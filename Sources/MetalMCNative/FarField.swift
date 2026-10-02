import Compression
import Foundation
import Metal
import MetalMCCore
import simd

// Far field (prototype, METALMC_FARFIELD=<level>): LOD levels from <level> up are drawn by a per-pixel height-field
// ray march instead of quads (docs/far-field-design.md). The quads' cost is vertex invocations, and at those levels
// 95% of them draw nothing (terrain seen at grazing angles); a ray march costs per pixel and has exact visibility.
//
// Data: every LOD node at those levels also keeps two words per column (LodBuild.farColumns): the ground (or water) at
// block precision, and a tree canopy over it as a slab rays can pass under. Per level there's a ring: a W x W window of
// that level's columns around the camera (one slice of a 2D array texture), and a max pyramid of column tops over it.
// A ring is toroidal (cell (x, z) of its level at texel (x mod W, z mod W)) and updated on the GPU, at most one ring per
// frame, when the camera moves 64 of its cells (the cells the window moved onto) or nodes in its window come, go or
// change columns (those nodes' cells; nodes are rebuilt whenever their region is saved, mostly with the same columns);
// until then it's drawn with the window it holds. A bitmap of the area drawn by quads (levels below the far field's, 64-block cells) is rebuilt every frame and
// tested where a ray reaches a column: columns in it count as empty, so rays only find terrain the quads don't draw.
//
// Drawing: right after the LOD's opaque quads, a shell around the camera at the horizontal distance of the nearest tile
// the far field draws (a 64-sided prism from the world bottom up to the highest terrain in the rings or the camera,
// and its floor). Every hit lies beyond it, so rays start there, the early depth test skips pixels with anything drawn
// nearer than the shell, and rays that leave through its top (over all the terrain) never start. Each pixel marches ring 0 until its ray leaves ring 0's
// window, then ring 1, and so on (each ring's cells are about 3.5 px wide where its window ends). A hit writes the
// column's color (top or side face, vanilla's lightmap, water over its floor, fog) and its depth.

/// The finest LOD level drawn by the far field (METALMC_FARFIELD, -PfarField; default 1); 0: off (quads at every level).
let lodFarFieldLevel = max(0, Int(ProcessInfo.processInfo.environment["METALMC_FARFIELD"] ?? "") ?? 1)
let lodFarFieldOn = lodFarFieldLevel > 0
/// Real columns' heights at block precision (LodFarCell), carried from the region files through every level. With
/// METALMC_EXP=ffvoxeltops they come from the voxels as before: tops rounded to 2-256 blocks, which stood real terrain
/// up to a voxel above the generated terrain next to it (spires along their seam).
let lodFarCells = lodFarFieldOn && !experiments.contains("ffvoxeltops")
/// Tree canopies as slabs over the ground (a second word per column). METALMC_EXP=ffpillars draws them as columns down
/// to the ground as before.
let lodFarCanopy = !experiments.contains("ffpillars")
/// Towers (columns v3): a coarse cell whose ground is two heights (a spire or a peak on a lower base, the top of a cliff
/// the cell straddles) is drawn as its base plus the high part in a rectangle of the cell, at an eighth of a cell's
/// precision, instead of one mean height over the whole cell (LodFarCell.addTower, at every merge from the blocks up;
/// the level-2 quadrant cache keeps them). METALMC_EXP=ffnotowers: one height as before.
let lodFarTowers = lodFarCells && !experiments.contains("ffnotowers")
/// Debug (METALMC_EXP=ffsteps): color hits by the number of march steps (green few, red many), misses dark blue.
let farFieldSteps = experiments.contains("ffsteps")
/// METALMC_FFGAIN: scales the light on land and trees (not water), standing in for the occlusion of detail finer than a
/// cell (the quads' voxels keep steps the cells' mean heights smooth away, and with them corners that darken). Default
/// 0.93, fitted against the level-0 answer key (far band 3.80 at 1, 3.43 at 0.96, 3.32 at 0.93, with the two-occluder
/// side foot; the quads 3.05).
let farFieldGain = max(0, Float(ProcessInfo.processInfo.environment["METALMC_FFGAIN"] ?? "") ?? 0.93)
/// METALMC_FFSTART: the pyramid level each ring's march starts at. From the top (11 with 2048-cell rings) rays over
/// everything leave a ring in one step, but every ray that meets terrain first descends 11 levels; a ray over the terrain
/// climbs a level a step anyway. Default 6, measured offline at the panel's resolution (the march's GPU time, top → 6):
/// the flight's view 4.02 → 3.51 ms, from 260 blocks up 6.04 → 5.50, near the ground 2.94 → 2.38; 6 of 7.7 M pixels
/// differ (rays at the step cap).
let farFieldStartLevel = Int(ProcessInfo.processInfo.environment["METALMC_FFSTART"] ?? "") ?? 6
/// Debug (METALMC_EXP=fflog): log every ring refill (with its time), to line them up with long frames.
let farFieldLogFills = experiments.contains("fflog")
/// The horizon profile (ff_profile; METALMC_EXP=ffnoprofile: none): per ring and azimuth, the steepest rise at which the
/// camera sees any of the ring's terrain, so a ray rising faster passes the ring without marching it (most of the sky
/// just over the horizon, and rays crossing near rings to far terrain).
let farFieldProfile = !experiments.contains("ffnoprofile")
let farFieldProfileBins = 1024
let farFieldProfileLevel = 5
/// Cells per ring side (METALMC_FFWIDTH, a power of two): each level's ring reaches half this many of its cells from the
/// camera, about as far as the quads use that level.
let farFieldWidth = Int(ProcessInfo.processInfo.environment["METALMC_FFWIDTH"] ?? "") ?? 2048
let farFieldMips = farFieldWidth.trailingZeroBitCount + 1   // max pyramid levels: W ... 1
let farFieldCoverCells = 256    // coverage bitmap side, 64-block cells (16 km)

/// Tree materials that are trunks (logs, stems): a crown's underside is its lowest leaves, not the trunk under them.
let lodTrunkMaterial: [Bool] = {
    var t = [Bool](repeating: false, count: 256)
    for m in [Mat.log, .cherryWood, .mushroomStem] { t[Int(m.rawValue)] = true }
    return t
}()

/// How a partial canopy is drawn (the offline evaluation's switch, mmc_debug_far_eval): 0 keeps it in a share of cells
/// equal to its cover (picked by a hash of the cell), 1 in cells at least half covered, 2 in every cell with any.
var lodFarCanopyMode = 0
/// Debug (mmc_debug_far_eval): merge heights by their maximum instead of their mean (bit 0 ground, bit 1 canopy).
var lodFarMergeMax = 0

/// Far field: a column as the far field sees it, at block precision, at any level (columns v2, docs/far-field-design.md).
/// Real columns get theirs from the blocks when a region is read (LodBuild.regionGrid: one per 2 x 2 blocks), and every
/// downsampling merges 2 x 2 of them (LodGrid.downsample, the region's cached level-2 quadrant included), so a level's
/// heights don't depend on its voxels, whose tops round to 2-256 blocks. Generated columns get theirs from the generator's
/// heights (LodFarStore.fill). Each height is a mean over the part of the cell it describes: the dry ground over the dry
/// part, the water and the bed under it over the wet part, the canopy over the part under trees. Means compose exactly
/// from level to level, and a lone tree or tower doesn't stand up as a whole cell as it would with the maximum.
struct LodFarCell: Equatable {
    /// Height units per block.
    static let unit = 16
    var ground: UInt16 = 0          // top of the dry ground, in 1/16 blocks above the world bottom
    var bed: UInt16 = 0             // top of the ground under the water
    var water: UInt16 = 0           // top of the top water block
    var canopyTop: UInt16 = 0       // top of the tree canopy
    var canopyBottom: UInt16 = 0    // its underside (the lowest leaves)
    var area: UInt8 = 0             // share of the cell with data, 0-255 (0: none; the voxels decide)
    var wet: UInt8 = 0              // share of that under water
    var cover: UInt8 = 0            // share of that under canopy
    var groundMat: UInt8 = 0        // the dry ground's top block
    var underMat: UInt8 = 0         // what its sides show a few blocks down
    var bedMat: UInt8 = 0
    var waterMat: UInt8 = 0
    var canopyMat: UInt8 = 0
    // Towers (every merge from the blocks up): the dry ground as two heights where one can't hold it, a base over
    // the cell and a high part standing in a rectangle of it (a spire, a peak, the top of a cliff the cell straddles).
    // `ground` stays the mean over both. peakShare 0: none.
    var peak: UInt16 = 0            // the high part's top, 1/16 blocks above the world bottom
    var peakShare: UInt8 = 0        // its share of the dry ground (1-254)
    var peakRect: UInt16 = 0        // its rectangle in eighths of the cell: x0 | (x1 - 1) << 3 | z0 << 6 | (z1 - 1) << 9
    var peakMat: UInt8 = 0          // its top block

    /// The base under a tower: the mean of the dry ground that isn't the high part.
    var towerBase: Int { (Int(ground) * 255 - Int(peak) * Int(peakShare) + (255 - Int(peakShare)) / 2) / (255 - Int(peakShare)) }

    /// 2 x 2 cells to their parent: shares add up, heights are means weighted by the share they describe, and each
    /// material is the one most of that share has (ties: the later cell). `towerLevel` > 0: the parent's level, and it
    /// gets a tower where its ground is two heights (addTower).
    static func merge(_ a: LodFarCell, _ b: LodFarCell, _ c: LodFarCell, _ d: LodFarCell, towerLevel: Int = 0) -> LodFarCell {
        @inline(__always) func at(_ i: Int) -> LodFarCell { i == 0 ? a : (i == 1 ? b : (i == 2 ? c : d)) }
        var areaSum = 0, dryW = 0, wetW = 0, covW = 0
        var g = 0, bd = 0, wt = 0, ct = 0, cb = 0, gMax = 0, ctMax = 0
        var dw = (0, 0, 0, 0), ww = (0, 0, 0, 0), cw = (0, 0, 0, 0)
        for i in 0..<4 {
            let x = at(i), ar = Int(x.area)
            if ar == 0 { continue }
            let dwi = ar * (255 - Int(x.wet)), wwi = ar * Int(x.wet), cwi = ar * Int(x.cover)
            switch i {
            case 0: dw.0 = dwi; ww.0 = wwi; cw.0 = cwi
            case 1: dw.1 = dwi; ww.1 = wwi; cw.1 = cwi
            case 2: dw.2 = dwi; ww.2 = wwi; cw.2 = cwi
            default: dw.3 = dwi; ww.3 = wwi; cw.3 = cwi
            }
            areaSum += ar
            dryW += dwi; wetW += wwi; covW += cwi
            g += dwi * Int(x.ground); bd += wwi * Int(x.bed); wt += wwi * Int(x.water)
            ct += cwi * Int(x.canopyTop); cb += cwi * Int(x.canopyBottom)
            if dwi > 0 { gMax = max(gMax, Int(x.ground)) }
            if cwi > 0 { ctMax = max(ctMax, Int(x.canopyTop)) }
        }
        var p = LodFarCell()
        if areaSum == 0 { return p }
        p.area = UInt8((areaSum + 3) / 4)
        p.wet = UInt8((wetW + areaSum / 2) / areaSum)
        p.cover = UInt8((covW + areaSum / 2) / areaSum)
        if dryW > 0 { p.ground = UInt16(lodFarMergeMax & 1 != 0 ? gMax : (g + dryW / 2) / dryW) }
        if wetW > 0 { p.bed = UInt16((bd + wetW / 2) / wetW); p.water = UInt16((wt + wetW / 2) / wetW) }
        if covW > 0 {
            p.canopyTop = UInt16(lodFarMergeMax & 2 != 0 ? ctMax : (ct + covW / 2) / covW)
            p.canopyBottom = UInt16((cb + covW / 2) / covW)
        }
        @inline(__always) func weight(_ w: (Int, Int, Int, Int), _ i: Int) -> Int { i == 0 ? w.0 : (i == 1 ? w.1 : (i == 2 ? w.2 : w.3)) }
        @inline(__always) func vote(_ w: (Int, Int, Int, Int), _ mat: (LodFarCell) -> UInt8) -> UInt8 {
            var best: UInt8 = 0, bestW = 0
            for i in 0..<4 where weight(w, i) > 0 {
                let m = mat(at(i))
                var sum = 0
                for j in 0..<4 where weight(w, j) > 0 && mat(at(j)) == m { sum += weight(w, j) }
                if sum >= bestW { best = m; bestW = sum }
            }
            return best
        }
        p.groundMat = vote(dw) { $0.groundMat }
        p.underMat = vote(dw) { $0.underMat }
        p.bedMat = vote(ww) { $0.bedMat }
        p.waterMat = vote(ww) { $0.waterMat }
        p.canopyMat = vote(cw) { $0.canopyMat }
        if towerLevel > 0 && lodFarTowers && dryW > 0 { p.addTower(a, b, c, d, dw, level: towerLevel) }
        return p
    }

    /// The parent's tower from its children's dry ground as parts: each child's base over its quadrant and its high part
    /// in its rectangle (one part over the quadrant if it has no tower). The parts split in two by height where that
    /// explains the most of their spread (Otsu's split); the high side is the tower if it stands at least a quarter of
    /// the cell (and 2 blocks) over the rest and its rectangle isn't most of the cell. High parts far apart (a rectangle
    /// much bigger than their area) would stand the whole cell up: then only the most prominent one is the tower.
    private mutating func addTower(_ a: LodFarCell, _ b: LodFarCell, _ c: LodFarCell, _ d: LodFarCell,
                                   _ dw: (Int, Int, Int, Int), level: Int) {
        let u = LodFarCell.unit, minRise = u * max(2, (1 << level) / 4)
        var h = SIMD8<Int>(repeating: 0), w = SIMD8<Int>(repeating: 0)   // height (1/16 blocks), weight
        var rx0 = SIMD8<Int>(repeating: 0), rx1 = SIMD8<Int>(repeating: 0), rz0 = SIMD8<Int>(repeating: 0), rz1 = SIMD8<Int>(repeating: 0)
        var mat = SIMD8<Int>(repeating: 0)
        var n = 0, lo = Int.max, hi = Int.min
        for i in 0..<4 {
            let x = i == 0 ? a : (i == 1 ? b : (i == 2 ? c : d))
            let dwi = i == 0 ? dw.0 : (i == 1 ? dw.1 : (i == 2 ? dw.2 : dw.3))
            if dwi == 0 { continue }
            let qx = (i & 1) * 8, qz = (i >> 1) * 8   // the child's quadrant, in sixteenths of the parent
            if x.peakShare > 0 {
                let hw = dwi * Int(x.peakShare) / 255
                h[n] = x.towerBase; w[n] = dwi - hw; mat[n] = Int(x.groundMat)
                rx0[n] = qx; rx1[n] = qx + 8; rz0[n] = qz; rz1[n] = qz + 8
                lo = min(lo, h[n]); hi = max(hi, h[n])
                n += 1
                let r = Int(x.peakRect)   // the child's eighths are the parent's sixteenths
                h[n] = Int(x.peak); w[n] = hw; mat[n] = Int(x.peakMat)
                rx0[n] = qx + (r & 7); rx1[n] = qx + ((r >> 3) & 7) + 1; rz0[n] = qz + ((r >> 6) & 7); rz1[n] = qz + ((r >> 9) & 7) + 1
            } else {
                h[n] = Int(x.ground); w[n] = dwi; mat[n] = Int(x.groundMat)
                rx0[n] = qx; rx1[n] = qx + 8; rz0[n] = qz; rz1[n] = qz + 8
            }
            lo = min(lo, h[n]); hi = max(hi, h[n])
            n += 1
        }
        // Most cells: no two parts far enough apart in height.
        if n < 2 || hi - lo < minRise { return }
        // Parts by height (insertion sort of at most 8), then the split with the most variance between the sides.
        var order = SIMD8<Int>(0, 1, 2, 3, 4, 5, 6, 7)
        for i in 1..<n {
            var j = i
            while j > 0 && h[order[j - 1]] > h[order[j]] { let t = order[j]; order[j] = order[j - 1]; order[j - 1] = t; j -= 1 }
        }
        var total = 0, totalH = 0
        for i in 0..<n { total += w[i]; totalH += w[i] * h[i] }
        if total == 0 { return }
        var best = 0.0, split = 0, lowW = 0, lowH = 0
        for k in 1..<n {
            lowW += w[order[k - 1]]; lowH += w[order[k - 1]] * h[order[k - 1]]
            let highW = total - lowW
            if lowW == 0 || highW == 0 { continue }
            let dm = Double(totalH - lowH) / Double(highW) - Double(lowH) / Double(lowW)
            let between = Double(lowW) * Double(highW) * dm * dm
            if between > best { best = between; split = k }
        }
        if split == 0 { return }
        var high = 0   // bit per part
        for k in split..<n { high |= 1 << order[k] }
        @inline(__always) func bounds(_ m: Int) -> (Int, Int, Int, Int, Int, Int) {   // x0, x1, z0, z1 (sixteenths), weight, weight * height
            var x0 = 16, x1 = 0, z0 = 16, z1 = 0, hw = 0, hh = 0
            for i in 0..<n where m & (1 << i) != 0 {
                x0 = min(x0, rx0[i]); x1 = max(x1, rx1[i]); z0 = min(z0, rz0[i]); z1 = max(z1, rz1[i])
                hw += w[i]; hh += w[i] * h[i]
            }
            return (x0, x1, z0, z1, hw, hh)
        }
        var (x0, x1, z0, z1, hw, hh) = bounds(high)
        if Double((x1 - x0) * (z1 - z0)) / 256 > 2 * Double(hw) / Double(total) + 0.125 {
            // Spread out: the part standing out most (weight times height over the mean) alone.
            let mean = Double(totalH) / Double(total)
            var top = -1, topScore = 0.0
            for i in 0..<n where high & (1 << i) != 0 {
                let sc = Double(w[i]) * (Double(h[i]) - mean)
                if sc > topScore { topScore = sc; top = i }
            }
            if top < 0 { return }
            high = 1 << top
            (x0, x1, z0, z1, hw, hh) = bounds(high)
        }
        let lw = total - hw
        if hw == 0 || lw == 0 { return }
        let peakH = (hh + hw / 2) / hw, baseH = (totalH - hh + lw / 2) / lw
        guard peakH - baseH >= minRise, (x1 - x0) * (z1 - z0) * 4 <= 256 * 3 else { return }
        // The high parts' top block: the one most of their weight has.
        var pm = 0, pmW = -1
        for i in 0..<n where high & (1 << i) != 0 {
            var sum = 0
            for j in 0..<n where high & (1 << j) != 0 && mat[j] == mat[i] { sum += w[j] }
            if sum > pmW { pmW = sum; pm = mat[i] }
        }
        // To eighths of the cell, outward.
        let ex0 = x0 / 2, ex1 = (x1 + 1) / 2, ez0 = z0 / 2, ez1 = (z1 + 1) / 2
        peak = UInt16(peakH)
        peakShare = UInt8(min(254, max(1, (hw * 255 + total / 2) / total)))
        peakRect = UInt16(ex0 | (ex1 - 1) << 3 | ez0 << 6 | (ez1 - 1) << 9)
        peakMat = UInt8(pm)
    }

    /// One block column of a chunk (`b`: 16 x 16 x 384 material ids laid out y, z, x from the world bottom, `top` the
    /// highest block that can be set) as a cell, `tint` its biome tint. Tree blocks at its top (snow on them too) are its
    /// canopy, down to the lowest leaves (a trunk under them is too thin to see from the far field's distances), over the
    /// water or the ground under them.
    static func block(_ b: UnsafePointer<UInt8>, x: Int, z: Int, top: Int, tint: UInt8) -> LodFarCell {
        let col = z * 16 + x
        let airK = MaterialKind.air.rawValue, waterK = MaterialKind.water.rawValue, u = unit
        let snow = Mat.snow.rawValue
        @inline(__always) func m(_ y: Int) -> UInt8 { lodReduceInput[Int(b[y * 256 + col])] }
        @inline(__always) func kind(_ y: Int) -> UInt8 { lodKinds[Int(m(y))] }
        // Snow resting on leaves belongs to the tree (snowy spruces carry it on every tier).
        @inline(__always) func tree(_ y: Int) -> Bool { lodTreeMaterial[Int(m(y))] || (m(y) == snow && y > 0 && lodTreeMaterial[Int(m(y - 1))]) }
        var c = LodFarCell()
        var y = top
        while y >= 0 && kind(y) == airK { y -= 1 }
        if y < 0 { return c }   // no blocks: no data
        c.area = 255
        if tree(y) {
            c.canopyTop = UInt16((y + 1) * u)
            c.canopyMat = lodTinted(m(y), tint)
            c.cover = 255
            var leaf = -1, low = y
            while y >= 0 {
                if tree(y) {
                    low = y
                    if !lodTrunkMaterial[Int(m(y))] { leaf = y }
                } else if kind(y) != airK {
                    break
                }
                y -= 1
            }
            c.canopyBottom = UInt16((leaf >= 0 ? leaf : low) * u)
            if y < 0 { return c }
        }
        if kind(y) == waterK {
            c.wet = 255
            c.water = UInt16((y + 1) * u)
            c.waterMat = lodTinted(m(y), tint)
            while y >= 0 && (kind(y) == waterK || kind(y) == airK) { y -= 1 }
            c.bed = UInt16((y + 1) * u)
            c.bedMat = y >= 0 ? lodTinted(m(y), tint) : 0
        } else {
            c.ground = UInt16((y + 1) * u)
            c.groundMat = lodTinted(m(y), tint)
            // The sides: the block 3 under the top (dirt under grass, also under a snow layer), else the first solid
            // block under that (the bed under ice).
            var k = max(0, y - 3)
            while k > 0 && (kind(k) == airK || kind(k) == waterK) { k -= 1 }
            c.underMat = kind(k) == airK || kind(k) == waterK ? c.groundMat : lodTinted(m(k), tint)
        }
        return c
    }

    /// Column `i` of `g`'s voxels as a cell, for columns without one (live chunks, servers): the top of the top voxel,
    /// the materials, and a canopy where tree voxels lie over air.
    static func voxels(_ g: LodGrid, column i: Int) -> LodFarCell {
        var c = LodFarCell()
        let layer = lodNodeVoxels * lodNodeVoxels, s = (1 << g.level) * LodFarCell.unit
        let airK = MaterialKind.air.rawValue
        var y = g.height - 1
        while y >= 0 && lodKinds[Int(g.v[y * layer + i])] == airK { y -= 1 }
        if y < 0 { return c }
        c.area = 255
        if lodTreeMaterial[Int(g.v[y * layer + i])] {
            c.canopyTop = UInt16((y + 1) * s)
            c.canopyMat = g.v[y * layer + i]
            c.cover = 255
            var leaf = -1, low = y
            while y >= 0 {
                let t = g.v[y * layer + i]
                if lodTreeMaterial[Int(t)] {
                    low = y
                    if !lodTrunkMaterial[Int(t)] { leaf = y }
                } else if lodKinds[Int(t)] != airK {
                    break
                }
                y -= 1
            }
            c.canopyBottom = UInt16((leaf >= 0 ? leaf : low) * s)
            if y < 0 { return c }
        }
        let top = g.v[y * layer + i]
        if lodIsWater(top) {
            c.wet = 255
            // The water's top block ends a block under the voxel's top: the far field draws water to one block over
            // its top block (words), which puts it at the voxel top like the quads.
            c.water = UInt16((y + 1) * s - LodFarCell.unit)
            c.waterMat = top
            while y >= 0 && (lodIsWater(g.v[y * layer + i]) || lodKinds[Int(g.v[y * layer + i])] == airK) { y -= 1 }
            c.bed = UInt16((y + 1) * s)
            c.bedMat = y >= 0 ? g.v[y * layer + i] : 0
        } else {
            c.ground = UInt16((y + 1) * s)
            c.groundMat = top
            c.underMat = top
            if y > 0, g.v[(y - 1) * layer + i] != 0, !lodIsWater(g.v[(y - 1) * layer + i]) { c.underMat = g.v[(y - 1) * layer + i] }
        }
        return c
    }

    /// Whether the cell's canopy is drawn (it covers `cover` of the cell): column (x, z) of a level-`level` node. By
    /// default a partial canopy stays in about `cover` of such cells, which keeps the share of trees (and their color)
    /// at every level, as the generator's sparse woods do; the hash mixes in the cell's heights so neighboring nodes
    /// don't repeat one pattern.
    @inline(__always) func keepsCanopy(x: Int, z: Int, level: Int, mode: Int = lodFarCanopyMode) -> Bool {
        switch mode {
        case 1: return cover >= 128
        case 2: return cover > 0
        default:
            return cover == 255 || Double(cover) / 255 > lodHash(x &+ Int(ground) &* 7919, z &+ Int(canopyTop) &* 104_729, 0xCA40 &+ UInt64(level))
        }
    }

    /// The far field's two words for the cell (one RG32 texel of a ring per column). Word 0: bits 0-8 the ground's top
    /// (blocks above the world bottom; 0: no column), 9-15 blocks of water over it (to one block over the top water
    /// block's top, which the shader drops 10/9 block to vanilla's surface, as the LOD's quads do from their voxel tops),
    /// 16-23 the top material (the bed's under water), 24-31 the water's material if there's water, else what the
    /// sides show under the top. Word 1, the canopy (0: none): bits 0-8 its top and 9-17 its underside (blocks above the
    /// world bottom), 18-25 its material. Or, with bit 31 set, a tower (columns v3): bits 0-8 the high part's top, 9-20
    /// its rectangle (peakRect), 21-28 its top block; word 0 then holds the base under it. (x, z): the column in its node,
    /// which picks the cells that keep a partial canopy.
    func words(x: Int, z: Int, level: Int, mode: Int = lodFarCanopyMode) -> (UInt32, UInt32) {
        guard area > 0 else { return (0, 0) }
        let u = LodFarCell.unit
        @inline(__always) func blocks(_ h: UInt16) -> Int { (Int(h) + u / 2) / u }
        var w0: UInt32, top: Int
        let dryMat: UInt8
        if wet >= 128 {
            let surface = max(2, blocks(water) + 1), floor = min(surface - 1, max(1, blocks(bed), surface - 127))
            w0 = UInt32(floor) | UInt32(surface - floor) << 9 | UInt32(bedMat) << 16 | UInt32(waterMat) << 24
            top = surface
            dryMat = bedMat
        } else {
            top = min(511, max(1, blocks(ground)))
            w0 = UInt32(top) | UInt32(groundMat) << 16 | UInt32(underMat) << 24
            dryMat = groundMat
        }
        let ct = min(511, blocks(canopyTop))
        // A tower (dry cells): the base's top in word 0, the high part in word 1 (flag bit 31) unless trees cover at
        // least half the cell (the canopy then).
        if peakShare > 0 && wet < 128 && lodFarTowers && !(cover >= 128 && ct > top) {
            let base = min(511, max(1, (towerBase + u / 2) / u)), pt = min(511, blocks(peak))
            if pt > base {
                return (UInt32(base) | UInt32(groundMat) << 16 | UInt32(underMat) << 24,
                        1 << 31 | UInt32(pt) | UInt32(peakRect) << 9 | UInt32(peakMat) << 21)
            }
        }
        guard cover > 0, ct > top, keepsCanopy(x: x, z: z, level: level, mode: mode) else { return (w0, 0) }
        if !lodFarCanopy { return (UInt32(ct) | UInt32(canopyMat) << 16 | UInt32(dryMat) << 24, 0) }
        let cb = min(ct - 1, blocks(canopyBottom))
        return (w0, UInt32(ct) | UInt32(cb) << 9 | UInt32(canopyMat) << 18)
    }

    /// Cells as bytes (24 per cell, little-endian, in field order), LZFSE-compressed: a region's cached quadrant.
    static func encode(_ cells: ArraySlice<LodFarCell>) -> [UInt8] {
        var raw = [UInt8]()
        raw.reserveCapacity(cells.count * 24)
        for c in cells {
            for h in [c.ground, c.bed, c.water, c.canopyTop, c.canopyBottom] { raw.append(UInt8(h & 255)); raw.append(UInt8(h >> 8)) }
            raw += [c.area, c.wet, c.cover, c.groundMat, c.underMat, c.bedMat, c.waterMat, c.canopyMat]
            raw += [UInt8(c.peak & 255), UInt8(c.peak >> 8), c.peakShare, UInt8(c.peakRect & 255), UInt8(c.peakRect >> 8), c.peakMat]
        }
        var packed = [UInt8](repeating: 0, count: raw.count + 1024)
        let n = raw.withUnsafeBufferPointer { compression_encode_buffer(&packed, packed.count, $0.baseAddress!, $0.count, nil, COMPRESSION_LZFSE) }
        return n > 0 ? Array(packed[0..<n]) : []
    }

    static func decode(_ bytes: [UInt8], count: Int) -> [LodFarCell]? {
        guard !bytes.isEmpty else { return nil }
        var raw = [UInt8](repeating: 0, count: count * 24)
        let got = bytes.withUnsafeBufferPointer { compression_decode_buffer(&raw, raw.count, $0.baseAddress!, $0.count, nil, COMPRESSION_LZFSE) }
        guard got == raw.count else { return nil }
        var out = [LodFarCell](repeating: LodFarCell(), count: count)
        for i in 0..<count {
            let o = i * 24
            @inline(__always) func h(_ k: Int) -> UInt16 { UInt16(raw[o + 2 * k]) | UInt16(raw[o + 2 * k + 1]) << 8 }
            out[i] = LodFarCell(ground: h(0), bed: h(1), water: h(2), canopyTop: h(3), canopyBottom: h(4),
                                area: raw[o + 10], wet: raw[o + 11], cover: raw[o + 12], groundMat: raw[o + 13],
                                underMat: raw[o + 14], bedMat: raw[o + 15], waterMat: raw[o + 16], canopyMat: raw[o + 17],
                                peak: h(9), peakShare: raw[o + 20], peakRect: UInt16(raw[o + 21]) | UInt16(raw[o + 22]) << 8,
                                peakMat: raw[o + 23])
        }
        return out
    }
}

extension LodBuild {
    /// Two words per column of a node's grid for the far field (LodFarCell.words), from the grid's cells where it has
    /// them (real columns at block precision, generated ones from the generator's heights), else from its voxels.
    static func farColumns(_ g: LodGrid) -> [UInt32] {
        let n = lodNodeVoxels, layer = n * n
        var out = [UInt32](repeating: 0, count: 2 * layer)
        for i in 0..<layer {
            let (w0, w1): (UInt32, UInt32)
            if !g.far.isEmpty && g.far[i].area != 0 {
                (w0, w1) = g.far[i].words(x: i % n, z: i / n, level: g.level)
            } else if !lodFarCanopy {
                (w0, w1) = (voxelWord(g, i), 0)
            } else {
                (w0, w1) = LodFarCell.voxels(g, column: i).words(x: i % n, z: i / n, level: g.level)
            }
            out[2 * i] = w0
            out[2 * i + 1] = w1
        }
        return out
    }

    /// The column word from the voxels as the first far field made it (METALMC_EXP=ffpillars): the top voxel's top,
    /// trees included, and the voxel under it for the sides.
    static func voxelWord(_ g: LodGrid, _ i: Int) -> UInt32 {
        let layer = lodNodeVoxels * lodNodeVoxels
        var y = g.height - 1
        while y >= 0 && g.v[y * layer + i] == 0 { y -= 1 }
        if y < 0 { return 0 }
        var waterTop = -1
        var waterMat: UInt8 = 0
        if lodIsWater(g.v[y * layer + i]) {
            waterTop = y
            waterMat = g.v[y * layer + i]
            while y >= 0 && (lodIsWater(g.v[y * layer + i]) || g.v[y * layer + i] == 0) { y -= 1 }
        }
        let top: UInt8 = y >= 0 ? g.v[y * layer + i] : 0
        var sub = top
        if y > 0, g.v[(y - 1) * layer + i] != 0, !lodIsWater(g.v[(y - 1) * layer + i]) { sub = g.v[(y - 1) * layer + i] }
        let s = 1 << g.level
        return UInt32((y + 1) * s) | UInt32(waterTop >= 0 ? min(127, (waterTop - y) * s) : 0) << 9
            | UInt32(top) << 16 | UInt32(waterTop >= 0 ? waterMat : sub) << 24
    }
}

/// Uniforms of the march (must match FarUniforms in the shader).
struct FarUniforms {
    var invViewProj: simd_float4x4          // inverse of the jittered projection * view rotation (camera-relative)
    var viewport: SIMD4<Float>               // width, height, highest terrain in the rings (blocks above the bottom), rings
    var cam: SIMD4<Float>                    // x: camera height above the world bottom (blocks); y: water alpha;
                                             // z: shell radius (blocks); w: shell top relative to the camera
    var ring: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
                                             // per ring: camera in ring coordinates (cells, xy), cell size in blocks (z)
    var ringCover: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
                                             // per ring: its coordinates' origin in blocks from the coverage bitmap's
                                             // corner (xy), its window's first cell in its coordinates (zw)
}

let farFieldShaderSource = """
#include <metal_stdlib>
using namespace metal;

// Lit mode (METALMC_EXP=lit, Lit.swift): the march also writes the terrain G-buffer.
#define LIT_MODE \(litEnabled ? 1 : 0)
\(litShaderHeader)
// Offline only (FARTEST_STATS, FarFieldDebug): per-pixel march counters into a buffer FarFieldDebug binds.
#define FF_STATS 0

struct LodUniforms {
    float4x4 proj;
    float4x4 view;
    float4 fogColor;
    float envStart, envEnd, rdStart, rdEnd;
    float discardRadius;
    float sky;
    float alpha;
    float lightmapOn;
    float4 camFrac;
    float4 camInSection;
    int4 seamInfo;
};
struct FarUniforms {
    float4x4 invViewProj;
    float4 viewport;
    float4 cam;
    float4 ring[8];
    float4 ringCover[8];
};
struct FillParams { int2 nodeCell; int2 base; int2 size; uint ring; uint level; };
struct ClearParams { int2 base; int2 size; uint ring; uint pad; };

constant int W = \(farFieldWidth);
constant int TOP = \(farFieldMips - 1);
constant int COVER = \(farFieldCoverCells);
#define FF_PROFILE \(farFieldProfile ? 1 : 0)
constant int PBINS = \(farFieldProfileBins);   // azimuth bins of the horizon profile
constant int PLEVEL = \(farFieldProfileLevel);   // the pyramid level it's built from

// A ring is toroidal: the cell (x, z) of its level is texel (x mod W, z mod W), so a window that moves only needs the
// cells it moved onto written. Clears a rectangle of cells (base, size).
kernel void ff_clear(uint2 gid [[thread_position_in_grid]], constant ClearParams& p [[buffer(0)]],
                     texture2d_array<uint, access::write> data [[texture(0)]],
                     texture2d_array<ushort, access::write> heights [[texture(1)]]) {
    if (gid.x >= uint(p.size.x) || gid.y >= uint(p.size.y)) return;
    uint2 t = uint2((p.base + int2(gid)) & (W - 1));
    data.write(uint4(0), t, p.ring);
    heights.write(ushort4(0), t, p.ring);
}

// A rectangle (base, size) of one node's columns (two words each: LodFarCell.words) into its level's ring. The pyramid
// holds the top of the ground or water, or of the canopy (or tower) over it.
kernel void ff_fill(uint2 gid [[thread_position_in_grid]], constant FillParams& p [[buffer(0)]],
                    const device uint2* cols [[buffer(1)]],
                    texture2d_array<uint, access::write> data [[texture(0)]],
                    texture2d_array<ushort, access::write> heights [[texture(1)]]) {
    if (gid.x >= uint(p.size.x) || gid.y >= uint(p.size.y)) return;
    int2 cell = p.base + int2(gid), i = cell - p.nodeCell;
    uint2 t = uint2(cell & (W - 1));
    uint2 c = cols[i.y * 256 + i.x];
    data.write(uint4(c, 0u, 0u), t, p.ring);
    // What a ray can meet in the column: the ground's top, or the water's surface (the march draws it 10/9 block under
    // its word's top: the block under, rounded up), or the canopy's or tower's top. The water's word top made rays
    // skimming the sea dip to every column under it and miss the surface a block lower, a step down and a column test
    // per cell.
    uint depth = (c.x >> 9) & 127u;
    uint top = max((c.x & 511u) + depth - (depth != 0u ? 1u : 0u), c.y & 511u);
    heights.write(ushort4(ushort(c.x == 0u ? 0u : top)), t, p.ring);
}

kernel void ff_mip(uint2 gid [[thread_position_in_grid]], constant uint& slice [[buffer(0)]],
                   texture2d_array<ushort, access::read> src [[texture(0)]],
                   texture2d_array<ushort, access::write> dst [[texture(1)]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint2 s = gid * 2u;
    ushort a = max(src.read(s, slice).r, src.read(s + uint2(1, 0), slice).r);
    ushort b = max(src.read(s + uint2(0, 1), slice).r, src.read(s + uint2(1, 1), slice).r);
    dst.write(ushort4(max(a, b)), gid, slice);
}

// Floats as ints that order the same, so an atomic max on the ints is a max on the floats.
static int ffOrdered(float x) { int i = as_type<int>(x); return i >= 0 ? i : i ^ 0x7FFFFFFF; }
static float ffUnordered(int i) { return as_type<float>(i >= 0 ? i : i ^ 0x7FFFFFFF); }

// The horizon profile, per frame: per ring and azimuth bin, the steepest rise (blocks up per block of horizontal
// distance) at which the camera sees any of the ring's terrain. One thread per cell of a coarse level of each ring's
// pyramid: its terrain is no higher than its maximum and no nearer than the cell (nor than the shell, where rays start),
// so seen from the camera it rises at most (maximum - camera) / nearest distance, or over the farthest distance for
// terrain under the camera; that goes into every bin the cell spans, and one more each side for rounding. Cells the
// ring's march never reaches are left out: inside the shell, and inside the window of the ring before (its rays are in
// that ring there). The buffer starts at -infinity (filled with 0x80 bytes). Per ring and band of distance as well, to
// start a ring's march where its terrain may first be met, saved little (260 up 3%, elsewhere nothing).
kernel void ff_profile(uint3 gid [[thread_position_in_grid]], constant FarUniforms& f [[buffer(25)]],
                       texture2d_array<ushort, access::read> heights [[texture(28)]],
                       device atomic_int* prof [[buffer(30)]]) {
    uint r = gid.z, n = uint(W >> PLEVEL);
    if (gid.x >= n || gid.y >= n || r >= uint(f.viewport.w)) return;
    float cs = float(1 << PLEVEL);
    float2 c = floor(f.ringCover[r].zw / cs) + float2(gid.xy);   // the cell, ring coordinates at level PLEVEL
    float h = float(heights.read(uint2(int2(c)) & (n - 1u), r, PLEVEL).r);
    if (h <= 0.0) return;
    float s = f.ring[r].z, shell = f.cam.z * 0.999;
    float2 lo = (c * cs - f.ring[r].xy) * s, hi = lo + cs * s;   // its square relative to the camera, blocks
    float far = length(max(abs(lo), abs(hi)));
    if (far < shell) return;
    if (r > 0) {
        float sp = f.ring[r - 1].z;
        float2 plo = (f.ringCover[r - 1].zw - f.ring[r - 1].xy) * sp, phi = plo + float(W) * sp;
        if (all(lo >= plo) && all(hi <= phi)) return;
    }
    float2 dn = max(max(lo, -hi), 0.0);
    float nearD = max(length(dn), shell), up = h - f.cam.x;
    int v = ffOrdered(up >= 0.0 ? up / nearD : up / max(far, nearD));
    device atomic_int* row = prof + r * uint(PBINS);
    if (all(dn == 0.0)) {   // the camera over the cell: every direction
        for (int b = 0; b < PBINS; b++) atomic_fetch_max_explicit(&row[b], v, memory_order_relaxed);
        return;
    }
    // The bins between its corners' directions (a square the camera isn't over spans less than half a turn).
    float2 mid = (lo + hi) * 0.5;
    float ac = atan2(mid.y, mid.x), a0 = 0.0, a1 = 0.0;
    for (int k = 0; k < 4; k++) {
        float2 p = float2((k & 1) ? hi.x : lo.x, (k & 2) ? hi.y : lo.y);
        float da = atan2(p.y, p.x) - ac;
        da -= 2.0 * M_PI_F * round(da / (2.0 * M_PI_F));
        a0 = min(a0, da); a1 = max(a1, da);
    }
    float perBin = float(PBINS) / (2.0 * M_PI_F);
    int b0 = int(floor((ac + a0 + M_PI_F) * perBin)) - 1, b1 = int(floor((ac + a1 + M_PI_F) * perBin)) + 1;
    for (int b = b0; b <= b1; b++) atomic_fetch_max_explicit(&row[((b % PBINS) + PBINS) % PBINS], v, memory_order_relaxed);
}

// The shell: 64 wall segments (6 vertices each) inscribed in the circle of radius cam.z, then the floor (3 each).
vertex float4 ff_vs(uint vid [[vertex_id]], constant LodUniforms& u [[buffer(19)]], constant FarUniforms& f [[buffer(25)]]) {
    const uint wallCorner[6] = { 0, 1, 1, 0, 1, 0 }, wallTop[6] = { 0, 0, 1, 0, 1, 1 };
    float yb = -f.cam.x, yt = f.cam.w;
    float3 rel;
    if (vid < 384u) {
        uint seg = vid / 6u, k = vid % 6u;
        float a = float(seg + wallCorner[k]) * (2.0 * M_PI_F / 64.0);
        rel = float3(f.cam.z * cos(a), wallTop[k] ? yt : yb, f.cam.z * sin(a));
    } else {
        uint seg = (vid - 384u) / 3u, k = (vid - 384u) % 3u;
        float a = float(seg + (k == 2u ? 1u : 0u)) * (2.0 * M_PI_F / 64.0);
        rel = k == 0u ? float3(0.0, yb, 0.0) : float3(f.cam.z * cos(a), yb, f.cam.z * sin(a));
    }
    float4 clip = u.proj * (u.view * float4(rel, 1.0));
    clip.y = -clip.y;
    return clip;
}

constant float kShade[6] = { 0.6, 0.6, 1.0, 0.5, 0.8, 0.8 };
// The LOD's ambient occlusion steps for 0-3 occluders at levels 1 and up (kAO in the LOD shader).
constant float kAO[4] = { 1.0, 0.88, 0.76, 0.64 };

// A side face's foot: vanilla's smooth lighting counts two occluders at a step's bottom corners (the ground in front
// and the corner beside it), as the quads' voxels do; METALMC_EXP=ffsideao1 counts one as before (brighter by a tenth).
constant float kSideFoot = \(experiments.contains("ffsideao1") ? "0.88" : "0.76");   // kAO[1] or kAO[2]

// Sky light levels the ground under a canopy loses (vanilla's leaves dim sky light a level per block).
constant float kUnderCanopy = 3.0;

// A neighbor's height for ambient occlusion: its ground's top, or with `canopy` the top of its canopy if that's higher.
static float solidAt(texture2d_array<uint, access::read> data, int2 c, uint r, bool canopy) {
    uint2 w = data.read(uint2(c & (W - 1)), r).rg;
    return canopy ? float(max(w.x & 511u, w.y & 511u)) : float(w.x & 511u);
}
// Ambient occlusion like the LOD's (vanilla's smooth lighting per voxel corner, bilinear inside the face). A top face's
// corners count taller neighbors (both sides: 3); a side face darkens toward its foot, where the ground in front of it
// occludes its lowest voxel's bottom corners. A canopy's top counts its neighbors' canopies too (crowns shade each other).
static float columnAO(texture2d_array<uint, access::read> data, int2 c, uint r, float s, int face, float top, float2 f, float y,
                      bool canopy) {
    if (face == 2) {
        // The 8 neighbors once each (the 4 corners share them): -x, +x, -z, +z, then the diagonals in corner order.
        bool ex0 = solidAt(data, c + int2(-1, 0), r, canopy) > top, ex1 = solidAt(data, c + int2(1, 0), r, canopy) > top;
        bool ez0 = solidAt(data, c + int2(0, -1), r, canopy) > top, ez1 = solidAt(data, c + int2(0, 1), r, canopy) > top;
        float o[4];
        for (int k = 0; k < 4; k++) {
            int2 dd = int2((k & 1) ? 1 : -1, (k & 2) ? 1 : -1);
            bool a = (k & 1) ? ex1 : ex0, b = (k & 2) ? ez1 : ez0;
            bool g = solidAt(data, c + dd, r, canopy) > top;
            o[k] = kAO[(a && b) ? 3 : int(a) + int(b) + int(g)];
        }
        return mix(mix(o[0], o[1], f.x), mix(o[2], o[3], f.x), f.y);
    }
    int2 front = face == 0 ? int2(1, 0) : (face == 1 ? int2(-1, 0) : (face == 4 ? int2(0, 1) : int2(0, -1)));
    float v = (y - solidAt(data, c + front, r, false)) / s;
    return v < 1.0 ? mix(kSideFoot, 1.0, saturate(v)) : 1.0;
}

static float3 lodLight(constant LodUniforms& u, texture2d<float> lightmap, sampler s, float skyLevel) {
    if (u.lightmapOn > 0.5) return lightmap.sample(s, float2(0.5 / 16.0, (skyLevel + 0.5) / 16.0), level(0)).rgb;
    return float3(mix(0.2, 1.0, u.sky) * (0.04 + 0.96 * skyLevel / 15.0));
}
static float linearFog(float d, float s, float e) {
    if (d <= s) return 0.0;
    if (d >= e) return 1.0;
    return (d - s) / (e - s);
}

#if FF_STATS
// The counters are written to a buffer, which would run the shader on pixels the early depth test skips unless early
// tests are forced, and then it can't write depth: the depth goes to an unbound color slot (the counters don't need it).
#define FF_EARLY [[early_fragment_tests]]
#define FF_DEPTH [[color(7)]]
#else
#define FF_EARLY
#define FF_DEPTH [[depth(less)]]
#endif
#if LIT_MODE
struct FFOut { float4 color [[color(0)]]; uint2 gbuf [[color(1)]]; float depth FF_DEPTH; };
#else
struct FFOut { float4 color [[color(0)]]; float depth FF_DEPTH; };
#endif
struct LodSpriteGPU { float4 top; float4 side; float4 luma; };

// Texture detail like the LOD's (lodShade): the texel's luma relative to the texture's mean luma, from a mip picked for
// the pixel's footprint on the face (the march has no smooth derivatives at column edges, so no gradients).
static float detail(constant LodUniforms& u, constant LodSpriteGPU* sprites, texture2d<float> atlas, sampler as, uint m,
                    int face, float3 rel, float mip) {
    if (u.camFrac.w < 0.5) return 1.0;
    uint mat = (m >= \(lodWaterBase)u && m < \(lodWaterBase + 32)u) ? \(Mat.water.rawValue)u
             : ((m >= \(lodLeavesBase)u && m < \(lodLeavesBase + 32)u) ? \(Mat.leaves.rawValue)u
             : ((m >= \(lodGrassBase)u && m < \(lodGrassBase + 32)u) ? \(Mat.grass.rawValue)u : m));
    float3 wp = rel + u.camFrac.xyz;
    float2 bc = face < 2 ? float2(wp.z, -wp.y) : (face < 4 ? wp.xz : float2(wp.x, -wp.y));
    bool top = face == 2 || face == 3;
    float4 rect = top ? sprites[mat].top : sprites[mat].side;
    float4 t = atlas.sample(as, rect.xy + fract(bc) * (rect.zw - rect.xy), level(mip));
    float luma = dot(t.rgb, float3(0.2126, 0.7152, 0.0722));
    float mean = top ? sprites[mat].luma.x : sprites[mat].luma.y;
    return clamp(mix(mat == \(Mat.water.rawValue)u ? 1.0 : 0.75, luma / max(mean, 0.02), t.a), 0.0, 2.0);
}

FF_EARLY fragment FFOut ff_fs(float4 pos [[position]], constant LodUniforms& u [[buffer(19)]], constant FarUniforms& f [[buffer(25)]],
                     constant float4* colors [[buffer(26)]],
                     texture2d_array<uint, access::read> data [[texture(27)]],
                     texture2d_array<ushort, access::read> heights [[texture(28)]],
                     texture2d<float> lightmap [[texture(29)]], sampler ls [[sampler(14)]],
                     constant LodSpriteGPU* sprites [[buffer(20)]], texture2d<float> atlas [[texture(30)]],
                     sampler atlasSampler [[sampler(15)]], const device uchar* cover [[buffer(28)]]
#if FF_PROFILE
                     , const device int* prof [[buffer(30)]]
#endif
#if FF_STATS
                     , device uint4* stats [[buffer(29)]]
#endif
                     ) {
#if FF_STATS
    // Per pixel: steps by kind (advances at a level over 0, descents, level-0 column tests and those that missed,
    // climbs undone by the next step), rings entered, steps per ring, the hit's ring and step in its ring.
    uint stAdv = 0, stDesc = 0, stDip = 0, stColMiss = 0, stRings = 0, stWasted = 0, stHitStep = 0;
    uint stRing[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    bool stClimbed = false;
#endif
    float2 uv = pos.xy / f.viewport.xy;
    float4 hp = f.invViewProj * float4(uv * 2.0 - 1.0, 1.0, 1.0);
    float3 dir = normalize(hp.xyz / hp.w);
    uint rings = uint(f.viewport.w);
    float t = f.cam.z / max(length(dir.xz), 1e-6) * 0.999;   // nothing to find inside the shell
    // Nothing to find once a rising ray is over the highest terrain.
    float tMax = dir.y > 0.0 ? (f.viewport.z - f.cam.x) / dir.y : INFINITY;
#if FF_PROFILE
    // The ray's rise per block of horizontal distance (a hair less, for rounding) and its azimuth bin, as ff_profile
    // has them: a ring whose terrain all rises less steeply in this direction can't be hit, the ray goes on to the next.
    float rise = dir.y / max(length(dir.xz), 1e-6);
    rise -= 1e-6 + abs(rise) * 1e-5;
    int pbin = clamp(int(floor((atan2(dir.z, dir.x) + M_PI_F) * (float(PBINS) / (2.0 * M_PI_F)))), 0, PBINS - 1);
#endif
    int steps = 0;
    bool hit = false;
    float tHit = 0.0;
    int face = 2;
    uint2 col = uint2(0);
    bool onCanopy = false, onTower = false;
    float s = 1.0;
    int2 hitCell = int2(0);
    uint hitRing = 0;
    float2 hitFrac = float2(0.0);
    for (uint r = 0; r < rings && !hit; r++) {
        float4 rc = f.ring[r];
        s = rc.z;
        // Ring coordinates: cells of the ring's level, offset by a multiple of W from the world's, so the window is
        // [win, win + W) and a cell's texel is the cell mod W (at pyramid level l, mod W >> l).
        float2 win = f.ringCover[r].zw;
        float3 o = float3(rc.x, f.cam.x, rc.y);
        float3 d = float3(dir.x / s, dir.y, dir.z / s);
        float2 inv = 1.0 / d.xz;
        float2 ta = (win - o.xz) * inv, tb = (win + float(W) - o.xz) * inv;
        float2 tmn = min(ta, tb), tmx = max(ta, tb);
        float tEnter = max(max(tmn.x, tmn.y), t);
        float tLeave = min(min(tmx.x, tmx.y), tMax);
        if (!(tEnter < tLeave)) continue;
#if FF_PROFILE
        if (rise > ffUnordered(prof[r * uint(PBINS) + uint(pbin)])) { t = tLeave; continue; }   // over all its terrain here
#endif
#if FF_STATS
        stRings++;
#endif
        int lastAxis = tmn.x > tmn.y ? 0 : 1;
        int l = min(TOP, \(farFieldStartLevel));   // a ray over the terrain climbs a level a step (METALMC_FFSTART)
        float tc = tEnter;
        // The ray's level-0 cell, kept inside the window (its cell at level l is it shifted right l bits; the window's
        // cells are never negative), and its height where it entered that cell. A crossing sets the crossed axis's cell
        // to the one past the boundary and only the other from the ray: from the ray alone, rounding put rays nearly
        // along x or z back into the cell they had just left toward -x or -z, and they spent their steps there (a wedge
        // of the next ring's terrain, cliff sides, drawn near the camera; METALMC_EXP=ffepsstep steps as then). Found
        // once per move instead of from the position at every step (floors, divides and the window's clamp at the
        // step's level): offline, 7% of the march's time at the same steps.
        float2 qmax = win + float(W - 1);
        uint2 qi = uint2(clamp(o.xz + d.xz * tc, win, qmax));
        uint2 up = uint2(d.xz > 0.0);
        float yA = o.y + d.y * tc;
        for (int i = 0; i < 192; i++) {
            steps++;
            uint2 ci = qi >> uint(l);
            float hmax = float(heights.read(ci & (uint(W - 1) >> uint(l)), r, l).r);
            float2 nb = float2((ci + up) << uint(l));
            float2 tt = (nb - o.xz) * inv;
            float tExit = min(min(tt.x, tt.y), tLeave);
            float yB = o.y + d.y * tExit;
            float2 cell = float2(ci);
            bool next = false, column = false;   // on to the next cell at this level; a column to test
#if FF_STATS
            stRing[r]++;
            bool stWasClimb = stClimbed;
            stClimbed = false;
#endif
            if (hmax <= 0.0 || min(yA, yB) > hmax) {
                next = true;
#if FF_STATS
                if (l > 0) stAdv++;
#endif
                // Up a level only into a parent not tested from here: a falling ray that stays in the same parent leaves
                // it as low as before and would fail its test again (a step to fail, one to come back down). So unless
                // the ray rises, only when the boundary it crosses is the parent's too (an even one at this level).
                uint nbI = tt.x < tt.y ? ci.x + up.x : ci.y + up.y;
                if (d.y >= 0.0 || (nbI & 1u) == 0u) {
#if FF_STATS
                    stClimbed = l < TOP;
#endif
                    l = min(l + 1, TOP);
                }
            } else if (l > 0) {
#if FF_STATS
                stDesc++;
                if (stWasClimb) stWasted++;
#endif
                l--;
            } else {
                // A column in the quads' area: they draw it, so it's empty here.
                float2 cb = floor((f.ringCover[r].xy + cell * s) / 64.0);
                next = all(cb >= 0.0) && all(cb < float(COVER)) && cover[int(cb.y) * COVER + int(cb.x)] != 0;
                column = !next;
#if FF_STATS
                stDip++;
#endif
            }
            if (column) {
                uint2 cw = data.read(uint2(int2(cell) & (W - 1)), r).rg;
                // Water surfaces sit 10/9 block below the voxel's top, as the LOD draws them (kWaterSurfaceDrop).
                float top = float((cw.x & 511u) + ((cw.x >> 9) & 127u)) - (((cw.x >> 9) & 127u) != 0u ? 10.0 / 9.0 : 0.0);
                // The canopy, a slab from its underside to its top over the ground (LodFarCell.words). Or a tower: the
                // ground's high part, a box over a rectangle of the cell up to its top, on the base word 0 holds.
                bool tower = (cw.y >> 31) != 0u;
                bool slab = cw.y != 0u && !tower;
                float cTop = float(cw.y & 511u), cBot = float((cw.y >> 9) & 511u);
                int sideFace = lastAxis == 0 ? (d.x > 0.0 ? 1 : 0) : (d.z > 0.0 ? 5 : 4);
                // The ray enters the column through a side (below the ground's top, or into the slab), else meets a top on
                // its way down (the slab's from over it, the ground's from under it) or the slab's underside on its way
                // up. Between the ground and the slab it passes under the trees to the next column.
                int kind = 0;   // 1: the ground (or water), 2: the canopy
                if (yA <= top) { kind = 1; tHit = tc; face = sideFace; }
                else if (slab && yA >= cBot && yA <= cTop) { kind = 2; tHit = tc; face = sideFace; }
                else if (d.y < 0.0) {
                    if (slab && yA > cTop) {
                        if (yB <= cTop) { kind = 2; tHit = (cTop - o.y) / d.y; face = 2; }
                    } else if (yB <= top) { kind = 1; tHit = (top - o.y) / d.y; face = 2; }
                } else if (slab && yA < cBot && yB >= cBot) { kind = 2; tHit = (cBot - o.y) / d.y; face = 3; }
                if (tower && !(kind == 1 && tHit <= tc)) {
                    uint rr = (cw.y >> 9) & 4095u;
                    float2 r0 = cell + float2(float(rr & 7u), float((rr >> 6) & 7u)) * 0.125;
                    float2 r1 = cell + float2(float(((rr >> 3) & 7u) + 1u), float(((rr >> 9) & 7u) + 1u)) * 0.125;
                    float pTop = float(cw.y & 511u);
                    float2 ta = (r0 - o.xz) * inv, tb = (r1 - o.xz) * inv;
                    float2 tn = min(ta, tb), tf = max(ta, tb);
                    float tyIn = d.y < 0.0 ? (pTop - o.y) / d.y : (o.y <= pTop ? -INFINITY : INFINITY);
                    float tyOut = d.y > 0.0 ? (pTop - o.y) / d.y : INFINITY;
                    float txz = max(tn.x, tn.y);
                    float tIn = max(max(txz, tyIn), tc), tOut = min(min(min(tf.x, tf.y), tyOut), tExit);
                    if (tIn <= tOut && (kind == 0 || tIn < tHit)) {
                        kind = 3;
                        tHit = tIn;
                        face = tIn <= tc ? sideFace : (tyIn >= txz ? 2 : (tn.x >= tn.y ? (d.x > 0.0 ? 1 : 0) : (d.z > 0.0 ? 5 : 4)));
                    }
                }
                if (kind != 0) {
                    tHit = clamp(tHit, tc, tExit);
                    col = cw;
                    onCanopy = kind == 2;
                    onTower = kind == 3;
                    hitCell = int2(cell);
                    hitRing = r;
                    hitFrac = saturate(o.xz + d.xz * tHit - cell);
                    hit = true;
#if FF_STATS
                    stHitStep = uint(i);
#endif
                    break;
                }
#if FF_STATS
                stColMiss++;
#endif
                next = true;
            }
            if (next) {
                if (tExit >= tLeave) break;
                lastAxis = tt.x < tt.y ? 0 : 1;
                if (\(experiments.contains("ffepsstep") ? "true" : "false")) {
                    tc = tExit + max(tExit * 1e-5, 1e-3);
                    yA = o.y + d.y * tc;
                    qi = uint2(clamp(o.xz + d.xz * tc, win, qmax));
                } else {
                    tc = tExit;
                    yA = yB;
                    // The cell past the crossed boundary (nb: this step's level, l may have risen), the other axis's from
                    // the ray, chosen before the one clamp to the window.
                    qi = uint2(clamp(select(o.xz + d.xz * tExit, nb - float2(1u - up), bool2(lastAxis == 0, lastAxis != 0)), win, qmax));
                }
            }
        }
        t = tLeave;
    }
    FFOut out;
#if LIT_MODE
    out.gbuf = uint2(0u);
#endif
#if FF_STATS
    {
        uint px = uint(pos.y) * uint(f.viewport.x) + uint(pos.x);
        stats[2 * px] = uint4(uint(steps) | stRings << 16, stAdv | stDesc << 16, stDip | stColMiss << 16,
                              (hit ? hitRing : 15u) | stWasted << 8 | stHitStep << 20 | 1u << 31);
        stats[2 * px + 1] = uint4(min(stRing[0], 255u) | min(stRing[1], 255u) << 8 | min(stRing[2], 255u) << 16 | min(stRing[3], 255u) << 24,
                                  min(stRing[4], 255u) | min(stRing[5], 255u) << 8 | min(stRing[6], 255u) << 16 | min(stRing[7], 255u) << 24,
                                  0u, 0u);
    }
#endif
    if (!hit) {
        if (\(farFieldSteps ? "true" : "false")) { out.color = float4(0.0, 0.0, 0.25, 1.0); out.depth = 0.0; return out; }
        discard_fragment();
        return out;
    }
    float3 rel = dir * tHit;
    float y = f.cam.x + rel.y;
    uint solid = col.x & 511u, depthVox = (col.x >> 9) & 127u;   // blocks
    float solidTop = float(solid);
    uint topMat = (col.x >> 16) & 255u, lowMat = col.x >> 24;
    // Under a canopy the ground gets less of the sky.
    float sky = max(0.0, float(u.seamInfo.z) - (col.y != 0u && (col.y >> 31) == 0u && !onCanopy ? kUnderCanopy : 0.0));
    float3 light = lodLight(u, lightmap, ls, sky);
    // About one texel per pixel: the pixel's footprint on the face, in blocks, times 16 texels per block.
    float3 n = face == 2 || face == 3 ? float3(0.0, 1.0, 0.0) : (face < 2 ? float3(1.0, 0.0, 0.0) : float3(0.0, 0.0, 1.0));
    float footprint = tHit * 2.0 / (f.viewport.y * u.proj[1][1]) / max(abs(dot(dir, n)), 0.05);
    float mip = max(0.0, log2(footprint * 16.0));
    float3 color;
#if LIT_MODE
    // Lit mode: the G-buffer's albedo, AO, face and sky light; water and what's seen through it stay unlit (LIT_NONE).
    float3 litAlbedo = float3(0.0);
    float litAO = 1.0, litSky = sky;
    uint litFace = LIT_NONE;\(litWater ? "\n    bool litWaterTop = false;   // water (METALMC_EXP=water): a water surface, flagged for the relight's reflections" : "")
#endif
    if (onCanopy) {
        // The canopy: its top with the corner occlusion of taller crowns around it, its sides darker toward the underside
        // (the crown shades itself), its underside lit by what gets under the trees.
        uint cm = (col.y >> 18) & 255u;
        float cTop = float(col.y & 511u), cBot = float((col.y >> 9) & 511u);
        float3 lit = face == 3 ? lodLight(u, lightmap, ls, max(0.0, sky - kUnderCanopy)) : light;
        float ao = face == 2 ? columnAO(data, hitCell, hitRing, s, 2, cTop, hitFrac, y, true)
                 : (face == 3 ? 1.0 : mix(kAO[1], 1.0, saturate((y - cBot) / max(1.0, cTop - cBot))));
#if LIT_MODE
        litAlbedo = colors[cm * 3u + (face == 2 ? 0u : (face == 3 ? 2u : 1u))].rgb * detail(u, sprites, atlas, atlasSampler, cm, face, rel, mip);
        color = litAlbedo * kShade[face] * lit * ao;
        litAO = ao;
        litSky = face == 3 ? max(0.0, sky - kUnderCanopy) : sky;
        litFace = uint(face);
#else
        color = colors[cm * 3u + (face == 2 ? 0u : (face == 3 ? 2u : 1u))].rgb * kShade[face] * lit
              * detail(u, sprites, atlas, atlasSampler, cm, face, rel, mip) * ao;
#endif
    } else if (onTower) {
        // A tower's high part: its own top block, and its sides of it too where it differs from the base's (a stone peak
        // on grass), else the base's side block under the top few blocks; the sides darken toward the base.
        float pTop = float(col.y & 511u);
        uint pm = (col.y >> 21) & 255u;
        uint mat = (face == 2 || y >= pTop - min(s, 8.0) || pm != topMat) ? pm : lowMat;
        float ao = face == 2 ? 1.0 : mix(kSideFoot, 1.0, saturate((y - solidTop) / s));
#if LIT_MODE
        litAlbedo = colors[mat * 3u + (face == 2 ? 0u : 1u)].rgb * detail(u, sprites, atlas, atlasSampler, mat, face, rel, mip);
        litAO = ao;
        litFace = uint(face);
        color = litAlbedo * kShade[face] * light * ao;
#else
        color = colors[mat * 3u + (face == 2 ? 0u : 1u)].rgb * kShade[face] * light
              * detail(u, sprites, atlas, atlasSampler, mat, face, rel, mip) * ao;
#endif
    } else if (depthVox > 0u && face == 2) {
        // Water over its floor, like the LOD's translucent water over its meshed floor (lit for the water above it).
        float depthBlocks = min(15.0, float(depthVox) - 1.0);
        float3 floorColor = colors[topMat * 3u].rgb * detail(u, sprites, atlas, atlasSampler, topMat, 2, rel, mip)
                          * lodLight(u, lightmap, ls, max(0.0, sky - depthBlocks));
        float a = f.cam.y;
        color = colors[lowMat * 3u].rgb * light * a + floorColor * (1.0 - a);\(litWater ? "\n#if LIT_MODE\n        litWaterTop = true;\n#endif" : "")
    } else if (depthVox > 0u) {
        // The side of a water column. Rays that pass under the quads' water reach the first far-field column from the
        // side, below its surface: that's terrain seen through the water (the quads' water surface is drawn over it),
        // so it's the floor (or the solid side), lit for the water above it, not another water surface.
        float waterTop = float(solid + depthVox) - 10.0 / 9.0;
        bool side = y < solidTop;
        float depthBlocks = min(15.0, waterTop - (side ? y : solidTop));
        color = colors[topMat * 3u + (side ? 1u : 0u)].rgb * (side ? kShade[face] : 1.0)
              * detail(u, sprites, atlas, atlasSampler, topMat, side ? face : 2, rel, mip) * lodLight(u, lightmap, ls, max(0.0, sky - depthBlocks));
    } else {
        uint mat = (face == 2 || y >= solidTop - min(s, 8.0)) ? topMat : lowMat;
#if LIT_MODE
        litAlbedo = colors[mat * 3u + (face == 2 ? 0u : 1u)].rgb * detail(u, sprites, atlas, atlasSampler, mat, face, rel, mip);
        litAO = columnAO(data, hitCell, hitRing, s, face, solidTop, hitFrac, y, false);
        litFace = uint(face);
        color = litAlbedo * kShade[face] * light * litAO;
#else
        color = colors[mat * 3u + (face == 2 ? 0u : 1u)].rgb * kShade[face] * light
              * detail(u, sprites, atlas, atlasSampler, mat, face, rel, mip)
              * columnAO(data, hitCell, hitRing, s, face, solidTop, hitFrac, y, false);
#endif
    }
    if (depthVox == 0u || onCanopy || onTower) {
        // Land and trees (METALMC_FFGAIN): as occlusion, so lit mode's G-buffer predicts this color and relights it alike.
        color *= \(farFieldGain);
#if LIT_MODE
        litAO *= \(farFieldGain);
#endif
    }
    if (\(farFieldSteps ? "true" : "false")) color = mix(float3(0.0, 1.0, 0.0), float3(1.0, 0.0, 0.0), saturate(float(steps) / 128.0));
    float horiz = length(rel.xz);
    float fog = max(linearFog(length(rel), u.envStart, u.envEnd), linearFog(max(horiz, abs(rel.y)), u.rdStart, u.rdEnd));
    out.color = float4(mix(color, u.fogColor.rgb, fog * u.fogColor.a), 1.0);
    float4 clip = u.proj * (u.view * float4(rel, 1.0));
    out.depth = min(clip.z / clip.w, pos.z);
#if LIT_MODE
    out.gbuf = \(litWater ? "litWaterTop ? litPackWater(2u, out.depth, litSky, 0.0) : " : "")litPack(litAlbedo, litAO, litFace, out.depth, litSky, 0.0);
#endif
    return out;
}
"""

/// Fill and clear parameters (must match FillParams and ClearParams in the shader).
struct FarFillParams {
    var nodeCell: SIMD2<Int32>
    var base: SIMD2<Int32>
    var size: SIMD2<Int32>
    var ring: UInt32
    var level: UInt32
}
struct FarClearParams {
    var base: SIMD2<Int32>
    var size: SIMD2<Int32>
    var ring: UInt32
    var pad: UInt32 = 0
}

final class FarField: @unchecked Sendable {
    static let shared = FarField()

    private var library: MTLLibrary?
    private var drawLibrary: MTLLibrary?   // offline (setDrawSource): the draw's shaders from another source
    /// Offline (FarFieldDebug, FARTEST_STATS): the march's per-pixel counters (a shader with FF_STATS set).
    var statsBuffer: MTLBuffer?
    /// The last draw's horizon profile pass (a command buffer of its own), for the offline timing.
    private(set) var lastProfileCB: MTLCommandBuffer?
    private var clearPipe: MTLComputePipelineState?
    private var fillPipe: MTLComputePipelineState?
    private var mipPipe: MTLComputePipelineState?
    private var drawPipes: [String: MTLRenderPipelineState] = [:]
    private var profilePipe: MTLComputePipelineState?   // ff_profile
    private var profileBuffer: MTLBuffer?
    private var compiling = false
    private let lock = NSLock()

    private var data: MTLTexture?          // RG32Uint, W x W x rings: packed columns (LodBuild.farColumns)
    private var heights: MTLTexture?       // R16Uint with mips, W x W x rings: column tops in blocks above the world bottom
    private var mipViews: [MTLTexture] = []
    private var rings = 0
    private var origins: [SIMD2<Int>] = []  // per ring: the window's first cell (of its level) for the current camera
    private var filledOrigins: [SIMD2<Int>] = []   // per ring: the same for the window it holds
    private var filledNodes: [[SIMD2<Int>: Int]] = []   // per ring: the nodes written into it (first cell: columns hash)
    private var filledTops: [Int] = []                  // per ring: their highest column top (blocks above the bottom)
    private var filled: [Bool] = []
    private var lastFill: [Int] = []
    private var coverOrigin = SIMD2<Int>(0, 0)
    private var cover = [UInt8](repeating: 0, count: farFieldCoverCells * farFieldCoverCells)
    private var coverBuffer: MTLBuffer?    // the bitmap the current frame draws with (a new one whenever it changes)
    private var lastCover: [UInt8] = []
    private(set) var ready = false
    private var maxTop = Double(lodWorldHeight)   // highest terrain in the rings' nodes (blocks above the world bottom)
    var fills = 0

    /// Compiles the shaders in the background; nil until they're ready.
    private func ensurePipelines() -> Bool {
        lock.lock(); defer { lock.unlock() }
        if clearPipe != nil { return true }
        if compiling { return false }
        compiling = true
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            do {
                let lib = try ctx.device.makeLibrary(source: farFieldShaderSource, options: nil)
                let c = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "ff_clear")!)
                let f = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "ff_fill")!)
                let m = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "ff_mip")!)
                let p = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "ff_profile")!)
                lock.lock(); library = lib; clearPipe = c; fillPipe = f; mipPipe = m; profilePipe = p; lock.unlock()
                log("far field: shaders compiled")
            } catch {
                log("far field: shader compile failed: \(error)")
            }
        }
        return false
    }

    /// Offline (FarFieldDebug): draws with the shaders of `source` (a variant of farFieldShaderSource), or the built-in
    /// ones again for nil. The rings' fills keep the built-in shaders.
    func setDrawSource(_ source: String?) -> Bool {
        var lib: MTLLibrary?
        if let source {
            do { lib = try ctx.device.makeLibrary(source: source, options: nil) } catch {
                log("far field: draw source failed: \(error)")
                return false
            }
        }
        lock.lock(); drawLibrary = lib; drawPipes = [:]; lock.unlock()
        return true
    }

    private func drawPipe(colorFormats: [MTLPixelFormat], depth: MTLPixelFormat, builtin: Bool = false) -> MTLRenderPipelineState? {
        let key = colorFormats.map { String($0.rawValue) }.joined(separator: ",") + "/\(depth.rawValue)" + (builtin ? "/b" : "")
        lock.lock(); defer { lock.unlock() }
        if let p = drawPipes[key] { return p }
        guard let library = builtin ? library : drawLibrary ?? library else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.label = "MetalMC far field"
        d.vertexFunction = library.makeFunction(name: "ff_vs")
        d.fragmentFunction = library.makeFunction(name: "ff_fs")
        for (i, f) in colorFormats.enumerated() {
            d.colorAttachments[i].pixelFormat = f
            // Extra targets are left alone, except lit mode's G-buffer (Lit.swift).
            if i > 0 && !litWritesGbuffer(i, f) { d.colorAttachments[i].writeMask = [] }
        }
        d.depthAttachmentPixelFormat = depth
        do {
            let p = try ctx.device.makeRenderPipelineState(descriptor: d)
            drawPipes[key] = p
            return p
        } catch {
            log("far field: pipeline failed: \(error)")
            return nil
        }
    }

    private func ensureTextures(rings n: Int) -> Bool {
        if n == rings, data != nil { return true }
        let d = MTLTextureDescriptor()
        d.textureType = .type2DArray
        d.width = farFieldWidth
        d.height = farFieldWidth
        d.arrayLength = n
        d.pixelFormat = .rg32Uint
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        guard let dt = ctx.device.makeTexture(descriptor: d) else { return false }
        d.pixelFormat = .r16Uint
        d.mipmapLevelCount = farFieldMips
        guard let ht = ctx.device.makeTexture(descriptor: d) else { return false }
        dt.label = "MetalMC far field columns"
        ht.label = "MetalMC far field heights"
        mipViews = (0..<farFieldMips).compactMap { ht.makeTextureView(pixelFormat: .r16Uint, textureType: .type2DArray, levels: $0..<($0 + 1), slices: 0..<n) }
        data = dt
        heights = ht
        rings = n
        filledNodes = [[SIMD2<Int>: Int]](repeating: [:], count: n)
        filledTops = [Int](repeating: 0, count: n)
        filled = [Bool](repeating: false, count: n)
        filledOrigins = [SIMD2<Int>](repeating: .zero, count: n)
        lastFill = [Int](repeating: 0, count: n)
        ready = false
        return mipViews.count == farFieldMips
    }

    @inline(__always) private static func floorDiv(_ a: Int, _ b: Int) -> Int { a >= 0 ? a / b : -((-a + b - 1) / b) }

    /// Before the LOD draws: rewrites the rings if the camera moved far enough, the nodes changed or the area drawn by
    /// quads changed. Returns true if the far field draws this frame (then the LOD skips its levels' quads).
    func prepare(meshes: [LodNodeKey: LodMeshNode], generation: Int, chosen: [(LodMeshNode, UInt16)], maxLevel: Int,
                 cx: Double, cz: Double) -> Bool {
        let k = lodFarFieldLevel
        guard k > 0, maxLevel >= k, ensurePipelines(), ensureTextures(rings: min(8, maxLevel - k + 1)) else { return false }
        camX = cx
        camZ = cz
        // Ring windows: the camera's cell, rounded down to 64 cells, minus half the width.
        origins = (0..<rings).map { r in
            let s = 1 << (k + r)
            let ccx = Self.floorDiv(Int(cx.rounded(.down)), s), ccz = Self.floorDiv(Int(cz.rounded(.down)), s)
            return SIMD2(Self.floorDiv(ccx, 64) * 64 - farFieldWidth / 2, Self.floorDiv(ccz, 64) * 64 - farFieldWidth / 2)
        }
        // The quads' area: chosen tiles below the far field's level, in 64-block cells.
        let half = farFieldCoverCells / 2
        coverOrigin = SIMD2(Self.floorDiv(Int(cx.rounded(.down)), 1024) * 16 - half, Self.floorDiv(Int(cz.rounded(.down)), 1024) * 16 - half)
        for i in 0..<cover.count { cover[i] = 0 }
        for (n, mask) in chosen where n.level < k {
            let tileBlocks = lodTileVoxels << n.level, span = tileBlocks / 64
            for t in 0..<16 where mask & (1 << UInt16(t)) != 0 {
                let bx = Self.floorDiv(n.x0 + (t % 4) * tileBlocks, 64) - coverOrigin.x
                let bz = Self.floorDiv(n.z0 + (t / 4) * tileBlocks, 64) - coverOrigin.y
                for z in max(0, bz)..<min(farFieldCoverCells, bz + span) {
                    for x in max(0, bx)..<min(farFieldCoverCells, bx + span) { cover[z * farFieldCoverCells + x] = 1 }
                }
            }
        }
        if cover != lastCover || coverBuffer == nil {
            guard let b = ctx.device.makeBuffer(bytes: cover, length: cover.count, options: [.storageModeShared]) else { return false }
            coverBuffer = b
            lastCover = cover
        }
        // Each ring's nodes in its window, and the rings whose window moved or whose nodes came, went or changed columns.
        let W = farFieldWidth
        var perRing = [[SIMD2<Int>: LodMeshNode]](repeating: [:], count: rings)
        for (key, node) in meshes {
            let r = key.level - k
            guard r >= 0, r < rings, node.columns != nil else { continue }
            let cell = SIMD2(node.x0 >> key.level, node.z0 >> key.level)
            let o = origins[r]
            if cell.x + 256 <= o.x || cell.y + 256 <= o.y || cell.x >= o.x + W || cell.y >= o.y + W { continue }
            perRing[r][cell] = node
        }
        func stale(_ r: Int) -> Bool {
            if !filled[r] || filledOrigins[r] != origins[r] || perRing[r].count != filledNodes[r].count { return true }
            for (cell, node) in perRing[r] where filledNodes[r][cell] != node.columnsHash { return true }
            return false
        }
        let staleRings = (0..<rings).filter(stale)
        if staleRings.isEmpty { return ready }
        let r = staleRings.min { lastFill[$0] < lastFill[$1] }!   // the one waiting longest
        // What to write (rectangles of cells of the ring's level, low corner and size): the cells the window moved onto
        // (all of it the first time or after a jump of a window or more) get cleared and every node's columns there;
        // elsewhere only nodes that are new or changed get written, and the cells of nodes that went get cleared.
        let o = origins[r], o1 = o &+ SIMD2(W, W), old = filledOrigins[r], old1 = old &+ SIMD2(W, W)
        let fresh = !filled[r] || abs(o.x - old.x) >= W || abs(o.y - old.y) >= W
        func clip(_ a0: SIMD2<Int>, _ a1: SIMD2<Int>, _ b0: SIMD2<Int>, _ b1: SIMD2<Int>) -> (SIMD2<Int>, SIMD2<Int>)? {
            let lo = SIMD2(max(a0.x, b0.x), max(a0.y, b0.y)), hi = SIMD2(min(a1.x, b1.x), min(a1.y, b1.y))
            return lo.x < hi.x && lo.y < hi.y ? (lo, hi &- lo) : nil
        }
        var exposed: [(SIMD2<Int>, SIMD2<Int>)] = []   // low and high corners
        if fresh {
            exposed = [(o, o1)]
        } else {
            if o.x < old.x { exposed.append((o, SIMD2(old.x, o1.y))) } else if o.x > old.x { exposed.append((SIMD2(old1.x, o.y), o1)) }
            let sx0 = max(o.x, old.x), sx1 = min(o1.x, old1.x)
            if o.y < old.y { exposed.append((SIMD2(sx0, o.y), SIMD2(sx1, old.y))) } else if o.y > old.y { exposed.append((SIMD2(sx0, old1.y), SIMD2(sx1, o1.y))) }
        }
        var clears: [(SIMD2<Int>, SIMD2<Int>)] = []
        for (lo, hi) in exposed { if let c = clip(lo, hi, o, o1) { clears.append(c) } }
        if !fresh {
            let k0 = SIMD2(max(o.x, old.x), max(o.y, old.y)), k1 = SIMD2(min(o1.x, old1.x), min(o1.y, old1.y))   // held and still in view
            for (cell, _) in filledNodes[r] where perRing[r][cell] == nil {
                if let c = clip(cell, cell &+ SIMD2(256, 256), k0, k1) { clears.append(c) }
            }
        }
        var writes: [(LodMeshNode, SIMD2<Int>, SIMD2<Int>, SIMD2<Int>)] = []   // node, its first cell, rectangle
        for (cell, node) in perRing[r] {
            let n1 = cell &+ SIMD2(256, 256)
            if fresh || filledNodes[r][cell] != node.columnsHash {
                if let (b, sz) = clip(cell, n1, o, o1) { writes.append((node, cell, b, sz)) }
            } else {
                for (lo, hi) in exposed { if let (b, sz) = clip(cell, n1, lo, hi) { writes.append((node, cell, b, sz)) } }
            }
        }
        guard let data, let clearPipe, let fillPipe, let mipPipe,
              let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return false }
        cb.label = "MetalMC far field fill"
        var cells = 0
        enc.setTexture(data, index: 0)
        enc.setTexture(mipViews[0], index: 1)
        enc.setComputePipelineState(clearPipe)
        for (b, sz) in clears {
            var p = FarClearParams(base: SIMD2(Int32(b.x), Int32(b.y)), size: SIMD2(Int32(sz.x), Int32(sz.y)), ring: UInt32(r))
            enc.setBytes(&p, length: MemoryLayout<FarClearParams>.stride, index: 0)
            enc.dispatchThreads(MTLSize(width: sz.x, height: sz.y, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            cells += sz.x * sz.y
        }
        enc.setComputePipelineState(fillPipe)
        for (node, cell, b, sz) in writes {
            var p = FarFillParams(nodeCell: SIMD2(Int32(cell.x), Int32(cell.y)), base: SIMD2(Int32(b.x), Int32(b.y)),
                                  size: SIMD2(Int32(sz.x), Int32(sz.y)), ring: UInt32(r), level: UInt32(k + r))
            enc.setBytes(&p, length: MemoryLayout<FarFillParams>.stride, index: 0)
            enc.setBuffer(node.columns, offset: 0, index: 1)
            enc.dispatchThreads(MTLSize(width: sz.x, height: sz.y, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            cells += sz.x * sz.y
        }
        var slice = UInt32(r)
        enc.setComputePipelineState(mipPipe)
        enc.setBytes(&slice, length: 4, index: 0)
        for m in 1..<farFieldMips {
            enc.setTexture(mipViews[m - 1], index: 0)
            enc.setTexture(mipViews[m], index: 1)
            let w = max(1, farFieldWidth >> m)
            enc.dispatchThreads(MTLSize(width: w, height: w, depth: 1), threadsPerThreadgroup: MTLSize(width: min(16, w), height: min(16, w), depth: 1))
        }
        enc.endEncoding()
        // Committed now, ahead of the frame's own command buffer (committed at the end of the frame), so this frame's
        // draw reads the new rings.
        cb.commit()
        self.fills += 1
        filled[r] = true
        filledOrigins[r] = origins[r]
        filledNodes[r] = perRing[r].mapValues { $0.columnsHash }
        // The highest terrain in what the rings hold (each as of its last fill), not in the nodes they will hold: until a
        // ring's refill its old nodes are still drawn, and a shell top under them cut them off.
        filledTops[r] = perRing[r].values.map { $0.columnsTop }.max() ?? 0
        maxTop = Double(min(filledTops.max() ?? 0, lodWorldHeight))
        lastFill[r] = self.fills
        if self.fills % 20 == 1 || farFieldLogFills {
            log("far field: fill \(self.fills): ring \(r) of \(rings) from level \(k), \(fresh ? "all" : "\(clears.count) cleared and \(writes.count) written rectangles,") \(cells) cells, \(staleRings.count) stale")
        }
        ready = ready || filled.allSatisfy { $0 }
        return ready
    }

    /// Draws the march inside the LOD's pass (after its opaque quads). `u` is the LOD's uniforms for this frame. `builtin`
    /// (offline): with the built-in shaders whatever setDrawSource set.
    func draw(_ enc: MTLRenderCommandEncoder, u: LodUniforms, colors: MTLBuffer, lightmap: MTLTexture, lightSampler: MTLSamplerState?,
              cy: Double, nearest: Double, builtin: Bool = false) {
        guard ready, let data, let heights, let coverBuffer,
              let pipe = drawPipe(colorFormats: ctx.passColorFormats, depth: ctx.passDepthFormat, builtin: builtin) else { return }
        let k = lodFarFieldLevel
        let vp = u.proj * u.view
        var f = FarUniforms(invViewProj: vp.inverse,
                            viewport: SIMD4(Float(ctx.passWidth), Float(ctx.passHeight), Float(maxTop), Float(rings)),
                            cam: SIMD4(Float(cy - Double(lodWorldMinY)), lodWaterAlpha, Float(nearest),
                                       Float(max(maxTop - (cy - Double(lodWorldMinY)), 0) + 1)),
                            ring: (.zero, .zero, .zero, .zero, .zero, .zero, .zero, .zero),
                            ringCover: (.zero, .zero, .zero, .zero, .zero, .zero, .zero, .zero))
        withUnsafeMutableBytes(of: &f.ring) { raw in
            let rp = raw.bindMemory(to: SIMD4<Float>.self)
            for r in 0..<rings {
                let s = Double(1 << (k + r))
                // Ring coordinates (the march's): the world's cells less a multiple of W, so the window starts at its
                // first cell's texel.
                let base = filledOrigins[r] &- (filledOrigins[r] & SIMD2(repeating: farFieldWidth - 1))
                rp[r] = SIMD4(Float(camX / s - Double(base.x)), Float(camZ / s - Double(base.y)), Float(s), 0)
            }
        }
        withUnsafeMutableBytes(of: &f.ringCover) { raw in
            let rp = raw.bindMemory(to: SIMD4<Float>.self)
            for r in 0..<rings {
                let s = 1 << (k + r)
                let win = filledOrigins[r] & SIMD2(repeating: farFieldWidth - 1), base = filledOrigins[r] &- win
                rp[r] = SIMD4(Float(base.x * s - coverOrigin.x * 64), Float(base.y * s - coverOrigin.y * 64), Float(win.x), Float(win.y))
            }
        }
        var uu = u
        enc.setRenderPipelineState(pipe)
        enc.setDepthStencilState(ctx.depthState(compare: .greaterEqual, write: true))
        enc.setCullMode(.none)
        enc.setVertexBytes(&f, length: MemoryLayout<FarUniforms>.stride, index: 25)
        enc.setFragmentBytes(&f, length: MemoryLayout<FarUniforms>.stride, index: 25)
        enc.setVertexBytes(&uu, length: MemoryLayout<LodUniforms>.stride, index: 19)
        enc.setFragmentBytes(&uu, length: MemoryLayout<LodUniforms>.stride, index: 19)
        enc.setFragmentBuffer(colors, offset: 0, index: 26)
        enc.setFragmentTexture(data, index: 27)
        enc.setFragmentTexture(heights, index: 28)
        enc.setFragmentTexture(lightmap, index: 29)
        enc.setFragmentSamplerState(lightSampler, index: 14)
        enc.setFragmentBuffer(coverBuffer, offset: 0, index: 28)
        if let statsBuffer { enc.setFragmentBuffer(statsBuffer, offset: 0, index: 29) }
        // The horizon profile for this camera (ff_profile), in a command buffer of its own committed now: the frame's, with
        // this pass, is committed at the end of the frame, so it runs first. Its GPU time is not in the main pass's.
        lastProfileCB = nil
        if farFieldProfile {
            let len = rings * farFieldProfileBins * 4
            if (profileBuffer?.length ?? 0) < len { profileBuffer = ctx.device.makeBuffer(length: len, options: [.storageModePrivate]) }
            guard let pp = profilePipe, let pb = profileBuffer, let cb = ctx.queue.makeCommandBuffer(),
                  let bl = cb.makeBlitCommandEncoder() else { return }
            cb.label = "MetalMC far field horizon profile"
            bl.fill(buffer: pb, range: 0..<len, value: 0x80)   // as ints, under every float's (ffOrdered)
            bl.endEncoding()
            guard let ce = cb.makeComputeCommandEncoder() else { return }
            ce.setComputePipelineState(pp)
            ce.setBytes(&f, length: MemoryLayout<FarUniforms>.stride, index: 25)
            ce.setTexture(heights, index: 28)
            ce.setBuffer(pb, offset: 0, index: 30)
            let n = farFieldWidth >> farFieldProfileLevel
            ce.dispatchThreads(MTLSize(width: n, height: n, depth: rings), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            ce.endEncoding()
            cb.commit()
            lastProfileCB = cb
            enc.setFragmentBuffer(pb, offset: 0, index: 30)
        }
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 64 * 9)
        enc.setCullMode(.back)
    }
    var camX = 0.0, camZ = 0.0
}
