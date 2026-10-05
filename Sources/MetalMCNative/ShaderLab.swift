import Foundation
import Metal

// Lab mode, the look half (docs/lab-mode.md): our own Metal shaders from files, recompiled while the game runs.
//
// Every library of ours that is compiled from MSL at runtime goes through ShaderLab.library(name, source, ...). Without a
// shader directory that is makeLibrary(source:) and nothing more. With METALMC_SHADERDIR=<dir> (gradle -PshaderDir=<dir>):
// - The source comes from <dir>/<name>.metal, written there from the built-in source the first time. The shared headers
//   become files of their own (#include "sky_header.metal" and so on), so an edit to one reaches every library that uses
//   it: lit_relight_header.metal is the relight in lit.metal and in taa.metal (the anti-aliasing's resolve, which relights
//   as it loads when TAA is on).
// - A watcher polls the files' modification times twice a second and recompiles a library whose files changed, off the
//   render thread. If it compiles, and still has every function the old one had, the library's reload hook runs on the
//   render thread between frames (mmc_submit): it drops the old library and the pipelines made from it (or swaps new ones
//   in), so the next frame builds them from the new library. A compile error is logged with the file and line (the
//   expanded source carries #line directives) and the old pipelines keep running.
// - <dir>/.orig/ keeps the built-in source each file was written from. A file without edits follows the built-in source
//   when that changes (another builder's edit merged in); an edited one is kept, with a warning and the new built-in source
//   in .orig/<name>.metal.new to merge from.
// - Every event is also appended to <dir>/shaderlab.log.

enum ShaderLab {
    /// METALMC_SHADERDIR as an absolute path; nil: the built-in sources only (the default).
    static let dir: String? = {
        guard let d = ProcessInfo.processInfo.environment["METALMC_SHADERDIR"], !d.isEmpty else { return nil }
        let url = URL(fileURLWithPath: (d as NSString).expandingTildeInPath).standardizedFileURL
        do {
            try FileManager.default.createDirectory(at: url.appendingPathComponent(".orig"), withIntermediateDirectories: true)
        } catch {
            log("shaderlab: can't use \(url.path) (\(error)); built-in shaders only")
            return nil
        }
        return url.path
    }()

    /// Compiles one of our shader libraries. `name`: the file's name without .metal (and the library's name in the log; one
    /// file per distinct source). `reload` (shader directory only): runs on the render thread between frames when the
    /// library's files changed and compiled; it gets the new library, and the next call here returns it too, at once. It
    /// should drop the old library and what was made from it, or swap new pipelines in. One per name and options: a later
    /// call's replaces an earlier one's.
    static func library(_ name: String, _ source: String, options: MTLCompileOptions? = nil, device: MTLDevice = ctx.device,
                        reload: ((MTLLibrary) -> Void)? = nil) throws -> MTLLibrary {
        guard dir != nil else { return try device.makeLibrary(source: source, options: options) }
        return try ShaderLabFiles.shared.library(name, source, options, device, reload)
    }

    /// Render thread, between frames (mmc_submit): runs the reload hooks of libraries that changed. Nothing without a
    /// shader directory.
    @inline(__always) static func frameBoundary() {
        if dir != nil { ShaderLabFiles.shared.drain() }
    }
}

/// Lab mode (the control file's `reload`): recompiles every library from its files now, changed or not, and queues their
/// reload hooks for the next frame. Writes a summary into `out`. Returns the number that failed to compile, -1 without a
/// shader directory.
@_cdecl("mmc_shaderlab_reload")
public func mmc_shaderlab_reload(_ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    guard let dir = ShaderLab.dir else {
        writeError("no shader directory (METALMC_SHADERDIR, gradle -PshaderDir)", out, len)
        return -1
    }
    let r = ShaderLabFiles.shared.reloadAll()
    writeError("\(r.ok) of \(r.ok + r.failed.count) libraries compiled from \(dir)"
               + (r.failed.isEmpty ? "" : "; failed (the old pipelines keep running): " + r.failed.joined(separator: ", ")), out, len)
    return Int32(r.failed.count)
}

private final class ShaderLabFiles: @unchecked Sendable {
    static let shared = ShaderLabFiles()

    /// One library: a source compiled with one set of options.
    final class Variant {
        let name: String
        let options: MTLCompileOptions?
        let device: MTLDevice
        var builtin: String
        var lib: MTLLibrary?
        var text = ""                     // the expanded text `lib` came from ("" for the built-in source)
        var badText = ""                  // the expanded text that failed last (not compiled or reported again until it changes)
        var deps: [String] = []           // the files `text` came from (absolute paths, its own first)
        var stamps: [Stamp] = []          // their modification times and sizes when it was read
        var reload: ((MTLLibrary) -> Void)?
        init(name: String, options: MTLCompileOptions?, device: MTLDevice, builtin: String) {
            self.name = name
            self.options = options
            self.device = device
            self.builtin = builtin
        }
    }

    struct Stamp: Equatable {
        var seconds = 0, nanos = 0, size: Int64 = -1
    }

    /// The shared headers: written to files of their own and #included where a source holds them on whole lines.
    private let headers: [(file: String, text: String)]
    private let lock = NSLock()
    private var variants: [String: Variant] = [:]
    private var order: [String] = []        // variant keys, in first-use order
    private var synced = Set<String>()      // files checked against their built-in source this session
    private var pending: [(Variant, MTLLibrary)] = []
    private var watcher: DispatchSourceTimer?
    private let queue = DispatchQueue(label: "metalmc.shaderlab", qos: .utility)
    private let dir: String

    init() {
        dir = ShaderLab.dir ?? ""
        let all: [(file: String, text: String)] = [
            ("sky_header.metal", skyShaderHeader), ("lit_header.metal", litShaderHeader),
            ("lit_relight_header.metal", litRelightHeader), ("gi_upsample_header.metal", giUpsampleHeader)]
        headers = all.filter { $0.text.utf8.count >= 64 }
        let readme = dir + "/README.txt"
        if !FileManager.default.fileExists(atPath: readme) {
            try? """
            MetalMC lab mode: these files are the game's own Metal shaders (docs/lab-mode.md in the repository).
            The game (started with METALMC_SHADERDIR set to this directory, gradle -PshaderDir) wrote each from its built-in
            source the first time it compiled it. Edit and save one: within about a second the game recompiles it and
            rebuilds the pipelines made from it. A compile error goes to shaderlab.log (and the game log) with the file and
            line, and the old shaders keep running. *_header.metal files are shared: #include "x.metal" lines pull them in.
            .orig/ holds the built-in source each file was written from: `diff -u .orig/lit.metal lit.metal` shows your edits
            (to port back into the Swift string they came from). Delete a file to get the built-in source again.

            """.write(toFile: readme, atomically: true, encoding: .utf8)
        }
        note("shaders from \(dir): written there from the built-in sources where missing, recompiled when saved")
    }

    /// Logs to the game log and to <dir>/shaderlab.log.
    private func note(_ s: String) {
        log("shaderlab: " + s)
        let line = ShaderLabFiles.clock.string(from: Date()) + " " + s + "\n"
        let path = dir + "/shaderlab.log"
        if let h = FileHandle(forWritingAtPath: path) {
            h.seekToEndOfFile()
            h.write(line.data(using: .utf8)!)
            try? h.close()
        } else {
            try? line.write(toFile: path, atomically: false, encoding: .utf8)
        }
    }

    private static let clock: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    private static func optionsKey(_ o: MTLCompileOptions?) -> String {
        guard let o else { return "" }
        let macros = (o.preprocessorMacros ?? [:]).map { "\($0.key)=\($0.value)" }.sorted().joined(separator: ",")
        return "|\(o.languageVersion.rawValue)|\(o.preserveInvariance ? 1 : 0)|\(macros)"
    }

    func library(_ name: String, _ source: String, _ options: MTLCompileOptions?, _ device: MTLDevice,
                 _ reload: ((MTLLibrary) -> Void)?) throws -> MTLLibrary {
        let key = name + ShaderLabFiles.optionsKey(options)
        lock.lock()
        let v: Variant
        if let old = variants[key] {
            v = old
        } else {
            v = Variant(name: name, options: options?.copy() as? MTLCompileOptions, device: device, builtin: source)
            variants[key] = v
            order.append(key)
        }
        v.builtin = source
        if let reload { v.reload = reload }
        let firstUse = !synced.contains(name + ".metal")
        lock.unlock()
        if firstUse {
            // The file (and the headers it includes): written from the built-in source where missing, checked against it.
            let (text, used) = dumpText(source)
            for h in used { sync(h.file, builtin: h.text) }
            sync(name + ".metal", builtin: text)
        }
        startWatcher()
        let (text, deps, stamps) = read(name)
        lock.lock()
        v.deps = deps
        v.stamps = stamps
        if let lib = v.lib, text == nil || v.text == text {
            lock.unlock()
            return lib
        }
        let last = v.lib
        let known = text != nil && v.badText == text
        lock.unlock()
        if let text {
            if known, let last { return last }
            do {
                let t0 = DispatchTime.now().uptimeNanoseconds
                let lib = try device.makeLibrary(source: text, options: options)
                lock.lock(); v.lib = lib; v.text = text; v.badText = ""; lock.unlock()
                if firstUse {
                    note("\(name).metal compiled in \((DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000) ms"
                         + (deps.count > 1 ? " (with " + deps.dropFirst().map { ($0 as NSString).lastPathComponent }.joined(separator: ", ") + ")" : ""))
                }
                return lib
            } catch {
                if !known { report(name, error) }
                lock.lock(); v.badText = text; lock.unlock()
                if let last { return last }
                note("\(name): the built-in source until \(name).metal compiles")
            }
        }
        // No file to read, or it failed with nothing compiled before: the built-in source (throws as it always did).
        let lib = try device.makeLibrary(source: source, options: options)
        lock.lock(); v.lib = lib; v.text = ""; lock.unlock()
        return lib
    }

    /// Render thread, between frames: the reload hooks of the libraries that changed, by name.
    func drain() {
        lock.lock()
        if pending.isEmpty {
            lock.unlock()
            return
        }
        let work = pending.map { ($0.0, $0.1, $0.0.reload) }
        pending = []
        lock.unlock()
        var names: [String] = []
        for (v, lib, hook) in work {
            hook?(lib)
            if !names.contains(v.name) { names.append(v.name) }
        }
        note("reloaded " + names.joined(separator: ", ") + ": their pipelines are rebuilt from the new shaders this frame")
    }

    /// The control file's `reload`: every library recompiled from its files now (on the watcher's queue, so not twice).
    func reloadAll() -> (ok: Int, failed: [String]) {
        queue.sync { poll(force: true) }
    }

    private func startWatcher() {
        lock.lock(); defer { lock.unlock() }
        if watcher != nil { return }
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + 0.5, repeating: 0.5, leeway: .milliseconds(100))
        t.setEventHandler { [weak self] in _ = self?.poll(force: false) }
        t.resume()
        watcher = t
    }

    /// The watcher (its queue): recompiles the libraries whose files changed (all of them with `force`) and queues their
    /// reloads. Returns how many compiled and which failed.
    @discardableResult
    private func poll(force: Bool) -> (ok: Int, failed: [String]) {
        lock.lock()
        let all = order.compactMap { variants[$0] }
        lock.unlock()
        var ok = 0
        var failed: [String] = []
        var stampCache: [String: Stamp] = [:]
        for v in all {
            lock.lock()
            let deps = v.deps, stamps = v.stamps, text0 = v.text, bad0 = v.badText, old = v.lib
            lock.unlock()
            if deps.isEmpty || old == nil { continue }
            var now: [Stamp] = []
            for p in deps {
                if stampCache[p] == nil { stampCache[p] = ShaderLabFiles.stamp(p) }
                now.append(stampCache[p]!)
            }
            if !force && now == stamps { continue }
            let (read, newDeps, newStamps) = self.read(v.name)
            lock.lock(); v.deps = newDeps; v.stamps = newStamps; lock.unlock()
            guard let text = read else { continue }   // the file is gone: keep what runs
            if !force && (text == text0 || text == bad0) { continue }
            let t0 = DispatchTime.now().uptimeNanoseconds
            do {
                let lib = try v.device.makeLibrary(source: text, options: v.options)
                if let old {
                    let missing = Set(old.functionNames).subtracting(lib.functionNames).sorted()
                    if !missing.isEmpty {
                        note("\(v.name).metal compiled, but has no \(missing.joined(separator: ", ")) any more (the game looks "
                             + "them up by name): the old pipelines keep running")
                        lock.lock(); v.badText = text; lock.unlock()
                        failed.append(v.name)
                        continue
                    }
                }
                lock.lock(); v.lib = lib; v.text = text; v.badText = ""; pending.append((v, lib)); lock.unlock()
                ok += 1
                note("\(v.name).metal compiled in \((DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000) ms; reloading next frame")
            } catch {
                if force || text != bad0 { report(v.name, error) }
                lock.lock(); v.badText = text; lock.unlock()
                failed.append(v.name)
            }
        }
        return (ok, failed)
    }

    /// A compile error, with the file and line of each error (from the #line directives).
    private func report(_ name: String, _ error: Error) {
        let lines = (error as NSError).localizedDescription.split(separator: "\n").map(String.init)
        let errors = lines.filter { $0.contains("error:") }
        let shown = (errors.isEmpty ? lines : errors).prefix(12).map { "  " + $0 }
        note("\(name).metal failed to compile; the old pipelines keep running:\n" + shown.joined(separator: "\n")
             + (errors.count > 12 ? "\n  (\(errors.count - 12) more errors)" : ""))
    }

    private static func stamp(_ path: String) -> Stamp {
        var st = stat()
        guard stat(path, &st) == 0 else { return Stamp() }
        return Stamp(seconds: st.st_mtimespec.tv_sec, nanos: st.st_mtimespec.tv_nsec, size: Int64(st.st_size))
    }

    /// A library's file, expanded (nil if it can't be read), the files it came from and their stamps. Read again if a
    /// file changed while it was read, so the stamps never claim a newer file than the text.
    private func read(_ name: String) -> (String?, [String], [Stamp]) {
        var result: (String?, [String], [Stamp]) = (nil, [], [])
        for _ in 0..<3 {
            var deps: [String] = []
            let before = result.1.map(ShaderLabFiles.stamp)
            let text = expand(dir + "/" + name + ".metal", &deps, depth: 0)
            let stamps = deps.map(ShaderLabFiles.stamp)
            result = (text, deps, stamps)
            if before == stamps { break }
        }
        return result
    }

    /// The name in an `#include "name.metal"` line, if the line is one.
    private static func includeName(_ line: Substring) -> String? {
        let t = line.drop { $0 == " " || $0 == "\t" }
        guard t.hasPrefix("#include") else { return nil }
        let rest = t.dropFirst("#include".count).drop { $0 == " " || $0 == "\t" }
        guard rest.first == "\"", let end = rest.dropFirst().firstIndex(of: "\"") else { return nil }
        let name = String(rest[rest.index(after: rest.startIndex)..<end])
        return name.hasSuffix(".metal") && !name.contains("/") ? name : nil
    }

    /// A file's text with its #include "x.metal" lines replaced by those files (from the directory), and #line directives
    /// so that compile errors name the file and line they're in. `deps` gets every file read.
    private func expand(_ path: String, _ deps: inout [String], depth: Int) -> String? {
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
        deps.append(path)
        var out = "#line 1 \"\(path)\"\n"
        var n = 0
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            n += 1
            if depth < 8, let inc = ShaderLabFiles.includeName(line), let body = expand(dir + "/" + inc, &deps, depth: depth + 1) {
                out += body + "\n#line \(n + 1) \"\(path)\"\n"
                continue
            }
            out += line
            out += "\n"
        }
        return out
    }

    /// The text to write for a built-in source: each shared header it holds on whole lines becomes an #include line, if that
    /// expands back to the same code (checked: the same lines, blank ones aside); otherwise the source as it is.
    private func dumpText(_ source: String) -> (String, [(file: String, text: String)]) {
        var text = source
        var used: [(file: String, text: String)] = []
        for h in headers.sorted(by: { $0.text.utf8.count > $1.text.utf8.count }) {
            var out = ""
            var rest = Substring(text)
            var found = false
            while let r = rest.range(of: h.text) {
                let before = rest[..<r.lowerBound]
                let prev: Character? = before.last ?? out.last
                let next: Character? = rest[r.upperBound...].first
                out += before
                let startOK = prev == nil || prev == "\n" || h.text.first == "\n"
                let endOK = next == nil || next == "\n" || h.text.last == "\n"
                if startOK && endOK {
                    out += (prev == nil || prev == "\n" ? "" : "\n") + "#include \"\(h.file)\"" + (next == nil || next == "\n" ? "" : "\n")
                    found = true
                } else {
                    out += rest[r]
                }
                rest = rest[r.upperBound...]
            }
            out += rest
            if found {
                text = out
                used.append(h)
            }
        }
        if used.isEmpty { return (source, []) }
        // Expand with the built-in headers and compare.
        var back = ""
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            if let inc = ShaderLabFiles.includeName(line), let h = used.first(where: { $0.file == inc }) {
                back += h.text + "\n"
            } else {
                back += line + "\n"
            }
        }
        func lines(_ s: String) -> [Substring] { s.split(separator: "\n").filter { !$0.allSatisfy { $0 == " " || $0 == "\t" } } }
        if lines(back) != lines(source) {
            note("the shared headers don't split cleanly out of a source; writing it whole")
            return (source, [])
        }
        return (text, used)
    }

    /// Checks a file against the built-in source it should hold (once a session): writes it if it's missing, follows the
    /// built-in source if the file had no edits, keeps the file's edits (with a warning if the built-in source moved).
    private func sync(_ file: String, builtin: String) {
        lock.lock()
        if synced.contains(file) {
            lock.unlock()
            return
        }
        synced.insert(file)
        lock.unlock()
        let path = dir + "/" + file, origPath = dir + "/.orig/" + file
        let current = try? String(contentsOfFile: path, encoding: .utf8)
        let base = try? String(contentsOfFile: origPath, encoding: .utf8)
        func write(_ p: String, _ s: String) {
            do { try s.write(toFile: p, atomically: true, encoding: .utf8) } catch { note("can't write \(p): \(error)") }
        }
        guard let current else {
            write(path, builtin)
            write(origPath, builtin)
            note("wrote \(file) (the built-in source)")
            return
        }
        if current == builtin {
            if base != builtin { write(origPath, builtin) }
            return
        }
        guard let base else {
            write(origPath, builtin)
            note("\(file) isn't the built-in source and has no record of what it came from: using the file")
            return
        }
        if current == base {
            write(path, builtin)
            write(origPath, builtin)
            note("\(file): the built-in source changed and the file had no edits: rewritten from it")
        } else if builtin != base {
            write(origPath + ".new", builtin)
            note("WARNING \(file) has edits, and the built-in source changed since it was written: using the file. The new "
                 + "built-in source is .orig/\(file).new (diff it against .orig/\(file) and merge; or delete \(file) to start over)")
        } else {
            note("\(file) has edits: using them")
        }
    }
}
