// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "MetalMC",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "MetalMCViewer", targets: ["MetalMCViewer"]),
        // Native side of the in-game Metal backend, loaded from Java through the FFM API.
        .library(name: "MetalMCNative", type: .dynamic, targets: ["MetalMCNative"]),
    ],
    targets: [
        // World-format decoding and materials, shared by the standalone viewer and the in-game library.
        .target(
            name: "MetalMCCore",
            path: "Sources/MetalMCCore",
            swiftSettings: [.unsafeFlags(["-Ounchecked"], .when(configuration: .release))]
        ),
        .executableTarget(
            name: "MetalMCViewer",
            dependencies: ["MetalMCCore"],
            path: "Sources/MetalMCViewer",
            swiftSettings: [.unsafeFlags(["-Ounchecked"], .when(configuration: .release))]
        ),
        .target(
            name: "MetalMCNative",
            dependencies: ["MetalMCCore"],
            path: "Sources/MetalMCNative",
            swiftSettings: [.unsafeFlags(["-Ounchecked"], .when(configuration: .release))]
        ),
    ]
)
