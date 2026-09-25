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
        .executableTarget(
            name: "MetalMCViewer",
            path: "Sources/MetalMCViewer",
            swiftSettings: [.unsafeFlags(["-Ounchecked"], .when(configuration: .release))]
        ),
        .target(
            name: "MetalMCNative",
            path: "Sources/MetalMCNative"
        ),
    ]
)
