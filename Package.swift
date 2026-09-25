// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "MetalMC",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "MetalMCViewer",
            path: "Sources/MetalMCViewer",
            swiftSettings: [.unsafeFlags(["-Ounchecked"], .when(configuration: .release))]
        )
    ]
)
