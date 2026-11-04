// swift-tools-version:5.9
// Captain for Mac — what's on your screen, to one of your agents.
// Build the app with ./build.sh (this package makes the binary inside it).
import PackageDescription

let package = Package(
    name: "CaptainForMac",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "CaptainForMac",
            path: "Sources/CaptainForMac",
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("ApplicationServices"),
                .linkedFramework("ScreenCaptureKit"),
                .linkedFramework("Vision"),
                .linkedFramework("Speech"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("Security"),
            ]
        ),
    ]
)
