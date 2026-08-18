// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "AnnouncementsSummarizer",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .executable(
            name: "announcements-summarizer",
            targets: ["AnnouncementsSummarizer"]
        )
    ],
    targets: [
        .executableTarget(
            name: "AnnouncementsSummarizer",
            path: "Sources"
        )
    ]
)
