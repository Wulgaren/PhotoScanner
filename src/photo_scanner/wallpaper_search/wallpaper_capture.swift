import AppKit
import CoreGraphics
import Foundation
import ImageIO
import ScreenCaptureKit
import UniformTypeIdentifiers

@main
struct Main {
    static func main() async {
        _ = NSApplication.shared
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            FileHandle.standardError.write(Data("usage: wallpaper-capture <out.png>\n".utf8))
            exit(2)
        }
        let out = args[1]
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
            guard let window = content.windows.first(where: {
                ($0.owningApplication?.applicationName ?? "") == "WindowManager"
                    && ($0.title ?? "") == "Wallpaper"
            }) else {
                FileHandle.standardError.write(Data("no WindowManager Wallpaper window\n".utf8))
                exit(1)
            }
            let filter = SCContentFilter(desktopIndependentWindow: window)
            let config = SCStreamConfiguration()
            config.width = max(1, Int(window.frame.width * 2))
            config.height = max(1, Int(window.frame.height * 2))
            config.showsCursor = false
            config.captureResolution = .best
            let shot = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
            let url = URL(fileURLWithPath: out)
            guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
                FileHandle.standardError.write(Data("could not write \(out)\n".utf8))
                exit(1)
            }
            CGImageDestinationAddImage(dest, shot, nil)
            if !CGImageDestinationFinalize(dest) {
                FileHandle.standardError.write(Data("failed to finalize \(out)\n".utf8))
                exit(1)
            }
        } catch {
            FileHandle.standardError.write(Data("ERROR: \(error)\n".utf8))
            exit(1)
        }
    }
}
