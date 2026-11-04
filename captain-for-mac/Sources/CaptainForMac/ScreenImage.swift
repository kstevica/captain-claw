import AppKit
import CoreGraphics
import ScreenCaptureKit
import Vision

// The screenshot (when the person asks for one) and reading text off it
// (OCR) for apps that don't share their text with Accessibility.

enum ScreenImage {
    static var allowed: Bool { CGPreflightScreenCaptureAccess() }

    static func askForPermission() { _ = CGRequestScreenCaptureAccess() }

    /// The front window of *pid* (the one nearest *windowFrame*), or the whole
    /// display under *pointer* without Captain's own windows. Long side at
    /// most *maxSide* pixels.
    static func capture(mode: ScreenshotMode, pid: pid_t, windowFrame: CGRect?, pointer: CGPoint,
                        maxSide: CGFloat = 2400) async -> CGImage? {
        guard mode != .off, allowed else { return nil }
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            let filter: SCContentFilter
            let size: CGSize
            switch mode {
            case .window:
                let mine = content.windows.filter {
                    $0.owningApplication?.processID == pid && $0.windowLayer == 0 && $0.frame.width > 40 && $0.frame.height > 40
                }
                guard !mine.isEmpty else { return nil }
                let pick: SCWindow
                if let wf = windowFrame {
                    pick = mine.min { distance($0.frame, wf) < distance($1.frame, wf) } ?? mine[0]
                } else {
                    pick = mine[0]
                }
                filter = SCContentFilter(desktopIndependentWindow: pick)
                size = pick.frame.size
            case .screen, .off:
                let display = content.displays.first { $0.frame.contains(pointer) } ?? content.displays.first
                guard let display else { return nil }
                let me = content.applications.filter { $0.processID == ProcessInfo.processInfo.processIdentifier }
                filter = SCContentFilter(display: display, excludingApplications: me, exceptingWindows: [])
                size = display.frame.size
            }
            let config = SCStreamConfiguration()
            let scale = CGFloat(filter.pointPixelScale)
            var w = size.width * scale
            var h = size.height * scale
            let longest = max(w, h)
            if longest > maxSide {
                w = w * maxSide / longest
                h = h * maxSide / longest
            }
            config.width = max(1, Int(w))
            config.height = max(1, Int(h))
            config.showsCursor = false
            config.ignoreShadowsSingleWindow = true
            return try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
        } catch {
            return nil
        }
    }

    private static func distance(_ a: CGRect, _ b: CGRect) -> CGFloat {
        abs(a.minX - b.minX) + abs(a.minY - b.minY) + abs(a.width - b.width) + abs(a.height - b.height)
    }

    static func png(_ image: CGImage) -> Data? {
        NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])
    }

    /// The text in *image*, top to bottom, on this Mac (Vision).
    static func ocr(_ image: CGImage, mode: String = "accurate") -> String {
        let req = VNRecognizeTextRequest()
        req.recognitionLevel = mode == "fast" ? .fast : .accurate
        req.usesLanguageCorrection = mode != "fast"
        if mode == "detect" {
            req.automaticallyDetectsLanguage = true
        } else {
            // The person's own languages (detecting them on every line is slow).
            let supported = (try? req.supportedRecognitionLanguages()) ?? []
            let mine = Locale.preferredLanguages.compactMap { pref -> String? in
                let base = pref.split(separator: "-").first.map(String.init) ?? pref
                return supported.first { $0 == pref } ?? supported.first { $0.hasPrefix(base + "-") || $0 == base }
            }
            var langs: [String] = []
            for l in mine + ["en-US"] where !langs.contains(l) { langs.append(l) }
            req.recognitionLanguages = langs
        }
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        do { try handler.perform([req]) } catch { return "" }
        let obs = (req.results ?? []).compactMap { o -> (CGRect, String)? in
            guard let s = o.topCandidates(1).first?.string, !s.isEmpty else { return nil }
            return (o.boundingBox, s)
        }
        // Vision's boxes start bottom-left; read top to bottom, then left to right.
        let sorted = obs.sorted { a, b in
            if abs(a.0.midY - b.0.midY) > min(a.0.height, b.0.height) * 0.5 { return a.0.midY > b.0.midY }
            return a.0.minX < b.0.minX
        }
        var out = ""
        var lastY: CGFloat?
        var lastH: CGFloat = 0
        for (box, s) in sorted {
            if let y = lastY, abs(y - box.midY) < min(lastH, box.height) * 0.5 {
                out += "  " + s
            } else {
                if lastY != nil { out += "\n" }
                out += s
            }
            lastY = box.midY
            lastH = box.height
        }
        return out
    }
}
