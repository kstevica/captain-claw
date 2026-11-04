import AppKit
import ApplicationServices
import Foundation

// Reads what's on the screen through Accessibility: the front app and its
// window, the selection (the part the person means), what's under the
// pointer, and the window's text — what's on screen first, the rest after.

struct PointerInfo {
    var x: Double = 0          // screen points, top-left origin (as Accessibility counts)
    var y: Double = 0
    var app: String = ""
    var role: String = ""      // "button", "text", "link"…
    var text: String = ""      // the thing itself: its title, label or value
    var near: String = ""      // the line or paragraph around the pointer, when the app says
}

struct ScreenCapture {
    var appName = ""
    var bundleID = ""
    var pid: pid_t = 0
    var windowTitle = ""
    var url = ""
    var selection = ""
    var selectionFrom = ""     // "accessibility" | "copy"
    var visibleText = ""
    var moreText = ""          // the rest of the window or document, not on screen
    var textFrom = "accessibility"   // "accessibility" | "ocr" | "none"
    var pointer: PointerInfo?
    var windowFrame: CGRect?   // top-left origin
    var screenshot: Data?
    var screenshotPixels: CGSize = .zero
    var capturedAt = Date()
    var trusted = true
    var notes: [String] = []
    var timings: [String: Int] = [:]   // ms per step (the probe prints them)
    var stats: [String: Int] = [:]

    var hasText: Bool { !visibleText.isEmpty || !moreText.isEmpty }
}

// ── Small Accessibility helpers ──────────────────────────────────────

enum AX {
    static func raw(_ el: AXUIElement, _ name: String) -> (AXError, AnyObject?) {
        var v: AnyObject?
        let err = AXUIElementCopyAttributeValue(el, name as CFString, &v)
        return (err, v)
    }

    static func attr(_ el: AXUIElement, _ name: String) -> AnyObject? {
        let (err, v) = raw(el, name)
        return err == .success ? v : nil
    }

    static func string(_ el: AXUIElement, _ name: String) -> String? {
        guard let v = attr(el, name) else { return nil }
        if let s = v as? String { return s }
        if let a = v as? NSAttributedString { return a.string }
        return nil
    }

    static func element(_ el: AXUIElement, _ name: String) -> AXUIElement? {
        guard let v = attr(el, name), CFGetTypeID(v) == AXUIElementGetTypeID() else { return nil }
        return (v as! AXUIElement)
    }

    static func elements(_ v: AnyObject?) -> [AXUIElement] {
        guard let arr = v as? [AnyObject] else { return [] }
        return arr.compactMap { CFGetTypeID($0) == AXUIElementGetTypeID() ? ($0 as! AXUIElement) : nil }
    }

    static func point(_ v: AnyObject?) -> CGPoint? {
        guard let v, CFGetTypeID(v) == AXValueGetTypeID() else { return nil }
        var p = CGPoint.zero
        return AXValueGetValue(v as! AXValue, .cgPoint, &p) ? p : nil
    }

    static func size(_ v: AnyObject?) -> CGSize? {
        guard let v, CFGetTypeID(v) == AXValueGetTypeID() else { return nil }
        var s = CGSize.zero
        return AXValueGetValue(v as! AXValue, .cgSize, &s) ? s : nil
    }

    static func range(_ v: AnyObject?) -> CFRange? {
        guard let v, CFGetTypeID(v) == AXValueGetTypeID() else { return nil }
        var r = CFRange(location: 0, length: 0)
        return AXValueGetValue(v as! AXValue, .cfRange, &r) ? r : nil
    }

    static func frame(_ el: AXUIElement) -> CGRect? {
        guard let p = point(attr(el, kAXPositionAttribute)), let s = size(attr(el, kAXSizeAttribute)) else { return nil }
        return CGRect(origin: p, size: s)
    }

    static func param(_ el: AXUIElement, _ name: String, _ arg: AnyObject) -> AnyObject? {
        var v: AnyObject?
        return AXUIElementCopyParameterizedAttributeValue(el, name as CFString, arg, &v) == .success ? v : nil
    }

    /// Several attributes in one round trip; a missing one comes back nil.
    static func many(_ el: AXUIElement, _ names: [String]) -> [AnyObject?] {
        var out: CFArray?
        let err = AXUIElementCopyMultipleAttributeValues(el, names as CFArray, AXCopyMultipleAttributeOptions(rawValue: 0), &out)
        guard err == .success, let arr = out as [AnyObject]?, arr.count == names.count else {
            return names.map { attr(el, $0) }
        }
        return arr.map { v in
            if CFGetTypeID(v) == AXValueGetTypeID(), AXValueGetType(v as! AXValue) == .axError { return nil }
            if v is NSNull { return nil }
            return v
        }
    }

    static func text(_ v: AnyObject?) -> String? {
        if let s = v as? String { return s }
        if let a = v as? NSAttributedString { return a.string }
        return nil
    }

    static func pid(_ el: AXUIElement) -> pid_t {
        var p: pid_t = 0
        AXUIElementGetPid(el, &p)
        return p
    }

    static func cfValue(_ r: CFRange) -> AXValue? {
        var r = r
        return AXValueCreate(.cfRange, &r)
    }

    static func cfValue(_ p: CGPoint) -> AXValue? {
        var p = p
        return AXValueCreate(.cgPoint, &p)
    }
}

// ── The reader ───────────────────────────────────────────────────────

final class ScreenReader {
    static let chromium: Set<String> = [
        "com.google.Chrome", "com.google.Chrome.beta", "com.google.Chrome.dev", "com.google.Chrome.canary",
        "com.brave.Browser", "com.microsoft.edgemac", "company.thebrowser.Browser", "company.thebrowser.dia", "com.vivaldi.Vivaldi",
        "com.operasoftware.Opera", "org.chromium.Chromium",
    ]

    private var webOpened: Set<pid_t> = []

    static var trusted: Bool { AXIsProcessTrusted() }

    static func askForTrust() {
        let opt = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        _ = AXIsProcessTrustedWithOptions([opt: true] as CFDictionary)
    }

    /// Chromium browsers and Electron apps build their accessibility tree only
    /// when asked; ask once per process (and give it a moment the first time).
    private func openWebAccessibility(_ app: NSRunningApplication, _ appEl: AXUIElement) -> Bool {
        let bundle = app.bundleIdentifier ?? ""
        let electron = app.bundleURL.map {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("Contents/Frameworks/Electron Framework.framework").path)
        } ?? false
        guard Self.chromium.contains(bundle) || electron else { return false }
        guard !webOpened.contains(app.processIdentifier) else { return false }
        webOpened.insert(app.processIdentifier)
        let ok = AXUIElementSetAttributeValue(appEl, "AXManualAccessibility" as CFString, kCFBooleanTrue) == .success
        if !ok && Self.chromium.contains(bundle) {
            AXUIElementSetAttributeValue(appEl, "AXEnhancedUserInterface" as CFString, kCFBooleanTrue)
        }
        return true
    }

    static let terminals: Set<String> = [
        "com.apple.Terminal", "com.googlecode.iterm2", "com.mitchellh.ghostty", "dev.warp.Warp-Stable",
        "net.kovidgoyal.kitty", "org.alacritty", "com.github.wez.wezterm", "co.zeit.hyper",
    ]
    static let browsers: Set<String> = chromium.union(["com.apple.Safari", "com.apple.SafariTechnologyPreview",
                                                        "org.mozilla.firefox", "org.mozilla.firefoxdeveloperedition"])

    /// What's on screen in *app* (the one the person was in), with the
    /// pointer at *mouse* (top-left screen points).
    func capture(app: NSRunningApplication, mouse: CGPoint, prefs: Prefs) -> ScreenCapture {
        var cap = ScreenCapture()
        cap.appName = app.localizedName ?? ""
        cap.bundleID = app.bundleIdentifier ?? ""
        cap.pid = app.processIdentifier
        cap.trusted = ScreenReader.trusted
        guard cap.trusted else {
            cap.textFrom = "none"
            cap.notes.append("Accessibility isn't allowed yet, so Captain can't read the screen's text.")
            return cap
        }
        var clock = Date()
        func lap(_ name: String) {
            cap.timings[name] = Int(Date().timeIntervalSince(clock) * 1000)
            clock = Date()
        }
        let appEl = AXUIElementCreateApplication(app.processIdentifier)
        AXUIElementSetMessagingTimeout(appEl, 0.25)
        if openWebAccessibility(app, appEl) {
            Thread.sleep(forTimeInterval: 0.45)
        }

        let window = AX.element(appEl, kAXFocusedWindowAttribute) ?? AX.element(appEl, kAXMainWindowAttribute)
            ?? AX.elements(AX.attr(appEl, kAXWindowsAttribute)).first
        if let window {
            cap.windowTitle = AX.string(window, kAXTitleAttribute) ?? ""
            cap.windowFrame = AX.frame(window)
        }
        let focused = AX.element(appEl, kAXFocusedUIElementAttribute)
        lap("window")

        // The window's text (and the page, when there is one).
        var web: AXUIElement? = nil
        if prefs.includeScreenText, let window {
            let screenRect = NSScreen.screens.reduce(CGRect.null) { $0.union(Self.topLeft($1.frame)) }
            let viewport = (cap.windowFrame ?? screenRect).intersection(screenRect)
            let walker = TextWalker(viewport: viewport.isNull ? screenRect : viewport,
                                    tailFirst: Self.terminals.contains(cap.bundleID))
            walker.walk(window)
            cap.visibleText = walker.visibleText()
            cap.moreText = walker.hiddenText()
            web = walker.webArea
            cap.stats = walker.stats
            if !cap.hasText { cap.textFrom = "none" }
        } else {
            cap.textFrom = "none"
        }
        lap("text")
        if web == nil {
            web = findWebArea(from: focused, window: Self.browsers.contains(cap.bundleID) ? window : nil)
        }
        if let web, let u = AX.attr(web, "AXURL") {
            if let url = u as? URL { cap.url = url.absoluteString } else if let s = u as? String { cap.url = s }
        }
        if cap.url.isEmpty, let window, let doc = AX.string(window, kAXDocumentAttribute) {
            cap.url = doc
        }
        // A page's address, not an app's insides (vscode-file://, file://…).
        if !(cap.url.hasPrefix("http://") || cap.url.hasPrefix("https://")) { cap.url = "" }
        lap("url")

        // The selection: what the person means.
        let (sel, supported) = readSelection(focused: focused, web: web)
        if !sel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            cap.selection = sel
            cap.selectionFrom = "accessibility"
        } else if !supported && prefs.copyFallback {
            if let copied = copySelection(), !copied.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                cap.selection = copied
                cap.selectionFrom = "copy"
            }
        }
        lap("selection")

        if prefs.includePointer {
            cap.pointer = readPointer(mouse)
        }
        lap("pointer")
        return cap
    }

    /// An NSScreen frame (bottom-left origin) in Accessibility's top-left points.
    static func topLeft(_ r: CGRect) -> CGRect {
        let h = NSScreen.screens.first?.frame.height ?? r.maxY
        return CGRect(x: r.minX, y: h - r.maxY, width: r.width, height: r.height)
    }

    private func findWebArea(from focused: AXUIElement?, window: AXUIElement?) -> AXUIElement? {
        var el = focused
        var hops = 0
        while let cur = el, hops < 40 {
            if AX.string(cur, kAXRoleAttribute) == "AXWebArea" { return cur }
            el = AX.element(cur, kAXParentAttribute)
            hops += 1
        }
        guard let window else { return nil }
        var queue: [AXUIElement] = [window]
        var seen = 0
        while !queue.isEmpty && seen < 600 {
            let cur = queue.removeFirst()
            seen += 1
            if AX.string(cur, kAXRoleAttribute) == "AXWebArea" { return cur }
            queue.append(contentsOf: AX.elements(AX.attr(cur, kAXChildrenAttribute)))
        }
        return nil
    }

    /// The selected text, and whether the app can tell at all (when it can
    /// and says nothing is selected, nothing is).
    private func readSelection(focused: AXUIElement?, web: AXUIElement?) -> (String, Bool) {
        var supported = false
        if let focused {
            let (err, v) = AX.raw(focused, kAXSelectedTextAttribute)
            if err == .success {
                supported = true
                if let s = AX.text(v), !s.isEmpty { return (s, true) }
            }
        }
        // Web pages (Safari, Chrome…): the selection is a text-marker range.
        for el in [web, focused].compactMap({ $0 }) {
            if let range = AX.attr(el, "AXSelectedTextMarkerRange") {
                supported = true
                if let s = AX.text(AX.param(el, "AXStringForTextMarkerRange", range)), !s.isEmpty { return (s, true) }
            }
        }
        return ("", supported)
    }

    /// ⌘C into the clipboard and back: for apps that don't share their
    /// selection. The clipboard is put back as it was.
    private func copySelection() -> String? {
        let pb = NSPasteboard.general
        let saved: [[(NSPasteboard.PasteboardType, Data)]] = (pb.pasteboardItems ?? []).map { item in
            item.types.compactMap { t in item.data(forType: t).map { (t, $0) } }
        }
        let before = pb.changeCount
        let src = CGEventSource(stateID: .hidSystemState)
        let down = CGEvent(keyboardEventSource: src, virtualKey: 8, keyDown: true)
        let up = CGEvent(keyboardEventSource: src, virtualKey: 8, keyDown: false)
        down?.flags = .maskCommand
        up?.flags = .maskCommand
        down?.post(tap: .cghidEventTap)
        up?.post(tap: .cghidEventTap)
        let start = Date()
        while pb.changeCount == before && Date().timeIntervalSince(start) < 0.4 {
            Thread.sleep(forTimeInterval: 0.02)
        }
        guard pb.changeCount != before else { return nil }
        let text = pb.string(forType: .string)
        pb.clearContents()
        let items: [NSPasteboardItem] = saved.map { pairs in
            let item = NSPasteboardItem()
            for (t, d) in pairs { item.setData(d, forType: t) }
            return item
        }
        if !items.isEmpty { pb.writeObjects(items) }
        return text
    }

    /// What's under the pointer: the thing itself and the line around it.
    private func readPointer(_ mouse: CGPoint) -> PointerInfo {
        var info = PointerInfo(x: mouse.x, y: mouse.y)
        let system = AXUIElementCreateSystemWide()
        AXUIElementSetMessagingTimeout(system, 0.4)
        var hit: AXUIElement?
        guard AXUIElementCopyElementAtPosition(system, Float(mouse.x), Float(mouse.y), &hit) == .success, let el = hit else {
            return info
        }
        let pid = AX.pid(el)
        info.app = NSRunningApplication(processIdentifier: pid)?.localizedName ?? ""
        var cur: AXUIElement? = el
        var hops = 0
        while let c = cur, hops < 5 {
            let vals = AX.many(c, [kAXRoleAttribute, kAXTitleAttribute, kAXDescriptionAttribute, kAXValueAttribute])
            let role = (vals[0] as? String) ?? ""
            if hops == 0 { info.role = Self.roleName(role) }
            let text = [AX.text(vals[1]), AX.text(vals[2]), AX.text(vals[3])]
                .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
                .first { !$0.isEmpty } ?? ""
            if !text.isEmpty {
                if hops > 0 && info.role.isEmpty { info.role = Self.roleName(role) }
                info.text = String(text.prefix(600))
                break
            }
            cur = AX.element(c, kAXParentAttribute)
            hops += 1
        }
        info.near = nearText(el, mouse) ?? ""
        if info.near == info.text { info.near = "" }
        return info
    }

    /// The line under the pointer and the ones beside it.
    private func nearText(_ el: AXUIElement, _ mouse: CGPoint) -> String? {
        if let pt = AX.cfValue(mouse) {
            // Cocoa text (TextEdit, Notes, Mail, Xcode…): a character index, then its lines.
            // (An app with nothing there answers NSNotFound: no line.)
            if let r = AX.range(AX.param(el, kAXRangeForPositionParameterizedAttribute, pt)),
               r.location >= 0, r.location < 50_000_000,
               let lineNo = (AX.param(el, kAXLineForIndexParameterizedAttribute, NSNumber(value: r.location)) as? NSNumber)?.intValue,
               lineNo >= 0, lineNo < 10_000_000 {
                var parts: [String] = []
                for n in max(0, lineNo - 1)...(lineNo + 1) {
                    if let lr = AX.range(AX.param(el, kAXRangeForLineParameterizedAttribute, NSNumber(value: n))),
                       let v = AX.cfValue(lr),
                       let s = AX.text(AX.param(el, kAXStringForRangeParameterizedAttribute, v)) {
                        parts.append(s.trimmingCharacters(in: .newlines))
                    }
                }
                let joined = parts.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
                if !joined.isEmpty { return String(joined.prefix(1200)) }
            }
            // Web pages: a text marker, then its paragraph.
            if let marker = AX.param(el, "AXTextMarkerForPosition", pt) {
                for name in ["AXParagraphTextMarkerRangeForTextMarker", "AXLineTextMarkerRangeForTextMarker"] {
                    if let range = AX.param(el, name, marker),
                       let s = AX.text(AX.param(el, "AXStringForTextMarkerRange", range))?
                        .trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty {
                        return String(s.prefix(1200))
                    }
                }
            }
        }
        return nil
    }

    static func roleName(_ role: String) -> String {
        switch role {
        case "AXStaticText": return "text"
        case "AXTextField", "AXSearchField": return "text field"
        case "AXTextArea": return "text area"
        case "AXButton", "AXMenuButton", "AXPopUpButton": return "button"
        case "AXLink": return "link"
        case "AXImage": return "picture"
        case "AXCell", "AXRow": return "row"
        case "AXCheckBox": return "checkbox"
        case "AXRadioButton": return "option"
        case "AXHeading": return "heading"
        case "AXTab", "AXTabGroup": return "tab"
        case "AXMenuItem": return "menu item"
        case "AXWebArea": return "page"
        case "AXGroup", "": return ""
        default:
            return role.hasPrefix("AX") ? String(role.dropFirst(2)).lowercased() : role.lowercased()
        }
    }
}

// ── The window's text, in reading order ──────────────────────────────

final class TextWalker {
    struct Piece {
        var text: String
        var frame: CGRect?
        var heading: Bool
    }

    let viewport: CGRect
    let tailFirst: Bool            // a terminal: the end of its text is what's on screen
    private(set) var webArea: AXUIElement?
    private var maxDepth = 0
    private var depthCuts = 0
    private var visible: [Piece] = []
    private var hidden: [Piece] = []
    private var visibleChars = 0
    private var hiddenChars = 0
    private var nodes = 0
    private let deadline = Date().addingTimeInterval(1.6)
    private let maxNodes = 9000
    private let maxVisible = 40_000
    private let maxHidden = 60_000

    private static let skip: Set<String> = [
        "AXMenuBar", "AXMenu", "AXMenuItem", "AXScrollBar", "AXValueIndicator", "AXSplitter", "AXGrowArea",
        "AXIncrementor", "AXBusyIndicator", "AXProgressIndicator", "AXImage", "AXRuler", "AXRulerMarker",
        "AXDockItem", "AXColorWell",
    ]
    private static let names = [kAXRoleAttribute, kAXValueAttribute, kAXTitleAttribute, kAXDescriptionAttribute,
                                kAXChildrenAttribute, kAXPositionAttribute, kAXSizeAttribute]

    init(viewport: CGRect, tailFirst: Bool = false) {
        self.viewport = viewport
        self.tailFirst = tailFirst
    }

    var stats: [String: Int] {
        ["nodes": nodes, "max_depth": maxDepth, "depth_cuts": depthCuts, "pieces": visible.count + hidden.count]
    }

    private var full: Bool { visibleChars >= maxVisible && hiddenChars >= maxHidden }

    func walk(_ el: AXUIElement, depth: Int = 0, parentFrame: CGRect? = nil) {
        guard depth < 120 else { depthCuts += 1; return }
        guard nodes < maxNodes, !full, Date() < deadline else { return }
        nodes += 1
        maxDepth = max(maxDepth, depth)
        let v = AX.many(el, Self.names)
        let role = (v[0] as? String) ?? ""
        if Self.skip.contains(role) { return }
        if role == "AXWebArea" && webArea == nil { webArea = el }
        var frame: CGRect? = nil
        if let p = AX.point(v[5]), let s = AX.size(v[6]) { frame = CGRect(origin: p, size: s) }
        let box = frame ?? parentFrame

        switch role {
        case "AXStaticText":
            let t = AX.text(v[1]) ?? AX.text(v[2]) ?? AX.text(v[3]) ?? ""
            add(t, box, heading: false)
            return
        case "AXTextArea", "AXTextField", "AXComboBox", "AXSearchField":
            textField(el, value: AX.text(v[1]) ?? "", frame: box)
            return
        case "AXHeading":
            let title = AX.text(v[2]) ?? ""
            if !title.isEmpty {
                add(title, box, heading: true)
                return
            }
            let sub = TextWalker(viewport: viewport)
            for c in AX.elements(v[4]) { sub.walk(c, depth: depth + 1, parentFrame: box) }
            let t = (sub.visible + sub.hidden).map(\.text).joined(separator: " ")
            nodes += sub.nodes
            add(t, box, heading: true)
            return
        default:
            break
        }
        for c in AX.elements(v[4]) {
            walk(c, depth: depth + 1, parentFrame: box)
            if nodes >= maxNodes || Date() >= deadline { return }
        }
    }

    private func textField(_ el: AXUIElement, value: String, frame: CGRect?) {
        guard !value.isEmpty else { return }
        // A terminal's scrollback: the end is on screen, the start long gone.
        if tailFirst && value.count > 3000 {
            let shownCount = 3000
            let shown = String(value.suffix(shownCount))
            let rest = String(value.dropLast(shownCount).suffix(maxHidden))
            add(rest, nil, heading: false, forceHidden: true)
            add(shown, frame, heading: false, forceVisible: true)
            return
        }
        // A long document: what's on screen, then the rest.
        if value.count > 600, let r = AX.range(AX.attr(el, kAXVisibleCharacterRangeAttribute)), r.length > 0,
           let rv = AX.cfValue(r), let shown = AX.text(AX.param(el, kAXStringForRangeParameterizedAttribute, rv)) {
            add(shown, frame, heading: false, forceVisible: true)
            let ns = value as NSString
            let before = r.location > 0 ? ns.substring(to: min(r.location, ns.length)) : ""
            let afterStart = min(r.location + r.length, ns.length)
            let after = afterStart < ns.length ? ns.substring(from: afterStart) : ""
            if !before.isEmpty { add(before, nil, heading: false, forceHidden: true) }
            if !after.isEmpty { add(after, nil, heading: false, forceHidden: true) }
            return
        }
        add(value, frame, heading: false)
    }

    private func add(_ raw: String, _ frame: CGRect?, heading: Bool, forceVisible: Bool = false, forceHidden: Bool = false) {
        let t = raw.replacingOccurrences(of: "\u{00A0}", with: " ").replacingOccurrences(of: "\u{FFFC}", with: "")
        guard !t.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let onScreen: Bool
        if forceVisible { onScreen = true } else if forceHidden { onScreen = false } else if let f = frame {
            onScreen = f.width > 0 && f.height > 0 && f.intersects(viewport.insetBy(dx: -2, dy: -2))
        } else { onScreen = true }
        let piece = Piece(text: t, frame: frame, heading: heading)
        if onScreen {
            guard visibleChars < maxVisible else { return }
            visible.append(piece)
            visibleChars += t.count
        } else {
            guard hiddenChars < maxHidden else { return }
            hidden.append(piece)
            hiddenChars += t.count
        }
    }

    func visibleText() -> String { Self.join(visible, cap: maxVisible) }
    func hiddenText() -> String { Self.join(hidden, cap: maxHidden) }

    /// Pieces on one line join with a space; a new line starts a new line.
    static func join(_ pieces: [Piece], cap: Int) -> String {
        var out = ""
        var last: Piece?
        var lastLine = ""
        for p in pieces {
            let text = p.heading ? "## " + p.text.trimmingCharacters(in: .whitespacesAndNewlines) : p.text
            if text.trimmingCharacters(in: .whitespacesAndNewlines) == lastLine.trimmingCharacters(in: .whitespacesAndNewlines),
               text.count < 200 {
                continue  // the same label twice in a row
            }
            if let l = last {
                if !p.heading && !l.heading, let a = l.frame, let b = p.frame,
                   abs(a.midY - b.midY) < max(4, min(a.height, b.height) * 0.5), b.minX >= a.minX {
                    let glue = (out.last?.isWhitespace ?? true) || (text.first.map { ".,;:!?)]}”’".contains($0) } ?? false) ? "" : " "
                    out += glue + text
                } else {
                    out += (p.heading || l.heading) ? "\n\n" : "\n"
                    out += text
                }
            } else {
                out += text
            }
            last = p
            lastLine = text
            if out.count > cap { break }
        }
        var lines: [String] = []
        var blank = 0
        for line in out.components(separatedBy: "\n") {
            let t = line.replacingOccurrences(of: "[ \t]+", with: " ", options: .regularExpression)
                .trimmingCharacters(in: .whitespaces)
            if t.isEmpty { blank += 1; if blank > 1 { continue } } else { blank = 0 }
            lines.append(t)
        }
        let joined = lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        return joined.count > cap ? String(joined.prefix(cap)) + "\n…" : joined
    }
}
