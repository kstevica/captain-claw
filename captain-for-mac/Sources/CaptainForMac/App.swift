import AppKit
import Combine
import SwiftUI

// Captain for Mac: a menu-bar app. Click its icon (or the floating button),
// or hold fn for a few seconds: it reads what's on screen and opens a panel
// beside it to send that — with what you type or say — to one of your agents.

@main
enum Main {
    static func main() {
        let args = CommandLine.arguments
        if args.contains("--probe") {
            Probe.run(args)
            return
        }
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.setActivationPolicy(.accessory)
        app.run()
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var status: NSStatusItem?
    private var panel: PanelController?
    private var settings: NSWindow?
    private var floating: FloatingButton?
    private let hold = HoldKeyMonitor()
    private var bag: Set<AnyCancellable> = []

    func applicationDidFinishLaunching(_ note: Notification) {
        setUp()
    }

    private func setUp() {
        let captain = Captain.shared
        let prefs = Prefs.shared
        let panel = PanelController(captain: captain, prefs: prefs)
        self.panel = panel
        captain.showPanel = { [weak panel] mouse in panel?.show(near: mouse) }
        captain.hidePanel = { [weak panel] in panel?.hide() }
        captain.focusPanel = { [weak panel] in panel?.focus() }
        captain.openSettings = { [weak self] in self?.showSettings() }

        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let b = item.button {
            b.image = NSImage(systemSymbolName: "dot.viewfinder", accessibilityDescription: "Captain")
            b.image?.isTemplate = true
            b.target = self
            b.action = #selector(statusClicked(_:))
            b.sendAction(on: [.leftMouseUp, .rightMouseUp])
            b.toolTip = "Captain — click to ask about what's on screen"
        }
        status = item

        hold.onFire = { MainActor.assumeIsolated { Captain.shared.trigger() } }
        hold.onReleaseAfterFire = { MainActor.assumeIsolated { Captain.shared.keyReleased() } }
        applyHold(prefs)
        hold.start()
        prefs.$holdKey.combineLatest(prefs.$holdSeconds)
            .receive(on: RunLoop.main)
            .sink { [weak self] _, _ in DispatchQueue.main.async { self?.applyHold(Prefs.shared) } }
            .store(in: &bag)
        prefs.$floatingButton
            .receive(on: RunLoop.main)
            .sink { [weak self] on in self?.setFloating(on) }
            .store(in: &bag)

        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--demo") {
            Demo.run(out: URL(fileURLWithPath: i + 1 < args.count ? args[i + 1] : NSTemporaryDirectory()))
            return
        }
        if args.contains("--selftest-hold") {
            SelfTest.hold()
            return
        }
        if args.contains("--selftest") {
            SelfTest.run(args)
            return
        }
        if !ScreenReader.trusted { ScreenReader.askForTrust() }
        if captain.connected {
            Task { await captain.refreshMe() }
        } else {
            showSettings()
        }
    }

    private func applyHold(_ prefs: Prefs) {
        hold.key = prefs.holdKey
        hold.seconds = prefs.holdSeconds
    }

    private func setFloating(_ on: Bool) {
        if on {
            if floating == nil { floating = FloatingButton { MainActor.assumeIsolated { Captain.shared.trigger() } } }
            floating?.show()
        } else {
            floating?.hide()
        }
    }

    @objc private func statusClicked(_ sender: NSStatusBarButton) {
        let ev = NSApp.currentEvent
        if ev?.type == .rightMouseUp || ev?.modifierFlags.contains(.control) == true {
            showMenu()
        } else {
            Captain.shared.trigger()
        }
    }

    private func showMenu() {
        let menu = NSMenu()
        let prefs = Prefs.shared
        let ask = NSMenuItem(title: "Ask about the screen", action: #selector(menuAsk), keyEquivalent: "")
        ask.target = self
        menu.addItem(ask)
        if prefs.holdKey != .off {
            let hint = NSMenuItem(title: String(format: "Or hold %@ for %.1f s", prefs.holdKey.short, prefs.holdSeconds),
                                  action: nil, keyEquivalent: "")
            hint.isEnabled = false
            menu.addItem(hint)
        }
        menu.addItem(.separator())
        let deck = NSMenuItem(title: "Open Flight Deck", action: #selector(menuDeck), keyEquivalent: "")
        deck.target = self
        menu.addItem(deck)
        let set = NSMenuItem(title: "Settings…", action: #selector(menuSettings), keyEquivalent: ",")
        set.target = self
        menu.addItem(set)
        menu.addItem(.separator())
        menu.addItem(NSMenuItem(title: "Quit Captain", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
        status?.menu = menu
        status?.button?.performClick(nil)
        status?.menu = nil
    }

    @objc private func menuAsk() { Captain.shared.trigger() }
    @objc private func menuDeck() { Captain.shared.openDeck() }
    @objc private func menuSettings() { showSettings() }

    func showSettings() {
        if settings == nil {
            let host = NSHostingController(rootView: SettingsView().environmentObject(Captain.shared).environmentObject(Prefs.shared))
            let w = NSWindow(contentViewController: host)
            w.title = "Captain Settings"
            w.styleMask = [.titled, .closable, .miniaturizable]
            w.isReleasedWhenClosed = false
            w.center()
            settings = w
        }
        NSApp.activate(ignoringOtherApps: true)
        settings?.makeKeyAndOrderFront(nil)
    }
}

// ── The panel window ─────────────────────────────────────────────────

final class CaptainPanel: NSPanel {
    var onEscape: (() -> Void)?
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
    override func cancelOperation(_ sender: Any?) { onEscape?() }
}

@MainActor
final class PanelController: NSObject, NSWindowDelegate {
    let panel: CaptainPanel
    private var top: CGFloat?

    init(captain: Captain, prefs: Prefs) {
        panel = CaptainPanel(contentRect: NSRect(x: 0, y: 0, width: 460, height: 300),
                             styleMask: [.nonactivatingPanel, .borderless, .fullSizeContentView],
                             backing: .buffered, defer: false)
        super.init()
        panel.isMovableByWindowBackground = true
        panel.level = .floating
        panel.hidesOnDeactivate = false
        panel.isReleasedWhenClosed = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = true
        panel.delegate = self
        panel.onEscape = { MainActor.assumeIsolated { Captain.shared.close() } }
        let host = NSHostingController(rootView: PanelView().environmentObject(captain).environmentObject(prefs))
        host.sizingOptions = [.preferredContentSize]
        panel.contentViewController = host
        panel.contentView?.wantsLayer = true
        panel.contentView?.layer?.cornerRadius = 14
        panel.contentView?.layer?.masksToBounds = true
    }

    /// Beside the pointer's screen edge that's farther from it, top third.
    func show(near mouse: CGPoint?) {
        if let mouse, !panel.isVisible {
            let h = NSScreen.screens.first?.frame.height ?? 0
            let cocoa = NSPoint(x: mouse.x, y: h - mouse.y)
            let screen = NSScreen.screens.first { $0.frame.contains(cocoa) } ?? NSScreen.main ?? NSScreen.screens[0]
            let vf = screen.visibleFrame
            let w = panel.frame.width
            let x = cocoa.x > vf.midX ? vf.minX + 24 : vf.maxX - w - 24
            let topY = vf.maxY - 60
            top = topY
            panel.setFrameOrigin(NSPoint(x: x, y: topY - panel.frame.height))
        }
        panel.orderFrontRegardless()
    }

    /// The keyboard to the panel (the app it came from stays the active one).
    func focus() {
        panel.orderFrontRegardless()
        panel.makeKey()
    }

    func hide() {
        panel.orderOut(nil)
    }

    func windowDidResize(_ notification: Notification) {
        // Grow downwards: the top stays where it was put.
        guard let top else { return }
        let f = panel.frame
        if abs(f.maxY - top) > 0.5 {
            panel.setFrameOrigin(NSPoint(x: f.minX, y: top - f.height))
        }
    }

    func windowDidMove(_ notification: Notification) {
        top = panel.frame.maxY
    }
}

// ── The floating button ──────────────────────────────────────────────

final class FloatingButton {
    private let panel: NSPanel
    private let side: CGFloat = 44

    init(onClick: @escaping () -> Void) {
        panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: side, height: side),
                        styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        panel.isReleasedWhenClosed = false
        let v = FloatingView(frame: NSRect(x: 0, y: 0, width: side, height: side))
        v.onClick = onClick
        v.onMoved = { origin in
            UserDefaults.standard.set(NSStringFromPoint(origin), forKey: "floatingOrigin")
        }
        panel.contentView = v
        if let s = UserDefaults.standard.string(forKey: "floatingOrigin") {
            panel.setFrameOrigin(NSPointFromString(s))
        } else if let vf = NSScreen.main?.visibleFrame {
            panel.setFrameOrigin(NSPoint(x: vf.maxX - side - 16, y: vf.midY))
        }
    }

    func show() { panel.orderFrontRegardless() }
    func hide() { panel.orderOut(nil) }
}

final class FloatingView: NSView {
    var onClick: (() -> Void)?
    var onMoved: ((NSPoint) -> Void)?
    private var downAt: NSPoint?
    private var startOrigin: NSPoint?
    private var dragged = false
    private var hovering = false

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func updateTrackingAreas() {
        trackingAreas.forEach(removeTrackingArea)
        addTrackingArea(NSTrackingArea(rect: bounds, options: [.mouseEnteredAndExited, .activeAlways], owner: self))
    }

    override func mouseEntered(with event: NSEvent) { hovering = true; needsDisplay = true }
    override func mouseExited(with event: NSEvent) { hovering = false; needsDisplay = true }

    override func draw(_ dirtyRect: NSRect) {
        let r = bounds.insetBy(dx: 2, dy: 2)
        NSColor(calibratedRed: 0.13, green: 0.12, blue: 0.2, alpha: hovering ? 0.95 : 0.82).setFill()
        NSBezierPath(ovalIn: r).fill()
        NSColor(calibratedRed: 0.55, green: 0.45, blue: 1.0, alpha: 0.9).setStroke()
        let ring = NSBezierPath(ovalIn: r.insetBy(dx: 1, dy: 1))
        ring.lineWidth = 1.5
        ring.stroke()
        if let icon = NSApp.applicationIconImage {
            icon.draw(in: r.insetBy(dx: 7, dy: 7))
        }
    }

    override func mouseDown(with event: NSEvent) {
        downAt = NSEvent.mouseLocation
        startOrigin = window?.frame.origin
        dragged = false
    }

    override func mouseDragged(with event: NSEvent) {
        guard let downAt, let startOrigin, let window else { return }
        let now = NSEvent.mouseLocation
        let dx = now.x - downAt.x, dy = now.y - downAt.y
        if abs(dx) + abs(dy) > 3 { dragged = true }
        if dragged { window.setFrameOrigin(NSPoint(x: startOrigin.x + dx, y: startOrigin.y + dy)) }
    }

    override func mouseUp(with event: NSEvent) {
        if dragged {
            if let o = window?.frame.origin { onMoved?(o) }
        } else {
            onClick?()
        }
        downAt = nil
        dragged = false
    }
}

// ── --probe: read the screen once and print it (for checking the reader) ──

enum Probe {
    static func run(_ args: [String]) {
        let delay = args.firstIndex(of: "--probe").flatMap { i in i + 1 < args.count ? Double(args[i + 1]) : nil } ?? 2
        let shot = args.firstIndex(of: "--shot").flatMap { i in i + 1 < args.count ? ScreenshotMode(rawValue: args[i + 1]) : nil } ?? .off
        let app = NSApplication.shared
        app.setActivationPolicy(.prohibited)
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
            Task { @MainActor in
                let prefs = Prefs.shared
                let wanted = args.firstIndex(of: "--app").flatMap { i in i + 1 < args.count ? args[i + 1] : nil }
                let front = wanted.map { id in NSWorkspace.shared.runningApplications.first { $0.bundleIdentifier == id } }
                    ?? NSWorkspace.shared.frontmostApplication
                var out: [String: Any] = ["trusted": ScreenReader.trusted, "screen_recording": ScreenImage.allowed]
                guard let front else {
                    print(json(out))
                    exit(0)
                }
                let mouse = Captain.mouseTopLeft()
                let reader = ScreenReader()
                let started = Date()
                var cap = reader.capture(app: front, mouse: mouse, prefs: prefs)
                out["read_ms"] = Int(Date().timeIntervalSince(started) * 1000)
                if shot != .off, let img = await ScreenImage.capture(mode: shot, pid: cap.pid, windowFrame: cap.windowFrame, pointer: mouse) {
                    out["screenshot"] = "\(img.width)x\(img.height)"
                    if args.contains("--ocr") {
                        let t0 = Date()
                        let mode = args.firstIndex(of: "--ocr").flatMap { i in i + 1 < args.count ? args[i + 1] : nil } ?? "accurate"
                        let text = ScreenImage.ocr(img, mode: mode)
                        out["ocr_mode"] = mode
                        out["ocr_ms"] = Int(Date().timeIntervalSince(t0) * 1000)
                        out["ocr_chars"] = text.count
                        out["ocr_head"] = String(text.prefix(400))
                    }
                    cap.screenshot = ScreenImage.png(img)
                }
                out["app"] = cap.appName
                out["bundle"] = cap.bundleID
                out["window"] = cap.windowTitle
                out["url"] = cap.url
                out["selection"] = String(cap.selection.prefix(400))
                out["selection_from"] = cap.selectionFrom
                out["visible_chars"] = cap.visibleText.count
                out["more_chars"] = cap.moreText.count
                out["visible_head"] = String(cap.visibleText.prefix(600))
                if let p = cap.pointer {
                    out["pointer"] = ["x": Int(p.x), "y": Int(p.y), "app": p.app, "role": p.role, "text": p.text,
                                      "near": String(p.near.prefix(300))]
                }
                out["notes"] = cap.notes
                out["timings"] = cap.timings
                out["stats"] = cap.stats
                if args.contains("--doc") {
                    print(ScreenDoc.markdown(cap, screenText: true, selection: true, pointer: true))
                }
                print(json(out))
                exit(0)
            }
        }
        app.run()
    }

    static func json(_ o: Any) -> String {
        guard let d = try? JSONSerialization.data(withJSONObject: o, options: [.prettyPrinted, .sortedKeys]) else { return "{}" }
        return String(data: d, encoding: .utf8) ?? "{}"
    }
}
