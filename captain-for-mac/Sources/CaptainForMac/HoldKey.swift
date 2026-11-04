import AppKit

// Hold one key alone (fn by default) for a few seconds → Captain opens.
// Any other key or modifier meanwhile, or letting go early, cancels it, so
// fn + F-keys, ⌘-shortcuts and quick taps work as always. A ring at the
// pointer fills while it counts.

final class HoldKeyMonitor {
    var key: HoldKey = .fn
    var seconds: Double = 3
    /// Fires once the key was held long enough.
    var onFire: (() -> Void)?
    /// The key went up after it fired (hold-to-talk ends here).
    var onReleaseAfterFire: (() -> Void)?

    private var monitors: [Any] = []
    private var downAt: Date?
    private var timer: Timer?
    private var fired = false
    private let ring = HoldRing()

    func start() {
        stop()
        let mask: NSEvent.EventTypeMask = [.flagsChanged, .keyDown]
        if let g = NSEvent.addGlobalMonitorForEvents(matching: mask, handler: { [weak self] e in self?.handle(e) }) {
            monitors.append(g)
        }
        if let l = NSEvent.addLocalMonitorForEvents(matching: mask, handler: { [weak self] e in self?.handle(e); return e }) {
            monitors.append(l)
        }
    }

    func stop() {
        monitors.forEach { NSEvent.removeMonitor($0) }
        monitors.removeAll()
        reset()
    }

    var isHeld: Bool { downAt != nil }

    /// An event as the monitors would pass it (the self-test feeds its own).
    func feed(_ e: NSEvent) { handle(e) }

    private func handle(_ e: NSEvent) {
        guard key != .off, let code = key.keyCode else { return }
        if e.type == .keyDown {
            if !fired { reset() }
            return
        }
        if e.keyCode == code {
            if isDown(e) {
                if downAt == nil && aloneHeld(e) { begin() }
            } else {
                released()
            }
        } else if downAt != nil && !fired {
            reset()   // another modifier: a shortcut, not a hold
        }
    }

    private func isDown(_ e: NSEvent) -> Bool {
        if let bit = key.deviceBit { return (e.modifierFlags.rawValue & bit) != 0 }
        return e.modifierFlags.contains(key.flag)
    }

    private func aloneHeld(_ e: NSEvent) -> Bool {
        var others = e.modifierFlags.intersection(.deviceIndependentFlagsMask)
        others.remove(.capsLock)
        others.remove(.numericPad)
        others.remove(key.flag)
        if key == .fn { others.remove(.function) }
        return others.isEmpty
    }

    private func begin() {
        downAt = Date()
        fired = false
        timer?.invalidate()
        let t = Timer(timeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in self?.tick() }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    private func tick() {
        guard let downAt, !fired else { return }
        let held = Date().timeIntervalSince(downAt)
        let frac = min(1, held / max(0.3, seconds))
        // Quick presses stay invisible.
        if held > min(0.35, seconds * 0.5) { ring.show(progress: frac) }
        if frac >= 1 {
            fired = true
            timer?.invalidate()
            timer = nil
            ring.flash()
            onFire?()
        }
    }

    private func released() {
        let wasFired = fired
        reset()
        if wasFired { onReleaseAfterFire?() }
    }

    private func reset() {
        timer?.invalidate()
        timer = nil
        downAt = nil
        fired = false
        ring.hide()
    }
}

/// The little ring by the pointer that fills while the key is held.
final class HoldRing {
    private var panel: NSPanel?
    private var view: RingView?
    private let side: CGFloat = 46

    private func make() {
        let p = NSPanel(contentRect: NSRect(x: 0, y: 0, width: side, height: side),
                        styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        p.isOpaque = false
        p.backgroundColor = .clear
        p.hasShadow = false
        p.level = .screenSaver
        p.ignoresMouseEvents = true
        p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]
        let v = RingView(frame: NSRect(x: 0, y: 0, width: side, height: side))
        p.contentView = v
        panel = p
        view = v
    }

    func show(progress: Double) {
        if panel == nil { make() }
        guard let panel, let view else { return }
        let m = NSEvent.mouseLocation
        panel.setFrameOrigin(NSPoint(x: m.x + 14, y: m.y - side - 6))
        view.progress = progress
        view.done = false
        view.needsDisplay = true
        if !panel.isVisible { panel.orderFrontRegardless() }
    }

    func flash() {
        guard let panel, let view else { return }
        view.progress = 1
        view.done = true
        view.needsDisplay = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) { [weak panel] in panel?.orderOut(nil) }
    }

    func hide() {
        panel?.orderOut(nil)
    }
}

final class RingView: NSView {
    var progress: Double = 0
    var done = false

    override func draw(_ dirtyRect: NSRect) {
        let r = bounds.insetBy(dx: 7, dy: 7)
        let center = NSPoint(x: r.midX, y: r.midY)
        let radius = r.width / 2
        NSColor.black.withAlphaComponent(0.55).setFill()
        NSBezierPath(ovalIn: bounds.insetBy(dx: 3, dy: 3)).fill()
        let track = NSBezierPath()
        track.appendArc(withCenter: center, radius: radius, startAngle: 0, endAngle: 360)
        track.lineWidth = 4
        NSColor.white.withAlphaComponent(0.18).setStroke()
        track.stroke()
        let arc = NSBezierPath()
        arc.appendArc(withCenter: center, radius: radius, startAngle: 90, endAngle: 90 - 360 * CGFloat(progress), clockwise: true)
        arc.lineWidth = 4
        arc.lineCapStyle = .round
        (done ? NSColor.systemGreen : NSColor(calibratedRed: 0.55, green: 0.45, blue: 1.0, alpha: 1)).setStroke()
        arc.stroke()
        let dot = NSBezierPath(ovalIn: NSRect(x: center.x - 4, y: center.y - 4, width: 8, height: 8))
        NSColor.white.withAlphaComponent(done ? 1 : 0.8).setFill()
        dot.fill()
    }
}
