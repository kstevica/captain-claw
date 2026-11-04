import AppKit
import ScreenCaptureKit

// --selftest "<task>" [--followup "<question>"] [--out <dir>] [--pair]
// Runs the real app end to end against a deck — pair (with --pair), read the
// screen, send the task, follow the run, ask a follow-up — snapshotting the
// panel at each step, then prints what happened as JSON and quits.
// (CAPTAIN_FOR_MAC_NO_BROWSER=1 keeps pairing from opening a browser.)

@MainActor
enum SelfTest {
    /// --selftest-hold: the hold-key logic on synthetic events (nothing is
    /// sent to the system): a tap, a shortcut and a long hold.
    static func hold() {
        let m = HoldKeyMonitor()
        m.key = .fn
        m.seconds = 1.0
        var fired = 0, released = 0
        m.onFire = { fired += 1 }
        m.onReleaseAfterFire = { released += 1 }
        func flags(_ down: Bool, code: UInt16 = 63, extra: NSEvent.ModifierFlags = []) -> NSEvent {
            NSEvent.keyEvent(with: .flagsChanged, location: .zero, modifierFlags: (down ? NSEvent.ModifierFlags.function : NSEvent.ModifierFlags()).union(extra),
                             timestamp: 0, windowNumber: 0, context: nil, characters: "", charactersIgnoringModifiers: "",
                             isARepeat: false, keyCode: code)!
        }
        let key = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [.function], timestamp: 0, windowNumber: 0,
                                   context: nil, characters: "a", charactersIgnoringModifiers: "a", isARepeat: false, keyCode: 0)!
        var log: [String] = []
        func after(_ s: Double, _ f: @escaping () -> Void) { DispatchQueue.main.asyncAfter(deadline: .now() + s, execute: f) }
        // 1. a tap: down, up at 0.3 s
        m.feed(flags(true)); after(0.3) { m.feed(flags(false)); log.append("tap fired=\(fired)") }
        // 2. fn + a key at 0.5 s (a shortcut), held past the time
        after(0.6) { m.feed(flags(true)) }
        after(0.9) { m.feed(key) }
        after(1.9) { log.append("shortcut fired=\(fired)"); m.feed(flags(false)) }
        // 3. fn + shift: not alone
        after(2.1) { m.feed(flags(true, extra: [.shift])) }
        after(3.3) { log.append("with shift fired=\(fired)"); m.feed(flags(false)) }
        // 4. a long hold, then let go
        after(3.5) { m.feed(flags(true)) }
        after(4.7) { log.append("long hold fired=\(fired)"); m.feed(flags(false)) }
        after(4.9) {
            log.append("released=\(released)")
            let ok = log == ["tap fired=0", "shortcut fired=0", "with shift fired=0", "long hold fired=1", "released=1"]
            print(Probe.json(["log": log, "ok": ok]))
            exit(ok ? 0 : 1)
        }
    }

    static func run(_ args: [String]) {
        func value(_ flag: String) -> String? {
            args.firstIndex(of: flag).flatMap { i in i + 1 < args.count ? args[i + 1] : nil }
        }
        let task = value("--selftest") ?? "What is this?"
        let follow = value("--followup")
        let out = URL(fileURLWithPath: value("--out") ?? NSTemporaryDirectory())
        try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let captain = Captain.shared
        var report: [String: Any] = [:]
        var shots: [String] = []

        Task { @MainActor in
            @MainActor func until(_ what: String, _ seconds: Double, _ cond: () -> Bool) async -> Bool {
                let end = Date().addingTimeInterval(seconds)
                while Date() < end {
                    if cond() { return true }
                    try? await Task.sleep(nanoseconds: 100_000_000)
                }
                report["timeout"] = what
                return false
            }
            @MainActor func snap(_ name: String) async {
                try? await Task.sleep(nanoseconds: 400_000_000)
                if let png = await panelPNG() {
                    let url = out.appendingPathComponent("\(name).png")
                    try? png.write(to: url)
                    shots.append(url.path)
                }
            }
            @MainActor func finish() -> Never {
                report["shots"] = shots
                report["pair_state"] = captain.pairState
                report["me_error"] = captain.meError ?? ""
                report["connected"] = captain.connected
                report["answer"] = captain.answer
                report["error"] = captain.error ?? captain.runError ?? ""
                report["events"] = captain.events.map(\.text)
                report["files"] = captain.files.map(\.path)
                report["history"] = captain.history.map { ["asked": $0.asked, "answer": String($0.answer.prefix(200))] }
                print(Probe.json(report))
                exit(0)
            }

            if args.contains("--pair") {
                captain.openSettings?()
                captain.connect()
                guard await until("pairing", 30, { captain.connected }) else { finish() }
                report["paired"] = true
                await captain.refreshMe()
                await snapSettings(out: out, shots: &shots)
            }
            await captain.refreshMe()
            report["agent"] = captain.agent?.name ?? ""
            report["actions"] = captain.actions.map(\.label)
            captain.trigger()
            await snap("1-reading")
            guard await until("compose", 15, { captain.phase == .compose }) else { finish() }
            report["captured"] = [
                "app": captain.capture?.appName ?? "", "window": captain.capture?.windowTitle ?? "",
                "visible_chars": captain.capture?.visibleText.count ?? 0,
                "selection_chars": captain.capture?.selection.count ?? 0,
                "pointer": captain.capture?.pointer?.text ?? "",
                "screenshot": captain.capture?.screenshot?.count ?? 0,
            ]
            await snap("2-compose")
            captain.task = task
            await snap("3-typed")
            captain.send()
            guard await until("running", 20, { captain.phase == .running || captain.error != nil }) else { finish() }
            if captain.error != nil { finish() }
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            await snap("4-working")
            guard await until("answer", 90, { captain.phase == .finished }) else { finish() }
            await snap("5-answer")
            report["first_answer"] = captain.answer
            if let follow {
                captain.followUp = follow
                captain.sendFollowUp()
                guard await until("follow-up", 90, {
                    captain.phase == .finished && captain.history.count == 1 && !captain.answer.isEmpty
                }) else { finish() }
                await snap("6-followup")
            }
            finish()
        }
    }

    static func snapSettings(out: URL, shots: inout [String]) async {
        try? await Task.sleep(nanoseconds: 600_000_000)
        if let png = await windowPNG(titled: "Captain Settings") {
            let url = out.appendingPathComponent("0-settings.png")
            try? png.write(to: url)
            shots.append(url.path)
        }
    }

    /// The panel (Captain's own floating window) as a PNG.
    static func panelPNG() async -> Data? { await windowPNG(titled: nil) }

    static func windowPNG(titled: String?) async -> Data? {
        guard let content = try? await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true) else { return nil }
        let me = ProcessInfo.processInfo.processIdentifier
        let mine = content.windows.filter { $0.owningApplication?.processID == me && $0.frame.width > 100 }
        let pick = titled.map { t in mine.first { $0.title == t } } ?? mine.filter { $0.windowLayer > 0 }.max { $0.frame.height < $1.frame.height }
        guard let w = pick ?? mine.first else { return nil }
        let filter = SCContentFilter(desktopIndependentWindow: w)
        let config = SCStreamConfiguration()
        let scale = CGFloat(filter.pointPixelScale)
        config.width = Int(w.frame.width * scale)
        config.height = Int(w.frame.height * scale)
        config.showsCursor = false
        guard let img = try? await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config) else { return nil }
        return ScreenImage.png(img)
    }
}

// --demo <dir>: the panel with a fictional scene (Velora, a small insurance
// consultancy) for the website — nothing is read from the screen and no deck
// is called. Dark and light shots of composing, working and the answer.
@MainActor
enum Demo {
    static let actions: [ScreenAction] = [
        ("explain", "Explain", "lightbulb"), ("summarize", "Summarize", "summary"), ("reply", "Draft a reply", "reply"),
        ("rewrite", "Improve", "pen"), ("translate", "Translate", "languages"), ("troubleshoot", "What's wrong?", "wrench"),
        ("tasks", "Make tasks", "checklist"), ("remember", "Remember", "brain"), ("save", "Save it", "bookmark"),
    ].map { ScreenAction(id: $0.0, label: $0.1, icon: $0.2, hint: nil, needs_text: nil) }

    static let me = MeInfo(
        user: DeckUser(name: "Maya Horvat", email: "maya@velora.example"),
        deck: MeInfo.Deck(name: "Velora"),
        agents: [
            Agent(name: "Inbox Assistant", slug: "inbox-assistant", description: nil, status: "running"),
            Agent(name: "Policy Analyst", slug: "policy-analyst", description: nil, status: "running"),
            Agent(name: "Researcher", slug: "researcher", description: nil, status: "asleep"),
        ],
        screen: MeInfo.Screen(folder: "screen-clips", actions: actions))

    struct Scene {
        let name: String
        let agent: String
        let capture: ScreenCapture
        let task: String
        let asked: String
        let step: String
        let live: String
        let answer: String
        var shot = false
    }

    static func mail() -> Scene {
        var c = ScreenCapture()
        c.appName = "Mail"; c.bundleID = "com.apple.mail"
        c.windowTitle = "Re: Fleet policy renewal — Harbour Logistics"
        c.selection = "Before we sign, can you confirm two things: does the €2,500 excess stay the same if we add the four new vans, and is the telematics discount still available if the vans join mid-year?"
        c.selectionFrom = "accessibility"
        c.visibleText = String(repeating: "x", count: 3412)
        c.pointer = PointerInfo(x: 640, y: 380, app: "Mail", role: "text", text: "does the €2,500 excess stay the same")
        return Scene(
            name: "mail", agent: "inbox-assistant", capture: c,
            task: "Yes to both — telematics from the month they join. Friendly, short.",
            asked: "Draft a reply: Yes to both — telematics from the month they join. Friendly, short.",
            step: "Reading: selection.md",
            live: "Hi Tomas,\n\nThanks for checking before you sign. Yes on both counts: the €2,500 excess",
            answer: """
            Hi Tomas,

            Thanks for checking before you sign. Yes on both counts:

            - **The excess stays at €2,500** with the four new vans on the policy.
            - **The telematics discount applies** to each van from the month it joins — no need to wait for the renewal date.

            I'll send the updated schedule once the vans' registrations are in. Anything else before you sign?

            Best,
            Maya
            """)
    }

    static func clause() -> Scene {
        var c = ScreenCapture()
        c.appName = "Preview"; c.bundleID = "com.apple.Preview"
        c.windowTitle = "Harbour Logistics — Commercial Motor Wording 2026.pdf"
        c.selection = "4.3 Unattended vehicle. We will not pay for theft of or from the vehicle while it is left unattended unless all doors, windows and other openings are closed and locked and the keys are removed and kept out of sight."
        c.selectionFrom = "accessibility"
        c.visibleText = String(repeating: "x", count: 5820)
        c.moreText = String(repeating: "x", count: 41_200)
        c.screenshot = Data([0])
        c.screenshotPixels = CGSize(width: 2400, height: 1500)
        c.pointer = PointerInfo(x: 520, y: 610, app: "Preview", role: "text", text: "4.3 Unattended vehicle")
        return Scene(
            name: "clause", agent: "policy-analyst", capture: c, task: "", asked: "Explain",
            step: "Comparing with section 7, Exclusions",
            live: "**In short:** a theft from an unlocked van, or with the keys left in it",
            answer: """
            **In short:** a theft from an unlocked van, or one with the keys left in it, isn't covered — even for a minute.

            - **What counts as unattended:** nobody in or right beside the van who can see it.
            - **All four conditions** must hold: doors, windows and other openings closed *and* locked, keys removed *and* out of sight.
            - **Watch for:** couriers leaving the engine running at drops — that alone voids a theft claim.

            Section 7.2 adds the same rule for tools left in the van overnight. Worth a line in Harbour's driver briefing.
            """, shot: true)
    }

    static func run(out: URL) {
        try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let captain = Captain.shared
        let savedAgent = UserDefaults.standard.object(forKey: "agentSlug")
        Task { @MainActor in
            captain.demoMode = true
            captain.me = me
            for scene in [mail(), clause()] {
                for look in ["dark", "light"] {
                    NSApp.appearance = NSAppearance(named: look == "dark" ? .darkAqua : .aqua)
                    Prefs.shared.agentSlug = scene.agent
                    captain.capture = scene.capture
                    captain.useSelection = true
                    captain.useScreen = true
                    captain.usePointer = true
                    captain.useShot = scene.shot
                    captain.task = scene.task
                    captain.error = nil
                    captain.history = []
                    captain.phase = .compose
                    captain.showPanel?(CGPoint(x: 300, y: 300))
                    captain.focusPanel?()
                    await shot(out, "\(scene.name)-1-compose-\(look)")
                    captain.asked = scene.asked
                    captain.events = [RunEvent(kind: "status", text: scene.step, tool: nil)]
                    captain.startedAt = Date().addingTimeInterval(-6)
                    captain.answer = ""
                    captain.live = scene.live
                    captain.phase = .running
                    await shot(out, "\(scene.name)-2-working-\(look)")
                    captain.live = ""
                    captain.answer = scene.answer
                    captain.phase = .finished
                    await shot(out, "\(scene.name)-3-answer-\(look)")
                    captain.hidePanel?()
                }
            }
            // The person's own choice back as it was.
            if let savedAgent { UserDefaults.standard.set(savedAgent, forKey: "agentSlug") }
            else { UserDefaults.standard.removeObject(forKey: "agentSlug") }
            print(out.path)
            exit(0)
        }
    }

    static func shot(_ out: URL, _ name: String) async {
        try? await Task.sleep(nanoseconds: 900_000_000)
        if let png = await SelfTest.panelPNG() {
            try? png.write(to: out.appendingPathComponent("\(name).png"))
        }
    }
}
