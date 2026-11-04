import AppKit
import Combine
import Foundation

/// One question and its answer, kept while the panel follows up on a capture.
struct Exchange: Identifiable {
    let id = UUID()
    let asked: String
    let answer: String
    let failed: Bool
}

/// The app's state: what was captured, what the person asks, the agent's run.
@MainActor
final class Captain: ObservableObject {
    static let shared = Captain()

    let prefs = Prefs.shared
    let deck = DeckClient()
    let reader = ScreenReader()
    let voice = VoiceInput()

    enum Phase { case capturing, compose, sending, running, finished }

    @Published var phase: Phase = .compose
    @Published var capture: ScreenCapture?
    @Published var useSelection = true
    @Published var useScreen = true
    @Published var usePointer = true
    @Published var useShot = true
    @Published var task = ""
    @Published var error: String?

    @Published var me: MeInfo?
    @Published var meError: String?
    @Published var loadingMe = false

    // The run
    @Published var clipID: String?
    @Published var folder: String?
    @Published var asked = ""
    @Published var events: [RunEvent] = []
    @Published var live = ""
    @Published var answer = ""
    @Published var runError: String?
    @Published var files: [RunFile] = []
    @Published var startedAt = Date()
    @Published var history: [Exchange] = []
    @Published var followUp = ""
    @Published var note: String?          // a passing line ("Copied")

    // Pairing
    @Published var pairing: PairStart?
    @Published var pairState = ""

    var showPanel: ((CGPoint?) -> Void)?
    var focusPanel: (() -> Void)?
    var hidePanel: (() -> Void)?
    var openSettings: (() -> Void)?

    private var nextEvent = 0
    private var pollTask: Task<Void, Never>?
    private var pairTask: Task<Void, Never>?
    private var lastApp: NSRunningApplication?
    private var sourceApp: NSRunningApplication?
    private var bag: Set<AnyCancellable> = []
    private var captureSerial = 0

    private init() {
        deck.base = prefs.deckBase
        deck.token = Self.testToken ?? Keychain.token(for: prefs.deckBase)
        prefs.$deckURL
            .dropFirst()
            .removeDuplicates()
            .debounce(for: .milliseconds(600), scheduler: RunLoop.main)
            .sink { [weak self] _ in self?.deckChanged() }
            .store(in: &bag)
        NSWorkspace.shared.notificationCenter.publisher(for: NSWorkspace.didActivateApplicationNotification)
            .sink { [weak self] n in
                guard let app = n.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication,
                      app.processIdentifier != ProcessInfo.processInfo.processIdentifier else { return }
                self?.lastApp = app
            }
            .store(in: &bag)
        voice.onFinish = { [weak self] text, _ in self?.voiceDone(text) }
        lastApp = NSWorkspace.shared.frontmostApplication
    }

    /// For checking the app against a test deck: a token that never touches the Keychain.
    static let testToken = ProcessInfo.processInfo.environment["CAPTAIN_FOR_MAC_TOKEN"]
    static let testing = ProcessInfo.processInfo.environment["CAPTAIN_FOR_MAC_NO_BROWSER"] == "1"

    static let sentFiles: Set<String> = ["clip.json", "screen.md", "selection.md", "screenshot.png"]

    static var label: String { "Captain for Mac on \(Host.current().localizedName ?? "this Mac")" }

    var connected: Bool { deck.connected }

    var agent: Agent? {
        guard let agents = me?.agents, !agents.isEmpty else { return nil }
        return agents.first { $0.slug == prefs.agentSlug } ?? agents.first { $0.running } ?? agents.first
    }

    var actions: [ScreenAction] {
        (me?.screen?.actions ?? Self.fallbackActions).filter { $0.id != "auto" && $0.id != "ask" }
    }

    static let fallbackActions: [ScreenAction] = [
        ScreenAction(id: "explain", label: "Explain", icon: "lightbulb", hint: nil, needs_text: nil),
        ScreenAction(id: "summarize", label: "Summarize", icon: "summary", hint: nil, needs_text: nil),
        ScreenAction(id: "reply", label: "Draft a reply", icon: "reply", hint: nil, needs_text: nil),
    ]

    var appForPaste: NSRunningApplication? { sourceApp }

    // ── The deck ──

    private func deckChanged() {
        // The same address written again (a field re-setting it) is no change.
        guard prefs.deckBase != deck.base else { return }
        pairTask?.cancel()
        pairing = nil
        pairState = ""
        deck.base = prefs.deckBase
        deck.token = Keychain.token(for: prefs.deckBase)
        me = nil
        meError = nil
        if connected { Task { await refreshMe() } }
    }

    func refreshMe() async {
        guard connected else {
            me = nil
            return
        }
        loadingMe = true
        defer { loadingMe = false }
        do {
            let info = try await deck.me()
            me = info
            meError = nil
            if !info.agents.contains(where: { $0.slug == prefs.agentSlug }),
               let pick = info.agents.first(where: { $0.running }) ?? info.agents.first {
                prefs.agentSlug = pick.slug
            }
        } catch let e as DeckError where e.status == 401 {
            Keychain.removeToken(for: deck.base)
            deck.token = nil
            me = nil
            meError = "Flight Deck doesn't know this Mac any more — connect it again."
        } catch {
            meError = error.localizedDescription
        }
    }

    func connect() {
        let base = prefs.deckBase
        guard !base.isEmpty else {
            pairState = "Type the Flight Deck address first (like http://localhost:25080)."
            return
        }
        pairTask?.cancel()
        deck.base = base
        deck.token = nil
        me = nil
        pairState = "Asking Flight Deck…"
        pairTask = Task { [weak self] in
            guard let self else { return }
            do {
                let p = try await self.deck.pairStart(label: Self.label)
                self.pairing = p
                self.pairState = "Approve the code in the browser window that opened."
                if !Self.testing, let url = URL(string: base + p.approve_path) { NSWorkspace.shared.open(url) }
                let until = Date().addingTimeInterval(TimeInterval(max(60, p.expires_in)))
                while Date() < until && !Task.isCancelled {
                    try await Task.sleep(nanoseconds: UInt64(max(1, p.interval)) * 1_000_000_000)
                    let r = try await self.deck.pairPoll(p.pair_id)
                    switch r.status {
                    case "approved":
                        guard let token = r.token else { break }
                        if !Self.testing { Keychain.setToken(token, for: base) }
                        self.deck.token = token
                        self.pairing = nil
                        self.pairState = ""
                        await self.refreshMe()
                        return
                    case "denied":
                        self.pairing = nil
                        self.pairState = "Turned away in Flight Deck."
                        return
                    case "expired":
                        self.pairing = nil
                        self.pairState = "The code expired — press Connect again."
                        return
                    default:
                        continue
                    }
                }
                if !Task.isCancelled {
                    self.pairing = nil
                    self.pairState = "The code expired — press Connect again."
                }
            } catch is CancellationError {
                return
            } catch {
                self.pairing = nil
                self.pairState = error.localizedDescription
            }
        }
    }

    func cancelPairing() {
        pairTask?.cancel()
        pairing = nil
        pairState = ""
    }

    func disconnect() {
        let base = deck.base
        Task {
            await deck.disconnect()
            Keychain.removeToken(for: base)
            deck.token = nil
            me = nil
        }
    }

    // ── A capture ──

    /// The button, the menu or the held key: read what's on screen, open the panel.
    func trigger() {
        if phase == .sending || phase == .running {
            showPanel?(nil)
            focusPanel?()
            return
        }
        let front = NSWorkspace.shared.frontmostApplication
        let app = (front?.processIdentifier == ProcessInfo.processInfo.processIdentifier ? nil : front) ?? lastApp
        let mouse = Self.mouseTopLeft()
        sourceApp = app
        resetRun()
        task = ""
        error = nil
        capture = nil
        phase = .capturing
        // Shown, but the keyboard stays with the app until the reading is
        // done: the ⌘C fallback for the selection must reach that app.
        showPanel?(mouse)
        if me == nil && connected { Task { await refreshMe() } }
        guard let app else {
            capture = ScreenCapture()
            phase = .compose
            focusPanel?()
            return
        }
        captureSerial += 1
        let serial = captureSerial
        let reader = self.reader
        let prefs = self.prefs
        Task.detached(priority: .userInitiated) {
            var cap = reader.capture(app: app, mouse: mouse, prefs: prefs)
            let mode = prefs.screenshotMode
            var image: CGImage? = nil
            if mode != .off {
                image = await ScreenImage.capture(mode: mode, pid: cap.pid, windowFrame: cap.windowFrame, pointer: mouse)
                if image == nil && !ScreenImage.allowed {
                    cap.notes.append("Screen Recording isn't allowed, so there's no screenshot.")
                }
            }
            if prefs.ocrFallback && prefs.includeScreenText && cap.visibleText.count < 120 && ScreenImage.allowed {
                var source = mode == .window ? image : nil
                if source == nil {
                    source = await ScreenImage.capture(mode: .window, pid: cap.pid, windowFrame: cap.windowFrame, pointer: mouse)
                }
                if let source {
                    let text = ScreenImage.ocr(source)
                    if text.count > cap.visibleText.count + 40 {
                        if !cap.visibleText.isEmpty {
                            cap.moreText = [cap.visibleText, cap.moreText].filter { !$0.isEmpty }.joined(separator: "\n\n")
                        }
                        cap.visibleText = text
                        cap.textFrom = "ocr"
                    }
                }
            }
            if let image {
                cap.screenshot = ScreenImage.png(image)
                cap.screenshotPixels = CGSize(width: image.width, height: image.height)
            }
            let done = cap
            await MainActor.run {
                guard serial == self.captureSerial else { return }
                self.captured(done)
            }
        }
    }

    private func captured(_ cap: ScreenCapture) {
        capture = cap
        useSelection = !cap.selection.isEmpty
        useScreen = cap.hasText
        usePointer = !(cap.pointer?.text ?? "").isEmpty || !(cap.pointer?.near ?? "").isEmpty
        useShot = cap.screenshot != nil
        phase = .compose
        focusPanel?()
        if prefs.voiceMode == .auto { startVoice() }
    }

    static func mouseTopLeft() -> CGPoint {
        let m = NSEvent.mouseLocation
        let h = NSScreen.screens.first?.frame.height ?? 0
        return CGPoint(x: m.x, y: h - m.y)
    }

    // ── Voice ──

    func startVoice() {
        guard prefs.voiceMode != .off else { return }
        voice.start(locale: prefs.voiceLocale, onDevice: prefs.onDeviceSpeech)
    }

    func toggleVoice() {
        voice.listening ? voice.stop() : startVoice()
    }

    /// The held key went up after it opened Captain: hold-to-talk ends.
    func keyReleased() {
        if voice.listening { voice.stop() }
    }

    private func voiceDone(_ text: String) {
        guard !text.isEmpty else { return }
        switch phase {
        case .compose:
            task = task.isEmpty ? text : task.trimmingCharacters(in: .whitespaces) + " " + text
            if prefs.voiceAutoSend { send() }
        case .finished, .running:
            followUp = followUp.isEmpty ? text : followUp + " " + text
            if prefs.voiceAutoSend && phase == .finished { sendFollowUp() }
        default:
            break
        }
    }

    // ── Sending ──

    func send(action: String? = nil) {
        guard phase == .compose, let cap = capture else { return }
        if voice.listening {
            let heard = voice.text
            voice.cancel()
            if !heard.isEmpty { task = task.isEmpty ? heard : task + " " + heard }
        }
        guard connected else {
            error = "Connect Captain to your Flight Deck first."
            openSettings?()
            return
        }
        guard let agent else {
            error = me == nil ? "Still loading your agents — try again in a moment." : "You have no agents to send to."
            return
        }
        let note = task.trimmingCharacters(in: .whitespacesAndNewlines)
        let act = action ?? "auto"
        var screen: [String: Any] = [
            "app": cap.appName, "bundle": cap.bundleID, "window": cap.windowTitle, "url": cap.url,
            "text_from": useScreen ? cap.textFrom : "none",
        ]
        if usePointer, let p = cap.pointer, !(p.text.isEmpty && p.near.isEmpty) {
            screen["pointer"] = ["role": p.role, "text": p.text, "app": p.app, "x": Int(p.x), "y": Int(p.y)]
        }
        let meta: [String: Any] = [
            "kind": "screen", "agent": agent.slug, "action": act, "note": note, "browser": Self.label,
            "captured_at": ISO8601DateFormatter().string(from: cap.capturedAt), "screen": screen,
        ]
        var parts = [FormPart(name: "screen", filename: "screen.md", type: "text/markdown",
                              data: Data(ScreenDoc.markdown(cap, screenText: useScreen, selection: useSelection,
                                                            pointer: usePointer).utf8))]
        if useSelection && !cap.selection.isEmpty {
            parts.append(FormPart(name: "selection", filename: "selection.md", type: "text/markdown",
                                  data: Data(ScreenDoc.selection(cap).utf8)))
        }
        if useShot, let png = cap.screenshot {
            parts.append(FormPart(name: "screenshot", filename: "screenshot.png", type: "image/png", data: png))
        }
        let label = actions.first { $0.id == act }?.label
        error = nil
        phase = .sending
        Task {
            do {
                let r = try await deck.send(meta: meta, parts: parts)
                folder = r.folder
                history = []
                begin(r.clip_id, asked: note.isEmpty ? (label ?? "Handle it") : (label.map { "\($0): \(note)" } ?? note))
            } catch {
                self.error = error.localizedDescription
                phase = .compose
            }
        }
    }

    func sendFollowUp() {
        let note = followUp.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !note.isEmpty, let folder, let agent, phase == .finished else { return }
        if voice.listening { voice.cancel() }
        let cap = capture ?? ScreenCapture()
        let meta: [String: Any] = [
            "kind": "screen", "agent": agent.slug, "action": "ask", "note": note, "browser": Self.label,
            "follow_up": ["folder": folder], "screen": ["app": cap.appName, "window": cap.windowTitle],
        ]
        history.append(Exchange(asked: asked, answer: answer.isEmpty ? (runError ?? "") : answer, failed: answer.isEmpty))
        followUp = ""
        phase = .sending
        Task {
            do {
                let r = try await deck.send(meta: meta, parts: [])
                begin(r.clip_id, asked: note)
            } catch {
                runError = error.localizedDescription
                phase = .finished
            }
        }
    }

    private func begin(_ id: String, asked: String) {
        clipID = id
        self.asked = asked
        events = []
        live = ""
        answer = ""
        runError = nil
        files = []
        nextEvent = 0
        startedAt = Date()
        phase = .running
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            var misses = 0
            while !Task.isCancelled {
                guard let self, self.clipID == id else { return }
                do {
                    let r = try await self.deck.run(id, since: self.nextEvent)
                    misses = 0
                    self.apply(r)
                    if r.done {
                        self.phase = .finished
                        return
                    }
                } catch {
                    misses += 1
                    if misses > 10 {
                        self.runError = error.localizedDescription
                        self.phase = .finished
                        return
                    }
                }
                try? await Task.sleep(nanoseconds: 700_000_000)
            }
        }
    }

    private func apply(_ r: RunState) {
        if !r.events.isEmpty { events.append(contentsOf: r.events) }
        nextEvent = r.next
        live = r.live ?? ""
        if let res = r.result { answer = res }
        if let e = r.error { runError = e }
        // What the agent made: what this app sent is no news.
        if let f = r.files { files = f.filter { !Self.sentFiles.contains($0.path) } }
    }

    func stop() {
        guard let id = clipID else { return }
        Task { await deck.stop(id) }
    }

    private func resetRun() {
        pollTask?.cancel()
        pollTask = nil
        clipID = nil
        folder = nil
        asked = ""
        events = []
        live = ""
        answer = ""
        runError = nil
        files = []
        history = []
        followUp = ""
        note = nil
        if voice.listening { voice.cancel() }
    }

    func close() {
        if voice.listening { voice.cancel() }
        hidePanel?()
    }

    // ── With the answer ──

    func copyAnswer() {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(answer, forType: .string)
        flash("Copied")
    }

    /// Back into the app the person was in, with the answer pasted where they were.
    func pasteBack() {
        guard !answer.isEmpty else { return }
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(answer, forType: .string)
        hidePanel?()
        sourceApp?.activate()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
            let src = CGEventSource(stateID: .hidSystemState)
            let down = CGEvent(keyboardEventSource: src, virtualKey: 9, keyDown: true)
            let up = CGEvent(keyboardEventSource: src, virtualKey: 9, keyDown: false)
            down?.flags = .maskCommand
            up?.flags = .maskCommand
            down?.post(tap: .cghidEventTap)
            up?.post(tap: .cghidEventTap)
        }
    }

    func openDeck() {
        if let url = URL(string: prefs.deckBase.isEmpty ? "http://localhost:25080" : prefs.deckBase) {
            NSWorkspace.shared.open(url)
        }
    }

    func newTopic() {
        guard let agent else { return }
        Task {
            do {
                try await deck.newTopic(agent.slug)
                flash("New topic with \(agent.name)")
            } catch {
                flash(error.localizedDescription)
            }
        }
    }

    /// A file the agent left in the capture's folder: into Downloads/Captain, then opened.
    func open(_ file: RunFile) {
        guard let id = clipID else { return }
        Task {
            do {
                let data = try await deck.file(id, path: file.path)
                let downloads = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
                    ?? FileManager.default.temporaryDirectory
                let dir = downloads.appendingPathComponent("Captain").appendingPathComponent(folder ?? "capture")
                let target = dir.appendingPathComponent(file.path)
                try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                try data.write(to: target)
                NSWorkspace.shared.open(target)
            } catch {
                flash(error.localizedDescription)
            }
        }
    }

    func flash(_ text: String) {
        note = text
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
            if self?.note == text { self?.note = nil }
        }
    }
}

// ── screen.md: what the agent reads first ────────────────────────────

enum ScreenDoc {
    static func markdown(_ c: ScreenCapture, screenText: Bool, selection: Bool, pointer: Bool) -> String {
        var out: [String] = []
        let title = c.windowTitle.isEmpty ? c.appName : c.windowTitle
        out.append("# \(title.isEmpty ? "Screen" : title)")
        out.append("")
        if !c.appName.isEmpty { out.append("- App: \(c.appName)\(c.bundleID.isEmpty ? "" : " (\(c.bundleID))")") }
        if !c.windowTitle.isEmpty && c.windowTitle != c.appName { out.append("- Window: \(c.windowTitle)") }
        if !c.url.isEmpty { out.append("- Address: \(c.url)") }
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss ZZZZZ"
        out.append("- Captured: \(f.string(from: c.capturedAt))")
        if screenText && c.textFrom == "ocr" { out.append("- The text below was read from the screen image (OCR) — the app doesn't share it.") }
        if selection && !c.selection.isEmpty {
            out += ["", "## Selected (what I mean)", "", quote(c.selection)]
        }
        if pointer, let p = c.pointer, !(p.text.isEmpty && p.near.isEmpty) {
            out += ["", "## Under the pointer", ""]
            var line = p.role.isEmpty ? "" : "\(p.role.capitalizedFirst): "
            line += p.text.isEmpty ? "(no label)" : "“\(p.text)”"
            if !p.app.isEmpty && p.app != c.appName { line += " — in \(p.app)" }
            out.append(line)
            if !p.near.isEmpty { out += ["", "Around it:", "", quote(p.near)] }
        }
        if screenText {
            if !c.visibleText.isEmpty { out += ["", "## On screen", "", c.visibleText] }
            if !c.moreText.isEmpty { out += ["", "## The rest of the window (not on screen)", "", c.moreText] }
            if !c.hasText { out += ["", "(No text could be read from this window.)"] }
        }
        return out.joined(separator: "\n") + "\n"
    }

    static func selection(_ c: ScreenCapture) -> String {
        var head = "# Selected in \(c.appName.isEmpty ? "the app" : c.appName)"
        if !c.windowTitle.isEmpty && c.windowTitle != c.appName { head += " — \(c.windowTitle)" }
        return head + "\n\n" + c.selection + "\n"
    }

    private static func quote(_ s: String) -> String {
        s.components(separatedBy: "\n").map { "> " + $0 }.joined(separator: "\n")
    }
}

extension String {
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}
