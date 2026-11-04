import AppKit
import Combine
import SwiftUI

// The panel: what Captain read (chips the person can switch off), the task
// (typed or spoken), one-click actions, then the agent's work and answer.

let accent = Color(red: 0.49, green: 0.40, blue: 0.98)

struct PanelView: View {
    @EnvironmentObject var captain: Captain
    @EnvironmentObject var prefs: Prefs

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Header()
            Divider().opacity(0.4)
            Group {
                switch captain.phase {
                case .capturing, .compose, .sending:
                    ComposeView()
                case .running, .finished:
                    RunView()
                }
            }
            .padding(14)
        }
        .frame(width: 460)
        .background(VisualEffect().ignoresSafeArea())
        .font(.system(size: prefs.textSize))
    }
}

struct VisualEffect: NSViewRepresentable {
    func makeNSView(context: Context) -> NSVisualEffectView {
        let v = NSVisualEffectView()
        v.material = .popover
        v.blendingMode = .behindWindow
        v.state = .active
        return v
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {}
}

// ── Header: who it goes to ───────────────────────────────────────────

struct Header: View {
    @EnvironmentObject var captain: Captain
    @EnvironmentObject var prefs: Prefs

    var body: some View {
        HStack(spacing: 8) {
            Image(nsImage: NSApp.applicationIconImage)
                .resizable()
                .frame(width: 22, height: 22)
            Text("Captain").font(.system(size: 14, weight: .semibold))
            Spacer()
            if captain.connected {
                AgentMenu()
            } else {
                Button("Connect…") { captain.openSettings?() }
                    .buttonStyle(.borderless)
                    .foregroundStyle(accent)
            }
            Menu {
                if let a = captain.agent {
                    Button("New topic with \(a.name)") { captain.newTopic() }
                }
                Button("Open Flight Deck") { captain.openDeck() }
                Divider()
                Button("Settings…") { captain.openSettings?() }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            Button {
                captain.close()
            } label: {
                Image(systemName: "xmark").font(.system(size: 11, weight: .semibold))
            }
            .buttonStyle(.borderless)
            .help("Close (Esc)")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }
}

struct AgentMenu: View {
    @EnvironmentObject var captain: Captain
    @EnvironmentObject var prefs: Prefs

    var body: some View {
        Menu {
            if let agents = captain.me?.agents, !agents.isEmpty {
                ForEach(agents) { a in
                    Button {
                        prefs.agentSlug = a.slug
                        captain.objectWillChange.send()
                    } label: {
                        Text(a.name + (a.running ? "" : a.asleep ? " · asleep" : " · \(a.status ?? "off")"))
                    }
                }
            } else if captain.loadingMe {
                Text("Loading your agents…")
            } else {
                Text(captain.meError ?? "No agents")
                Button("Try again") { Task { await captain.refreshMe() } }
            }
        } label: {
            HStack(spacing: 5) {
                Circle()
                    .fill(captain.agent?.running == true ? Color.green : Color.secondary.opacity(0.6))
                    .frame(width: 7, height: 7)
                Text(captain.agent?.name ?? (captain.loadingMe ? "Loading…" : "Pick an agent"))
                    .lineLimit(1)
            }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help("Which agent gets it")
    }
}

// ── Compose ──────────────────────────────────────────────────────────

struct ComposeView: View {
    @EnvironmentObject var captain: Captain
    @EnvironmentObject var prefs: Prefs

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ContextCard()
            TaskBox(text: $captain.task,
                    placeholder: captain.agent.map { "What should \($0.name) do? Empty = Handle it" } ?? "What should Captain do?",
                    onSubmit: { captain.send() })
            if let problem = captain.voice.problem {
                Label(problem, systemImage: "mic.slash").font(.caption).foregroundStyle(.orange)
            }
            ActionChips()
            if let err = captain.error {
                Label(err, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Text(keyHint).font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button {
                    captain.send()
                } label: {
                    HStack(spacing: 6) {
                        if captain.phase == .sending { ProgressView().controlSize(.small) }
                        Text(captain.phase == .sending ? "Sending…" : "Send")
                        Image(systemName: "return")
                    }
                    .padding(.horizontal, 6)
                }
                .buttonStyle(.borderedProminent)
                .tint(accent)
                .disabled(captain.phase != .compose)
                .keyboardShortcut(.defaultAction)
            }
        }
    }

    var keyHint: String {
        prefs.holdKey == .off ? "⏎ send · ⇧⏎ new line · esc close"
            : "⏎ send · ⇧⏎ new line · hold \(prefs.holdKey.short) to ask again"
    }
}

struct ContextCard: View {
    @EnvironmentObject var captain: Captain

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if captain.phase == .capturing || captain.capture == nil {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Reading the screen…").foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
            } else if let cap = captain.capture {
                HStack(alignment: .top, spacing: 10) {
                    AppIcon(bundleID: cap.bundleID).frame(width: 30, height: 30)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(cap.appName.isEmpty ? "No app" : cap.appName).font(.system(size: 13, weight: .semibold))
                        if !cap.windowTitle.isEmpty && cap.windowTitle != cap.appName {
                            Text(cap.windowTitle).lineLimit(1).truncationMode(.middle).foregroundStyle(.secondary)
                        }
                        if !cap.url.isEmpty {
                            Text(cap.url).font(.caption).lineLimit(1).truncationMode(.middle).foregroundStyle(.tertiary)
                        }
                    }
                    Spacer(minLength: 0)
                }
                FlowLayout(spacing: 6) {
                    if !cap.selection.isEmpty {
                        Chip(icon: "text.cursor", title: "Selection",
                             detail: count(cap.selection) + (cap.selectionFrom == "copy" ? " · copied" : ""),
                             on: $captain.useSelection, preview: cap.selection, strong: true)
                    }
                    if cap.hasText {
                        Chip(icon: cap.textFrom == "ocr" ? "text.viewfinder" : "text.alignleft",
                             title: cap.textFrom == "ocr" ? "Screen text (OCR)" : "Screen text",
                             detail: count(cap.visibleText + cap.moreText),
                             on: $captain.useScreen, preview: cap.visibleText.isEmpty ? cap.moreText : cap.visibleText)
                    }
                    if let p = cap.pointer, !(p.text.isEmpty && p.near.isEmpty) {
                        Chip(icon: "cursorarrow.rays", title: "Pointer",
                             detail: p.text.isEmpty ? (p.role.isEmpty ? "here" : p.role) : "“\(String(p.text.prefix(22)))\(p.text.count > 22 ? "…" : "")”",
                             on: $captain.usePointer, preview: [p.text, p.near].filter { !$0.isEmpty }.joined(separator: "\n\n"))
                    }
                    if cap.screenshot != nil {
                        Chip(icon: "camera.viewfinder", title: "Screenshot",
                             detail: "\(Int(cap.screenshotPixels.width))×\(Int(cap.screenshotPixels.height))",
                             on: $captain.useShot, preview: nil)
                    }
                }
                ForEach(cap.notes, id: \.self) { n in
                    HStack(alignment: .top, spacing: 6) {
                        Image(systemName: "exclamationmark.circle").foregroundStyle(.orange)
                        Text(n).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                        if !cap.trusted {
                            Button("Allow…") { ScreenReader.askForTrust(); Permissions.open(.accessibility) }
                                .controlSize(.small)
                        }
                    }
                }
            }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.primary.opacity(0.05)))
    }

    func count(_ s: String) -> String {
        let n = s.count
        if n >= 1000 { return String(format: "%.1fk chars", Double(n) / 1000) }
        return "\(n) chars"
    }
}

struct AppIcon: View {
    let bundleID: String
    var body: some View {
        if let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID) {
            Image(nsImage: NSWorkspace.shared.icon(forFile: url.path)).resizable()
        } else {
            Image(systemName: "macwindow").resizable().scaledToFit().foregroundStyle(.secondary)
        }
    }
}

struct Chip: View {
    let icon: String
    let title: String
    let detail: String
    @Binding var on: Bool
    let preview: String?
    var strong = false
    @State private var showing = false

    var body: some View {
        HStack(spacing: 5) {
            Button {
                on.toggle()
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: on ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(on ? (strong ? accent : Color.primary.opacity(0.7)) : Color.secondary)
                    Image(systemName: icon).foregroundStyle(.secondary)
                    Text(title).fontWeight(strong ? .semibold : .regular)
                    Text(detail).foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.plain)
            if let preview, !preview.isEmpty {
                Button {
                    showing.toggle()
                } label: {
                    Image(systemName: "eye").foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("See what Captain read")
                .popover(isPresented: $showing, arrowEdge: .bottom) {
                    ScrollView {
                        Text(String(preview.prefix(6000)))
                            .font(.system(size: 12))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(12)
                    }
                    .frame(width: 380, height: 260)
                }
            }
        }
        .font(.system(size: 12))
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Capsule().fill(on ? (strong ? accent.opacity(0.16) : Color.primary.opacity(0.08)) : Color.clear))
        .overlay(Capsule().stroke(Color.primary.opacity(on ? 0 : 0.15)))
        .opacity(on ? 1 : 0.7)
        .help(on ? "Sent with your task — click to leave it out" : "Left out — click to send it")
    }
}

struct ActionChips: View {
    @EnvironmentObject var captain: Captain

    var body: some View {
        FlowLayout(spacing: 6) {
            ForEach(captain.actions) { a in
                Button {
                    captain.send(action: a.id)
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: Self.symbol(a.icon))
                        Text(a.label)
                    }
                    .font(.system(size: 12))
                    .padding(.horizontal, 9)
                    .padding(.vertical, 5)
                    .background(Capsule().fill(Color.primary.opacity(0.07)))
                }
                .buttonStyle(.plain)
                .disabled(captain.phase != .compose)
                .help(a.hint ?? a.label)
            }
        }
    }

    static func symbol(_ icon: String?) -> String {
        switch icon ?? "" {
        case "sparkles": return "sparkles"
        case "lightbulb": return "lightbulb"
        case "summary": return "text.append"
        case "reply": return "arrowshape.turn.up.left"
        case "pen": return "pencil.line"
        case "languages": return "character.bubble"
        case "wrench": return "wrench.and.screwdriver"
        case "checklist": return "checklist"
        case "brain": return "brain"
        case "bookmark": return "bookmark"
        case "ask": return "questionmark.bubble"
        default: return "circle.grid.2x2"
        }
    }
}

// ── The task box, typed or spoken ────────────────────────────────────

struct TaskBox: View {
    @EnvironmentObject var captain: Captain
    @EnvironmentObject var prefs: Prefs
    @ObservedObject var voice = Captain.shared.voice
    @Binding var text: String
    let placeholder: String
    let onSubmit: () -> Void
    var minHeight: CGFloat = 58

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            ZStack(alignment: .topLeading) {
                TaskTextView(text: $text, onSubmit: onSubmit, fontSize: prefs.textSize + 1)
                    .frame(minHeight: minHeight, maxHeight: 130)
                if text.isEmpty {
                    Text(voice.listening ? (voice.text.isEmpty ? "Listening… say what you need" : voice.text) : placeholder)
                        .foregroundStyle(voice.listening ? Color.primary.opacity(0.75) : Color.secondary)
                        .font(.system(size: prefs.textSize + 1))
                        .padding(.top, 1)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                } else if voice.listening && !voice.text.isEmpty {
                    EmptyView()
                }
            }
            if prefs.voiceMode != .off {
                MicButton(listening: voice.listening, level: voice.level) { captain.toggleVoice() }
            }
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .textBackgroundColor).opacity(0.7)))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(voice.listening ? Color.red.opacity(0.6) : accent.opacity(0.35)))
        .overlay(alignment: .bottomLeading) {
            if voice.listening && !text.isEmpty && !voice.text.isEmpty {
                Text("🎙 " + voice.text).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    .padding(.horizontal, 12).padding(.bottom, 4)
            }
        }
    }
}

struct MicButton: View {
    let listening: Bool
    let level: Float
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle().fill(listening ? Color.red.opacity(0.18) : Color.primary.opacity(0.07))
                if listening {
                    Circle().stroke(Color.red.opacity(0.7), lineWidth: 2)
                        .scaleEffect(1 + CGFloat(level) * 0.35)
                        .animation(.easeOut(duration: 0.12), value: level)
                }
                Image(systemName: listening ? "stop.fill" : "mic.fill")
                    .foregroundStyle(listening ? Color.red : Color.primary.opacity(0.8))
                    .font(.system(size: 13))
            }
            .frame(width: 32, height: 32)
        }
        .buttonStyle(.plain)
        .help(listening ? "Stop listening" : "Say it")
    }
}

/// A multi-line field where Enter sends and Shift+Enter makes a new line.
struct TaskTextView: NSViewRepresentable {
    @Binding var text: String
    let onSubmit: () -> Void
    var fontSize: CGFloat = 14

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView()
        scroll.drawsBackground = false
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.borderType = .noBorder
        let tv = SubmitTextView()
        tv.delegate = context.coordinator
        tv.onSubmit = { context.coordinator.parent.onSubmit() }
        tv.isRichText = false
        tv.allowsUndo = true
        tv.drawsBackground = false
        tv.font = .systemFont(ofSize: fontSize)
        tv.textContainerInset = NSSize(width: 0, height: 1)
        tv.isAutomaticQuoteSubstitutionEnabled = false
        tv.isVerticallyResizable = true
        tv.isHorizontallyResizable = false
        tv.autoresizingMask = [.width]
        tv.textContainer?.widthTracksTextView = true
        tv.string = text
        scroll.documentView = tv
        DispatchQueue.main.async { tv.window?.makeFirstResponder(tv) }
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let tv = scroll.documentView as? SubmitTextView else { return }
        if tv.string != text {
            tv.string = text
            tv.setSelectedRange(NSRange(location: (text as NSString).length, length: 0))
        }
        if tv.font?.pointSize != fontSize { tv.font = .systemFont(ofSize: fontSize) }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: TaskTextView
        init(_ p: TaskTextView) { parent = p }

        func textDidChange(_ n: Notification) {
            guard let tv = n.object as? NSTextView else { return }
            parent.text = tv.string
        }
    }
}

final class SubmitTextView: NSTextView {
    var onSubmit: (() -> Void)?

    override func keyDown(with event: NSEvent) {
        let enter = event.keyCode == 36 || event.keyCode == 76
        if enter && !event.modifierFlags.contains(.shift) && !hasMarkedText() {
            onSubmit?()
            return
        }
        super.keyDown(with: event)
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.window?.makeFirstResponder(self)
        }
    }
}

// ── The run: the agent at work, then its answer ──────────────────────

struct RunView: View {
    @EnvironmentObject var captain: Captain
    @EnvironmentObject var prefs: Prefs
    @State private var contentHeight: CGFloat = 80

    var running: Bool { captain.phase == .running }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let cap = captain.capture {
                HStack(spacing: 6) {
                    AppIcon(bundleID: cap.bundleID).frame(width: 16, height: 16)
                    Text(cap.windowTitle.isEmpty ? cap.appName : cap.windowTitle)
                        .lineLimit(1).truncationMode(.middle).foregroundStyle(.secondary).font(.caption)
                    Spacer()
                    Button("New capture") { captain.trigger() }
                        .buttonStyle(.borderless).font(.caption).disabled(running)
                }
            }
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        ForEach(captain.history) { ex in
                            Question(text: ex.asked)
                            if ex.failed {
                                Text(ex.answer).foregroundStyle(.red)
                            } else {
                                MarkdownView(text: ex.answer, size: prefs.textSize)
                            }
                            Divider().opacity(0.4)
                        }
                        Question(text: captain.asked)
                        if running {
                            TimelineView(.periodic(from: .now, by: 1)) { ctx in
                                Working(events: captain.events, startedAt: captain.startedAt, now: ctx.date)
                            }
                            if !captain.live.isEmpty {
                                MarkdownView(text: captain.live, size: prefs.textSize).opacity(0.8)
                            } else if !captain.answer.isEmpty {
                                MarkdownView(text: captain.answer, size: prefs.textSize)
                            }
                        } else {
                            if !captain.answer.isEmpty {
                                MarkdownView(text: captain.answer, size: prefs.textSize)
                            }
                            if let err = captain.runError {
                                Label(err, systemImage: "exclamationmark.triangle.fill")
                                    .foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
                            }
                        }
                        if !running && !captain.files.isEmpty {
                            FilesList()
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding(.trailing, 6)
                    .background(GeometryReader { g in Color.clear.preference(key: HeightKey.self, value: g.size.height) })
                }
                .frame(height: min(max(contentHeight, 60), 420))
                .onPreferenceChange(HeightKey.self) { contentHeight = $0 }
                .onChange(of: captain.live) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
                .onChange(of: captain.events.count) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
                .onChange(of: captain.phase) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
            }
            HStack(spacing: 8) {
                if running {
                    Button {
                        captain.stop()
                    } label: {
                        Label("Stop", systemImage: "stop.circle")
                    }
                } else {
                    Button {
                        captain.copyAnswer()
                    } label: {
                        Label("Copy", systemImage: "doc.on.doc")
                    }
                    .disabled(captain.answer.isEmpty)
                    Button {
                        captain.pasteBack()
                    } label: {
                        Label("Paste into \(captain.appForPaste?.localizedName ?? "the app")", systemImage: "arrow.down.doc")
                    }
                    .disabled(captain.answer.isEmpty || captain.appForPaste == nil)
                    .help("Back to the app you were in, with the answer pasted where you were")
                }
                Spacer()
                if let n = captain.note {
                    Text(n).font(.caption).foregroundStyle(.secondary).transition(.opacity)
                }
            }
            .controlSize(.small)
            if !running {
                HStack(alignment: .bottom, spacing: 8) {
                    TaskBox(text: $captain.followUp, placeholder: "Ask a follow-up…",
                            onSubmit: { captain.sendFollowUp() }, minHeight: 22)
                    Button {
                        captain.sendFollowUp()
                    } label: {
                        Image(systemName: "arrow.up.circle.fill").font(.system(size: 22))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(captain.followUp.trimmingCharacters(in: .whitespaces).isEmpty ? Color.secondary : accent)
                    .disabled(captain.followUp.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
        }
    }
}

struct HeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = max(value, nextValue()) }
}

struct Question: View {
    let text: String
    var body: some View {
        HStack {
            Spacer(minLength: 40)
            Text(text)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(RoundedRectangle(cornerRadius: 10).fill(accent.opacity(0.18)))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

struct Working: View {
    let events: [RunEvent]
    let startedAt: Date
    let now: Date

    var body: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            Text(events.last?.text ?? "Sent — waiting for the agent…")
                .lineLimit(1).truncationMode(.tail).foregroundStyle(.secondary)
            Spacer()
            Text(elapsed).font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
        }
    }

    var elapsed: String {
        let s = max(0, Int(now.timeIntervalSince(startedAt)))
        return s < 60 ? "\(s)s" : "\(s / 60)m \(s % 60)s"
    }
}

struct FilesList: View {
    @EnvironmentObject var captain: Captain

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("In the capture's folder").font(.caption).foregroundStyle(.secondary)
            ForEach(captain.files) { f in
                Button {
                    captain.open(f)
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: Self.icon(f.path)).foregroundStyle(.secondary)
                        Text(f.path).lineLimit(1).truncationMode(.middle)
                        Spacer()
                        Text(ByteCountFormatter.string(fromByteCount: Int64(f.size), countStyle: .file))
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                }
                .buttonStyle(.plain)
                .help("Download to Downloads/Captain and open")
            }
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.primary.opacity(0.04)))
    }

    static func icon(_ path: String) -> String {
        let ext = (path as NSString).pathExtension.lowercased()
        switch ext {
        case "png", "jpg", "jpeg", "gif", "webp": return "photo"
        case "csv", "xlsx": return "tablecells"
        case "pdf": return "doc.richtext"
        default: return "doc.text"
        }
    }
}

// ── A small flow layout for chips ────────────────────────────────────

struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 420
        var x: CGFloat = 0, y: CGFloat = 0, rowH: CGFloat = 0, maxX: CGFloat = 0
        for v in subviews {
            let s = v.sizeThatFits(.unspecified)
            if x > 0 && x + s.width > width {
                x = 0
                y += rowH + spacing
                rowH = 0
            }
            x += s.width + spacing
            maxX = max(maxX, x - spacing)
            rowH = max(rowH, s.height)
        }
        return CGSize(width: min(width, maxX), height: y + rowH)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, rowH: CGFloat = 0
        for v in subviews {
            let s = v.sizeThatFits(.unspecified)
            if x > bounds.minX && x + s.width > bounds.maxX {
                x = bounds.minX
                y += rowH + spacing
                rowH = 0
            }
            v.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(s))
            x += s.width + spacing
            rowH = max(rowH, s.height)
        }
    }
}
