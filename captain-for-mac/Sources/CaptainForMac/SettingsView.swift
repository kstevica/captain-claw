import AVFoundation
import Combine
import ServiceManagement
import Speech
import SwiftUI

enum Permissions {
    enum Pane: String {
        case accessibility = "Privacy_Accessibility"
        case screen = "Privacy_ScreenCapture"
        case microphone = "Privacy_Microphone"
        case speech = "Privacy_SpeechRecognition"
    }

    static func open(_ pane: Pane) {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(pane.rawValue)") {
            NSWorkspace.shared.open(url)
        }
    }
}

struct SettingsView: View {
    @EnvironmentObject var captain: Captain
    @EnvironmentObject var prefs: Prefs

    var body: some View {
        TabView {
            ConnectionTab().tabItem { Label("Flight Deck", systemImage: "antenna.radiowaves.left.and.right") }
            TriggerTab().tabItem { Label("Trigger", systemImage: "keyboard") }
            CaptureTab().tabItem { Label("What it reads", systemImage: "text.viewfinder") }
            VoiceTab().tabItem { Label("Voice", systemImage: "mic") }
            PermissionsTab().tabItem { Label("Permissions", systemImage: "lock.shield") }
        }
        .padding(18)
        .frame(width: 560, height: 470)
    }
}

struct ConnectionTab: View {
    @EnvironmentObject var captain: Captain
    @EnvironmentObject var prefs: Prefs

    var body: some View {
        Form {
            Section {
                TextField("Flight Deck address", text: $prefs.deckURL, prompt: Text("http://localhost:25080"))
                    .textFieldStyle(.roundedBorder)
                if captain.connected {
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.seal.fill").foregroundStyle(.green)
                        VStack(alignment: .leading, spacing: 2) {
                            if let me = captain.me {
                                Text("Connected as \(me.user.name.isEmpty ? me.user.email : me.user.name) to \(me.deck.name)")
                                Text("\(me.agents.count) agent\(me.agents.count == 1 ? "" : "s") · captures go to \(me.screen?.folder ?? "screen-clips")")
                                    .font(.caption).foregroundStyle(.secondary)
                            } else if let err = captain.meError {
                                Text(err).foregroundStyle(.orange)
                            } else {
                                Text("Connected")
                            }
                        }
                        Spacer()
                        Button("Disconnect") { captain.disconnect() }
                    }
                } else if let p = captain.pairing {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Approve this code in Flight Deck (a browser window opened):")
                        Text(p.code)
                            .font(.system(size: 26, weight: .semibold, design: .monospaced))
                            .tracking(4)
                            .foregroundStyle(accent)
                            .textSelection(.enabled)
                        HStack {
                            ProgressView().controlSize(.small)
                            Text("Waiting for the approval…").foregroundStyle(.secondary)
                            Spacer()
                            Button("Open it again") {
                                if let url = URL(string: prefs.deckBase + p.approve_path) { NSWorkspace.shared.open(url) }
                            }
                            Button("Cancel") { captain.cancelPairing() }
                        }
                    }
                } else {
                    HStack {
                        Text("Not connected").foregroundStyle(.secondary)
                        Spacer()
                        Button("Connect") { captain.connect() }
                            .buttonStyle(.borderedProminent).tint(accent)
                            .disabled(prefs.deckBase.isEmpty)
                    }
                }
                if !captain.pairState.isEmpty && captain.pairing == nil {
                    Text(captain.pairState).font(.caption).foregroundStyle(.orange)
                }
            } header: {
                Text("Your Flight Deck")
            } footer: {
                Text("Captain asks the deck for a connection; you approve it there, signed in. The app gets a token of its own — remove it any time in Flight Deck → Connections → Browser.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if captain.connected, let agents = captain.me?.agents, !agents.isEmpty {
                Section("Send to") {
                    Picker("Agent", selection: $prefs.agentSlug) {
                        ForEach(agents) { a in
                            Text(a.name + (a.running ? "" : " · \(a.status ?? "")")).tag(a.slug)
                        }
                    }
                }
            }
            Section("Look") {
                Picker("Text size", selection: $prefs.textSize) {
                    Text("Small").tag(12.0)
                    Text("Medium").tag(13.0)
                    Text("Large").tag(15.0)
                    Text("Larger").tag(17.0)
                }
                LaunchAtLogin()
            }
        }
        .formStyle(.grouped)
        .task { if captain.connected && captain.me == nil { await captain.refreshMe() } }
    }
}

struct LaunchAtLogin: View {
    @State private var on = SMAppService.mainApp.status == .enabled
    @State private var problem: String?

    var body: some View {
        VStack(alignment: .leading) {
            Toggle("Open Captain when I log in", isOn: Binding(get: { on }, set: { want in
                do {
                    if want { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
                    on = want
                    problem = nil
                } catch {
                    problem = error.localizedDescription
                }
            }))
            if let problem { Text(problem).font(.caption).foregroundStyle(.orange) }
        }
    }
}

struct TriggerTab: View {
    @EnvironmentObject var prefs: Prefs

    var body: some View {
        Form {
            Section {
                Picker("Hold this key", selection: $prefs.holdKey) {
                    ForEach(HoldKey.allCases) { k in Text(k.label).tag(k) }
                }
                if prefs.holdKey != .off {
                    HStack {
                        Text("For")
                        Slider(value: $prefs.holdSeconds, in: 0.5...5, step: 0.5)
                        Text(String(format: "%.1f s", prefs.holdSeconds)).monospacedDigit().frame(width: 44, alignment: .trailing)
                    }
                }
            } header: {
                Text("Hold a key")
            } footer: {
                Text(prefs.holdKey == .fn
                     ? "Hold fn alone; a ring by the pointer fills, then Captain opens. Any other key cancels it, so fn shortcuts keep working. If the 🌐 key also opens emoji or dictation, set System Settings → Keyboard → “Press 🌐 key to” → Do Nothing."
                     : "Hold the key alone; a ring by the pointer fills, then Captain opens. Any other key cancels it.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section {
                Toggle("Show a floating Captain button", isOn: $prefs.floatingButton)
                Text("Click it — or the Captain icon in the menu bar — to ask about what's on screen. Drag it anywhere.")
                    .font(.caption).foregroundStyle(.secondary)
            } header: {
                Text("Or click a button")
            }
            Section {
                Text("While Captain is open, keep holding the key and talk: letting go ends it (set Voice → Listen when Captain opens).")
                    .font(.caption).foregroundStyle(.secondary)
            } header: {
                Text("Hold to talk")
            }
        }
        .formStyle(.grouped)
    }
}

struct CaptureTab: View {
    @EnvironmentObject var prefs: Prefs

    var body: some View {
        Form {
            Section {
                Toggle("The window's text", isOn: $prefs.includeScreenText)
                Toggle("What's under the pointer", isOn: $prefs.includePointer)
                Toggle("Read the screen image when an app doesn't share its text (OCR)", isOn: $prefs.ocrFallback)
            } header: {
                Text("Read")
            } footer: {
                Text("The selection always goes first — it's what you mean; the rest is context. Each piece shows as a chip you can switch off before sending.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section {
                Toggle("Copy the selection when the app doesn't share it", isOn: $prefs.copyFallback)
                Text("Presses ⌘C for you and puts your clipboard back as it was.")
                    .font(.caption).foregroundStyle(.secondary)
            } header: {
                Text("Selection")
            }
            Section {
                Picker("Screenshot", selection: $prefs.screenshotMode) {
                    ForEach(ScreenshotMode.allCases) { m in Text(m.label).tag(m) }
                }
                Text("The agent sees the picture too. Needs Screen Recording (Permissions).")
                    .font(.caption).foregroundStyle(.secondary)
            } header: {
                Text("Screenshot")
            }
        }
        .formStyle(.grouped)
    }
}

struct VoiceTab: View {
    @EnvironmentObject var prefs: Prefs
    private let locales = VoiceInput.locales()

    var body: some View {
        Form {
            Section {
                Picker("Microphone", selection: $prefs.voiceMode) {
                    ForEach(VoiceMode.allCases) { m in Text(m.label).tag(m) }
                }
                Toggle("Send when I stop talking", isOn: $prefs.voiceAutoSend)
                    .disabled(prefs.voiceMode == .off)
            } header: {
                Text("Say the task")
            } footer: {
                Text("Your words go into the task box as you speak; a pause ends it. With “Send when I stop talking”, a pause (or letting go of the held key) sends it.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Language") {
                Picker("Language", selection: $prefs.voiceLocale) {
                    Text("This Mac's (\(Locale.current.identifier))").tag("")
                    ForEach(locales, id: \.identifier) { l in
                        Text(Locale.current.localizedString(forIdentifier: l.identifier) ?? l.identifier).tag(l.identifier)
                    }
                }
                Toggle("Only on this Mac (no audio leaves it)", isOn: $prefs.onDeviceSpeech)
                Text("Apple's speech recognition. Not every language works offline; with this off, Apple may process the audio.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .disabled(false)
    }
}

struct PermissionsTab: View {
    @State private var tick = 0
    private let timer = Timer.publish(every: 1.5, on: .main, in: .common).autoconnect()

    var body: some View {
        Form {
            Section {
                row("Accessibility", "Read the screen's text, the selection and the held key.",
                    ScreenReader.trusted, ask: { ScreenReader.askForTrust() }, pane: .accessibility)
                row("Screen Recording", "Screenshots, and reading text off the screen (OCR).",
                    ScreenImage.allowed, ask: { ScreenImage.askForPermission() }, pane: .screen)
                row("Microphone", "Saying the task.", VoiceInput.micAllowed,
                    ask: { AVCaptureDevice.requestAccess(for: .audio) { _ in } }, pane: .microphone)
                row("Speech Recognition", "Turning what you say into text.", VoiceInput.speechAllowed,
                    ask: { SFSpeechRecognizer.requestAuthorization { _ in } }, pane: .speech)
            } header: {
                Text("What macOS lets Captain do")
            } footer: {
                Text("Screen Recording takes effect after Captain restarts. A new build of the app asks again.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .id(tick)
            Section {
                Button("Restart Captain") {
                    let path = Bundle.main.bundlePath
                    let task = Process()
                    task.launchPath = "/bin/sh"
                    task.arguments = ["-c", "sleep 0.6; open \"\(path)\""]
                    try? task.run()
                    NSApp.terminate(nil)
                }
            }
        }
        .formStyle(.grouped)
        .onReceive(timer) { _ in tick += 1 }
    }

    private func row(_ name: String, _ why: String, _ ok: Bool, ask: @escaping () -> Void, pane: Permissions.Pane) -> some View {
        HStack(alignment: .top) {
            Image(systemName: ok ? "checkmark.circle.fill" : "circle.dashed")
                .foregroundStyle(ok ? .green : .orange)
            VStack(alignment: .leading, spacing: 2) {
                Text(name)
                Text(why).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if !ok {
                Button("Allow…") {
                    ask()
                    Permissions.open(pane)
                }
            }
        }
    }
}
