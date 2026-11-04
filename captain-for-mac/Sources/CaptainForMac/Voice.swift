import AVFoundation
import Foundation
import Speech

// The task, spoken: Apple's speech recognition (on this Mac when the
// language allows and the person wants it). The words go into the task box
// as they come; a pause after speaking ends it.

final class VoiceInput: NSObject, ObservableObject {
    @Published var listening = false
    @Published var level: Float = 0
    @Published var text = ""
    @Published var problem: String?

    /// The words when listening ended; *bySilence* when a pause ended it.
    var onFinish: ((String, Bool) -> Void)?
    var silenceSeconds: Double = 1.8

    private let engine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var recognizer: SFSpeechRecognizer?
    private var started = Date()
    private var lastChange = Date()
    private var watch: Timer?
    private var finishing = false
    private var endedBySilence = false
    private var tapInstalled = false

    static func locales() -> [Locale] {
        SFSpeechRecognizer.supportedLocales().sorted {
            ($0.localizedString(forIdentifier: $0.identifier) ?? $0.identifier)
                < ($1.localizedString(forIdentifier: $1.identifier) ?? $1.identifier)
        }
    }

    static var speechAllowed: Bool { SFSpeechRecognizer.authorizationStatus() == .authorized }
    static var micAllowed: Bool { AVCaptureDevice.authorizationStatus(for: .audio) == .authorized }

    func toggle(locale: String, onDevice: Bool) {
        listening ? stop() : start(locale: locale, onDevice: onDevice)
    }

    func start(locale: String, onDevice: Bool) {
        guard !listening, !finishing else { return }
        problem = nil
        SFSpeechRecognizer.requestAuthorization { status in
            guard status == .authorized else {
                DispatchQueue.main.async {
                    self.problem = "Speech recognition isn't allowed — System Settings → Privacy & Security → Speech Recognition."
                }
                return
            }
            AVCaptureDevice.requestAccess(for: .audio) { ok in
                DispatchQueue.main.async {
                    guard ok else {
                        self.problem = "The microphone isn't allowed — System Settings → Privacy & Security → Microphone."
                        return
                    }
                    self.begin(locale: locale, onDevice: onDevice)
                }
            }
        }
    }

    private func begin(locale: String, onDevice: Bool) {
        let loc = locale.isEmpty ? Locale.current : Locale(identifier: locale)
        guard let rec = SFSpeechRecognizer(locale: loc) ?? SFSpeechRecognizer(), rec.isAvailable else {
            problem = "Speech recognition isn't available for \(loc.identifier) right now."
            return
        }
        recognizer = rec
        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        req.addsPunctuation = true
        if onDevice && rec.supportsOnDeviceRecognition { req.requiresOnDeviceRecognition = true }
        request = req
        text = ""
        endedBySilence = false
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0 else {
            problem = "No microphone found."
            return
        }
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            req.append(buffer)
            self?.meter(buffer)
        }
        tapInstalled = true
        engine.prepare()
        do { try engine.start() } catch {
            cleanup()
            problem = "The microphone didn't start: \(error.localizedDescription)"
            return
        }
        task = rec.recognitionTask(with: req) { [weak self] result, error in
            DispatchQueue.main.async {
                guard let self else { return }
                if let result {
                    let t = result.bestTranscription.formattedString
                    if t != self.text {
                        self.text = t
                        self.lastChange = Date()
                    }
                    if result.isFinal { self.complete() }
                }
                if error != nil && (self.listening || self.finishing) { self.complete() }
            }
        }
        listening = true
        started = Date()
        lastChange = Date()
        watch?.invalidate()
        let w = Timer(timeInterval: 0.25, repeats: true) { [weak self] _ in self?.check() }
        RunLoop.main.add(w, forMode: .common)
        watch = w
    }

    private func check() {
        guard listening else { return }
        let now = Date()
        if !text.isEmpty && now.timeIntervalSince(lastChange) > silenceSeconds {
            endedBySilence = true
            stop()
        } else if text.isEmpty && now.timeIntervalSince(started) > 12 {
            stop()
        }
    }

    private func meter(_ buffer: AVAudioPCMBuffer) {
        guard let ch = buffer.floatChannelData?[0] else { return }
        let n = Int(buffer.frameLength)
        guard n > 0 else { return }
        var sum: Float = 0
        for i in stride(from: 0, to: n, by: 4) { sum += ch[i] * ch[i] }
        let rms = sqrt(sum / Float(max(1, n / 4)))
        let db = 20 * log10(max(rms, 0.000_01))
        let lvl = max(0, min(1, (db + 55) / 45))
        DispatchQueue.main.async { self.level = self.level * 0.6 + lvl * 0.4 }
    }

    /// Stop listening; the last words still come (a moment later at most).
    func stop() {
        guard listening else { return }
        listening = false
        finishing = true
        watch?.invalidate()
        watch = nil
        if engine.isRunning { engine.stop() }
        if tapInstalled { engine.inputNode.removeTap(onBus: 0); tapInstalled = false }
        request?.endAudio()
        level = 0
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.9) { [weak self] in self?.complete() }
    }

    /// Stop without using the words.
    func cancel() {
        guard listening || finishing else { return }
        discard = true
        stop()
        task?.cancel()
        complete()
    }

    private var discard = false

    private func complete() {
        guard finishing || listening else { return }
        defer { discard = false }
        if listening {
            listening = false
            if engine.isRunning { engine.stop() }
            if tapInstalled { engine.inputNode.removeTap(onBus: 0); tapInstalled = false }
            request?.endAudio()
        }
        finishing = false
        watch?.invalidate()
        watch = nil
        task?.finish()
        task = nil
        request = nil
        level = 0
        if !discard { onFinish?(text.trimmingCharacters(in: .whitespacesAndNewlines), endedBySilence) }
    }

    private func cleanup() {
        if engine.isRunning { engine.stop() }
        if tapInstalled { engine.inputNode.removeTap(onBus: 0); tapInstalled = false }
        request = nil
        task = nil
        listening = false
        finishing = false
    }
}
