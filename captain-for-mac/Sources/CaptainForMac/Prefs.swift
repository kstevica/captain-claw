import AppKit
import Combine
import Foundation

/// The key that, held alone for a while, asks Captain.
enum HoldKey: String, CaseIterable, Identifiable {
    case fn, rightCommand, rightOption, rightShift, rightControl, off

    var id: String { rawValue }

    var label: String {
        switch self {
        case .fn: return "fn / 🌐"
        case .rightCommand: return "Right ⌘"
        case .rightOption: return "Right ⌥"
        case .rightShift: return "Right ⇧"
        case .rightControl: return "Right ⌃"
        case .off: return "Off"
        }
    }

    /// The key's own code in a flagsChanged event.
    var keyCode: UInt16? {
        switch self {
        case .fn: return 63
        case .rightCommand: return 54
        case .rightOption: return 61
        case .rightShift: return 60
        case .rightControl: return 62
        case .off: return nil
        }
    }

    /// The modifier it sets (device independent).
    var flag: NSEvent.ModifierFlags {
        switch self {
        case .fn: return .function
        case .rightCommand: return .command
        case .rightOption: return .option
        case .rightShift: return .shift
        case .rightControl: return .control
        case .off: return []
        }
    }

    /// The device-dependent bit that tells the right-hand key from the left.
    var deviceBit: UInt? {
        switch self {
        case .rightCommand: return 0x10
        case .rightOption: return 0x40
        case .rightShift: return 0x04
        case .rightControl: return 0x2000
        default: return nil
        }
    }

    var short: String {
        switch self {
        case .fn: return "fn"
        case .rightCommand: return "right ⌘"
        case .rightOption: return "right ⌥"
        case .rightShift: return "right ⇧"
        case .rightControl: return "right ⌃"
        case .off: return ""
        }
    }
}

enum ScreenshotMode: String, CaseIterable, Identifiable {
    case off, window, screen
    var id: String { rawValue }
    var label: String {
        switch self {
        case .off: return "Off"
        case .window: return "The front window"
        case .screen: return "The whole screen"
        }
    }
}

enum VoiceMode: String, CaseIterable, Identifiable {
    case off, button, auto
    var id: String { rawValue }
    var label: String {
        switch self {
        case .off: return "Off"
        case .button: return "Mic button only"
        case .auto: return "Listen when Captain opens"
        }
    }
}

/// Everything the person sets, kept in UserDefaults (the token is in the Keychain).
final class Prefs: ObservableObject {
    static let shared = Prefs()
    private let d = UserDefaults.standard

    @Published var deckURL: String { didSet { d.set(deckURL, forKey: "deckURL") } }
    @Published var agentSlug: String { didSet { d.set(agentSlug, forKey: "agentSlug") } }
    @Published var holdKey: HoldKey { didSet { d.set(holdKey.rawValue, forKey: "holdKey") } }
    @Published var holdSeconds: Double { didSet { d.set(holdSeconds, forKey: "holdSeconds") } }
    @Published var floatingButton: Bool { didSet { d.set(floatingButton, forKey: "floatingButton") } }
    @Published var includeScreenText: Bool { didSet { d.set(includeScreenText, forKey: "includeScreenText") } }
    @Published var includePointer: Bool { didSet { d.set(includePointer, forKey: "includePointer") } }
    @Published var copyFallback: Bool { didSet { d.set(copyFallback, forKey: "copyFallback") } }
    @Published var screenshotMode: ScreenshotMode { didSet { d.set(screenshotMode.rawValue, forKey: "screenshotMode") } }
    @Published var ocrFallback: Bool { didSet { d.set(ocrFallback, forKey: "ocrFallback") } }
    @Published var voiceMode: VoiceMode { didSet { d.set(voiceMode.rawValue, forKey: "voiceMode") } }
    @Published var voiceAutoSend: Bool { didSet { d.set(voiceAutoSend, forKey: "voiceAutoSend") } }
    @Published var voiceLocale: String { didSet { d.set(voiceLocale, forKey: "voiceLocale") } }
    @Published var onDeviceSpeech: Bool { didSet { d.set(onDeviceSpeech, forKey: "onDeviceSpeech") } }
    @Published var textSize: Double { didSet { d.set(textSize, forKey: "textSize") } }

    private init() {
        d.register(defaults: [
            "deckURL": "http://localhost:25080",
            "agentSlug": "",
            "holdKey": HoldKey.fn.rawValue,
            "holdSeconds": 3.0,
            "floatingButton": false,
            "includeScreenText": true,
            "includePointer": true,
            "copyFallback": true,
            "screenshotMode": ScreenshotMode.off.rawValue,
            "ocrFallback": true,
            "voiceMode": VoiceMode.button.rawValue,
            "voiceAutoSend": false,
            "voiceLocale": "",
            "onDeviceSpeech": false,
            "textSize": 13.0,
        ])
        deckURL = d.string(forKey: "deckURL") ?? "http://localhost:25080"
        agentSlug = d.string(forKey: "agentSlug") ?? ""
        holdKey = HoldKey(rawValue: d.string(forKey: "holdKey") ?? "") ?? .fn
        holdSeconds = min(5, max(0.5, d.double(forKey: "holdSeconds")))
        floatingButton = d.bool(forKey: "floatingButton")
        includeScreenText = d.bool(forKey: "includeScreenText")
        includePointer = d.bool(forKey: "includePointer")
        copyFallback = d.bool(forKey: "copyFallback")
        screenshotMode = ScreenshotMode(rawValue: d.string(forKey: "screenshotMode") ?? "") ?? .off
        ocrFallback = d.bool(forKey: "ocrFallback")
        voiceMode = VoiceMode(rawValue: d.string(forKey: "voiceMode") ?? "") ?? .button
        voiceAutoSend = d.bool(forKey: "voiceAutoSend")
        voiceLocale = d.string(forKey: "voiceLocale") ?? ""
        onDeviceSpeech = d.bool(forKey: "onDeviceSpeech")
        textSize = min(18, max(11, d.double(forKey: "textSize")))
    }

    /// The deck's address without a trailing slash ("" when it isn't one).
    var deckBase: String {
        var s = deckURL.trimmingCharacters(in: .whitespacesAndNewlines)
        if !s.isEmpty && !s.contains("://") { s = "http://" + s }
        while s.hasSuffix("/") { s.removeLast() }
        guard let u = URL(string: s), u.host != nil else { return "" }
        return s
    }
}

/// The deck's token, one per deck address.
enum Keychain {
    static let service = "Captain for Mac"

    static func token(for deck: String) -> String? {
        let q: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: deck,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var out: AnyObject?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let data = out as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func setToken(_ token: String, for deck: String) {
        removeToken(for: deck)
        let q: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: deck,
            kSecAttrLabel as String: "Captain for Mac — \(deck)",
            kSecValueData as String: Data(token.utf8),
        ]
        SecItemAdd(q as CFDictionary, nil)
    }

    static func removeToken(for deck: String) {
        let q: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: deck,
        ]
        SecItemDelete(q as CFDictionary)
    }
}
