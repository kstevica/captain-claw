import Foundation

// The deck's /fd/clip API — the same one Captain for Chrome uses, with
// kind "screen" for what comes from the Mac.

struct DeckUser: Decodable, Hashable {
    let name: String
    let email: String
}

struct Agent: Decodable, Identifiable, Hashable {
    let name: String
    let slug: String
    let description: String?
    let status: String?
    var id: String { slug }
    var running: Bool { (status ?? "") == "running" }
    var asleep: Bool { (status ?? "") == "asleep" }
}

struct ScreenAction: Decodable, Identifiable, Hashable {
    let id: String
    let label: String
    let icon: String?
    let hint: String?
    let needs_text: Bool?
}

struct MeInfo: Decodable {
    struct Deck: Decodable { let name: String }
    struct Screen: Decodable {
        let folder: String
        let actions: [ScreenAction]
    }
    let user: DeckUser
    let deck: Deck
    let agents: [Agent]
    let screen: Screen?
}

struct PairStart: Decodable {
    let pair_id: String
    let code: String
    let approve_path: String
    let expires_in: Int
    let interval: Int
}

struct PairPoll: Decodable {
    let status: String
    let token: String?
    let user: DeckUser?
}

struct SendResult: Decodable {
    let clip_id: String
    let folder: String
    let vfs: String
    let agent: String?
}

struct RunEvent: Decodable, Hashable {
    let kind: String
    let text: String
    let tool: String?
}

struct RunFile: Decodable, Hashable, Identifiable {
    let path: String
    let size: Int
    var id: String { path }
}

struct RunState: Decodable {
    let status: String
    let events: [RunEvent]
    let next: Int
    let live: String?
    let done: Bool
    let result: String?
    let error: String?
    let files: [RunFile]?
    let vfs: String?
    let folder: String?
}

struct DeckError: LocalizedError {
    let message: String
    let status: Int
    var errorDescription: String? { message }
}

/// One part of a multipart body.
struct FormPart {
    let name: String
    let filename: String
    let type: String
    let data: Data
}

final class DeckClient {
    var base: String = ""
    var token: String?

    private let session: URLSession = {
        let c = URLSessionConfiguration.ephemeral
        c.timeoutIntervalForRequest = 60
        c.timeoutIntervalForResource = 300
        return URLSession(configuration: c)
    }()

    var connected: Bool { !base.isEmpty && (token ?? "").isEmpty == false }

    private func request(_ method: String, _ path: String, body: Data? = nil, type: String? = nil,
                         auth: Bool = true, timeout: TimeInterval = 60) async throws -> Data {
        guard let url = URL(string: base + path), !base.isEmpty else {
            throw DeckError(message: "Set the Flight Deck address in Settings first.", status: 0)
        }
        var req = URLRequest(url: url, timeoutInterval: timeout)
        req.httpMethod = method
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        if let type { req.setValue(type, forHTTPHeaderField: "Content-Type") }
        if auth, let token, !token.isEmpty { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        req.httpBody = body
        let data: Data
        let resp: URLResponse
        do {
            (data, resp) = try await session.data(for: req)
        } catch {
            throw DeckError(message: "Can't reach Flight Deck at \(base) — is it running? (\(error.localizedDescription))", status: 0)
        }
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else {
            throw DeckError(message: Self.detail(data) ?? "Flight Deck answered \(code).", status: code)
        }
        return data
    }

    static func detail(_ data: Data) -> String? {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            let text = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return text.isEmpty || text.count > 300 ? nil : text
        }
        if let s = obj["detail"] as? String { return s }
        if let list = obj["detail"] as? [[String: Any]], let first = list.first, let msg = first["msg"] as? String { return msg }
        if let s = obj["message"] as? String { return s }
        return nil
    }

    private func json<T: Decodable>(_ type: T.Type, _ data: Data) throws -> T {
        do { return try JSONDecoder().decode(T.self, from: data) } catch {
            throw DeckError(message: "Flight Deck answered something this app doesn't understand — is it up to date?", status: 0)
        }
    }

    private func body(_ obj: [String: Any]) -> Data { (try? JSONSerialization.data(withJSONObject: obj)) ?? Data() }

    // ── Pairing ──

    func pairStart(label: String) async throws -> PairStart {
        let data = try await request("POST", "/fd/clip/pair", body: body(["browser": label, "kind": "desktop"]),
                                     type: "application/json", auth: false)
        return try json(PairStart.self, data)
    }

    func pairPoll(_ id: String) async throws -> PairPoll {
        let data = try await request("GET", "/fd/clip/pair/\(id)", auth: false)
        return try json(PairPoll.self, data)
    }

    func disconnect() async {
        _ = try? await request("POST", "/fd/clip/disconnect", timeout: 10)
    }

    // ── What the app needs ──

    func me() async throws -> MeInfo {
        try json(MeInfo.self, try await request("GET", "/fd/clip/me", timeout: 20))
    }

    func send(meta: [String: Any], parts: [FormPart]) async throws -> SendResult {
        let boundary = "captain-\(UUID().uuidString)"
        var data = Data()
        func add(_ s: String) { data.append(Data(s.utf8)) }
        let metaJSON = (try? JSONSerialization.data(withJSONObject: meta)) ?? Data("{}".utf8)
        add("--\(boundary)\r\nContent-Disposition: form-data; name=\"meta\"; filename=\"meta.json\"\r\nContent-Type: application/json\r\n\r\n")
        data.append(metaJSON)
        add("\r\n")
        for p in parts {
            let fname = p.filename.replacingOccurrences(of: "\"", with: "'")
            add("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(p.name)\"; filename=\"\(fname)\"\r\nContent-Type: \(p.type)\r\n\r\n")
            data.append(p.data)
            add("\r\n")
        }
        add("--\(boundary)--\r\n")
        let out = try await request("POST", "/fd/clip/send", body: data, type: "multipart/form-data; boundary=\(boundary)",
                                    timeout: 180)
        return try json(SendResult.self, out)
    }

    func run(_ id: String, since: Int) async throws -> RunState {
        try json(RunState.self, try await request("GET", "/fd/clip/runs/\(id)?since=\(since)", timeout: 20))
    }

    func stop(_ id: String) async {
        _ = try? await request("POST", "/fd/clip/runs/\(id)/stop", timeout: 10)
    }

    func file(_ id: String, path: String) async throws -> Data {
        let q = path.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed.subtracting(CharacterSet(charactersIn: "&=+"))) ?? path
        return try await request("GET", "/fd/clip/runs/\(id)/file?path=\(q)", timeout: 120)
    }

    func newTopic(_ slug: String) async throws {
        _ = try await request("POST", "/fd/clip/agents/\(slug)/new-topic", timeout: 40)
    }
}
