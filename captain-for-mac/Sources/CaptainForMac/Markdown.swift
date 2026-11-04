import SwiftUI

// The agent's answer as Markdown: headings, lists, quotes, code and tables
// (as text), with inline bold/italic/code/links.

struct MarkdownView: View {
    let text: String
    var size: CGFloat = 13

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            ForEach(Array(MD.blocks(text).enumerated()), id: \.offset) { _, b in
                block(b)
            }
        }
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func block(_ b: MD.Block) -> some View {
        switch b {
        case .heading(let level, let s):
            MD.inline(s)
                .font(.system(size: size + (level <= 1 ? 5 : level == 2 ? 3 : 1), weight: .semibold))
                .padding(.top, 2)
                .fixedSize(horizontal: false, vertical: true)
        case .paragraph(let s):
            MD.inline(s).font(.system(size: size)).fixedSize(horizontal: false, vertical: true)
        case .item(let marker, let s, let indent):
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(marker).font(.system(size: size)).foregroundStyle(.secondary)
                    .frame(minWidth: 12, alignment: .trailing)
                MD.inline(s).font(.system(size: size)).fixedSize(horizontal: false, vertical: true)
            }
            .padding(.leading, CGFloat(indent) * 14)
        case .quote(let s):
            HStack(alignment: .top, spacing: 8) {
                Rectangle().fill(Color.secondary.opacity(0.4)).frame(width: 3)
                MD.inline(s).font(.system(size: size)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        case .code(let s):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(s).font(.system(size: size - 1, design: .monospaced)).padding(8)
            }
            .background(RoundedRectangle(cornerRadius: 6).fill(Color.primary.opacity(0.07)))
        case .rule:
            Divider()
        }
    }
}

enum MD {
    enum Block {
        case heading(Int, String)
        case paragraph(String)
        case item(String, String, Int)
        case quote(String)
        case code(String)
        case rule
    }

    static func inline(_ s: String) -> Text {
        if let a = try? AttributedString(markdown: s, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)) {
            return Text(a)
        }
        return Text(s)
    }

    static func blocks(_ text: String) -> [Block] {
        var out: [Block] = []
        var para: [String] = []
        var quote: [String] = []
        var code: [String]? = nil
        var table: [String] = []

        func flushPara() {
            if !para.isEmpty { out.append(.paragraph(para.joined(separator: "\n"))); para = [] }
        }
        func flushQuote() {
            if !quote.isEmpty { out.append(.quote(quote.joined(separator: "\n"))); quote = [] }
        }
        func flushTable() {
            if !table.isEmpty { out.append(.code(table.joined(separator: "\n"))); table = [] }
        }
        func flushAll() { flushPara(); flushQuote(); flushTable() }

        for raw in text.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n") {
            let line = raw
            let t = line.trimmingCharacters(in: .whitespaces)
            if var c = code {
                if t.hasPrefix("```") {
                    out.append(.code(c.joined(separator: "\n")))
                    code = nil
                } else {
                    c.append(line)
                    code = c
                }
                continue
            }
            if t.hasPrefix("```") {
                flushAll()
                code = []
                continue
            }
            if t.isEmpty {
                flushAll()
                continue
            }
            if t.hasPrefix("|") {
                flushPara(); flushQuote()
                if t.replacingOccurrences(of: "[|:\\- ]", with: "", options: .regularExpression).isEmpty { continue }
                table.append(t)
                continue
            } else {
                flushTable()
            }
            if t == "---" || t == "***" || t == "___" {
                flushAll()
                out.append(.rule)
                continue
            }
            if t.hasPrefix("#") {
                let level = t.prefix(while: { $0 == "#" }).count
                if level <= 6, t.dropFirst(level).first == " " {
                    flushAll()
                    out.append(.heading(level, String(t.dropFirst(level + 1))))
                    continue
                }
            }
            if t.hasPrefix(">") {
                flushPara()
                quote.append(String(t.dropFirst()).trimmingCharacters(in: .whitespaces))
                continue
            }
            let indent = (line.prefix(while: { $0 == " " }).count) / 2
            if t.hasPrefix("- ") || t.hasPrefix("* ") || t.hasPrefix("+ ") {
                flushPara(); flushQuote()
                var body = String(t.dropFirst(2))
                var marker = "•"
                if body.hasPrefix("[ ] ") { marker = "☐"; body = String(body.dropFirst(4)) }
                else if body.hasPrefix("[x] ") || body.hasPrefix("[X] ") { marker = "☑"; body = String(body.dropFirst(4)) }
                out.append(.item(marker, body, indent))
                continue
            }
            if let m = t.range(of: "^[0-9]{1,3}[.)] ", options: .regularExpression) {
                flushPara(); flushQuote()
                out.append(.item(String(t[m]).trimmingCharacters(in: .whitespaces), String(t[m.upperBound...]), indent))
                continue
            }
            flushQuote()
            para.append(t)
        }
        if let c = code { out.append(.code(c.joined(separator: "\n"))) }
        flushAll()
        return out
    }
}
