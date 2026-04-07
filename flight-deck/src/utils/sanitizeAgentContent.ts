/**
 * Shared sanitizers for agent text output.
 *
 * These strip:
 *  - <think>/<thinking>/<reasoning>/<reflection>/<inner_monologue>/<scratchpad> blocks
 *  - Council protocol headers (SUITABILITY / ACTION / TARGET) — both as a
 *    structured block at the top and as inline prose prefixes
 *  - Memory retrieval echoes ([session] ..., [tool] ..., (score=0.xx),
 *    sessions/xxx.txt:NN)
 *  - Insight echoes ([fact] (imp:N) ..., [contact] (imp:N) ..., etc.)
 *  - Instruction-block echoes ("You are participating in a Council...",
 *    "Verbosity rule:", "Other participants:", etc.)
 *
 * Used by both councilStore and chatStore so chat windows benefit from the
 * same protections as council sessions.
 */

/**
 * Strip thinking/reasoning blocks emitted as raw tags.
 *
 * Handles <think>, <thinking>, <reasoning>, <reflection>, <inner_monologue>,
 * and <scratchpad> — both self-closing and unclosed forms.
 */
export function stripThinkingBlocks(text: string): string {
  let result = text
  const thinkTags = ['think', 'thinking', 'reasoning', 'reflection', 'inner_monologue', 'scratchpad']
  for (const tag of thinkTags) {
    const blockRe = new RegExp(`<${tag}>[\\s\\S]*?</${tag}>`, 'gi')
    result = result.replace(blockRe, '')
    const unclosedRe = new RegExp(`<${tag}>(?:(?!</${tag}>)[\\s\\S])*$`, 'gi')
    result = result.replace(unclosedRe, '')
  }
  return result
}

/**
 * Sanitize an agent response for display.
 *
 * Strategy: look for the SUITABILITY/ACTION/TARGET header block and take
 * everything AFTER the last header line. If no headers are found, fall back
 * to aggressive line-by-line cleaning.
 */
export function sanitizeAgentContent(raw: string): string {
  // First, strip all thinking/reasoning blocks
  let text = stripThinkingBlocks(raw)

  // Strip [session] and [tool] memory retrieval blocks (multi-line)
  text = text.replace(/^\[session\]\s.*(?:\n(?:\s.*|\(score=.*|sessions\/.*|\n))*$/gm, '')
  text = text.replace(/^\[tool\]\s.*(?:\n(?:\s.*|\(score=.*|\n))*$/gm, '')
  text = text.replace(/^\[user\]\s*(?:COUNCIL ROUND|SUITABILITY).*$/gm, '')

  // Header separator: colon, em-dash, en-dash, or hyphen (with optional surrounding whitespace)
  const SEP = '\\s*[:\\-–—]\\s*'

  // "Protocol" headers carry a structured value (numeric score / enum action)
  // and can be dropped wholesale. "Prose" headers carry free-form text after
  // the separator and we want to keep the prose, only stripping the prefix.
  const protoSuitRe = new RegExp(`^\\**SUITABILITY\\**${SEP}[\\d.]+\\s*$`, 'i')
  const protoActionRe = new RegExp(`^\\**ACTION\\**${SEP}(answer|respond|challenge|refine|broaden|pass)\\s*$`, 'i')

  // Strip lines where all three headers are on one line
  text = text.replace(new RegExp(`^\\**SUITABILITY\\**${SEP}[\\d.]+\\s+\\**ACTION\\**${SEP}\\w+\\s+\\**TARGET\\**${SEP}.+$`, 'gmi'), '')
  // Two-header combos on one line
  text = text.replace(new RegExp(`^\\**SUITABILITY\\**${SEP}[\\d.]+\\s+\\**ACTION\\**${SEP}\\w+\\s*$`, 'gmi'), '')

  // Find the last *protocol* header line.
  const lines = text.split('\n')
  let lastHeaderIdx = -1

  const stripMd = (s: string) => s.replace(/\*{1,2}|_{1,2}/g, '')

  const targetHeaderRe = new RegExp(`^TARGET${SEP}`, 'i')

  // Numeric / enum value detectors (after the separator)
  const suitValueRe = new RegExp(`^SUITABILITY${SEP}[\\d.]+\\s*$`, 'i')
  const actionValueRe = new RegExp(`^ACTION${SEP}(answer|respond|challenge|refine|broaden|pass)\\s*$`, 'i')

  // Also recognize prose-style protocol headers like "ACTION: Introduce identity..."
  // when they appear at the very top of the message (chat-mode leak).
  const proseSuitTopRe = new RegExp(`^\\**SUITABILITY\\**${SEP}.+$`, 'i')
  const proseActionTopRe = new RegExp(`^\\**ACTION\\**${SEP}.+$`, 'i')

  let sawProtocolHeader = false
  for (let i = 0; i < lines.length; i++) {
    const trimmed = stripMd(lines[i].trim())
    if (suitValueRe.test(trimmed)) { lastHeaderIdx = i; sawProtocolHeader = true }
    if (actionValueRe.test(trimmed)) { lastHeaderIdx = i; sawProtocolHeader = true }
    // TARGET is only a protocol delimiter if we've already seen a protocol SUITABILITY/ACTION
    // line above it. Otherwise "TARGET — long prose" is content, not a header.
    if (sawProtocolHeader && targetHeaderRe.test(trimmed)) lastHeaderIdx = i
  }

  // Chat-mode fallback: if no structured value headers were found, but the
  // first non-blank lines are prose-style "SUITABILITY: 100%" / "ACTION: ..."
  // headers, treat them as a header block to strip.
  if (lastHeaderIdx < 0) {
    let firstNonBlank = -1
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].trim()) { firstNonBlank = i; break }
    }
    if (firstNonBlank >= 0) {
      let cursor = firstNonBlank
      let matched = false
      while (cursor < lines.length) {
        const trimmed = stripMd(lines[cursor].trim())
        if (!trimmed) { cursor++; continue }
        if (proseSuitTopRe.test(trimmed) || proseActionTopRe.test(trimmed) || targetHeaderRe.test(trimmed)) {
          lastHeaderIdx = cursor
          matched = true
          cursor++
          continue
        }
        break
      }
      if (!matched) lastHeaderIdx = -1
    }
  }

  let content: string
  if (lastHeaderIdx >= 0) {
    content = lines.slice(lastHeaderIdx + 1).join('\n')
  } else {
    content = text
  }

  // Strip prose-style header prefixes: "**SUITABILITY** — text" → "text".
  const proseHeaderPrefix = new RegExp(`^\\s*\\**(SUITABILITY|ACTION|TARGET)\\**${SEP}`, 'i')
  content = content
    .split('\n')
    .map(line => line.replace(proseHeaderPrefix, ''))
    .join('\n')

  // Final cleanup: strip any remaining noise lines that leaked through
  const cleaned = content.split('\n').filter(line => {
    const t = line.trim()
    if (!t) return true  // preserve blank lines for formatting
    const tb = t.replace(/\*{1,2}|_{1,2}/g, '')
    if (protoSuitRe.test(tb)) return false
    if (protoActionRe.test(tb)) return false
    // Memory retrieval
    if (/^\[session\]\s/i.test(t) || /^\[tool\]\s/i.test(t)) return false
    if (/^\(score=[\d.]+\)/.test(t)) return false
    if (/^sessions\/\w+\.txt:\d+/.test(t)) return false
    // Insight echoes
    if (/^\[(fact|contact|decision|preference|deadline|project|workflow)\]\s*\(imp:\d+\)/i.test(t)) return false
    if (/^[-•*]\s*\[(fact|contact|decision|preference|deadline|project|workflow)\]\s*\(imp:\d+\)/i.test(t)) return false
    // Instruction echoes
    if (/^#{1,3}\s*(Task|Instructions|Items to process|Context)\b/i.test(t)) return false
    if (/^Provide your contribution now/i.test(t)) return false
    if (/^Remember to start with SUITABILITY/i.test(t)) return false
    if (/^Structure your response starting with/i.test(t)) return false
    if (/^You are participating in a Council/i.test(t)) return false
    if (/^Session type:/i.test(t)) return false
    if (/^Verbosity rule:/i.test(t)) return false
    if (/^Other participants:/i.test(t)) return false
    if (/^COUNCIL ROUND \d/i.test(t)) return false
    if (/^Discussion so far:/i.test(t)) return false
    if (/^Follow the verbosity rule/i.test(t)) return false
    if (/^Before each of your contributions/i.test(t)) return false
    if (/^Formulate a (contribution|response) that/i.test(t)) return false
    if (/^Analyze the (existing|ongoing) (arguments|debate|discussion)/i.test(t)) return false
    if (/^\\n#{1,3}\s/.test(t)) return false
    return true
  })

  return cleaned.join('\n').trim()
}
