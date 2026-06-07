import Prism from 'prismjs'

// Custom Prism grammar for the Captain Claw Flow DSL (see captain_claw/
// flight_deck/flow_dsl.py and FLOWS.md). Line-oriented: a flow = directives +
// `step <id>:` blocks with indented fields, `{{...}}` interpolation, `#` line
// comments. Order matters — strings/comments are matched (greedy) first so a
// `#` inside a string or `{{` inside a comment isn't mis-tokenized.
Prism.languages.flow = {
  string: {
    pattern: /"(?:\\.|[^"\\\r\n])*"|'(?:\\.|[^'\\\r\n])*'/,
    greedy: true,
  },
  comment: {
    pattern: /#.*/,
    greedy: true,
  },
  // {{ trigger.x }} / {{ steps.id.output }} interpolation
  template: {
    pattern: /\{\{[^{}]*\}\}/,
    inside: {
      punctuation: /\{\{|\}\}|\./,
      keyword: /\b(?:trigger|steps|args|vars|calls|joins|error)\b/,
    },
  },
  // Top-level directives at the start of a line.
  directive: {
    pattern: /(^[ \t]*)(?:flow|description|priority|trigger|step|output)\b/m,
    lookbehind: true,
  },
  // Step types + control-flow / rule keywords.
  keyword:
    /\b(?:tool|agent|vision|input|emit|branch|gosub|return|spawn|join|error|set|foreach|while|sleep|wait|if|elif|else|stop|when|in|on|with|arg|deny|retry|timeout|prompt|attach|image|channel|has|contains|matches|from_waid|mime|regex|and|or|not|always)\b/,
  // Channels / execution targets.
  builtin: {
    pattern: /\b(?:any|whatsapp|web|glasses|fd|capability|same|log)\b/,
    alias: 'function',
  },
  boolean: /\b(?:true|false|null)\b/,
  // `name:` field labels.
  field: {
    pattern: /\b[A-Za-z_]\w*(?=\s*:)/,
    alias: 'attr-name',
  },
  operator: /->|[=!<>]=|[<>=+*/-]/,
  number: /\b\d+(?:\.\d+)?\b/,
  punctuation: /[{}()[\],:]/,
}
