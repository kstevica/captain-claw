import type { Room } from '../../services/roomsApi'

// Plain helpers the data room's views share (no components here, so fast
// refresh keeps working in the files that hold them).

export function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

export function count(n: number, one: string, many = `${one}s`): string {
  return `${n.toLocaleString()} ${n === 1 ? one : many}`
}

/** "company" → "companies", "address" → "addresses", "person" → "people". */
export function plural(word: string, n: number): string {
  if (n === 1) return word
  const special: Record<string, string> = { person: 'people', 'date of birth': 'dates of birth' }
  if (special[word]) return special[word]
  if (/[^aeiou]y$/i.test(word)) return `${word.slice(0, -1)}ies`
  if (/(s|x|z|ch|sh)$/i.test(word)) return `${word}es`
  return `${word}s`
}

export function fileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / 1024 / 1024).toFixed(bytes < 10 * 1024 * 1024 ? 1 : 0)} MB`
}

/** 950 → "950", 48_200 → "48K", 1_234_567 → "1.2M". */
export function compact(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e4) return `${Math.round(n / 1e3)}K`
  return n.toLocaleString()
}

/** "about 3 min", "about 1 h 20 min", "under a minute". */
export function duration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return ''
  if (seconds < 60) return 'under a minute'
  const m = Math.round(seconds / 60)
  if (m < 60) return `about ${m} min`
  const h = Math.floor(m / 60), rest = m % 60
  return `about ${h} h${rest ? ` ${rest} min` : ''}`
}

/** 42 → "0:42", 3725 → "1:02:05". */
export function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}` : `${m}:${String(r).padStart(2, '0')}`
}

// The code pass's passes, in order (a final check, when there is one, comes last).
const PASS_ORDER: Record<string, number> = { '': 0, looking: 0, matching: 1, replacing: 2, checking: 2.5 }

/** How far the code pass is through its round, 0..1 (looking, matching, replacing). */
export function replaceShare(r?: { pass: string; done: number; of: number } | null): number {
  if (!r) return 0
  const within = r.of ? Math.min(1, r.done / r.of) : 0
  return Math.min(1, ((PASS_ORDER[r.pass] ?? 0) + within) / 3)
}

/** Time left in the code pass's current pass (null until there's enough to go on). */
export function passEta(r: { done: number; of: number; started_at: number } | null | undefined, now: number): number | null {
  if (!r || r.done <= 0 || now - r.started_at < 2) return null
  return Math.max(0, ((r.of - r.done) * (now - r.started_at)) / r.done)
}

type EtaJob = {
  phase?: string; step?: string; chunk?: number; chunks?: number; chunks_started_at?: number | null
  started_at?: number; phase_started_at?: number
  bytes_total?: number; bytes_done?: number; current_size?: number; file_index?: number; file_count?: number
  replace?: { pass: string; done: number; of: number; started_at: number }
}

// Rough shares of a file's time, step by step. Anonymizing: reading a little,
// the trusted model (when it reads) most, then the code pass — which is most
// of it for a big workbook. The sweep is the code pass alone; the check is
// mostly making the text Claude reads.
const CHECK_SHARE: Record<string, number> = {
  reading: 0.02, text: 0.1, leaks: 0.7, hiding: 0.72, replacing: 0.75, saving: 0.8, writing: 0.8, indexing: 0.85,
}

export function fileShare(job: EtaJob): number {
  const chunks = job.chunks || 0, chunk = job.chunk || 0
  const wrapping = job.step === 'saving' || job.step === 'writing'
  if (job.phase === 'sweeping') {
    return wrapping ? 0.9 : job.step === 'replacing' ? 0.1 + 0.8 * replaceShare(job.replace) : 0.02
  }
  if (job.phase === 'checking') return CHECK_SHARE[job.step || ''] ?? 0
  return wrapping ? 0.97
    : job.step === 'replacing' ? 0.5 + 0.45 * replaceShare(job.replace)
      : job.step === 'model' && chunks ? (chunk / chunks) * 0.45 + 0.05 : 0.02
}

/** Time left for the current file and for the whole phase (the files to
 *  anonymize, a sweep round, the check), from how fast it went so far; null
 *  where there isn't enough to go on yet. */
export function runEta(job: EtaJob, now: number): { file: number | null; run: number | null; fileShare: number; share: number } {
  const chunks = job.chunks || 0, chunk = job.chunk || 0
  const part = fileShare(job)
  let file: number | null = null
  if (job.step === 'model' && chunk > 0 && job.chunks_started_at) {
    const per = (now - job.chunks_started_at) / chunk
    file = Math.max(0, (chunks - chunk) * per)
  } else if (job.step === 'replacing') {
    file = passEta(job.replace, now)
  }
  let run: number | null = null
  const total = job.bytes_total || 0
  const done = (job.bytes_done || 0) + (job.current_size || 0) * part
  const since = job.phase_started_at || job.started_at
  const spent = since ? now - since : 0
  if (total > 0 && done > total * 0.02 && spent > 10) run = Math.max(0, ((total - done) * spent) / done)
  // How far the phase is, 0..1: by size, or by files when sizes aren't known.
  const share = total > 0 ? Math.min(1, done / total)
    : job.file_count ? Math.min(1, ((job.file_index || 1) - 1 + part) / job.file_count) : 0
  return { file, run, fileShare: part, share }
}

export function tokenKind(token: string): string {
  return token.replace(/^⟦|⟧$/g, '').replace(/_\d+(:[a-z0-9]+)?$/, '')
}

/** "⟦ORG_7:2⟧" → "⟦ORG_7⟧": the entity a spelling belongs to. */
export function baseToken(token: string): string {
  return token.replace(/:[a-z0-9]{1,3}⟧$/, '⟧')
}

export type Range = readonly [number, number]

const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** Where *needles* appear in *text*, case-insensitively, longest first, no overlaps. */
export function findAll(text: string, needles: string[]): Range[] {
  const words = [...new Set(needles.map((n) => n.trim()).filter((n) => n.length > 0))].sort((a, b) => b.length - a.length)
  if (!words.length) return []
  const re = new RegExp(words.map(esc).join('|'), 'giu')
  const out: Range[] = []
  for (const m of text.matchAll(re)) {
    if (!m[0]) continue
    out.push([m.index ?? 0, (m.index ?? 0) + m[0].length])
  }
  return out
}

export function findRe(text: string, re: RegExp): Range[] {
  const g = new RegExp(re.source, re.flags.includes('g') ? re.flags : `${re.flags}g`)
  return [...text.matchAll(g)].filter((m) => m[0]).map((m) => [m.index ?? 0, (m.index ?? 0) + m[0].length] as Range)
}

export const STATE_LABEL: Record<Room['state'], string> = {
  setup: 'Setting up', running: 'Running', review: 'Ready to check', released: 'Released',
}
export const STATE_TONE: Record<Room['state'], 'zinc' | 'sky' | 'amber' | 'emerald'> = {
  setup: 'zinc', running: 'sky', review: 'amber', released: 'emerald',
}
