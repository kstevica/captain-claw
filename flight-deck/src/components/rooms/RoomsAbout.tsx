import { useEffect, useState } from 'react'
import { BookOpenText, ChevronDown, ChevronUp, Loader2, Sparkles, Square } from 'lucide-react'
import { TOKEN_RE, type VaultEntry } from '../../services/anonymizeApi'
import type { FileAbout, Room, RoomFile } from '../../services/roomsApi'
import { Button, Pill, Segmented } from '../console/parts'
import { Marked } from './roomParts'
import { baseToken, count, duration } from './roomText'

// What each file of a room is about: one line, a paragraph, a summary,
// keywords and topics — written by the trusted model from the text Claude
// reads, so Claude (and you) can pick files without reading them all.

/** *text* with the room's tokens as the real names (for its owner's page). */
export function withNames(text: string, byToken: Map<string, VaultEntry>): string {
  return text.replace(TOKEN_RE, (t) => byToken.get(baseToken(t))?.original || t)
}

/** Describing the room's files, in the background: n of N, the file it's on, time left. */
export function DescribeStrip({ room, files, stopping, onStop }: {
  room: Room
  files: RoomFile[]
  stopping: boolean
  onStop: () => void
}) {
  const d = room.describing
  const [now, setNow] = useState(() => Date.now() / 1000)
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => clearInterval(t)
  }, [])
  if (!d) return null
  const n = d.done + d.failed
  const left = Math.max(0, d.of - n)
  const per = n > 0 && d.started_at ? (now - d.started_at) / n : null
  const file = d.current ? files.find((f) => f.id === d.current) : undefined
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-zinc-800 bg-sky-500/[0.04] px-4 py-2 text-xs">
      <Loader2 className="h-3.5 w-3.5 animate-spin text-sky-500" />
      <span className="font-medium text-zinc-100"
        title="The trusted model reads the text Claude reads and writes what each file is about: one line, a paragraph, a summary, keywords and topics. Claude uses them to find the right files.">
        Describing files
      </span>
      <span className="tabular-nums text-zinc-500">{n} of {d.of} done</span>
      <span className="h-1.5 w-40 overflow-hidden rounded-full bg-zinc-800">
        <span className="block h-full rounded-full bg-sky-500 transition-all duration-700"
          style={{ width: `${d.of ? (n / d.of) * 100 : 0}%` }} />
      </span>
      {per !== null && left > 0 && <span className="text-zinc-400">{duration(per * left)} left</span>}
      {d.failed > 0 && <Pill tone="amber">{d.failed} failed</Pill>}
      {d.waiting_for_slot
        ? <span className="text-zinc-500">waiting for a room's run to finish</span>
        : file && <span className="max-w-[24rem] truncate text-zinc-300" title={file.path}>{file.path.split('/').pop()}</span>}
      <span className="ml-auto flex items-center gap-3">
        {d.model && <span className="hidden text-zinc-500 xl:inline">{d.model}</span>}
        <Button icon={Square} busy={stopping} onClick={onStop} className="!px-2.5 !py-1 !text-xs">Stop</Button>
      </span>
    </div>
  )
}

/** How far the room's files are described, and the buttons to describe them. */
export function DescribeActions({ room, busy, onDescribe }: {
  room: Room
  busy: string
  onDescribe: (again: boolean) => void
}) {
  const c = room.counts
  if (!room.trusted_model || room.describing || !c.done) return null
  const needed = c.describe_needed || 0
  return (
    <>
      {needed > 0 && (
        <Button icon={Sparkles} busy={busy === 'describe'} onClick={() => onDescribe(false)}
          title="The trusted model writes what each file is about — one line, a paragraph, a summary, keywords and topics — for files that have none yet, whose text changed, or whose description failed. Claude finds files with them.">
          Describe {count(needed, 'file')}
        </Button>
      )}
      {(c.described || 0) > 0 && (
        <Button variant="ghost" icon={Sparkles} busy={busy === 'describe-all'} onClick={() => onDescribe(true)}
          title="Write every clean file's description again, from scratch.">
          {needed > 0 ? 'Describe all again' : 'Describe again'}
        </Button>
      )}
    </>
  )
}

const STATE: Record<string, { label: string; tone: 'zinc' | 'amber' | 'red' | 'sky' }> = {
  waiting: { label: 'Waiting to be described', tone: 'sky' },
  failed: { label: 'Describing failed', tone: 'amber' },
  held: { label: 'Held back', tone: 'red' },
  unavailable: { label: 'No trusted model', tone: 'zinc' },
}

/** One file's description, on its page: in real names (or as Claude reads it). */
export function AboutPanel({ file, about, byToken, canDescribe, busy, onDescribe }: {
  file: RoomFile
  about: FileAbout | null | undefined
  byToken: Map<string, VaultEntry>
  canDescribe: boolean
  busy: boolean
  onDescribe: () => void
}) {
  const [open, setOpen] = useState(false)
  const [asClaude, setAsClaude] = useState<'real' | 'claude'>('real')
  if (file.state !== 'done') return null
  const has = !!about?.one
  const state = about?.state || ''
  const st = STATE[state]
  const real = about?.real
  const showReal = asClaude === 'real'
  const text = (key: 'one' | 'paragraph' | 'summary') => {
    const t = showReal ? (real?.[key] ?? withNames(about?.[key] || '', byToken)) : about?.[key] || ''
    return showReal ? t : <Marked text={t} ranges={[]} tokens />
  }
  const terms = (key: 'keywords' | 'topics') => (showReal ? real?.[key] : about?.[key]) || about?.[key] || []
  const chip = (t: string, i: number, tone: string) => (
    <span key={i} className={`inline-flex max-w-full items-center rounded-full px-2 py-0.5 text-[11px] ${tone}`}>
      <span className="truncate">{showReal ? t : <Marked text={t} ranges={[]} tokens />}</span>
    </span>
  )

  return (
    <section className="border-b border-zinc-800 px-4 py-3">
      <div className="flex items-start gap-2.5">
        <BookOpenText className="mt-0.5 h-4 w-4 shrink-0 text-sky-500" />
        <div className="min-w-0 flex-1 space-y-1.5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-zinc-100">About this file</span>
            {st && <Pill tone={st.tone}>{state === 'waiting' ? <span className="inline-flex items-center gap-1"><Loader2 className="h-3 w-3 animate-spin" />{st.label}</span> : st.label}</Pill>}
            {about?.stale && <Pill tone="amber" title="Its text changed after it was described — it's described again on the next pass.">Text changed since</Pill>}
            {about?.cut && state === 'done' && <span className="text-[11px] text-zinc-500">from parts of a long file</span>}
            <div className="ml-auto flex items-center gap-2">
              {has && (
                <Segmented label="Show" value={asClaude} onChange={setAsClaude} options={[
                  { value: 'real', label: 'Real names' }, { value: 'claude', label: 'As Claude reads it' },
                ]} />
              )}
              {canDescribe && state !== 'waiting' && (
                <Button icon={Sparkles} busy={busy} onClick={onDescribe} className="!px-2.5 !py-1 !text-xs">
                  {has ? 'Describe again' : 'Describe'}
                </Button>
              )}
            </div>
          </div>
          {about?.error && state !== 'waiting' && (
            <p className={`text-xs ${state === 'held' ? 'text-red-600 dark:text-red-400' : 'text-amber-700 dark:text-amber-300'}`}>{about.error}</p>
          )}
          {!has && !about?.error && state !== 'waiting' && (
            <p className="text-xs text-zinc-500">
              {canDescribe ? 'Not described yet. Claude finds files by what they are about — describe it, or the whole room from Files.'
                : 'Not described: describing needs a trusted model (an admin sets one up under Anonymization).'}
            </p>
          )}
          {has && (
            <>
              <p className="text-[13.5px] font-medium leading-relaxed text-zinc-200">{text('one')}</p>
              {(terms('keywords').length > 0 || terms('topics').length > 0) && (
                <div className="flex flex-wrap gap-1.5">
                  {terms('topics').map((t, i) => chip(t, i, 'bg-sky-500/10 text-sky-800 dark:text-sky-300'))}
                  {terms('keywords').map((t, i) => chip(t, i + 100, 'bg-zinc-800 text-zinc-300'))}
                </div>
              )}
              {(about?.paragraph || about?.summary) && (
                <button type="button" onClick={() => setOpen(!open)}
                  className="inline-flex items-center gap-1 text-xs font-medium text-sky-700 hover:underline dark:text-sky-400">
                  {open ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                  {open ? 'Less' : 'Paragraph and summary'}
                </button>
              )}
              {open && (
                <div className="max-h-72 space-y-3 overflow-y-auto rounded-xl border border-zinc-800 bg-zinc-950/40 p-3 text-[13px] leading-relaxed text-zinc-300">
                  {about?.paragraph && <p>{text('paragraph')}</p>}
                  {about?.summary && (
                    <div className="space-y-2 border-t border-zinc-800 pt-3">
                      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-zinc-500">Summary</div>
                      {(showReal ? (real?.summary ?? withNames(about.summary, byToken)) : about.summary)
                        .split(/\n\n+/).map((p, i) => <p key={i}>{showReal ? p : <Marked text={p} ranges={[]} tokens />}</p>)}
                    </div>
                  )}
                  {about?.model && <div className="text-[11px] text-zinc-500">Written by {about.model}</div>}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  )
}
