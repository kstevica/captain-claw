import { useCallback, useEffect, useState } from 'react'
import { ArrowLeftRight, Check, CheckCheck, Combine, Loader2, X } from 'lucide-react'
import { roomsApi, type AlikeMerged, type AlikePair, type Room } from '../../services/roomsApi'
import { Button, EmptyState, Field, Notice, Segmented, inputCls } from '../console/parts'
import { SectionHead } from './RoomsInbox'
import { TokenChip } from './roomParts'
import { count, errorText } from './roomText'

// "Same party?": tokens code couldn't prove are one person, company or
// address — a one-letter typo, a first name alone, a surname that changed
// between files, a company's short name — for the room's owner to say. Code
// scores each pair (how sure it is the two are one); the slider shows the
// pairs on one side of a score, and one click answers all of them: "Same"
// for those at or above it, "Not the same" for those under. "Same" makes the
// second token the first one's spelling in the vault and in every copy in the
// room; "Not the same" is remembered.

type Kind = 'all' | 'PERSON' | 'ORG' | 'ADDRESS'
type Side = 'above' | 'below'

const KIND_LABEL: Record<AlikePair['kind'], string> = { PERSON: 'People', ORG: 'Companies', ADDRESS: 'Addresses' }
const DEFAULT_SCORE = 90

function readPref(key: string): { side: Side; score: number } {
  try {
    const v = JSON.parse(localStorage.getItem(key) || 'null')
    if (v && (v.side === 'above' || v.side === 'below') && Number.isInteger(v.score) && v.score >= 0 && v.score <= 99) {
      return { side: v.side, score: v.score }
    }
  } catch { /* no storage here: the defaults */ }
  return { side: 'above', score: DEFAULT_SCORE }
}

function writePref(key: string, side: Side, score: number) {
  try { localStorage.setItem(key, JSON.stringify({ side, score })) } catch { /* not kept: fine */ }
}

export function AlikeSection({ room, disabled, onChanged }: {
  room: Room
  disabled: boolean
  onChanged: (count: number) => void
}) {
  const prefKey = `rooms.alike.${room.id}`
  const [pairs, setPairs] = useState<AlikePair[] | null>(null)
  const [merged, setMerged] = useState<AlikeMerged[]>([])
  const [kind, setKind] = useState<Kind>('all')
  const [side, setSide] = useState<Side>(() => readPref(prefKey).side)
  const [score, setScore] = useState(() => readPref(prefKey).score)
  // Pairs whose arrows were clicked: the second one is kept.
  const [swapped, setSwapped] = useState<Set<string>>(() => new Set())
  const [confirming, setConfirming] = useState(false)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [keepTok, setKeepTok] = useState('')
  const [otherTok, setOtherTok] = useState('')

  useEffect(() => { writePref(prefKey, side, score) }, [prefKey, side, score])
  useEffect(() => { setConfirming(false) }, [kind, side, score])

  const load = useCallback(async () => {
    try {
      const r = await roomsApi.alike(room.id)
      setPairs(r.suggestions)
      setMerged(r.merged)
      onChanged(r.suggestions.length)
    } catch (e) {
      setError(errorText(e))
      setPairs([])
    }
  }, [room.id, onChanged])
  useEffect(() => { void load() }, [load])

  const sides = (p: AlikePair) => (swapped.has(p.key) ? { keep: p.b, other: p.a } : { keep: p.a, other: p.b })

  const same = async (keep: string, other: string, label: string) => {
    setBusy(label); setError(''); setNotice('')
    try {
      const r = await roomsApi.same(room.id, keep, other)
      setNotice(`${other} is ${keep} now — ${count(r.rewritten, 'file')} rewritten in place`
        + (r.requeued ? `, ${count(r.requeued, 'file')} to run again (click Run at the top)` : '') + '.')
      setKeepTok(''); setOtherTok('')
      await load()
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy('')
    }
  }

  const drop = (keys: Set<string>) => setPairs((cur) => {
    const next = (cur || []).filter((x) => !keys.has(x.key))
    onChanged(next.length)
    return next
  })

  const notSame = async (p: AlikePair) => {
    setBusy(`no:${p.key}`); setError(''); setNotice('')
    try {
      await roomsApi.notSame(room.id, p.a.token, p.b.token)
      drop(new Set([p.key]))
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy('')
    }
  }

  const ofKind = (pairs || []).filter((p) => kind === 'all' || p.kind === kind)
  const above = ofKind.filter((p) => p.score >= score)
  const below = ofKind.filter((p) => p.score < score)
  const shown = side === 'above' ? above : below
  const kinds = (k: AlikePair['kind']) => (pairs || []).filter((p) => p.kind === k).length

  const answerAll = async () => {
    const batch = shown
    setConfirming(false); setBusy('all'); setError(''); setNotice('')
    try {
      if (side === 'above') {
        const r = await roomsApi.sameMany(room.id, batch.map((p) => {
          const s = sides(p)
          return { keep: s.keep.token, other: s.other.token }
        }))
        setNotice(`${count(r.merged, 'pair')} made one — ${count(r.rewritten, 'file')} rewritten in place`
          + (r.requeued ? `, ${count(r.requeued, 'file')} to run again (click Run at the top)` : '')
          + (r.skipped.length ? `. ${count(r.skipped.length, 'pair')} left as they were: ${r.skipped[0].why.replace(/\.$/, '')}`
            + (r.skipped.length > 1 ? ' (and the like)' : '') : '') + '.')
        await load()
      } else {
        await roomsApi.notSameMany(room.id, batch.map((p) => ({ a: p.a.token, b: p.b.token })))
        drop(new Set(batch.map((p) => p.key)))
        setNotice(`${count(batch.length, 'pair')} won't be asked about again.`)
      }
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy('')
    }
  }

  return (
    <div className="space-y-4">
      <SectionHead icon={Combine} title="Same party?"
        hint="Code gives one token to what it can prove is one person, company or address. These it couldn't prove: a one-letter typo, a first name alone, a surname that changed between files, a company's short name. Each has code's score — how sure it is the two are one. Say which are one, a pair at a time or all at one score and above; the second token becomes the first one's spelling in every file, without a new run. Only you see the real names." />
      {error && <Notice tone="amber">{error}</Notice>}
      {notice && <Notice tone="emerald">{notice}</Notice>}
      {disabled && <Notice tone="amber">The room is running — answer when it's done.</Notice>}

      {pairs === null ? (
        <div className="flex items-center gap-2 px-1 py-6 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> Looking…</div>
      ) : pairs.length === 0 ? (
        <EmptyState icon={Combine} title="Nothing to ask">
          {room.counts.done === 0 ? 'Run the room first: the tokens come from its files.' : 'No two tokens look like one party.'}
        </EmptyState>
      ) : (
        <div className="overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900/30">
          <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800 px-3 py-2.5">
            <Segmented label="Show" value={kind} onChange={setKind} options={[
              { value: 'all', label: 'All', count: pairs.length },
              ...(['PERSON', 'ORG', 'ADDRESS'] as const).filter((k) => kinds(k) > 0)
                .map((k) => ({ value: k as Kind, label: KIND_LABEL[k], count: kinds(k) })),
            ]} />
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2.5 border-b border-zinc-800 px-3 py-2.5">
            <Segmented label="Code's score" value={side} onChange={setSide} options={[
              { value: 'above', label: `${score}% or more`, count: above.length },
              { value: 'below', label: `Under ${score}%`, count: below.length },
            ]} />
            <label className="flex min-w-[11rem] flex-1 items-center gap-2 text-xs text-zinc-400">
              Score
              <input type="range" min={0} max={99} step={1} value={score} onChange={(e) => setScore(Number(e.target.value))}
                className="min-w-0 flex-1 accent-violet-500" aria-label="Code's score" aria-valuetext={`${score}%`} />
              <span className="w-9 shrink-0 text-right tabular-nums text-zinc-300">{score}%</span>
            </label>
            {confirming ? (
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs text-zinc-300">
                  {side === 'above'
                    ? `Make ${count(shown.length, 'pair')} one? Each second token becomes the first one's spelling — it can't be undone here.`
                    : `${count(shown.length, 'pair')} not the same? They won't be asked about again.`}
                </span>
                <Button variant={side === 'above' ? 'primary' : 'secondary'} icon={side === 'above' ? CheckCheck : X}
                  onClick={() => void answerAll()}>{side === 'above' ? 'Yes, make them one' : 'Yes, not the same'}</Button>
                <Button variant="ghost" onClick={() => setConfirming(false)}>Cancel</Button>
              </div>
            ) : (
              <Button variant={side === 'above' ? 'primary' : 'secondary'} icon={side === 'above' ? CheckCheck : X}
                busy={busy === 'all'} disabled={disabled || !!busy || shown.length === 0} onClick={() => setConfirming(true)}>
                {side === 'above' ? `Same for all ${shown.length}` : `Not the same for all ${shown.length}`}
              </Button>
            )}
          </div>
          {shown.length === 0 ? (
            <p className="px-4 py-6 text-sm text-zinc-500">
              {side === 'above' ? `No pair scores ${score}% or more.` : `No pair scores under ${score}%.`} Move the slider to see others.
            </p>
          ) : (
            <ul className="divide-y divide-zinc-800/60">
              {shown.map((p) => <PairRow key={p.key} p={p} swapped={swapped.has(p.key)} busy={busy} disabled={disabled}
                onSwap={() => setSwapped((cur) => {
                  const next = new Set(cur)
                  if (!next.delete(p.key)) next.add(p.key)
                  return next
                })}
                onSame={(keep, other) => same(keep, other, `same:${p.key}`)} onNotSame={() => notSame(p)} />)}
            </ul>
          )}
        </div>
      )}

      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-4">
        <div className="text-sm font-medium text-zinc-100">Two tokens you know are one</div>
        <p className="mt-0.5 text-xs leading-relaxed text-zinc-500">
          Not in the list? Give both tokens as they are written (⟦PERSON_3⟧, without a :2). The second becomes the first one's spelling.
        </p>
        <div className="mt-3 flex flex-wrap items-end gap-2">
          <Field label="Keep"><input className={`${inputCls} w-44 font-mono text-xs`} placeholder="⟦PERSON_3⟧" value={keepTok}
            onChange={(e) => setKeepTok(e.target.value.trim())} /></Field>
          <Field label="Is the same as"><input className={`${inputCls} w-44 font-mono text-xs`} placeholder="⟦PERSON_9⟧" value={otherTok}
            onChange={(e) => setOtherTok(e.target.value.trim())} /></Field>
          <Button icon={Combine} busy={busy === 'manual'} disabled={disabled || !keepTok || !otherTok}
            onClick={() => same(keepTok, otherTok, 'manual')}>Make one</Button>
        </div>
      </div>

      {merged.length > 0 && (
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-4">
          <div className="text-sm font-medium text-zinc-100">Made one</div>
          <ul className="mt-2 flex flex-wrap gap-2">
            {merged.map((m) => (
              <li key={m.token} className="inline-flex items-center gap-1.5 rounded-full border border-zinc-700 bg-zinc-950 px-2.5 py-1 text-xs">
                <TokenChip token={m.token} /><span className="text-zinc-300">{m.text}</span>
                <span className="text-zinc-500">→</span><TokenChip token={m.into} />
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function scoreTone(score: number): string {
  if (score >= 90) return 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
  if (score >= 70) return 'border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300'
  return 'border-zinc-700 bg-zinc-800/60 text-zinc-400'
}

function PairRow({ p, swapped, busy, disabled, onSwap, onSame, onNotSame }: {
  p: AlikePair
  swapped: boolean
  busy: string
  disabled: boolean
  onSwap: () => void
  onSame: (keep: string, other: string) => void
  onNotSame: () => void
}) {
  // The one in more files is kept by default; the arrows swap it.
  const keep = swapped ? p.b : p.a
  const other = swapped ? p.a : p.b
  const side = (s: typeof p.a, note: string) => (
    <div className="min-w-0">
      <div className="flex min-w-0 items-start gap-2">
        <TokenChip token={s.token} />
        <span className="min-w-0 break-words text-sm leading-5 text-zinc-100">{s.text}</span>
      </div>
      <div className="mt-0.5 text-[11px] text-zinc-500">
        <span className="tabular-nums">{count(s.files, 'file')}</span> · {note}
      </div>
    </div>
  )
  const why = [p.why, ...(p.notes || []).filter((n) => n !== p.why)].join(' · ')
  return (
    <li className="grid items-center gap-x-3 gap-y-2 px-4 py-3 @3xl:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)_auto]">
      {side(keep, 'keep this one')}
      <button type="button" onClick={onSwap} title="Keep the other one" aria-label="Keep the other one"
        className="justify-self-start rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
        <ArrowLeftRight className="h-4 w-4" />
      </button>
      {side(other, why)}
      <div className="flex shrink-0 items-center gap-2">
        <span title={`Code's score: how sure it is these are one — ${why}`}
          className={`inline-flex w-12 justify-center rounded-lg border px-1.5 py-1 text-xs font-semibold tabular-nums ${scoreTone(p.score)}`}>
          {p.score}%
        </span>
        <Button variant="primary" icon={Check} busy={busy === `same:${p.key}`} disabled={disabled || !!busy}
          onClick={() => onSame(keep.token, other.token)}>Same</Button>
        <Button icon={X} busy={busy === `no:${p.key}`} disabled={disabled || !!busy} onClick={onNotSame}>Not the same</Button>
      </div>
    </li>
  )
}
