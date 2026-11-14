import { useCallback, useEffect, useState } from 'react'
import { ArrowLeftRight, Check, Combine, Loader2, X } from 'lucide-react'
import { roomsApi, type AlikeMerged, type AlikePair, type Room } from '../../services/roomsApi'
import { Button, EmptyState, Field, Notice, Segmented, inputCls } from '../console/parts'
import { SectionHead } from './RoomsInbox'
import { TokenChip } from './roomParts'
import { count, errorText } from './roomText'

// "Same party?": tokens code couldn't prove are one person, company or
// address — a one-letter typo, a first name alone, a surname that changed
// between files, a company's short name — for the room's owner to say. "Same"
// makes the second token the first one's spelling in the vault and in every
// copy in the room; "Not the same" is remembered.

type Kind = 'all' | 'PERSON' | 'ORG' | 'ADDRESS'

const KIND_LABEL: Record<AlikePair['kind'], string> = { PERSON: 'People', ORG: 'Companies', ADDRESS: 'Addresses' }

export function AlikeSection({ room, disabled, onChanged }: {
  room: Room
  disabled: boolean
  onChanged: (count: number) => void
}) {
  const [pairs, setPairs] = useState<AlikePair[] | null>(null)
  const [merged, setMerged] = useState<AlikeMerged[]>([])
  const [kind, setKind] = useState<Kind>('all')
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [keepTok, setKeepTok] = useState('')
  const [otherTok, setOtherTok] = useState('')

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

  const notSame = async (p: AlikePair) => {
    setBusy(`no:${p.key}`); setError(''); setNotice('')
    try {
      await roomsApi.notSame(room.id, p.a.token, p.b.token)
      setPairs((cur) => {
        const next = (cur || []).filter((x) => x.key !== p.key)
        onChanged(next.length)
        return next
      })
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy('')
    }
  }

  const shown = (pairs || []).filter((p) => kind === 'all' || p.kind === kind)
  const kinds = (k: AlikePair['kind']) => (pairs || []).filter((p) => p.kind === k).length

  return (
    <div className="space-y-4">
      <SectionHead icon={Combine} title="Same party?"
        hint="Code gives one token to what it can prove is one person, company or address. These it couldn't prove: a one-letter typo, a first name alone, a surname that changed between files, a company's short name. Say which are one — the second token becomes the first one's spelling in every file, without a new run. Only you see the real names." />
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
          <ul className="divide-y divide-zinc-800/60">
            {shown.map((p) => <PairRow key={p.key} p={p} busy={busy} disabled={disabled}
              onSame={(keep, other) => same(keep, other, `same:${p.key}`)} onNotSame={() => notSame(p)} />)}
          </ul>
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

function PairRow({ p, busy, disabled, onSame, onNotSame }: {
  p: AlikePair
  busy: string
  disabled: boolean
  onSame: (keep: string, other: string) => void
  onNotSame: () => void
}) {
  // The one in more files is kept by default; the arrows swap it.
  const [swap, setSwap] = useState(false)
  const keep = swap ? p.b : p.a
  const other = swap ? p.a : p.b
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
  return (
    <li className="grid items-center gap-x-3 gap-y-2 px-4 py-3 @3xl:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)_auto]">
      {side(keep, 'keep this one')}
      <button type="button" onClick={() => setSwap(!swap)} title="Keep the other one" aria-label="Keep the other one"
        className="justify-self-start rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
        <ArrowLeftRight className="h-4 w-4" />
      </button>
      {side(other, p.why)}
      <div className="flex shrink-0 items-center gap-2">
        <Button variant="primary" icon={Check} busy={busy === `same:${p.key}`} disabled={disabled || !!busy}
          onClick={() => onSame(keep.token, other.token)}>Same</Button>
        <Button icon={X} busy={busy === `no:${p.key}`} disabled={disabled || !!busy} onClick={onNotSame}>Not the same</Button>
      </div>
    </li>
  )
}
