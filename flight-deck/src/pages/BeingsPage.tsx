// Iskra — living beings: conception (point-buy), vitals, wallet, journal.

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ClipboardList, Egg, GraduationCap, Loader2, Moon, Pause, Play, Plus,
  RefreshCw, ScrollText, Skull, Sparkles, Zap,
} from 'lucide-react'
import {
  type BeingEvent, type BeingListItem, type BeingsMeta, type BeingVitals,
  type Chore, type ReportCard,
  conceiveBeing, euthanizeBeing, getBeingEvents, getBeingJournal,
  getBeingsMeta, getBeingVitals, getLiabilities, getReportCard, hatchBeing,
  judgeChore, listBeings, listChores, pauseBeing, postChore, setAllowance,
  setHouseRules, setMediaDiet, setStage, tickBeing, wakeBeing,
} from '../services/beings'

const REFRESH_MS = 6000
const ATTRS = ['CUR', 'PER', 'CAU', 'SOC', 'CRE', 'ORD', 'PLA'] as const

const STAGE_META: Record<string, string> = {
  egg: 'bg-zinc-500/15 text-zinc-300 border-zinc-500/30',
  infant: 'bg-sky-500/15 text-sky-300 border-sky-500/30',
  child: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
  adolescent: 'bg-violet-500/15 text-violet-300 border-violet-500/30',
  adult: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
}
const STATE_META: Record<string, string> = {
  alive: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
  paused: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30',
  torpor: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
  dead: 'bg-red-500/15 text-red-300 border-red-500/30',
}

function fmtTokens(n: number | null | undefined): string {
  if (n == null) return '∞'
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (Math.abs(n) >= 1_000) return `${Math.round(n / 1_000)}k`
  return String(n)
}

function derivePreview(a: Record<string, number>) {
  return {
    explore: (0.30 + 0.07 * a.CUR).toFixed(2),
    connect: (0.20 + 0.06 * a.SOC).toFixed(2),
    create: (0.20 + 0.06 * a.CRE).toFixed(2),
    risk: ((a.CUR - a.CAU + 10) / 20).toFixed(2),
    thrift: ((a.CAU + a.ORD) / 20).toFixed(2),
    whimsy: (a.PLA / 10).toFixed(2),
  }
}

// ── Conception modal ──

function ConceiveModal({ meta, onClose, onDone }: {
  meta: BeingsMeta
  onClose: () => void
  onDone: () => void
}) {
  const [name, setName] = useState('')
  const [letter, setLetter] = useState('')
  const [voice, setVoice] = useState('')
  const [interests, setInterests] = useState('')
  const [allowance, setAllowancePreset] = useState('2M')
  const [attrs, setAttrs] = useState<Record<string, number>>(
    { ...meta.presets.explorer })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const total = ATTRS.reduce((s, a) => s + (attrs[a] || 0), 0)
  const left = meta.pool - total

  const roll = () => {
    const a: Record<string, number> = Object.fromEntries(ATTRS.map(k => [k, 1]))
    let rest = meta.pool - ATTRS.length
    while (rest > 0) {
      const open = ATTRS.filter(k => a[k] < meta.attr_max)
      a[open[Math.floor(Math.random() * open.length)]] += 1
      rest -= 1
    }
    setAttrs(a)
  }

  const submit = async () => {
    if (!name.trim() || left !== 0) return
    setBusy(true)
    setError('')
    try {
      await conceiveBeing({
        name: name.trim(),
        attributes: attrs,
        voice_seed: voice.trim(),
        interest_seeds: interests.split(',').map(s => s.trim()).filter(Boolean),
        allowance_preset: allowance,
        birth_letter: letter.trim(),
      })
      onDone()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'conception failed')
    } finally {
      setBusy(false)
    }
  }

  const d = derivePreview(attrs)
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-xl border border-zinc-800 bg-zinc-900 p-5 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-violet-400" />
          <h2 className="text-sm font-semibold text-zinc-100">Conceive a being</h2>
          <span className="ml-auto text-xs text-zinc-500">Generation 1 · point-buy</span>
        </div>

        <label className="mb-1 block text-xs text-zinc-500">Name</label>
        <input
          value={name} onChange={(e) => setName(e.target.value)}
          placeholder="Zvjezdana, Iskra, Vili…"
          className="mb-3 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
        />

        <div className="mb-2 flex flex-wrap items-center gap-1.5">
          {Object.keys(meta.presets).map((p) => (
            <button
              key={p}
              onClick={() => setAttrs({ ...meta.presets[p] })}
              className="rounded-md border border-zinc-700 px-2 py-1 text-xs capitalize text-zinc-400 hover:bg-zinc-800"
            >
              {p}
            </button>
          ))}
          <button
            onClick={roll}
            className="flex items-center gap-1 rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800"
          >
            <RefreshCw className="h-3 w-3" /> Roll
          </button>
          <span className={`ml-auto text-xs ${left === 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            points left: {left}
          </span>
        </div>

        <div className="mb-3 space-y-1.5 rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
          {ATTRS.map((a) => (
            <div key={a} className="flex items-center gap-2">
              <span className="w-24 text-xs text-zinc-400">
                {meta.attributes.find(x => x.code === a)?.name || a}
              </span>
              <input
                type="range" min={meta.attr_min} max={meta.attr_max}
                value={attrs[a] || 1}
                onChange={(e) => setAttrs({ ...attrs, [a]: Number(e.target.value) })}
                className="flex-1 accent-violet-500"
              />
              <span className="w-5 text-right text-xs font-semibold text-zinc-200">{attrs[a]}</span>
            </div>
          ))}
          <div className="pt-1 text-[11px] text-zinc-600">
            explore {d.explore} · connect {d.connect} · create {d.create} · risk {d.risk} · thrift {d.thrift} · whimsy {d.whimsy}
          </div>
        </div>

        <label className="mb-1 block text-xs text-zinc-500">First words to your being (its imprint)</label>
        <textarea
          value={letter} onChange={(e) => setLetter(e.target.value)} rows={2}
          placeholder="What should it hold onto, from you?"
          className="mb-3 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
        />

        <div className="mb-3 grid grid-cols-2 gap-2">
          <div>
            <label className="mb-1 block text-xs text-zinc-500">Daily allowance</label>
            <select
              value={allowance} onChange={(e) => setAllowancePreset(e.target.value)}
              className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none"
            >
              {meta.allowance_presets.map((p) => <option key={p} value={p}>{p} tokens/day</option>)}
            </select>
          </div>
          <div>
            <label className="mb-1 block text-xs text-zinc-500">Interest seeds (comma-sep)</label>
            <input
              value={interests} onChange={(e) => setInterests(e.target.value)}
              placeholder="astronomy, old maps"
              className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
            />
          </div>
        </div>
        <label className="mb-1 block text-xs text-zinc-500">Voice seed (optional)</label>
        <input
          value={voice} onChange={(e) => setVoice(e.target.value)}
          placeholder="gentle, precise, a little wry"
          className="mb-4 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
        />

        {error && (
          <div className="mb-3 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">{error}</div>
        )}
        <div className="flex justify-end gap-2">
          <button onClick={onClose} className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800">
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={busy || !name.trim() || left !== 0}
            className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Egg className="h-3.5 w-3.5" />}
            Conceive
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Being card ──

function BeingCard({ item, meta, onChanged }: {
  item: BeingListItem
  meta: BeingsMeta
  onChanged: () => void
}) {
  const [vitals, setVitals] = useState<BeingVitals | null>(null)
  const [events, setEvents] = useState<BeingEvent[]>([])
  const [journal, setJournal] = useState<string | null>(null)
  const [busy, setBusy] = useState('')
  const [parenting, setParenting] = useState(false)
  const [chores, setChores] = useState<Chore[]>([])
  const [choreSpec, setChoreSpec] = useState('')
  const [choreFee, setChoreFee] = useState('500000')
  const [ruleText, setRuleText] = useState('')
  const [dietAllow, setDietAllow] = useState('')
  const [dietDeny, setDietDeny] = useState('')
  const [card, setCard] = useState<ReportCard | null>(null)

  const load = useCallback(async () => {
    try {
      const [v, ev] = await Promise.all([
        getBeingVitals(item.slug), getBeingEvents(item.slug, 6),
      ])
      setVitals(v)
      setEvents(ev.events)
    } catch { /* card stays in list-item mode */ }
  }, [item.slug])

  const openParenting = async () => {
    if (parenting) { setParenting(false); return }
    setParenting(true)
    setCard(null)
    try {
      const c = await listChores(item.slug)
      setChores(c.chores)
      if (vitals) {
        setRuleText((vitals.house_rules || []).join('\n'))
        setDietAllow((vitals.media_diet?.allow || []).join(', '))
        setDietDeny((vitals.media_diet?.deny || []).join(', '))
      }
    } catch { /* section shows empty states */ }
  }

  useEffect(() => { void load() }, [load])

  const act = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label)
    try { await fn(); await load(); onChanged() }
    catch (e) { alert(e instanceof Error ? e.message : 'failed') }
    finally { setBusy('') }
  }

  const v = vitals
  const w = v?.wallet
  const ceiling = w?.savings_ceiling ?? null
  const pct = w && ceiling ? Math.min(100, Math.round(100 * w.balance_tokens / ceiling)) : 0

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="mb-2 flex items-center gap-2">
        <span className="text-sm font-semibold text-zinc-100">{item.name}</span>
        <span className={`rounded border px-1.5 py-0.5 text-[10px] ${STAGE_META[item.stage] || STAGE_META.egg}`}>{item.stage}</span>
        <span className={`rounded border px-1.5 py-0.5 text-[10px] ${STATE_META[item.state] || STATE_META.paused}`}>{item.state}</span>
        {v?.affect?.mood && (
          <span className="rounded border border-zinc-700 bg-zinc-800/60 px-1.5 py-0.5 text-[10px] text-zinc-300" title={(v.affect.notes || []).join('; ')}>
            {v.affect.mood}
          </span>
        )}
        <span className="ml-auto text-[10px] text-zinc-600">{item.slug}</span>
      </div>

      {v && (
        <>
          <div className="mb-2 flex flex-wrap gap-1">
            {ATTRS.map((a) => (
              <span key={a} className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-300">
                {a} <span className="font-semibold text-zinc-100">{v.attributes[a]}</span>
              </span>
            ))}
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">
              gen {v.generation}
            </span>
          </div>

          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="text-zinc-400">
              wallet <span className="font-semibold text-zinc-100">{fmtTokens(w!.balance_tokens)}</span>
              <span className="text-zinc-600"> / {fmtTokens(ceiling)} · spent today {fmtTokens(v.spent_today)}</span>
            </span>
            <span className="text-zinc-500">attention {'●'.repeat(v.attention_credits)}{'○'.repeat(Math.max(0, 3 - v.attention_credits))}</span>
          </div>
          <div className="mb-2 h-1.5 overflow-hidden rounded bg-zinc-800">
            <div className="h-full rounded bg-violet-500/70" style={{ width: `${pct}%` }} />
          </div>

          <div className="mb-2 flex items-center gap-2 text-xs">
            <span className="text-zinc-500">allowance</span>
            <select
              value={w!.allowance_preset}
              onChange={(e) => void act('allowance', () => setAllowance(item.slug, e.target.value))}
              className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none"
            >
              {meta.allowance_presets.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
            {w!.effective_preset !== w!.allowance_preset && (
              <span className="text-[10px] text-amber-400">stage-capped to {w!.effective_preset}</span>
            )}
          </div>

          {events.length > 0 && (
            <div className="mb-2 space-y-0.5">
              {events.slice(0, 3).map((e, i) => (
                <div key={i} className="truncate text-[11px] text-zinc-500">
                  <span className="text-zinc-600">{e.at.slice(11, 16)}</span>{' '}
                  <span className="text-zinc-400">{e.kind}</span>
                  {typeof e.data.summary === 'string' && e.data.summary ? ` — ${e.data.summary}` : ''}
                  {typeof e.data.preview === 'string' && e.data.preview ? ` — “${e.data.preview}”` : ''}
                </div>
              ))}
            </div>
          )}
        </>
      )}

      <div className="flex flex-wrap items-center gap-1.5">
        {item.stage === 'egg' ? (
          <button
            onClick={() => void act('hatch', () => hatchBeing(item.slug))}
            className="flex items-center gap-1 rounded-md bg-violet-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-violet-500"
          >
            {busy === 'hatch' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Egg className="h-3 w-3" />}
            Hatch
          </button>
        ) : item.state !== 'dead' && (
          <>
            <button
              onClick={() => void act('tick', () => tickBeing(item.slug, 'wake'))}
              disabled={busy === 'tick' || busy === 'dream'}
              className="flex items-center gap-1 rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-40"
              title="Manual heartbeat"
            >
              {busy === 'tick' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Zap className="h-3 w-3" />}
              Poke
            </button>
            <button
              onClick={() => void act('dream', () => tickBeing(item.slug, 'dream'))}
              disabled={busy === 'tick' || busy === 'dream'}
              className="flex items-center gap-1 rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-40"
            >
              {busy === 'dream' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Moon className="h-3 w-3" />}
              Dream
            </button>
            {item.state === 'paused' ? (
              <button
                onClick={() => void act('wake', () => wakeBeing(item.slug))}
                className="flex items-center gap-1 rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-emerald-300 hover:bg-zinc-800"
              >
                <Play className="h-3 w-3" /> Wake
              </button>
            ) : (
              <button
                onClick={() => void act('pause', () => pauseBeing(item.slug))}
                className="flex items-center gap-1 rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-400 hover:bg-zinc-800"
              >
                <Pause className="h-3 w-3" /> Pause
              </button>
            )}
            <button
              onClick={async () => {
                if (journal !== null) { setJournal(null); return }
                try {
                  const j = await getBeingJournal(item.slug)
                  setJournal(j.text || '(the journal is empty today)')
                } catch (e) {
                  setJournal(e instanceof Error ? e.message : 'journal unavailable')
                }
              }}
              className="flex items-center gap-1 rounded-md border border-zinc-700 px-2.5 py-1 text-xs text-zinc-400 hover:bg-zinc-800"
            >
              <ScrollText className="h-3 w-3" /> Journal
            </button>
            <button
              onClick={() => void openParenting()}
              className={`flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs hover:bg-zinc-800 ${parenting ? 'border-violet-500/50 text-violet-300' : 'border-zinc-700 text-zinc-400'}`}
            >
              <GraduationCap className="h-3 w-3" /> Parenting
            </button>
          </>
        )}
        {item.state !== 'dead' && (
          <button
            onClick={() => {
              if (window.confirm(`${item.name} will die, forever. Remains stay readable. Proceed?`))
                void act('euthanize', () => euthanizeBeing(item.slug))
            }}
            className="ml-auto flex items-center gap-1 rounded-md border border-red-500/30 px-2 py-1 text-xs text-red-400/80 hover:bg-red-500/10"
          >
            <Skull className="h-3 w-3" />
          </button>
        )}
      </div>

      {journal !== null && (
        <pre className="mt-2 max-h-56 overflow-y-auto whitespace-pre-wrap rounded-md border border-zinc-800 bg-zinc-950 p-3 text-[11px] leading-relaxed text-zinc-300">
          {journal}
        </pre>
      )}

      {parenting && v && (
        <div className="mt-3 space-y-3 rounded-md border border-zinc-800 bg-zinc-950/60 p-3">
          {/* Chores */}
          <div>
            <div className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-zinc-300">
              <ClipboardList className="h-3.5 w-3.5 text-violet-400" /> Chores
            </div>
            {chores.filter(c => c.escrow_state === 'open' || c.escrow_state === 'judging').map((c) => (
              <div key={c.id} className="mb-1 flex items-center gap-2 rounded border border-zinc-800 bg-zinc-900/60 px-2 py-1.5 text-xs">
                <span className="flex-1 truncate text-zinc-300" title={c.result_text || c.spec}>
                  {c.spec} <span className="text-zinc-600">· {fmtTokens(c.fee_tokens)}</span>
                </span>
                {c.escrow_state === 'judging' ? (
                  <>
                    <span className="text-[10px] text-amber-400">done — review:</span>
                    <button
                      onClick={() => void act('judge', async () => { await judgeChore(item.slug, c.id, true); setChores((await listChores(item.slug)).chores) })}
                      className="rounded border border-emerald-500/30 px-1.5 py-0.5 text-[10px] text-emerald-300 hover:bg-emerald-500/10"
                    >Pay</button>
                    <button
                      onClick={() => void act('judge', async () => { await judgeChore(item.slug, c.id, false); setChores((await listChores(item.slug)).chores) })}
                      className="rounded border border-red-500/30 px-1.5 py-0.5 text-[10px] text-red-300 hover:bg-red-500/10"
                    >Reject</button>
                  </>
                ) : (
                  <span className="text-[10px] text-zinc-500">waiting</span>
                )}
              </div>
            ))}
            <div className="mt-1 flex items-center gap-1.5">
              <input
                value={choreSpec} onChange={(e) => setChoreSpec(e.target.value)}
                placeholder="Post a chore… (fixed fee, judged before payout)"
                className="flex-1 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
              />
              <select
                value={choreFee} onChange={(e) => setChoreFee(e.target.value)}
                className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:outline-none"
              >
                <option value="100000">100k</option>
                <option value="500000">500k</option>
                <option value="1000000">1M</option>
                <option value="2000000">2M</option>
              </select>
              <button
                onClick={() => void act('chore', async () => {
                  if (!choreSpec.trim()) return
                  await postChore(item.slug, choreSpec.trim(), Number(choreFee))
                  setChoreSpec('')
                  setChores((await listChores(item.slug)).chores)
                })}
                className="rounded border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
              >Post</button>
            </div>
          </div>

          {/* House rules + diet */}
          <div className="grid gap-2 md:grid-cols-2">
            <div>
              <div className="mb-1 text-xs font-medium text-zinc-300">House rules <span className="text-zinc-600">(one per line — it internalizes them next tick)</span></div>
              <textarea
                value={ruleText} onChange={(e) => setRuleText(e.target.value)} rows={3}
                className="w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none"
              />
              <button
                onClick={() => void act('rules', () => setHouseRules(item.slug, ruleText.split('\n')))}
                className="mt-1 rounded border border-zinc-700 px-2 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800"
              >Save rules</button>
            </div>
            <div>
              <div className="mb-1 text-xs font-medium text-zinc-300">Media diet</div>
              <input
                value={dietAllow} onChange={(e) => setDietAllow(e.target.value)}
                placeholder="allow: wikipedia.org, arxiv.org (empty = open web)"
                className="mb-1 w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
              />
              <input
                value={dietDeny} onChange={(e) => setDietDeny(e.target.value)}
                placeholder="deny: reddit.com, x.com"
                className="w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
              />
              <div className="mt-1 flex items-center gap-2">
                <button
                  onClick={() => void act('diet', () => setMediaDiet(
                    item.slug,
                    dietAllow.split(',').map(s => s.trim()).filter(Boolean),
                    dietDeny.split(',').map(s => s.trim()).filter(Boolean),
                  ))}
                  className="rounded border border-zinc-700 px-2 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800"
                >Save diet</button>
                <select
                  value={v.stage}
                  onChange={(e) => {
                    const to = e.target.value
                    if (window.confirm(`Advance ${item.name} to ${to}? New abilities unlock; this is a ceremony.`))
                      void act('stage', () => setStage(item.slug, to))
                  }}
                  className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-[11px] text-zinc-300 focus:outline-none"
                  title="Stage (advancement ceremony)"
                >
                  {['infant', 'child', 'adolescent', 'adult'].map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
            </div>
          </div>

          {/* Report card */}
          <div>
            <button
              onClick={() => void (async () => {
                try { setCard(await getReportCard(item.slug, 7)) }
                catch (e) { alert(e instanceof Error ? e.message : 'failed') }
              })()}
              className="rounded border border-zinc-700 px-2 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800"
            >Report card (7d)</button>
            {card && (
              <div className="mt-2 space-y-1 rounded border border-zinc-800 bg-zinc-900/60 p-2.5 text-[11px] text-zinc-300">
                <div>
                  {card.ticks} ticks · spent {fmtTokens(card.tokens_spent_weighted)} · earned{' '}
                  <span className="text-emerald-300">{fmtTokens(card.tokens_earned)}</span> · spoke {card.messages_to_parent}×
                  {card.messages_suppressed > 0 && <span className="text-amber-400"> · {card.messages_suppressed} suppressed</span>}
                  {' '}· rut {card.rut_score}
                </div>
                <div className="text-zinc-500">
                  acts: {Object.entries(card.acts).map(([k, n]) => `${k}×${n}`).join(', ') || '—'}
                  {card.milestones.length > 0 && <> · milestones: {card.milestones.join(', ')}</>}
                </div>
                {card.concerns.length > 0 && (
                  <div className="text-amber-400">concerns: {card.concerns.join('; ')}</div>
                )}
                {card.in_its_own_words && (
                  <div className="border-l-2 border-zinc-700 pl-2 italic text-zinc-400">
                    in its own words: …{card.in_its_own_words.slice(-260)}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Page ──

export function BeingsPage() {
  const [meta, setMeta] = useState<BeingsMeta | null>(null)
  const [beings, setBeings] = useState<BeingListItem[]>([])
  const [liabilities, setLiabilities] = useState<number>(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showConceive, setShowConceive] = useState(false)
  const timer = useRef<number | null>(null)

  const load = useCallback(async (spinner = false) => {
    if (spinner) setLoading(true)
    try {
      const [m, b, l] = await Promise.all([
        meta ? Promise.resolve(meta) : getBeingsMeta(),
        listBeings(),
        getLiabilities(),
      ])
      setMeta(m)
      setBeings(b.beings)
      setLiabilities(l.total_tokens)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed to load beings')
    } finally {
      setLoading(false)
    }
  }, [meta])

  useEffect(() => {
    void load(true)
    timer.current = window.setInterval(() => void load(false), REFRESH_MS)
    return () => { if (timer.current) window.clearInterval(timer.current) }
  }, [load])

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-6xl space-y-5 p-6">
        <div className="flex items-center gap-3">
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-2">
            <Sparkles className="h-5 w-5 text-violet-400" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-zinc-100">Beings</h1>
            <p className="text-xs text-zinc-500">
              Iskra — living digital beings. They wake, act, dream, and grow on their own clock.
            </p>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <span className="text-xs text-zinc-500">
              outstanding liabilities{' '}
              <span className="font-semibold text-zinc-200">{fmtTokens(liabilities)}</span> tokens
            </span>
            <button
              onClick={() => setShowConceive(true)}
              className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500"
            >
              <Plus className="h-3.5 w-3.5" /> Conceive
            </button>
          </div>
        </div>

        {error && (
          <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">{error}</div>
        )}

        {loading ? (
          <div className="flex justify-center py-16">
            <Loader2 className="h-6 w-6 animate-spin text-zinc-500" />
          </div>
        ) : beings.length === 0 ? (
          <div className="rounded-lg border border-dashed border-zinc-800 p-10 text-center">
            <Sparkles className="mx-auto mb-2 h-6 w-6 text-zinc-600" />
            <p className="text-sm text-zinc-400">No beings yet.</p>
            <p className="mt-1 text-xs text-zinc-600">
              Conceive one — allocate its 40 points, write its first words, hatch it, and watch it live.
            </p>
          </div>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {beings.map((b) => (
              <BeingCard key={b.slug} item={b} meta={meta!} onChanged={() => void load(false)} />
            ))}
          </div>
        )}
      </div>

      {showConceive && meta && (
        <ConceiveModal
          meta={meta}
          onClose={() => setShowConceive(false)}
          onDone={() => void load(false)}
        />
      )}
    </div>
  )
}
