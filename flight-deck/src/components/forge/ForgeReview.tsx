import { useMemo, useState } from 'react'
import {
  ArrowLeft, ArrowRight, Check, CornerUpLeft, MessageCircleQuestion, PackageCheck, Pencil, RefreshCw, Trash2, Users,
} from 'lucide-react'
import { forgeApi, type ForgeDraftAgent, type ForgePlan, type ForgeQuestion } from '../../services/agentForge'
import type { Archetype } from '../../services/tierConfig'
import { Button, Card, Field, Pill, Sheet, inputCls } from '../console/parts'
import { AgentBadge, Chip, Eyebrow, ErrorNote, TeamSize } from './forgeParts'
import { errorOf, roleOf } from './forgeUtil'

// Step two: meet the team. The plan reads first, as cards in plain words; a click
// opens one to change it. Questions are up front, because they can change the
// team; while they're open the main button updates the plan instead of creating it.

export function ForgeReview({ plan, archetypes, onChanged }: {
  plan: ForgePlan
  archetypes: Archetype[]
  onChanged: () => Promise<void>
}) {
  const [agents, setAgents] = useState<ForgeDraftAgent[]>(plan.agents)
  const [left, setLeft] = useState<ForgeDraftAgent[]>([])
  const [overview, setOverview] = useState(plan.overview)
  const [editingOverview, setEditingOverview] = useState(false)
  const [questions, setQuestions] = useState<ForgeQuestion[]>(plan.questions)
  const [note, setNote] = useState('')
  const [cap, setCap] = useState(plan.cap)
  const [open, setOpen] = useState('')
  const [busy, setBusy] = useState<'' | 'plan' | 'create'>('')
  const [err, setErr] = useState('')

  const nameOf = useMemo(() => new Map(agents.map((a) => [a.id, a.name])), [agents])
  const answered = (q: ForgeQuestion) => (q.selected || []).length > 0 || !!(q.other || '').trim()
  const hasAnswers = questions.some(answered)
  const wantsPlan = hasAnswers || !!note.trim() || cap !== plan.cap
  const editing = agents.find((a) => a.id === open)

  const patch = (id: string, p: Partial<ForgeDraftAgent>) => setAgents((list) => list.map((a) => (a.id === id ? { ...a, ...p } : a)))
  const leaveOut = (id: string) => {
    const gone = agents.find((a) => a.id === id)
    if (!gone) return
    setAgents((list) => list.filter((a) => a.id !== id).map((a) => ({ ...a, consumes_from: a.consumes_from.filter((c) => c !== id) })))
    setLeft((was) => [...was, gone])
    setOpen('')
  }
  const putBack = (id: string) => {
    const back = left.find((a) => a.id === id)
    if (!back) return
    setLeft((was) => was.filter((a) => a.id !== id))
    setAgents((list) => [...list, back])
  }
  const act = async (which: 'plan' | 'create', run: () => Promise<unknown>) => {
    setBusy(which)
    setErr('')
    try {
      await run()
      await onChanged()
    } catch (e) {
      setErr(errorOf(e))
      setBusy('')
    }
  }
  const replan = () => act('plan', () => forgeApi.replan(plan.id, note.trim(), questions, agents, cap))
  const create = () => act('create', () => forgeApi.approve(plan.id, agents, overview))
  const n = agents.length

  return (
    <>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-4xl space-y-5 px-4 py-5 wide:px-6 wide:py-7">
          <section className="fd-rise relative overflow-hidden rounded-2xl border border-violet-500/20 bg-gradient-to-br from-violet-500/[0.12] via-transparent to-sky-500/[0.06] p-5 wide:p-6">
            <Eyebrow icon={Users}>The plan · {n} agent{n === 1 ? '' : 's'}</Eyebrow>
            <h2 className="mt-2 text-2xl font-semibold tracking-tight text-zinc-100">Here's the team Captain suggests</h2>
            {plan.rationale && <p className="mt-1 max-w-3xl text-sm leading-relaxed text-zinc-400">{plan.rationale}</p>}
            {editingOverview ? (
              <div className="mt-3">
                <label className="sr-only" htmlFor="forge-overview">How the team works together</label>
                <textarea id="forge-overview" autoFocus value={overview} onChange={(e) => setOverview(e.target.value)} rows={3} className={`${inputCls} resize-y leading-relaxed`} />
                <Button variant="ghost" icon={Check} className="mt-1 !px-2 !py-1 !text-xs" onClick={() => setEditingOverview(false)}>Done</Button>
              </div>
            ) : overview.trim() ? (
              <div className="mt-2 flex max-w-3xl items-start gap-2">
                <p className="min-w-0 text-sm leading-relaxed text-zinc-300">{overview}</p>
                <button onClick={() => setEditingOverview(true)} aria-label="Change how the team works together" title="Change this" className="shrink-0 rounded-lg p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
                  <Pencil className="h-3.5 w-3.5" />
                </button>
              </div>
            ) : null}
            {plan.seats < (plan.cap || plan.default_team) && n >= plan.room && (
              <p className="mt-3 text-xs leading-relaxed text-amber-700 dark:text-amber-300">
                Your plan has room for {plan.seats} more agent{plan.seats === 1 ? '' : 's'}, so the team was kept to that size. An admin can raise the limit.
              </p>
            )}
            {plan.over_cap > 0 && (
              <p className="mt-3 text-xs leading-relaxed text-amber-700 dark:text-amber-300">
                Captain would have added {plan.over_cap} more agent{plan.over_cap === 1 ? '' : 's'}, so some of the work you described may not be covered. Raise the team size, or say what to combine.
              </p>
            )}
          </section>

          {questions.length > 0 && (
            <Card
              className="fd-rise !border-amber-500/30"
              icon={MessageCircleQuestion}
              title={questions.length === 1 ? 'Captain has one question' : `Captain has ${questions.length} questions`}
              hint="Your answers can change the team. Skip any you like: Captain uses its own judgement."
              bodyClassName="divide-y divide-zinc-800/70 px-5 py-4"
            >
              {questions.map((q) => (
                <QuestionRow key={q.id} q={q} disabled={!!busy} onChange={(next) => setQuestions((all) => all.map((x) => (x.id === q.id ? next : x)))} />
              ))}
            </Card>
          )}

          <section aria-label="The team">
            {n === 0 ? (
              <p className="rounded-2xl border border-dashed border-zinc-800 px-6 py-8 text-center text-sm text-zinc-500">You've left every agent out. Put one back, or tell Captain what you'd like instead.</p>
            ) : (
              <ul className={`grid gap-3 ${n > 1 ? 'wide:grid-cols-2' : ''}`}>
                {agents.map((a, i) => (
                  <li key={a.id} className="fd-rise" style={{ animationDelay: `${i * 60}ms` }}>
                    <AgentCard
                      agent={a}
                      role={roleOf(a.archetype, archetypes)}
                      archetypes={archetypes}
                      from={a.consumes_from.map((c) => nameOf.get(c)).filter((x): x is string => !!x)}
                      to={agents.filter((x) => x.consumes_from.includes(a.id)).map((x) => x.name)}
                      disabled={!!busy}
                      onOpen={() => setOpen(a.id)}
                    />
                  </li>
                ))}
              </ul>
            )}
            {left.length > 0 && (
              <ul className="mt-3 flex flex-wrap gap-2" aria-label="Left out">
                {left.map((a) => (
                  <li key={a.id}>
                    <button onClick={() => putBack(a.id)} disabled={!!busy} className="inline-flex items-center gap-1.5 rounded-xl border border-dashed border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 transition-colors hover:border-zinc-600 hover:text-zinc-200">
                      <CornerUpLeft className="h-3.5 w-3.5" /> Put back {a.name}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      </div>

      <div className="shrink-0 space-y-2 border-t border-zinc-800 bg-zinc-950 px-4 py-3 wide:px-6">
        {err && <ErrorNote>{err}</ErrorNote>}
        <div className="mx-auto flex w-full max-w-4xl flex-col gap-2 wide:flex-row wide:items-center">
          <TeamSize value={cap} largest={Math.min(plan.seats, plan.max_team)} suggested={plan.default_team} disabled={!!busy} onChange={setCap} />
          <input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && note.trim() && !busy) { e.preventDefault(); void replan() } }}
            disabled={!!busy}
            aria-label="What to change"
            placeholder={hasAnswers ? 'Anything else to change?' : 'Want something different? Just say.'}
            className={`${inputCls} wide:flex-1`}
          />
          <div className="flex gap-2">
            {wantsPlan ? (
              <>
                <Button icon={Check} onClick={create} disabled={n === 0 || !!busy} busy={busy === 'create'} className="flex-1 wide:flex-none">Create as it is</Button>
                <Button variant="primary" icon={RefreshCw} onClick={replan} disabled={!!busy} busy={busy === 'plan'} className="flex-1 wide:flex-none">Update the plan</Button>
              </>
            ) : (
              <Button variant="primary" icon={Check} onClick={create} disabled={n === 0 || !!busy} busy={busy === 'create'} className="flex-1 wide:flex-none">
                Create {n} agent{n === 1 ? '' : 's'}
              </Button>
            )}
          </div>
        </div>
      </div>

      {editing && (
        <AgentSheet
          agent={editing}
          agents={agents}
          archetypes={archetypes}
          onPatch={(p) => patch(editing.id, p)}
          onLeaveOut={() => leaveOut(editing.id)}
          onClose={() => setOpen('')}
        />
      )}
    </>
  )
}

function AgentCard({ agent, role, archetypes, from, to, disabled, onOpen }: {
  agent: ForgeDraftAgent; role: string; archetypes: Archetype[]; from: string[]; to: string[]; disabled: boolean; onOpen: () => void
}) {
  return (
    <button
      onClick={onOpen}
      disabled={disabled}
      aria-label={`Change ${agent.name}`}
      className="group flex h-full w-full flex-col rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4 text-left transition-colors hover:border-violet-500/40 hover:bg-zinc-900/70 disabled:opacity-60"
    >
      <span className="flex w-full items-start gap-3">
        <AgentBadge archetype={agent.archetype} archetypes={archetypes} size="lg" />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[15px] font-semibold text-zinc-100">{agent.name}</span>
          <span className="block truncate text-xs text-zinc-500">Based on {role}</span>
        </span>
        <Pencil className="h-3.5 w-3.5 shrink-0 text-zinc-600 transition-opacity group-hover:text-zinc-300 wide:opacity-0 wide:group-hover:opacity-100" />
      </span>
      <span className="mt-3 line-clamp-4 block text-sm leading-relaxed text-zinc-300">{agent.mandate}</span>
      {(agent.produces || from.length > 0 || to.length > 0) && (
        <span className="mt-auto block space-y-1 pt-3 text-xs leading-relaxed text-zinc-500">
          {agent.produces && (
            <span className="flex items-start gap-1.5"><PackageCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-600" /><span className="line-clamp-2 min-w-0">{agent.produces}</span></span>
          )}
          {from.length > 0 && (
            <span className="flex items-start gap-1.5"><ArrowLeft className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-600" /><span className="min-w-0">Builds on {from.join(', ')}</span></span>
          )}
          {to.length > 0 && (
            <span className="flex items-start gap-1.5"><ArrowRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-600" /><span className="min-w-0">Feeds {to.join(', ')}</span></span>
          )}
        </span>
      )}
    </button>
  )
}

function QuestionRow({ q, disabled, onChange }: { q: ForgeQuestion; disabled: boolean; onChange: (q: ForgeQuestion) => void }) {
  const selected = q.selected || []
  const [typing, setTyping] = useState(!!(q.other || '').trim() || q.options.length === 0)
  const toggle = (opt: string) => {
    const has = selected.includes(opt)
    onChange({ ...q, selected: q.multi ? (has ? selected.filter((o) => o !== opt) : [...selected, opt]) : (has ? [] : [opt]) })
  }
  return (
    <div className="py-3 first:pt-0 last:pb-0">
      <p className="text-sm font-medium leading-snug text-zinc-200">{q.question}</p>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {q.options.map((o) => <Chip key={o} on={selected.includes(o)} disabled={disabled} onClick={() => toggle(o)}>{o}</Chip>)}
        {q.options.length > 0 && <Chip on={typing} disabled={disabled} onClick={() => { if (typing) onChange({ ...q, other: '' }); setTyping(!typing) }}>Something else</Chip>}
      </div>
      {typing && (
        <input
          autoFocus={q.options.length > 0}
          value={q.other || ''}
          onChange={(e) => onChange({ ...q, other: e.target.value })}
          disabled={disabled}
          aria-label={`Your answer: ${q.question}`}
          placeholder="Your answer"
          className={`${inputCls} fd-rise mt-2`}
        />
      )}
    </div>
  )
}

function AgentSheet({ agent, agents, archetypes, onPatch, onLeaveOut, onClose }: {
  agent: ForgeDraftAgent; agents: ForgeDraftAgent[]; archetypes: Archetype[]
  onPatch: (p: Partial<ForgeDraftAgent>) => void; onLeaveOut: () => void; onClose: () => void
}) {
  const families = useMemo(() => {
    const byFamily = new Map<string, Archetype[]>()
    for (const a of archetypes) byFamily.set(a.family || 'Other', [...(byFamily.get(a.family || 'Other') || []), a])
    return [...byFamily.entries()].sort(([x], [y]) => x.localeCompare(y))
  }, [archetypes])
  const base = archetypes.find((a) => a.id === agent.archetype)
  const others = agents.filter((a) => a.id !== agent.id)
  return (
    <Sheet
      wide
      title={agent.name || 'This agent'}
      subtitle="Your changes stay here until you create the team."
      onClose={onClose}
      footer={
        <>
          <Button variant="danger" icon={Trash2} className="mr-auto" onClick={onLeaveOut}>Leave out</Button>
          <Button variant="primary" icon={Check} onClick={onClose}>Done</Button>
        </>
      }
    >
      <div className="space-y-4">
        <Field label="Name">
          <input value={agent.name} onChange={(e) => onPatch({ name: e.target.value })} maxLength={60} className={inputCls} />
        </Field>
        <Field label="Based on" hint={base?.description}>
          <select value={agent.archetype} onChange={(e) => onPatch({ archetype: e.target.value })} className={inputCls}>
            {!base && <option value={agent.archetype}>{agent.archetype}</option>}
            {families.map(([family, list]) => (
              <optgroup key={family} label={family}>
                {list.map((a) => <option key={a.id} value={a.id}>{a.role || a.id}</option>)}
              </optgroup>
            ))}
          </select>
        </Field>
        <Field label="Its job">
          <textarea value={agent.mandate} onChange={(e) => onPatch({ mandate: e.target.value })} rows={8} className={`${inputCls} resize-y leading-relaxed`} />
        </Field>
        <Field label="What it delivers">
          <textarea value={agent.produces} onChange={(e) => onPatch({ produces: e.target.value })} rows={2} className={`${inputCls} resize-y leading-relaxed`} />
        </Field>
        {others.length > 0 && (
          <Field label="Builds on the work of">
            <div className="flex flex-wrap gap-1.5">
              {others.map((a) => {
                const on = agent.consumes_from.includes(a.id)
                return (
                  <Chip key={a.id} on={on} onClick={() => onPatch({ consumes_from: on ? agent.consumes_from.filter((c) => c !== a.id) : [...agent.consumes_from, a.id] })}>
                    {a.name}
                  </Chip>
                )
              })}
              {!agent.consumes_from.length && <Pill>Starts from scratch</Pill>}
            </div>
          </Field>
        )}
        <Field label="How teammates get its work">
          <textarea value={agent.hand_off_notes} onChange={(e) => onPatch({ hand_off_notes: e.target.value })} rows={3} className={`${inputCls} resize-y leading-relaxed`} />
        </Field>
      </div>
    </Sheet>
  )
}
