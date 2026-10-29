import { useState } from 'react'
import type { ReactNode } from 'react'
import { AlertTriangle, FileText, Library, Pin, PinOff } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { AgentPreset, PresetSource } from '../../services/agentPreset'
import { GROUPS, GROUP_ORDER, USAGE_LOOKS, keptIn, type Group, type Review } from './agentSetup'

// ── The review step of AgentSetupDialog ──────────────────────────────
//
// What the owner's model pulled out of their instructions and documents, one
// group at a time (a sidebar of groups on a wide screen, chips on a phone):
// every item kept or left out, editable, with the document it came from.

const fieldCls = 'w-full rounded-lg border border-zinc-800 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/60 focus:outline-none disabled:opacity-60'

type View = Group | 'notices'

export function PresetReview({ review, disabled, onChange }: {
  review: Review
  disabled: boolean
  onChange: (r: Review) => void
}) {
  const { preset, off, prepared } = review
  const readDocs = prepared.docs.filter((d) => d.read).length
  const notices = [...preset.conflicts.map((t) => ({ t, conflict: true })), ...preset.warnings.map((t) => ({ t, conflict: false }))]
  const groups = GROUP_ORDER.filter((g) => preset[g].length > 0)
  // Conflicts first: they are what the person most needs to see.
  const [view, setView] = useState<View>(() => (preset.conflicts.length ? 'notices' : groups[0] || 'notices'))

  const toggle = (key: string) => {
    const next = new Set(off)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    onChange({ ...review, off: next })
  }
  const setAll = (g: Group, keep: boolean) => {
    const next = new Set(off)
    preset[g].forEach((_, i) => (keep ? next.delete(`${g}:${i}`) : next.add(`${g}:${i}`)))
    onChange({ ...review, off: next })
  }
  function edit<G extends Group>(g: G, i: number, patch: Partial<AgentPreset[G][number]>) {
    const list = preset[g].map((item, j) => (j === i ? { ...item, ...patch } : item))
    onChange({ ...review, preset: { ...preset, [g]: list } })
  }

  if (groups.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center px-8 py-10 text-center">
        <p className="text-sm text-zinc-300">Nothing to set up came out of this.</p>
        <p className="mt-1 text-xs text-zinc-500">Go back and add more, or close — the agent works as it is.</p>
        {notices.length > 0 && <div className="mt-4 w-full max-w-lg text-left"><Notices notices={notices} /></div>}
      </div>
    )
  }

  const nav: { id: View; icon: LucideIcon; title: string; badge: string; warn?: boolean }[] = [
    ...(notices.length ? [{ id: 'notices' as View, icon: AlertTriangle, title: 'Check first', badge: String(notices.length), warn: true }] : []),
    ...groups.map((g) => ({ id: g as View, icon: GROUPS[g].icon, title: GROUPS[g].title, badge: `${keptIn(review, g)}/${preset[g].length}` })),
  ]

  return (
    <div className="flex min-h-0 flex-1 flex-col wide:flex-row">
      {/* Groups: a sidebar on a wide screen, a row of chips on a phone. */}
      <nav className="shrink-0 border-b border-zinc-800 wide:w-60 wide:border-b-0 wide:border-r wide:bg-zinc-950/40" aria-label="What it starts with">
        <p className="hidden px-4 pt-4 pb-2 text-[11px] leading-snug text-zinc-500 wide:block">
          {prepared.steps > 0 ? (
            <>
              Read by <span className="text-zinc-400">{prepared.model || 'your model'}</span>
              {prepared.steps > 1 && ` in ${prepared.steps} steps`}
              {readDocs > 0 && ` from ${readDocs} document${readDocs === 1 ? '' : 's'}`}.
            </>
          ) : 'Nothing was read — the documents are kept as knowledge only.'}
        </p>
        <ul className="flex gap-1.5 overflow-x-auto px-3 py-2.5 wide:flex-col wide:gap-0.5 wide:overflow-visible wide:px-2 wide:py-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {nav.map(({ id, icon: Icon, title, badge, warn }) => {
            const active = view === id
            return (
              <li key={id} className="shrink-0">
                <button
                  onClick={() => setView(id)}
                  aria-current={active}
                  className={`flex w-full items-center gap-2 rounded-lg border px-2.5 py-1.5 text-left text-sm transition-colors wide:border-transparent ${
                    active
                      ? 'border-violet-500/40 bg-violet-500/10 text-violet-700 dark:text-violet-200'
                      : 'border-zinc-800 text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200'
                  }`}
                >
                  <Icon className={`h-3.5 w-3.5 shrink-0 ${warn ? 'text-amber-500' : active ? '' : 'text-zinc-500'}`} />
                  <span className="whitespace-nowrap wide:flex-1 wide:truncate">{title}</span>
                  <span className={`rounded-md px-1.5 text-[10px] tabular-nums ${
                    warn ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300' : 'bg-zinc-800 text-zinc-400'
                  }`}>{badge}</span>
                </button>
              </li>
            )
          })}
        </ul>
      </nav>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 wide:px-6">
        <div className="mx-auto max-w-3xl">
          {view === 'notices' ? (
            <section>
              <h4 className="text-sm font-semibold text-zinc-100">Check these first</h4>
              <p className="mb-3 text-xs text-zinc-500">Where your material contradicts itself, and anything that didn't come through.</p>
              <Notices notices={notices} />
            </section>
          ) : (
            <GroupView
              group={view}
              review={review}
              disabled={disabled}
              onToggle={toggle}
              onAll={(keep) => setAll(view, keep)}
              onEdit={(i, patch) => edit(view, i, patch as never)}
            />
          )}
        </div>
      </div>
    </div>
  )
}

function Notices({ notices }: { notices: { t: string; conflict: boolean }[] }) {
  return (
    <ul className="space-y-1.5 rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2.5">
      {notices.map(({ t, conflict }, i) => (
        <li key={i} className="flex items-start gap-2 text-xs text-amber-700 dark:text-amber-300">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{conflict && <span className="font-medium">Conflict: </span>}{t}</span>
        </li>
      ))}
    </ul>
  )
}

function GroupView({ group, review, disabled, onToggle, onAll, onEdit }: {
  group: Group
  review: Review
  disabled: boolean
  onToggle: (key: string) => void
  onAll: (keep: boolean) => void
  onEdit: (i: number, patch: Record<string, unknown>) => void
}) {
  const { icon: Icon, title, blurb } = GROUPS[group]
  const items = review.preset[group]
  const kept = keptIn(review, group)
  const docs = new Map(review.prepared.docs.map((d) => [d.name, d]))
  const found = group === 'knowledge' ? 0 : review.prepared.found?.[group] ?? 0
  return (
    <section>
      <div className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-1">
        <Icon className="h-4 w-4 shrink-0 text-violet-500 dark:text-violet-400" />
        <div className="min-w-0 flex-1">
          <h4 className="text-sm font-semibold text-zinc-100">{title} <span className="font-normal text-zinc-500">· {kept} of {items.length} kept</span></h4>
          <p className="text-xs text-zinc-500">{blurb}</p>
          {found > items.length && (
            <p className="mt-1 text-xs text-zinc-500">
              The read found {found.toLocaleString()}; {items.length === 1 ? 'this is the one' : `these are the ${items.length.toLocaleString()}`} found most often.
              The agent can still look up the rest in its documents.
            </p>
          )}
        </div>
        <div className="flex gap-1 text-[11px]">
          <button disabled={disabled || kept === items.length} onClick={() => onAll(true)} className="rounded-md px-2 py-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40">Keep all</button>
          <button disabled={disabled || kept === 0} onClick={() => onAll(false)} className="rounded-md px-2 py-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-40">Leave all out</button>
        </div>
      </div>
      <ul className="space-y-2">
        {items.map((item, i) => {
          const key = `${group}:${i}`
          const on = !review.off.has(key)
          const doc = group === 'knowledge' ? docs.get((item as { doc: string }).doc) : undefined
          return (
            <li key={key} className={`flex gap-2.5 rounded-xl border px-3 py-2.5 transition-colors ${on ? 'border-zinc-800 bg-zinc-900/60' : 'border-zinc-800/60 bg-transparent opacity-60'}`}>
              <input
                type="checkbox"
                checked={on}
                disabled={disabled}
                onChange={() => onToggle(key)}
                aria-label={on ? 'Leave this out' : 'Keep this'}
                className="mt-1.5 h-4 w-4 shrink-0 accent-violet-600"
              />
              <div className="min-w-0 flex-1 space-y-1.5">
                <ItemFields group={group} item={item} disabled={disabled || !on} onEdit={(patch) => onEdit(i, patch)} />
                {(group === 'rules' || group === 'escalation') && (
                  <AlwaysToggle always={!!(item as { always?: boolean }).always} disabled={disabled || !on}
                    onChange={(always) => onEdit(i, { always })} />
                )}
                {doc && (
                  <p className="flex flex-wrap items-center gap-1.5 text-[10px] text-zinc-500">
                    <span className="inline-flex items-center gap-1">
                      {doc.origin === 'library' ? <Library className="h-3 w-3" /> : <FileText className="h-3 w-3" />}
                      {doc.name}{doc.agent ? ` · from ${doc.agent}` : ''}
                    </span>
                    {doc.usage !== 'auto' && (
                      <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-400">{USAGE_LOOKS[doc.usage].label}</span>
                    )}
                    {doc.pages > 0 && <span>{doc.pages_exact ? '' : 'about '}{doc.pages.toLocaleString()} page{doc.pages === 1 ? '' : 's'}</span>}
                    {!doc.read && <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-400">Knowledge only</span>}
                    {!!doc.tables && (
                      <span className="rounded bg-emerald-500/10 px-1.5 py-0.5 text-emerald-700 dark:text-emerald-400">
                        {doc.tables === 1 ? 'Also a table' : `Also ${doc.tables} tables`}
                      </span>
                    )}
                  </p>
                )}
              </div>
            </li>
          )
        })}
      </ul>
    </section>
  )
}

/** Whether a rule is in its instructions all the time, or reaches it with the messages it bears on. */
function AlwaysToggle({ always, disabled, onChange }: { always: boolean; disabled: boolean; onChange: (v: boolean) => void }) {
  return (
    <button type="button" onClick={() => onChange(!always)} disabled={disabled} aria-pressed={always}
      title={always ? 'In its instructions, for every conversation — click to bring it up only when it matters'
        : 'It\'s reminded of this when a message is about it — click to keep it in mind all the time'}
      className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] transition-colors disabled:opacity-50 ${always
        ? 'bg-violet-500/10 text-violet-700 hover:bg-violet-500/20 dark:text-violet-300'
        : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'}`}>
      {always ? <Pin className="h-3 w-3" /> : <PinOff className="h-3 w-3" />}
      {always ? 'Always in mind' : 'When it matters'}
    </button>
  )
}

function SourceLine({ source }: { source?: PresetSource }) {
  if (!source || (!source.doc && !source.section)) return null
  const from = source.doc === 'instructions' ? 'your instructions' : source.doc
  return <p className="truncate text-[10px] text-zinc-600">from {from}{source.section ? ` · ${source.section}` : ''}</p>
}

function Lines({ value, disabled, rows, placeholder, onChange }: {
  value: string[]; disabled: boolean; rows: number; placeholder: string; onChange: (v: string[]) => void
}) {
  return (
    <textarea
      value={value.join('\n')}
      disabled={disabled}
      rows={rows}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value.split('\n'))}
      onBlur={(e) => onChange(e.target.value.split('\n').map((s) => s.trim()).filter(Boolean))}
      className={`${fieldCls} resize-y`}
    />
  )
}

function ItemFields({ group, item, disabled, onEdit }: {
  group: Group
  item: AgentPreset[Group][number]
  disabled: boolean
  onEdit: (patch: Record<string, unknown>) => void
}): ReactNode {
  const it = item as unknown as Record<string, unknown>
  const text = (key: string, placeholder: string, rows = 1) => (
    rows > 1
      ? <textarea value={String(it[key] ?? '')} disabled={disabled} rows={rows} placeholder={placeholder}
          onChange={(e) => onEdit({ [key]: e.target.value })} className={`${fieldCls} resize-y`} />
      : <input value={String(it[key] ?? '')} disabled={disabled} placeholder={placeholder}
          onChange={(e) => onEdit({ [key]: e.target.value })} className={fieldCls} />
  )
  const source = <SourceLine source={it.source as PresetSource | undefined} />
  switch (group) {
    case 'rules':
    case 'escalation':
      return <>{text('text', 'The rule', 2)}{source}</>
    case 'facts':
      return <>{text('content', 'The fact', 2)}{source}</>
    case 'intents':
      return (
        <>
          <div className="flex flex-col gap-1.5 wide:flex-row">
            <div className="wide:w-40 wide:shrink-0">{text('name', 'name')}</div>
            <div className="min-w-0 flex-1">{text('when', 'When it applies')}</div>
          </div>
          <Lines value={(it.steps as string[]) || []} disabled={disabled} rows={3} placeholder="One step per line"
            onChange={(steps) => onEdit({ steps })} />
          {source}
        </>
      )
    case 'templates':
      return <>{text('name', 'Template name')}{text('body', 'The format', 4)}{source}</>
    case 'glossary':
      return (
        <>
          <div className="flex flex-col gap-1.5 wide:flex-row">
            <div className="wide:w-40 wide:shrink-0">{text('term', 'Term')}</div>
            <div className="min-w-0 flex-1">{text('meaning', 'What it means')}</div>
          </div>
          {source}
        </>
      )
    case 'knowledge':
      return (
        <>
          {text('title', 'Title')}
          {text('summary', 'What is in it, and when to look there', 2)}
        </>
      )
  }
}
