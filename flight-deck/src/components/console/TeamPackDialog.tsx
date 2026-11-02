import { useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import { createPortal } from 'react-dom'
import {
  Check, ChevronDown, FileText, Layers, Loader2, Paperclip, RefreshCw, Sparkles, Trash2, Users, X,
} from 'lucide-react'
import {
  DISCLAIMER_CHOICES, outlineTeamPack, publishTeamPack, writeTeamInstructions,
  type PackArchetype, type PackFamily, type PackOutline,
} from '../../services/archetypes'
import { TEAM_COLORS, TEAM_ICONS, archetypeLook, lookFor, shelves } from '../layout/archetypeLook'
import { ErrorNote } from '../forge/forgeParts'
import { errorOf, fmtSize } from '../forge/forgeUtil'
import { Button, inputCls, labelCls } from './parts'

// ── Generate a group ─────────────────────────────────────────────────
//
// An admin says what their people need. Captain drafts a group, a family and the
// archetypes in them; the admin changes what they like. Then each one's instructions
// are written in a call of its own, three at a time, so one that fails can be tried
// again without losing the rest. Publishing makes them available to everyone.

const ACCEPT = '.pdf,.docx,.xlsx,.pptx,.txt,.md,.csv,.json'
const MAX_FILE_BYTES = 25 * 1024 * 1024
const WRITERS = 3
const NEW_GROUP = '__new__'

type WriteState = 'idle' | 'writing' | 'done' | 'failed'
interface Card extends PackArchetype { key: string; text: string; state: WriteState; error: string; open: boolean }

export function TeamPackDialog({ existingFamilies, onClose, onPublished }: {
  existingFamilies: string[]
  onClose: () => void
  onPublished: () => void
}) {
  const [brief, setBrief] = useState('')
  const [count, setCount] = useState(0)
  const [files, setFiles] = useState<File[]>([])
  const [drafting, setDrafting] = useState(false)
  const [err, setErr] = useState('')
  const [outline, setOutline] = useState<PackOutline | null>(null)
  const [groupChoice, setGroupChoice] = useState(NEW_GROUP)
  const [groupLabel, setGroupLabel] = useState('')
  const [groupHint, setGroupHint] = useState('')
  const [family, setFamily] = useState<PackFamily>({ name: '', icon: 'Sparkles', color: 'violet' })
  const [cards, setCards] = useState<Card[]>([])
  const [writing, setWriting] = useState(false)
  const [publishing, setPublishing] = useState(false)
  const [published, setPublished] = useState<{ count: number; group: string; family: string } | null>(null)
  const picker = useRef<HTMLInputElement>(null)

  const patch = (key: string, change: Partial<Card>) => setCards((all) => all.map((c) => (c.key === key ? { ...c, ...change } : c)))

  const addFiles = (e: ChangeEvent<HTMLInputElement>) => {
    const list = Array.from(e.target.files ?? [])
    e.target.value = ''
    const tooBig = list.find((f) => f.size > MAX_FILE_BYTES)
    if (tooBig) { setErr(`${tooBig.name} is over 25 MB. Try a smaller copy.`); return }
    setErr('')
    setFiles((old) => [...old, ...list.filter((f) => !old.some((o) => o.name === f.name))])
  }

  const draft = async () => {
    if ((!brief.trim() && !files.length) || drafting) return
    setDrafting(true)
    setErr('')
    try {
      const out = await outlineTeamPack(brief.trim(), count, files)
      setOutline(out)
      setGroupLabel(out.group.label)
      setGroupHint(out.group.hint)
      setFamily(out.family)
      setCards(out.archetypes.map((a, i) => ({ ...a, key: `c${i}`, text: '', state: 'idle', error: '', open: false })))
    } catch (e) {
      setErr(errorOf(e))
    } finally {
      setDrafting(false)
    }
  }

  const writeOne = async (card: Card, all: Card[]) => {
    patch(card.key, { state: 'writing', error: '' })
    try {
      const text = await writeTeamInstructions({
        brief: brief.trim(),
        reference: outline?.reference || '',
        group: groupChoice === NEW_GROUP ? groupLabel : shelves().find((s) => s.id === groupChoice)?.label || '',
        family: family.name,
        role: card.role,
        description: card.description,
        keywords: card.keywords,
        tools: card.tools,
        disclaimer: card.disclaimer,
        others: all.filter((c) => c.key !== card.key).map((c) => ({ role: c.role, description: c.description })),
      })
      patch(card.key, { text, state: 'done' })
    } catch (e) {
      patch(card.key, { state: 'failed', error: errorOf(e) })
    }
  }

  const writeAll = async () => {
    setWriting(true)
    setErr('')
    const queue = cards.filter((c) => !c.text)
    await Promise.all(Array.from({ length: WRITERS }, async () => {
      for (let c = queue.shift(); c; c = queue.shift()) await writeOne(c, cards)
    }))
    setWriting(false)
  }

  const publish = async () => {
    setPublishing(true)
    setErr('')
    try {
      const out = await publishTeamPack({
        group: groupChoice === NEW_GROUP ? { label: groupLabel.trim(), hint: groupHint.trim() } : { id: groupChoice },
        family,
        archetypes: cards.map((c) => ({
          archetype_id: '', role: c.role, family: family.name, description: c.description, cognitive_mode: c.cognitive_mode,
          tier: c.tier, tools: c.tools, fleet_instructions: c.text, keywords: c.keywords, lead: false, reliability_seed: 0.7,
          runtime: '', disclaimer: c.disclaimer, icon: c.icon,
        })),
      })
      onPublished()
      setPublished({
        count: out.published,
        group: groupChoice === NEW_GROUP ? groupLabel.trim() : shelves().find((s) => s.id === groupChoice)?.label || '',
        family: out.family,
      })
    } catch (e) {
      setErr(errorOf(e))
    } finally {
      setPublishing(false)
    }
  }

  const asked = () => {
    if (!outline || published) return onClose()
    if (confirm('Close without publishing? This draft will be lost.')) onClose()
  }

  const allWritten = cards.length > 0 && cards.every((c) => c.text.trim())
  const named = !!family.name.trim() && (groupChoice !== NEW_GROUP || !!groupLabel.trim())
  const look = lookFor(family.icon, family.color)
  const FamilyIcon = look.icon
  const newFamily = !existingFamilies.includes(family.name.trim())

  return createPortal(
    <div className="fd-fade fixed inset-0 z-50 flex items-stretch justify-center bg-black/60 backdrop-blur-[2px] wide:items-center wide:p-4">
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Generate a group of archetypes"
        className="fd-sheet fd-card-wide flex h-[100dvh] w-full flex-col overflow-hidden bg-zinc-950 pb-[env(safe-area-inset-bottom)] shadow-2xl wide:h-[min(92vh,62rem)] wide:max-w-5xl wide:rounded-2xl wide:border wide:border-zinc-800 wide:pb-0"
      >
        <div className="flex shrink-0 items-center gap-3 border-b border-zinc-800 px-4 py-3 wide:px-5">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/10 text-violet-600 dark:text-violet-400">
            <Layers className="h-[18px] w-[18px]" />
          </span>
          <div className="min-w-0 flex-1">
            <h3 className="truncate text-base font-semibold text-zinc-100">Generate a group</h3>
            <p className="truncate text-xs text-zinc-500">A new group and family of archetypes, for everyone on this deck.</p>
          </div>
          <button onClick={asked} aria-label="Close" title="Close" className="rounded-lg p-2 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-[18px] w-[18px]" />
          </button>
        </div>

        {published ? (
          <div className="flex min-h-0 flex-1 flex-col items-center justify-center px-6 text-center">
            <span className="fd-rise flex h-14 w-14 items-center justify-center rounded-2xl bg-emerald-500/15 text-emerald-700 dark:text-emerald-300">
              <Check className="h-7 w-7" />
            </span>
            <h2 className="mt-4 text-2xl font-semibold tracking-tight text-zinc-100">Published to everyone</h2>
            <p className="mt-1.5 max-w-md text-sm leading-relaxed text-zinc-500">
              {published.count} archetype{published.count === 1 ? ' is' : 's are'} now in {published.group}, under {published.family}. Everyone on this deck sees them in New agent.
            </p>
            <Button variant="primary" icon={Check} className="mt-5" onClick={onClose}>Done</Button>
          </div>
        ) : !outline ? (
          <div className="min-h-0 flex-1 overflow-y-auto">
            <div className="mx-auto w-full max-w-3xl space-y-5 px-4 py-6 wide:py-10">
              <div className="fd-rise">
                <h2 className="text-2xl font-semibold tracking-tight text-zinc-100 wide:text-3xl">Who is this group for?</h2>
                <p className="mt-1.5 text-sm leading-relaxed text-zinc-500">Describe the people and the work. Captain drafts a group, a family and the archetypes in them, and nothing reaches anyone until you publish.</p>
              </div>
              <div className="fd-rise rounded-2xl border border-zinc-700 bg-zinc-900/40 shadow-sm transition-colors focus-within:border-violet-500/60 focus-within:ring-2 focus-within:ring-violet-500/15">
                <textarea
                  autoFocus
                  value={brief}
                  onChange={(e) => setBrief(e.target.value)}
                  disabled={drafting}
                  aria-label="Who is this group for?"
                  placeholder="We run three dental practices. Reception drowns in treatment plan questions and insurance letters, and each practice manager writes the same reports by hand…"
                  className="block max-h-[45vh] min-h-[8.5rem] w-full resize-none bg-transparent px-4 pb-2 pt-4 text-base leading-relaxed text-zinc-100 placeholder-zinc-600 [field-sizing:content] focus:outline-none disabled:opacity-60 wide:text-[15px]"
                />
                {files.length > 0 && (
                  <ul className="flex flex-wrap gap-1.5 px-3 pb-2" aria-label="Attached files">
                    {files.map((f) => (
                      <li key={f.name} className="flex max-w-full items-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-950 py-1 pl-2 pr-1 text-xs">
                        <FileText className="h-3.5 w-3.5 shrink-0 text-violet-600 dark:text-violet-400" />
                        <span className="truncate text-zinc-300">{f.name}</span>
                        <span className="shrink-0 tabular-nums text-zinc-600">{fmtSize(f.size)}</span>
                        <button onClick={() => setFiles((old) => old.filter((x) => x !== f))} disabled={drafting} aria-label={`Remove ${f.name}`} className="shrink-0 rounded p-0.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
                          <X className="h-3 w-3" />
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
                <div className="flex flex-wrap items-center gap-2 border-t border-zinc-800/80 p-2">
                  <button type="button" onClick={() => picker.current?.click()} disabled={drafting} className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50">
                    <Paperclip className="h-4 w-4" /> Attach
                  </button>
                  <input ref={picker} type="file" multiple hidden accept={ACCEPT} onChange={addFiles} />
                  <label className="inline-flex items-center gap-2 text-xs font-medium text-zinc-400">
                    <Users className="h-3.5 w-3.5 text-zinc-500" /> How many
                    <select value={count} onChange={(e) => setCount(Number(e.target.value))} disabled={drafting} className="rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-200 focus:border-violet-500/60 focus:outline-none">
                      <option value={0}>Captain decides</option>
                      {Array.from({ length: 12 }, (_, i) => i + 1).map((n) => <option key={n} value={n}>{n}</option>)}
                    </select>
                  </label>
                  <button onClick={draft} disabled={(!brief.trim() && !files.length) || drafting} className="ml-auto inline-flex shrink-0 items-center gap-1.5 rounded-xl bg-violet-600 px-3.5 py-2 text-sm font-medium text-white shadow-sm shadow-violet-900/20 transition-colors hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-50">
                    {drafting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                    {drafting ? 'Drafting…' : 'Draft the group'}
                  </button>
                </div>
              </div>
              {err && <ErrorNote>{err}</ErrorNote>}
            </div>
          </div>
        ) : (
          <>
            <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
              <div className="mx-auto w-full max-w-4xl space-y-5 px-4 py-5 wide:px-6 wide:py-6">
                <div className="grid gap-3 md:grid-cols-2">
                  <section aria-label="Group" className="space-y-3 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
                    <div>
                      <label className={labelCls} htmlFor="pack-group">Group</label>
                      <select id="pack-group" value={groupChoice} onChange={(e) => setGroupChoice(e.target.value)} className={inputCls}>
                        <option value={NEW_GROUP}>A new group</option>
                        {shelves().map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
                      </select>
                    </div>
                    {groupChoice === NEW_GROUP && (
                      <>
                        <div>
                          <label className={labelCls} htmlFor="pack-group-label">Its name</label>
                          <input id="pack-group-label" value={groupLabel} onChange={(e) => setGroupLabel(e.target.value)} maxLength={40} className={inputCls} />
                        </div>
                        <div>
                          <label className={labelCls} htmlFor="pack-group-hint">One line on who it's for</label>
                          <input id="pack-group-hint" value={groupHint} onChange={(e) => setGroupHint(e.target.value)} maxLength={140} className={inputCls} />
                        </div>
                      </>
                    )}
                  </section>
                  <section aria-label="Family" className="space-y-3 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
                    <div className="flex items-end gap-3">
                      <span className={`flex h-[38px] w-[38px] shrink-0 items-center justify-center rounded-xl ring-1 ${look.bg} ${look.text} ${look.ring}`}>
                        <FamilyIcon className="h-[18px] w-[18px]" />
                      </span>
                      <div className="min-w-0 flex-1">
                        <label className={labelCls} htmlFor="pack-family">Family</label>
                        <input id="pack-family" list="pack-families" value={family.name} onChange={(e) => setFamily({ ...family, name: e.target.value })} maxLength={40} className={inputCls} />
                        <datalist id="pack-families">{existingFamilies.map((f) => <option key={f} value={f} />)}</datalist>
                      </div>
                    </div>
                    {newFamily && (
                      <>
                        <div role="radiogroup" aria-label="Family colour" className="flex flex-wrap gap-1.5">
                          {TEAM_COLORS.map((c) => {
                            const swatch = lookFor(family.icon, c)
                            return (
                              <button key={c} type="button" role="radio" aria-checked={family.color === c} aria-label={c} title={c} onClick={() => setFamily({ ...family, color: c })}
                                className={`flex h-6 w-6 items-center justify-center rounded-full ring-1 ${swatch.bg} ${swatch.text} ${swatch.ring} ${family.color === c ? 'outline outline-2 outline-offset-1 outline-violet-500' : ''}`}>
                                <span className="h-2 w-2 rounded-full bg-current" />
                              </button>
                            )
                          })}
                        </div>
                        <div role="radiogroup" aria-label="Family icon" className="flex flex-wrap gap-1">
                          {Object.entries(TEAM_ICONS).map(([name, Icon]) => (
                            <button key={name} type="button" role="radio" aria-checked={family.icon === name} aria-label={name} title={name} onClick={() => setFamily({ ...family, icon: name })}
                              className={`flex h-7 w-7 items-center justify-center rounded-lg transition-colors ${family.icon === name ? 'bg-violet-500/15 text-violet-700 dark:text-violet-300' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200'}`}>
                              <Icon className="h-3.5 w-3.5" />
                            </button>
                          ))}
                        </div>
                      </>
                    )}
                  </section>
                </div>

                <ul className="space-y-2" aria-label="The archetypes">
                  {cards.map((c) => {
                    const cardLook = archetypeLook(c.key, family.name)
                    const CardIcon = (c.icon && TEAM_ICONS[c.icon]) || cardLook.icon
                    return (
                      <li key={c.key} className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-3">
                        <div className="flex items-start gap-3">
                          <span className={`mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ring-1 ${look.bg} ${look.text} ${look.ring}`}>
                            <CardIcon className="h-[18px] w-[18px]" />
                          </span>
                          <div className="min-w-0 flex-1 space-y-1">
                            <input aria-label="Role" value={c.role} onChange={(e) => patch(c.key, { role: e.target.value })} className="w-full rounded-md bg-transparent px-1 py-0.5 text-sm font-semibold text-zinc-100 hover:bg-zinc-800/60 focus:bg-zinc-800/60 focus:outline-none" />
                            <input aria-label="What it does" value={c.description} onChange={(e) => patch(c.key, { description: e.target.value })} className="w-full rounded-md bg-transparent px-1 py-0.5 text-xs text-zinc-500 hover:bg-zinc-800/60 focus:bg-zinc-800/60 focus:outline-none" />
                            <label className="flex flex-wrap items-center gap-1.5 px-1 pt-0.5 text-[11px] text-zinc-500">
                              Disclaimer
                              <select aria-label={`Disclaimer for ${c.role}`} value={c.disclaimer} onChange={(e) => patch(c.key, { disclaimer: e.target.value })} className={`rounded-md border bg-zinc-950 px-1.5 py-0.5 text-[11px] focus:border-violet-500/60 focus:outline-none ${c.disclaimer ? 'border-amber-500/40 text-amber-700 dark:text-amber-300' : 'border-zinc-800 text-zinc-400'}`}>
                                {DISCLAIMER_CHOICES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                              </select>
                            </label>
                          </div>
                          <WriteMark state={c.state} text={c.text} />
                          <button onClick={() => patch(c.key, { open: !c.open })} aria-expanded={c.open} aria-label={`Instructions for ${c.role}`} title="Instructions" className="shrink-0 rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
                            <ChevronDown className={`h-4 w-4 transition-transform ${c.open ? '' : '-rotate-90'}`} />
                          </button>
                          <button onClick={() => setCards((all) => all.filter((x) => x.key !== c.key))} disabled={c.state === 'writing'} aria-label={`Leave out ${c.role}`} title="Leave out" className="shrink-0 rounded-lg p-1.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-red-500 disabled:opacity-40">
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                        {c.state === 'failed' && (
                          <div className="mt-2 flex items-center gap-2 pl-12">
                            <p className="min-w-0 flex-1 text-xs text-red-700 dark:text-red-300">{c.error}</p>
                            <Button icon={RefreshCw} className="!px-2 !py-1 !text-xs" disabled={writing} onClick={() => { void writeOne(c, cards) }}>Try again</Button>
                          </div>
                        )}
                        {c.open && (
                          <div className="fd-rise mt-2 pl-12">
                            {c.text ? (
                              <textarea aria-label={`Instructions for ${c.role}`} value={c.text} onChange={(e) => patch(c.key, { text: e.target.value })} rows={14} className={`${inputCls} resize-y font-mono !text-xs leading-relaxed`} />
                            ) : (
                              <p className="text-xs text-zinc-500">Not written yet.</p>
                            )}
                          </div>
                        )}
                      </li>
                    )
                  })}
                </ul>
                {!cards.length && <p className="rounded-2xl border border-dashed border-zinc-800 px-6 py-8 text-center text-sm text-zinc-500">You've left them all out. Close this and start again.</p>}
                {err && <ErrorNote>{err}</ErrorNote>}
              </div>
            </div>

            <div className="shrink-0 border-t border-zinc-800 px-4 py-3 wide:px-6">
              <div className="mx-auto flex w-full max-w-4xl flex-col gap-2 wide:flex-row wide:items-center">
                <p className="min-w-0 flex-1 text-xs leading-relaxed text-zinc-500">
                  {allWritten
                    ? 'Captain wrote these from its own knowledge, with no web search. Read them before you publish, above all anything legal, medical or financial: everyone on this deck can use them.'
                    : 'Change anything above first: the names and descriptions go into how each one is written.'}
                </p>
                {allWritten ? (
                  <Button variant="primary" icon={Users} busy={publishing} disabled={!named || writing} onClick={publish}>Publish to everyone</Button>
                ) : (
                  <Button variant="primary" icon={Sparkles} busy={writing} disabled={!named || !cards.length} onClick={writeAll}>
                    {writing ? 'Writing…' : `Write ${cards.filter((c) => !c.text).length} set${cards.filter((c) => !c.text).length === 1 ? '' : 's'} of instructions`}
                  </Button>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>,
    document.body,
  )
}

function WriteMark({ state, text }: { state: WriteState; text: string }) {
  if (state === 'writing') return <Loader2 aria-label="Writing" className="mt-2 h-4 w-4 shrink-0 animate-spin text-violet-600 dark:text-violet-400" />
  if (text) return <Check aria-label="Written" className="mt-2 h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
  return null
}
