import { useState, useRef } from 'react'
import {
  Swords, Lightbulb, ClipboardCheck, Map, Play, Paperclip, X, MessageCircleQuestion, Bug, ShieldAlert, MessagesSquare,
  Sparkles, MousePointerClick, Check, Upload, FileText, Users, SlidersHorizontal, Crown, Search,
} from 'lucide-react'
import { AgentPicker, isOldManName } from './AgentPicker'
import { formatSize } from '../../services/fileTransfer'
import { archetypeMatches, useTierConfig } from '../../services/tierConfig'
import type {
  CouncilAgentDef, CreateSessionConfig, SessionType, Verbosity,
} from '../../stores/councilStore'
import { Button, Card, Field, inputCls, labelCls } from '../console/parts'
import { TONE_SQUARE, TYPE_LOOK } from './councilStyles'

const SESSION_TYPES: { id: SessionType; label: string; icon: typeof Swords; desc: string }[] = [
  { id: 'debate', label: 'Debate', icon: Swords, desc: 'Structured argumentation' },
  { id: 'brainstorm', label: 'Brainstorm', icon: Lightbulb, desc: 'Creative ideation' },
  { id: 'review', label: 'Review', icon: ClipboardCheck, desc: 'Critical analysis' },
  { id: 'planning', label: 'Planning', icon: Map, desc: 'Task decomposition' },
  { id: 'interview', label: 'Interview', icon: MessageCircleQuestion, desc: 'Knowledge extraction' },
  { id: 'troubleshoot', label: 'Troubleshoot', icon: Bug, desc: 'Problem diagnosis' },
  { id: 'critique', label: 'Critique', icon: ShieldAlert, desc: 'Adversarial stress-test' },
  { id: 'freeform', label: 'Freeform', icon: MessagesSquare, desc: 'Open conversation' },
]

const VERBOSITY_OPTIONS: { id: Verbosity; label: string; desc: string }[] = [
  { id: 'thought', label: 'Thought', desc: '1-2 sentences' },
  { id: 'message', label: 'Message', desc: 'Up to 5 sentences' },
  { id: 'short', label: 'Short', desc: 'Up to 3 paragraphs' },
  { id: 'medium', label: 'Medium', desc: 'Up to 5 paragraphs' },
  { id: 'long', label: 'Long', desc: 'Up to 10 paragraphs' },
]

// A choice among several, as a tile: violet when picked.
const CHOICE = 'rounded-2xl border text-left transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/40'
const CHOICE_ON = 'border-violet-500/60 bg-violet-500/[0.08] ring-1 ring-violet-500/30'
const CHOICE_OFF = 'border-zinc-800 bg-zinc-950/40 hover:-translate-y-px hover:border-zinc-700 hover:bg-zinc-800/60'

const RANGE = 'h-1.5 w-full cursor-pointer accent-violet-500 dark:[color-scheme:dark]'

function Value({ children }: { children: React.ReactNode }) {
  return <span className="rounded-md bg-violet-500/10 px-1.5 py-0.5 text-xs font-semibold tabular-nums text-violet-700 dark:text-violet-300">{children}</span>
}

function Ticks({ marks }: { marks: (number | string)[] }) {
  return (
    <div className="mt-1.5 flex justify-between text-[11px] tabular-nums text-zinc-500">
      {marks.map((m) => <span key={m}>{m}</span>)}
    </div>
  )
}

interface CouncilSetupProps {
  onStart: (cfg: CreateSessionConfig) => void | Promise<void>
  onCancel: () => void
}

export function CouncilSetup({ onStart, onCancel }: CouncilSetupProps) {
  const [title, setTitle] = useState('')
  const [topic, setTopic] = useState('')
  const [sessionType, setSessionType] = useState<SessionType>('brainstorm')
  const [verbosity, setVerbosity] = useState<Verbosity>('message')
  const [maxRounds, setMaxRounds] = useState(5)
  const [agents, setAgents] = useState<CouncilAgentDef[]>([])
  const [firstSpeaker, setFirstSpeaker] = useState('random')
  const [files, setFiles] = useState<File[]>([])
  const [dragOver, setDragOver] = useState(false)
  // Auto-assemble: spawn a task-modeled archetype panel instead of picking
  // already-running agents.
  const [autoAssemble, setAutoAssemble] = useState(true)
  const [maxAgents, setMaxAgents] = useState(4)
  // Optional hand-picked panel. Empty => the router auto-picks the specialists.
  const [selectedArchetypes, setSelectedArchetypes] = useState<string[]>([])
  const [teamQuery, setTeamQuery] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { tiers, envVars, registry } = useTierConfig()
  const archetypes = registry?.archetypes || []
  const toggleArchetype = (id: string) =>
    setSelectedArchetypes(prev => prev.includes(id) ? prev.filter(a => a !== id) : [...prev, id])

  const oldMan = agents.find(a => isOldManName(a.name))
  const moderatorMode = oldMan ? 'moderator' : 'round-robin'
  const canStart = !!topic.trim() && (autoAssemble || agents.length >= 2)

  const addFiles = (newFiles: FileList | File[]) => {
    const arr = Array.from(newFiles)
    setFiles(prev => {
      const names = new Set(prev.map(f => f.name))
      return [...prev, ...arr.filter(f => !names.has(f.name))]
    })
  }

  const removeFile = (name: string) => setFiles(prev => prev.filter(f => f.name !== name))

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files)
  }

  const handleStart = async () => {
    if (!canStart || submitting) return
    setSubmitting(true)
    try {
      await onStart({
        title: title.trim() || topic.slice(0, 60),
        topic: topic.trim(),
        sessionType,
        verbosity,
        maxRounds,
        moderatorMode: autoAssemble ? 'round-robin' : moderatorMode,
        moderatorAgentId: autoAssemble ? '' : (oldMan?.id || ''),
        agents: autoAssemble ? [] : agents,
        firstSpeaker: autoAssemble ? 'random' : firstSpeaker,
        files,
        autoAssemble,
        maxAgents,
        tiers: autoAssemble ? tiers : undefined,
        envVars: autoAssemble ? envVars : undefined,
        archetypeIds: autoAssemble && selectedArchetypes.length ? selectedArchetypes : undefined,
      })
    } catch (e) {
      setSubmitting(false)
      // Surface the failure inline rather than silently swallowing it.
      alert(e instanceof Error ? e.message : 'Failed to start council')
    }
    // On success the page navigates away; no need to clear submitting.
  }

  const panelModes = [
    { on: true, icon: Sparkles, label: 'Auto-assemble', desc: 'Spawn a panel of specialists modeled to the topic' },
    { on: false, icon: MousePointerClick, label: 'Pick agents', desc: 'Choose from your already-running agents' },
  ]

  return (
    <div className="@container space-y-6">
      <div className="grid items-start gap-6 @5xl:grid-cols-2">
        {/* ── Left: what, and how they talk ── */}
        <div className="space-y-6">
          <Card title="Topic" icon={FileText}>
            <div className="space-y-4">
              <Field label="Topic / Task *">
                <textarea
                  value={topic}
                  onChange={e => setTopic(e.target.value)}
                  placeholder="What should the council discuss or work on?"
                  rows={4}
                  className={`${inputCls} resize-none leading-relaxed`}
                />
              </Field>
              <Field label="Title (optional)">
                <input
                  value={title}
                  onChange={e => setTitle(e.target.value)}
                  placeholder="Council session title..."
                  className={inputCls}
                />
              </Field>
            </div>
          </Card>

          <Card title="Session Type" icon={MessagesSquare} className="@container">
            <div className="grid grid-cols-2 gap-2.5 @md:grid-cols-4">
              {SESSION_TYPES.map(t => {
                const on = sessionType === t.id
                return (
                  <button
                    key={t.id}
                    type="button"
                    onClick={() => setSessionType(t.id)}
                    aria-pressed={on}
                    className={`${CHOICE} flex min-w-0 flex-col p-3 ${on ? CHOICE_ON : CHOICE_OFF}`}
                  >
                    <span className="flex items-center justify-between">
                      <span className={`flex h-8 w-8 items-center justify-center rounded-lg ring-1 ${TONE_SQUARE[TYPE_LOOK[t.id].tone]}`}>
                        <t.icon className="h-4 w-4" />
                      </span>
                      {on && (
                        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-violet-600 text-white">
                          <Check className="h-3 w-3" />
                        </span>
                      )}
                    </span>
                    <span className="mt-2.5 block text-sm font-semibold leading-snug text-zinc-100">{t.label}</span>
                    <span className="mt-0.5 block text-[11px] leading-snug text-zinc-500">{t.desc}</span>
                  </button>
                )
              })}
            </div>
          </Card>

          <Card title="Discussion" icon={SlidersHorizontal} className="@container">
            <div className="space-y-6">
              <div>
                <label className={labelCls}>Response Verbosity</label>
                <div role="radiogroup" aria-label="Response Verbosity" className="grid grid-cols-2 gap-2 @md:grid-cols-5">
                  {VERBOSITY_OPTIONS.map(v => {
                    const on = verbosity === v.id
                    return (
                      <button
                        key={v.id}
                        type="button"
                        role="radio"
                        aria-checked={on}
                        onClick={() => setVerbosity(v.id)}
                        className={`${CHOICE} rounded-xl px-3 py-2.5 ${on ? CHOICE_ON : CHOICE_OFF}`}
                      >
                        <div className={`text-sm font-medium ${on ? 'text-violet-700 dark:text-violet-200' : 'text-zinc-200'}`}>{v.label}</div>
                        <div className="mt-0.5 text-[11px] leading-snug text-zinc-500">{v.desc}</div>
                      </button>
                    )
                  })}
                </div>
              </div>

              <div>
                <label className={`${labelCls} flex items-center gap-2`}>
                  Max Rounds <Value>{maxRounds}</Value>
                </label>
                <input
                  type="range"
                  min={1}
                  max={20}
                  value={maxRounds}
                  onChange={e => setMaxRounds(parseInt(e.target.value))}
                  className={RANGE}
                />
                <Ticks marks={[1, 10, 20]} />
              </div>

              {/* First Speaker */}
              {agents.length >= 2 && (
                <Field label="First Speaker">
                  <select
                    value={firstSpeaker}
                    onChange={e => setFirstSpeaker(e.target.value)}
                    className={inputCls}
                  >
                    <option value="random">Random</option>
                    {agents.filter(a => a.id !== oldMan?.id).map(a => (
                      <option key={a.id} value={a.id}>{a.name}</option>
                    ))}
                  </select>
                </Field>
              )}
            </div>
          </Card>
        </div>

        {/* ── Right: who takes part, and what they read ── */}
        <div className="space-y-6">
          <Card title="Panel" icon={Users} className="@container">
            <div className="space-y-4">
              <div className="grid gap-2.5 @sm:grid-cols-2">
                {panelModes.map(m => {
                  const on = autoAssemble === m.on
                  return (
                    <button
                      key={m.label}
                      type="button"
                      onClick={() => setAutoAssemble(m.on)}
                      aria-pressed={on}
                      className={`${CHOICE} flex items-start gap-3 p-3.5 ${on ? CHOICE_ON : CHOICE_OFF}`}
                    >
                      <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ring-1 ${TONE_SQUARE[on ? 'violet' : 'zinc']}`}>
                        <m.icon className="h-[18px] w-[18px]" />
                      </span>
                      <span className="min-w-0">
                        <span className="block text-sm font-semibold text-zinc-100">{m.label}</span>
                        <span className="mt-0.5 block text-xs leading-snug text-zinc-500">{m.desc}</span>
                      </span>
                    </button>
                  )
                })}
              </div>

              {autoAssemble ? (
                <div className="space-y-5 rounded-xl border border-zinc-800 bg-zinc-950/40 p-4">
                  {selectedArchetypes.length === 0 ? (
                    <div>
                      <label className={`${labelCls} flex items-center gap-2`}>
                        Panel size <Value>{maxAgents}</Value> specialists
                      </label>
                      <input
                        type="range"
                        min={2}
                        max={6}
                        value={maxAgents}
                        onChange={e => setMaxAgents(parseInt(e.target.value))}
                        className={RANGE}
                      />
                      <Ticks marks={[2, 4, 6]} />
                      <p className="mt-3 text-xs leading-relaxed text-zinc-500">
                        A router picks complementary archetypes for this topic and spawns each as a
                        fresh agent. They're torn down when you delete the session.
                      </p>
                    </div>
                  ) : (
                    <p className="text-xs leading-relaxed text-zinc-500">
                      Using <span className="font-semibold text-violet-700 dark:text-violet-300">{selectedArchetypes.length}</span> hand-picked
                      specialist{selectedArchetypes.length !== 1 ? 's' : ''}. Each is spawned fresh and
                      briefed for this topic; torn down when you delete the session.
                    </p>
                  )}

                  {/* Optional: hand-pick the specialists instead of auto-routing. */}
                  {archetypes.length > 0 && (
                    <div>
                      <div className="mb-2 flex items-center justify-between gap-2">
                        <label className="text-xs font-medium text-zinc-400">
                          Specialists {selectedArchetypes.length > 0
                            ? <span className="font-normal text-zinc-500">({selectedArchetypes.length} selected)</span>
                            : <span className="font-normal text-zinc-500">(optional — leave empty to auto-pick)</span>}
                        </label>
                        {selectedArchetypes.length > 0 && (
                          <button
                            type="button"
                            onClick={() => setSelectedArchetypes([])}
                            className="rounded-lg px-2 py-0.5 text-xs text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
                          >
                            Clear
                          </button>
                        )}
                      </div>
                      <div className="relative mb-2">
                        <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-500" />
                        <input
                          value={teamQuery}
                          onChange={(e) => setTeamQuery(e.target.value)}
                          placeholder="Filter by role, family or skill…"
                          aria-label="Filter archetypes"
                          className={`${inputCls} pl-8`}
                        />
                      </div>
                      <div className="grid max-h-56 gap-1.5 overflow-auto pr-1 @md:grid-cols-2">
                        {archetypes.filter((a) => selectedArchetypes.includes(a.id) || archetypeMatches(a, teamQuery)).map(a => {
                          const on = selectedArchetypes.includes(a.id)
                          return (
                            <button
                              key={a.id}
                              type="button"
                              onClick={() => toggleArchetype(a.id)}
                              className={`flex w-full min-w-0 items-start gap-2.5 rounded-xl border px-2.5 py-2 text-left transition-colors ${
                                on
                                  ? 'border-violet-500/50 bg-violet-500/[0.08]'
                                  : 'border-zinc-800 bg-zinc-900/40 hover:border-zinc-700 hover:bg-zinc-800/60'
                              }`}
                            >
                              <span className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-[5px] ${
                                on ? 'bg-violet-600 text-white' : 'border border-zinc-600'
                              }`}>
                                {on && <Check className="h-3 w-3" />}
                              </span>
                              <span className="min-w-0">
                                <span className="block truncate text-xs font-medium text-zinc-200">{a.role}</span>
                                <span className="block truncate text-[11px] text-zinc-500">{a.family}</span>
                              </span>
                            </button>
                          )
                        })}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-2.5">
                  <label className="flex flex-wrap items-center gap-2 text-xs font-medium text-zinc-400">
                    Select Agents ({agents.length} selected)
                    {oldMan && (
                      <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/10 px-2 py-0.5 text-[11px] font-medium text-amber-700 dark:text-amber-300">
                        <Crown className="h-3 w-3" /> Moderator mode (Old Man detected)
                      </span>
                    )}
                    {!oldMan && agents.length >= 2 && (
                      <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[11px] font-medium text-zinc-400">Round-robin mode</span>
                    )}
                  </label>
                  <AgentPicker selected={agents} onChange={setAgents} />
                </div>
              )}
            </div>
          </Card>

          <Card
            title="Attachments"
            icon={Paperclip}
            hint={files.length > 0 ? `${files.length} file${files.length !== 1 ? 's' : ''}` : undefined}
          >
            <div
              onDragOver={e => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`flex cursor-pointer flex-col items-center gap-1.5 rounded-xl border border-dashed px-4 py-6 text-center transition-colors ${
                dragOver
                  ? 'border-violet-500 bg-violet-500/10'
                  : 'border-zinc-700 bg-zinc-950/40 hover:border-zinc-600 hover:bg-zinc-800/40'
              }`}
            >
              <span className="mb-1 flex h-9 w-9 items-center justify-center rounded-xl bg-zinc-800 text-zinc-400">
                <Upload className="h-4 w-4" />
              </span>
              <span className="text-sm font-medium text-zinc-300">Drop files here or click to browse</span>
              <span className="text-xs text-zinc-500">Files will be uploaded to all agents at council start</span>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              className="hidden"
              onChange={e => { if (e.target.files?.length) { addFiles(e.target.files); e.target.value = '' } }}
            />
            {files.length > 0 && (
              <div className="mt-3 space-y-1.5">
                {files.map(f => (
                  <div key={f.name} className="flex items-center gap-2.5 rounded-xl border border-zinc-800 bg-zinc-950/40 px-3 py-2 text-sm">
                    <Paperclip className="h-3.5 w-3.5 flex-shrink-0 text-zinc-500" />
                    <span className="min-w-0 flex-1 truncate text-zinc-200">{f.name}</span>
                    <span className="flex-shrink-0 text-xs tabular-nums text-zinc-500">{formatSize(f.size)}</span>
                    <button
                      onClick={e => { e.stopPropagation(); removeFile(f.name) }}
                      className="flex-shrink-0 rounded-lg p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Actions */}
      <div className="sticky bottom-0 z-10 -mx-4 -mb-6 border-t border-zinc-800 bg-zinc-950/90 px-4 py-3 backdrop-blur wide:-mx-8 wide:-mb-8 wide:px-8 wide:py-4">
        <div className="flex flex-wrap items-center gap-3">
          {!canStart && !submitting && (
            <span className="mr-auto text-xs text-zinc-500">
              {!topic.trim() ? 'Write a topic to start.' : 'Pick at least two agents.'}
            </span>
          )}
          <div className="ml-auto flex items-center gap-2">
            <Button variant="ghost" onClick={onCancel} disabled={submitting}>
              Cancel
            </Button>
            <Button variant="primary" onClick={handleStart} disabled={!canStart} busy={submitting} icon={Play}>
              {submitting ? (autoAssemble ? 'Assembling panel…' : 'Starting…') : 'Start Council'}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
