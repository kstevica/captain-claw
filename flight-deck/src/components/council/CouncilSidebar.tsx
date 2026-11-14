import { useRef, useEffect, useState } from 'react'
import {
  VolumeX, Volume2, Pin, Crown, Loader2, Wrench,
  Activity, Leaf, Paperclip, Play, Plus, ChevronDown, ChevronRight, FolderOpen, Trash2,
} from 'lucide-react'
import { CouncilFileBrowser } from './CouncilFileBrowser'
import { useContainerStore } from '../../stores/containerStore'
import { useProcessStore } from '../../stores/processStore'
import { formatSize } from '../../services/fileTransfer'
import type { CouncilSession, ActivityLogEntry, MemoryRounds } from '../../stores/councilStore'
import { useCouncilStore } from '../../stores/councilStore'

import { suitabilityLabel } from '../../utils/suitability'
import { Button, Switch } from '../console/parts'
import { AgentAvatar, SectionLabel } from './councilLook'

const COGNITIVE_MODE_COLORS: Record<string, { dot: string; label: string }> = {
  neutra:     { dot: 'bg-zinc-500',    label: 'Neutra' },
  ionian:     { dot: 'bg-amber-400',   label: 'Ionian' },
  dorian:     { dot: 'bg-teal-400',    label: 'Dorian' },
  phrygian:   { dot: 'bg-red-400',     label: 'Phrygian' },
  lydian:     { dot: 'bg-violet-400',  label: 'Lydian' },
  mixolydian: { dot: 'bg-orange-400',  label: 'Mixolydian' },
  aeolian:    { dot: 'bg-blue-400',    label: 'Aeolian' },
  locrian:    { dot: 'bg-fuchsia-400', label: 'Locrian' },
}

const TYPE_LABELS: Record<string, string> = {
  debate: 'Debate',
  brainstorm: 'Brainstorm',
  review: 'Review',
  planning: 'Planning',
  interview: 'Interview',
  troubleshoot: 'Troubleshoot',
  critique: 'Critique',
  freeform: 'Freeform',
}

const VERBOSITY_LABELS: Record<string, string> = {
  thought: 'Thought',
  message: 'Message',
  short: 'Short',
  medium: 'Medium',
  long: 'Long',
}

const LOG_TYPE_COLORS: Record<string, string> = {
  tool: 'text-amber-700 dark:text-amber-400',
  status: 'text-zinc-400',
  speaking: 'text-violet-700 dark:text-violet-400',
  done: 'text-emerald-700 dark:text-emerald-400',
  system: 'text-cyan-700 dark:text-cyan-400',
  moderator: 'text-amber-700 dark:text-amber-400',
  error: 'text-red-700 dark:text-red-400',
  connect: 'text-emerald-700 dark:text-emerald-400',
  disconnect: 'text-red-700 dark:text-red-400',
  narration: 'text-sky-700 dark:text-sky-300 italic',
  reasoning: 'text-zinc-500 italic',
}

const STATUS_TEXT: Record<string, string> = {
  active: 'text-emerald-700 dark:text-emerald-400',
  synthesizing: 'text-violet-700 dark:text-violet-400',
  concluded: 'text-zinc-400',
}

/** One fact about the session: a small tile. */
function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0 rounded-xl border border-zinc-800 bg-zinc-950/40 px-3 py-1.5">
      <div className="text-[11px] font-medium text-zinc-500">{label}</div>
      <div className="truncate text-[13px] font-semibold text-zinc-100">{children}</div>
    </div>
  )
}

/** A switch with what it does on the left. */
function ToggleRow({ icon: Icon, label, title, on, onChange }: {
  icon?: typeof Play
  label: string
  title?: string
  on: boolean
  onChange: (on: boolean) => void
}) {
  return (
    <div className="flex items-center justify-between gap-3 py-[3px]" title={title}>
      <span className="flex items-center gap-2 text-sm text-zinc-300">
        {Icon && <Icon className="h-3.5 w-3.5 text-zinc-500" />} {label}
      </span>
      <Switch on={on} label={label} onChange={onChange} />
    </div>
  )
}

const EXTEND_OPTIONS = [3, 5, 10, 20]

interface CouncilSidebarProps {
  session: CouncilSession
  speaking: string
  activityLog: ActivityLogEntry[]
  autoAdvanceCountdown: number
  onMute: (agentId: string, muted: boolean) => void
  onPinClick: (messageId: number) => void
}

export function CouncilSidebar({
  session, speaking, activityLog, autoAdvanceCountdown,
  onMute, onPinClick,
}: CouncilSidebarProps) {
  const pinnedMessages = session.messages.filter(m => m.pinned)
  const logContainerRef = useRef<HTMLDivElement>(null)
  const [showExtend, setShowExtend] = useState(false)
  const [infoCollapsed, setInfoCollapsed] = useState(false)
  const [showFileBrowser, setShowFileBrowser] = useState(false)

  // Look up cognitive mode & eco mode per agent from desktop stores
  const containerCogModes = useContainerStore(s => s.cognitiveModeOverrides)
  const containerEcoModes = useContainerStore(s => s.ecoModeOverrides)
  const processCogModes = useProcessStore(s => s.cognitiveModeOverrides)
  const processEcoModes = useProcessStore(s => s.ecoModeOverrides)

  const getAgentModes = (agentId: string) => {
    const cogMode = containerCogModes[agentId] || processCogModes[agentId] || 'neutra'
    const ecoMode = containerEcoModes[agentId] ?? processEcoModes[agentId] ?? false
    return { cogMode, ecoMode }
  }

  const store = useCouncilStore

  // Auto-scroll activity log when anything is active or new entries arrive
  useEffect(() => {
    const el = logContainerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [activityLog.length, speaking])

  const isActive = session.status === 'active'
  const isConcluded = session.status === 'concluded'

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-zinc-900/20">
      {/* ─── Session Info ─── */}
      <div className="shrink-0 border-b border-zinc-800">
        <button
          onClick={() => setInfoCollapsed(!infoCollapsed)}
          aria-expanded={!infoCollapsed}
          className="flex w-full items-center gap-1.5 px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-zinc-500 transition-colors hover:bg-zinc-800/30 hover:text-zinc-300"
        >
          {infoCollapsed ? <ChevronRight className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          Session Info
          <span className="ml-auto text-xs font-normal normal-case tracking-normal text-zinc-500">
            {TYPE_LABELS[session.sessionType]} · {VERBOSITY_LABELS[session.verbosity]}
          </span>
        </button>

        {!infoCollapsed && (
          <div className="space-y-3 px-4 pb-4">
            {/* Compact info grid */}
            <div className="grid grid-cols-3 gap-2">
              <Fact label="Round">
                <span className="tabular-nums">{session.currentRound} / {session.maxRounds}</span>
                {session.extensions.length > 0 && (
                  <span className="ml-1 text-[11px] font-normal text-amber-700 dark:text-amber-400">
                    (was {session.originalMaxRounds})
                  </span>
                )}
              </Fact>
              <Fact label="Mode">
                {session.moderatorMode === 'moderator' ? 'Moderator' : 'Round-Robin'}
              </Fact>
              <Fact label="Status">
                <span className={`capitalize ${STATUS_TEXT[session.status] || 'text-zinc-500'}`}>{session.status}</span>
              </Fact>
            </div>

            {/* ─── Controls ─── */}
            <div className="rounded-xl border border-zinc-800 bg-zinc-950/40 px-3 py-1.5">
              {/* Memory selector */}
              <div className="flex items-center justify-between gap-3 py-[3px]">
                <span className="text-sm text-zinc-300">Memory</span>
                <select
                  value={session.memoryRounds}
                  onChange={e => store.getState().setMemoryRounds(Number(e.target.value) as MemoryRounds)}
                  className="rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-0.5 text-xs text-zinc-200 focus:border-violet-500/60 focus:outline-none"
                >
                  <option value={5}>5 rounds</option>
                  <option value={10}>10 rounds</option>
                  <option value={20}>20 rounds</option>
                  <option value={30}>30 rounds</option>
                  <option value={0}>Indefinite</option>
                </select>
              </div>

              {/* Auto-advance toggle */}
              {(isActive || isConcluded) && (
                <ToggleRow
                  icon={Play}
                  label="Auto-advance"
                  on={session.autoAdvance}
                  onChange={() => store.getState().setAutoAdvance(!session.autoAdvance)}
                />
              )}

              {/* Allow passing toggle. When OFF, participants are
                  forbidden from using ACTION: pass — the btw template
                  drops the option entirely and the speaker turn loop
                  retries once with a hard directive if the model
                  passes anyway. */}
              <ToggleRow
                label="Allow passing"
                title={
                  session.allowPass
                    ? 'Participants may use ACTION: pass when they have nothing to add'
                    : 'Participants must contribute every round; pass action is disabled'
                }
                on={session.allowPass}
                onChange={() => store.getState().setAllowPass(!session.allowPass)}
              />

              {/* Allow delegation toggle. When OFF (default), agents can't use
                  orchestration tools (flight_deck/task_contract/consult_peer)
                  during their turn — the council is the coordination layer, and
                  sub-delegation causes no-progress loops. Turn ON for sessions
                  where farming out subtasks is the point (e.g. planning). */}
              <ToggleRow
                label="Allow delegation"
                title={
                  session.allowDelegation
                    ? 'Agents may delegate / orchestrate during their turn'
                    : 'Agents contribute directly; delegation tools are disabled per turn'
                }
                on={session.allowDelegation}
                onChange={() => store.getState().setAllowDelegation(!session.allowDelegation)}
              />
            </div>

            {/* Auto-advance countdown indicator */}
            {autoAdvanceCountdown > 0 && (
              <div className="flex items-center gap-2 rounded-xl border border-violet-500/25 bg-violet-500/[0.07] px-3 py-2">
                <span className="h-2 w-2 animate-pulse rounded-full bg-violet-500" />
                <span className="text-xs font-medium text-violet-700 dark:text-violet-300">
                  Next round in {autoAdvanceCountdown}s
                </span>
                <button
                  onClick={() => store.getState().cancelAutoAdvance()}
                  className="ml-auto rounded-lg border border-violet-500/30 px-2 py-0.5 text-xs font-medium text-violet-700 transition-colors hover:bg-violet-500/15 dark:text-violet-300"
                >
                  Stop
                </button>
              </div>
            )}

            {/* Extend rounds · browse agent files */}
            <div>
              <div className="flex gap-2">
                {(isActive || isConcluded) && (
                  <button
                    onClick={() => setShowExtend(!showExtend)}
                    aria-expanded={showExtend}
                    className="flex min-w-0 flex-1 items-center justify-between gap-1.5 rounded-xl border border-zinc-700 bg-zinc-900/60 px-3 py-1.5 text-[13px] font-medium text-zinc-200 transition-colors hover:border-zinc-600 hover:bg-zinc-800"
                  >
                    <span className="flex min-w-0 items-center gap-1.5">
                      <Plus className="h-4 w-4 shrink-0 text-zinc-500" /> <span className="truncate">Extend Rounds</span>
                    </span>
                    <ChevronDown className={`h-4 w-4 shrink-0 text-zinc-500 transition-transform ${showExtend ? 'rotate-180' : ''}`} />
                  </button>
                )}
                <Button onClick={() => setShowFileBrowser(true)} icon={FolderOpen} className="min-w-0 flex-1 px-3! py-1.5! text-[13px]!">
                  <span className="truncate">Browse Agent Files</span>
                </Button>
              </div>
              {(isActive || isConcluded) && showExtend && (
                <div className="mt-2 grid grid-cols-4 gap-1.5">
                  {EXTEND_OPTIONS.map(n => (
                    <button
                      key={n}
                      onClick={() => { store.getState().extendRounds(n); setShowExtend(false) }}
                      className="rounded-lg border border-zinc-800 bg-zinc-950 py-1.5 text-xs font-semibold tabular-nums text-zinc-300 transition-colors hover:border-violet-500/50 hover:bg-violet-500/10 hover:text-violet-700 dark:hover:text-violet-200"
                    >
                      +{n}
                    </button>
                  ))}
                </div>
              )}
              {(isActive || isConcluded) && session.extensions.length > 0 && (
                <div className="mt-1.5 text-[11px] text-zinc-500">
                  Extended: {session.extensions.map(e => `+${e}`).join(', ')} from original {session.originalMaxRounds}
                </div>
              )}
            </div>

            {/* Attached files */}
            {session.fileRefs && session.fileRefs.length > 0 && (
              <div className="space-y-2">
                <SectionLabel icon={Paperclip}>Files ({session.fileRefs.length})</SectionLabel>
                <div className="space-y-1">
                  {session.fileRefs.map((f, i) => (
                    <div key={i} className="flex items-center gap-2 rounded-lg bg-zinc-950/40 px-2.5 py-1.5 text-xs text-zinc-300">
                      <Paperclip className="h-3 w-3 shrink-0 text-zinc-500" />
                      <span className="min-w-0 flex-1 truncate">{f.name}</span>
                      <span className="shrink-0 tabular-nums text-zinc-500">{formatSize(f.size)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        )}
      </div>

      {/* ─── Agents ─── */}
      <div className="shrink-0 space-y-2.5 border-b border-zinc-800 px-4 py-3">
        <SectionLabel
          aside={
            <>
              {session.spawnedSlugs.length > 0 && (
                <span
                  className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
                    session.agentsDisposed
                      ? 'bg-zinc-800 text-zinc-500'
                      : 'bg-amber-500/10 text-amber-700 dark:text-amber-300'
                  }`}
                  title="This panel was auto-assembled — these agents are temporary"
                >
                  {session.agentsDisposed ? 'Disposed' : 'Temporary'}
                </span>
              )}
              {session.spawnedSlugs.length > 0 && !session.agentsDisposed && (
                <button
                  onClick={() => {
                    if (window.confirm(`Dispose ${session.spawnedSlugs.length} temporary agent(s)? They were spawned for this council and will be removed from the fleet. The transcript and synthesis are kept.`)) {
                      store.getState().disposeAgents()
                    }
                  }}
                  className="flex items-center gap-1 rounded-lg border border-zinc-700 px-2 py-0.5 text-[11px] font-medium text-zinc-400 transition-colors hover:border-red-500/40 hover:bg-red-500/10 hover:text-red-600 dark:hover:text-red-400"
                  title="Remove the temporary agents spawned for this council"
                >
                  <Trash2 className="h-3 w-3" /> Dispose
                </button>
              )}
            </>
          }
        >
          Agents ({session.agents.length})
        </SectionLabel>
        <div className="space-y-1.5">
          {session.agents.map(agent => {
            const agentMsgs = session.messages.filter(
              m => m.agentId === agent.id && m.role === 'agent',
            )
            const avgSuitability = agentMsgs.length > 0
              ? agentMsgs.reduce((sum, m) => sum + m.suitability, 0) / agentMsgs.length
              : 0
            const hasMsgs = agentMsgs.length > 0
            const isSpeaking = speaking === agent.id
            const isMod = agent.id === session.moderatorAgentId
            const isTemp = session.spawnedSlugs.includes(agent.id)

            const { cogMode, ecoMode } = getAgentModes(agent.id)
            const modeInfo = COGNITIVE_MODE_COLORS[cogMode] || COGNITIVE_MODE_COLORS.neutra

            return (
              <div
                key={agent.id}
                className={`rounded-xl border px-2.5 py-2 transition-colors ${
                  isSpeaking ? 'border-violet-500/40 bg-violet-500/[0.07]' : 'border-zinc-800 bg-zinc-950/40'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <AgentAvatar name={agent.name} id={agent.id} size="sm">
                    {/* Connection indicator */}
                    <span
                      className={`absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full ring-2 ring-zinc-900 ${agent.connected ? 'bg-emerald-500' : 'bg-zinc-600'}`}
                      title={agent.connected ? 'Connected' : 'Not connected'}
                    />
                  </AgentAvatar>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="truncate text-sm font-medium text-zinc-100">{agent.name}</span>
                      {isTemp && (
                        <span
                          className="shrink-0 rounded-full bg-amber-500/10 px-1.5 text-[10px] font-medium uppercase text-amber-700 dark:text-amber-300"
                          title="Temporary — spawned for this council"
                        >
                          temp
                        </span>
                      )}
                      {isMod && <Crown className="h-3.5 w-3.5 shrink-0 text-amber-600 dark:text-amber-400" />}
                      {isSpeaking && <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-violet-600 dark:text-violet-400" />}
                      {/* Cognitive mode dot */}
                      {cogMode !== 'neutra' && (
                        <span
                          className={`h-2 w-2 shrink-0 rounded-full ${modeInfo.dot}`}
                          title={`${modeInfo.label} mode`}
                        />
                      )}
                      {/* Eco mode indicator */}
                      {ecoMode && (
                        <span title="Eco mode" className="shrink-0">
                          <Leaf className="h-3 w-3 text-emerald-600 dark:text-emerald-400" />
                        </span>
                      )}
                    </div>
                    {hasMsgs && (
                      <div
                        className="mt-1 flex items-center gap-1.5"
                        title={`Average self-rated suitability for the topic across ${agent.name}'s ${agentMsgs.length} contribution${agentMsgs.length === 1 ? '' : 's'}. Each agent reports a SUITABILITY score at the top of every turn; the moderator uses it to pick who speaks next.`}
                      >
                        <div className="h-1 w-12 overflow-hidden rounded-full bg-zinc-800">
                          <div
                            className="h-full rounded-full bg-violet-500"
                            style={{ width: `${avgSuitability * 100}%` }}
                          />
                        </div>
                        <span className="text-[11px] tabular-nums text-zinc-500">
                          {suitabilityLabel(avgSuitability)} ({Math.round(avgSuitability * 100)}%)
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Mute toggle */}
                  <button
                    onClick={() => onMute(agent.id, !agent.muted)}
                    className={`shrink-0 rounded-lg p-1.5 transition-colors ${
                      agent.muted ? 'bg-red-500/10 text-red-600 dark:text-red-400' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200'
                    }`}
                    title={agent.muted ? 'Unmute' : 'Mute'}
                  >
                    {agent.muted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
                  </button>
                </div>

                {/* Tool activity — shown when agent is speaking */}
                {isSpeaking && agent.statusText && (
                  <div className="mt-1.5 truncate pl-[42px] text-[11px] text-amber-700 dark:text-amber-400">
                    {agent.statusText}
                  </div>
                )}
                {isSpeaking && agent.toolHistory.length > 0 && (
                  <div className="mt-1 flex flex-wrap gap-1 pl-[42px]">
                    {agent.toolHistory.map((tool, i) => (
                      <span
                        key={`${tool}-${i}`}
                        className="inline-flex items-center gap-1 rounded-md bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400"
                      >
                        <Wrench className="h-2.5 w-2.5" />{tool}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* Pinned */}
        {pinnedMessages.length > 0 && (
          <div className="space-y-2 pt-2">
            <SectionLabel icon={Pin}>Pinned ({pinnedMessages.length})</SectionLabel>
            <div className="max-h-28 space-y-1 overflow-auto">
              {pinnedMessages.map(m => (
                <button
                  key={m.id}
                  onClick={() => onPinClick(m.id)}
                  className="line-clamp-1 w-full rounded-lg bg-zinc-950/40 px-2.5 py-1.5 text-left text-xs text-zinc-300 transition-colors hover:bg-zinc-800"
                >
                  <span className="font-medium text-violet-700 dark:text-violet-400">{m.agentName}:</span>{' '}
                  {m.content.slice(0, 80)}
                </button>
              ))}
            </div>
          </div>
        )}

      </div>

      {/* ─── Activity Log ─── */}
      <div className="flex min-h-[16rem] flex-1 flex-col">
        <div className="shrink-0 px-4 pb-2 pt-3">
          <SectionLabel icon={Activity} aside={<span className="text-[11px] tabular-nums text-zinc-500">{activityLog.length}</span>}>
            Activity Log
          </SectionLabel>
        </div>
        <div ref={logContainerRef} className="mx-4 mb-4 min-h-0 flex-1 overflow-auto rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 font-mono">
          {activityLog.length === 0 && (
            <div className="flex h-full items-center justify-center text-[11px] text-zinc-500">
              Activity will appear here...
            </div>
          )}
          {activityLog.map((entry, i) => {
            const time = new Date(entry.timestamp).toLocaleTimeString('en-US', {
              hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit',
            })
            return (
              <div key={i} className="flex gap-2 py-0.5 text-[11px] leading-snug">
                <span className="shrink-0 tabular-nums text-zinc-600">{time}</span>
                <span className={`shrink-0 ${LOG_TYPE_COLORS[entry.type] || 'text-zinc-400'}`}>
                  {entry.agentName || 'System'}
                </span>
                <span className="truncate text-zinc-500" title={entry.detail}>{entry.detail}</span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Council File Browser modal */}
      {showFileBrowser && (
        <CouncilFileBrowser
          agents={session.agents}
          sessionCreatedAt={session.createdAt}
          sessionConcludedAt={session.concludedAt}
          onClose={() => setShowFileBrowser(false)}
        />
      )}
    </div>
  )
}
