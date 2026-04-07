import { useRef, useEffect, useState } from 'react'
import {
  Wifi, WifiOff, VolumeX, Volume2, Pin, Crown, Loader2, Wrench,
  Activity, Leaf, Paperclip, Play, Plus, ChevronDown, ChevronRight, FolderOpen,
} from 'lucide-react'
import { VotingPanel } from './VotingPanel'
import { CouncilFileBrowser } from './CouncilFileBrowser'
import { useContainerStore } from '../../stores/containerStore'
import { useProcessStore } from '../../stores/processStore'
import { formatSize } from '../../services/fileTransfer'
import type { CouncilSession, ActivityLogEntry, MemoryRounds } from '../../stores/councilStore'
import { useCouncilStore } from '../../stores/councilStore'

import { suitabilityLabel } from '../../utils/suitability'

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
  tool: 'text-amber-400',
  status: 'text-zinc-400',
  speaking: 'text-violet-400',
  done: 'text-emerald-400',
  system: 'text-cyan-400',
  moderator: 'text-amber-400',
  error: 'text-red-400',
  connect: 'text-emerald-400',
  disconnect: 'text-red-400',
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
    <div className="flex h-full flex-col border-l border-zinc-700/50 bg-zinc-900/30">
      {/* ─── Session Info ─── */}
      <div className="shrink-0 border-b border-zinc-700/50">
        <button
          onClick={() => setInfoCollapsed(!infoCollapsed)}
          className="flex w-full items-center gap-1.5 px-3 py-2 text-xs font-medium uppercase tracking-wider text-zinc-400 hover:bg-zinc-800/30"
        >
          {infoCollapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          Session Info
          <span className="ml-auto text-[10px] font-normal normal-case text-zinc-600">
            {TYPE_LABELS[session.sessionType]} · {VERBOSITY_LABELS[session.verbosity]}
          </span>
        </button>

        {!infoCollapsed && (
          <div className="space-y-3 px-3 pb-3">
            {/* Compact info grid */}
            <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs">
              <div className="flex justify-between">
                <span className="text-zinc-500">Round</span>
                <span className="text-zinc-300">
                  {session.currentRound} / {session.maxRounds}
                  {session.extensions.length > 0 && (
                    <span className="ml-1 text-[10px] text-amber-400/70">
                      (was {session.originalMaxRounds})
                    </span>
                  )}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">Mode</span>
                <span className="text-zinc-300">
                  {session.moderatorMode === 'moderator' ? 'Moderator' : 'Round-Robin'}
                </span>
              </div>
              <div className="flex justify-between col-span-2">
                <span className="text-zinc-500">Status</span>
                <span className={`font-medium ${
                  session.status === 'active' ? 'text-emerald-400' :
                  session.status === 'synthesizing' ? 'text-violet-400' :
                  session.status === 'concluded' ? 'text-zinc-400' :
                  'text-zinc-500'
                }`}>{session.status}</span>
              </div>
            </div>

            {/* ─── Controls row ─── */}
            <div className="space-y-2">
              {/* Memory selector */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-zinc-500">Memory</span>
                <select
                  value={session.memoryRounds}
                  onChange={e => store.getState().setMemoryRounds(Number(e.target.value) as MemoryRounds)}
                  className="rounded border border-zinc-700/50 bg-zinc-800/50 px-1.5 py-0.5 text-[11px] text-zinc-300 focus:outline-none focus:border-violet-500/50"
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
                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-1.5 text-zinc-500">
                    <Play className="h-3 w-3" /> Auto-advance
                  </span>
                  <button
                    onClick={() => store.getState().setAutoAdvance(!session.autoAdvance)}
                    className={`relative h-5 w-9 rounded-full transition-colors ${
                      session.autoAdvance ? 'bg-violet-600' : 'bg-zinc-700'
                    }`}
                  >
                    <div className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform ${
                      session.autoAdvance ? 'translate-x-4' : 'translate-x-0.5'
                    }`} />
                  </button>
                </div>
              )}

              {/* Auto-advance countdown indicator */}
              {autoAdvanceCountdown > 0 && (
                <div className="flex items-center gap-2 rounded-md bg-violet-500/10 border border-violet-500/20 px-2 py-1.5">
                  <div className="flex items-center gap-1">
                    <div className="h-2 w-2 rounded-full bg-violet-400 animate-pulse" />
                    <span className="text-[11px] font-medium text-violet-300">
                      Next round in {autoAdvanceCountdown}s
                    </span>
                  </div>
                  <button
                    onClick={() => store.getState().cancelAutoAdvance()}
                    className="ml-auto rounded px-1.5 py-0.5 text-[10px] text-violet-400 hover:bg-violet-500/20 border border-violet-500/30"
                  >
                    Stop
                  </button>
                </div>
              )}

              {/* Extend rounds */}
              {(isActive || isConcluded) && (
                <div className="relative">
                  <button
                    onClick={() => setShowExtend(!showExtend)}
                    className="flex w-full items-center justify-between rounded-md border border-zinc-700/50 bg-zinc-800/30 px-2 py-1.5 text-xs text-zinc-400 hover:bg-zinc-700/30 hover:text-zinc-300 transition-colors"
                  >
                    <span className="flex items-center gap-1.5">
                      <Plus className="h-3 w-3" /> Extend Rounds
                    </span>
                    <ChevronDown className={`h-3 w-3 transition-transform ${showExtend ? 'rotate-180' : ''}`} />
                  </button>
                  {showExtend && (
                    <div className="mt-1 flex gap-1">
                      {EXTEND_OPTIONS.map(n => (
                        <button
                          key={n}
                          onClick={() => { store.getState().extendRounds(n); setShowExtend(false) }}
                          className="flex-1 rounded-md border border-zinc-700/50 bg-zinc-800/50 py-1 text-[11px] font-medium text-zinc-300 hover:bg-violet-500/10 hover:border-violet-500/30 hover:text-violet-300 transition-colors"
                        >
                          +{n}
                        </button>
                      ))}
                    </div>
                  )}
                  {session.extensions.length > 0 && (
                    <div className="mt-1 text-[10px] text-zinc-500">
                      Extended: {session.extensions.map(e => `+${e}`).join(', ')} from original {session.originalMaxRounds}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Attached files */}
            {session.fileRefs && session.fileRefs.length > 0 && (
              <div className="space-y-1">
                <h4 className="flex items-center gap-1 text-[10px] font-medium text-zinc-500 uppercase tracking-wider">
                  <Paperclip className="h-2.5 w-2.5" /> Files ({session.fileRefs.length})
                </h4>
                <div className="space-y-0.5">
                  {session.fileRefs.map((f, i) => (
                    <div key={i} className="flex items-center gap-1.5 text-[10px] text-zinc-400 truncate">
                      <Paperclip className="h-2.5 w-2.5 shrink-0 text-zinc-500" />
                      <span className="truncate">{f.name}</span>
                      <span className="shrink-0 text-zinc-600">{formatSize(f.size)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Browse agent files */}
            <button
              onClick={() => setShowFileBrowser(true)}
              className="flex w-full items-center justify-center gap-1.5 rounded-md border border-zinc-700/50 bg-zinc-800/30 px-2 py-1.5 text-xs text-zinc-400 hover:bg-zinc-700/30 hover:text-zinc-300 transition-colors"
            >
              <FolderOpen className="h-3 w-3" /> Browse Agent Files
            </button>
          </div>
        )}
      </div>

      {/* ─── Agents ─── */}
      <div className="shrink-0 border-b border-zinc-700/50 p-3 space-y-2 max-h-[40%] overflow-auto">
        <h3 className="text-xs font-medium text-zinc-400 uppercase tracking-wider">
          Agents ({session.agents.length})
        </h3>
        <div className="space-y-1">
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

            const { cogMode, ecoMode } = getAgentModes(agent.id)
            const modeInfo = COGNITIVE_MODE_COLORS[cogMode] || COGNITIVE_MODE_COLORS.neutra

            return (
              <div
                key={agent.id}
                className={`rounded-lg px-2 py-1.5 text-xs ${
                  isSpeaking ? 'bg-violet-500/10 border border-violet-500/30' : 'bg-zinc-800/30'
                }`}
              >
                <div className="flex items-center gap-2">
                  {/* Connection indicator */}
                  {agent.connected
                    ? <Wifi className="h-3 w-3 shrink-0 text-emerald-400" />
                    : <WifiOff className="h-3 w-3 shrink-0 text-zinc-500" />
                  }

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1">
                      <span className="truncate text-zinc-200">{agent.name}</span>
                      {isMod && <Crown className="h-3 w-3 text-amber-400 shrink-0" />}
                      {isSpeaking && <Loader2 className="h-3 w-3 text-violet-400 animate-spin shrink-0" />}
                      {/* Cognitive mode dot */}
                      {cogMode !== 'neutra' && (
                        <span
                          className={`h-2 w-2 rounded-full shrink-0 ${modeInfo.dot}`}
                          title={`${modeInfo.label} mode`}
                        />
                      )}
                      {/* Eco mode indicator */}
                      {ecoMode && (
                        <span title="Eco mode" className="shrink-0">
                          <Leaf className="h-2.5 w-2.5 text-emerald-400" />
                        </span>
                      )}
                    </div>
                    {hasMsgs && (
                      <div
                        className="flex items-center gap-1 mt-0.5"
                        title={`Average self-rated suitability for the topic across ${agent.name}'s ${agentMsgs.length} contribution${agentMsgs.length === 1 ? '' : 's'}. Each agent reports a SUITABILITY score at the top of every turn; the moderator uses it to pick who speaks next.`}
                      >
                        <div className="h-1 w-8 rounded-full bg-zinc-700 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-violet-500"
                            style={{ width: `${avgSuitability * 100}%` }}
                          />
                        </div>
                        <span className="text-[10px] text-zinc-500">
                          {suitabilityLabel(avgSuitability)} ({Math.round(avgSuitability * 100)}%)
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Mute toggle */}
                  <button
                    onClick={() => onMute(agent.id, !agent.muted)}
                    className={`shrink-0 rounded p-1 transition-colors ${
                      agent.muted ? 'text-red-400' : 'text-zinc-500 hover:text-zinc-300'
                    }`}
                    title={agent.muted ? 'Unmute' : 'Mute'}
                  >
                    {agent.muted ? <VolumeX className="h-3 w-3" /> : <Volume2 className="h-3 w-3" />}
                  </button>
                </div>

                {/* Tool activity — shown when agent is speaking */}
                {isSpeaking && agent.statusText && (
                  <div className="mt-1 pl-5 text-[10px] text-amber-400 truncate">
                    {agent.statusText}
                  </div>
                )}
                {isSpeaking && agent.toolHistory.length > 0 && (
                  <div className="mt-0.5 pl-5 flex flex-wrap gap-1">
                    {agent.toolHistory.map((tool, i) => (
                      <span
                        key={`${tool}-${i}`}
                        className="inline-flex items-center gap-0.5 rounded bg-zinc-700/50 px-1 py-0.5 text-[9px] text-zinc-400"
                      >
                        <Wrench className="h-2 w-2" />{tool}
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
          <div className="pt-2 space-y-1">
            <h4 className="flex items-center gap-1 text-[10px] font-medium text-zinc-500 uppercase tracking-wider">
              <Pin className="h-2.5 w-2.5" /> Pinned ({pinnedMessages.length})
            </h4>
            <div className="space-y-0.5 max-h-24 overflow-auto">
              {pinnedMessages.map(m => (
                <button
                  key={m.id}
                  onClick={() => onPinClick(m.id)}
                  className="w-full rounded bg-zinc-800/50 px-1.5 py-1 text-left text-[10px] text-zinc-300 hover:bg-zinc-700/50 line-clamp-1"
                >
                  <span className="font-medium text-violet-400">{m.agentName}:</span>{' '}
                  {m.content.slice(0, 80)}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Votes */}
        {session.votes.length > 0 && (
          <div className="pt-2">
            <VotingPanel votes={session.votes} />
          </div>
        )}
      </div>

      {/* ─── Activity Log ─── */}
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        <div className="shrink-0 flex items-center gap-1 px-3 py-2 border-b border-zinc-700/30">
          <Activity className="h-3 w-3 text-zinc-500" />
          <h3 className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Activity Log</h3>
          <span className="text-[10px] text-zinc-600 ml-auto">{activityLog.length}</span>
        </div>
        <div ref={logContainerRef} className="flex-1 overflow-auto px-2 py-1 font-mono">
          {activityLog.length === 0 && (
            <div className="flex items-center justify-center h-full text-[10px] text-zinc-600">
              Activity will appear here...
            </div>
          )}
          {activityLog.map((entry, i) => {
            const time = new Date(entry.timestamp).toLocaleTimeString('en-US', {
              hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit',
            })
            return (
              <div key={i} className="flex gap-1.5 py-0.5 text-[10px] leading-tight">
                <span className="shrink-0 text-zinc-600">{time}</span>
                <span className={`shrink-0 ${LOG_TYPE_COLORS[entry.type] || 'text-zinc-400'}`}>
                  {entry.agentName || 'System'}
                </span>
                <span className="text-zinc-500 truncate">{entry.detail}</span>
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
