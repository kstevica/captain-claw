import { useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle, ArrowLeft, Check, ChevronDown, ChevronRight, CircleDot, CircleStop, FileText, Globe, Info, Loader2,
  Megaphone, PanelRightOpen, Sparkles, Users, Wrench, X,
} from 'lucide-react'
import { useShallow } from 'zustand/react/shallow'
import { useChatStore } from '../../../stores/chatStore'
import {
  LIVE_HELPER, LIVE_TEAM, currentTeam, fmtElapsed, fmtTokens, helperTone, useTeamStore,
  type HelperState, type TeamHelper, type TeamRun, type TeamStep,
} from '../../../stores/teamStore'
import { AgentPicture } from '../AgentPicture'
import { FOCUS } from '../queue/QueueButton'

// Team up (captain_claw/team_up.py) in the chat: the card in the transcript
// where the agent offered the team, the pill in the conversation bar while it
// works, and the Team panel docked beside the chat — every helper live, and
// the person still talking to the agent in the composer next to it.

// ── Small pieces ───────────────────────────────────────────────────────────

function useNow(on: boolean, ms = 1000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!on) return
    const t = setInterval(() => setNow(Date.now()), ms)
    return () => clearInterval(t)
  }, [on, ms])
  return now
}

function elapsedOf(team: TeamRun, now: number): number {
  if (!team.started_at) return 0
  const end = LIVE_TEAM.includes(team.state) ? now / 1000 : team.ended_at || now / 1000
  return end - team.started_at
}

const TEAM_LOOK: Record<TeamRun['state'], { label: string; cls: string }> = {
  proposed: { label: 'Waiting for you', cls: 'bg-violet-500/12 text-violet-700 dark:text-violet-300' },
  running: { label: 'Working', cls: 'bg-violet-500/12 text-violet-700 dark:text-violet-300' },
  finishing: { label: 'Handing over', cls: 'bg-sky-500/12 text-sky-700 dark:text-sky-300' },
  done: { label: 'Done', cls: 'bg-emerald-500/12 text-emerald-700 dark:text-emerald-300' },
  stopped: { label: 'Stopped', cls: 'bg-zinc-800 text-zinc-400' },
  dismissed: { label: 'Not used', cls: 'bg-zinc-800 text-zinc-400' },
  failed: { label: 'Failed', cls: 'bg-red-500/12 text-red-700 dark:text-red-300' },
}

const HELPER_LOOK: Record<HelperState, string> = {
  waiting: 'Waiting', working: 'Working', stopping: 'Stopping', done: 'Done', failed: 'Failed', stopped: 'Stopped',
}

function StateChip({ team }: { team: TeamRun }) {
  const look = TEAM_LOOK[team.state]
  const live = team.state === 'running' || team.state === 'finishing'
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ${look.cls}`}>
      {live && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />}
      {look.label}
    </span>
  )
}

function HelperStateIcon({ state, className = 'h-3.5 w-3.5' }: { state: HelperState; className?: string }) {
  if (state === 'working' || state === 'waiting') return <Loader2 className={`${className} animate-spin text-violet-500`} />
  if (state === 'stopping') return <Loader2 className={`${className} animate-spin text-zinc-500`} />
  if (state === 'done') return <Check className={`${className} text-emerald-500`} />
  if (state === 'failed') return <AlertTriangle className={`${className} text-red-500`} />
  return <CircleStop className={`${className} text-zinc-500`} />
}

/** A helper's face: the agent's own picture, ringed in the helper's colour,
 *  with its number. */
function HelperAvatar({ agentId, agentName, index, size = 'md', dim = false, live = false }: {
  agentId: string; agentName: string; index: number; size?: 'sm' | 'md'; dim?: boolean; live?: boolean
}) {
  const tone = helperTone(index)
  const box = size === 'sm' ? 'h-7 w-7 text-[9px] tracking-tight' : 'h-8 w-8 text-[11px]'
  return (
    <span className="relative shrink-0">
      <span className={`relative flex ${box} items-center justify-center overflow-hidden rounded-full bg-[var(--fd-accent-soft)] font-semibold text-[var(--fd-accent-text)] ring-2 ${tone.ring} ${dim ? 'opacity-60' : ''}`}>
        <AgentPicture agentId={agentId} name={agentName} size="sm" dim={dim} />
      </span>
      <span className={`absolute -bottom-1 -right-1 flex h-3.5 min-w-[14px] items-center justify-center rounded-full px-[3px] text-[9px] font-bold leading-none text-white ring-2 ring-zinc-950 ${tone.dot}`}>
        {index + 1}
      </span>
      {live && <span className={`absolute -left-0.5 -top-0.5 h-2 w-2 animate-ping rounded-full ${tone.dot} opacity-60`} />}
    </span>
  )
}

function BudgetMeter({ team, compact = false }: { team: TeamRun; compact?: boolean }) {
  const pct = team.budget > 0 ? Math.min(100, (team.used / team.budget) * 100) : 0
  const tone = pct >= 100 ? 'bg-red-500' : pct >= 80 ? 'bg-amber-500' : 'bg-violet-500'
  return (
    <div className={compact ? 'min-w-[90px] flex-1' : ''} title={`${fmtTokens(team.used)} of ${fmtTokens(team.budget)} tokens`}>
      {!compact && (
        <div className="mb-1 flex items-center justify-between text-[11px] text-zinc-500">
          <span>Budget</span>
          <span className="font-mono tabular-nums">{fmtTokens(team.used)} / {fmtTokens(team.budget)}</span>
        </div>
      )}
      <div className="h-1.5 overflow-hidden rounded-full bg-zinc-800">
        <div className={`h-full rounded-full transition-[width] duration-700 ease-out ${tone}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

const STEP_ICON: Record<string, typeof Wrench> = {
  tool: Wrench, narration: Sparkles, file: FileText, tell: Megaphone, note: Info, llm: CircleDot,
}

function StepLine({ step, fresh }: { step: TeamStep; fresh?: boolean }) {
  const Icon = STEP_ICON[step.kind] ?? Info
  const tell = step.kind === 'tell'
  return (
    <li className={`flex items-start gap-2 text-xs leading-relaxed ${fresh ? 'fd-rise' : ''} ${
      step.kind === 'narration' ? 'italic text-zinc-400' : tell ? 'text-violet-700 dark:text-violet-300' : step.kind === 'note' ? 'text-zinc-500' : 'text-zinc-300'
    }`}>
      <Icon className={`mt-[3px] h-3 w-3 shrink-0 ${tell ? 'text-violet-500' : 'text-zinc-600'}`} />
      <span className="min-w-0 break-words">{tell ? <><span className="font-medium">From the agent:</span> {step.text}</> : step.text}</span>
    </li>
  )
}

// ── The card in the transcript ─────────────────────────────────────────────

export function TeamCard({ sessionKey, teamId, title, agentId, agentName }: {
  sessionKey: string; teamId: string; title: string; agentId: string; agentName: string
}) {
  const team = useTeamStore((s) => s.byKey[sessionKey]?.[teamId])
  const panelOpen = useTeamStore((s) => s.panel?.key === sessionKey && s.panel.teamId === teamId)
  const openPanel = useTeamStore((s) => s.openPanel)
  const closePanel = useTeamStore((s) => s.closePanel)
  const { teamAction, teamAlone } = useChatStore(useShallow((s) => ({ teamAction: s.teamAction, teamAlone: s.teamAlone })))
  const live = !!team && (team.state === 'running' || team.state === 'finishing')
  const now = useNow(live)

  if (!team) {
    return (
      <div className="my-3 flex items-center gap-2 rounded-2xl border border-zinc-800 bg-zinc-900/30 px-4 py-3 text-xs text-zinc-500">
        <Users className="h-4 w-4 text-zinc-600" />
        <span className="min-w-0 truncate">Team up · <span className="text-zinc-300">{title}</span> — its details went with an agent restart.</span>
      </div>
    )
  }

  if (team.state === 'dismissed') {
    return (
      <div className="my-2 flex items-center gap-2 px-1 text-xs text-zinc-500">
        <Users className="h-3.5 w-3.5 text-zinc-600" />
        <span className="min-w-0 truncate">Team up offered for <span className="text-zinc-400">{team.title}</span> — {agentName} does it alone.</span>
      </div>
    )
  }

  const working = team.helpers.filter((h) => LIVE_HELPER.includes(h.state)).length
  const done = team.helpers.filter((h) => h.state === 'done').length
  const proposed = team.state === 'proposed'
  const pending = team.pending

  return (
    <div
      data-team-card={team.id}
      className={`fd-rise my-3 overflow-hidden rounded-2xl border shadow-sm transition-colors ${
        proposed ? 'border-violet-500/35 bg-gradient-to-b from-violet-500/[0.07] to-zinc-900/40'
          : live ? 'border-violet-500/25 bg-zinc-900/50' : 'border-zinc-800 bg-zinc-900/40'
      }`}
    >
      {live && (
        <div className="relative h-0.5 overflow-hidden bg-violet-500/10">
          <div className="fm-running-bar absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-violet-500 to-transparent" />
        </div>
      )}

      <div className="flex items-start gap-3 px-4 pt-3.5">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-500/15 text-violet-600 dark:text-violet-300">
          <Users className="h-[18px] w-[18px]" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-violet-600 dark:text-violet-300">Team up</span>
            <StateChip team={team} />
          </div>
          <div className="mt-0.5 text-[15px] font-semibold leading-snug text-zinc-100">{team.title}</div>
          <div className="mt-0.5 text-xs text-zinc-500">
            {proposed
              ? <>{team.helpers.length} helpers — copies of {agentName}, working side by side</>
              : <>{team.helpers.length} helpers · {fmtElapsed(elapsedOf(team, now))} · <span className="font-mono tabular-nums">{fmtTokens(team.used)}</span> tokens</>}
          </div>
        </div>
      </div>

      <ul className="mt-3 space-y-0.5 px-2.5">
        {team.helpers.map((h, i) => (
          <li key={h.id} className="flex items-start gap-3 rounded-xl px-2 py-2 transition-colors hover:bg-zinc-800/40">
            <HelperAvatar agentId={agentId} agentName={agentName} index={i} size="sm"
              dim={h.state === 'stopped' || h.state === 'failed'} />
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className={`text-[13px] font-semibold ${helperTone(i).text}`}>{h.name}</span>
                {!proposed && <HelperStateIcon state={h.state} className="h-3 w-3" />}
                {team.browser_holder === h.id && LIVE_HELPER.includes(h.state) && (
                  <Globe className="h-3 w-3 text-sky-500" aria-label="Holds the browser" />
                )}
                {!proposed && h.tokens > 0 && (
                  <span className="ml-auto shrink-0 font-mono text-[11px] tabular-nums text-zinc-500">{fmtTokens(h.tokens)}</span>
                )}
              </div>
              {proposed ? (
                <p className="mt-0.5 line-clamp-2 text-xs leading-relaxed text-zinc-400">{h.goal}</p>
              ) : (
                <p key={h.last} className="fd-fade mt-0.5 truncate text-xs text-zinc-500">
                  {LIVE_HELPER.includes(h.state) ? (h.last || 'Getting started…')
                    : h.state === 'done' ? 'Done — result handed to ' + agentName
                    : h.state === 'failed' ? (h.error || 'Failed')
                    : `Stopped${h.stopped_by === 'budget' ? ' at the budget' : ''}`}
                </p>
              )}
            </div>
          </li>
        ))}
      </ul>

      {team.error && (
        <div className="mx-4 mt-2 flex items-start gap-2 rounded-lg bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{team.error}</span>
        </div>
      )}

      <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-zinc-800/70 px-4 py-3">
        {proposed ? (
          <>
            <p className="w-full text-[11px] leading-relaxed text-zinc-500 sm:w-auto sm:min-w-[180px] sm:flex-1">
              They work in the background — keep chatting with {agentName} meanwhile. Up to {fmtTokens(team.budget)} tokens.
            </p>
            <span className="flex-1 sm:hidden" />
            <button
              onClick={() => teamAlone(sessionKey, team.id)}
              disabled={!!pending}
              className={`h-9 rounded-full px-3.5 text-[13px] font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50 ${FOCUS}`}
            >
              Do it alone
            </button>
            <button
              onClick={() => { teamAction(sessionKey, team.id, 'start'); openPanel(sessionKey, team.id) }}
              disabled={!!pending}
              data-team-start
              className={`flex h-9 items-center gap-1.5 rounded-full bg-violet-600 px-4 text-[13px] font-semibold text-white shadow-sm shadow-violet-900/30 transition-all hover:bg-violet-500 active:scale-[0.98] disabled:opacity-60 ${FOCUS}`}
            >
              {pending === 'start' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Users className="h-4 w-4" />}
              Team up
            </button>
          </>
        ) : (
          <>
            {live ? <BudgetMeter team={team} compact /> : (
              <span className="mr-auto text-xs text-zinc-500">
                {team.state === 'done' ? `${done} of ${team.helpers.length} done · results went to ${agentName}`
                  : team.note ? team.note.charAt(0).toUpperCase() + team.note.slice(1) : TEAM_LOOK[team.state].label}
              </span>
            )}
            <button
              onClick={() => (panelOpen ? closePanel() : openPanel(sessionKey, team.id))}
              className={`flex h-8 items-center gap-1.5 rounded-full border border-zinc-700 px-3 text-xs font-medium text-zinc-300 transition-colors hover:border-zinc-600 hover:bg-zinc-800 ${FOCUS}`}
            >
              <PanelRightOpen className="h-3.5 w-3.5" />
              {panelOpen ? 'Hide details' : live ? (working > 0 ? `Watch ${working} live` : 'Watch live') : 'Details'}
            </button>
            {team.state === 'running' && (
              <button
                onClick={() => teamAction(sessionKey, team.id, 'stop')}
                disabled={pending === 'stop'}
                className={`flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-medium text-zinc-500 transition-colors hover:bg-red-500/10 hover:text-red-600 dark:hover:text-red-400 disabled:opacity-50 ${FOCUS}`}
              >
                <CircleStop className="h-3.5 w-3.5" />
                Stop all
              </button>
            )}
          </>
        )}
      </div>
    </div>
  )
}

// ── The pill in the conversation bar ───────────────────────────────────────

export function TeamPill({ sessionKey, compact = false }: { sessionKey: string; compact?: boolean }) {
  const team = useTeamStore((s) => currentTeam(s.byKey[sessionKey]))
  const panel = useTeamStore((s) => s.panel)
  const openPanel = useTeamStore((s) => s.openPanel)
  const closePanel = useTeamStore((s) => s.closePanel)
  if (!team) return null
  const open = panel?.key === sessionKey && panel.teamId === team.id
  const working = team.helpers.filter((h) => LIVE_HELPER.includes(h.state)).length
  const live = team.state === 'running' || team.state === 'finishing'
  const label = team.state === 'proposed' ? 'offered'
    : team.state === 'finishing' ? 'handing over'
    : live ? `${working} working`
    : TEAM_LOOK[team.state].label.toLowerCase()
  const title = open ? 'Hide the team' : `Team: ${team.title} · ${label}`
  return (
    <button
      onClick={() => (open ? closePanel() : openPanel(sessionKey, team.id))}
      title={title}
      aria-label={title}
      aria-pressed={open}
      data-team-pill
      className={`flex shrink-0 items-center gap-2 rounded-full border font-medium transition-colors ${
        compact ? 'mr-1 h-7 px-2 text-xs' : 'mr-1.5 h-9 px-2.5 text-[13px] wide:px-3'} ${FOCUS} ${
        open ? 'border-violet-500/40 bg-violet-600/15 text-violet-700 dark:text-violet-200'
          : live ? 'border-violet-500/30 text-zinc-200 hover:bg-zinc-800'
          : 'border-zinc-700 text-zinc-300 hover:border-zinc-600 hover:bg-zinc-800'
      }`}
    >
      <span className="flex -space-x-1">
        {team.helpers.slice(0, 4).map((h, i) => (
          <span key={h.id} className={`h-3 w-3 rounded-full ring-2 ring-zinc-950 ${helperTone(i).dot} ${
            h.state === 'working' ? 'animate-pulse' : h.state === 'stopped' || h.state === 'failed' ? 'opacity-40' : ''
          }`} />
        ))}
      </span>
      <span className={compact ? '' : 'hidden wide:inline'}>Team</span>
      <span className={`text-[12px] tabular-nums ${live ? 'text-violet-600 dark:text-violet-300' : 'text-zinc-500'}`}>{label}</span>
    </button>
  )
}

// ── The panel beside the chat ──────────────────────────────────────────────

export function TeamPanel({ sessionKey, agentId, agentName, phone = false }: {
  sessionKey: string; agentId: string; agentName: string; phone?: boolean
}) {
  const panel = useTeamStore((s) => s.panel)
  const team = useTeamStore((s) => (s.panel && s.panel.key === sessionKey ? s.byKey[sessionKey]?.[s.panel.teamId] : undefined))
  const closePanel = useTeamStore((s) => s.closePanel)
  const teamAction = useChatStore((s) => s.teamAction)
  const [tab, setTab] = useState<'helpers' | 'timeline'>('helpers')
  const live = !!team && (team.state === 'running' || team.state === 'finishing')
  const now = useNow(live)

  // Esc closes it (unless a dialog has the key).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !e.defaultPrevented) closePanel() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [closePanel])

  if (!panel || panel.key !== sessionKey || !team) return null

  return (
    <aside
      data-team-panel={team.id}
      className={`fd-drawer flex min-h-0 flex-col bg-zinc-950/95 ${phone ? 'w-full' : 'w-[400px] max-w-[45%] shrink-0 border-l border-zinc-800'}`}
      aria-label={`Team: ${team.title}`}
    >
      <div className="flex items-start gap-2.5 border-b border-zinc-800 px-4 pb-3 pt-3.5">
        {phone && (
          <button onClick={closePanel} className={`-ml-1 rounded-lg p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 ${FOCUS}`} aria-label="Back to the chat">
            <ArrowLeft className="h-4 w-4" />
          </button>
        )}
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-500/15 text-violet-600 dark:text-violet-300">
          <Users className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-semibold text-zinc-100">{team.title}</span>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-zinc-500">
            <StateChip team={team} />
            {team.started_at > 0 && <span className="font-mono tabular-nums">{fmtElapsed(elapsedOf(team, now))}</span>}
            <span>· {team.helpers.length} copies of {agentName}</span>
          </div>
        </div>
        {!phone && (
          <button onClick={closePanel} className={`rounded-lg p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 ${FOCUS}`} aria-label="Close the team panel">
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {team.state !== 'proposed' && (
        <div className="border-b border-zinc-800/70 px-4 py-3">
          <BudgetMeter team={team} />
          {team.warned && team.state === 'running' && (
            <p className="mt-1.5 text-[11px] text-amber-700 dark:text-amber-300">Close to the budget — helpers will be asked to wrap up.</p>
          )}
          {team.note && !live && <p className="mt-1.5 text-[11px] text-zinc-500">{team.note.charAt(0).toUpperCase() + team.note.slice(1)}</p>}
        </div>
      )}

      <div className="flex items-center gap-1 border-b border-zinc-800/70 px-3 py-1.5">
        {(['helpers', 'timeline'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${FOCUS} ${
              tab === t ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {t === 'helpers' ? 'Helpers' : 'Timeline'}
          </button>
        ))}
        {team.state === 'running' && (
          <button
            onClick={() => teamAction(sessionKey, team.id, 'stop')}
            disabled={team.pending === 'stop'}
            className={`ml-auto flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium text-zinc-500 transition-colors hover:bg-red-500/10 hover:text-red-600 dark:hover:text-red-400 disabled:opacity-50 ${FOCUS}`}
          >
            <CircleStop className="h-3.5 w-3.5" />
            Stop all
          </button>
        )}
      </div>

      <div className={`min-h-0 flex-1 ${tab === 'helpers' ? 'overflow-y-auto' : 'flex flex-col'}`}>
        {tab === 'helpers' ? (
          <div className="space-y-3 p-3">
            {team.helpers.map((h, i) => (
              <HelperCard key={h.id} team={team} helper={h} index={i} sessionKey={sessionKey}
                agentId={agentId} agentName={agentName} />
            ))}
            {team.state === 'finishing' && (
              <div className="flex items-center gap-2 rounded-xl border border-sky-500/25 bg-sky-500/10 px-3 py-2.5 text-xs text-sky-700 dark:text-sky-300">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Handing the results to {agentName}, who writes you one reply.
              </div>
            )}
          </div>
        ) : (
          <Timeline team={team} live={live} />
        )}
      </div>

      <div className="border-t border-zinc-800/70 px-4 py-2.5 text-[11px] leading-relaxed text-zinc-500">
        {live ? <>Keep talking to {agentName} in the chat — it sees the team’s progress and can steer or stop a helper.</>
          : team.state === 'proposed' ? <>Tap Team up on the card to start them.</>
          : <>Helpers changed nothing outside {agentName}; anything to send or book is up to {agentName}, with you.</>}
      </div>
    </aside>
  )
}

function HelperCard({ team, helper: h, index, sessionKey, agentId, agentName }: {
  team: TeamRun; helper: TeamHelper; index: number; sessionKey: string; agentId: string; agentName: string
}) {
  const teamAction = useChatStore((s) => s.teamAction)
  const [allSteps, setAllSteps] = useState(false)
  const [goalOpen, setGoalOpen] = useState(false)
  const [resultOpen, setResultOpen] = useState(false)
  const tone = helperTone(index)
  const live = LIVE_HELPER.includes(h.state)
  const steps = useMemo(() => h.steps.filter((s) => s.kind !== 'llm'), [h.steps])
  const shown = allSteps ? steps : steps.slice(-6)
  const lastLlm = live && h.steps.length > 0 && h.steps[h.steps.length - 1].kind === 'llm'

  return (
    <div
      data-helper={h.id}
      className={`rounded-xl border bg-zinc-900/40 transition-colors ${
        live ? 'border-zinc-700/80' : h.state === 'done' ? 'border-emerald-500/20' : 'border-zinc-800'
      }`}
    >
      <div className="flex items-center gap-2.5 px-3 pt-3">
        <HelperAvatar agentId={agentId} agentName={agentName} index={index} live={h.state === 'working'}
          dim={h.state === 'stopped' || h.state === 'failed'} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className={`truncate text-[13px] font-semibold ${tone.text}`}>{h.name}</span>
            {team.browser_holder === h.id && live && (
              <span className="flex items-center gap-0.5 rounded-full bg-sky-500/12 px-1.5 text-[10px] font-medium text-sky-700 dark:text-sky-300">
                <Globe className="h-2.5 w-2.5" /> browser
              </span>
            )}
          </div>
          <div className="flex items-center gap-1.5 text-[11px] text-zinc-500">
            <HelperStateIcon state={h.state} className="h-3 w-3" />
            <span>{HELPER_LOOK[h.state]}{h.state === 'stopped' && h.stopped_by === 'budget' ? ' at the budget' : ''}</span>
            <span>·</span>
            <span className="font-mono tabular-nums">{fmtTokens(h.tokens)}</span>
            {h.tool_count > 0 && <><span>·</span><span>{h.tool_count} tool{h.tool_count === 1 ? '' : 's'}</span></>}
          </div>
        </div>
        {(h.state === 'working' || h.state === 'waiting') && team.state === 'running' && (
          <button
            onClick={() => teamAction(sessionKey, team.id, 'stop', h.id)}
            title={`Stop ${h.name}`}
            aria-label={`Stop ${h.name}`}
            className={`rounded-lg p-1.5 text-zinc-600 transition-colors hover:bg-red-500/10 hover:text-red-500 ${FOCUS}`}
          >
            <CircleStop className="h-4 w-4" />
          </button>
        )}
      </div>

      <button
        onClick={() => setGoalOpen((o) => !o)}
        className={`mx-3 mt-2 block w-[calc(100%-1.5rem)] rounded-md text-left text-xs leading-relaxed text-zinc-400 ${goalOpen ? '' : 'line-clamp-2'} ${FOCUS}`}
        title={goalOpen ? 'Show less' : 'Show the whole part'}
      >
        {h.goal}
      </button>

      {shown.length > 0 && (
        <ul className="mx-3 mt-2.5 space-y-1.5 border-l border-zinc-800 pl-2.5">
          {steps.length > shown.length && !allSteps && (
            <li>
              <button onClick={() => setAllSteps(true)} className="text-[11px] text-zinc-500 hover:text-zinc-300">
                {steps.length - shown.length} earlier step{steps.length - shown.length === 1 ? '' : 's'}…
              </button>
            </li>
          )}
          {/* Stable keys: only a step that just arrived mounts (and rises in). */}
          {shown.map((s, i) => (
            <StepLine key={`${s.t}-${s.kind}-${steps.length - shown.length + i}`} step={s} fresh />
          ))}
        </ul>
      )}

      {live && (
        <div className="mx-3 mt-2.5 overflow-hidden rounded-lg">
          {h.draft ? (
            <div className="bg-zinc-800/40 px-2.5 py-2">
              <div className="mb-1 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                <Loader2 className="h-2.5 w-2.5 animate-spin" /> Writing its result
              </div>
              <p className="fd-fade-bottom line-clamp-3 whitespace-pre-wrap text-xs text-zinc-400">{h.draft.slice(-300)}</p>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-0.5 py-1 text-[11px] text-zinc-500">
              <span className="fd-shimmer h-1.5 w-16 rounded-full" />
              {lastLlm ? 'Thinking…' : h.state === 'stopping' ? 'Stopping…' : 'Working…'}
            </div>
          )}
        </div>
      )}

      {h.state === 'done' && h.result && (
        <div className="mx-3 mt-2.5">
          <button
            onClick={() => setResultOpen((o) => !o)}
            className={`flex items-center gap-1 text-[11px] font-semibold uppercase tracking-wider text-emerald-700 dark:text-emerald-300 ${FOCUS}`}
          >
            {resultOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            Result
          </button>
          <div className={`mt-1 whitespace-pre-wrap rounded-lg bg-zinc-800/40 px-2.5 py-2 text-xs leading-relaxed text-zinc-300 ${
            resultOpen ? 'max-h-[50vh] overflow-y-auto' : 'fd-fade-bottom line-clamp-4'
          }`}>
            {h.result}
          </div>
        </div>
      )}

      {h.error && h.state === 'failed' && (
        <div className="mx-3 mt-2.5 flex items-start gap-1.5 rounded-lg bg-red-500/10 px-2.5 py-2 text-xs text-red-700 dark:text-red-300">
          <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" /> {h.error}
        </div>
      )}

      {h.files.length > 0 && (
        <div className="mx-3 mt-2.5 flex flex-wrap gap-1.5">
          {h.files.slice(-6).map((f) => (
            <span key={f} className="flex max-w-full items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900/60 px-1.5 py-0.5 text-[11px] text-zinc-400">
              <FileText className="h-3 w-3 shrink-0" />
              <span className="truncate">{f.split('/').pop()}</span>
            </span>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between px-3 pb-2.5 pt-2 text-[11px] text-zinc-600">
        <span>{h.step_count} step{h.step_count === 1 ? '' : 's'}{h.model ? ` · ${h.model.split('/').pop()}` : ''}</span>
        {allSteps && steps.length > 6 && (
          <button onClick={() => setAllSteps(false)} className="text-zinc-500 hover:text-zinc-300">Fewer</button>
        )}
      </div>
    </div>
  )
}

function Timeline({ team, live }: { team: TeamRun; live: boolean }) {
  const rows = useMemo(() => {
    const out: Array<{ step: TeamStep; helper: TeamHelper; index: number }> = []
    team.helpers.forEach((helper, index) => {
      for (const step of helper.steps) if (step.kind !== 'llm') out.push({ step, helper, index })
    })
    return out.sort((a, b) => a.step.t - b.step.t)
  }, [team.helpers])
  // Follows the newest step while the team works, unless you scrolled up to read.
  const boxRef = useRef<HTMLDivElement | null>(null)
  const stick = useRef(true)
  useEffect(() => {
    const el = boxRef.current
    if (el && live && stick.current) el.scrollTop = el.scrollHeight
  }, [rows.length, live])
  const start = team.started_at || (rows[0]?.step.t ?? 0)

  if (rows.length === 0) {
    return <p className="p-6 text-center text-xs text-zinc-500">Nothing yet — steps show here as the helpers work.</p>
  }
  return (
    <div
      ref={boxRef}
      onScroll={(e) => {
        const el = e.currentTarget
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60
      }}
      className="min-h-0 flex-1 overflow-y-auto px-3 py-3"
    >
      <ol className="space-y-2">
        {rows.map(({ step, helper, index }, i) => {
          const Icon = STEP_ICON[step.kind] ?? Info
          const tone = helperTone(index)
          return (
            <li key={`${helper.id}-${step.t}-${i}`} className="flex items-start gap-2.5 text-xs">
              <span className="w-10 shrink-0 pt-px text-right font-mono text-[10px] tabular-nums text-zinc-600">
                {fmtElapsed(Math.max(0, step.t - start))}
              </span>
              <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${tone.dot}`} />
              <div className="min-w-0 flex-1">
                <span className={`mr-1.5 font-semibold ${tone.text}`}>{helper.name}</span>
                <Icon className="mr-1 inline h-3 w-3 -translate-y-px text-zinc-600" />
                <span className={step.kind === 'narration' ? 'italic text-zinc-400' : step.kind === 'tell' ? 'text-violet-700 dark:text-violet-300' : 'text-zinc-300'}>
                  {step.kind === 'tell' ? `From the agent: ${step.text}` : step.text}
                </span>
              </div>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
