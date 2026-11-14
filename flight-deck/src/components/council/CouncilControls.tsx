import { useState } from 'react'
import { Send, SkipForward, FileText, CheckCircle, ChevronDown, Download, MessageSquareQuote, Pause, RotateCcw, Trash2 } from 'lucide-react'
import type { CouncilSession } from '../../stores/councilStore'
import { Button } from '../console/parts'

// A bar of its own over the composer: what the round is doing, and what can be done.
const BAR = 'flex flex-wrap items-center gap-x-3 gap-y-2.5 rounded-xl border px-3.5 py-2.5'
const BAR_ZINC = `${BAR} border-zinc-800 bg-zinc-900/60`
// Buttons in a bar: the console's Button, a size smaller.
const SM = 'px-3! py-1.5! text-[13px]!'

interface CouncilControlsProps {
  session: CouncilSession
  speaking: string
  roundRunning: boolean
  generatingArtifact: string
  autoAdvanceCountdown: number
  onInject: (content: string) => void
  onDirectAddress: (agentId: string, content: string) => void
  onAdvanceRound: () => void
  onRestartRound: () => void
  onRequestSynthesis: () => void
  onConclude: () => void
  onCancelAutoAdvance: () => void
  onGenerateTldrs: () => void
  onExportMd: () => void
  onDisposeAgents: () => void
}

export function CouncilControls({
  session, speaking, roundRunning, generatingArtifact, autoAdvanceCountdown,
  onInject, onDirectAddress,
  onAdvanceRound, onRestartRound, onRequestSynthesis, onConclude, onCancelAutoAdvance,
  onGenerateTldrs, onExportMd, onDisposeAgents,
}: CouncilControlsProps) {
  const [input, setInput] = useState('')
  const [directTo, setDirectTo] = useState('')
  const [showDirect, setShowDirect] = useState(false)

  const isActive = session.status === 'active'
  const isBusy = !!speaking
  // This council's panel was auto-spawned (Auto-assemble) and is still live.
  const hasTempAgents = session.spawnedSlugs.length > 0 && !session.agentsDisposed
  const disposeAgents = () => {
    if (window.confirm(
      `Dispose ${session.spawnedSlugs.length} temporary agent(s)? They were spawned for this council and will be removed from the fleet. The transcript and synthesis are kept.`,
    )) onDisposeAgents()
  }

  // Check if any recent system message indicates the *current* round is complete.
  // We scan the last few system messages because an extension message
  // ("Council extended by +N rounds...") may have been appended after
  // the "Round X complete" message. We pin to currentRound so a stale
  // "Round 1 complete" doesn't bleed into Round 2 while agents are still speaking.
  const recentSysMsgs = [...session.messages]
    .reverse()
    .filter(m => m.role === 'system')
    .slice(0, 5)
  const completeMarker = `Round ${session.currentRound} complete`
  const isRoundComplete = !isBusy && recentSysMsgs.some(m => m.content.includes(completeMarker))
  // A round that has begun but isn't complete AND has no loop running in this tab
  // was interrupted (server stop/restart, or this session was just reloaded).
  const isInterrupted = isActive && !isRoundComplete && !roundRunning && session.currentRound >= 1
  // Actively running a round in this tab.
  const isWorking = isActive && !isRoundComplete && roundRunning
  // Council was extended after the round ended — user needs a Continue button
  const canContinue = isActive && !isBusy && isRoundComplete && session.currentRound < session.maxRounds

  const handleSend = () => {
    const msg = input.trim()
    if (!msg) return
    if (directTo) {
      onDirectAddress(directTo, msg)
    } else {
      onInject(msg)
    }
    setInput('')
    setDirectTo('')
    setShowDirect(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      handleSend()
    }
  }

  if (session.status === 'concluded') {
    const hasTldrs = session.artifacts?.some(a => a.kind === 'tldr')
    return (
      <div className="shrink-0 border-t border-zinc-800 bg-zinc-900/30 px-4 py-3 wide:px-6">
        <div className="flex flex-wrap items-center gap-2">
          <div className="mr-auto min-w-0">
            <div className="flex items-center gap-1.5 text-sm font-medium text-zinc-300">
              <CheckCircle className="h-4 w-4 text-emerald-600 dark:text-emerald-400" /> Session concluded.
            </div>
            {session.agentsDisposed
              ? <div className="mt-0.5 text-xs text-zinc-500">Temporary agents disposed.</div>
              : <div className="mt-0.5 text-xs text-zinc-500">Use sidebar to extend rounds.</div>}
          </div>
          <Button
            onClick={onGenerateTldrs}
            disabled={!!generatingArtifact}
            icon={MessageSquareQuote}
            className={SM}
          >
            {hasTldrs ? 'Regen TL;DRs' : 'Generate TL;DRs'}
          </Button>
          <Button onClick={onExportMd} icon={Download} className={SM}>
            Export .md
          </Button>
          {hasTempAgents && (
            <Button
              variant="danger"
              onClick={disposeAgents}
              icon={Trash2}
              className={SM}
              title="Remove the temporary agents spawned for this council"
            >
              Dispose agents
            </Button>
          )}
        </div>
      </div>
    )
  }

  if (session.status === 'synthesizing') {
    return (
      <div className="shrink-0 border-t border-zinc-800 bg-zinc-900/30 px-4 py-4 wide:px-6">
        <div className="flex items-center justify-center gap-2.5 text-sm text-zinc-400">
          <span className="flex gap-1">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-violet-500" />
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-violet-500" style={{ animationDelay: '0.2s' }} />
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-violet-500" style={{ animationDelay: '0.4s' }} />
          </span>
          Synthesizing and voting in progress...
        </div>
      </div>
    )
  }

  const confirmRestart = (what: string) => {
    if (window.confirm(`Restart round ${session.currentRound}? ${what}`)) onRestartRound()
  }

  return (
    <div className="shrink-0 space-y-2.5 border-t border-zinc-800 bg-zinc-900/30 px-4 py-3 wide:px-6">
      {/* In-session status (round actively running in this tab) */}
      {isWorking && (
        <div className={BAR_ZINC}>
          <span className="relative flex h-2 w-2 shrink-0">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-violet-500 opacity-60" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-violet-500" />
          </span>
          <span className="min-w-0 flex-1 text-sm text-zinc-300">
            Round {session.currentRound} in session{session.currentRound >= session.maxRounds ? ` / ${session.maxRounds}` : ''}...
          </span>
          <Button
            variant="danger"
            onClick={() => confirmRestart("This discards this round's messages and re-runs it from the top.")}
            icon={RotateCcw}
            className={SM}
            title="Stop the current round and run it again"
          >
            Restart round
          </Button>
        </div>
      )}

      {/* Interrupted round — no loop is running (server stopped/restarted or the
          session was just reopened). Offer to re-run it. */}
      {isInterrupted && (
        <div className={`${BAR} border-amber-500/30 bg-amber-500/[0.06]`}>
          <span className="min-w-0 flex-[1_1_16rem] text-sm leading-snug text-amber-700 dark:text-amber-300">
            Round {session.currentRound} was interrupted — it isn't running. Restart to re-run it, or synthesize/conclude with what's saved.
          </span>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <Button
              variant="primary"
              onClick={() => confirmRestart("This discards this round's partial messages and re-runs it from the top.")}
              icon={RotateCcw}
              className={SM}
            >
              Restart round {session.currentRound}
            </Button>
            <Button onClick={onRequestSynthesis} icon={FileText} className={SM}>
              Synthesize
            </Button>
            <Button onClick={onConclude} icon={CheckCircle} className={SM}>
              Conclude
            </Button>
          </div>
        </div>
      )}
      {/* Round control bar */}
      {isRoundComplete && (
        <div className="space-y-2.5">
          {/* Auto-advance countdown bar */}
          {autoAdvanceCountdown > 0 && (
            <div className={`${BAR} border-violet-500/25 bg-violet-500/[0.07]`}>
              <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-violet-500" />
              <span className="flex-1 text-sm font-medium text-violet-700 dark:text-violet-300">
                Auto-advancing in {autoAdvanceCountdown}s...
              </span>
              <Button variant="ghost" onClick={onCancelAutoAdvance} icon={Pause} className={SM}>
                Pause
              </Button>
            </div>
          )}
          <div className="space-y-2.5 rounded-xl border border-zinc-800 bg-zinc-900/60 px-3.5 py-3">
            <div className="flex items-start gap-3">
              <span className="min-w-0 flex-1 pt-1 text-sm text-zinc-300">
                Round {session.currentRound}{session.currentRound >= session.maxRounds ? ` / ${session.maxRounds}` : ''} complete.
                {session.currentRound >= session.maxRounds && (
                  <span className="text-amber-700 dark:text-amber-300"> Max rounds reached — extend from sidebar or conclude.</span>
                )}
              </span>
              <Button
                variant="ghost"
                onClick={() => confirmRestart("This discards this round's messages and re-runs it from the top.")}
                disabled={isBusy}
                icon={RotateCcw}
                className={`${SM} shrink-0 hover:text-red-600! dark:hover:text-red-400!`}
                title="Discard this round and run it again"
              >
                Restart
              </Button>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {canContinue && (
                <Button variant="primary" onClick={onAdvanceRound} disabled={isBusy} icon={SkipForward} className={SM}>
                  Continue (Round {session.currentRound + 1})
                </Button>
              )}
              {!canContinue && session.currentRound < session.maxRounds && (
                <Button variant="primary" onClick={onAdvanceRound} disabled={isBusy} icon={SkipForward} className={SM}>
                  Next Round
                </Button>
              )}
              <Button onClick={onRequestSynthesis} disabled={isBusy} icon={FileText} className={SM}>
                Synthesize
              </Button>
              <Button
                onClick={onGenerateTldrs}
                disabled={isBusy || !!generatingArtifact}
                icon={MessageSquareQuote}
                className={SM}
              >
                TL;DR
              </Button>
              <Button onClick={onConclude} disabled={isBusy} icon={CheckCircle} className={SM}>
                Conclude
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Message input */}
      {isActive && (
        <div className="rounded-2xl border border-zinc-700 bg-zinc-950 p-1.5 transition-colors focus-within:border-violet-500/60 focus-within:ring-2 focus-within:ring-violet-500/15">
          {showDirect && (
            <div className="mb-1.5 flex items-center gap-2 border-b border-zinc-800 px-2 pb-2 pt-1">
              <span className="text-xs text-zinc-500">Direct to:</span>
              <select
                value={directTo}
                onChange={e => setDirectTo(e.target.value)}
                className="rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/60 focus:outline-none"
              >
                <option value="">Everyone</option>
                {session.agents.filter(a => !a.muted).map(a => (
                  <option key={a.id} value={a.id}>{a.name}</option>
                ))}
              </select>
            </div>
          )}
          <div className="flex items-center gap-1">
            <button
              onClick={() => setShowDirect(!showDirect)}
              className={`shrink-0 rounded-xl p-2 transition-colors ${showDirect ? 'bg-violet-500/10 text-violet-600 dark:text-violet-400' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200'}`}
              title="Direct address"
              aria-pressed={showDirect}
            >
              <ChevronDown className={`h-4 w-4 transition-transform ${showDirect ? 'rotate-180' : ''}`} />
            </button>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={directTo ? 'Direct message...' : 'Inject a message into the discussion...'}
              className="min-w-0 flex-1 bg-transparent px-1.5 py-2 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              title="Send (Ctrl+Enter)"
              aria-label="Send"
              className="shrink-0 rounded-xl bg-violet-600 p-2.5 text-white shadow-sm shadow-violet-900/20 transition-colors hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
