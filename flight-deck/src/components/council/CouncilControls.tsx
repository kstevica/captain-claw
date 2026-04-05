import { useState } from 'react'
import { Send, SkipForward, FileText, CheckCircle, ChevronDown, Download, MessageSquareQuote } from 'lucide-react'
import type { CouncilSession } from '../../stores/councilStore'

interface CouncilControlsProps {
  session: CouncilSession
  speaking: string
  generatingArtifact: string
  onInject: (content: string) => void
  onDirectAddress: (agentId: string, content: string) => void
  onAdvanceRound: () => void
  onRequestSynthesis: () => void
  onConclude: () => void
  onGenerateTldrs: () => void
  onExportMd: () => void
}

export function CouncilControls({
  session, speaking, generatingArtifact, onInject, onDirectAddress,
  onAdvanceRound, onRequestSynthesis, onConclude,
  onGenerateTldrs, onExportMd,
}: CouncilControlsProps) {
  const [input, setInput] = useState('')
  const [directTo, setDirectTo] = useState('')
  const [showDirect, setShowDirect] = useState(false)

  const isActive = session.status === 'active'
  const isBusy = !!speaking

  // Check if the last system message indicates round completion
  const lastSysMsg = [...session.messages].reverse().find(m => m.role === 'system')
  const isRoundComplete = lastSysMsg?.content.includes('Round') && lastSysMsg?.content.includes('complete')

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
      <div className="border-t border-zinc-700/50 bg-zinc-900/50 px-4 py-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-zinc-500 mr-auto">Session concluded.</span>
          <button
            onClick={onGenerateTldrs}
            disabled={!!generatingArtifact}
            className="flex items-center gap-1 rounded-lg border border-violet-500/40 px-2.5 py-1.5 text-xs font-medium text-violet-400 hover:bg-violet-500/10 disabled:opacity-40"
          >
            <MessageSquareQuote className="h-3 w-3" /> {hasTldrs ? 'Regen TL;DRs' : 'Generate TL;DRs'}
          </button>
          <button
            onClick={onExportMd}
            className="flex items-center gap-1 rounded-lg border border-zinc-600 px-2.5 py-1.5 text-xs text-zinc-400 hover:bg-zinc-700/30"
          >
            <Download className="h-3 w-3" /> Export .md
          </button>
        </div>
      </div>
    )
  }

  if (session.status === 'synthesizing') {
    return (
      <div className="border-t border-zinc-700/50 bg-zinc-900/50 px-4 py-3 text-center text-sm text-zinc-400">
        Synthesizing and voting in progress...
      </div>
    )
  }

  return (
    <div className="border-t border-zinc-700/50 bg-zinc-900/50 px-4 py-3 space-y-2">
      {/* Round control bar */}
      {isRoundComplete && (
        <div className="flex items-center gap-2 rounded-lg bg-zinc-800/50 px-3 py-2">
          <span className="flex-1 text-xs text-zinc-400">
            Round {session.currentRound} complete.
            {session.currentRound >= session.maxRounds && ' Max rounds reached.'}
          </span>
          {session.currentRound < session.maxRounds && (
            <button
              onClick={onAdvanceRound}
              disabled={isBusy}
              className="flex items-center gap-1 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
            >
              <SkipForward className="h-3.5 w-3.5" /> Next Round
            </button>
          )}
          <button
            onClick={onRequestSynthesis}
            disabled={isBusy}
            className="flex items-center gap-1 rounded-lg border border-cyan-500/40 px-3 py-1.5 text-xs font-medium text-cyan-400 hover:bg-cyan-500/10 disabled:opacity-40"
          >
            <FileText className="h-3.5 w-3.5" /> Synthesize
          </button>
          <button
            onClick={onGenerateTldrs}
            disabled={isBusy || !!generatingArtifact}
            className="flex items-center gap-1 rounded-lg border border-violet-500/30 px-3 py-1.5 text-xs text-violet-400 hover:bg-violet-500/10 disabled:opacity-40"
          >
            <MessageSquareQuote className="h-3.5 w-3.5" /> TL;DR
          </button>
          <button
            onClick={onConclude}
            disabled={isBusy}
            className="flex items-center gap-1 rounded-lg border border-zinc-600 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-700/30 disabled:opacity-40"
          >
            <CheckCircle className="h-3.5 w-3.5" /> Conclude
          </button>
        </div>
      )}

      {/* Message input */}
      {isActive && (
        <div className="flex items-end gap-2">
          <div className="flex-1">
            {showDirect && (
              <div className="mb-1 flex items-center gap-2">
                <span className="text-[10px] text-zinc-500">Direct to:</span>
                <select
                  value={directTo}
                  onChange={e => setDirectTo(e.target.value)}
                  className="rounded border border-zinc-700/50 bg-zinc-800/50 px-2 py-0.5 text-xs text-zinc-300 focus:outline-none"
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
                className={`shrink-0 rounded p-1.5 transition-colors ${showDirect ? 'text-violet-400' : 'text-zinc-500 hover:text-zinc-300'}`}
                title="Direct address"
              >
                <ChevronDown className="h-4 w-4" />
              </button>
              <input
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={directTo ? 'Direct message...' : 'Inject a message into the discussion...'}
                className="flex-1 rounded-lg border border-zinc-700/50 bg-zinc-800/50 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-500 focus:border-violet-500/50 focus:outline-none"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim()}
                className="shrink-0 rounded-lg bg-violet-600 p-2 text-white hover:bg-violet-500 disabled:opacity-40"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
