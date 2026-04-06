import { useEffect, useState, useCallback, useRef } from 'react'
import { Users, Plus, ArrowLeft, MessageSquare, Lightbulb, ClipboardCheck, Target } from 'lucide-react'
import { useCouncilStore, type CreateSessionConfig } from '../stores/councilStore'
import { useOnboardingStore } from '../stores/onboardingStore'
import { CouncilSetup } from '../components/council/CouncilSetup'
import { CouncilDiscussion } from '../components/council/CouncilDiscussion'
import { CouncilControls } from '../components/council/CouncilControls'
import { CouncilSidebar } from '../components/council/CouncilSidebar'
import { SynthesisView } from '../components/council/SynthesisView'
import { TldrPanel } from '../components/council/TldrPanel'
import { SessionCard } from '../components/council/SessionCard'

type PageState = 'list' | 'setup' | 'active'

const SIDEBAR_MIN = 280
const SIDEBAR_MAX_RATIO = 0.65
const SIDEBAR_DEFAULT_RATIO = 0.5

export function CouncilPage() {
  const {
    sessions, activeSession, loading, speaking, generatingArtifact, activityLog,
    autoAdvanceCountdown,
    loadSessionList, createSession, loadSession, deleteSession, clearActive,
    startCouncil, advanceRound, requestSynthesis, concludeSession,
    cancelAutoAdvance,
    injectMessage, directAddress, muteAgent, pinMessage,
    connectAllAgents,
    generateTldrs, exportMinutesMd,
  } = useCouncilStore()

  const [pageState, setPageState] = useState<PageState>('list')
  const [sidebarRatio, setSidebarRatio] = useState(SIDEBAR_DEFAULT_RATIO)
  const containerRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)

  useEffect(() => {
    loadSessionList()
  }, [])

  // Sync page state with active session
  useEffect(() => {
    if (activeSession) setPageState('active')
  }, [activeSession?.id])

  // Drag handler for resizable splitter
  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    dragging.current = true

    const onMouseMove = (ev: MouseEvent) => {
      if (!dragging.current || !containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const totalWidth = rect.width
      const sidebarWidth = rect.right - ev.clientX
      const ratio = Math.max(SIDEBAR_MIN / totalWidth, Math.min(SIDEBAR_MAX_RATIO, sidebarWidth / totalWidth))
      setSidebarRatio(ratio)
    }
    const onMouseUp = () => {
      dragging.current = false
      document.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('mouseup', onMouseUp)
    }
    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('mouseup', onMouseUp)
  }, [])

  const onboarding = useOnboardingStore()

  const handleCreate = async (cfg: CreateSessionConfig) => {
    const id = await createSession(cfg)
    await loadSession(id)
    await startCouncil()
    setPageState('active')
    if (!onboarding.completed.council) onboarding.completeStep('council')
  }

  const handleOpen = async (id: string) => {
    await loadSession(id)
    // If session is active/synthesizing, reconnect agents
    const session = useCouncilStore.getState().activeSession
    if (session && (session.status === 'active' || session.status === 'synthesizing')) {
      connectAllAgents()
    }
    setPageState('active')
  }

  const handleDelete = async (id: string) => {
    await deleteSession(id)
  }

  const handleBack = () => {
    clearActive()
    setPageState('list')
    loadSessionList()
  }

  // Synthesis message for special rendering
  const synthesisMsg = activeSession?.messages.find(m => m.role === 'synthesis')
  const tldrs = activeSession?.artifacts?.filter(a => a.kind === 'tldr') || []

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="shrink-0 border-b border-zinc-700/50 bg-zinc-900/50 px-4 py-3 md:px-6">
        <div className="flex items-center gap-3">
          {pageState !== 'list' && (
            <button
              onClick={handleBack}
              className="rounded-lg p-1.5 text-zinc-400 hover:bg-zinc-700/50 hover:text-zinc-200"
            >
              <ArrowLeft className="h-4 w-4" />
            </button>
          )}
          <Users className="h-5 w-5 text-violet-400" />
          <div>
            <h1 className="text-sm font-semibold text-zinc-200">
              {pageState === 'list' && 'Council of Agents'}
              {pageState === 'setup' && 'New Council Session'}
              {pageState === 'active' && (activeSession?.title || 'Council Session')}
            </h1>
            {pageState === 'active' && activeSession && (
              <p className="text-xs text-zinc-500 line-clamp-1">{activeSession.topic}</p>
            )}
          </div>

          {pageState === 'list' && (
            <button
              onClick={() => setPageState('setup')}
              className="ml-auto flex items-center gap-1.5 rounded-lg bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500"
            >
              <Plus className="h-3.5 w-3.5" /> New Session
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {/* Session List */}
        {pageState === 'list' && (
          <div className="h-full overflow-auto p-4 md:p-6">
            {loading && sessions.length === 0 && (
              <div className="flex items-center justify-center py-12 text-sm text-zinc-500">
                Loading sessions...
              </div>
            )}

            {!loading && sessions.length === 0 && (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <Users className="h-12 w-12 text-zinc-600 mb-4" />
                <h2 className="text-lg font-medium text-zinc-300 mb-2">Council Sessions</h2>
                <p className="text-sm text-zinc-500 mb-6 max-w-lg">
                  Have your agents discuss topics in moderated multi-round sessions. Each agent self-scores its suitability
                  and can answer, challenge, refine, or broaden other agents' points. A synthesizer combines the best ideas at the end.
                </p>

                {/* Session type examples */}
                <div className="mb-6 grid grid-cols-2 gap-2 max-w-md w-full text-left sm:grid-cols-4">
                  {[
                    { icon: MessageSquare, label: 'Debate', desc: 'Argue perspectives' },
                    { icon: Lightbulb, label: 'Brainstorm', desc: 'Generate ideas' },
                    { icon: ClipboardCheck, label: 'Review', desc: 'Evaluate work' },
                    { icon: Target, label: 'Plan', desc: 'Strategy & goals' },
                  ].map(({ icon: Icon, label, desc }) => (
                    <div key={label} className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-3">
                      <Icon className="h-4 w-4 text-violet-400 mb-1.5" />
                      <p className="text-xs font-medium text-zinc-300">{label}</p>
                      <p className="text-[10px] text-zinc-600">{desc}</p>
                    </div>
                  ))}
                </div>

                <button
                  onClick={() => setPageState('setup')}
                  className="flex items-center gap-2 rounded-lg bg-violet-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-violet-500"
                >
                  <Plus className="h-4 w-4" /> Create First Session
                </button>
              </div>
            )}

            {sessions.length > 0 && (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {sessions.map(s => (
                  <SessionCard
                    key={s.id}
                    session={s}
                    onOpen={handleOpen}
                    onDelete={handleDelete}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Setup */}
        {pageState === 'setup' && (
          <div className="h-full overflow-auto p-4 md:p-6">
            <CouncilSetup
              onStart={handleCreate}
              onCancel={() => setPageState('list')}
            />
          </div>
        )}

        {/* Active Session */}
        {pageState === 'active' && activeSession && (
          <div ref={containerRef} className="flex h-full">
            {/* Main discussion area */}
            <div className="flex flex-col min-w-0" style={{ flex: `1 1 ${(1 - sidebarRatio) * 100}%` }}>
              {/* Synthesis view if present */}
              {synthesisMsg && (
                <div className="shrink-0 px-4 pt-3">
                  <SynthesisView message={synthesisMsg} />
                </div>
              )}

              {/* TL;DR panel */}
              {(tldrs.length > 0 || generatingArtifact === 'tldr') && (
                <div className="shrink-0 px-4 pt-3">
                  <TldrPanel
                    tldrs={tldrs}
                    generating={generatingArtifact === 'tldr'}
                    onGenerate={generateTldrs}
                  />
                </div>
              )}

              {/* Discussion thread */}
              <CouncilDiscussion
                session={activeSession}
                onPin={pinMessage}
                speaking={speaking}
              />

              {/* Controls */}
              <CouncilControls
                session={activeSession}
                speaking={speaking}
                generatingArtifact={generatingArtifact}
                autoAdvanceCountdown={autoAdvanceCountdown}
                onInject={injectMessage}
                onDirectAddress={directAddress}
                onAdvanceRound={advanceRound}
                onRequestSynthesis={requestSynthesis}
                onConclude={concludeSession}
                onCancelAutoAdvance={cancelAutoAdvance}
                onGenerateTldrs={generateTldrs}
                onExportMd={exportMinutesMd}
              />
            </div>

            {/* Resize handle */}
            <div
              onMouseDown={onMouseDown}
              className="hidden lg:flex w-1 shrink-0 cursor-col-resize items-center justify-center hover:bg-violet-500/20 active:bg-violet-500/30 transition-colors"
            >
              <div className="h-8 w-0.5 rounded-full bg-zinc-600" />
            </div>

            {/* Sidebar panel */}
            <div
              className="hidden lg:block shrink-0 overflow-hidden"
              style={{ width: `${sidebarRatio * 100}%`, minWidth: SIDEBAR_MIN }}
            >
              <CouncilSidebar
                session={activeSession}
                speaking={speaking}
                activityLog={activityLog}
                autoAdvanceCountdown={autoAdvanceCountdown}
                onMute={muteAgent}
                onPinClick={pinMessage}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
