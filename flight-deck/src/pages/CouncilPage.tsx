import { useEffect, useState, useCallback, useRef } from 'react'
import { Plus, MessagesSquare, Loader2 } from 'lucide-react'
import { useCouncilStore, type CreateSessionConfig } from '../stores/councilStore'
import { useOnboardingStore } from '../stores/onboardingStore'
import { CouncilSetup } from '../components/council/CouncilSetup'
import { CouncilDiscussion } from '../components/council/CouncilDiscussion'
import { CouncilControls } from '../components/council/CouncilControls'
import { CouncilSidebar } from '../components/council/CouncilSidebar'
import { SynthesisView } from '../components/council/SynthesisView'
import { TldrPanel } from '../components/council/TldrPanel'
import { ActionPointsPanel } from '../components/council/ActionPointsPanel'
import { SessionCard } from '../components/council/SessionCard'
import { CouncilHeader, IconSquare, StatusPill, TypePill } from '../components/council/councilLook'
import { TYPE_LOOK, TONE_SQUARE } from '../components/council/councilStyles'
import { ShareModal } from '../components/common/ShareModal'
import { Button, Page } from '../components/console/parts'
import type { CouncilSessionSummary, SessionType } from '../stores/councilStore'

type PageState = 'list' | 'setup' | 'active'

const SIDEBAR_MIN = 280
const SIDEBAR_MAX_RATIO = 0.65
const SIDEBAR_DEFAULT_RATIO = 0.5

// The four kinds of session the empty page shows, each in its colour.
const STARTERS: { type: SessionType; label: string; desc: string }[] = [
  { type: 'debate', label: 'Debate', desc: 'Argue perspectives' },
  { type: 'brainstorm', label: 'Brainstorm', desc: 'Generate ideas' },
  { type: 'review', label: 'Review', desc: 'Evaluate work' },
  { type: 'planning', label: 'Plan', desc: 'Strategy & goals' },
]

/** `onClose`: in a kiosk dialog, its Close. */
export function CouncilPage({ onClose }: { onClose?: () => void } = {}) {
  const {
    sessions, activeSession, loading, speaking, roundRunning, generatingArtifact, activityLog,
    autoAdvanceCountdown,
    loadSessionList, createSession, loadSession, deleteSession, cancelSession, clearActive,
    startCouncil, advanceRound, restartRound, requestSynthesis, concludeSession,
    cancelAutoAdvance,
    injectMessage, directAddress, muteAgent, pinMessage,
    connectAllAgents, disposeAgents,
    generateTldrs, generateActionPoints, sendActionPointToAgent, exportMinutesMd,
  } = useCouncilStore()

  const [pageState, setPageState] = useState<PageState>('list')
  const [shareTarget, setShareTarget] = useState<CouncilSessionSummary | null>(null)
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
  const actionPoints = activeSession?.artifacts?.filter(a => a.kind === 'action_points') || []
  const showTldrs = tldrs.length > 0 || generatingArtifact === 'tldr'
  const showActionPoints = !!synthesisMsg || actionPoints.length > 0 || generatingArtifact === 'action_points'

  const newSession = (
    <Button variant="primary" icon={Plus} onClick={() => setPageState('setup')}>New session</Button>
  )

  return (
    <div className="@container flex h-full flex-col">
      {shareTarget && (
        <ShareModal
          resourceType="council"
          resourceId={shareTarget.id}
          resourceName={shareTarget.title || 'Council session'}
          onClose={() => setShareTarget(null)}
        />
      )}

      {/* Session List */}
      {pageState === 'list' && (
        <div className="min-h-0 flex-1 overflow-y-auto">
          <Page full={!!onClose}>
            <CouncilHeader
              icon={MessagesSquare}
              title="Council"
              description="Your agents discuss a topic in rounds — answering and challenging each other — then sum up the best of it."
              action={sessions.length > 0 ? newSession : undefined}
              onClose={onClose}
            />

            {loading && sessions.length === 0 && (
              <div className="flex items-center justify-center gap-2 py-16 text-sm text-zinc-500">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading sessions...
              </div>
            )}

            {!loading && sessions.length === 0 && (
              <div className="relative overflow-hidden rounded-3xl border border-zinc-800 bg-zinc-900/40 px-5 py-10 text-center @xl:px-10 @xl:py-14">
                <div className="pointer-events-none absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-amber-500/[0.07] to-transparent" aria-hidden />
                <div className="relative">
                  <IconSquare icon={MessagesSquare} tone="amber" size="xl" className="mx-auto shadow-lg shadow-amber-500/10" />
                  <h2 className="mt-5 text-xl font-semibold tracking-tight text-zinc-100">Put a topic to your agents</h2>
                  <p className="mx-auto mt-2 max-w-lg text-sm leading-relaxed text-zinc-500">
                    Each agent rates how well the topic fits it, then answers, challenges or builds on the others,
                    round by round. At the end, the best ideas come together in one summary.
                  </p>

                  <div className="mx-auto mt-8 grid max-w-3xl grid-cols-2 gap-2.5 text-left @2xl:grid-cols-4">
                    {STARTERS.map(({ type, label, desc }) => {
                      const look = TYPE_LOOK[type]
                      const Icon = look.icon
                      return (
                        <div key={type} className="flex min-w-0 flex-col rounded-2xl border border-zinc-800 bg-zinc-950/40 p-3.5">
                          <span className={`flex h-10 w-10 items-center justify-center rounded-xl ring-1 ${TONE_SQUARE[look.tone]}`}>
                            <Icon className="h-5 w-5" />
                          </span>
                          <span className="mt-3 block text-sm font-semibold leading-snug text-zinc-100">{label}</span>
                          <span className="mt-0.5 block text-xs leading-snug text-zinc-500">{desc}</span>
                        </div>
                      )
                    })}
                  </div>

                  <Button variant="primary" icon={Plus} onClick={() => setPageState('setup')} className="mt-8 px-5! py-2.5!">
                    Start your first session
                  </Button>
                </div>
              </div>
            )}

            {sessions.length > 0 && (
              <section className="@container space-y-3">
                <div className="flex items-baseline gap-2 px-1">
                  <h2 className="text-sm font-semibold text-zinc-200">Sessions</h2>
                  <span className="text-xs tabular-nums text-zinc-500">{sessions.length}</span>
                </div>
                <div className="grid gap-3 @2xl:grid-cols-2 @6xl:grid-cols-3">
                  {sessions.map(s => (
                    <SessionCard
                      key={s.id}
                      session={s}
                      onOpen={handleOpen}
                      onDelete={handleDelete}
                      onCancel={cancelSession}
                      onShare={setShareTarget}
                    />
                  ))}
                </div>
              </section>
            )}
          </Page>
        </div>
      )}

      {/* Setup */}
      {pageState === 'setup' && (
        <div className="min-h-0 flex-1 overflow-y-auto">
          <Page full={!!onClose}>
            <CouncilHeader
              icon={MessagesSquare}
              title="New session"
              description="Set the topic, choose who takes part, and start."
              onBack={handleBack}
              onClose={onClose}
            />
            <CouncilSetup
              onStart={handleCreate}
              onCancel={() => setPageState('list')}
            />
          </Page>
        </div>
      )}

      {/* Active Session */}
      {pageState === 'active' && (
        <div className="shrink-0 border-b border-zinc-800 px-4 py-3.5 wide:px-6">
          <CouncilHeader
            compact
            icon={(activeSession && TYPE_LOOK[activeSession.sessionType]?.icon) || MessagesSquare}
            tone={(activeSession && TYPE_LOOK[activeSession.sessionType]?.tone) || 'amber'}
            title={activeSession?.title || 'Council Session'}
            meta={activeSession && (
              <>
                <TypePill type={activeSession.sessionType} />
                <StatusPill status={activeSession.status} />
                <span className="text-[11px] font-medium tabular-nums text-zinc-500">
                  R{activeSession.currentRound}/{activeSession.maxRounds}
                </span>
              </>
            )}
            description={activeSession?.topic}
            onBack={handleBack}
            onClose={onClose}
          />
        </div>
      )}
      {pageState === 'active' && activeSession && (
        <div ref={containerRef} className="flex min-h-0 flex-1">
          {/* Main discussion area */}
          <div className="flex min-w-0 flex-col" style={{ flex: `1 1 ${(1 - sidebarRatio) * 100}%` }}>
            {/* Synthesis, TL;DRs, action points: on top, never taking the whole column */}
            {(synthesisMsg || showTldrs || showActionPoints) && (
              <div className="@container max-h-[45%] shrink-0 space-y-3 overflow-y-auto border-b border-zinc-800/70 px-4 py-3 wide:px-6">
                {synthesisMsg && (
                  <SynthesisView message={synthesisMsg} votes={activeSession.votes} />
                )}

                {showTldrs && (
                  <TldrPanel
                    tldrs={tldrs}
                    generating={generatingArtifact === 'tldr'}
                    onGenerate={generateTldrs}
                  />
                )}

                {/* Action points panel — available once the council has been synthesized */}
                {showActionPoints && (
                  <ActionPointsPanel
                    artifacts={actionPoints}
                    generating={generatingArtifact === 'action_points'}
                    busy={!!speaking}
                    onGenerate={generateActionPoints}
                    onSend={sendActionPointToAgent}
                  />
                )}
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
              roundRunning={roundRunning}
              generatingArtifact={generatingArtifact}
              autoAdvanceCountdown={autoAdvanceCountdown}
              onInject={injectMessage}
              onDirectAddress={directAddress}
              onAdvanceRound={advanceRound}
              onRestartRound={restartRound}
              onRequestSynthesis={requestSynthesis}
              onConclude={concludeSession}
              onCancelAutoAdvance={cancelAutoAdvance}
              onGenerateTldrs={generateTldrs}
              onExportMd={exportMinutesMd}
              onDisposeAgents={disposeAgents}
            />
          </div>

          {/* Resize handle */}
          <div
            onMouseDown={onMouseDown}
            className="group hidden w-1.5 shrink-0 cursor-col-resize items-center justify-center border-l border-zinc-800 transition-colors hover:bg-violet-500/15 active:bg-violet-500/25 lg:flex"
          >
            <div className="h-10 w-1 rounded-full bg-zinc-700 transition-colors group-hover:bg-violet-500/60" />
          </div>

          {/* Sidebar panel */}
          <div
            className="hidden shrink-0 overflow-hidden lg:block"
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
  )
}
