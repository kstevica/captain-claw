import { useCallback, useEffect, useRef, useState } from 'react'
import { AlertTriangle, CheckCircle2, Download, GitCommitHorizontal, Loader2, RefreshCw } from 'lucide-react'
import {
  checkDeckUpdate, fetchDeckUpdate, setDeckUpdateMode, startDeckUpdate,
  type DeckUpdateJob, type DeckUpdateMode, type DeckUpdateStatus,
} from './consoleApi'
import { Page, PageHeader, Card, Button, Notice, Pill, Dialog, Segmented, EmptyState, SettingRow } from './parts'

// Updates: this deck's version, what's new on its branch, and the one button that
// stops the agents, pulls and restarts (deck_update.py). While one runs the page
// follows it, through the restart too, and opens the new version's page at the end.

const ACTIVE = ['rehearsing', 'waiting', 'stopping', 'pulling', 'restarting', 'booted', 'rolling_back']

const PHASE_LABEL: Record<string, string> = {
  rehearsing: 'Trying the new version on a spare port',
  waiting: 'Waiting for agents to finish',
  stopping: 'Stopping the agents',
  pulling: 'Pulling the new version',
  restarting: 'Restarting the deck',
  booted: 'Restarting the deck',
  rolling_back: 'Going back to the previous version',
}

const MODES: { value: DeckUpdateMode; label: string; hint: string }[] = [
  { value: 'off', label: 'Off', hint: 'The deck only looks when you press Check now.' },
  { value: 'notify', label: 'Tell me', hint: 'It looks every 15 minutes, and tells admins in the bell when there\'s a new version.' },
  { value: 'install', label: 'Install by itself', hint: 'It looks every 15 minutes, and installs a new version when no agent is mid-turn. Everything merged to this branch goes live, as long as it passes the test start.' },
]

const short = (sha?: string) => (sha || '').slice(0, 9)

function when(ts?: number): string {
  return ts ? new Date(ts * 1000).toLocaleString() : ''
}

function Result({ job }: { job: DeckUpdateJob }) {
  const took = job.finished_at && job.started_at ? ` in ${Math.round(job.finished_at - job.started_at)} seconds` : ''
  if (job.phase === 'done') {
    return <Notice tone="emerald">Updated from <code>{short(job.from)}</code> to <code>{short(job.to)}</code>{took}, started by {job.by}. The agents are back or on their way.</Notice>
  }
  if (job.phase === 'rolled_back') {
    return (
      <Notice tone="amber">
        <p>{job.error} The deck went back to <code>{short(job.from)}</code> and is running.</p>
        <p className="mt-1">It's no longer in the terminal it was started in. It logs to <code>{job.detached_log}</code>.</p>
      </Notice>
    )
  }
  if (job.phase === 'failed') {
    return (
      <Notice tone="red">
        <p>The update to <code>{short(job.to)}</code> didn't go through, and this deck stays on the version it's on.</p>
        <p className="mt-1 whitespace-pre-wrap break-words">{job.error}</p>
      </Notice>
    )
  }
  return null
}

function Progress({ job, gone }: { job: DeckUpdateJob; gone: boolean }) {
  return (
    <Card title={`Updating to ${short(job.to)}`} icon={RefreshCw} hint={`Started by ${job.by}`}>
      <ul className="space-y-2 text-sm text-zinc-400">
        {(job.steps || []).map((s) => (
          <li key={`${s.at}-${s.text}`} className="flex items-start gap-2">
            <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-zinc-600" />
            <span className="min-w-0 break-words">{s.text}</span>
          </li>
        ))}
        <li className="flex items-center gap-2 font-medium text-zinc-200">
          <Loader2 className="h-4 w-4 shrink-0 animate-spin text-violet-500" />
          {gone ? 'The deck is restarting. This page comes back by itself.' : PHASE_LABEL[job.phase || ''] || 'Working'}
        </li>
      </ul>
    </Card>
  )
}

export function UpdatesConsole() {
  const [status, setStatus] = useState<DeckUpdateStatus | null>(null)
  const [error, setError] = useState('')
  const [checking, setChecking] = useState(false)
  const [starting, setStarting] = useState(false)
  const [confirm, setConfirm] = useState(false)
  // Asked and not answered, while one runs: that's the restart.
  const [gone, setGone] = useState(false)
  const following = useRef(false)

  const load = useCallback(async () => {
    try {
      const next = await fetchDeckUpdate()
      setStatus(next)
      setGone(false)
      setError('')
      if (next.active) following.current = true
      else if (following.current) {
        following.current = false
        // The bundle may have changed under this page.
        if (next.job.phase === 'done') window.setTimeout(() => window.location.reload(), 1500)
      }
    } catch (err) {
      if (following.current) setGone(true)
      else setError(err instanceof Error ? err.message : 'Couldn\'t read the deck\'s version')
    }
  }, [])

  useEffect(() => { void load() }, [load])

  const live = gone || !!status?.active
  useEffect(() => {
    if (!live) return
    const timer = window.setInterval(() => { void load() }, 2000)
    return () => window.clearInterval(timer)
  }, [live, load])

  const check = async () => {
    setChecking(true)
    setError('')
    try {
      setStatus(await checkDeckUpdate())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Couldn\'t check')
    } finally {
      setChecking(false)
    }
  }

  const begin = async (wait: boolean) => {
    if (!status?.target) return
    setStarting(true)
    setError('')
    try {
      await startDeckUpdate(status.target, wait)
      following.current = true
      setConfirm(false)
      await load()
    } catch (err) {
      setConfirm(false)
      setError(err instanceof Error ? err.message : 'Couldn\'t start the update')
    } finally {
      setStarting(false)
    }
  }

  const changeMode = async (mode: DeckUpdateMode) => {
    setStatus((s) => (s ? { ...s, mode } : s))
    try {
      await setDeckUpdateMode(mode)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Couldn\'t save that')
      void load()
    }
  }

  const job = status?.job
  const active = gone || !!status?.active || ACTIVE.includes(job?.phase || '')
  const behind = status?.behind || 0
  const blocked = (status?.blockers || []).length > 0
  const busy = status?.agents_busy || []
  const mode = status?.mode || 'off'

  return (
    <Page>
      <PageHeader
        icon={RefreshCw}
        title="Updates"
        description="Bring this deck up to the newest version on its branch. It tries the new version first, stops the agents, pulls and restarts. If the new version doesn't come up, it goes back by itself."
        actions={status?.available && !active
          ? <Button icon={RefreshCw} busy={checking} onClick={() => { void check() }}>Check now</Button>
          : undefined}
      />

      {error && <Notice tone="red">{error}</Notice>}

      {!status && !error && (
        <div className="flex items-center gap-2 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> Loading…</div>
      )}

      {status && !status.available && (
        <EmptyState icon={AlertTriangle} title="This deck can't update itself">{status.why}</EmptyState>
      )}

      {job && active && <Progress job={job} gone={gone} />}
      {job && !active && <Result job={job} />}

      {status?.available && !active && (
        <>
          {status.fetch_error && <Notice tone="amber">{status.fetch_error}</Notice>}

          <Card title="This deck" icon={GitCommitHorizontal}>
            <div className="min-w-0 space-y-1">
              <p className="break-words text-sm font-medium text-zinc-100">{status.head_title}</p>
              <p className="break-words text-xs text-zinc-500">
                <code>{short(status.head)}</code> on {status.branch}, following {status.upstream}
                {status.remote_url ? ` at ${status.remote_url}` : ''}
              </p>
              <p className="text-xs text-zinc-500">
                {status.checked_at ? `Last checked ${when(status.checked_at)}.` : 'Not checked since the deck started.'}
              </p>
            </div>
          </Card>

          {behind === 0
            ? <Notice tone="zinc">{status.checked_at ? 'This is the newest version.' : 'Press Check now to see if there\'s a newer version.'}</Notice>
            : (
              <Card
                title={`${behind} new commit${behind === 1 ? '' : 's'}`}
                icon={Download}
                action={<Button variant="primary" icon={Download} disabled={blocked} onClick={() => setConfirm(true)}>Update now</Button>}
              >
                <div className="space-y-4">
                  {blocked && (
                    <Notice tone="amber">
                      <p className="font-medium">This can't be updated yet, and nothing has been touched.</p>
                      {status.blockers?.map((b) => <p key={b} className="mt-1 whitespace-pre-wrap break-words">{b}</p>)}
                    </Notice>
                  )}
                  {status.notes?.map((n) => <Notice key={n} tone="zinc">{n}</Notice>)}
                  <ul className="divide-y divide-zinc-800/70">
                    {status.commits?.map((c) => (
                      <li key={c.sha} className="flex flex-col gap-0.5 py-2 first:pt-0 last:pb-0">
                        <span className="break-words text-sm text-zinc-200">{c.title}</span>
                        <span className="text-xs text-zinc-500"><code>{short(c.sha)}</code> · {c.author} · {new Date(c.date).toLocaleDateString()}</span>
                      </li>
                    ))}
                  </ul>
                  {behind > (status.commits?.length || 0) && (
                    <p className="text-xs text-zinc-500">And {behind - (status.commits?.length || 0)} older ones.</p>
                  )}
                </div>
              </Card>
            )}

          <Card title="Automatically">
            <SettingRow title="When there's a new version" hint={MODES.find((m) => m.value === mode)?.hint}>
              <Segmented label="When there's a new version" value={mode} options={MODES} onChange={(m) => { void changeMode(m) }} />
            </SettingRow>
          </Card>
        </>
      )}

      {confirm && status && (
        <Dialog
          title={`Update this deck to ${short(status.target)}?`}
          icon={Download}
          onClose={() => setConfirm(false)}
          footer={(
            <>
              <Button variant="ghost" onClick={() => setConfirm(false)}>Cancel</Button>
              {busy.length > 0 && <Button variant="danger" disabled={starting} onClick={() => { void begin(false) }}>Stop them now</Button>}
              <Button variant="primary" busy={starting} onClick={() => { void begin(busy.length > 0) }}>
                {busy.length > 0 ? 'Wait for them, then update' : 'Update now'}
              </Button>
            </>
          )}
        >
          <div className="space-y-3 text-sm leading-relaxed text-zinc-400">
            <p>
              {status.agents_running
                ? `The ${status.agents_running} running agent${status.agents_running === 1 ? '' : 's'} stop and come back when the deck is up. `
                : ''}
              Everyone's page loses the deck for a moment.
              {status.crew_running ? ` ${status.crew_running} Crew task${status.crew_running === 1 ? '' : 's'} keep going through it.` : ''}
            </p>
            {busy.length > 0 && (
              <Notice tone="amber">
                <span className="flex flex-wrap items-center gap-1.5">
                  Working right now: {busy.map((name) => <Pill key={name} tone="amber">{name}</Pill>)}
                </span>
                <span className="mt-1 block">Waiting gives them up to 5 minutes. After that they're stopped anyway.</span>
              </Notice>
            )}
          </div>
        </Dialog>
      )}
    </Page>
  )
}
