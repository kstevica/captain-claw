import { useCallback, useEffect, useState } from 'react'
import { BadgeCheck, CircleAlert, Factory, Loader2, Rocket, Wand2 } from 'lucide-react'
import { api, post } from '../api'
import ManifestEditor from '../components/ManifestEditor'

type Manifest = Record<string, unknown>

interface PackSummary {
  slug: string; status: string; version: number; owner_id: string
  name: string; tagline: string; accent: string
  generation: string; eval: string
}

interface PackDetail {
  pack: Record<string, unknown> & {
    name?: string; tagline?: string
    theme?: Record<string, string>
    evals?: { brief?: string }[]
  }
  summary: PackSummary
  generation: { status?: string; message?: string }
  eval: { status?: string; verdict?: string; message?: string
          metrics?: Record<string, number | string> }
}

interface ProgressEvent { i: number; stage: string; message: string; ok?: boolean }

/** The live team-at-work feed of a factory run — same shape the customer
 * commission feed uses. Phase lines are banners; the rest are detail. */
function RunFeed({ events }: { events: ProgressEvent[] }) {
  if (events.length === 0) return null
  return (
    <div className="mt-2 max-h-56 overflow-y-auto rounded-lg border border-[var(--lp-border)] bg-[var(--lp-bg)] px-3 py-2 space-y-0.5 text-xs font-mono">
      {events.map((e) => (
        <div key={e.i} className={
          e.ok === false ? 'text-red-400'
          : e.stage === 'phase' ? 'font-bold mt-1.5'
          : 'text-[var(--lp-text-dim)]'}>
          {e.stage === 'phase' ? `— ${e.message} —` : `${e.stage}: ${e.message}`}
        </div>
      ))}
    </div>
  )
}

/** Pack Studio — the in-product factory: draft → generate → review →
 * evaluate (ship-gate) → publish. Creator/admin only (the header hides the
 * entry otherwise; the BFF enforces it regardless). */
export default function Studio() {
  const [packs, setPacks] = useState<PackSummary[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<PackDetail | null>(null)
  const [error, setError] = useState('')

  // Create form
  const [slug, setSlug] = useState('')
  const [name, setName] = useState('')

  // Editors
  const [manifest, setManifest] = useState<Manifest | null>(null)
  const [genInstructions, setGenInstructions] = useState('')
  const [busy, setBusy] = useState('')
  const [events, setEvents] = useState<ProgressEvent[]>([])

  const loadList = useCallback(async () => {
    const d = await api<{ packs: PackSummary[] }>('/api/packs')
    setPacks(d.packs)
  }, [])

  const loadDetail = useCallback(async (s: string) => {
    const d = await api<PackDetail>(`/api/packs/${s}`)
    setDetail(d)
    const m = { ...d.pack }
    delete (m as Record<string, unknown>).slug
    delete (m as Record<string, unknown>).pack_status
    delete (m as Record<string, unknown>).pack_version
    setManifest(m)
    return d
  }, [])

  useEffect(() => { void loadList() }, [loadList])
  useEffect(() => {
    setDetail(null)
    if (selected) void loadDetail(selected)
  }, [selected, loadDetail])

  // Poll while the factory is working. Generation is a single completion (no
  // feed); the EVALUATE golden run is the real multi-agent run, so stream its
  // live event feed (the team-at-work log).
  const evalRunning = detail?.eval.status === 'running'
  const working = detail?.generation.status === 'running' || evalRunning
  useEffect(() => {
    if (!selected) { setEvents([]); return }
    if (!working) return
    const tick = async () => {
      await loadDetail(selected)
      await loadList()
      if (evalRunning) {
        try {
          const p = await api<{ events: ProgressEvent[] }>(
            `/api/packs/${selected}/progress?phase=eval`)
          setEvents(p.events ?? [])
        } catch { /* best-effort */ }
      }
    }
    void tick()
    const iv = setInterval(() => { void tick() }, 2500)
    return () => clearInterval(iv)
  }, [selected, working, evalRunning, loadDetail, loadList])

  const act = async (action: () => Promise<unknown>, label: string) => {
    setBusy(label); setError('')
    try { await action(); if (selected) await loadDetail(selected); await loadList() }
    catch (e) { setError(e instanceof Error ? e.message : `${label} failed`) }
    finally { setBusy('') }
  }

  const createPack = () => act(async () => {
    await post('/api/packs', { slug: slug.trim(), name: name.trim() })
    setSelected(slug.trim()); setSlug(''); setName('')
  }, 'create')

  const saveManifest = () => act(async () => {
    if (!manifest) return
    await api(`/api/packs/${selected}`, { method: 'PUT',
                                          body: JSON.stringify({ manifest }) })
  }, 'save')

  const generate = () => act(async () => {
    await post(`/api/packs/${selected}/generate`, { instructions: genInstructions.trim() })
  }, 'generate')

  const evaluate = () => act(() => post(`/api/packs/${selected}/evaluate`, {}), 'evaluate')
  const publish = () => act(() => post(`/api/packs/${selected}/publish`, {}), 'publish')
  const cancel = (phase: 'generation' | 'eval') =>
    act(() => post(`/api/packs/${selected}/cancel?phase=${phase}`, {}), 'cancel')

  const evalGreen = detail?.eval.verdict === 'green'

  return (
    <div className="max-w-4xl mx-auto px-5 py-6 space-y-5">
      <div className="flex items-center gap-2">
        <Factory size={18} style={{ color: 'var(--lp-accent)' }} />
        <h1 className="text-xl font-bold">Pack Studio</h1>
        <span className="text-sm text-[var(--lp-text-dim)]">
          Forge a vertical desk: draft → generate → evaluate → publish.
        </span>
      </div>

      <div className="flex gap-2">
        <input value={slug} onChange={(e) => setSlug(e.target.value)}
               placeholder="slug (e.g. tender-desk)"
               className="w-48 rounded-lg bg-[var(--lp-surface)] border border-[var(--lp-border)] px-3 py-2 text-sm outline-none focus:border-[var(--lp-accent)]" />
        <input value={name} onChange={(e) => setName(e.target.value)}
               placeholder="Desk name"
               className="flex-1 rounded-lg bg-[var(--lp-surface)] border border-[var(--lp-border)] px-3 py-2 text-sm outline-none focus:border-[var(--lp-accent)]" />
        <button onClick={() => void createPack()}
                disabled={busy !== '' || !slug.trim() || !name.trim()}
                className="rounded-lg px-4 py-2 text-sm font-semibold text-black disabled:opacity-40"
                style={{ background: 'var(--lp-accent)' }}>
          New draft
        </button>
      </div>

      {error && <div className="text-sm text-red-400">{error}</div>}

      <div className="flex flex-wrap gap-2">
        {packs.map((p) => (
          <button key={p.slug} onClick={() => setSelected(p.slug)}
                  className={`rounded-lg border px-3 py-2 text-left transition-colors ${
                    selected === p.slug
                      ? 'border-[var(--lp-accent)]'
                      : 'border-[var(--lp-border)] hover:border-[var(--lp-text-dim)]'}`}>
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full inline-block"
                    style={{ background: p.accent || 'var(--lp-border)' }} />
              <span className="text-sm font-medium">{p.name}</span>
              <span className={`text-[10px] px-1.5 rounded-full border ${
                p.status === 'published'
                  ? 'border-emerald-700/60 text-emerald-400'
                  : 'border-[var(--lp-border)] text-[var(--lp-text-dim)]'}`}>
                {p.status}{p.status === 'published' ? ` v${p.version}` : ''}
              </span>
            </div>
            <div className="text-[11px] text-[var(--lp-text-dim)] mt-0.5">{p.slug}</div>
          </button>
        ))}
      </div>

      {detail && selected && (
        <div className="space-y-4">
          {/* Generate */}
          <div className="rounded-xl border border-[var(--lp-border)] bg-[var(--lp-surface)] p-4 space-y-2">
            <div className="flex items-center gap-2 text-sm font-semibold">
              <Wand2 size={14} style={{ color: 'var(--lp-accent)' }} /> Generate
              <span className="text-xs font-normal text-[var(--lp-text-dim)]">
                A research team drafts the whole manifest from your description.
              </span>
            </div>
            <textarea
              value={genInstructions}
              onChange={(e) => setGenInstructions(e.target.value)}
              placeholder="Describe the vertical: who the customer is, what they commission, domain vocabulary, the analyst rate to compare against…"
              rows={3}
              className="w-full rounded-lg bg-[var(--lp-bg)] border border-[var(--lp-border)] px-3 py-2 text-sm outline-none focus:border-[var(--lp-accent)] resize-y"
            />
            <div className="flex items-center gap-3">
              <button onClick={() => void generate()}
                      disabled={busy !== '' || working || !genInstructions.trim()}
                      className="rounded-lg px-3.5 py-1.5 text-sm font-semibold text-black disabled:opacity-40"
                      style={{ background: 'var(--lp-accent)' }}>
                Generate
              </button>
              {detail.generation.status === 'running' && (
                <span className="text-xs text-[var(--lp-text-dim)] flex items-center gap-1.5">
                  <Loader2 size={12} className="animate-spin" /> drafting the desk manifest…
                  <button onClick={() => void cancel('generation')} disabled={busy !== ''}
                          className="underline hover:text-[var(--lp-text)]">cancel</button>
                </span>
              )}
              {detail.generation.status === 'done' && (
                <span className="text-xs text-emerald-400">generated — review below</span>
              )}
              {detail.generation.status === 'error' && (
                <span className="text-xs text-red-400">{detail.generation.message}</span>
              )}
            </div>
          </div>

          {/* Review */}
          <div className="rounded-xl border border-[var(--lp-border)] bg-[var(--lp-surface)] p-4 space-y-3">
            <div className="text-sm font-semibold">Review the manifest</div>
            {manifest && <ManifestEditor value={manifest} onChange={setManifest} />}
            <div className="flex justify-end">
              <button onClick={() => void saveManifest()} disabled={busy !== '' || !manifest}
                      className="rounded-lg px-3.5 py-1.5 text-sm border border-[var(--lp-border)] hover:border-[var(--lp-accent)] disabled:opacity-40">
                Save manifest
              </button>
            </div>
          </div>

          {/* Evaluate + publish (the ship-gate) */}
          <div className="rounded-xl border border-[var(--lp-border)] bg-[var(--lp-surface)] p-4 space-y-3">
            <div className="flex items-center gap-2 text-sm font-semibold">
              {evalGreen
                ? <BadgeCheck size={15} className="text-emerald-400" />
                : <CircleAlert size={15} className="text-amber-400" />}
              Ship-gate
              <span className="text-xs font-normal text-[var(--lp-text-dim)]">
                The golden commission must pass its own receipts before this desk can publish.
              </span>
            </div>
            <div className="text-xs text-[var(--lp-text-dim)]">
              Golden task: {detail.pack.evals?.[0]?.brief ?? 'default overview task'}
            </div>
            <div className="flex items-center gap-3 flex-wrap">
              <button onClick={() => void evaluate()} disabled={busy !== '' || working}
                      className="rounded-lg px-3.5 py-1.5 text-sm border border-[var(--lp-border)] hover:border-[var(--lp-accent)] disabled:opacity-40">
                Run evaluation
              </button>
              {detail.eval.status === 'running' && (
                <span className="text-xs text-[var(--lp-text-dim)] flex items-center gap-1.5">
                  <Loader2 size={12} className="animate-spin" /> golden commission running…
                  <button onClick={() => void cancel('eval')} disabled={busy !== ''}
                          className="underline hover:text-[var(--lp-text)]">cancel</button>
                </span>
              )}
              {detail.eval.status === 'done' && (
                <span className={`text-xs px-2 py-0.5 rounded-full border ${
                  evalGreen ? 'border-emerald-700/60 text-emerald-400'
                            : 'border-red-700/60 text-red-400'}`}>
                  {detail.eval.verdict}
                </span>
              )}
              {detail.eval.status === 'error' && (
                <span className="text-xs text-red-400">{detail.eval.message}</span>
              )}
              <div className="flex-1" />
              <button onClick={() => void publish()}
                      disabled={busy !== '' || !evalGreen}
                      title={evalGreen ? 'Publish this desk' : 'Locked until the evaluation is green'}
                      className="rounded-lg px-4 py-1.5 text-sm font-semibold text-black disabled:opacity-40 flex items-center gap-1.5"
                      style={{ background: 'var(--lp-accent)' }}>
                <Rocket size={14} /> Publish
              </button>
              {detail.summary.status === 'published' && (
                <a href={`/desks/${selected}`}
                   className="text-sm underline" style={{ color: 'var(--lp-accent)' }}>
                  Open desk →
                </a>
              )}
            </div>
            {detail.eval.status === 'running' && <RunFeed events={events} />}
          </div>
        </div>
      )}
    </div>
  )
}
