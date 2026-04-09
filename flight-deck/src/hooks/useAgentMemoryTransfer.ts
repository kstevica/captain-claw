import { useState, useRef, useCallback } from 'react'
import { useAuthStore } from '../stores/authStore'

function fdAuthHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = {}
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

/**
 * Shared Export/Import Memory logic for agent cards.
 *
 * Talks to the Flight Deck proxy endpoints
 *   GET  /fd/agent-memory/{host}/{port}/export
 *   POST /fd/agent-memory/{host}/{port}/import
 * which forward to the target agent's /api/memory/{export,import}.
 *
 * By default only curated memory (insights + reflections) is transferred.
 * Opting into semantic export also ships a text-only dump of the agent's
 * semantic chunks — the importing agent re-embeds locally, so two agents
 * running different embedding models can still share knowledge.
 * See MEMORY_STRUCTURE.md.
 */
export interface AgentMemoryTarget {
  host: string
  port: number
  authToken: string
  /** Human-readable label used in filenames and confirmations. */
  label: string
}

export type MemoryTransferState = 'idle' | 'exporting' | 'importing'

export interface ExportOptions {
  /** Include text-only semantic chunks (target re-embeds on import). */
  includeSemantic?: boolean
  /** Max chunks to include (default 1000). */
  semanticLimit?: number
  /** Skip chunks shorter than this (default 100). */
  semanticMinChars?: number
}

export function useAgentMemoryTransfer(target: AgentMemoryTarget) {
  const [state, setState] = useState<MemoryTransferState>('idle')
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  // When true, the next import will route conflicting
  // decision/preference/workflow insights to the pending-review queue
  // instead of silently deduping them. The caller sets this via
  // `promptImport(true)` and it's consumed on the next file selection.
  const stageConflictsRef = useRef(false)

  const exportMemory = useCallback(async (opts: ExportOptions = {}) => {
    if (state !== 'idle') return
    setState('exporting')
    try {
      const params = new URLSearchParams({
        token: target.authToken || '',
        min_importance: '0',
        reflection_limit: '50',
      })
      if (opts.includeSemantic) {
        params.set('include_semantic', '1')
        params.set('semantic_limit', String(opts.semanticLimit ?? 1000))
        params.set('semantic_min_chars', String(opts.semanticMinChars ?? 100))
      }
      const url = `/fd/agent-memory/${encodeURIComponent(target.host)}/${target.port}/export?${params}`
      const res = await fetch(url, {
        headers: fdAuthHeaders(),
        credentials: 'include',
      })
      if (!res.ok) {
        throw new Error(`Export failed: ${res.status} ${await res.text().catch(() => '')}`)
      }
      const blob = await res.blob()
      const disposition = res.headers.get('content-disposition') || ''
      const match = /filename="?([^"]+)"?/.exec(disposition)
      let filename = match?.[1] || `captain-claw-memory-${target.label}.json`
      // Flag "full" bundles (including semantic) so they're easy to spot
      // on disk. The server filename doesn't know the caller's intent.
      if (opts.includeSemantic && !/-full\.json$/.test(filename)) {
        filename = filename.replace(/\.json$/, '-full.json')
      }
      const objectUrl = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = objectUrl
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(objectUrl)
    } catch (e) {
      console.error('Memory export failed', e)
      alert(`Memory export failed: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setState('idle')
    }
  }, [state, target.host, target.port, target.authToken, target.label])

  const promptImport = useCallback(
    (stageConflicts = false) => {
      if (state !== 'idle') return
      stageConflictsRef.current = stageConflicts
      fileInputRef.current?.click()
    },
    [state]
  )

  const handleFileSelected = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      e.target.value = ''
      if (!file) return
      setState('importing')
      const stageConflicts = stageConflictsRef.current
      stageConflictsRef.current = false
      try {
        const text = await file.text()
        // Fail fast on obviously-bad input.
        JSON.parse(text)
        const params = new URLSearchParams({
          token: target.authToken || '',
          min_importance: '0',
        })
        if (stageConflicts) params.set('stage_conflicts', '1')
        const url = `/fd/agent-memory/${encodeURIComponent(target.host)}/${target.port}/import?${params}`
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...fdAuthHeaders() },
          credentials: 'include',
          body: text,
        })
        if (!res.ok) {
          throw new Error(`Import failed: ${res.status} ${await res.text().catch(() => '')}`)
        }
        const result = await res.json()
        const ins = result.insights || {}
        const refs = result.reflections || {}
        const sem = result.semantic || {}
        const stagedLine =
          ins.staged && ins.staged > 0
            ? `\nStaged for review: ${ins.staged} (decision/preference/workflow conflicts)\nOpen "Pending Insights" from the Actions menu to approve/reject.`
            : ''
        let semanticLine = ''
        if (sem.available) {
          const embedNote = sem.embedding_skipped
            ? ' (embedding skipped — keyword-only retrieval for these rows)'
            : ''
          semanticLine =
            `\nSemantic — docs: ${sem.docs_upserted ?? 0}, chunks: ${sem.chunks_inserted ?? 0}, ` +
            `embedded: ${sem.embedded ?? 0}${embedNote}`
          if (sem.error) semanticLine += `\n  error: ${sem.error}`
        } else if (result.schema_version === 2 && (result.semantic || {}).note) {
          semanticLine = `\nSemantic — ${sem.note}`
        }
        alert(
          `Memory import complete (source: ${result.source_label || 'unknown'}${
            stageConflicts ? ', stage conflicts ON' : ''
          })\n\n` +
            `Insights — stored: ${ins.stored ?? 0}, deduped: ${ins.deduped ?? 0}, ` +
            `low importance: ${ins.skipped_low_importance ?? 0}, expired: ${ins.skipped_expired ?? 0}\n` +
            `Reflections — staged: ${refs.stored ?? 0}, duplicate: ${refs.skipped_duplicate ?? 0}` +
            semanticLine +
            stagedLine +
            `\n\nNote: imported reflections are staged and do NOT replace the active personality. ` +
            `Use "Merge Reflection" in the Actions menu to promote one.`
        )
      } catch (err) {
        console.error('Memory import failed', err)
        alert(`Memory import failed: ${err instanceof Error ? err.message : String(err)}`)
      } finally {
        setState('idle')
      }
    },
    [target.host, target.port, target.authToken]
  )

  return {
    state,
    busy: state !== 'idle',
    exportMemory,
    promptImport,
    fileInputRef,
    handleFileSelected,
  }
}
