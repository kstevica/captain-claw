import { useState } from 'react'
import type { AgentManifest, ActionDef } from '../types'
import { callTool, extractText } from '../mcp'
import { useAppRuntime, applyTemplates } from '../store'
import { uploadFile } from '../files'

interface Props {
  manifest: AgentManifest
  action: ActionDef
  prefill?: Record<string, string>
  prominent?: boolean
}

export function ActionButton({ manifest, action, prefill, prominent }: Props) {
  const [open, setOpen] = useState(false)
  const [values, setValues] = useState<Record<string, string>>({})
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const selectedEntity = useAppRuntime((s) => s.selectedEntity)

  const merged = { ...(action.prefill ?? {}), ...(prefill ?? {}) }
  const prefilledArgs = applyTemplates(merged, selectedEntity)

  const onOpen = () => {
    // seed form with prefilled (stringified) values + empty for the rest
    const seed: Record<string, string> = {}
    for (const [k, v] of Object.entries(prefilledArgs)) {
      seed[k] = v == null ? '' : String(v)
    }
    for (const k of Object.keys(action.inputs)) {
      if (!(k in seed)) seed[k] = ''
    }
    setValues(seed)
    setResult(null)
    setError(null)
    setOpen(true)
  }

  const onSubmit = async () => {
    setSubmitting(true)
    setError(null)
    setResult(null)
    const args: Record<string, unknown> = { ...prefilledArgs, ...values }
    const res = await callTool(manifest.agent.id, manifest.agent.mcp_server, action.mcp_tool, args)
    setSubmitting(false)
    if (!res.ok) {
      setError(res.error ?? 'unknown error')
      return
    }
    if (action.returns === 'markdown') {
      setResult(extractText(res.result))
    } else {
      setResult('Done.')
      setTimeout(() => setOpen(false), 600)
    }
  }

  return (
    <div className="inline-block">
      <button
        onClick={onOpen}
        className={
          prominent || action.prominent
            ? 'rounded-md bg-violet-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-violet-500'
            : 'rounded-md border border-zinc-700 bg-zinc-800/60 px-3 py-1.5 text-sm text-zinc-200 hover:bg-zinc-800'
        }
      >
        {action.label}
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={() => setOpen(false)}>
          <div
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-lg rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
          >
            <h3 className="mb-3 text-sm font-semibold text-zinc-100">{action.label}</h3>
            {action.description && (
              <p className="mb-3 text-xs text-zinc-500">{action.description}</p>
            )}

            <div className="space-y-3">
              {Object.entries(action.inputs).map(([key, def]) => (
                <div key={key}>
                  <label className="mb-1 block text-[11px] uppercase tracking-wide text-zinc-500">
                    {def.label ?? key}
                    {def.required && <span className="ml-1 text-red-400">*</span>}
                  </label>
                  {def.type === 'file' ? (
                    <FileInput
                      agentId={manifest.agent.id}
                      value={values[key] ?? ''}
                      onChange={(fileId) => setValues((v) => ({ ...v, [key]: fileId }))}
                    />
                  ) : def.type === 'text' || def.type === 'markdown' ? (
                    <textarea
                      value={values[key] ?? ''}
                      onChange={(e) => setValues((v) => ({ ...v, [key]: e.target.value }))}
                      className="w-full rounded border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100"
                      rows={4}
                    />
                  ) : def.type === 'enum' && def.values ? (
                    <select
                      value={values[key] ?? ''}
                      onChange={(e) => setValues((v) => ({ ...v, [key]: e.target.value }))}
                      className="w-full rounded border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100"
                    >
                      <option value="">—</option>
                      {def.values.map((v) => (
                        <option key={v} value={v}>{v}</option>
                      ))}
                    </select>
                  ) : (
                    <input
                      value={values[key] ?? ''}
                      onChange={(e) => setValues((v) => ({ ...v, [key]: e.target.value }))}
                      className="w-full rounded border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100"
                    />
                  )}
                </div>
              ))}
            </div>

            {error && <div className="mt-3 rounded bg-red-950/50 px-3 py-2 text-xs text-red-300">{error}</div>}
            {result && (
              <div className="mt-3 max-h-64 overflow-auto rounded bg-zinc-900 px-3 py-2 text-xs text-zinc-200 whitespace-pre-wrap">
                {result}
              </div>
            )}

            <div className="mt-4 flex justify-end gap-2">
              <button
                onClick={() => setOpen(false)}
                className="rounded px-3 py-1.5 text-sm text-zinc-400 hover:text-zinc-200"
              >
                Close
              </button>
              <button
                onClick={onSubmit}
                disabled={submitting}
                className="rounded bg-violet-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-50"
              >
                {submitting ? 'Running…' : 'Run'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

interface FileInputProps {
  agentId: string
  value: string
  onChange: (fileId: string) => void
}

function FileInput({ agentId, value, onChange }: FileInputProps) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const onPick = async (file: File) => {
    setBusy(true)
    setError(null)
    try {
      const meta = await uploadFile(agentId, file)
      onChange(meta.file_id)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-1">
      <input
        type="file"
        disabled={busy}
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) onPick(f)
          e.target.value = ''
        }}
        className="block w-full text-xs text-zinc-300 file:mr-3 file:rounded file:border-0 file:bg-zinc-800 file:px-3 file:py-1.5 file:text-xs file:text-zinc-200 hover:file:bg-zinc-700"
      />
      {value && (
        <div className="font-mono text-[10px] text-zinc-500">file_id: {value}</div>
      )}
      {busy && <div className="text-[10px] text-zinc-500">Uploading…</div>}
      {error && <div className="text-[10px] text-red-400">{error}</div>}
    </div>
  )
}
