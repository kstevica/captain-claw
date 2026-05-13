import type { AgentManifest, EntityDef } from '../types'
import { useAppRuntime } from '../store'

interface Props {
  manifest: AgentManifest
  entity: EntityDef
}

export function EntityDetail({ entity }: Props) {
  const selected = useAppRuntime((s) => s.selectedEntity)
  const clear = useAppRuntime((s) => s.clearEntity)

  if (!selected || selected.type !== entity.id) {
    return <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-4 text-xs text-zinc-500">No {entity.label.toLowerCase()} selected</div>
  }

  const data = selected.data ?? {}
  const titleField = Object.entries(entity.fields).find(([, f]) => f.title)?.[0] ?? 'name'
  const title = String(data[titleField] ?? selected.id)

  return (
    <section className="rounded-lg border border-zinc-800 bg-zinc-900/40">
      <header className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
        <div>
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">{entity.label}</div>
          <h2 className="text-lg font-semibold text-zinc-100">{title}</h2>
        </div>
        <button
          onClick={clear}
          className="rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
        >
          ✕
        </button>
      </header>
      <dl className="grid grid-cols-2 gap-3 px-4 py-3">
        {Object.entries(entity.fields).map(([key, def]) => (
          <div key={key}>
            <dt className="text-[10px] uppercase tracking-wide text-zinc-500">{def.label ?? key}</dt>
            <dd className="mt-0.5 text-sm text-zinc-200 break-words">
              {data[key] != null && data[key] !== '' ? String(data[key]) : <span className="text-zinc-600">—</span>}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
