import { X, Send, Tag, Wrench, Brain } from 'lucide-react'
import { useState } from 'react'
import type { InstanceInfo, Concern } from '../../types'
import { StatusBadge } from '../common/StatusBadge'

interface Props {
  instance: InstanceInfo
  concerns: Concern[]
  onClose: () => void
}

export function AgentDetail({ instance, concerns, onClose }: Props) {
  const [message, setMessage] = useState('')

  const instanceConcerns = concerns.filter(
    (c) => c.from_instance === instance.id || c.assigned_instance === instance.id
  )

  return (
    <div className="flex h-full flex-col border-l border-zinc-800 bg-zinc-900/70 w-[var(--fd-panel)]">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold">{instance.name || instance.id.slice(0, 12)}</h2>
          <span className="text-xs text-zinc-500 font-mono">{instance.id}</span>
        </div>
        <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Personas */}
        <Section icon={Brain} title="Personas">
          {instance.personas.length === 0 ? (
            <p className="text-xs text-zinc-600">No personas configured</p>
          ) : (
            instance.personas.map((p) => (
              <div key={p.name} className="mb-2 rounded-lg bg-zinc-950/50 p-3">
                <div className="text-sm font-medium">{p.name}</div>
                {p.description && <p className="mt-0.5 text-xs text-zinc-400">{p.description}</p>}
                {p.expertise_tags.length > 0 && (
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {p.expertise_tags.map((tag) => (
                      <span key={tag} className="flex items-center gap-0.5 rounded bg-zinc-800 px-1.5 py-0.5 text-xs text-zinc-500">
                        <Tag className="h-2.5 w-2.5" />{tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </Section>

        {/* Tools */}
        <Section icon={Wrench} title={`Tools (${instance.tools.length})`}>
          <div className="flex flex-wrap gap-1">
            {instance.tools.map((t) => (
              <span key={t} className="rounded bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400">{t}</span>
            ))}
          </div>
        </Section>

        {/* Concerns */}
        <Section icon={Send} title={`Concerns (${instanceConcerns.length})`}>
          {instanceConcerns.length === 0 ? (
            <p className="text-xs text-zinc-600">No recent concerns</p>
          ) : (
            instanceConcerns.slice(0, 20).map((c) => (
              <div key={c.id} className="mb-1.5 rounded-lg bg-zinc-950/50 p-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono text-zinc-500">{c.id.slice(0, 8)}</span>
                  <StatusBadge status={c.status} />
                </div>
                <p className="mt-1 text-xs text-zinc-300 line-clamp-2">{c.task}</p>
              </div>
            ))
          )}
        </Section>
      </div>

      {/* Quick message */}
      <div className="border-t border-zinc-800 p-3">
        <div className="flex gap-2">
          <input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Send a concern..."
            className="flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
          />
          <button
            className="rounded-lg bg-violet-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-40"
            disabled={!message.trim()}
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  )
}

function Section({ icon: Icon, title, children }: { icon: typeof Brain; title: string; children: React.ReactNode }) {
  return (
    <div className="border-b border-zinc-800/50 p-4">
      <div className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-zinc-500">
        <Icon className="h-3.5 w-3.5" />
        {title}
      </div>
      {children}
    </div>
  )
}
