const colors: Record<string, string> = {
  connected: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  disconnected: 'bg-zinc-700/30 text-zinc-500 border-zinc-600/30',
  running: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  completed: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  failed: 'bg-red-500/20 text-red-400 border-red-500/30',
  timeout: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  pending: 'bg-zinc-700/30 text-zinc-400 border-zinc-600/30',
  assigned: 'bg-violet-500/20 text-violet-400 border-violet-500/30',
  in_progress: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  responded: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  closed: 'bg-zinc-700/30 text-zinc-500 border-zinc-600/30',
  queued: 'bg-zinc-700/30 text-zinc-400 border-zinc-600/30',
  waiting: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  pending_approval: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  retrying: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  paused: 'bg-zinc-700/30 text-zinc-400 border-zinc-600/30',
  skipped: 'bg-zinc-700/30 text-zinc-500 border-zinc-600/30',
  draft: 'bg-zinc-700/30 text-zinc-400 border-zinc-600/30',
  ready: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
  decomposing: 'bg-violet-500/20 text-violet-400 border-violet-500/30',
  cancelled: 'bg-zinc-700/30 text-zinc-500 border-zinc-600/30',
}

const pulseStatuses = new Set(['running', 'in_progress', 'decomposing', 'retrying'])

export function StatusBadge({ status }: { status: string }) {
  const cls = colors[status] ?? colors.pending
  const pulse = pulseStatuses.has(status)

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium ${cls}`}>
      {pulse && (
        <span className="relative flex h-1.5 w-1.5">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-current opacity-75" />
          <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-current" />
        </span>
      )}
      {status.replace(/_/g, ' ')}
    </span>
  )
}
