import { useState } from 'react'
import {
  Bot, ClipboardPaste, Database, Download, Eye, Folder, Link2, Loader2, MessagesSquare, Plus, Trash2, UploadCloud,
  type LucideIcon,
} from 'lucide-react'
import {
  addFactoryData, addRunData, dropFactoryData, dropRunData, factoryDataUrl,
  type FactoryDataItem,
} from '../../services/firstMate'
import { vfsApi } from '../../services/vfsApi'
import { getFileTypeGroup, isViewable, type AgentFile } from '../../services/fileTransfer'
import { AddDataPanel, FileGlyph, type DataSource } from '../context/AddData'
import { sizeText } from '../../utils/format'
import { FileViewer } from '../agents/FileViewer'
import { errText } from './ui'
import type { RunDataState } from './useRunData'

// Extra data: what a factory hands every run (kept with the factory), and what one run
// was given (in its folder's extra/). Both come in through the shared Add data panel.

const START_CHIPS: { source: DataSource; label: string; icon: LucideIcon }[] = [
  { source: 'upload', label: 'Upload', icon: UploadCloud },
  { source: 'paste', label: 'Paste', icon: ClipboardPaste },
  { source: 'links', label: 'Links', icon: Link2 },
  { source: 'agents', label: "Agents' files", icon: Bot },
  { source: 'deck', label: 'Flight Deck', icon: Folder },
  { source: 'conversation', label: 'Conversation', icon: MessagesSquare },
]

const extOf = (name: string) => (name.includes('.') ? name.slice(name.lastIndexOf('.')).toLowerCase() : '')
const asFile = (name: string, size: number, mtime: number, logical: string): AgentFile => {
  const group = getFileTypeGroup({ extension: extOf(name), filename: name } as AgentFile)
  return {
    logical, physical: logical, filename: name, extension: extOf(name), exists: true, size, modified: mtime, mime_type: '',
    is_text: ['markdown', 'code', 'data', 'text', 'html'].includes(group), source: 'workspace',
  }
}

function whenText(at: string | number): string {
  const d = typeof at === 'number' ? new Date(at * 1000) : new Date(at)
  if (Number.isNaN(d.getTime())) return ''
  const s = (Date.now() - d.getTime()) / 1000
  if (s < 90) return 'just now'
  if (s < 5400) return `${Math.round(s / 60)} min ago`
  if (s < 129_600) return `${Math.round(s / 3600)} h ago`
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

interface Card { key: string; name: string; size: number; origin: string; when: string; view: string; download: string; logical: string; mtime: number }

/** One file of extra data: what it is, where it came from, and open / download / take off. */
function DataCard({ card, onOpen, onRemove, removing }: { card: Card; onOpen?: () => void; onRemove: () => void; removing: boolean }) {
  const btn = 'flex h-8 w-8 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200'
  return (
    <li className="group fd-fade flex items-center gap-3 rounded-2xl border border-[var(--fd-line)] bg-[var(--fd-surface-raised)] p-2.5 pr-1.5 transition-all hover:border-violet-500/30 hover:shadow-sm">
      <button onClick={onOpen} disabled={!onOpen} title={onOpen ? `Open ${card.name}` : card.name}
        className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-zinc-800/70 transition-transform enabled:hover:scale-105">
        <FileGlyph name={card.name} className="h-5 w-5" />
      </button>
      <div className="min-w-0 flex-1">
        <p className="truncate text-[13px] font-medium text-zinc-200" title={card.name}>{card.name}</p>
        <p className="truncate text-[11px] text-zinc-500">{[card.origin, sizeText(card.size), card.when].filter(Boolean).join(' · ')}</p>
      </div>
      {onOpen && <button onClick={onOpen} className={btn} title="Open" aria-label={`Open ${card.name}`}><Eye className="h-3.5 w-3.5" /></button>}
      <a href={card.download} className={btn} title="Download" aria-label={`Download ${card.name}`}><Download className="h-3.5 w-3.5" /></a>
      <button onClick={onRemove} disabled={removing} className={`${btn} hover:!text-red-600 dark:hover:!text-red-300`} title="Take it off" aria-label={`Take ${card.name} off`}>
        {removing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
      </button>
    </li>
  )
}

function useViewer() {
  const [viewing, setViewing] = useState<Card | null>(null)
  const open = (c: Card) => {
    const file = asFile(c.name, c.size, c.mtime, c.logical)
    if (getFileTypeGroup(file) === 'pdf') window.open(c.view, '_blank', 'noopener')
    else setViewing(c)
  }
  const canOpen = (c: Card) => isViewable(asFile(c.name, c.size, c.mtime, c.logical))
  const node = viewing && (
    <div data-above-drawer className="contents">
      <FileViewer file={asFile(viewing.name, viewing.size, viewing.mtime, viewing.logical)} readOnly
        urls={{ view: viewing.view, download: viewing.download }} onClose={() => setViewing(null)} />
    </div>
  )
  return { open, canOpen, node }
}

/** The factory's Data tab. `items` come with the factory; a change reloads it. */
export function FactoryData({ factoryId, title, items, onChanged, onNotice }: {
  factoryId: string
  title: string
  items: FactoryDataItem[]
  onChanged: () => void
  onNotice: (text: string) => void
}) {
  const [panel, setPanel] = useState<DataSource | null>(null)
  const [removing, setRemoving] = useState('')
  const [error, setError] = useState('')
  const viewer = useViewer()
  const total = items.reduce((n, i) => n + i.size, 0)
  const cards: Card[] = items.map((i) => ({
    key: i.id, name: i.name, size: i.size, origin: i.origin, when: whenText(i.added_at), mtime: Date.parse(i.added_at) / 1000,
    view: factoryDataUrl(factoryId, i.id), download: factoryDataUrl(factoryId, i.id, true), logical: `first-mate:${factoryId}/${i.name}`,
  }))

  const remove = async (id: string, name: string) => {
    setRemoving(id); setError('')
    try {
      await dropFactoryData(factoryId, id)
      onNotice(`${name} taken off: runs from now on won't get it`)
      onChanged()
    } catch (e) {
      setError(errText(e))
    } finally {
      setRemoving('')
    }
  }

  return (
    <div className="space-y-4">
      <section className="relative overflow-hidden rounded-2xl border border-[var(--fd-line)] bg-gradient-to-br from-violet-500/[0.08] via-transparent to-fuchsia-500/[0.06] p-4">
        <div className="flex items-start gap-3">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-fuchsia-500 text-white shadow-md shadow-violet-500/20">
            <Database className="h-5 w-5" />
          </span>
          <div className="min-w-0 flex-1">
            <h3 className="text-sm font-semibold text-zinc-100">Extra data</h3>
            <p className="mt-0.5 text-xs text-zinc-500">
              Every run from now on gets a copy in its folder, and its workers read what bears on the work. A list to start from,
              rules, examples, a conversation where you explained what you want.
            </p>
          </div>
          {items.length > 0 && (
            <button onClick={() => setPanel('upload')} className="inline-flex h-9 shrink-0 items-center gap-1.5 rounded-full bg-violet-600 px-3.5 text-xs font-semibold text-white shadow-sm transition-colors hover:bg-violet-500">
              <Plus className="h-3.5 w-3.5" />Add data
            </button>
          )}
        </div>
        {items.length === 0 && (
          <div className="mt-4 flex flex-wrap gap-2">
            {START_CHIPS.map(({ source, label, icon: Icon }) => (
              <button key={source} onClick={() => setPanel(source)}
                className="inline-flex items-center gap-1.5 rounded-full border border-[var(--fd-line)] bg-[var(--fd-surface-raised)] px-3 py-1.5 text-xs font-medium text-zinc-300 transition-all hover:-translate-y-px hover:border-violet-500/40 hover:text-zinc-100">
                <Icon className="h-3.5 w-3.5 text-violet-600 dark:text-violet-300" />{label}
              </button>
            ))}
          </div>
        )}
      </section>

      {error && <p role="alert" className="text-xs text-red-700 dark:text-red-300">{error}</p>}

      {items.length > 0 && (
        <>
          <p className="px-1 text-[11px] font-medium text-zinc-500">{items.length} file{items.length === 1 ? '' : 's'} · {sizeText(total)}</p>
          <ul className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,22rem),1fr))] gap-2">
            {cards.map((c) => (
              <DataCard key={c.key} card={c} removing={removing === c.key}
                onOpen={viewer.canOpen(c) ? () => viewer.open(c) : undefined} onRemove={() => void remove(c.key, c.name)} />
            ))}
          </ul>
        </>
      )}

      {panel && (
        <AddDataPanel start={panel} title={`Extra data for ${title}`}
          subtitle="Every run from now on gets these in its folder."
          action={(n) => (n ? `Add ${n} to the factory` : 'Add to the factory')}
          onDone={async (staged) => {
            const got = await addFactoryData(factoryId, staged.map((s) => s.id))
            onNotice(`${got.added.length} file${got.added.length === 1 ? '' : 's'} added: every run from now on gets ${got.added.length === 1 ? 'it' : 'them'}`)
            onChanged()
          }}
          onClose={() => setPanel(null)} />
      )}
      {viewer.node}
    </div>
  )
}

/** One run's extra data: its folder's extra/ (what the factory gave every run, and what
 *  was given to this one), with more to add while it can still use it. Shown on the
 *  run's Extra data tab. */
export function RunDataList({ runId, going, stopped, state }: { runId: string; going: boolean; stopped: boolean; state: RunDataState }) {
  const { data, setData, error, setError } = state
  const [panel, setPanel] = useState(false)
  const [removing, setRemoving] = useState('')
  const viewer = useViewer()

  const canAdd = going || stopped
  if (!data) {
    return error ? <p className="text-xs text-red-700 dark:text-red-300">{error}</p>
      : <p className="flex items-center gap-1.5 text-xs text-zinc-500"><Loader2 className="h-3 w-3 animate-spin" />Loading its data…</p>
  }
  const ref = { project: data.project, owner: '' }
  const cards: Card[] = data.files.map((f) => {
    const path = `${data.folder}/${f.path}`
    return {
      key: f.path, name: f.name, size: f.size, origin: f.origin, when: whenText(f.mtime), mtime: f.mtime, logical: `vfs:${data.project}/${path}`,
      view: vfsApi.viewUrl(ref, path, f.mtime), download: vfsApi.downloadUrl(ref, path),
    }
  })
  const total = data.files.reduce((n, f) => n + f.size, 0)

  const remove = async (path: string) => {
    setRemoving(path); setError('')
    try {
      setData(await dropRunData(runId, path))
    } catch (e) {
      setError(errText(e))
    } finally {
      setRemoving('')
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <p className="min-w-0 flex-1 text-xs text-zinc-500">
          {cards.length
            ? <>{cards.length} file{cards.length === 1 ? '' : 's'} · {sizeText(total)} in its <span className="font-mono text-[11px]">extra/</span> folder: what the factory gives every run, and what was added to this one.</>
            : canAdd ? 'Nothing given yet. What you add here, its next steps read.' : 'It was given no extra data.'}
        </p>
        {canAdd && (
          <button onClick={() => setPanel(true)} className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-full bg-violet-600 px-3 text-xs font-semibold text-white shadow-sm transition-colors hover:bg-violet-500">
            <Plus className="h-3.5 w-3.5" />Add data
          </button>
        )}
      </div>
      {error && <p role="alert" className="text-xs text-red-700 dark:text-red-300">{error}</p>}
      {cards.length > 0 && (
        <ul className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,22rem),1fr))] gap-2">
          {cards.map((c) => (
            <DataCard key={c.key} card={c} removing={removing === c.key}
              onOpen={viewer.canOpen(c) ? () => viewer.open(c) : undefined} onRemove={() => void remove(c.key)} />
          ))}
        </ul>
      )}
      {panel && (
        <AddDataPanel title="Extra data for this run"
          subtitle={going ? 'Its next steps see these.' : 'It sees these when you press Continue.'}
          action={(n) => (n ? `Add ${n} to this run` : 'Add to this run')}
          onDone={async (staged) => { setData(await addRunData(runId, staged.map((s) => s.id))) }}
          onClose={() => setPanel(false)} />
      )}
      {viewer.node}
    </div>
  )
}
