import { useEffect, useState, type ReactNode } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { AlertTriangle, Download, FileText, FileUp, Inbox, Trash2 } from 'lucide-react'
import { roomsApi, type Delivery, type DeliverySummary } from '../../services/roomsApi'
import { untrustedUrlTransform } from '../../utils/untrustedContent'
import { Button, Card, Dialog, Notice, Pill, Segmented } from '../console/parts'
import { TokenText } from '../anonymize/anonParts'
import { kindLabel, kindTone } from '../anonymize/anonFormat'

// The Inbox: what Claude sent back to a room, with the real names put back
// here, for the owner only. Claude's markdown is someone else's writing, and
// once restored it holds real names: it shows no pictures and no live links —
// a picture Claude planted would fetch an address with those names in it, and
// a link would send them on with one careless click.

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

function when(ts: number): string {
  return new Date(ts * 1000).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
}

function Token({ token }: { token: string }) {
  const kind = token.replace(/^⟦|⟧$/g, '').replace(/_\d+(:[a-z0-9]+)?$/, '')
  return <span className={`rounded px-1 py-px font-mono text-[0.85em] ${kindTone(kind)}`} title={kindLabel(kind)}>{token}</span>
}

function Tokens({ list }: { list: string[] }) {
  return <>{list.slice(0, 10).map((t) => <Token key={t} token={t} />)}{list.length > 10 ? ' …' : ''}</>
}

/** A link shown as its text only; its address in the tooltip, never followed. */
function InertLink({ href, children }: { href?: string; children?: ReactNode }) {
  return <span className="underline decoration-dotted" title={href ? `Link (not opened): ${href}` : undefined}>{children}</span>
}

function InertPicture({ alt }: { alt?: string }) {
  return <span className="text-zinc-500">[picture{alt ? `: ${alt}` : ''}]</span>
}

const KIND_LABEL: Record<string, string> = { workbook: 'Excel', deck: 'PowerPoint', document: 'Word' }
const EXT_LABEL: Record<string, string> = {
  xlsx: 'Excel', pptx: 'PowerPoint', docx: 'Word', md: 'Markdown', markdown: 'Markdown', csv: 'CSV', tsv: 'TSV',
  txt: 'Text', json: 'JSON',
}
const DROP_ACCEPT = '.docx,.xlsx,.pptx,.md,.markdown,.txt,.csv,.tsv,.json'

function kindLabelOf(r: { kind?: string; file?: string }): string {
  if (r.kind === 'file') return EXT_LABEL[r.file || ''] || (r.file || 'File').toUpperCase()
  return KIND_LABEL[r.kind || ''] || ''
}

export function InboxCard({ roomId, onChanged, onDialog }: {
  roomId: string
  onChanged: () => void
  /** Told when a delivery opens or closes, so the page's own Escape stays out of the way. */
  onDialog?: (open: boolean) => void
}) {
  const [items, setItems] = useState<DeliverySummary[] | null>(null)
  const [open, setOpenRaw] = useState<number | null>(null)
  const setOpen = (seq: number | null) => { setOpenRaw(seq); onDialog?.(seq !== null) }
  const [error, setError] = useState('')
  const [tick, setTick] = useState(0)
  const [dropping, setDropping] = useState(false)

  useEffect(() => {
    roomsApi.deliveries(roomId).then((r) => setItems(r.deliveries)).catch((e) => setError(errorText(e)))
  }, [roomId, tick])

  const refresh = () => { setTick((t) => t + 1); onChanged() }

  const drop = async (file: File | undefined) => {
    if (!file) return
    setDropping(true); setError('')
    try {
      const r = await roomsApi.dropFile(roomId, file)
      refresh()
      setOpen(r.seq)
    } catch (e) {
      setError(errorText(e))
    } finally {
      setDropping(false)
    }
  }

  return (
    <Card title="From Claude" icon={Inbox}
      hint="What Claude delivered, with the real names put back here — for you alone. A file Claude made itself can be added here too: download it from Claude, then add it."
      action={
        <label className={`inline-flex cursor-pointer items-center gap-1.5 rounded-xl border border-zinc-700 bg-zinc-900/60 px-3.5 py-2 text-sm font-medium text-zinc-200 hover:border-zinc-600 hover:bg-zinc-800 ${dropping ? 'pointer-events-none opacity-50' : ''}`}>
          <FileUp className="h-4 w-4" />
          {dropping ? 'Putting the names back…' : 'Add a file Claude made'}
          <input type="file" className="sr-only" accept={DROP_ACCEPT} disabled={dropping}
            onChange={(e) => { void drop(e.target.files?.[0]); e.target.value = '' }} />
        </label>
      }
      bodyClassName="p-0">
      {error && <div className="px-5 pt-4"><Notice tone="red">{error}</Notice></div>}
      {items === null ? (
        <div className="px-5 py-6 text-sm text-zinc-500">Loading…</div>
      ) : items.length === 0 ? (
        <div className="px-5 py-6 text-sm text-zinc-500">
          Nothing yet. Ask Claude to deliver its result with the Captain Data Rooms connector.
        </div>
      ) : (
        <ul className="divide-y divide-zinc-800/80">
          {items.map((d) => (
            <li key={d.seq}>
              <button type="button" onClick={() => setOpen(d.seq)}
                className="flex w-full flex-wrap items-center gap-2 px-5 py-3 text-left hover:bg-zinc-800/30">
                {!d.seen && <span className="h-2 w-2 shrink-0 rounded-full bg-violet-500" aria-label="new" />}
                <span className="font-mono text-xs text-zinc-500">#{d.seq}</span>
                <span className="min-w-0 flex-1 truncate font-medium text-zinc-200">{d.title}</span>
                {d.report.kind && <Pill tone="sky" icon={FileText}>{kindLabelOf(d.report)}</Pill>}
                {d.report.no_tokens && <Pill tone="amber">no tokens</Pill>}
                {d.report.real_values.length > 0 && <Pill tone="red" icon={AlertTriangle}>real values</Pill>}
                {d.report.not_in_room.length > 0 && <Pill tone="amber">{d.report.not_in_room.length} unknown</Pill>}
                <Pill>{d.report.restored} put back</Pill>
                <span className="text-xs text-zinc-500">{when(d.at)} · {d.app}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {open !== null && (
        <DeliveryDialog roomId={roomId} seq={open} onClose={() => { setOpen(null); refresh() }}
          onDeleted={() => { setOpen(null); refresh() }} />
      )}
    </Card>
  )
}

function DeliveryDialog({ roomId, seq, onClose, onDeleted }: {
  roomId: string
  seq: number
  onClose: () => void
  onDeleted: () => void
}) {
  const [d, setD] = useState<Delivery | null>(null)
  const [view, setView] = useState<'names' | 'tokens'>('names')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')
  const [confirm, setConfirm] = useState(false)

  useEffect(() => {
    roomsApi.delivery(roomId, seq).then(setD).catch((e) => setError(errorText(e)))
  }, [roomId, seq])

  const act = async (label: string, work: () => Promise<unknown>) => {
    setBusy(label); setError('')
    try { await work() } catch (e) { setError(errorText(e)) } finally { setBusy('') }
  }

  return (
    <Dialog title={d ? `#${d.seq} · ${view === 'names' ? d.title : d.title_tokens}` : `#${seq}`} icon={FileText}
      onClose={onClose} wide large footer={d && (
        <div className="flex flex-wrap items-center justify-end gap-2">
          {confirm ? (
            <>
              <span className="mr-auto text-sm text-zinc-400">Delete this delivery? It can't be undone.</span>
              <Button onClick={() => setConfirm(false)}>Keep it</Button>
              <Button variant="danger" busy={busy === 'delete'}
                onClick={() => act('delete', async () => { await roomsApi.deleteDelivery(roomId, d.seq); onDeleted() })}>Delete</Button>
            </>
          ) : (
            <>
              <Button variant="danger" icon={Trash2} className="mr-auto" onClick={() => setConfirm(true)}>Delete</Button>
              {d.report.kind ? (
                <Button variant="primary" icon={Download} busy={busy === 'file'}
                  onClick={() => act('file', () => roomsApi.downloadDelivery(roomId, d.seq, 'file', d.title, d.report.file))}>
                  {kindLabelOf(d.report)} file
                </Button>
              ) : (
                <>
                  <Button icon={Download} busy={busy === 'md'}
                    onClick={() => act('md', () => roomsApi.downloadDelivery(roomId, d.seq, 'md', d.title))}>Markdown</Button>
                  <Button variant="primary" icon={Download} busy={busy === 'docx'}
                    onClick={() => act('docx', () => roomsApi.downloadDelivery(roomId, d.seq, 'docx', d.title))}>Word</Button>
                </>
              )}
            </>
          )}
        </div>
      )}>
      {error && <Notice tone="red">{error}</Notice>}
      {!d ? <div className="text-sm text-zinc-500">Loading…</div> : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-500">
            <span>{when(d.at)} · {d.app} · {d.report.characters.toLocaleString()} characters
              {d.report.parts > 1 ? ` in ${d.report.parts} parts` : ''} · {d.report.restored} put back</span>
            <div className="ml-auto">
              <Segmented label="Show" value={view} onChange={setView} options={[
                { value: 'names', label: 'With the real names' },
                { value: 'tokens', label: 'As Claude wrote it' },
              ]} />
            </div>
          </div>
          {d.report.real_values.length > 0 && (
            <Notice tone="red">
              <div className="flex flex-wrap items-center gap-1">
                Claude wrote the real value of <Tokens list={d.report.real_values} /> itself — it was told it some
                other way, or guessed it. It wasn't told it was right.
              </div>
            </Notice>
          )}
          {d.report.no_tokens && (
            <Notice tone="amber">
              No tokens were found in this file, so nothing could be put back. Claude probably wrote a codename or
              descriptions instead of the tokens — ask it to keep the tokens (⟦ORG_7⟧ …) in files it makes.
            </Notice>
          )}
          {d.report.kind && (
            <Notice tone="zinc">
              {d.report.kind === 'file'
                ? `${/^[AEIOU]/.test(kindLabelOf(d.report)) ? 'An' : 'A'} ${kindLabelOf(d.report)} file Claude made itself${d.report.source === 'owner' ? ', added by you' : ''}, with the real names put back inside.`
                : `${d.report.kind === 'workbook' ? 'An' : 'A'} ${kindLabelOf(d.report)} file, built here from what Claude sent, with the real names put back inside.`}
              {' '}Below is its text.
              {d.report.notes && d.report.notes.length > 0 && (
                <ul className="mt-1 space-y-0.5 text-xs opacity-80">{d.report.notes.slice(0, 8).map((n, i) => <li key={i}>· {n}</li>)}</ul>
              )}
            </Notice>
          )}
          {d.report.not_in_room.length > 0 && (
            <Notice tone="amber">
              <div className="flex flex-wrap items-center gap-1">
                <Tokens list={d.report.not_in_room} /> {d.report.not_in_room.length === 1 ? "isn't" : "aren't"} in
                this room, so {d.report.not_in_room.length === 1 ? 'it stays' : 'they stay'} as Claude wrote{' '}
                {d.report.not_in_room.length === 1 ? 'it' : 'them'} — Claude made {d.report.not_in_room.length === 1 ? 'it' : 'them'} up.
              </div>
            </Notice>
          )}
          {view === 'tokens' ? (
            <TokenText text={d.body_tokens} className="max-h-[60vh] overflow-auto" />
          ) : d.format === 'markdown' ? (
            <div className="fd-file-markdown max-h-[60vh] overflow-auto rounded-xl border border-zinc-800 bg-zinc-950 p-5">
              <Markdown remarkPlugins={[remarkGfm]} urlTransform={untrustedUrlTransform}
                components={{ a: InertLink, img: InertPicture }}>
                {d.body}
              </Markdown>
            </div>
          ) : (
            <div className="max-h-[60vh] overflow-auto whitespace-pre-wrap break-words rounded-xl border border-zinc-800 bg-zinc-950 p-5 text-sm text-zinc-200">
              {d.body}
            </div>
          )}
        </div>
      )}
    </Dialog>
  )
}
