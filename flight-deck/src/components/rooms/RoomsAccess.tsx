import { useCallback, useEffect, useState } from 'react'
import { AlertTriangle, BookOpen, Check, Copy, Download, Plug, Unplug } from 'lucide-react'
import { roomsApi, type RoomAlarm, type RoomExport, type RoomGrant, type RoomRead, type RoomSummary } from '../../services/roomsApi'
import { Button, Card, Notice, Pill } from '../console/parts'
import { kindLabel, kindTone } from '../anonymize/anonFormat'

// Claude's side of the data rooms: how to connect Claude, which connected app
// reads which rooms (changeable, revocable at once), and per room what Claude
// read and any alarm — a real value showing up in what Claude sent.

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

function ago(ts: number): string {
  const s = Math.max(0, Date.now() / 1000 - ts)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  return new Date(ts * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
}

function Token({ token }: { token: string }) {
  const kind = token.replace(/^⟦|⟧$/g, '').replace(/_\d+(:[a-z0-9]+)?$/, '')
  return <span className={`rounded px-1 py-px font-mono text-[0.85em] ${kindTone(kind)}`} title={kindLabel(kind)}>{token}</span>
}

const TOOL_LABEL: Record<string, string> = {
  room_overview: 'Overview', room_list_files: 'File list', room_read: 'Read', room_search: 'Search',
  room_get_file: 'File', room_deliver: 'Delivered', room_deliver_workbook: 'Delivered Excel',
  room_deliver_deck: 'Delivered PowerPoint', room_deliver_document: 'Delivered Word',
  room_deliver_file: 'Delivered a file', export: 'Download for Claude', room_tasks: 'Took tasks',
  room_get_delivery: 'Read a delivery',
}

function size(bytes: number): string {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export function ExportCard({ roomId, released, onDone, headless = false }: {
  roomId: string; released: number; onDone?: () => void; headless?: boolean
}) {
  const [textOnly, setTextOnly] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [done, setDone] = useState<RoomExport | null>(null)

  const run = async () => {
    setBusy(true); setError(''); setDone(null)
    try {
      const e = await roomsApi.exportRoom(roomId, textOnly)
      setDone(e)
      await roomsApi.downloadExport(roomId, e)
      onDone?.()
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy(false)
    }
  }

  const button = <Button variant="primary" icon={Download} busy={busy} disabled={released === 0} onClick={run}>Download ZIP</Button>
  return (
    <Card title={headless ? undefined : 'Download for Claude'} icon={headless ? undefined : Download}
      hint={headless ? undefined : 'The released files as a ZIP to add to a Claude Project — anonymized, with the roles and a README telling Claude the rules.'}
      action={headless ? undefined : button}>
      {headless && (
        <p className="mb-4 text-sm text-zinc-400">
          {released} released file{released === 1 ? '' : 's'}, anonymized, with the roles and a README telling Claude
          the rules. Add the ZIP to a Claude Project.
        </p>
      )}
      <label className="flex items-start gap-2 text-sm text-zinc-300">
        <input type="checkbox" className="mt-0.5" checked={textOnly} onChange={(e) => setTextOnly(e.target.checked)} />
        <span>Text only <span className="text-zinc-500">— Word, Excel and PowerPoint as markdown: smaller, and every Claude reads it.</span></span>
      </label>
      {headless && <div className="mt-4 flex justify-end">{button}</div>}
      {error && <div className="mt-3"><Notice tone="red">{error}</Notice></div>}
      {done && (
        <div className="mt-3 space-y-2">
          <Notice tone="emerald">
            {done.files} file(s) in the ZIP ({size(done.size)}). Results come back through the
            connector, or as a file you add under From Claude.
          </Notice>
          {done.held_back.length > 0 && (
            <Notice tone="red">
              Left out, because they still hold a real value: {done.held_back.slice(0, 8).join(', ')}
              {done.held_back.length > 8 ? ' …' : ''}. Check them, then run the room again.
            </Notice>
          )}
        </div>
      )}
    </Card>
  )
}

export function ClaudeAccessCard({ rooms }: { rooms: RoomSummary[] }) {
  const [grants, setGrants] = useState<RoomGrant[] | null>(null)
  const [url, setUrl] = useState('')
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState('')

  const reload = useCallback(() => {
    roomsApi.access()
      .then((r) => { setGrants(r.grants); setUrl(r.connector_url) })
      .catch((e) => setError(errorText(e)))
  }, [])
  useEffect(() => {
    roomsApi.access()
      .then((r) => { setGrants(r.grants); setUrl(r.connector_url) })
      .catch((e) => setError(errorText(e)))
  }, [])

  return (
    <Card title="Claude's access" icon={Plug}
      hint="Claude reads released files of the rooms you choose — anonymized text only — through the Captain Data Rooms connector.">
      {error && <Notice tone="red">{error}</Notice>}
      <div className="space-y-2 text-sm text-zinc-400">
        <p>
          In Claude: Settings → Connectors → Add custom connector, named "Captain Data Rooms", with this address.
          You sign in to Flight Deck and tick the rooms Claude may read.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <code className="min-w-0 max-w-full truncate rounded-lg border border-zinc-800 bg-zinc-950 px-2 py-1.5 font-mono text-xs text-zinc-200">{url || '…'}</code>
          <Button icon={copied ? Check : Copy} disabled={!url} onClick={async () => {
            try { await navigator.clipboard.writeText(url); setCopied(true); setTimeout(() => setCopied(false), 1500) } catch { /* refused */ }
          }}>{copied ? 'Copied' : 'Copy'}</Button>
        </div>
        <p className="text-xs text-zinc-500">The address must be reachable from the internet over HTTPS, and not blocked for Claude by a firewall or Cloudflare's AI-bot blocking.</p>
      </div>
      {grants && grants.length > 0 && (
        <ul className="mt-4 divide-y divide-zinc-800/80 rounded-xl border border-zinc-800">
          {grants.map((g) => <GrantRow key={g.client_id} grant={g} rooms={rooms} onChanged={reload} onError={setError} />)}
        </ul>
      )}
      {grants && grants.length === 0 && <p className="mt-4 text-sm text-zinc-500">No app is connected to your data rooms.</p>}
    </Card>
  )
}

function GrantRow({ grant, rooms, onChanged, onError }: {
  grant: RoomGrant
  rooms: RoomSummary[]
  onChanged: () => void
  onError: (e: string) => void
}) {
  const [chosen, setChosen] = useState<Set<string>>(() => new Set(grant.rooms.map((r) => r.id)))
  const [all, setAll] = useState(grant.all_rooms)
  const [busy, setBusy] = useState('')
  const dirty = all !== grant.all_rooms || chosen.size !== grant.rooms.length || grant.rooms.some((r) => !chosen.has(r.id))

  const save = async () => {
    setBusy('save'); onError('')
    try { await roomsApi.setAccess(grant.client_id, [...chosen], all); onChanged() } catch (e) { onError(errorText(e)) } finally { setBusy('') }
  }
  const revoke = async () => {
    setBusy('revoke'); onError('')
    try { await roomsApi.revokeAccess(grant.client_id); onChanged() } catch (e) { onError(errorText(e)) } finally { setBusy('') }
  }

  return (
    <li className="space-y-3 px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-zinc-200">{grant.app}</span>
        <span className="text-xs text-zinc-500">connected {ago(grant.created_at)}</span>
        <div className="ml-auto flex gap-2">
          {dirty && <Button variant="primary" busy={busy === 'save'} onClick={save}>Save</Button>}
          <Button variant="danger" icon={Unplug} busy={busy === 'revoke'} onClick={revoke}>Disconnect</Button>
        </div>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-2 text-sm">
        <label className="flex items-center gap-2 text-zinc-300">
          <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} />
          All rooms, and new ones
        </label>
        {rooms.map((r) => (
          <label key={r.id} className={`flex items-center gap-2 ${all ? 'text-zinc-600' : 'text-zinc-300'}`}>
            <input type="checkbox" disabled={all} checked={all || chosen.has(r.id)} onChange={(e) => {
              const next = new Set(chosen)
              if (e.target.checked) next.add(r.id); else next.delete(r.id)
              setChosen(next)
            }} />
            {r.codename}
          </label>
        ))}
      </div>
    </li>
  )
}

export function AlarmsNotice({ roomId, alarms, onSeen }: { roomId: string; alarms: RoomAlarm[]; onSeen: () => void }) {
  const unseen = alarms.filter((a) => !a.seen)
  if (!unseen.length) return null
  const tokens = [...new Set(unseen.flatMap((a) => a.tokens))]
  return (
    <Notice tone="red">
      <div className="flex flex-wrap items-start gap-3">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
        <div className="min-w-0 flex-1 space-y-1">
          <div className="font-medium">Claude sent a real value from this room ({unseen.length === 1 ? 'once' : `${unseen.length} times`}).</div>
          <div className="flex flex-wrap items-center gap-1">The real value of {tokens.slice(0, 8).map((t) => <Token key={t} token={t} />)} was in what Claude asked for.</div>
          <div className="text-xs opacity-80">
            It didn't come from this room — Claude only gets tokens here. It got there some other way: typed in the chat,
            a file given to Claude directly, or a guess. Check that conversation.
          </div>
        </div>
        <Button onClick={async () => { await roomsApi.alarmsSeen(roomId); onSeen() }}>Got it</Button>
      </div>
    </Notice>
  )
}

export function ReadsCard({ reads, headless = false }: { reads: RoomRead[]; headless?: boolean }) {
  const [all, setAll] = useState(false)
  const shown = all ? reads : reads.slice(0, headless ? 40 : 12)
  return (
    <Card title={headless ? undefined : 'What Claude read'} icon={headless ? undefined : BookOpen}
      hint={headless ? undefined : 'Every call through the connector: which file and page, never the text Claude got or what it searched for.'}
      bodyClassName="p-0">
      {reads.length === 0 ? (
        <div className="px-5 py-6 text-sm text-zinc-500">Nothing yet.</div>
      ) : (
        <>
          <ul className="divide-y divide-zinc-800/80">
            {shown.map((r, i) => (
              <li key={i} className="flex flex-wrap items-center gap-2 px-5 py-2 text-sm">
                <span className="w-20 shrink-0 text-xs text-zinc-500">{ago(r.at)}</span>
                <Pill>{TOOL_LABEL[r.tool] || r.tool}</Pill>
                {r.path && <span className="min-w-0 truncate text-zinc-300" title={r.path}>{r.path}{r.page ? ` · page ${r.page}` : ''}</span>}
                {r.alarm && <Pill tone="red">alarm</Pill>}
                <span className="ml-auto text-xs text-zinc-500">{r.app}</span>
              </li>
            ))}
          </ul>
          {reads.length > (headless ? 40 : 12) && (
            <div className="border-t border-zinc-800/80 px-5 py-2">
              <button type="button" className="text-xs text-zinc-400 hover:text-zinc-200" onClick={() => setAll(!all)}>
                {all ? 'Show fewer' : `Show all ${reads.length}`}
              </button>
            </div>
          )}
        </>
      )}
    </Card>
  )
}
