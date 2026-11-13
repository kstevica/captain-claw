import { useCallback, useEffect, useState } from 'react'
import { FolderLock, Inbox, ListTodo, Plus, X } from 'lucide-react'
import { roomsApi, type RoomSummary } from '../../services/roomsApi'
import { Button, Card, EmptyState, Field, Notice, Page, PageHeader, Pill, inputCls } from '../console/parts'
import { when } from '../anonymize/anonFormat'
import { FullscreenButton } from '../library/libraryParts'
import { ClaudeAccessCard } from './RoomsAccess'
import { RoomView } from './RoomView'
import { STATE_LABEL, STATE_TONE, count, errorText } from './roomText'

// Data rooms: a virtual data room — a ZIP or a folder of files — anonymized
// into one vault. The person checks what came out, gives the parties roles,
// and releases the room; then Claude reads it through the Captain connector
// and never sees a real name. Results Claude sends back are restored here.

export function RoomsPage({ phone = false, onClose }: { phone?: boolean; onClose?: () => void }) {
  const [rooms, setRooms] = useState<RoomSummary[] | null>(null)
  const [openId, setOpenId] = useState<string | null>(null)
  const [codename, setCodename] = useState('')
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const reload = useCallback(async () => {
    try {
      setRooms((await roomsApi.list()).rooms)
    } catch (e) {
      setError(errorText(e))
    }
  }, [])

  useEffect(() => { void reload() }, [reload])

  useEffect(() => {
    if (!onClose || openId) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, openId])

  if (openId) {
    return <RoomView id={openId} phone={phone} onBack={() => { setOpenId(null); void reload() }} onClose={onClose} />
  }

  const create = async () => {
    setBusy(true); setError('')
    try {
      const room = await roomsApi.create(codename.trim(), name.trim())
      setCodename(''); setName('')
      setOpenId(room.id)
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="@container h-full overflow-y-auto">
      <Page full={!!onClose}>
        <PageHeader
          icon={FolderLock}
          title="Data rooms"
          description="Put a data room's files here — a ZIP or a folder. Captain anonymizes all of them into one vault; you check the result and release it. Claude then reads it through the Captain connector with every name replaced by a token, and what Claude sends back is restored here, for you alone."
          actions={<><FullscreenButton />{onClose && <Button variant="ghost" icon={X} onClick={onClose} aria-label="Close" />}</>}
        />
        {error && <Notice tone="red">{error}</Notice>}
        <Card title="New data room" icon={Plus} hint="The codename is what Claude sees — don't use a real name there.">
          <div className="grid gap-4 @lg:grid-cols-2">
            <Field label="Codename (Claude sees this)">
              <input className={inputCls} value={codename} onChange={(e) => setCodename(e.target.value)}
                placeholder="Project Falcon" maxLength={80} />
            </Field>
            <Field label="Your name for it (only you see this)">
              <input className={inputCls} value={name} onChange={(e) => setName(e.target.value)}
                placeholder="The deal's real name" maxLength={200} />
            </Field>
          </div>
          <div className="mt-4">
            <Button variant="primary" icon={Plus} busy={busy} disabled={!codename.trim()} onClick={create}>
              Create data room
            </Button>
          </div>
        </Card>
        {rooms && rooms.length > 0 && <ClaudeAccessCard rooms={rooms} />}
        {rooms === null ? null : rooms.length === 0 ? (
          <EmptyState icon={FolderLock} title="No data rooms yet">
            Create one above, then add a ZIP or a folder.
          </EmptyState>
        ) : (
          <div className="grid gap-3 @2xl:grid-cols-2">
            {rooms.map((r) => (
              <button key={r.id} type="button" onClick={() => setOpenId(r.id)}
                className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4 text-left transition-colors hover:border-zinc-600">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate font-medium text-zinc-100">{r.codename}</div>
                    {r.name && <div className="truncate text-sm text-zinc-500">{r.name}</div>}
                  </div>
                  <Pill tone={STATE_TONE[r.state]}>{r.job.running ? 'Running' : STATE_LABEL[r.state]}</Pill>
                </div>
                <div className="mt-3 flex flex-wrap gap-1.5 text-xs text-zinc-500">
                  <span>{count(r.counts.files, 'file')}</span>
                  {r.counts.released > 0 && <Pill tone="emerald">{r.counts.released} released</Pill>}
                  {r.counts.blocked > 0 && <Pill tone="red">{r.counts.blocked} blocked</Pill>}
                  {r.counts.failed > 0 && <Pill tone="amber">{r.counts.failed} failed</Pill>}
                  {r.inbox.unseen > 0 && <Pill tone="violet" icon={Inbox}>{r.inbox.unseen} new from Claude</Pill>}
                  {(r.tasks?.open || 0) > 0 && <Pill tone="amber" icon={ListTodo}>{r.tasks!.open} for Claude</Pill>}
                  <span className="ml-auto">{when(r.updated_at)}</span>
                </div>
              </button>
            ))}
          </div>
        )}
      </Page>
    </div>
  )
}
