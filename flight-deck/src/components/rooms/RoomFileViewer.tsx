import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, Bot, ChevronDown, ChevronUp, Download, Eye, Link2, Link2Off, Lock, Tag, Type, User } from 'lucide-react'
import { roomsApi, type FileText, type RoomFile } from '../../services/roomsApi'
import { TOKEN_RE, type VaultEntry } from '../../services/anonymizeApi'
import { Button, Notice, Pill, Segmented } from '../console/parts'
import { kindLabel } from '../anonymize/anonFormat'
import { IconBtn, Marked, Picker, RoomDialog, SearchField, TokenChip, type PickOption } from './roomParts'
import { baseToken, count, errorText, findAll, findRe, plural, tokenKind, type Range } from './roomText'

// One file of a room: the original next to what Claude reads, scrolling
// together line by line. Search by words, by a token, or by a real name — the
// token on Claude's side and the name on yours light up together.

type Mode = 'text' | 'token' | 'name'

const FILE_STATE: Record<string, { label: string; tone: 'emerald' | 'red' | 'amber' | 'sky' | 'zinc' }> = {
  done: { label: 'Clean', tone: 'emerald' }, blocked: { label: 'Blocked', tone: 'red' }, failed: { label: 'Failed', tone: 'amber' },
  new: { label: 'Not run yet', tone: 'sky' }, unread: { label: "Can't be read", tone: 'zinc' },
}

function lines(text: string): { line: string; at: number }[] {
  const out: { line: string; at: number }[] = []
  let at = 0
  for (const line of text.split('\n')) {
    out.push({ line, at })
    at += line.length + 1
  }
  return out
}

/** The first child of *el* whose top is at or above *y* (children are lines). */
function lineAt(el: HTMLElement, y: number): number {
  const kids = el.children
  let lo = 0, hi = kids.length - 1, best = 0
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if ((kids[mid] as HTMLElement).offsetTop <= y) { best = mid; lo = mid + 1 } else hi = mid - 1
  }
  return best
}

export function RoomFileViewer({ roomId, file, entries, onClose }: {
  roomId: string
  file: RoomFile
  entries: VaultEntry[] | null
  onClose: () => void
}) {
  const [text, setText] = useState<FileText | null>(null)
  const [error, setError] = useState('')
  const [mode, setMode] = useState<Mode>('text')
  const [query, setQuery] = useState('')
  const [token, setToken] = useState('')
  const [at, setAt] = useState(0)
  const [sync, setSync] = useState(true)
  const [side, setSide] = useState<'original' | 'clean'>('clean')
  const [busy, setBusy] = useState(false)
  const left = useRef<HTMLDivElement>(null)
  const right = useRef<HTMLDivElement>(null)
  const quiet = useRef<{ l: boolean; r: boolean }>({ l: false, r: false })

  useEffect(() => {
    roomsApi.text(roomId, file.id, true).then(setText).catch((e) => setError(errorText(e)))
  }, [roomId, file.id])

  const original = text?.original || ''
  const clean = text?.clean || ''
  const oLines = useMemo(() => lines(original), [original])
  const cLines = useMemo(() => lines(clean), [clean])

  const byToken = useMemo(() => new Map((entries || []).map((e) => [e.token, e])), [entries])
  const inFile = useMemo(() => {
    const m = new Map<string, number>()
    for (const x of clean.matchAll(TOKEN_RE)) { const b = baseToken(x[0]); m.set(b, (m.get(b) || 0) + 1) }
    return m
  }, [clean])
  const kinds = useMemo(() => {
    const k = new Map<string, number>()
    for (const t of inFile.keys()) { const kind = tokenKind(t); k.set(kind, (k.get(kind) || 0) + 1) }
    return [...k.entries()].sort((a, b) => b[1] - a[1])
  }, [inFile])

  const options: PickOption[] = useMemo(() => {
    const list = [...inFile.entries()].sort((a, b) => {
      const ka = tokenKind(a[0]), kb = tokenKind(b[0])
      return ka === kb ? Number(a[0].match(/_(\d+)⟧/)?.[1] || 0) - Number(b[0].match(/_(\d+)⟧/)?.[1] || 0) : ka.localeCompare(kb)
    })
    return list.map(([t, n]) => {
      const name = byToken.get(t)?.original || ''
      const group = kindLabel(tokenKind(t)).replace(/^./, (c) => c.toUpperCase())
      return mode === 'name'
        ? { value: t, group, search: `${name} ${t}`, meta: <TokenChip token={t} />,
            label: <span className="text-zinc-100">{name || <span className="text-zinc-500">(not in the vault)</span>}</span> }
        : { value: t, group, search: `${t} ${name}`, meta: `${n}×`,
            label: <span className="inline-flex min-w-0 items-center gap-2"><TokenChip token={t} /><span className="truncate text-zinc-500">{name}</span></span> }
    }).sort((a, b) => (a.group || '').localeCompare(b.group || ''))
  }, [inFile, byToken, mode])

  // What lights up on each side.
  const [lRanges, rRanges] = useMemo((): [Range[], Range[]] => {
    if (mode === 'text') {
      const q = query.trim()
      return q.length < 2 ? [[], []] : [findAll(original, [q]), findAll(clean, [q])]
    }
    if (!token) return [[], []]
    const m = token.match(/^⟦([A-Z][A-Z0-9_]*?)_(\d+)⟧$/)
    const re = m ? new RegExp(`⟦${m[1]}_${m[2]}(?::[a-z0-9]{1,3})?⟧`) : null
    const e = byToken.get(token)
    const names = e ? [e.original, ...e.forms, ...(e.spellings || [])] : []
    return [findAll(original, names), re ? findRe(clean, re) : []]
  }, [mode, query, token, original, clean, byToken])

  const primary: 'l' | 'r' = rRanges.length ? 'r' : 'l'
  const total = primary === 'r' ? rRanges.length : lRanges.length
  const cur = total ? Math.min(at, total - 1) : -1

  const step = useCallback((dir: 1 | -1) => { if (total) setAt((a) => (Math.min(a, total - 1) + dir + total) % total) }, [total])

  // The current match in view (in its pane; the other follows by scroll sync).
  useEffect(() => {
    if (cur < 0) return
    const pane = primary === 'r' ? right.current : left.current
    const el = pane?.querySelector('[data-current]') as HTMLElement | null
    if (!pane || !el) return
    const box = pane.getBoundingClientRect(), r = el.getBoundingClientRect()
    pane.scrollTo({ top: pane.scrollTop + (r.top - box.top) - box.height / 3, behavior: 'smooth' })
  }, [cur, primary, lRanges, rRanges, side])

  // Scroll together: the same line at the top of both panes.
  const follow = useCallback((from: 'l' | 'r') => {
    if (!sync) return
    if (quiet.current[from]) { quiet.current[from] = false; return }
    const a = from === 'l' ? left.current : right.current
    const b = from === 'l' ? right.current : left.current
    const la = a?.firstElementChild as HTMLElement | null, lb = b?.firstElementChild as HTMLElement | null
    if (!a || !b || !la || !lb) return
    let top: number
    if (la.children.length === lb.children.length && la.children.length > 1) {
      const i = lineAt(la, a.scrollTop)
      const ea = la.children[i] as HTMLElement, eb = lb.children[i] as HTMLElement
      const frac = ea.offsetHeight ? (a.scrollTop - ea.offsetTop) / ea.offsetHeight : 0
      top = eb.offsetTop + frac * eb.offsetHeight
    } else {
      const span = a.scrollHeight - a.clientHeight
      top = span > 0 ? (a.scrollTop / span) * (b.scrollHeight - b.clientHeight) : 0
    }
    if (Math.abs(b.scrollTop - top) < 1) return
    quiet.current[from === 'l' ? 'r' : 'l'] = true
    b.scrollTop = top
  }, [sync])

  const pickToken = (t: string) => { setMode('token'); setToken(t); setAt(0) }
  const replaced = Object.values(file.entities).reduce((a, b) => a + b, 0)
  const st = FILE_STATE[file.state] || FILE_STATE.unread
  const name = file.path.split('/').pop() || file.path
  const folder = file.path.includes('/') ? file.path.slice(0, file.path.lastIndexOf('/')) : ''

  const pane = (which: 'l' | 'r') => {
    const isR = which === 'r'
    const ls = isR ? cLines : oLines
    const ranges = isR ? rRanges : lRanges
    const curHere = primary === which ? cur : -1
    return (
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-2 text-xs">
          {isR ? <Bot className="h-3.5 w-3.5 text-violet-500" /> : <Lock className="h-3.5 w-3.5 text-zinc-500" />}
          <span className="font-medium text-zinc-300">{isR ? 'What Claude reads' : 'Original'}</span>
          <span className="truncate text-zinc-500">{isR ? (text?.clean_path || file.clean_path) : 'only you see this'}</span>
          {ranges.length > 0 && (
            <span className="ml-auto shrink-0 rounded-full bg-yellow-200 px-2 py-0.5 font-medium tabular-nums text-yellow-950 dark:bg-yellow-400/20 dark:text-yellow-200">
              {ranges.length} found
            </span>
          )}
        </div>
        <div ref={isR ? right : left} onScroll={() => follow(which)}
          className="relative min-h-0 flex-1 overflow-auto bg-zinc-950 px-5 py-4">
          <div className="font-[450] text-[13.5px] leading-[1.7] text-zinc-300">
            {ls.map(({ line, at: off }, i) => (
              <div key={i} className="min-h-[1.7em] whitespace-pre-wrap break-words">
                {line
                  ? <Marked text={line} offset={off} ranges={ranges} current={curHere} tokens={isR}
                      onToken={isR ? pickToken : undefined} activeToken={isR && mode !== 'text' ? token : undefined} />
                  : '​'}
              </div>
            ))}
          </div>
          {isR && !clean && <div className="text-sm text-zinc-500">Not run yet.</div>}
        </div>
      </div>
    )
  }

  return (
    <RoomDialog icon={Eye} onClose={onClose} scroll={false}
      title={name}
      subtitle={<>
        {folder && <span className="truncate">{folder}</span>}
        <Pill tone={st.tone}>{st.label}</Pill>
        {file.released && <Pill tone="emerald">Released</Pill>}
        {replaced > 0 && <span>{count(replaced, 'name')} replaced</span>}
        {kinds.slice(0, 5).map(([k, n]) => <span key={k} className="hidden md:inline">· {n} {plural(kindLabel(k), n)}</span>)}
      </>}
      actions={(file.state === 'done' || file.state === 'blocked') && (
        <Button icon={Download} busy={busy} onClick={async () => {
          setBusy(true)
          try { await roomsApi.downloadClean(roomId, file) } catch (e) { setError(errorText(e)) } finally { setBusy(false) }
        }}>Anonymized copy</Button>
      )}>
      <div className="flex flex-wrap items-center gap-2 border-b border-zinc-800 bg-zinc-900/30 px-4 py-2.5">
        <Segmented label="Search by" value={mode} onChange={(m) => { setMode(m); setAt(0) }} options={[
          { value: 'text', label: <span className="inline-flex items-center gap-1.5"><Type className="h-3.5 w-3.5" />Text</span> },
          { value: 'token', label: <span className="inline-flex items-center gap-1.5"><Tag className="h-3.5 w-3.5" />Token</span> },
          { value: 'name', label: <span className="inline-flex items-center gap-1.5"><User className="h-3.5 w-3.5" />Real name</span> },
        ]} />
        {mode === 'text' ? (
          <SearchField value={query} onChange={(v) => { setQuery(v); setAt(0) }} placeholder="Words in either side…"
            total={query.trim().length >= 2 ? total : undefined} at={cur} onStep={step} className="min-w-[14rem] flex-1" autoFocus />
        ) : (
          <div className="flex min-w-[14rem] flex-1 items-center gap-2">
            <Picker className="flex-1" value={token} onChange={(v) => { setToken(v); setAt(0) }} options={options}
              icon={mode === 'name' ? User : Tag}
              placeholder={mode === 'name' ? 'Pick a real name in this file' : 'Pick a token in this file'}
              empty={entries === null ? 'Loading the names…' : 'No tokens in this file.'} />
            {token && (
              <div className="flex shrink-0 items-center gap-1 text-xs tabular-nums text-zinc-400">
                <span>{total ? `${cur + 1} / ${total}` : 'none'}</span>
                <IconBtn label="Previous" onClick={() => step(-1)}><ChevronUp className="h-4 w-4" /></IconBtn>
                <IconBtn label="Next" onClick={() => step(1)}><ChevronDown className="h-4 w-4" /></IconBtn>
              </div>
            )}
          </div>
        )}
        <IconBtn label={sync ? 'Scrolling together — click to scroll each side on its own' : 'Scroll both sides together'}
          active={sync} onClick={() => setSync(!sync)}>
          {sync ? <Link2 className="h-4 w-4" /> : <Link2Off className="h-4 w-4" />}
        </IconBtn>
        <div className="md:hidden">
          <Segmented label="Side" value={side} onChange={setSide} options={[
            { value: 'original', label: 'Original' }, { value: 'clean', label: 'Claude' },
          ]} />
        </div>
      </div>
      {(error || file.error || file.warnings.length > 0) && (
        <div className="space-y-2 border-b border-zinc-800 px-4 py-2.5">
          {error && <Notice tone="red">{error}</Notice>}
          {file.error && <Notice tone={file.state === 'blocked' ? 'red' : 'amber'}>{file.error}</Notice>}
          {file.warnings.length > 0 && (
            <div className="flex items-start gap-2 text-xs text-amber-700 dark:text-amber-300">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{file.warnings.join(' · ')}</span>
            </div>
          )}
        </div>
      )}
      {!text ? (
        <div className="flex flex-1 items-center justify-center text-sm text-zinc-500">{error ? '' : 'Loading…'}</div>
      ) : (
        <div className="flex min-h-0 flex-1">
          <div className={`min-h-0 min-w-0 flex-1 flex-col border-r border-zinc-800 md:flex ${side === 'original' ? 'flex' : 'hidden'}`}>{pane('l')}</div>
          <div className={`min-h-0 min-w-0 flex-1 flex-col md:flex ${side === 'clean' ? 'flex' : 'hidden'}`}>{pane('r')}</div>
        </div>
      )}
    </RoomDialog>
  )
}
