import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle, Bot, ChevronDown, ChevronUp, Download, Eye, EyeOff, Image as ImageIcon, Link2, Link2Off, Lock,
  ShieldAlert, Tag, Type, User,
} from 'lucide-react'
import { roomsApi, type FileText, type Leftover, type RoomFile } from '../../services/roomsApi'
import { TOKEN_RE, type VaultEntry } from '../../services/anonymizeApi'
import { Button, Notice, Pill, Segmented } from '../console/parts'
import { kindLabel } from '../anonymize/anonFormat'
import { IconBtn, Marked, Picker, RoomDialog, SearchField, TokenChip, type PickOption } from './roomParts'
import {
  baseToken, colName, count, errorText, findAll, findRe, parseSheets, plural, sheetAt, tokenKind,
  type Range, type Sheet, type SheetCell,
} from './roomText'

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

// ── a workbook, as sheets ──────────────────────────────────────────

interface MarkProps {
  ranges: Range[]
  current: number
  tokens: boolean
  onToken?: (token: string) => void
  activeToken?: string
}

// An amount, a count, a date or a time: right-aligned, as a spreadsheet does.
const NUMBERISH = /^[-+(]?\d[\d.,: ]*(?:%|\))?$|^\d{4}-\d{2}-\d{2}(?: \d{1,2}:\d{2}(?::\d{2})?)?$/
const GRID_HEAD = 'border-b border-r border-zinc-800 bg-zinc-900 py-1 text-[11px] font-normal text-zinc-500'

/** A cell's text: line breaks kept (<br> in the room's text), search marks and tokens drawn. */
function CellText({ cell, ...mark }: { cell: SheetCell } & MarkProps) {
  const parts = cell.text.split('<br>')
  // Where each part starts: the parts before it and a <br> (4 characters) after each.
  const starts = parts.map((_, i) => cell.at + parts.slice(0, i).reduce((n, p) => n + p.length + 4, 0))
  return (
    <>
      {parts.map((p, i) => (
        <Fragment key={i}>{i > 0 && <br />}<Marked text={p} offset={starts[i]} {...mark} /></Fragment>
      ))}
    </>
  )
}

/** Column widths (in characters) both sides share, so the two grids line up
 *  cell for cell: the longest line in each column, on either side. */
function sheetWidths(...sheets: (Sheet | undefined)[]): number[] {
  const out: number[] = []
  for (const s of sheets) {
    for (const row of s?.rows || []) {
      row.forEach((cell, c) => {
        const longest = Math.max(...cell.text.split('<br>').map((p) => p.length))
        out[c] = Math.max(out[c] || 0, longest)
      })
    }
  }
  return out.map((n) => Math.min(40, Math.max(4, n + 1)))
}

/** One sheet as a grid: column letters, row numbers, the cells. */
function SheetGrid({ sheet, widths, ...mark }: { sheet: Sheet; widths: number[] } & MarkProps) {
  if (sheet.kind === 'pictures' || !sheet.rows.length) {
    return (
      <div data-lines="" className="space-y-2 px-5 py-4 text-[13px] leading-relaxed text-zinc-300">
        {sheet.notes.length === 0 && <div className="text-zinc-500">Nothing on this sheet.</div>}
        {sheet.notes.map((n, i) => (
          <div key={i} className="whitespace-pre-wrap break-words"><Marked text={n.text} offset={n.at} {...mark} /></div>
        ))}
      </div>
    )
  }
  const cols = Array.from({ length: Math.max(1, sheet.width, widths.length) }, (_, i) => i)
  const ch = cols.map((c) => widths[c] || 4)
  // Fixed widths, one line per cell (a cell's own line breaks kept): both
  // sides draw the same grid, whatever names or tokens are in it.
  return (
    <table className="table-fixed border-separate border-spacing-0 text-[12.5px] leading-5 text-zinc-200"
      style={{ width: `calc(3rem + ${ch.reduce((a, b) => a + b, 0)}ch + ${cols.length}rem)` }}>
      <colgroup>
        <col style={{ width: '3rem' }} />
        {ch.map((w, c) => <col key={c} style={{ width: `calc(${w}ch + 1rem)` }} />)}
      </colgroup>
      <thead>
        <tr>
          <th className={`${GRID_HEAD} sticky left-0 top-0 z-20`} />
          {cols.map((c) => (
            <th key={c} className={`${GRID_HEAD} sticky top-0 z-10 px-2 text-center`}>{colName(c)}</th>
          ))}
        </tr>
      </thead>
      <tbody data-lines="">
        {sheet.rows.map((row, r) => (
          <tr key={r}>
            <th className={`${GRID_HEAD} sticky left-0 z-10 px-2 text-right tabular-nums`}>{r + 1}</th>
            {cols.map((c) => {
              const cell = row[c]
              // The room's text names an empty cell of the first row by its letter.
              const blank = !cell || !cell.text || (r === 0 && cell.text === colName(c))
              return (
                <td key={c} title={blank ? undefined : cell.text.replaceAll('<br>', '\n')}
                  className={`overflow-hidden text-ellipsis whitespace-nowrap border-b border-r border-zinc-800 px-2 py-1 align-top ${
                    !blank && NUMBERISH.test(cell.text) ? 'text-right tabular-nums' : ''}`}>
                  {blank ? null : <CellText cell={cell} {...mark} />}
                </td>
              )
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** A workbook's sheets as tabs, as a spreadsheet shows them — with how many
 *  search matches each has. */
function SheetTabs({ sheets, active, onPick, ranges, tokens }: {
  sheets: Sheet[]; active: number; onPick: (i: number) => void; ranges: Range[]; tokens: boolean
}) {
  return (
    <div className="flex shrink-0 items-center gap-1 overflow-x-auto border-t border-zinc-800 bg-zinc-900/40 px-2 py-1.5" role="tablist">
      {sheets.map((s, i) => {
        const n = ranges.filter((r) => r[0] >= s.start && r[0] < s.end).length
        return (
          <button key={i} type="button" role="tab" aria-selected={i === active} onClick={() => onPick(i)}
            className={`inline-flex shrink-0 items-center gap-1.5 rounded-md px-2.5 py-1 text-xs transition-colors ${
              i === active ? 'bg-zinc-950 font-medium text-zinc-100 shadow-sm ring-1 ring-zinc-700' : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'}`}>
            {s.kind === 'pictures' && <ImageIcon className="h-3 w-3" />}
            <span className="max-w-[14rem] truncate">
              {tokens && s.kind === 'sheet' ? <Marked text={s.name} offset={s.nameAt} ranges={[]} tokens /> : s.name}
            </span>
            {n > 0 && (
              <span className="rounded-full bg-yellow-200 px-1.5 text-[10px] font-medium tabular-nums text-yellow-950 dark:bg-yellow-400/20 dark:text-yellow-200">{n}</span>
            )}
          </button>
        )
      })}
    </div>
  )
}

/** What the leak check found in this file, as written: what it hid with the
 *  name's own token, and what is still there (why the file is blocked) — with
 *  a way to show each, and to say a word isn't a name here. */
function LeftoversPanel({ roomId, leftovers, byToken, onShowText, onShowToken, onChanged }: {
  roomId: string
  leftovers: { hidden: Leftover[]; left: Leftover[] }
  byToken: Map<string, VaultEntry>
  onShowText: (text: string) => void
  onShowToken: (token: string) => void
  onChanged?: () => void
}) {
  const [asking, setAsking] = useState('')
  const [busy, setBusy] = useState('')
  const [said, setSaid] = useState<Record<string, string>>({})
  const [error, setError] = useState('')
  const { hidden, left } = leftovers
  if (!hidden.length && !left.length) return null

  const allow = async (key: string, x: Leftover, rerun: boolean) => {
    setBusy(key); setError('')
    try {
      const r = await roomsApi.allow(roomId, { text: x.text, tokens: [x.token], rerun })
      setSaid((s) => ({
        ...s, [key]: rerun
          ? `Left as written from now on. ${r.requeued ? `${count(r.requeued, 'file')} will run again from the original` : 'Run the room'} to bring the word back.`
          : 'Left as written. Click Check again at the top to clear the block.',
      }))
      setAsking('')
      onChanged?.()
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy('')
    }
  }

  const row = (x: Leftover, i: number, kind: 'left' | 'hidden') => {
    const key = `${kind}:${x.token}:${x.text}:${i}`
    const real = byToken.get(x.token)?.original || ''
    return (
      <li key={key} className="rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2">
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5 text-sm">
          {x.text
            ? <mark className="rounded-md bg-yellow-200 px-1.5 py-px font-medium text-yellow-950 dark:bg-yellow-400/20 dark:text-yellow-200">{x.text}</mark>
            : <span className="text-xs italic text-zinc-500">found, but not placed in the text</span>}
          <span className="text-xs text-zinc-500">{kind === 'hidden' ? 'now' : 'is'}</span>
          <TokenChip token={x.token} onClick={() => onShowToken(x.token)} />
          {real && real.toLowerCase() !== x.text.toLowerCase() && (
            <span className="max-w-[16rem] truncate text-xs text-zinc-500" title={real}>({real})</span>
          )}
          <span className="text-xs tabular-nums text-zinc-500">
            {x.count}×{x.where === 'name' ? ' · in the file’s name' : ''}
          </span>
          <span className="ml-auto flex flex-wrap items-center gap-1.5">
            {said[key] ? <span className="text-xs text-emerald-700 dark:text-emerald-300">{said[key]}</span> : asking === key ? (
              <>
                <span className="text-xs text-zinc-300">Leave “{x.text}” as written in every file of this room?</span>
                <Button variant="primary" className="!px-2.5 !py-1 !text-xs" busy={busy === key}
                  onClick={() => allow(key, x, kind === 'hidden')}>Yes, it’s not a name</Button>
                <Button className="!px-2.5 !py-1 !text-xs" onClick={() => setAsking('')}>Cancel</Button>
              </>
            ) : (
              <>
                {x.text && x.where !== 'name' && (
                  <Button className="!px-2.5 !py-1 !text-xs" icon={Eye}
                    onClick={() => (kind === 'hidden' ? onShowToken(x.token) : onShowText(x.text))}>Show</Button>
                )}
                {x.text && (
                  <Button className="!px-2.5 !py-1 !text-xs" onClick={() => setAsking(key)}>
                    {kind === 'hidden' ? 'Not a name — put it back' : 'Not a name here'}
                  </Button>
                )}
              </>
            )}
          </span>
        </div>
      </li>
    )
  }

  return (
    <div className="max-h-[40%] shrink-0 space-y-3 overflow-auto border-b border-zinc-800 px-4 py-3">
      {error && <Notice tone="red">{error}</Notice>}
      {left.length > 0 && (
        <section className="rounded-2xl border border-red-500/30 bg-red-500/[0.04] p-3">
          <div className="flex items-start gap-2.5">
            <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-red-500" />
            <div className="min-w-0">
              <div className="text-sm font-medium text-zinc-100">Still in this file — why it’s blocked</div>
              <p className="mt-0.5 text-xs leading-relaxed text-zinc-500">
                Captain couldn’t put a token in its place: the name may run across two cells, sit in a sheet name or
                a formula, or in the file’s name. If a word isn’t a name here, leave it as written. If it is, remove
                the file, or fix the original and add it again.
              </p>
            </div>
          </div>
          <ul className="mt-2.5 space-y-1.5">{left.map((x, i) => row(x, i, 'left'))}</ul>
        </section>
      )}
      {hidden.length > 0 && (
        <section className="rounded-2xl border border-zinc-800 bg-zinc-900/30 p-3">
          <div className="flex items-start gap-2.5">
            <EyeOff className="mt-0.5 h-4 w-4 shrink-0 text-violet-500" />
            <div className="min-w-0">
              <div className="text-sm font-medium text-zinc-100">Hidden after the check</div>
              <p className="mt-0.5 text-xs leading-relaxed text-zinc-500">
                The sweep leaves a surname on its own, or a company written in lower case, to the trusted model. The
                check still found these real values, so each got the token of the name it belongs to.
              </p>
            </div>
          </div>
          <ul className="mt-2.5 space-y-1.5">{hidden.map((x, i) => row(x, i, 'hidden'))}</ul>
        </section>
      )}
    </div>
  )
}

export function RoomFileViewer({ roomId, file, entries, onClose, onChanged }: {
  roomId: string
  file: RoomFile
  entries: VaultEntry[] | null
  onClose: () => void
  onChanged?: () => void
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
  // A workbook: its sheets as tabs and grids, on both sides.
  const oSheets = useMemo(() => parseSheets(original), [original])
  const cSheets = useMemo(() => parseSheets(clean), [clean])
  // The sheet you picked — until the search moves to a match on another one.
  const [pick, setPick] = useState<{ idx: number; at: number }>({ idx: 0, at: -1 })

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

  const pSheets = primary === 'r' ? cSheets : oSheets
  const pRanges = primary === 'r' ? rRanges : lRanges
  const matchSheet = cur >= 0 && pSheets && pRanges[cur] ? sheetAt(pSheets, pRanges[cur][0]) : -1
  const sheetIdx = matchSheet >= 0 && pick.at !== cur ? matchSheet : pick.idx
  const pickSheet = (i: number) => setPick({ idx: i, at: cur })
  const widths = useMemo(() => sheetWidths(oSheets?.[sheetIdx], cSheets?.[sheetIdx]), [oSheets, cSheets, sheetIdx])

  // The current match in view (in its pane; the other follows by scroll sync).
  useEffect(() => {
    if (cur < 0) return
    const pane = primary === 'r' ? right.current : left.current
    const el = pane?.querySelector('[data-current]') as HTMLElement | null
    if (!pane || !el) return
    const box = pane.getBoundingClientRect(), r = el.getBoundingClientRect()
    pane.scrollTo({
      top: pane.scrollTop + (r.top - box.top) - box.height / 3,
      left: r.left < box.left + 48 || r.right > box.right ? pane.scrollLeft + (r.left - box.left) - box.width / 4 : pane.scrollLeft,
      behavior: 'smooth',
    })
  }, [cur, primary, lRanges, rRanges, side, sheetIdx])

  // Scroll together: the same line at the top of both panes.
  const follow = useCallback((from: 'l' | 'r') => {
    if (!sync) return
    if (quiet.current[from]) { quiet.current[from] = false; return }
    const a = from === 'l' ? left.current : right.current
    const b = from === 'l' ? right.current : left.current
    // The lines (or a sheet's rows) of each side.
    const linesOf = (el: HTMLElement | null) =>
      (el?.querySelector('[data-lines]') as HTMLElement | null) ?? (el?.firstElementChild as HTMLElement | null)
    const la = linesOf(a), lb = linesOf(b)
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
    // Side to side too (a wide sheet): the same share of the way across.
    const wide = a.scrollWidth - a.clientWidth
    const leftTo = wide > 0 ? (a.scrollLeft / wide) * Math.max(0, b.scrollWidth - b.clientWidth) : 0
    if (Math.abs(b.scrollTop - top) < 1 && Math.abs(b.scrollLeft - leftTo) < 1) return
    quiet.current[from === 'l' ? 'r' : 'l'] = true
    b.scrollTop = top
    b.scrollLeft = leftTo
  }, [sync])

  const pickToken = (t: string) => { setMode('token'); setToken(t); setAt(0) }
  // A blocked file whose leftovers are listed below: the list says it better than the error line.
  const explained = file.state === 'blocked' && (text?.leftovers?.left.length || 0) > 0
  const replaced = Object.values(file.entities).reduce((a, b) => a + b, 0)
  const st = FILE_STATE[file.state] || FILE_STATE.unread
  const name = file.path.split('/').pop() || file.path
  const folder = file.path.includes('/') ? file.path.slice(0, file.path.lastIndexOf('/')) : ''

  const pane = (which: 'l' | 'r') => {
    const isR = which === 'r'
    const ls = isR ? cLines : oLines
    const ranges = isR ? rRanges : lRanges
    const curHere = primary === which ? cur : -1
    const sheets = isR ? cSheets : oSheets
    const cut = isR ? text?.clean_cut : text?.original_cut
    const mark: MarkProps = {
      ranges, current: curHere, tokens: isR, onToken: isR ? pickToken : undefined,
      activeToken: isR && mode !== 'text' ? token : undefined,
    }
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
        {sheets ? (
          <>
            <div ref={isR ? right : left} onScroll={() => follow(which)} className="relative min-h-0 flex-1 overflow-auto bg-zinc-950">
              {sheets[sheetIdx]
                ? <SheetGrid sheet={sheets[sheetIdx]} widths={widths} {...mark} />
                : <div className="px-5 py-4 text-sm text-zinc-500">No such sheet on this side.</div>}
              {cut && (
                <div className="sticky left-0 px-4 py-3 text-xs text-zinc-500">
                  Showing the start of this file — the rest is in the {isR ? 'anonymized copy' : 'original'}.
                </div>
              )}
            </div>
            <SheetTabs sheets={sheets} active={sheetIdx} onPick={pickSheet} ranges={ranges} tokens={isR} />
          </>
        ) : (
        <div ref={isR ? right : left} onScroll={() => follow(which)}
          className="relative min-h-0 flex-1 overflow-auto bg-zinc-950 px-5 py-4">
          <div data-lines="" className="font-[450] text-[13.5px] leading-[1.7] text-zinc-300">
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
          {cut && <div className="pt-3 text-xs text-zinc-500">Showing the start of this file — the rest is in the {isR ? 'anonymized copy' : 'original'}.</div>}
        </div>
        )}
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
      {(error || (file.error && !explained) || file.warnings.length > 0) && (
        <div className="space-y-2 border-b border-zinc-800 px-4 py-2.5">
          {error && <Notice tone="red">{error}</Notice>}
          {file.error && !explained && <Notice tone={file.state === 'blocked' ? 'red' : 'amber'}>{file.error}</Notice>}
          {file.warnings.length > 0 && (
            <div className="flex items-start gap-2 text-xs text-amber-700 dark:text-amber-300">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{file.warnings.join(' · ')}</span>
            </div>
          )}
        </div>
      )}
      {text?.leftovers && (
        <LeftoversPanel roomId={roomId} leftovers={text.leftovers} byToken={byToken} onChanged={onChanged}
          onShowText={(t) => { setMode('text'); setQuery(t); setAt(0) }} onShowToken={pickToken} />
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
