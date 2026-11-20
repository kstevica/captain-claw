import { useCallback, useEffect, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  AlertTriangle, ChevronDown, ChevronLeft, ChevronRight, ChevronUp, ChevronsUpDown, Download, Loader2, Plus,
  RotateCcw, Save, Trash2, X,
} from 'lucide-react'
import { vfsApi, type FolderRef } from '../../services/vfsApi'
import type { ColumnKind } from '../../services/dataLibraryApi'
import { parseCsv, sniffSeparator, toCsv } from '../../utils/csv'
import { Cell, CellEditor, KindIcon, SearchBox, type Col } from './DataLibrary'

// A CSV or TSV file in a VFS folder as a table, the way the Data library shows a
// table: typed columns, search, sort, a page at a time, and cells changed in
// place (double-click). Rows can be added and deleted. Nothing is written until
// Save, which writes the file back with the separator it came with.

const PAGE = 100
const MAX_ROWS = 50_000
const URL_RE = /^https?:\/\//i
const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/
const NUM_RE = /^-?\d+(?:[.,]\d+)?$/

/** What a column holds, from its values: number, email, url or text. */
function kindOf(values: string[]): ColumnKind {
  const filled = values.filter((v) => v.trim() !== '').slice(0, 200)
  if (!filled.length) return 'text'
  if (filled.every((v) => NUM_RE.test(v.trim()))) return 'number'
  if (filled.every((v) => EMAIL_RE.test(v.trim()))) return 'email'
  if (filled.every((v) => URL_RE.test(v.trim()))) return 'url'
  return 'text'
}

interface Row { id: number; cells: string[] }

export function CsvTable({ fileRef: given, path, name, extension, onClose, onSaved }: {
  fileRef: FolderRef
  path: string
  name: string
  extension: string
  onClose: () => void
  /** The file was written back. */
  onSaved?: () => void
}) {
  const [header, setHeader] = useState<string[]>([])
  const [rows, setRows] = useState<Row[]>([])
  const [sep, setSep] = useState(',')
  const [loadError, setLoadError] = useState('')
  const [loading, setLoading] = useState(true)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState('')
  const [saved, setSaved] = useState(false)
  const [q, setQ] = useState('')
  const [sort, setSort] = useState<{ col: number; dir: 'asc' | 'desc' } | null>(null)
  const [page, setPage] = useState(0)
  const [editing, setEditing] = useState<{ id: number; col: number } | null>(null)
  const [nextId, setNextId] = useState(0)
  // The file it shows, by what it names: a caller that builds the object anew on every
  // render (a drawer that reloads every few seconds) mustn't read the file again, and
  // throw away unsaved edits, each time.
  const fileRef = useMemo<FolderRef>(() => ({ project: given.project, owner: given.owner }), [given.project, given.owner])

  const load = useCallback(async () => {
    setLoading(true); setLoadError('')
    try {
      const r = await fetch(vfsApi.viewUrl(fileRef, path, Date.now() / 1000), { credentials: 'include' })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const text = await r.text()
      const s = sniffSeparator(text, extension)
      const all = parseCsv(text, s)
      if (all.length - 1 > MAX_ROWS) throw new Error(`It has over ${MAX_ROWS.toLocaleString()} rows: download it to work on it.`)
      const head = all[0] ?? []
      const width = Math.max(head.length, ...all.slice(1).map((r) => r.length), 1)
      const cols = Array.from({ length: width }, (_, i) => head[i] ?? `column ${i + 1}`)
      setSep(s)
      setHeader(cols)
      setRows(all.slice(1).map((cells, id) => ({ id, cells: Array.from({ length: width }, (_, i) => cells[i] ?? '') })))
      setNextId(all.length)
      setDirty(false); setSaved(false); setEditing(null)
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "Couldn't read the file")
    } finally {
      setLoading(false)
    }
  }, [fileRef, path, extension])

  useEffect(() => { void load() }, [load])

  const kinds = useMemo(() => header.map((_, i) => kindOf(rows.map((r) => r.cells[i] ?? ''))), [header, rows])
  const terms = useMemo(() => q.toLowerCase().split(/\s+/).filter(Boolean), [q])
  const shown = useMemo(() => {
    let list = terms.length
      ? rows.filter((r) => { const hay = r.cells.join(' ').toLowerCase(); return terms.every((t) => hay.includes(t)) })
      : rows
    if (sort) {
      const numeric = kinds[sort.col] === 'number'
      list = [...list].sort((a, b) => {
        const x = a.cells[sort.col] ?? ''
        const y = b.cells[sort.col] ?? ''
        const c = numeric ? (parseFloat(x.replace(',', '.')) || 0) - (parseFloat(y.replace(',', '.')) || 0)
          : x.localeCompare(y, undefined, { numeric: true, sensitivity: 'base' })
        return sort.dir === 'asc' ? c : -c
      })
    }
    return list
  }, [rows, terms, sort, kinds])
  const pages = Math.max(1, Math.ceil(shown.length / PAGE))
  const at = Math.min(page, pages - 1)
  const pageRows = shown.slice(at * PAGE, at * PAGE + PAGE)

  const change = (id: number, col: number, value: string) => {
    setRows((rs) => rs.map((r) => (r.id === id ? { ...r, cells: r.cells.map((c, i) => (i === col ? value : c)) } : r)))
    setDirty(true); setSaved(false)
  }
  const addRow = () => {
    const id = nextId
    setNextId(id + 1)
    setRows((rs) => [...rs, { id, cells: header.map(() => '') }])
    setQ(''); setSort(null)
    setPage(Math.floor(rows.length / PAGE))
    setEditing({ id, col: 0 })
    setDirty(true); setSaved(false)
  }
  const deleteRow = (id: number) => {
    setRows((rs) => rs.filter((r) => r.id !== id))
    if (editing?.id === id) setEditing(null)
    setDirty(true); setSaved(false)
  }
  const save = async () => {
    setSaving(true); setSaveError('')
    try {
      await vfsApi.write(fileRef, path, toCsv([header, ...rows.map((r) => r.cells)], sep))
      setDirty(false); setSaved(true)
      onSaved?.()
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Couldn't save the file")
    } finally {
      setSaving(false)
    }
  }
  const close = useCallback(() => {
    if (dirty && !window.confirm('Close without saving your changes?')) return
    onClose()
  }, [dirty, onClose])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !editing) { e.stopPropagation(); close() }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's' && dirty && !saving) { e.preventDefault(); void save() }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  })

  const toggleSort = (col: number) => setSort((s) => (s?.col !== col ? { col, dir: 'asc' } : s.dir === 'asc' ? { col, dir: 'desc' } : null))
  const btn = 'flex shrink-0 items-center gap-1.5 rounded-lg border border-zinc-800 px-2.5 py-1.5 text-xs text-zinc-300 transition-colors hover:bg-zinc-800/60 disabled:opacity-40'
  const pager = 'flex h-7 w-7 items-center justify-center rounded-lg text-zinc-400 hover:bg-zinc-800 disabled:opacity-30'

  return createPortal(
    <div data-above-drawer className="fixed inset-0 z-[80] flex items-stretch justify-center bg-black/60 p-0 wide:p-6" onMouseDown={(e) => { if (e.target === e.currentTarget) close() }}>
      <div role="dialog" aria-label={`${name} as a table`} className="flex min-h-0 w-full max-w-7xl flex-col overflow-hidden bg-zinc-950 shadow-2xl wide:rounded-2xl wide:border wide:border-zinc-800">
        <div className="flex shrink-0 items-center gap-3 border-b border-zinc-800 px-4 py-3">
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm font-semibold text-zinc-100" title={path}>{name}</div>
            <div className="text-xs text-zinc-500">
              {loading ? 'Reading…' : `${rows.length.toLocaleString()} rows · ${header.length} columns · ${sep === '\t' ? 'tab' : `"${sep}"`}-separated`}
              {dirty && <span className="ml-2 text-amber-700 dark:text-amber-300">Changed, not saved</span>}
              {saved && !dirty && <span className="ml-2 text-emerald-600 dark:text-emerald-400">Saved</span>}
            </div>
          </div>
          {dirty && (
            <button onClick={() => { if (window.confirm('Drop your changes and read the file again?')) void load() }} className={btn} title="Drop your changes">
              <RotateCcw className="h-3.5 w-3.5" />Revert
            </button>
          )}
          <button onClick={() => void save()} disabled={!dirty || saving}
            className="flex shrink-0 items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-emerald-500 disabled:opacity-40"
            title="Write the file back (Ctrl/⌘ S)">
            {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}Save
          </button>
          <a href={vfsApi.downloadUrl(fileRef, path)} className={btn} title="Download the file as it is saved"><Download className="h-3.5 w-3.5" /></a>
          <button onClick={close} aria-label="Close" className="flex h-8 w-8 items-center justify-center rounded-full text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100">
            <X className="h-4 w-4" />
          </button>
        </div>
        {saveError && <p className="shrink-0 border-b border-zinc-800 px-4 py-2 text-xs text-red-600 dark:text-red-400">{saveError}</p>}
        <div className="flex shrink-0 items-center gap-2 border-b border-zinc-800 px-3 py-2">
          <div className="max-w-sm flex-1"><SearchBox value={q} onChange={(v) => { setQ(v); setPage(0) }} placeholder="Search the rows…" small /></div>
          <button onClick={addRow} disabled={loading || !!loadError} className={btn} title="Add a row by hand"><Plus className="h-3.5 w-3.5" />Add row</button>
          <span className="ml-auto shrink-0 text-xs tabular-nums text-zinc-500">
            {shown.length ? `${(at * PAGE + 1).toLocaleString()}–${Math.min(shown.length, at * PAGE + PAGE).toLocaleString()} of ${shown.length.toLocaleString()}${terms.length ? ' matching' : ''}` : terms.length ? 'No matching rows' : ''}
          </span>
          <button onClick={() => setPage(Math.max(0, at - 1))} disabled={at === 0} className={pager} aria-label="Previous page"><ChevronLeft className="h-4 w-4" /></button>
          <button onClick={() => setPage(Math.min(pages - 1, at + 1))} disabled={at >= pages - 1} className={pager} aria-label="Next page"><ChevronRight className="h-4 w-4" /></button>
        </div>
        <div className="min-h-0 flex-1 overflow-auto">
          {loadError ? (
            <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center text-sm text-zinc-500">
              <AlertTriangle className="h-5 w-5 text-red-500" />{loadError}
            </div>
          ) : loading ? (
            <div className="flex h-full justify-center pt-12 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
          ) : !pageRows.length ? (
            <div className="flex h-full items-center justify-center px-6 text-sm text-zinc-500">{terms.length ? 'No row matches that search.' : 'This file has no rows yet.'}</div>
          ) : (
            <table className="w-max min-w-full border-separate border-spacing-0 text-[13px]">
              <thead className="sticky top-0 z-10">
                <tr>
                  <th className="sticky left-0 z-20 border-b border-zinc-800 bg-zinc-900 px-3 py-2 text-right text-[11px] font-medium text-zinc-500">#</th>
                  {header.map((h, i) => {
                    const sorted = sort?.col === i ? sort.dir : null
                    return (
                      <th key={i} onClick={() => toggleSort(i)} title={`${h} · click to sort`}
                        className="cursor-pointer select-none border-b border-zinc-800 bg-zinc-900 px-3 py-2 font-medium text-zinc-300 hover:text-zinc-100">
                        <span className={`flex items-center gap-1.5 whitespace-nowrap ${kinds[i] === 'number' ? 'justify-end' : ''}`}>
                          <KindIcon kind={kinds[i]} />{h}
                          {sorted === 'asc' ? <ChevronUp className="h-3.5 w-3.5 text-emerald-500" />
                            : sorted === 'desc' ? <ChevronDown className="h-3.5 w-3.5 text-emerald-500" />
                              : <ChevronsUpDown className="h-3 w-3 text-zinc-600" />}
                        </span>
                      </th>
                    )
                  })}
                </tr>
              </thead>
              <tbody>
                {pageRows.map((r, n) => (
                  <tr key={r.id} className="group hover:bg-zinc-900/70">
                    <td className="sticky left-0 border-b border-zinc-800/60 bg-zinc-950 px-2 py-1.5 text-right text-[11px] tabular-nums text-zinc-600">
                      <span className="flex items-center justify-end gap-1">
                        <button onClick={() => deleteRow(r.id)} title="Delete this row" aria-label={`Delete row ${at * PAGE + n + 1}`}
                          className="rounded p-0.5 text-zinc-500 opacity-30 transition-opacity hover:bg-red-500/10 hover:text-red-500 focus:opacity-100 group-hover:opacity-100">
                          <Trash2 className="h-3 w-3" />
                        </button>
                        {at * PAGE + n + 1}
                      </span>
                    </td>
                    {header.map((h, i) => {
                      const here = editing?.id === r.id && editing.col === i
                      const col: Col = { name: h, type: 'text' }
                      const cell = <Cell value={r.cells[i] === '' ? null : r.cells[i]} kind={kinds[i]} type="text" terms={terms} />
                      return (
                        <td key={i} onDoubleClick={() => setEditing({ id: r.id, col: i })}
                          title={here ? undefined : `${r.cells[i] ? `${r.cells[i]}\n` : ''}Double-click to change`}
                          className={`relative max-w-[320px] border-b border-zinc-800/60 px-3 py-1.5 text-zinc-300 ${here ? 'whitespace-nowrap' : 'truncate'} ${kinds[i] === 'number' ? 'text-right tabular-nums' : ''}`}>
                          {here ? (
                            <>
                              <span className="invisible block truncate">{cell}</span>
                              <div onClick={(e) => e.stopPropagation()}
                                className="absolute left-0 top-0 z-30 w-[max(100%,300px)] whitespace-normal rounded-lg border border-zinc-700 bg-zinc-900 p-2 text-left shadow-xl">
                                <CellEditor col={col} value={r.cells[i]} compact
                                  onSave={async (draft) => { change(r.id, i, draft); return '' }}
                                  onDone={() => setEditing(null)} />
                              </div>
                            </>
                          ) : cell}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>,
    document.body,
  )
}
