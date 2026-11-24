import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  AlertTriangle, ChevronDown, ChevronLeft, ChevronRight, ChevronUp, ChevronsUpDown, Download, Loader2, Plus,
  Pencil, RotateCcw, Save, Trash2, X,
} from 'lucide-react'
import { vfsApi, type FolderRef } from '../../services/vfsApi'
import type { ColumnKind } from '../../services/dataLibraryApi'
import { parseCsv, sniffSeparator, toCsv } from '../../utils/csv'
import { isSheetExt, readXlsx } from '../../utils/xlsx'
import { Cell, CellEditor, KindIcon, SearchBox, type Col } from './DataLibrary'
import { filterKindOf, localValueStats, makeMatcher, useColumnFilters, type ColumnFilter } from './columnFilters'
import { FilterButton, FilterChips, FilterPopover } from './ColumnFilterUI'

// A CSV, TSV or Excel file as a table, the way the Data library shows a table:
// typed columns, search, sort, a filter on any column (as many as you like), a
// page at a time. A CSV or TSV in a VFS folder can be changed in place
// (double-click a cell), rows added and deleted; nothing is written until
// Save, which writes the file back with the separator it came with. An Excel
// workbook (.xlsx) is read-only here, a tab for each of its sheets — and so
// is any file given only by its addresses (`urls`), from whatever viewer.

const PAGE = 100
const MAX_ROWS = 50_000
/** A workbook bigger than this is read in Excel, not here. */
const MAX_SHEET_BYTES = 40 * 1024 * 1024
const URL_RE = /^https?:\/\//i
const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/
const NUM_RE = /^-?\d+(?:[.,]\d+)?%?$/
const DATE_RE = /^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?$/
const BOOL_RE = /^(true|false)$/i

/** What a column holds, from its values: number, date, yes/no, email, url or text. */
function kindOf(values: string[]): ColumnKind {
  const filled = values.filter((v) => v.trim() !== '').slice(0, 200)
  if (!filled.length) return 'text'
  if (filled.every((v) => NUM_RE.test(v.trim()))) return 'number'
  if (filled.every((v) => DATE_RE.test(v.trim()))) return 'date'
  if (filled.every((v) => BOOL_RE.test(v.trim()))) return 'boolean'
  if (filled.every((v) => EMAIL_RE.test(v.trim()))) return 'email'
  if (filled.every((v) => URL_RE.test(v.trim()))) return 'url'
  return 'text'
}

interface Row { id: number; cells: string[] }
/** One sheet (a CSV is one): its header, its rows, and whether rows were left out. */
interface Tab { name: string; header: string[]; rows: Row[]; cut: boolean }

/** Rows as read (the first the header) → a sheet, every row as wide as the widest. */
function toTab(name: string, all: string[][], cut: boolean): Tab {
  const head = all[0] ?? []
  const width = Math.max(head.length, ...all.slice(1).map((r) => r.length), 1)
  return {
    name,
    header: Array.from({ length: width }, (_, i) => head[i]?.trim() || `column ${i + 1}`),
    rows: all.slice(1).map((cells, id) => ({ id, cells: Array.from({ length: width }, (_, i) => cells[i] ?? '') })),
    cut,
  }
}

export function CsvTable({ fileRef: given, path = '', urls, name, extension, onClose, onSaved, onEditText }: {
  /** A file in a VFS folder: read from there, and (a CSV or TSV) saved back. */
  fileRef?: FolderRef
  path?: string
  /** Or the file at these addresses, read-only (any viewer's routes). */
  urls?: { view: string; download: string }
  name: string
  extension: string
  onClose: () => void
  /** The file was written back. */
  onSaved?: () => void
  /** Change the file as text instead (a viewer that can edit it). */
  onEditText?: () => void
}) {
  const workbook = isSheetExt(extension)
  const [book, setBook] = useState<Tab[]>([])
  const [sheet, setSheet] = useState(0)
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
  const filt = useColumnFilters()
  // The file it shows, by what it names: a caller that builds the object anew on every
  // render (a drawer that reloads every few seconds) mustn't read the file again, and
  // throw away unsaved edits, each time.
  const project = given?.project
  const owner = given?.owner
  const fileRef = useMemo<FolderRef | null>(
    () => (project !== undefined && owner !== undefined ? { project, owner } : null), [project, owner],
  )
  // Given addresses may change with every sign-in token refresh: the file stays the same.
  const urlsRef = useRef(urls)
  urlsRef.current = urls
  const fromUrls = !!urls
  const editable = !workbook && !fromUrls && !!fileRef
  const downloadUrl = urls ? urls.download : fileRef ? vfsApi.downloadUrl(fileRef, path) : ''

  const show = (tab: Tab | undefined) => {
    setHeader(tab?.header ?? [])
    setRows(tab?.rows ?? [])
    setQ(''); setSort(null); setPage(0); filt.clear()
  }

  const load = useCallback(async () => {
    setLoading(true); setLoadError('')
    try {
      const url = urlsRef.current?.view ?? (fileRef ? vfsApi.viewUrl(fileRef, path, Date.now() / 1000) : '')
      const r = await fetch(url, { credentials: 'include' })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      let tabs: Tab[]
      if (workbook) {
        if (Number(r.headers.get('content-length') ?? 0) > MAX_SHEET_BYTES) throw new Error('This workbook is too big to show here: download it to open it in Excel.')
        const buf = await r.arrayBuffer()
        if (buf.byteLength > MAX_SHEET_BYTES) throw new Error('This workbook is too big to show here: download it to open it in Excel.')
        const wb = await readXlsx(buf, { maxRows: MAX_ROWS + 1 })
        tabs = wb.sheets.map((s) => toTab(s.name, s.rows, s.cut))
      } else {
        const text = await r.text()
        const s = sniffSeparator(text, extension)
        const all = parseCsv(text, s)
        if (all.length - 1 > MAX_ROWS) throw new Error(`It has over ${MAX_ROWS.toLocaleString()} rows: download it to work on it.`)
        setSep(s)
        tabs = [toTab(name, all, false)]
      }
      setBook(tabs)
      setSheet(0)
      show(tabs[0])
      setNextId(tabs[0]?.rows.length ?? 0)
      setDirty(false); setSaved(false); setEditing(null)
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "Couldn't read the file")
    } finally {
      setLoading(false)
    }
    // `show` and `filt` only set state; `name` is just a sheet label.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileRef, path, extension, workbook, fromUrls])

  useEffect(() => { void load() }, [load])

  const pickSheet = (i: number) => {
    if (i === sheet) return
    setSheet(i)
    show(book[i])
    setEditing(null)
  }

  const kinds = useMemo(() => header.map((_, i) => kindOf(rows.map((r) => r.cells[i] ?? ''))), [header, rows])
  const fkinds = useMemo(() => kinds.map((k) => filterKindOf(k)), [kinds])
  const terms = useMemo(() => q.toLowerCase().split(/\s+/).filter(Boolean), [q])
  const narrowed = terms.length > 0 || filt.filters.length > 0
  /** The rows the search and these filters leave (top / bottom, the average
   *  and repeats measured over the whole column). */
  const narrow = useCallback((filters: ColumnFilter[]) => {
    let list = terms.length
      ? rows.filter((r) => { const hay = r.cells.join(' ').toLowerCase(); return terms.every((t) => hay.includes(t)) })
      : rows
    if (filters.length) {
      const tests = filters.map((f) => {
        const i = Number(f.col)
        const test = makeMatcher(f, fkinds[i] ?? 'text', rows.map((r) => r.cells[i]))
        return (r: Row) => test(r.cells[i])
      })
      list = list.filter((r) => tests.every((t) => t(r)))
    }
    return list
  }, [rows, terms, fkinds])
  /** A column's values for its filter, among the rows the other filters leave. */
  const statsFor = (i: number) => (search: string) => Promise.resolve(
    localValueStats(narrow(filt.filters.filter((f) => f.col !== String(i))).map((r) => r.cells[i]), fkinds[i] ?? 'text', search))
  const shown = useMemo(() => {
    let list = narrow(filt.filters)
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
  }, [narrow, sort, kinds, filt.filters])
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
    // The new row shows where it lands: nothing hides or moves it.
    setQ(''); setSort(null); filt.clear()
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
    if (!editable || !fileRef) return
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
      if (e.key === 'Escape') {
        // A filter or a cell being changed closes itself first.
        if ((e.target as Element | null)?.closest?.('[data-cell-editor]')) return
        if (filt.menu) { e.stopPropagation(); filt.setMenu(null); return }
        if (!editing) { e.stopPropagation(); close() }
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's' && dirty && !saving) { e.preventDefault(); void save() }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  })

  const toggleSort = (col: number) => setSort((s) => (s?.col !== col ? { col, dir: 'asc' } : s.dir === 'asc' ? { col, dir: 'desc' } : null))
  const btn = 'flex shrink-0 items-center gap-1.5 rounded-lg border border-zinc-800 px-2.5 py-1.5 text-xs text-zinc-300 transition-colors hover:bg-zinc-800/60 disabled:opacity-40'
  const pager = 'flex h-7 w-7 items-center justify-center rounded-lg text-zinc-400 hover:bg-zinc-800 disabled:opacity-30'
  const tab = book[sheet]
  const what = workbook
    ? [book.length > 1 ? `Sheet “${tab?.name ?? ''}” of ${book.length}` : 'Excel workbook', 'read-only here'].join(' · ')
    : `${sep === '\t' ? 'tab' : `"${sep}"`}-separated${editable ? '' : ' · read-only here'}`
  const openMenu = filt.menu
  const menuCol = openMenu ? Number(openMenu.col) : -1

  return createPortal(
    <div data-above-drawer className="fixed inset-0 z-[80] flex items-stretch justify-center bg-black/60 p-0 wide:p-6" onMouseDown={(e) => { if (e.target === e.currentTarget) close() }}>
      <div role="dialog" aria-label={`${name} as a table`} className="flex min-h-0 w-full max-w-7xl flex-col overflow-hidden bg-zinc-950 shadow-2xl wide:rounded-2xl wide:border wide:border-zinc-800">
        <div className="flex shrink-0 items-center gap-3 border-b border-zinc-800 px-4 py-3">
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm font-semibold text-zinc-100" title={path || name}>{name}</div>
            <div className="text-xs text-zinc-500">
              {loading ? 'Reading…' : `${rows.length.toLocaleString()} rows · ${header.length} columns · ${what}`}
              {tab?.cut && <span className="ml-2 text-amber-700 dark:text-amber-300">Only its first {MAX_ROWS.toLocaleString()} rows</span>}
              {dirty && <span className="ml-2 text-amber-700 dark:text-amber-300">Changed, not saved</span>}
              {saved && !dirty && <span className="ml-2 text-emerald-600 dark:text-emerald-400">Saved</span>}
            </div>
          </div>
          {editable && dirty && (
            <button onClick={() => { if (window.confirm('Drop your changes and read the file again?')) void load() }} className={btn} title="Drop your changes">
              <RotateCcw className="h-3.5 w-3.5" />Revert
            </button>
          )}
          {editable && (
            <button onClick={() => void save()} disabled={!dirty || saving}
              className="flex shrink-0 items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-emerald-500 disabled:opacity-40"
              title="Write the file back (Ctrl/⌘ S)">
              {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}Save
            </button>
          )}
          {onEditText && (
            <button onClick={onEditText} className={btn} title="Change the file as text">
              <Pencil className="h-3.5 w-3.5" />Edit
            </button>
          )}
          {downloadUrl && (
            <a href={downloadUrl} className={btn} title={editable ? 'Download the file as it is saved' : 'Download the file'} aria-label="Download">
              <Download className="h-3.5 w-3.5" />
            </a>
          )}
          <button onClick={close} aria-label="Close" className="flex h-8 w-8 items-center justify-center rounded-full text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100">
            <X className="h-4 w-4" />
          </button>
        </div>
        {saveError && <p className="shrink-0 border-b border-zinc-800 px-4 py-2 text-xs text-red-600 dark:text-red-400">{saveError}</p>}
        {book.length > 1 && (
          <div role="tablist" aria-label="Sheets" className="flex shrink-0 gap-1 overflow-x-auto border-b border-zinc-800 px-3 pt-2">
            {book.map((b, i) => (
              <button key={`${i}:${b.name}`} role="tab" aria-selected={i === sheet} onClick={() => pickSheet(i)}
                className={`shrink-0 rounded-t-lg border border-b-0 px-3 py-1.5 text-xs transition-colors ${i === sheet
                  ? 'border-zinc-700 bg-zinc-900 font-medium text-zinc-100'
                  : 'border-transparent text-zinc-500 hover:bg-zinc-900/60 hover:text-zinc-200'}`}>
                {b.name}<span className="ml-1.5 tabular-nums text-zinc-500">{b.rows.length.toLocaleString()}</span>
              </button>
            ))}
          </div>
        )}
        <div className="flex shrink-0 items-center gap-2 border-b border-zinc-800 px-3 py-2">
          <div className="max-w-sm flex-1"><SearchBox value={q} onChange={(v) => { setQ(v); setPage(0) }} placeholder="Search the rows…" small /></div>
          {editable && <button onClick={addRow} disabled={loading || !!loadError} className={btn} title="Add a row by hand"><Plus className="h-3.5 w-3.5" />Add row</button>}
          <span className="ml-auto shrink-0 text-xs tabular-nums text-zinc-500">
            {shown.length ? `${(at * PAGE + 1).toLocaleString()}–${Math.min(shown.length, at * PAGE + PAGE).toLocaleString()} of ${shown.length.toLocaleString()}${narrowed ? ' matching' : ''}` : narrowed ? 'No matching rows' : ''}
          </span>
          <button onClick={() => setPage(Math.max(0, at - 1))} disabled={at === 0} className={pager} aria-label="Previous page"><ChevronLeft className="h-4 w-4" /></button>
          <button onClick={() => setPage(Math.min(pages - 1, at + 1))} disabled={at >= pages - 1} className={pager} aria-label="Next page"><ChevronRight className="h-4 w-4" /></button>
        </div>
        <FilterChips filters={filt.filters} labelOf={(c) => header[Number(c)] ?? c} kindOf={(c) => fkinds[Number(c)] ?? 'text'}
          onEdit={(col, anchor) => filt.setMenu({ col, anchor })}
          onRemove={(c) => { filt.remove(c); setPage(0) }} onClear={() => { filt.clear(); setPage(0) }} />
        <div className="min-h-0 flex-1 overflow-auto">
          {loadError ? (
            <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center text-sm text-zinc-500">
              <AlertTriangle className="h-5 w-5 text-red-500" />{loadError}
            </div>
          ) : loading ? (
            <div className="flex h-full justify-center pt-12 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
          ) : !rows.length ? (
            <div className="flex h-full items-center justify-center px-6 text-sm text-zinc-500">{workbook ? 'This sheet has no rows under its header.' : 'This file has no rows yet.'}</div>
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
                          <FilterButton label={h} active={!!filt.of(String(i))} onOpen={(anchor) => filt.setMenu({ col: String(i), anchor })} />
                        </span>
                      </th>
                    )
                  })}
                </tr>
              </thead>
              <tbody>
                {!pageRows.length && (
                  <tr>
                    <td colSpan={header.length + 1} className="px-6 py-12 text-center text-sm text-zinc-500">
                      No row matches {filt.filters.length && terms.length ? 'the search and the filters' : filt.filters.length ? (filt.filters.length > 1 ? 'these filters' : 'that filter') : 'that search'}.
                    </td>
                  </tr>
                )}
                {pageRows.map((r, n) => (
                  <tr key={r.id} className="group hover:bg-zinc-900/70">
                    <td className="sticky left-0 border-b border-zinc-800/60 bg-zinc-950 px-2 py-1.5 text-right text-[11px] tabular-nums text-zinc-600">
                      <span className="flex items-center justify-end gap-1">
                        {editable && (
                          <button onClick={() => deleteRow(r.id)} title="Delete this row" aria-label={`Delete row ${at * PAGE + n + 1}`}
                            className="rounded p-0.5 text-zinc-500 opacity-30 transition-opacity hover:bg-red-500/10 hover:text-red-500 focus:opacity-100 group-hover:opacity-100">
                            <Trash2 className="h-3 w-3" />
                          </button>
                        )}
                        {at * PAGE + n + 1}
                      </span>
                    </td>
                    {header.map((h, i) => {
                      const here = editable && editing?.id === r.id && editing.col === i
                      const col: Col = { name: h, type: 'text' }
                      const cell = <Cell value={r.cells[i] === '' ? null : r.cells[i]} kind={kinds[i]} type="text" terms={terms} />
                      return (
                        <td key={i} onDoubleClick={editable ? () => setEditing({ id: r.id, col: i }) : undefined}
                          title={here ? undefined : editable ? `${r.cells[i] ? `${r.cells[i]}\n` : ''}Double-click to change` : r.cells[i] || undefined}
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
      {openMenu && menuCol >= 0 && menuCol < header.length && (
        <FilterPopover key={openMenu.col} anchor={openMenu.anchor} label={header[menuCol]} fk={fkinds[menuCol] ?? 'text'}
          current={filt.of(openMenu.col)} loadStats={statsFor(menuCol)} onApply={(f) => { filt.apply(openMenu.col, f); setPage(0) }} onClose={() => filt.setMenu(null)} />
      )}
    </div>,
    document.body,
  )
}
