// The reading room, in first person (FPV plan Phase 4): the same per-iskra
// file browser the 2D map shows when you click a building — a place whose
// work lives in a folder (Library → reading reports, Garden → garden
// pages, Workshop → skills, …). Reuses the map's exact machinery
// (folderFor / shortName / isBoilerplate + GFM rendering); the file source
// is injected so the PARENT reads authed self-files and a PUBLIC visitor
// reads the un-gated public files, with no other difference.

import { useEffect, useMemo, useState } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ChevronLeft, X } from 'lucide-react'
import type { VillageBeingPos, VillagePlace } from '../../../services/beings'
import { IskraAvatar } from '../avatars'
import { folderFor, isBoilerplate, shortName } from '../places'

interface FileRow { path: string }
export interface ReaderApi {
  files: (slug: string) => Promise<{ files: FileRow[] }>
  file: (slug: string, path: string) => Promise<{ text: string }>
}

export default function BuildingReader({ place, beings, api, onClose }: {
  place: VillagePlace
  beings: VillageBeingPos[]
  api: ReaderApi
  onClose: () => void
}) {
  const fmap = useMemo(() => folderFor(place), [place])
  const [files, setFiles] = useState<Record<string, FileRow[]>>({})
  const [loading, setLoading] = useState(true)
  const [open, setOpen] = useState<{ slug: string; name: string; path: string } | null>(null)
  const [text, setText] = useState('')

  const slugKey = beings.map((b) => b.slug).join(',')
  useEffect(() => {
    setFiles({}); setOpen(null); setLoading(true)
    if (!fmap) { setLoading(false); return }
    let dead = false
    void Promise.all(beings.map(async (b) => {
      try {
        const r = await api.files(b.slug)
        const fs = r.files.filter((f) => f.path.startsWith(fmap.folder)
          && !isBoilerplate(f.path)
          && (!fmap.excl || !f.path.startsWith(fmap.excl)))
        return [b.slug, fs] as const
      } catch { return [b.slug, [] as FileRow[]] as const }
    })).then((pairs) => {
      if (!dead) { setFiles(Object.fromEntries(pairs)); setLoading(false) }
    })
    return () => { dead = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [place.id, fmap?.folder, slugKey])

  useEffect(() => {
    if (!open) { setText(''); return }
    let dead = false
    void api.file(open.slug, open.path)
      .then((r) => { if (!dead) setText(r.text) })
      .catch(() => { if (!dead) setText('(could not read this one)') })
    return () => { dead = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open?.slug, open?.path])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { if (open) setOpen(null); else onClose() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  const anyFiles = Object.values(files).some((fs) => fs.length > 0)

  return (
    <div className="absolute inset-0 grid place-items-center bg-[#0c0f0a]/60 backdrop-blur-[2px]">
      <div className="flex max-h-[86vh] w-[min(94vw,560px)] flex-col rounded-xl border border-[#4a4436] bg-[#171410]/95 text-[#e8e2cf]">
        <div className="flex items-center gap-2 border-b border-[#4a4436] px-4 py-2.5">
          <span className="text-[14px] font-semibold">{place.name}</span>
          {fmap && <span className="text-[11px] text-[#8d8571]">· {fmap.label}</span>}
          <button onClick={onClose}
            className="ml-auto rounded-md border border-[#4a4436] p-1 text-[#b9b19a] transition-colors hover:bg-[#2a251d] hover:text-[#e8e2cf]">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          {!fmap ? (
            <p className="text-[12px] text-[#8d8571]">Nothing is kept here to read.</p>
          ) : open ? (
            <div>
              <button onClick={() => setOpen(null)}
                className="mb-2 flex items-center gap-1 text-[11px] text-violet-300 hover:text-violet-200">
                <ChevronLeft className="h-3.5 w-3.5" /> {fmap.label}
              </button>
              <div className="mb-2 text-[12px] font-medium">
                {open.name} <span className="text-[#8d8571]">· {shortName(open.path)}</span>
              </div>
              <div className="fd-fpv-markdown rounded-lg border border-[#4a4436] bg-[#0c0f0a]/60 p-3 text-[13px]">
                <Markdown remarkPlugins={[remarkGfm]}>{text || '…'}</Markdown>
              </div>
            </div>
          ) : loading ? (
            <p className="text-[12px] text-[#8d8571]">opening the reading room…</p>
          ) : !anyFiles ? (
            <p className="text-[12px] text-[#8d8571]">
              Nothing here yet — the Iskre haven't left work in {fmap.label}.
            </p>
          ) : (
            <div className="space-y-3">
              {beings.filter((b) => (files[b.slug] || []).length > 0).map((b) => (
                <div key={b.slug}>
                  <div className="mb-1 flex items-center gap-1.5 text-[12px] text-[#b9b19a]">
                    {b.avatar && <IskraAvatar c={b.avatar.c} p={b.avatar.p} size={15} />}{b.name}
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {(files[b.slug] || []).map((f) => (
                      <button key={f.path}
                        onClick={() => setOpen({ slug: b.slug, name: b.name, path: f.path })}
                        className="rounded-md border border-[#4a4436] bg-[#0c0f0a]/50 px-2 py-1 text-[11px] text-[#e8e2cf] transition-colors hover:border-violet-400/50 hover:bg-[#2a251d]">
                        {shortName(f.path)}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
