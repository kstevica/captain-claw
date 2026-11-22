import { useCallback, useEffect, useState } from 'react'
import { FolderOpen, Loader2, Check, AlertCircle } from 'lucide-react'
import { fetchJson } from '../../stores/googleAuthStore'

/** GET /fd/google/picker: what the Google Picker needs, for this user. */
interface PickerConfig {
  ready: boolean
  reason: '' | 'not_configured' | 'not_connected' | 'no_drive_scope' | 'full_drive' | 'no_api_key' | 'no_app_id'
  client_id?: string
  app_id?: string
  api_key?: string
  scope?: string
  login_hint?: string
}

interface PickedFile {
  id: string
  name: string
  url: string
}

/* eslint-disable @typescript-eslint/no-explicit-any */
declare global {
  interface Window {
    gapi?: any
    google?: any
  }
}

const _scripts = new Map<string, Promise<void>>()

function loadScript(src: string): Promise<void> {
  let loading = _scripts.get(src)
  if (!loading) {
    loading = new Promise<void>((resolve, reject) => {
      const el = document.createElement('script')
      el.src = src
      el.async = true
      el.onload = () => resolve()
      el.onerror = () => {
        _scripts.delete(src)
        reject(new Error("Couldn't load Google's file picker. Check your connection and try again."))
      }
      document.head.appendChild(el)
    })
    _scripts.set(src, loading)
  }
  return loading
}

async function loadGoogle(): Promise<void> {
  await Promise.all([
    loadScript('https://apis.google.com/js/api.js'),
    loadScript('https://accounts.google.com/gsi/client'),
  ])
  await new Promise<void>((resolve, reject) => {
    window.gapi.load('picker', {
      callback: () => resolve(),
      onerror: () => reject(new Error("Couldn't load Google's file picker.")),
    })
  })
}

/** A file-only token straight from Google for this deck's OAuth client: the
 *  stored token (which may also reach Gmail and Calendar) never comes to the
 *  page. What is picked with it is granted to the app, so the agents' own
 *  connection opens it afterwards. */
function fileOnlyToken(cfg: PickerConfig): Promise<string> {
  return new Promise((resolve, reject) => {
    const client = window.google.accounts.oauth2.initTokenClient({
      client_id: cfg.client_id,
      scope: cfg.scope,
      hint: cfg.login_hint || undefined,
      callback: (resp: any) => {
        if (resp?.access_token) resolve(resp.access_token)
        else reject(new Error(resp?.error_description || resp?.error || 'Google gave no access.'))
      },
      error_callback: (err: any) => {
        reject(new Error(err?.type === 'popup_closed' ? 'The Google window was closed.'
          : err?.type === 'popup_failed_to_open' ? 'Your browser blocked the Google window — allow pop-ups for this site.'
          : (err?.message || 'Google sign-in did not finish.')))
      },
    })
    client.requestAccessToken({ prompt: '' })
  })
}

function showPicker(cfg: PickerConfig, token: string): Promise<PickedFile[]> {
  return new Promise((resolve) => {
    const g = window.google.picker
    const myDrive = new g.DocsView(g.ViewId.DOCS).setIncludeFolders(true).setSelectFolderEnabled(false)
    const sharedDrives = new g.DocsView(g.ViewId.DOCS).setEnableDrives(true).setIncludeFolders(true)
      .setSelectFolderEnabled(false)
    const picker = new g.PickerBuilder()
      .addView(myDrive)
      .addView(sharedDrives)
      .enableFeature(g.Feature.MULTISELECT_ENABLED)
      .enableFeature(g.Feature.SUPPORT_DRIVES)
      .setOAuthToken(token)
      .setDeveloperKey(cfg.api_key)
      .setAppId(cfg.app_id)
      .setTitle('Pick files your agents may open')
      .setCallback((data: any) => {
        const action = data[g.Response.ACTION]
        if (action === g.Action.PICKED) {
          const docs: any[] = data[g.Response.DOCUMENTS] || []
          resolve(docs.map((d) => ({
            id: String(d[g.Document.ID] || ''),
            name: String(d[g.Document.NAME] || 'Untitled'),
            url: String(d[g.Document.URL] || ''),
          })))
        } else if (action === g.Action.CANCEL) {
          resolve([])
        }
      })
      .build()
    picker.setVisible(true)
  })
}
/* eslint-enable @typescript-eslint/no-explicit-any */

/** Connections → Google: give the agents single Drive files. Under the per-file
 *  Drive scope they open only files the app made or the person picked; a
 *  pasted link to the person's own Sheet otherwise can't be opened. */
export default function GoogleDrivePickerSection({ canManageClient }: { canManageClient: boolean }) {
  const [cfg, setCfg] = useState<PickerConfig | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [picked, setPicked] = useState<PickedFile[]>([])

  const load = useCallback(async () => {
    try {
      setCfg(await fetchJson('/fd/google/picker') as PickerConfig)
    } catch {
      setCfg(null)
    }
  }, [])

  useEffect(() => { void load() }, [load])

  const pick = async () => {
    if (!cfg?.ready) return
    setBusy(true)
    setError('')
    try {
      await loadGoogle()
      const token = await fileOnlyToken(cfg)
      const files = await showPicker(cfg, token)
      if (files.length) {
        setPicked((prev) => [...files, ...prev.filter((p) => !files.some((f) => f.id === p.id))])
      }
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    } finally {
      setBusy(false)
    }
  }

  // Full Drive access: the agents open any file already, nothing to pick.
  if (!cfg || cfg.reason === 'full_drive' || cfg.reason === 'not_connected' || cfg.reason === 'not_configured') {
    return null
  }

  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-950/50 px-3 py-2.5 space-y-2">
      <div className="flex items-center gap-2">
        <FolderOpen className="h-4 w-4 text-zinc-400" />
        <div className="text-sm text-zinc-200">Files from your Google Drive</div>
      </div>
      <div className="text-xs text-zinc-400 leading-relaxed">
        Your agents open only the Drive files they made or the ones you pick here — nothing else in
        your Drive. Pick a Sheet, Doc or file you want them to read or fill in.
      </div>

      {cfg.ready ? (
        <button
          onClick={pick}
          disabled={busy}
          className="inline-flex items-center gap-2 rounded-md border border-zinc-700 hover:bg-zinc-800 px-3 py-1.5 text-sm text-zinc-200 disabled:opacity-50"
        >
          {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FolderOpen className="h-3.5 w-3.5" />}
          Pick files from Google Drive
        </button>
      ) : (
        <div className="text-xs text-zinc-500">
          {cfg.reason === 'no_api_key' && (canManageClient
            ? 'To let people pick files, add the Picker API key in the credentials below.'
            : "Picking single files isn't set up on this deck yet — ask your admin.")}
          {cfg.reason === 'no_drive_scope' && 'Your Google connection has no Drive access.'}
          {cfg.reason === 'no_app_id' && "This deck's Google client ID doesn't name its Cloud project, so files can't be picked."}
        </div>
      )}

      {error && (
        <div className="flex items-start gap-1.5 text-xs text-red-400">
          <AlertCircle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {picked.length > 0 && (
        <div className="text-xs text-zinc-300 space-y-1">
          <div className="text-zinc-500">Your agents can now open:</div>
          {picked.map((f) => (
            <div key={f.id} className="flex items-center gap-1.5">
              <Check className="h-3 w-3 text-emerald-400 shrink-0" />
              {f.url ? (
                <a href={f.url} target="_blank" rel="noreferrer" className="truncate hover:underline">{f.name}</a>
              ) : <span className="truncate">{f.name}</span>}
            </div>
          ))}
          <div className="text-zinc-500">Paste a file's link in the chat if the agent hasn't got it.</div>
        </div>
      )}
    </div>
  )
}
