import { useEffect, useRef, useState } from 'react'
import { ImagePlus, Loader2, Palette, Radio, Save, Trash2 } from 'lucide-react'
import { DEFAULT_INSTANCE_NAME, useAuthStore } from '../../stores/authStore'
import { deleteBrandingLogo, uploadBrandingLogo } from '../../services/sharedIntegrations'
import { Switch } from './Switch'

// Admin → Settings → Branding: the deck's name (top left of the panels and on
// the sign-in page), a short description for the sign-in page, and a logo —
// small in the panels, large on the sign-in page — and whether the sign-in
// page's footer says "Powered by Captain" with its links. Name, description
// and that switch are system config (PUT /fd/admin/config); the logo is
// uploaded on its own (captain_claw/flight_deck/branding_routes.py).

const NAME_MAX = 60
const DESCRIPTION_MAX = 300
const LOGO_MAX_BYTES = 512 * 1024
const LOGO_TYPES = 'image/png,image/jpeg,image/webp,image/gif,image/svg+xml'

interface Props {
  name: string
  description: string
  poweredBy: boolean
  onSave: (patch: { instance_name?: string; instance_description?: string; show_powered_by?: boolean }) => Promise<{
    instance_name: string
    instance_description: string
    show_powered_by: boolean
  }>
}

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result || ''))
    reader.onerror = () => reject(new Error("That file couldn't be read"))
    reader.readAsDataURL(file)
  })
}

export function BrandingAdmin({ name, description, poweredBy, onSave }: Props) {
  const logo = useAuthStore((s) => s.branding.logo)
  const setBranding = useAuthStore((s) => s.setBranding)
  const [nameDraft, setNameDraft] = useState(name)
  const [descDraft, setDescDraft] = useState(description)
  const [saving, setSaving] = useState(false)
  const [poweredByBusy, setPoweredByBusy] = useState(false)
  const [logoBusy, setLogoBusy] = useState(false)
  const [error, setError] = useState('')
  const [flash, setFlash] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)

  useEffect(() => { setNameDraft(name) }, [name])
  useEffect(() => { setDescDraft(description) }, [description])

  const dirty = nameDraft.trim() !== name || descDraft.trim() !== description

  const done = (msg: string) => {
    setFlash(msg)
    window.setTimeout(() => setFlash(''), 2000)
  }

  const save = async () => {
    setSaving(true)
    setError('')
    try {
      const next = await onSave({ instance_name: nameDraft, instance_description: descDraft })
      setBranding({ name: next.instance_name || DEFAULT_INSTANCE_NAME, description: next.instance_description })
      setNameDraft(next.instance_name)
      setDescDraft(next.instance_description)
      done('Saved')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save')
    } finally {
      setSaving(false)
    }
  }

  const togglePoweredBy = async (on: boolean) => {
    setPoweredByBusy(true)
    setError('')
    try {
      const next = await onSave({ show_powered_by: on })
      setBranding({ poweredBy: next.show_powered_by })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save')
    } finally {
      setPoweredByBusy(false)
    }
  }

  const upload = async (file: File | undefined) => {
    if (!file) return
    setError('')
    if (file.size > LOGO_MAX_BYTES) {
      setError(`The logo is at most ${LOGO_MAX_BYTES / 1024} KB — this one is ${Math.ceil(file.size / 1024)} KB.`)
      return
    }
    setLogoBusy(true)
    try {
      const { logo: url } = await uploadBrandingLogo(await readAsDataUrl(file))
      setBranding({ logo: url })
      done('Logo saved')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to upload the logo')
    } finally {
      setLogoBusy(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  const removeLogo = async () => {
    if (!confirm('Remove the logo? The panels and the sign-in page go back to the Captain icon.')) return
    setLogoBusy(true)
    setError('')
    try {
      await deleteBrandingLogo()
      setBranding({ logo: '' })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to remove the logo')
    } finally {
      setLogoBusy(false)
    }
  }

  const input = 'w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none'

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 space-y-4">
      <div>
        <h3 className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
          <Palette className="h-4 w-4 text-violet-400" />
          Branding
        </h3>
        <p className="text-xs text-zinc-500 mt-0.5">
          What people see on the sign-in page and at the top left of the panels. Signed-in users see a change
          on their next reload.
        </p>
      </div>

      <div className="flex flex-col gap-4 sm:flex-row">
        {/* Logo */}
        <div className="shrink-0">
          <div className="mb-1 text-sm text-zinc-300">Logo</div>
          <div className="flex h-24 w-24 items-center justify-center overflow-hidden rounded-xl border border-zinc-700 bg-zinc-950 p-2">
            {logoBusy ? (
              <Loader2 className="h-5 w-5 animate-spin text-zinc-500" />
            ) : logo ? (
              <img src={logo} alt="Logo" className="h-full w-full object-contain" />
            ) : (
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600">
                <Radio className="h-6 w-6 text-white" />
              </div>
            )}
          </div>
          <input
            ref={fileInput}
            type="file"
            accept={LOGO_TYPES}
            className="hidden"
            onChange={(e) => upload(e.target.files?.[0])}
          />
          <div className="mt-2 flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => fileInput.current?.click()}
              disabled={logoBusy}
              className="flex items-center gap-1 rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-50"
            >
              <ImagePlus className="h-3 w-3" />
              {logo ? 'Replace' : 'Upload'}
            </button>
            {logo && (
              <button
                type="button"
                onClick={removeLogo}
                disabled={logoBusy}
                title="Remove the logo"
                className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-red-400 disabled:opacity-50"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
          <p className="mt-1 w-32 text-[11px] leading-snug text-zinc-500">PNG, JPEG, WebP, GIF or SVG, up to 512 KB. Square looks best.</p>
        </div>

        {/* Name + description */}
        <div className="min-w-0 flex-1 space-y-3">
          <div>
            <label className="block text-sm text-zinc-300 mb-1">Instance name</label>
            <input
              type="text"
              value={nameDraft}
              onChange={(e) => setNameDraft(e.target.value)}
              maxLength={NAME_MAX}
              placeholder={DEFAULT_INSTANCE_NAME}
              className={input}
            />
            <p className="mt-1 text-xs text-zinc-500">Instead of “{DEFAULT_INSTANCE_NAME}”. Leave it blank for the default.</p>
          </div>
          <div>
            <label className="block text-sm text-zinc-300 mb-1">Short description</label>
            <textarea
              value={descDraft}
              onChange={(e) => setDescDraft(e.target.value)}
              maxLength={DESCRIPTION_MAX}
              rows={3}
              placeholder="One or two sentences for the sign-in page"
              className={`${input} resize-none`}
            />
            <p className="mt-1 text-xs text-zinc-500">
              Shown on the sign-in and sign-up page, under the name. {descDraft.length}/{DESCRIPTION_MAX}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={save}
              disabled={saving || !dirty}
              className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
            >
              {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
              Save
            </button>
            {flash && <span className="text-xs text-emerald-600 dark:text-emerald-400">{flash}</span>}
          </div>
        </div>
      </div>

      <label className="flex items-center justify-between gap-4 border-t border-zinc-800 pt-4">
        <div>
          <div className="text-sm text-zinc-300">“Powered by Captain” on the sign-in page</div>
          <div className="text-xs text-zinc-500 mt-0.5">
            Links in the footer to captain-claw.com and Captain's repository on GitHub. Off: only the
            version shows there.
          </div>
        </div>
        <Switch
          on={poweredBy}
          disabled={poweredByBusy}
          label="“Powered by Captain” on the sign-in page"
          onToggle={() => togglePoweredBy(!poweredBy)}
        />
      </label>

      {error && (
        <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
          {error}
        </div>
      )}
    </div>
  )
}
