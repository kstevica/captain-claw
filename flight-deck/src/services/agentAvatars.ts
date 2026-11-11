// Pictures a person uploads for their agents (/fd/agent-avatars,
// captain_claw/flight_deck/agent_avatar_routes.py). Scaled down here to a small
// square before they go up; read back with the person's sign-in (an <img> can't
// send it), once per visit, then shown from memory.

import { useAuthStore, refreshAccessToken } from '../stores/authStore'

const BASE = '/fd/agent-avatars'
/** Sides of the square that goes up: sharp at three times the largest place it shows. */
const SIDE = 192
const MAX_BYTES = 256 * 1024  // the server's limit

function headers(json = false): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  return {
    ...(json ? { 'Content-Type': 'application/json' } : {}),
    ...(authEnabled && token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

async function call(path: string, init: RequestInit & { json?: boolean } = {}): Promise<Response> {
  const { json, ...rest } = init
  const build = (): RequestInit => ({ ...rest, headers: headers(json), credentials: 'include' })
  let res = await fetch(`${BASE}${path}`, build())
  if (res.status === 401 && useAuthStore.getState().authEnabled && (await refreshAccessToken())) {
    res = await fetch(`${BASE}${path}`, build())
  }
  return res
}

async function detail(res: Response): Promise<string> {
  try {
    const body = await res.json()
    if (typeof body?.detail === 'string') return body.detail
  } catch { /* not JSON */ }
  return `HTTP ${res.status}`
}

/** The file as a small square (the middle of it), ready to upload. */
export async function squareDataUrl(file: File): Promise<string> {
  if (!file.type.startsWith('image/')) throw new Error('Pick a picture: PNG, JPEG, WebP or GIF.')
  const url = URL.createObjectURL(file)
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image()
      el.onload = () => resolve(el)
      el.onerror = () => reject(new Error('That picture could not be opened.'))
      el.src = url
    })
    const w = img.naturalWidth, h = img.naturalHeight
    if (!w || !h) throw new Error('That picture could not be opened.')
    const crop = Math.min(w, h)
    const side = Math.min(SIDE, crop)
    const canvas = document.createElement('canvas')
    canvas.width = side
    canvas.height = side
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('This browser can’t prepare pictures.')
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(img, (w - crop) / 2, (h - crop) / 2, crop, crop, 0, 0, side, side)
    // WebP where the browser writes it (it keeps transparency), else PNG; a
    // PNG photo too large for the server goes as JPEG.
    let out = canvas.toDataURL('image/webp', 0.88)
    if (!out.startsWith('data:image/webp')) out = canvas.toDataURL('image/png')
    if (out.length * 0.75 > MAX_BYTES) out = canvas.toDataURL('image/jpeg', 0.86)
    return out
  } finally {
    URL.revokeObjectURL(url)
  }
}

/** Uploads a picture; its id. */
export async function uploadAvatarImage(dataUrl: string): Promise<string> {
  const res = await call('', { method: 'POST', json: true, body: JSON.stringify({ data_url: dataUrl }) })
  if (!res.ok) throw new Error(await detail(res))
  const { id } = await res.json()
  if (typeof id !== 'string') throw new Error('The server did not keep the picture.')
  return id
}

export async function deleteAvatarImage(id: string): Promise<void> {
  try {
    await call(`/${id}`, { method: 'DELETE' })
  } catch { /* stays on the server: harmless */ }
  const shown = loaded.get(id)
  loaded.delete(id)
  pending.delete(id)
  if (shown) URL.revokeObjectURL(shown)
}

// Read once a visit: an id never changes what it holds.
const loaded = new Map<string, string>()
const pending = new Map<string, Promise<string | null>>()

/** Already read this visit: its object URL. */
export function avatarImageNow(id: string): string | null {
  return loaded.get(id) ?? null
}

/** Its object URL, or null when it can't be read (gone, or offline: tried again next time). */
export function loadAvatarImage(id: string): Promise<string | null> {
  const have = loaded.get(id)
  if (have) return Promise.resolve(have)
  let p = pending.get(id)
  if (!p) {
    p = call(`/${id}`)
      .then((res) => (res.ok ? res.blob() : null))
      .then((blob) => {
        if (!blob) return null
        const url = URL.createObjectURL(blob)
        loaded.set(id, url)
        return url
      })
      .catch(() => null)
      .finally(() => pending.delete(id))
    pending.set(id, p)
  }
  return p
}

/** A picture just uploaded: shown from the file at once, not read back. */
export function rememberAvatarImage(id: string, dataUrl: string): void {
  if (!loaded.has(id)) loaded.set(id, dataUrl)
}
