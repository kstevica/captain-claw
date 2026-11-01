/**
 * Hosting — a user's VFS folders on the web, as a static site or a running app
 * (`captain_claw/flight_deck/hosting_routes.py`). Every read is the caller's own.
 */

import { api } from './fileLibraryApi'

export interface HostedSite {
  name: string
  kind: 'static' | 'app'
  project: string
  subdir: string
  start_cmd: string
  running: boolean
  port: number | null
  /** Its path on this deck, like /vfs/my-site/. */
  url: string
  /** Its folder is gone, so there's nothing to serve. */
  missing: boolean
  /** Since the deck last restarted. */
  visit_count: number
}

export interface SitePublish {
  name: string
  kind: 'static' | 'app'
  project: string
  subdir: string
  start_cmd: string
}

const path = (name: string, rest = '') => `/fd/hosting/${encodeURIComponent(name)}${rest}`

export const hostingApi = {
  list: () => api<{ entries: HostedSite[] }>('/fd/hosting'),
  publish: (body: SitePublish) => api<HostedSite>('/fd/hosting', { method: 'POST', body: JSON.stringify(body) }),
  start: (name: string) => api<{ ok: boolean; message: string }>(path(name, '/start'), { method: 'POST' }),
  stop: (name: string) => api<{ ok: boolean; message: string }>(path(name, '/stop'), { method: 'POST' }),
  remove: (name: string) => api<{ ok: boolean }>(path(name), { method: 'DELETE' }),
  folders: (project: string) => api<{ folders: string[] }>(`/fd/hosting/folders?project=${encodeURIComponent(project)}`),
  projects: () => api<{ projects: { name: string }[] }>('/fd/vfs/projects'),
}
