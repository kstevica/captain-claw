import { useAuthStore } from '../stores/authStore'

/** A file on an agent's disk, viewable in the browser through Flight Deck's
 *  proxy (chat images, the queue's picture thumbnails). An <img> or a new tab
 *  sends no Authorization header, so on a deck with sign-in the session
 *  token rides along as `fd_token` (as the file browser's view URLs do). */
export function agentFileViewUrl(path: string, host?: string, port?: number, auth?: string): string {
  const params = new URLSearchParams({ path })
  if (auth) params.set('token', auth)
  const { token, authEnabled } = useAuthStore.getState()
  if (authEnabled && token) params.set('fd_token', token)
  return `/fd/agent-file-view/${encodeURIComponent(host || 'localhost')}/${port}?${params}`
}
