// REST client for cross-user resource sharing (/fd/shares).

import { useAuthStore, refreshAccessToken } from '../stores/authStore'

const FD_BASE = '/fd'

function _authHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`
  return headers
}

async function fdFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const _state = useAuthStore.getState()
  if (_state.authEnabled === true && !_state.token) {
    const refreshed = await refreshAccessToken()
    if (!refreshed) throw new Error('Not authenticated')
  }
  const res = await fetch(`${FD_BASE}${path}`, {
    headers: _authHeaders(),
    credentials: 'include',
    ...init,
  })
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    const refreshed = await refreshAccessToken()
    if (refreshed) {
      const retry = await fetch(`${FD_BASE}${path}`, {
        headers: _authHeaders(), credentials: 'include', ...init,
      })
      if (retry.ok) return retry.json()
    }
    useAuthStore.getState().clearAuth()
    throw new Error('Session expired')
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status}`)
  }
  return res.json()
}

// ── Types ──

export type ResourceType = 'archetype' | 'code' | 'basna' | 'council' | 'vfs'
export type Permission = 'view' | 'edit'

export interface ShareUser {
  id: string
  email: string
  display_name: string
}

export interface ResourceShare {
  grantee_id: string
  grantee_email: string
  grantee_name: string
  permission: Permission
}

export interface SharedWithMe {
  resource_type: ResourceType
  resource_id: string
  owner_id: string
  owner_email: string
  owner_name: string
  permission: Permission
}

// ── Endpoints ──

export const listShareUsers = () =>
  fdFetch<{ users: ShareUser[] }>('/shares/users').then((r) => r.users)

export const listResourceShares = (resourceType: ResourceType, resourceId: string) =>
  fdFetch<{ shares: ResourceShare[] }>(
    `/shares?resource_type=${resourceType}&resource_id=${encodeURIComponent(resourceId)}`,
  ).then((r) => r.shares)

export const listSharedWithMe = (resourceType?: ResourceType) =>
  fdFetch<{ shares: SharedWithMe[] }>(
    `/shares/mine${resourceType ? `?resource_type=${resourceType}` : ''}`,
  ).then((r) => r.shares)

export const createShare = (
  resourceType: ResourceType, resourceId: string, granteeId: string, permission: Permission,
) =>
  fdFetch<{ ok: boolean }>('/shares', {
    method: 'POST',
    body: JSON.stringify({
      resource_type: resourceType, resource_id: resourceId,
      grantee_id: granteeId, permission,
    }),
  })

export const deleteShare = (resourceType: ResourceType, resourceId: string, granteeId: string) =>
  fdFetch<{ ok: boolean }>(
    `/shares?resource_type=${resourceType}&resource_id=${encodeURIComponent(resourceId)}&grantee_id=${granteeId}`,
    { method: 'DELETE' },
  )

export const leaveShare = (resourceType: ResourceType, resourceId: string, ownerId: string) =>
  fdFetch<{ ok: boolean }>(
    `/shares/leave?resource_type=${resourceType}&resource_id=${encodeURIComponent(resourceId)}&owner_id=${ownerId}`,
    { method: 'DELETE' },
  )
