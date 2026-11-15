import { api } from './fileLibraryApi'

// Approval cards (captain_claw/flight_deck/approvals.py): what an agent must
// ask before it pays, sends, posts, shares, deletes or stores a password. The
// card itself is a question in PersonAsks (source "approval").

export type ApprovalKind = 'pay' | 'send' | 'post' | 'share' | 'delete' | 'credentials'
export type ApprovalMode = 'off' | 'ask' | 'always_ask' | 'people_only'

export interface ApprovalPolicy {
  policy: Record<ApprovalKind, ApprovalMode>
  kinds: { kind: ApprovalKind; label: string }[]
  modes: { mode: ApprovalMode; label: string }[]
}

export interface AlwaysAllowed {
  agent: string
  agent_name: string
  kind: ApprovalKind
  kind_label: string
}

export const MODE_HINTS: Record<ApprovalMode, string> = {
  off: 'Agents do it without asking.',
  ask: 'Agents ask first; people can allow it once, for the conversation, or always for that agent.',
  always_ask: 'Agents ask every time, with no "always".',
  people_only: "Agents never do it: people do it themselves.",
}

export const approvalsApi = {
  policy: () => api<ApprovalPolicy>('/fd/approvals/policy'),
  setPolicy: (policy: Partial<Record<ApprovalKind, ApprovalMode>>) =>
    api<ApprovalPolicy>('/fd/approvals/policy', { method: 'PUT', body: JSON.stringify({ policy }) }),
  mine: () => api<{ always: AlwaysAllowed[] }>('/fd/approvals/mine'),
  forget: (agent: string, kind: ApprovalKind) =>
    api<{ ok: boolean }>(`/fd/approvals/mine/${encodeURIComponent(agent)}/${encodeURIComponent(kind)}`, { method: 'DELETE' }),
}
