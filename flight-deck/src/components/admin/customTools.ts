// Custom tools (external_tools on the server): the admin's two switches, worded
// once for both Admin → Settings and the console's Settings page.

export const CUSTOM_TOOLS_TITLE = 'Custom tools'

export const CUSTOM_TOOLS_HINT =
  'Agents get Python tools that Crew’s coding agents build for them, and Flight Deck runs each call with time and memory limits. ' +
  'Making a tool needs Crew.'

export const EVERYBODY_APPROVAL_TITLE = 'Approve tools shared with everybody'

export const EVERYBODY_APPROVAL_HINT =
  'A tool someone offers to everybody waits for an admin before others can add it. Off: everyone is offered it at once. ' +
  'Pending ones are under Custom tools → Approvals.'

export const SANDBOX_TITLE = 'Run tools only in the sandbox'

export const SANDBOX_HINT =
  'On Linux each tool runs in a Landlock sandbox: it reads only its own files and the system’s, writes only its call’s folder, ' +
  'has no network unless it says it needs it, and can’t signal anything outside. On: a host without the sandbox runs no tools.'

/** One line on what this host offers, from /fd/external-tools/status. */
export function sandboxLine(s: { os: string; landlock: number; files: boolean; network: boolean; signals: boolean } | undefined): string {
  if (!s) return ''
  if (!s.files) return `This host has no sandbox (${s.os === 'darwin' ? 'macOS' : s.os}): tools run as Flight Deck’s own user and can reach what it can.`
  const parts = ['files', ...(s.network ? ['network'] : []), ...(s.signals ? ['signals'] : [])]
  return `This host has the sandbox (Landlock ${s.landlock}: ${parts.join(', ')}).`
}
