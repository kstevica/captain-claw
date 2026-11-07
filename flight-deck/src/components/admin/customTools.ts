// Custom tools (external_tools on the server): the admin's two switches, worded
// once for both Admin → Settings and the console's Settings page.

export const CUSTOM_TOOLS_TITLE = 'Custom tools'

export const CUSTOM_TOOLS_HINT =
  'Agents get Python tools that Crew’s coding agents build for them, and Flight Deck runs each call with time and memory limits. ' +
  'Making a tool needs Crew. A tool runs as Flight Deck’s own user, so it can reach what a Crew worker can.'

export const EVERYBODY_APPROVAL_TITLE = 'Approve tools shared with everybody'

export const EVERYBODY_APPROVAL_HINT =
  'A tool someone offers to everybody waits for an admin before others can add it. Off: everyone is offered it at once. ' +
  'Pending ones are under Custom tools → Approvals.'
