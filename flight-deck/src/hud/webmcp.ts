// Optional voice control: Meta AI calls tools the page registers on
// document.modelContext (WebMCP; off by default, enabled per device). The app
// stays fully usable by D-pad without it.
//
// Rules from Meta's ai-glasses-webapp-webmcp skill:
//   - feature-detect document.modelContext.registerTool, never polyfill;
//   - registerTool may throw synchronously or return a promise / handle /
//     undefined — guard every call, swallow a rejected promise;
//   - scalar parameters only; enum/default/min/max are dropped, so constraints
//     live in descriptions and are validated in execute();
//   - names ≤ 57 effective chars, not reserved (openUrl, goBack, …);
//   - there is no failure flag: handled problems are ordinary JSON results
//     with `error`, `message` and `next_action`; every result ends with
//     `next_action` ("speech ends the turn");
//   - unregister through the AbortSignal passed at registration.

import { findAgent } from './agentsStore'
import { chat, useHudChat } from './chat/chatStore'
import { speakableText } from './chat/text'
import { getRoute, navigate, type Route, type Tab } from './router'

interface ToolProperty { type: 'string' | 'boolean' | 'integer' | 'number'; description: string }

interface HudTool {
  name: string
  description: string
  inputSchema: { type: 'object'; properties: Record<string, ToolProperty>; required?: string[] }
  annotations?: { readOnlyHint?: boolean }
  execute: (input: Record<string, unknown>) => string
}

interface ModelContextLike {
  registerTool?: (tool: unknown, options?: { signal?: AbortSignal }) => unknown
}

function problem(message: string, next_action: string): string {
  return JSON.stringify({ error: true, message, next_action })
}

function screenName(r: Route): string {
  return r.v === 'agent' ? r.t : r.v
}

/** The agent the wearer is looking at (route) or chatting with (session). */
function currentAgentId(): string | null {
  const r = getRoute()
  return 'a' in r ? r.a : useHudChat.getState().agentId
}

function lastReply(): string {
  const msgs = useHudChat.getState().messages
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === 'assistant') return speakableText(msgs[i].text, 600)
  }
  return ''
}

const sendMessage: HudTool = {
  name: 'claw_send_message',
  description:
    'Send the wearer\'s message to the Captain Claw agent open on the glasses. Pass the whole message in one call. '
    + 'The reply appears on screen when the agent finishes (often after several seconds); '
    + 'call claw_get_state later to hear it.',
  inputSchema: {
    type: 'object',
    properties: {
      text: { type: 'string', description: 'The message, in the wearer\'s words. Up to 4000 characters.' },
    },
    required: ['text'],
  },
  execute: (input) => {
    const text = typeof input?.text === 'string' ? input.text.trim() : ''
    if (!text) return problem('The message is empty.', 'Ask the wearer what to send, then call again with the text.')
    if (text.length > 4000) return problem('The message is longer than 4000 characters.', 'Shorten it and call again.')
    const st = useHudChat.getState()
    const agent = st.agentId ? findAgent(st.agentId) : null
    if (!st.agentId) {
      return problem('No agent is open.', 'Ask the wearer to pick an agent, or call claw_open_screen with screen "agents". Stop and talk.')
    }
    if (st.closed) return problem(st.error || 'The connection to the agent was closed.', 'Tell the wearer the agent is unavailable. Stop and talk.')
    if (!st.connected) return problem('Still connecting to the agent.', 'Tell the wearer to try again in a few seconds. Stop and talk.')
    if (st.busy) {
      return problem('The agent is still working on the previous message.', 'Tell the wearer to wait for the current reply, then try again. Stop and talk.')
    }
    if (!chat.send(text)) return problem('The message could not be sent.', 'Tell the wearer to try again. Stop and talk.')
    // Show the conversation if the wearer is on another tab of this agent.
    const r = getRoute()
    if (r.v === 'agent' && r.a === st.agentId && r.t !== 'chat') navigate({ v: 'agent', a: r.a, t: 'chat' }, { replace: true })
    return JSON.stringify({
      sent: true,
      agent: agent?.name ?? null,
      next_action: 'Tell the wearer in one short sentence that the message was sent and the reply will appear on the display. Stop and talk.',
    })
  },
}

const getState: HudTool = {
  name: 'claw_get_state',
  description:
    'Report what Captain Claw shows: the screen, the open agent, whether it is connected or still working, '
    + 'its latest reply as plain text, and any approval waiting on the display. Call this before acting.',
  inputSchema: { type: 'object', properties: {} },
  annotations: { readOnlyHint: true },
  execute: () => {
    const st = useHudChat.getState()
    const agentId = currentAgentId()
    const agent = agentId ? findAgent(agentId) : null
    const attached = !!st.agentId && st.agentId === agentId
    const reply = attached ? lastReply() : ''
    let next_action = 'Answer the wearer from this state in one or two short sentences. Stop and talk.'
    if (attached && st.approval) next_action = 'Tell the wearer the agent asks for approval and that they can Approve or Deny on the display. Stop and talk.'
    else if (attached && st.busy) next_action = 'Tell the wearer the agent is still working. Stop and talk.'
    else if (reply) next_action = 'Read or summarize latest_reply for the wearer in one or two short sentences. Stop and talk.'
    return JSON.stringify({
      screen: screenName(getRoute()),
      agent: agent?.name ?? null,
      connected: attached && st.connected,
      busy: attached && st.busy,
      status: attached && st.busy ? st.status : '',
      latest_reply: reply,
      approval_pending: attached && st.approval ? st.approval.message.slice(0, 300) : null,
      suggested_replies: attached && !st.busy ? st.nextSteps.slice(0, 4).map((s) => s.label) : [],
      unread_replies: st.unread,
      next_action,
    })
  },
}

const SCREENS = new Set(['chat', 'files', 'data', 'agents'])

const openScreen: HudTool = {
  name: 'claw_open_screen',
  description: 'Open a Captain Claw screen: chat, files or data of the current agent, or agents for the agent list.',
  inputSchema: {
    type: 'object',
    properties: {
      screen: { type: 'string', description: 'One of: chat, files, data, agents.' },
    },
    required: ['screen'],
  },
  execute: (input) => {
    const screen = typeof input?.screen === 'string' ? input.screen.trim().toLowerCase() : ''
    if (!SCREENS.has(screen)) return problem(`Unknown screen "${screen}".`, 'Call again with screen chat, files, data or agents.')
    const r = getRoute()
    if (screen === 'agents') {
      if (r.v !== 'agents') navigate({ v: 'agents' })
      return JSON.stringify({ opened: 'agents', next_action: 'Confirm in one short sentence. Stop and talk.' })
    }
    const agentId = currentAgentId()
    const agent = agentId ? findAgent(agentId) : null
    if (!agent) {
      return problem('No agent is open.', 'Ask the wearer which agent to open, or call claw_open_screen with screen "agents". Stop and talk.')
    }
    if (!agent.running) return problem(`${agent.name} is not running.`, 'Tell the wearer to start it from Flight Deck. Stop and talk.')
    const tab = screen as Tab
    if ((tab === 'files' && !agent.caps.files) || (tab === 'data' && !agent.caps.data)) {
      return problem(`${agent.name} does not share its ${tab}.`, 'Tell the wearer this screen is not available for this agent. Stop and talk.')
    }
    if (!(r.v === 'agent' && r.a === agent.id && r.t === tab)) {
      // A tab switch replaces (Back leaves the agent); from a deeper screen, push.
      navigate({ v: 'agent', a: agent.id, t: tab }, { replace: r.v === 'agent' && r.a === agent.id })
    }
    return JSON.stringify({ opened: tab, agent: agent.name, next_action: 'Confirm in one short sentence. Stop and talk.' })
  },
}

/** Register one tool; a failure must never break the app or the other tools. */
function safeRegister(ctx: ModelContextLike, tool: HudTool, signal: AbortSignal): void {
  const wrapped = {
    ...tool,
    execute: (input: unknown) => {
      try {
        return tool.execute(input && typeof input === 'object' ? (input as Record<string, unknown>) : {})
      } catch (e) {
        return problem(e instanceof Error ? e.message : 'Something went wrong.', 'Tell the wearer it did not work. Stop and talk.')
      }
    },
  }
  try {
    const result = ctx.registerTool!(wrapped, { signal })
    if (result && typeof (result as PromiseLike<unknown>).then === 'function') {
      (result as PromiseLike<unknown>).then(undefined, () => { /* host rejected this tool */ })
    }
  } catch {
    /* host rejected this tool (bad definition, aborted signal) */
  }
}

/** Register the HUD's WebMCP tools if the host supports it. Returns a cleanup. */
export function registerWebMcpTools(): () => void {
  const ctx = (document as unknown as { modelContext?: ModelContextLike }).modelContext
  if (!ctx || typeof ctx.registerTool !== 'function') return () => {}
  const ac = new AbortController()
  for (const tool of [getState, sendMessage, openScreen]) safeRegister(ctx, tool, ac.signal)
  return () => ac.abort()
}
