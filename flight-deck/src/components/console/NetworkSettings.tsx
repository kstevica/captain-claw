import { useCallback, useEffect, useState } from 'react'
import { Cable, Share2, Save, Stethoscope, Lock, Trash2, Loader2, CheckCircle2, XCircle, Bot, Globe, Ticket } from 'lucide-react'
import {
  getBotportConfig, saveBotportConfig, getBotportAgents, probeBotport, botportEnvVar, setBotportShare, parseJoinCode,
  type BotportConfig, type BotportConfigPatch, type BotportLocalAgent, type BotportProbe,
} from '../../services/botportLink'
import { useNetworkShareStore } from '../../stores/networkShareStore'
import { Switch } from '../admin/Switch'
import { Card, SettingRow, Segmented, Notice, Button, Field, Pill, Dot, inputCls } from './parts'

// BotPort → the link's settings (admin): where the hub is and who this deck is
// on it, then what other decks may use here. Both cards save through the same
// PUT /fd/botport/config, each with only the fields it changed — so saving one
// never touches the other's half-typed edits.

type Save = (patch: BotportConfigPatch) => Promise<BotportConfig>

// The server's own rule (botport_link.valid_deck_name).
const DECK_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$/

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

/** "a, b ,a" → ['a', 'b']. */
function splitList(s: string): string[] {
  return [...new Set(s.split(',').map((x) => x.trim()).filter(Boolean))]
}

function sameList(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((x, i) => x === b[i])
}

/** The settings cards: load the config once, then each card saves its part. */
export function NetworkSettings({ onSaved }: { onSaved: () => void }) {
  const [config, setConfig] = useState<BotportConfig | null>(null)
  const [agents, setAgents] = useState<BotportLocalAgent[] | null>(null)
  const [error, setError] = useState('')

  const loadAgents = useCallback(() => getBotportAgents().then(
    (r) => setAgents(r.agents),
    () => setAgents((a) => a ?? []),
  ), [])

  useEffect(() => {
    let cancelled = false
    getBotportConfig()
      .then((c) => { if (!cancelled) setConfig(c) })
      .catch((e) => { if (!cancelled) setError(errText(e)) })
    loadAgents()
    return () => { cancelled = true }
  }, [loadAgents])

  const save = useCallback<Save>(async (patch) => {
    const next = await saveBotportConfig(patch)
    setConfig(next)
    onSaved()
    // Agent cards show what is on the network: let them catch up.
    void useNetworkShareStore.getState().fetch(true)
    // Which agents count as exposed follows the saved list (and users' shares).
    if (patch.expose || patch.user_sharing !== undefined) loadAgents()
    return next
  }, [onSaved, loadAgents])

  // 403 on a deck with sign-in off, from another machine: say why.
  if (!config) {
    return error ? <Notice tone="red">{error}</Notice> : (
      <Card icon={Cable} title="Connection">
        <div className="flex items-center gap-2 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> Loading settings…</div>
      </Card>
    )
  }
  return (
    <>
      <ConnectionCard config={config} save={save} />
      <ExposeCard config={config} agents={agents} save={save} reloadAgents={loadAgents} />
    </>
  )
}

/** One card's save: busy, the error it got, a short "Saved". */
function useSaving(save: Save) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [flash, setFlash] = useState('')
  const run = async (patch: BotportConfigPatch, done = 'Saved'): Promise<BotportConfig | null> => {
    setBusy(true)
    setError('')
    try {
      const next = await save(patch)
      setFlash(done)
      window.setTimeout(() => setFlash(''), 2000)
      return next
    } catch (e) {
      setError(errText(e))
      return null
    } finally {
      setBusy(false)
    }
  }
  return { busy, error, flash, run }
}

function LockHint({ field }: { field: string }) {
  return (
    <p className="mt-1.5 flex items-center gap-1 text-[11px] text-amber-700 dark:text-amber-400">
      <Lock className="h-3 w-3 shrink-0" /> Set by {botportEnvVar(field)} in the environment
    </p>
  )
}

/** What a hub check found. */
export function ProbeLine({ probe }: { probe: BotportProbe }) {
  return probe.ok ? (
    <p className="flex items-start gap-1.5 text-xs text-emerald-700 dark:text-emerald-400">
      <CheckCircle2 className="mt-px h-3.5 w-3.5 shrink-0" />
      <span className="min-w-0 break-words">
        BotPort{probe.version ? ` v${probe.version.replace(/^v/, '')}` : ''} reachable
        {probe.protocol ? ` (protocol ${probe.protocol})` : ''}
      </span>
    </p>
  ) : (
    <p className="flex items-start gap-1.5 text-xs text-red-700 dark:text-red-400">
      <XCircle className="mt-px h-3.5 w-3.5 shrink-0" />
      <span className="min-w-0 break-words">{probe.error || 'The hub did not answer'}</span>
    </p>
  )
}

// ── Connection ──

interface ConnDraft {
  enabled: boolean
  url: string
  deck_name: string
  key: string
  secret: string
  allow_outbound: boolean
  outbound_users: boolean
}

const connDraft = (c: BotportConfig): ConnDraft => ({
  enabled: c.enabled, url: c.url, deck_name: c.deck_name, key: c.key, secret: '', allow_outbound: c.allow_outbound,
  outbound_users: c.outbound_users ?? false,
})

function ConnectionCard({ config, save }: { config: BotportConfig; save: Save }) {
  const [d, setD] = useState(() => connDraft(config))
  const [probe, setProbe] = useState<BotportProbe | null>(null)
  const [probing, setProbing] = useState(false)
  const { busy, error, flash, run } = useSaving(save)
  const locked = (f: string) => config.env_locked.includes(f)
  const set = <K extends keyof ConnDraft>(k: K, v: ConnDraft[K]) => setD((p) => ({ ...p, [k]: v }))
  const [join, setJoin] = useState('')
  const [joined, setJoined] = useState('')
  const credFields = ['url', 'deck_name', 'key', 'secret']
  const joinable = credFields.some((f) => !locked(f))
  const joinBad = join.trim().length > 8 && parseJoinCode(join) === null

  // A join code from the hub's Decks page fills in the whole connection (the
  // fields the environment pins keep their values); Save then connects.
  const applyJoin = (text: string) => {
    setJoin(text)
    const j = parseJoinCode(text)
    if (!j) return
    setD((p) => ({
      ...p,
      enabled: locked('enabled') ? p.enabled : true,
      url: locked('url') ? p.url : j.url,
      deck_name: locked('deck_name') ? p.deck_name : j.deck_name,
      key: locked('key') ? p.key : j.key,
      secret: locked('secret') ? p.secret : j.secret,
    }))
    setProbe(null)
    setJoin('')
    setJoined(j.deck_name)
  }

  // Only what changed (a pinned field can't change: its input is disabled).
  const patch: BotportConfigPatch = {}
  if (d.enabled !== config.enabled) patch.enabled = d.enabled
  if (d.url.trim() !== config.url) patch.url = d.url.trim()
  if (d.deck_name.trim() !== config.deck_name) patch.deck_name = d.deck_name.trim()
  if (d.key.trim() !== config.key) patch.key = d.key.trim()
  if (d.secret.trim()) patch.secret = d.secret.trim()
  if (d.allow_outbound !== config.allow_outbound) patch.allow_outbound = d.allow_outbound
  if (d.outbound_users !== (config.outbound_users ?? false)) patch.outbound_users = d.outbound_users
  const dirty = Object.keys(patch).length > 0
  const badName = d.deck_name.trim() !== '' && !DECK_NAME_RE.test(d.deck_name.trim())

  const submit = async () => {
    const next = await run(patch)
    if (next) {
      setD(connDraft(next))
      setJoined('')
    }
  }

  const test = async () => {
    setProbing(true)
    setProbe(null)
    try {
      setProbe(await probeBotport(d.url.trim()))
    } catch (e) {
      setProbe({ ok: false, error: errText(e) })
    } finally {
      setProbing(false)
    }
  }

  const removeSecret = async () => {
    if (!confirm('Remove the stored secret? On a hub with sign-in on, this deck can no longer connect until you enter a new one.')) return
    const next = await run({ clear_secret: true }, 'Secret removed')
    if (next) set('secret', '')
  }

  return (
    <Card
      icon={Cable}
      title="Connection"
      hint="The hub this deck connects to and its name there. Credentials come from the hub's admin."
    >
      <div className="space-y-5">
        {joinable && (
          <div className="rounded-xl border border-cyan-500/25 bg-gradient-to-br from-cyan-500/[0.06] via-transparent to-violet-500/[0.07] p-4">
            <div className="flex items-start gap-3">
              <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-400 via-indigo-400 to-violet-500 text-white shadow-md shadow-cyan-500/20">
                <Ticket className="h-4 w-4" />
              </span>
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-zinc-100">Join code</div>
                <p className="mt-0.5 text-xs leading-relaxed text-zinc-500">
                  {config.key && config.secret_set ? (
                    <>A new one from the hub (after it replaced this deck's secret, say)? Paste it here and Save.</>
                  ) : (
                    <>
                      The hub's dashboard makes one under <span className="font-medium text-zinc-300">Decks → Add Flight Deck</span>.
                      Paste it here and the address, name, key and secret below fill in; then Save.
                    </>
                  )}
                </p>
                <input
                  value={join}
                  onChange={(e) => applyJoin(e.target.value)}
                  placeholder="bpj1.…"
                  aria-label="Join code"
                  spellCheck={false}
                  autoComplete="off"
                  className={`${inputCls} mt-2.5 font-mono`}
                />
                {joinBad && (
                  <p className="mt-1.5 text-[11px] text-red-700 dark:text-red-400">
                    That isn't a whole join code. Copy it again from the hub's Decks page: it starts with bpj1.
                  </p>
                )}
                {joined && (
                  <div className="mt-2.5">
                    <Notice tone="emerald">
                      Filled in for <span className="font-mono">{joined}</span>
                      {credFields.some(locked) ? ' (fields pinned by the environment keep their values)' : ''}. Save to connect.
                    </Notice>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        <div className="divide-y divide-zinc-800/80">
          <SettingRow
            title="Connect to BotPort"
            hint={locked('enabled') ? <LockHint field="enabled" /> : 'Keep a link open to the hub. Off: this deck is not on the network.'}
          >
            <Switch on={d.enabled} disabled={locked('enabled')} label="Connect to BotPort" onToggle={() => set('enabled', !d.enabled)} />
          </SettingRow>
          <SettingRow
            title="Let this deck's agents consult other decks"
            hint={locked('allow_outbound') ? <LockHint field="allow_outbound" />
              : 'Off: agents here can\'t reach remote agents. Other decks can still use the agents you expose below.'}
          >
            <Switch
              on={d.allow_outbound}
              disabled={locked('allow_outbound')}
              label="Let this deck's agents consult other decks"
              onToggle={() => set('allow_outbound', !d.allow_outbound)}
            />
          </SettingRow>
          {d.allow_outbound && (
            <SettingRow
              title="Every user's agents too"
              hint={locked('outbound_users') ? <LockHint field="outbound_users" />
                : d.outbound_users
                  ? 'Every user\'s agents can see and consult the agents other decks share.'
                  : 'Off: only admins and their agents reach other decks — usually your own decks, which other users shouldn\'t see.'}
            >
              <Switch
                on={d.outbound_users}
                disabled={locked('outbound_users')}
                label="Every user's agents too"
                onToggle={() => set('outbound_users', !d.outbound_users)}
              />
            </SettingRow>
          )}
        </div>

        <Field
          label="Hub address"
          hint={<>
            For example <code className="text-zinc-400">ws://192.168.1.10:23180/ws</code> or{' '}
            <code className="text-zinc-400">wss://botport.example.com/ws</code> — http(s)://host:port and host:port work too.
            {config.normalized_url && d.url.trim() === config.url && (
              <span className="mt-0.5 block break-all">Connects to <code className="text-zinc-400">{config.normalized_url}</code></span>
            )}
          </>}
        >
          <div className="flex gap-2">
            <input
              value={d.url}
              onChange={(e) => {
                // A join code pasted here instead of in its own box works too.
                if (parseJoinCode(e.target.value)) applyJoin(e.target.value)
                else { set('url', e.target.value); setProbe(null) }
              }}
              disabled={locked('url')}
              placeholder="ws://host:23180/ws"
              spellCheck={false}
              autoComplete="off"
              className={`${inputCls} min-w-0 font-mono`}
            />
            <Button icon={Stethoscope} busy={probing} disabled={!d.url.trim()} onClick={test} title="Check the hub answers at this address (does not connect)">
              Test
            </Button>
          </div>
          {locked('url') && <LockHint field="url" />}
          {probe && <div className="mt-2"><ProbeLine probe={probe} /></div>}
        </Field>

        <Field
          label="Deck name"
          hint={<>
            This deck's name on the network: letters, digits, - _ . (no spaces or @). Other decks see its agents as{' '}
            <code className="text-zinc-400">Agent@{d.deck_name.trim() || 'name'}</code>. On a hub with sign-in, the hub ties the name to the key.
          </>}
        >
          <input
            value={d.deck_name}
            onChange={(e) => set('deck_name', e.target.value)}
            disabled={locked('deck_name')}
            placeholder="e.g. acme-hq"
            maxLength={63}
            spellCheck={false}
            autoComplete="off"
            className={inputCls}
          />
          {locked('deck_name') && <LockHint field="deck_name" />}
          {badName && (
            <p className="mt-1.5 text-[11px] text-red-700 dark:text-red-400">Only letters, digits, - _ and . — starting with a letter or digit.</p>
          )}
        </Field>

        <div className="grid grid-cols-1 gap-4 wide:grid-cols-2">
          <Field label="Key">
            <input
              value={d.key}
              onChange={(e) => set('key', e.target.value)}
              disabled={locked('key')}
              placeholder="From the hub admin"
              spellCheck={false}
              autoComplete="off"
              className={`${inputCls} font-mono`}
            />
            {locked('key') && <LockHint field="key" />}
          </Field>
          <Field label="Secret">
            <div className="flex gap-2">
              <input
                type="password"
                value={d.secret}
                onChange={(e) => set('secret', e.target.value)}
                disabled={locked('secret')}
                placeholder={config.secret_set ? '••• set — type to replace' : 'From the hub admin'}
                autoComplete="new-password"
                className={`${inputCls} min-w-0 font-mono`}
              />
              {config.secret_set && !locked('secret') && (
                <Button variant="danger" icon={Trash2} disabled={busy} onClick={removeSecret} title="Remove the stored secret">
                  Remove
                </Button>
              )}
            </div>
            {locked('secret') && <LockHint field="secret" />}
          </Field>
        </div>
        <p className="-mt-2 text-[11px] leading-relaxed text-zinc-500">
          The hub's dashboard makes both (Decks → Add Flight Deck), as does <code className="text-zinc-400">botport gen-key &lt;deck name&gt;</code> on
          the hub. Leave them blank for a hub without sign-in. The secret is never shown again.
        </p>

        {error && <Notice tone="red">{error}</Notice>}

        <div className="flex flex-wrap items-center justify-end gap-2">
          {dirty && <Button variant="ghost" disabled={busy} onClick={() => setD(connDraft(config))}>Discard</Button>}
          <Button variant="primary" icon={Save} busy={busy} disabled={!dirty || badName} onClick={submit}>
            {flash || 'Save'}
          </Button>
        </div>
      </div>
    </Card>
  )
}

// ── Exposed agents ──

interface ExposeDraft {
  all: boolean
  picked: string[]
  acceptAny: boolean
  acceptList: string
  maxConcurrent: string
  timeout: string
  denyTools: string
  userSharing: boolean
}

const exposeDraft = (c: BotportConfig): ExposeDraft => ({
  all: c.expose.includes('*'),
  picked: c.expose.filter((x) => x !== '*'),
  acceptAny: c.accept_from.includes('*'),
  acceptList: c.accept_from.filter((x) => x !== '*').join(', '),
  maxConcurrent: String(c.max_concurrent),
  timeout: String(Math.round(c.max_inbound_timeout)),
  denyTools: c.deny_tools.join(', '),
  userSharing: c.user_sharing ?? true,
})

/** Does an `expose` entry name this agent (by slug or name, any case)? */
function names(a: BotportLocalAgent, entry: string): boolean {
  const e = entry.toLowerCase()
  return (!!a.slug && a.slug.toLowerCase() === e) || (!!a.name && a.name.toLowerCase() === e)
}

function intIn(s: string, lo: number, hi: number): number | null {
  const n = Number(s.trim())
  return Number.isInteger(n) && n >= lo && n <= hi ? n : null
}

function ExposeCard({ config, agents, save, reloadAgents }: {
  config: BotportConfig
  agents: BotportLocalAgent[] | null
  save: Save
  reloadAgents: () => void
}) {
  const [d, setD] = useState(() => exposeDraft(config))
  const { busy, error, flash, run } = useSaving(save)
  const locked = (f: string) => config.env_locked.includes(f)
  const set = <K extends keyof ExposeDraft>(k: K, v: ExposeDraft[K]) => setD((p) => ({ ...p, [k]: v }))

  const maxConcurrent = intIn(d.maxConcurrent, 1, 64)
  const timeout = intIn(d.timeout, 30, 3600)

  const patch: BotportConfigPatch = {}
  const expose = d.all ? ['*'] : d.picked
  if (!sameList(expose, config.expose)) patch.expose = expose
  const accept = d.acceptAny ? ['*'] : splitList(d.acceptList)
  if (!sameList(accept, config.accept_from)) patch.accept_from = accept
  if (maxConcurrent !== null && maxConcurrent !== config.max_concurrent) patch.max_concurrent = maxConcurrent
  if (timeout !== null && timeout !== Math.round(config.max_inbound_timeout)) patch.max_inbound_timeout = timeout
  const deny = splitList(d.denyTools)
  if (!sameList(deny, config.deny_tools)) patch.deny_tools = deny
  if (d.userSharing !== (config.user_sharing ?? true)) patch.user_sharing = d.userSharing
  const dirty = Object.keys(patch).length > 0
  const invalid = maxConcurrent === null || timeout === null

  const submit = async () => {
    const next = await run(patch)
    if (next) setD(exposeDraft(next))
  }

  const isPicked = (a: BotportLocalAgent) => d.picked.some((x) => names(a, x))
  const toggle = (a: BotportLocalAgent) => set('picked', isPicked(a)
    ? d.picked.filter((x) => !names(a, x))
    : [...d.picked, a.slug || a.name])
  // Picked names no agent here has any more: kept until the admin removes them.
  const ghosts = agents ? d.picked.filter((x) => !agents.some((a) => names(a, x))) : []

  const toggleAll = () => {
    if (!d.all && !confirm(
      'Expose every agent on this deck — including ones added later? Other decks on the hub (those you accept requests from) '
      + 'can then hand any of them work. On a deck with sign-in, only agents someone owns are exposed.',
    )) return
    set('all', !d.all)
  }

  // An owner shared it themselves: the admin may take it off the network.
  const [revoking, setRevoking] = useState('')
  const [revokeError, setRevokeError] = useState('')
  const revoke = async (a: BotportLocalAgent) => {
    const name = a.name || a.slug
    if (!confirm(`Take '${name}' off the network? Its owner shared it; they can share it again unless you turn sharing by users off.`)) return
    setRevoking(a.slug || a.name)
    setRevokeError('')
    try {
      await setBotportShare(a.slug || a.name, false, a.kind)
      reloadAgents()
      void useNetworkShareStore.getState().fetch(true)
    } catch (e) {
      setRevokeError(errText(e))
    } finally {
      setRevoking('')
    }
  }

  // With "every agent" saved, the server says which ones it really exposes.
  const shownChecked = (a: BotportLocalAgent) => (d.all ? (config.expose.includes('*') ? a.exposed : true) : isPicked(a))
  const exposeLocked = locked('expose')

  return (
    <Card
      icon={Share2}
      title="Exposed agents"
      hint="Which of this deck's agents other decks may use, and on what terms. Nothing is exposed until you pick it."
    >
      <div className="space-y-5">
        <SettingRow
          title="Expose every agent"
          hint={exposeLocked ? <LockHint field="expose" /> : 'Every agent on this deck, now and later. Off: only the ones ticked below.'}
        >
          <Switch on={d.all} disabled={exposeLocked} label="Expose every agent" onToggle={toggleAll} />
        </SettingRow>

        <SettingRow
          title="Let users share their own agents"
          hint={locked('user_sharing') ? <LockHint field="user_sharing" /> : (d.userSharing
            ? 'Each user can put the agents they own on the network from the agent\'s menu (Share on network), next to the ones you pick here.'
            : 'Off: only the list below decides. Users\' shares are kept but don\'t count until you turn this back on.')}
        >
          <Switch on={d.userSharing} disabled={locked('user_sharing')} label="Let users share their own agents"
            onToggle={() => set('userSharing', !d.userSharing)} />
        </SettingRow>

        <div className="space-y-1.5">
          {agents === null ? (
            <div className="flex items-center gap-2 text-sm text-zinc-500"><Loader2 className="h-4 w-4 animate-spin" /> Loading agents…</div>
          ) : agents.length === 0 && ghosts.length === 0 ? (
            <div className="rounded-xl border border-dashed border-zinc-800 px-4 py-6 text-center text-xs text-zinc-500">
              <Bot className="mx-auto mb-2 h-5 w-5 text-zinc-600" />
              No agents on this deck yet.
            </div>
          ) : (
            <>
              {agents.map((a) => (
                <label
                  key={a.slug || a.name}
                  className={`flex items-start gap-3 rounded-xl border border-zinc-800 bg-zinc-950/40 px-3 py-2.5 ${
                    d.all || exposeLocked ? 'opacity-80' : 'cursor-pointer hover:border-zinc-700'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={shownChecked(a)}
                    disabled={d.all || exposeLocked}
                    onChange={() => toggle(a)}
                    className="mt-0.5 h-4 w-4 shrink-0 accent-violet-600"
                  />
                  <span className="min-w-0 flex-1">
                    <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      <span className="min-w-0 truncate text-sm font-medium text-zinc-200">{a.name || a.slug}</span>
                      {a.kind && <Pill>{a.kind}</Pill>}
                      <span className="flex items-center gap-1 text-[11px] text-zinc-500">
                        <Dot tone={a.status === 'running' ? 'on' : 'off'} />{a.status || 'unknown'}
                      </span>
                      {a.by_owner && (
                        <>
                          <Pill tone={config.user_sharing ? 'sky' : 'zinc'} icon={Globe}
                            title={config.user_sharing ? 'Its owner shared it on the network' : 'Its owner shared it, but sharing by users is off'}>
                            shared by owner{config.user_sharing ? '' : ' (off)'}
                          </Pill>
                          <button
                            type="button"
                            disabled={revoking === (a.slug || a.name)}
                            onClick={(e) => { e.preventDefault(); e.stopPropagation(); void revoke(a) }}
                            className="rounded-md px-1.5 py-0.5 text-[11px] text-zinc-500 hover:bg-zinc-800 hover:text-red-600 disabled:opacity-50 dark:hover:text-red-400"
                          >
                            {revoking === (a.slug || a.name) ? 'Stopping…' : 'Stop'}
                          </button>
                        </>
                      )}
                    </span>
                    {a.description && <span className="mt-0.5 block truncate text-xs text-zinc-500" title={a.description}>{a.description}</span>}
                  </span>
                </label>
              ))}
              {!d.all && ghosts.map((g) => (
                <div key={`ghost:${g}`} className="flex items-center gap-3 rounded-xl border border-dashed border-zinc-800 px-3 py-2.5">
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm text-zinc-400">{g}</span>
                    <span className="block text-[11px] text-zinc-500">Not on this deck any more</span>
                  </span>
                  <button
                    type="button"
                    disabled={exposeLocked}
                    onClick={() => set('picked', d.picked.filter((x) => x !== g))}
                    title="Remove from the exposed list"
                    aria-label={`Remove ${g}`}
                    className="rounded-lg p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-red-600 disabled:opacity-50 dark:hover:text-red-400"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
            </>
          )}
          {d.all && (
            <p className="text-[11px] text-zinc-500">Every agent is exposed. Flight Deck's own workers never are.</p>
          )}
          {revokeError && <Notice tone="red">{revokeError}</Notice>}
        </div>

        <div className="space-y-2 border-t border-zinc-800/80 pt-5">
          <div className="text-sm font-medium text-zinc-200">Accept requests from</div>
          <Segmented
            value={d.acceptAny ? 'any' : 'listed'}
            label="Accept requests from"
            onChange={(v) => { if (!locked('accept_from')) set('acceptAny', v === 'any') }}
            options={[
              { value: 'any', label: 'Any deck on the hub' },
              { value: 'listed', label: 'Only these decks' },
            ]}
          />
          {!d.acceptAny && (
            <input
              value={d.acceptList}
              onChange={(e) => set('acceptList', e.target.value)}
              disabled={locked('accept_from')}
              placeholder="deck-a, deck-b"
              spellCheck={false}
              autoComplete="off"
              className={inputCls}
            />
          )}
          {locked('accept_from') ? <LockHint field="accept_from" /> : (
            <p className="text-[11px] leading-relaxed text-zinc-500">
              {d.acceptAny
                ? 'Any deck connected to the hub may send work to the exposed agents. On a hub reachable from the internet, list the decks you trust instead.'
                : 'Deck names, separated by commas. Empty: no deck can send requests.'}
            </p>
          )}
        </div>

        <div className="grid grid-cols-1 gap-4 wide:grid-cols-2">
          <Field label="Remote tasks at once" hint={locked('max_concurrent') ? undefined : 'How many requests from other decks run here together (1–64); beyond that they are turned away with "busy — retry".'}>
            <input
              type="number"
              min={1}
              max={64}
              value={d.maxConcurrent}
              onChange={(e) => set('maxConcurrent', e.target.value)}
              disabled={locked('max_concurrent')}
              className={inputCls}
            />
            {locked('max_concurrent') && <LockHint field="max_concurrent" />}
            {maxConcurrent === null && <p className="mt-1.5 text-[11px] text-red-700 dark:text-red-400">A whole number from 1 to 64.</p>}
          </Field>
          <Field label="Longest remote task (seconds)" hint={`A remote task is stopped after this long (30–3600)${timeout && timeout >= 60 ? ` — about ${Math.round(timeout / 60)} min` : ''}.`}>
            <input
              type="number"
              min={30}
              max={3600}
              step={30}
              value={d.timeout}
              onChange={(e) => set('timeout', e.target.value)}
              className={inputCls}
            />
            {timeout === null && <p className="mt-1.5 text-[11px] text-red-700 dark:text-red-400">A whole number from 30 to 3600.</p>}
          </Field>
        </div>

        <Field
          label="Tools remote requests may not use"
          hint={locked('deny_tools') ? undefined : <>
            Comma-separated, e.g. <code className="text-zinc-400">shell, scripts, write, edit</code>. On top of the
            network tools and the tools bound to the owner's own session and memory (cron, sessions, history,
            personality, …), which a remote request never gets.
          </>}
        >
          <input
            value={d.denyTools}
            onChange={(e) => set('denyTools', e.target.value)}
            disabled={locked('deny_tools')}
            placeholder="shell, scripts, write, edit"
            spellCheck={false}
            autoComplete="off"
            className={`${inputCls} font-mono`}
          />
          {locked('deny_tools') && <LockHint field="deny_tools" />}
        </Field>

        {error && <Notice tone="red">{error}</Notice>}

        <div className="flex flex-wrap items-center justify-end gap-2">
          {dirty && <Button variant="ghost" disabled={busy} onClick={() => setD(exposeDraft(config))}>Discard</Button>}
          <Button variant="primary" icon={Save} busy={busy} disabled={!dirty || invalid} onClick={submit}>
            {flash || 'Save'}
          </Button>
        </div>
      </div>
    </Card>
  )
}
