import { useEffect, useMemo, useState } from 'react'
import { AtSign, Check, IdCard, Inbox, Loader2, Mail, Users } from 'lucide-react'
import { AgentDialog } from './KioskAgentDialogs'
import { SwitchTrack } from '../common/SwitchTrack'
import { colleaguesApi, type ColleagueAgent, type ColleaguesView, type GmailReplies } from '../../services/colleaguesApi'
import { GmailReplyChoice } from '../agents/AgentMailSetting'
import { useNotificationStore } from '../../stores/notificationStore'

// ── Identity & email: one agent as a colleague ───────────────────────
//
// Who it is to the rest of the team (a role and a line on what it does), the
// names it answers to in email, its own email — an IMAP inbox that becomes
// only its own, or aliases that land in your inbox — whether it picks up the
// email meant for it, and which colleagues it may hand work to on the
// Taskboard. Flight Deck: colleagues.py; the mail watcher: agent_mail.py.

const field = 'mt-1 w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-violet-500/60 focus:outline-none'
const label = 'text-xs font-medium text-zinc-300'
const hint = 'mt-1 text-[11px] leading-relaxed text-zinc-500'

const splitList = (text: string) => text.split(/[\n,;]+/).map((s) => s.trim()).filter(Boolean)

export function KioskAgentIdentityDialog({ slug, agentName, onClose }: {
  slug: string
  agentName: string
  onClose: () => void
}) {
  const [view, setView] = useState<ColleaguesView | null>(null)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [role, setRole] = useState('')
  const [about, setAbout] = useState('')
  const [names, setNames] = useState('')
  const [addresses, setAddresses] = useState('')
  const [inbox, setInbox] = useState('')
  const [pickUp, setPickUp] = useState(false)
  const [gmailReplies, setGmailReplies] = useState<GmailReplies>('off')
  const [handsAll, setHandsAll] = useState(true)
  const [handsTo, setHandsTo] = useState<string[]>([])

  useEffect(() => {
    let live = true
    colleaguesApi.list().then((v) => {
      if (!live) return
      setView(v)
      const me = v.agents.find((a) => a.slug === slug)
      if (!me) { setError('This agent isn’t yours.'); return }
      fill(me)
    }).catch((e) => live && setError(String(e?.message || e)))
    return () => { live = false }
  }, [slug])

  const fill = (me: ColleagueAgent) => {
    setRole(me.role)
    setAbout(me.about)
    setNames(me.names_set ? me.names.join(', ') : '')
    setAddresses(me.addresses.join(', '))
    setInbox(me.inbox)
    setPickUp(me.pick_up_mail)
    setGmailReplies(me.gmail_replies || 'off')
    setHandsAll(me.hands_to === 'all')
    setHandsTo(me.hands_to === 'all' ? [] : me.hands_to)
  }

  const others = useMemo(() => (view?.agents || []).filter((a) => a.slug !== slug), [view, slug])
  const chosenInbox = view?.inboxes.find((i) => i.id === inbox)
  const takenBy = chosenInbox ? view?.agents.find((a) => a.slug !== slug && a.inbox === chosenInbox.id) : undefined
  const watched = view ? [view.mail.gmail && 'Gmail', view.mail.outlook && 'Outlook', view.mail.imap && 'your IMAP inboxes']
    .filter(Boolean).join(', ') : ''
  const answersTo = splitList(names)

  const save = async () => {
    setSaving(true)
    setError('')
    try {
      const { agent } = await colleaguesApi.save(slug, {
        role, about, names: answersTo, addresses: splitList(addresses), inbox, pick_up_mail: pickUp, gmail_replies: gmailReplies,
        hands_to: handsAll ? 'all' : handsTo,
      })
      setView((v) => v && { ...v, agents: v.agents.map((a) => (a.slug === slug ? { ...a, ...agent } : a)) })
      fill(agent)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
      useNotificationStore.getState().add('success', 'Saved', `${agentName}'s identity is saved.`)
    } catch (e) {
      setError(String((e as Error)?.message || e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <AgentDialog title="Identity & email" icon={IdCard} agentName={agentName} size="md" onClose={onClose}>
      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
        {!view && !error && (
          <div className="flex h-40 items-center justify-center text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
        )}
        {error && <div className="mb-3 rounded-lg border border-red-500/30 bg-red-500/5 px-3 py-2 text-sm text-red-300">{error}</div>}
        {view && (
          <div className="flex flex-col gap-5">
            <section>
              <h3 className="flex items-center gap-1.5 text-sm font-semibold text-zinc-100"><Users className="h-4 w-4 text-violet-400" />Who it is to the team</h3>
              <p className={hint}>Your other agents see this when they look for someone to hand work to.</p>
              <label className="mt-3 block">
                <span className={label}>Role</span>
                <input className={field} value={role} maxLength={60} placeholder="e.g. Researcher" onChange={(e) => setRole(e.target.value)} />
              </label>
              <label className="mt-3 block">
                <span className={label}>What it does</span>
                <textarea className={`${field} min-h-[4.5rem] resize-y`} value={about} maxLength={300}
                  placeholder="e.g. Finds numbers and sources for client reports" onChange={(e) => setAbout(e.target.value)} />
              </label>
            </section>

            <section>
              <h3 className="flex items-center gap-1.5 text-sm font-semibold text-zinc-100"><Mail className="h-4 w-4 text-rose-400" />Its email</h3>
              <label className="mt-3 block">
                <span className={label}>Its own inbox</span>
                <select className={field} value={inbox} onChange={(e) => setInbox(e.target.value)}>
                  <option value="">None</option>
                  {view.inboxes.map((i) => (
                    <option key={i.id} value={i.id}>{i.label ? `${i.label} — ${i.email}` : i.email}</option>
                  ))}
                </select>
                <p className={hint}>
                  {view.inboxes.length === 0
                    ? 'Connect a mailbox such as researcher@yourfirm.com under Connections → Email (IMAP) to give it one.'
                    : chosenInbox
                      ? `Only ${agentName} will use ${chosenInbox.email}. ${chosenInbox.sending_enabled
                        ? 'Sending is on for it: replies go out within its allowlist and daily limit.'
                        : 'Sending is off for it, so replies wait as drafts — turn it on in Connections → Email (IMAP).'}`
                      : 'A mailbox under Connections → Email (IMAP) that becomes only this agent’s.'}
                  {takenBy ? ` It is ${takenBy.name}'s now; saving moves it here.` : ''}
                </p>
              </label>
              <label className="mt-3 block">
                <span className={label}>Addresses that land in your inbox</span>
                <input className={field} value={addresses} placeholder="you+researcher@gmail.com"
                  onChange={(e) => setAddresses(e.target.value)} />
                <p className={hint}>Mail sent to these reaches your own inbox and goes to {agentName}. Separate several with commas.</p>
              </label>
              <label className="mt-3 block">
                <span className={label}>Names it answers to</span>
                <input className={field} value={names} placeholder={agentName} onChange={(e) => setNames(e.target.value)} />
                <p className={hint}>
                  An email with one of these in the subject, or that addresses it in the body (“Hi {answersTo[0] || agentName},”,
                  “{answersTo[0] || agentName}, please…”, “@{answersTo[0] || agentName}”), is for {agentName}. Empty: its name.
                </p>
              </label>
              <button
                type="button" role="switch" aria-checked={pickUp} onClick={() => setPickUp((v) => !v)}
                className="mt-4 flex w-full items-start gap-3 rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-3 text-left hover:border-zinc-700"
              >
                <span className="mt-0.5"><SwitchTrack on={pickUp} /></span>
                <span className="min-w-0">
                  <span className="block text-sm font-medium text-zinc-100">Pick up email meant for {agentName}</span>
                  <span className={hint}>
                    {watched
                      ? `It starts on what comes to its own inbox and its addresses, and on mail in ${watched} that names it. Each one is a task on your Taskboard.`
                      : 'Connect Gmail, Outlook or an IMAP inbox under Connections first.'}
                    {' '}Replies from its own inbox go out only if that inbox may send; replies in Gmail go as you choose below;
                    anywhere else they wait as drafts for you.
                  </span>
                </span>
              </button>
              {pickUp && view.mail.gmail && (
                <div className="mt-3">
                  <GmailReplyChoice value={gmailReplies} onChange={setGmailReplies} />
                </div>
              )}
            </section>

            <section>
              <h3 className="flex items-center gap-1.5 text-sm font-semibold text-zinc-100"><AtSign className="h-4 w-4 text-sky-400" />Colleagues it may hand work to</h3>
              <p className={hint}>
                A hand-over is a task of its own on your Taskboard. The colleague starts at once, never with more permission than {agentName} had,
                and its answer comes back to {agentName}.
              </p>
              {others.length === 0 ? (
                <p className="mt-2 text-sm text-zinc-500">You have no other agent yet.</p>
              ) : (
                <div className="mt-2 flex flex-col gap-1.5">
                  <label className="flex items-center gap-2 text-sm text-zinc-200">
                    <input type="radio" checked={handsAll} onChange={() => setHandsAll(true)} /> All of them
                  </label>
                  <label className="flex items-center gap-2 text-sm text-zinc-200">
                    <input type="radio" checked={!handsAll} onChange={() => setHandsAll(false)} /> Only these
                  </label>
                  {!handsAll && (
                    <div className="ml-6 mt-1 flex flex-col gap-1">
                      {others.map((a) => (
                        <label key={a.slug} className="flex items-center gap-2 text-sm text-zinc-300">
                          <input
                            type="checkbox" checked={handsTo.includes(a.slug)}
                            onChange={(e) => setHandsTo((cur) => e.target.checked ? [...cur, a.slug] : cur.filter((s) => s !== a.slug))}
                          />
                          <span>{a.name}</span>
                          {a.role && <span className="text-xs text-zinc-500">— {a.role}</span>}
                        </label>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </section>
          </div>
        )}
      </div>
      {view && (
        <div className="flex shrink-0 items-center justify-end gap-2 border-t border-zinc-800 px-5 py-3">
          <span className="mr-auto flex items-center gap-1.5 text-[11px] text-zinc-500">
            <Inbox className="h-3.5 w-3.5" />Email it gets shows on your Taskboard.
          </span>
          <button onClick={onClose} className="rounded-lg px-3 py-2 text-sm text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200">Close</button>
          <button
            onClick={save} disabled={saving}
            className="flex items-center gap-1.5 rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-60"
          >
            {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : saved ? <Check className="h-4 w-4" /> : null}
            {saved ? 'Saved' : 'Save'}
          </button>
        </div>
      )}
    </AgentDialog>
  )
}
