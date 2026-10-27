import { useEffect, useState } from 'react'
import { Gauge, Hourglass, KeyRound, RotateCcw } from 'lucide-react'
import type { QuotaAllowance, QuotaStatus } from '../../services/profile'
import { canSpendReset, quotaLimited, useQuotaStore } from '../../stores/quotaStore'
import { useByokStore } from '../../stores/byokStore'

// ── Daily plan allowances, as the person sees them ──
//
// PlanUsageBar sits at the top of the chat, under the lane tabs: in the simple
// (kiosk) layout it is always there while the plan has daily limits; in the
// full layout only once an allowance runs low. When one is spent the composer
// is closed and the bar says why and when it comes back. PlanUsageMeters is the Info dialog's
// detailed view. "Tokens" are "AI usage" here — a kiosk user needn't know
// what a token is. Someone an admin gave free resets can spend one from the
// bar (once spent or running low) or the Info dialog to start the day over.
// Where the deck allows it, the bar also offers "Use my own keys" (byok.py):
// on their own API keys a person has no daily limits.

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, '')}M`
  if (n >= 1_000) return `${Math.round(n / 1_000)}k`
  return String(n)
}

/** The reset moment in the person's own clock, e.g. "02:00". */
function resetClock(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

/** How long until *iso*, e.g. "5 h 12 min". */
function resetIn(iso: string): string {
  const ms = new Date(iso).getTime() - Date.now()
  if (!Number.isFinite(ms)) return ''
  const minutes = Math.max(1, Math.round(ms / 60000))
  const h = Math.floor(minutes / 60)
  const m = minutes % 60
  return h ? `${h} h ${m} min` : `${m} min`
}

function pct(a: QuotaAllowance): number {
  return a.limit > 0 ? Math.min(100, Math.round((a.used / a.limit) * 100)) : 0
}

/** Why the composer is closed, in a sentence. */
function quotaBlockedText(s: QuotaStatus): string {
  const what = s.blocked === 'tokens'
    ? "today's AI usage"
    : `today's ${s.messages.limit} message${s.messages.limit === 1 ? '' : 's'}`
  return `You've used ${what} on the ${s.plan_label} plan.`
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`

/** "Use a free reset", then "Reset now" to confirm: they're scarce. *tone*
 *  colours the buttons for the strip they sit on. */
function FreeReset({ s, tone, compact = false }: { s: QuotaStatus; tone: string; compact?: boolean }) {
  const spendReset = useQuotaStore((st) => st.spendReset)
  const [asking, setAsking] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  if (!canSpendReset(s)) return null
  const left = s.resets_left
  const uses = left === 1 ? 'This uses your last free reset.' : `This uses 1 of your ${left} free resets.`
  const btn =`inline-flex shrink-0 items-center gap-1 rounded-md border font-medium disabled:opacity-50 ${compact ? 'px-1.5 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs'}`

  const spend = async () => {
    setBusy(true); setErr('')
    try {
      await spendReset()
      setAsking(false)
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Couldn't use the reset")
    }
    setBusy(false)
  }

  if (!asking) {
    return (
      <span className="inline-flex flex-wrap items-center gap-2">
        <button
          onClick={() => { setAsking(true); setErr('') }}
          title={`Start today's messages and AI usage over. You have ${plural(left, 'free reset')}.`}
          className={`${btn} ${tone}`}
        >
          <RotateCcw className="h-3 w-3" />
          {compact ? `Free reset (${left})` : 'Use a free reset'}
        </button>
        {!compact && <span className="text-xs opacity-90">{left} left</span>}
        {err && <span className="text-xs">{err}</span>}
      </span>
    )
  }
  return (
    <span className="inline-flex flex-wrap items-center gap-2">
      {compact ? (
        <span className="text-[11px]" title={uses}>Start today over?</span>
      ) : (
        <span className="text-xs">Start today over? {uses}</span>
      )}
      <button
        onClick={spend}
        disabled={busy}
        className={`${btn} border-violet-600 bg-violet-600 text-white hover:bg-violet-500`}
      >
        {busy ? 'Resetting…' : 'Reset now'}
      </button>
      <button onClick={() => setAsking(false)} disabled={busy} className={`${btn} ${tone}`}>
        Cancel
      </button>
    </span>
  )
}

/** "Use my own keys" — where the deck allows them and a layout can open the dialog. */
function OwnKeysButton({ s, tone, compact = false }: { s: QuotaStatus; tone: string; compact?: boolean }) {
  const host = useByokStore((st) => st.host)
  const openDialog = useByokStore((st) => st.openDialog)
  if (!s.byok_allowed || !host) return null
  return (
    <button
      onClick={openDialog}
      title="Run your agents on your own API keys — no daily limits"
      className={`inline-flex shrink-0 items-center gap-1 rounded-md border font-medium ${compact ? 'px-1.5 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs'} ${tone}`}
    >
      <KeyRound className="h-3 w-3" />
      {compact ? 'Own keys' : 'Use my own keys'}
    </button>
  )
}

/** Keeps the allowances fresh while they matter: on mount, and every minute
 *  (and on focus) while one is spent — an admin may give it back. */
function useQuotaRefresh(): QuotaStatus | null {
  const status = useQuotaStore((s) => s.status)
  const refresh = useQuotaStore((s) => s.refresh)
  useEffect(() => { void refresh() }, [refresh])
  const blocked = !!status?.blocked
  useEffect(() => {
    if (!blocked) return
    const again = () => { void refresh() }
    window.addEventListener('focus', again)
    const t = setInterval(again, 60_000)
    return () => { window.removeEventListener('focus', again); clearInterval(t) }
  }, [blocked, refresh])
  return status
}

export function PlanUsageBar({ simple }: { simple: boolean }) {
  const status = useQuotaRefresh()
  if (status?.byok && simple) {
    return (
      <div className="flex items-center gap-2 border-b border-zinc-800/60 px-3 py-1.5 text-xs text-zinc-500"
        title="Your agents run on your own API keys: no daily limits, and your provider bills you directly.">
        <KeyRound className="h-3.5 w-3.5 shrink-0 text-emerald-500" />
        <span className="min-w-0 flex-1 truncate">On your own keys · no daily limits</span>
        <span className="shrink-0">{fmtTokens(status.byok_tokens ?? 0)} AI tokens today</span>
      </div>
    )
  }
  if (!quotaLimited(status)) return null
  const s = status

  if (s.blocked) {
    return (
      <div role="status" className="flex items-start gap-2.5 border-b border-red-500/30 bg-red-500/10 px-3 py-2.5 text-sm text-red-700 dark:text-red-300">
        <Hourglass className="mt-0.5 h-4 w-4 shrink-0" />
        <div className="min-w-0">
          <div className="font-medium">{quotaBlockedText(s)}</div>
          <div className="text-xs opacity-90">
            You can chat again at {resetClock(s.resets_at)} (in {resetIn(s.resets_at)}).
            {!canSpendReset(s)
              ? ' Ask your admin if you need a bigger plan.'
              : s.resets_left === 1
                ? ' Or start over now with your free reset.'
                : ` Or start over now with one of your ${s.resets_left} free resets.`}
          </div>
          {(canSpendReset(s) || s.byok_allowed) && (
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <FreeReset s={s} tone="border-red-500/40 hover:bg-red-500/10" />
              <OwnKeysButton s={s} tone="border-red-500/40 hover:bg-red-500/10" />
            </div>
          )}
        </div>
      </div>
    )
  }

  if (!simple && !s.warning) return null
  const parts: string[] = []
  if (s.messages.limit > 0) {
    const left = s.messages.remaining ?? 0
    parts.push(`${left} of ${s.messages.limit} message${s.messages.limit === 1 ? '' : 's'} left today`)
  }
  if (s.tokens.limit > 0) parts.push(`AI usage ${pct(s.tokens)}%`)
  const low = !!s.warning
  return (
    <div
      className={`flex items-center gap-2 border-b px-3 py-1.5 text-xs ${low
        ? 'border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-400'
        : 'border-zinc-800/60 text-zinc-500'}`}
      title={`${s.plan_label} plan · ${s.messages.limit > 0 ? `${s.messages.used} of ${s.messages.limit} messages` : 'unlimited messages'} · ${s.tokens.limit > 0 ? `${fmtTokens(s.tokens.used)} of ${fmtTokens(s.tokens.limit)} AI tokens` : 'unlimited AI usage'} today`}
    >
      <Gauge className="h-3.5 w-3.5 shrink-0" />
      <span className="min-w-0 flex-1 truncate">{parts.join(' · ')}</span>
      {low && <FreeReset s={s} compact tone="border-amber-500/40 hover:bg-amber-500/10" />}
      {low && <OwnKeysButton s={s} compact tone="border-amber-500/40 hover:bg-amber-500/10" />}
      <span className="shrink-0">Resets {resetClock(s.resets_at)}</span>
    </div>
  )
}

function Meter({ label, a, fmt }: { label: string; a: QuotaAllowance; fmt: (n: number) => string }) {
  const p = pct(a)
  const tone = p >= 100 ? 'bg-red-500' : p >= 80 ? 'bg-amber-500' : 'bg-violet-500'
  return (
    <div className="rounded-lg bg-zinc-900 px-3 py-2">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[11px] text-zinc-500">{label}</span>
        <span className="text-sm font-medium text-zinc-200">
          {a.limit > 0 ? `${fmt(a.used)} of ${fmt(a.limit)}` : `${fmt(a.used)} · unlimited`}
        </span>
      </div>
      {a.limit > 0 && (
        <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-zinc-800">
          <div className={`h-full rounded-full ${tone}`} style={{ width: `${p}%` }} />
        </div>
      )}
    </div>
  )
}

/** Today's allowances in the Info dialog. */
export function PlanUsageMeters({ initial }: { initial?: QuotaStatus }) {
  const live = useQuotaRefresh()
  const s = live ?? initial ?? null
  if (!s) return null
  if (s.byok) {
    return (
      <div className="mt-3">
        <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">Today</div>
        <div className="rounded-lg bg-zinc-900 px-3 py-2">
          <div className="flex items-baseline justify-between gap-2">
            <span className="flex items-center gap-1.5 text-[11px] text-zinc-500"><KeyRound className="h-3 w-3 text-emerald-500" /> On your own keys</span>
            <span className="text-sm font-medium text-zinc-200">{fmtTokens(s.byok_tokens ?? 0)} AI tokens · no limits</span>
          </div>
        </div>
        <p className="mt-1.5 text-[11px] text-zinc-500">
          Your agents run on your own API keys, so the {s.plan_label} plan's daily limits don't apply. Your provider bills you for what they use.
        </p>
      </div>
    )
  }
  return (
    <div className="mt-3">
      <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">Today</div>
      <div className="grid grid-cols-1 gap-2 wide:grid-cols-2">
        <Meter label="Messages" a={s.messages} fmt={String} />
        <Meter label="AI tokens" a={s.tokens} fmt={fmtTokens} />
      </div>
      <p className="mt-1.5 text-[11px] text-zinc-500">
        {s.exempt
          ? "Admins aren't held to the daily limits."
          : s.blocked
            ? `${quotaBlockedText(s)} You can chat again at ${resetClock(s.resets_at)} (in ${resetIn(s.resets_at)}).`
            : `Resets every day at ${resetClock(s.resets_at)} your time (midnight UTC). A reply already under way always finishes.`}
      </p>
      {!s.exempt && (s.resets_left ?? 0) > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-zinc-400">
          {canSpendReset(s)
            ? <FreeReset s={s} tone="border-zinc-700 text-zinc-300 hover:border-violet-500/40 hover:text-zinc-100" />
            : `You have ${plural(s.resets_left, 'free reset')} to start a day over once you've used some of it.`}
        </div>
      )}
    </div>
  )
}
