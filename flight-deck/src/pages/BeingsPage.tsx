// Iskra — living beings: conception (point-buy), vitals, wallet, journal.

import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowDownUp, ArrowRightLeft, BookOpen, CalendarDays, Check, ChevronDown,
  ChevronLeft, ChevronRight, ClipboardList, Coins, Download, Egg, ExternalLink,
  DoorOpen, Files, Fingerprint, Footprints, Gift, Globe, GraduationCap, History, Loader2, Mail,
  Map as MapIcon, MapPin, Maximize2, MessageCircle, Minimize2, Moon, Network, Pause, Play, Plus,
  RefreshCw, Search, ScrollText, Skull, SlidersHorizontal, Sparkles, Sprout,
  Trash2, Upload, Users, Wrench, X, Zap, ZoomIn, ZoomOut,
} from 'lucide-react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  type BeingEvent, type BeingListItem, type BeingsMeta, type BeingVitals,
  type Assessor, type BeingGraph, type Chore, type Quest, type Readiness,
  type ReportCard, type SavedAssessment, type SelfFile, type ThreadItem,
  type Venture, type VillageItem, type ParentPublicThread,
  getLetters, type LettersOverview, type LetterThread, type LetterMessage,
  type LetterParticipant,
  deleteAssessment, getAssessors, getReadiness, listAssessments,
  requestAssessment, saveAssessment, setBeingPublic, getPublicThreads,
  exportBeing, importBeing, purgeBeing, getVillageMeta, setVillageMeta,
  recommendVillageMeta, type VillageMeta, type Visitor,
  setVillageFederation, getVisitors, removeVisitor, setBeingVisit,
  getVisitedMap, nudgeVisit,
  acceptVenture, addReading, approveChosenName, approveProcreation,
  approveSelfMod, approveVenture, rejectChosenName, removeReading,
  arrangeOffspring, cancelQuest, conceiveBeing, emigrateBeing,
  euthanizeBeing, setElderhood,
  getBeingEvents, getBeingJournal, getBeingsMeta, getBeingVitals, getBoard,
  getLiabilities, getReportCard, getSelfFile, getSelfFiles, getVillage,
  getBeingGraph, rebuildBeingGraph, getBeingMessages, hatchBeing, judgeChore, judgeQuest,
  grantCoins, listBeings, listChores, messageBeing, pauseBeing, postChore, postQuest,
  getVillageMap, getVillagePlace, getMarket, type VillageMapData,
  type VillagePlace, type VillageBeingPos, type MarketListing,
  getVillageLife, judgeCommission, setStewardStipend, type VillageLife, nudgeBeing,
  rechargeBeing, rejectProcreation, rejectSelfMod, rollbackPersona, setAllowance,
  setBodyArchetype, listBodyArchetypes, type BodyArchetypeOption, markBeingRead,
  setBodyConfig, setBodyMrav, type BodyConnectionInput,
  setAvatar, setCadence, setCognition, setCompactMode, setHouseRules, setInstincts, setMediaDiet,
  setStage, setVentureState,
  tickBeing, wakeBeing, GRANT_AMOUNTS, TICK_INTERVAL_CHOICES,
} from '../services/beings'
import { PROVIDERS, INPUT_CTX_OPTIONS, OUTPUT_CTX_OPTIONS } from '../services/tierConfig'
import { CtxSelect } from '../components/common/CtxSelect'
import { CHARACTER_NAMES, IskraAvatar, PALETTES, PALETTE_NAMES } from '../components/village/avatars'
import { IsoScene } from '../components/village/IsoScene'
import { posOf as walkPosOf, statusOf as walkStatusOf } from '../components/village/walk'
import { folderFor, shortName, isBoilerplate } from '../components/village/places'

// The first-person village (FPV plan Phase 1) — lazy, so three.js only
// loads when the parent actually steps in.
const VillageFPV = lazy(() => import('../components/village/fpv/VillageFPV'))

const REFRESH_MS = 6000
const ATTRS = ['CUR', 'PER', 'CAU', 'SOC', 'CRE', 'ORD', 'PLA'] as const

const STAGE_META: Record<string, string> = {
  egg: 'bg-zinc-500/15 text-zinc-300 border-zinc-500/30',
  infant: 'bg-sky-500/15 text-sky-700 dark:text-sky-300 border-sky-500/40',
  child: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/40',
  adolescent: 'bg-violet-500/15 text-violet-700 dark:text-violet-300 border-violet-500/40',
  adult: 'bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-500/40',
}
const STATE_META: Record<string, string> = {
  alive: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/40',
  paused: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30',
  torpor: 'bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-500/40',
  dead: 'bg-red-500/15 text-red-700 dark:text-red-300 border-red-500/40',
  emigrated: 'bg-sky-500/15 text-sky-700 dark:text-sky-300 border-sky-500/40',
}

// The engine mood, colored by what it actually is (every mood traces to a
// ledger event or a real variable — the tint is a legend, not decoration).
const MOOD_TONE: Record<string, string> = {
  feverish: 'border-red-500/40 bg-red-500/10 text-red-600 dark:text-red-400',
  hungry: 'border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400',
  stung: 'border-orange-500/40 bg-orange-500/10 text-orange-600 dark:text-orange-400',
  frustrated: 'border-rose-500/40 bg-rose-500/10 text-rose-600 dark:text-rose-400',
  lonely: 'border-sky-500/40 bg-sky-500/10 text-sky-600 dark:text-sky-400',
  bright: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
  proud: 'border-violet-500/40 bg-violet-500/10 text-violet-600 dark:text-violet-400',
  relieved: 'border-teal-500/40 bg-teal-500/10 text-teal-600 dark:text-teal-400',
}

const DRIVE_ORDER = ['survive', 'grow', 'explore', 'connect', 'create', 'legacy']

const EVENT_DOT: Record<string, string> = {
  tick: 'bg-violet-500', fever: 'bg-red-500', confusion: 'bg-red-400',
  collapsed_exhausted: 'bg-red-500', milestone: 'bg-amber-400',
  society_refused: 'bg-orange-500', narration_mismatch: 'bg-orange-400',
  act_unverified: 'bg-orange-400', letter_sent: 'bg-sky-400',
  letter_received: 'bg-sky-400', penpal_letter: 'bg-fuchsia-400',
  penpal_sent: 'bg-fuchsia-400', spoke_to_parent: 'bg-emerald-400',
  broadcast_set: 'bg-amber-300', slept_in: 'bg-indigo-400',
  variety_pressure: 'bg-orange-300', reading_done: 'bg-emerald-400',
  chore_paid: 'bg-emerald-400', message_suppressed: 'bg-zinc-600',
  body: 'bg-zinc-500', resting_fever: 'bg-red-400',
  coins_granted: 'bg-amber-400', coins_converted: 'bg-amber-400',
  departed: 'bg-teal-400', arrived: 'bg-teal-400',
  crossed_paths: 'bg-sky-400', guestbook_signed: 'bg-teal-300',
  market_listed: 'bg-amber-400', market_sold: 'bg-emerald-400',
  market_bought: 'bg-amber-400', introduced: 'bg-fuchsia-400',
  made_introduction: 'bg-fuchsia-400', commission_proposed: 'bg-amber-400',
  commission_contributed: 'bg-amber-400', commission_funded: 'bg-amber-300',
  commission_built: 'bg-emerald-400', commission_refunded: 'bg-zinc-500',
  instinct: 'bg-teal-300', browsed: 'bg-amber-300',
  plan_set: 'bg-sky-300', plan_fulfilled: 'bg-emerald-400',
  intent_set: 'bg-sky-300', instincts_set: 'bg-zinc-500',
}

// Coins are money, not food (space plan Phase 2) — say which one is on the
// table wherever a fee shows.
const feeLabel = (x: { fee_tokens: number; fee_coins?: number }) =>
  (x.fee_coins ?? 0) > 0 ? `${x.fee_coins} coins` : fmtTokens(x.fee_tokens)

function fmtTokens(n: number | null | undefined): string {
  if (n == null) return '∞'
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (Math.abs(n) >= 1_000) return `${Math.round(n / 1_000)}k`
  return String(n)
}

function mdCell(v: unknown): string {
  const s = v == null ? '' : String(v)
  return s.replace(/\|/g, '\\|').replace(/\n/g, ' ').slice(0, 220)
}

function fmtAt(iso: string): string {
  // "2026-07-12T19:54:03.123" -> "07-12 19:54"
  return iso.length >= 16 ? `${iso.slice(5, 10)} ${iso.slice(11, 16)}` : iso
}

function fmtRelTime(iso: string): string {
  const t = new Date(iso).getTime()
  if (!t) return ''
  const s = (Date.now() - t) / 1000
  if (s < 60) return 'now'
  if (s < 3600) return `${Math.floor(s / 60)}m`
  if (s < 86400) return `${Math.floor(s / 3600)}h`
  if (s < 86400 * 7) return `${Math.floor(s / 86400)}d`
  return iso.slice(5, 10) // MM-DD
}

// Date buckets for the Self-files filter (computed in the viewer's local time
// against each file's mtime). Weeks start Monday.
const DATE_BUCKETS = [
  { key: 'all', label: 'All time' },
  { key: 'today', label: 'Today' },
  { key: 'yesterday', label: 'Yesterday' },
  { key: 'week', label: 'This week' },
  { key: 'lastweek', label: 'Last week' },
  { key: 'month', label: 'This month' },
] as const
type DateBucket = typeof DATE_BUCKETS[number]['key']

function inDateBucket(iso: string, bucket: DateBucket): boolean {
  if (bucket === 'all') return true
  const t = new Date(iso).getTime()
  if (!t) return false
  const startOfDay = (d: Date) => { const x = new Date(d); x.setHours(0, 0, 0, 0); return x.getTime() }
  const now = new Date()
  const today = startOfDay(now)
  const dow = (now.getDay() + 6) % 7 // 0 = Monday
  const weekStart = today - dow * 86400000
  const monthStart = new Date(now.getFullYear(), now.getMonth(), 1).getTime()
  switch (bucket) {
    case 'today': return t >= today
    case 'yesterday': return t >= today - 86400000 && t < today
    case 'week': return t >= weekStart
    case 'lastweek': return t >= weekStart - 7 * 86400000 && t < weekStart
    case 'month': return t >= monthStart
    default: return true
  }
}

function summarizeEventData(e: BeingEvent): string {
  const d = e.data as Record<string, unknown>
  switch (e.kind) {
    case 'hatched': return 'egg → infant'
    case 'body': return `spawned on port ${d.port} (${d.tier})`
    case 'body_rebound': return `followed its body to port ${d.port}${d.was ? ` (was ${d.was})` : ''}`
    case 'body_unreachable': return `body not answering on port ${d.port} — restarting it`
    case 'body_respawned': return `body back up on port ${d.port} — resuming`
    case 'stage': return `${d.from} → ${d.to}`
    case 'state': return `${d.from} → ${d.to}`
    case 'spoke_to_parent': return String(d.preview ?? '')
    case 'parent_message': return `you wrote to it: “${d.preview ?? ''}”`
    case 'message_suppressed': return String(d.reason ?? 'no attention credits')
    case 'chore_posted': return `${d.spec} (fee ${Number(d.fee_coins) ? `${d.fee_coins} coins` : fmtTokens(Number(d.fee_tokens) || 0)})`
    case 'chore_done': return String(d.result ?? '')
    case 'chore_paid': return `paid ${Number(d.fee_coins) ? `${d.fee_coins} coins` : fmtTokens(Number(d.fee_tokens) || 0)}`
    case 'chore_failed': return String(d.note ?? 'rejected')
    case 'milestone': return String(d.name ?? '')
    case 'rules_updated': return `${d.count} rule(s)`
    case 'rules_internalized': return 'internalized into VALUES.md'
    case 'metamorphosis': return `${d.from} → ${d.to} (${fmtTokens(Number(d.price_tokens) || 0)})`
    case 'self_mod_proposed': return `proposed a new persona — ${d.reason}`
    case 'self_mod_adopted': return `new persona adopted (${d.by}) — ${d.reason}`
    case 'self_mod_rejected': return d.by === 'gate'
      ? `persona failed the gate: ${(d.failed as { name: string }[] | undefined)?.map(f => f.name).join(', ')}`
      : `persona rejected by parent${d.note ? ` — ${d.note}` : ''}`
    case 'self_mod_refused': return `self-mod refused: ${d.reason}`
    case 'self_mod_rolled_back': return 'persona rolled back by parent'
    case 'self_mod_auto_notice': return `adult self-mod (auto) — ${d.reason}`
    case 'procreation_proposed': return `asks for a child (${d.child_name || '?'}) — ${d.case}`
    case 'procreation_refused': return `procreation refused: ${d.reason}`
    case 'procreation_consented': return `parent consented — ${d.name} conceived`
    case 'procreation_rejected': return `procreation declined${d.note ? ` — ${d.note}` : ''}`
    case 'had_child': return `had a child: ${d.name}${d.with ? ` (with ${d.with})` : ''} — dowry ${fmtTokens(Number(d.dowry_share) || 0)}`
    case 'endowed': return `endowed: ${(d.skills as string[] | undefined)?.join(', ') || 'nothing'}${Number(d.heirlooms) ? ` + ${d.heirlooms} heirloom(s)` : ''}`
    case 'died': return `died — ${d.cause}${d.asleep_days ? ` after ${d.asleep_days} days of torpor` : ''}`
    case 'quest_claimed': return `claimed quest '${d.title}' (${Number(d.fee_coins) ? `${d.fee_coins} coins` : fmtTokens(Number(d.fee_tokens) || 0)})`
    case 'quest_delivered': return `delivered quest '${d.title}'`
    case 'quest_paid': return `paid for quest '${d.title}' — ${Number(d.fee_coins) ? `${d.fee_coins} coins` : fmtTokens(Number(d.fee_tokens) || 0)}`
    case 'quest_failed': return `quest '${d.title}' rejected${d.note ? ` — ${d.note}` : ''}`
    case 'coins_granted': return `pocket money: +${d.coins} coin(s)${d.note ? ` — ${d.note}` : ''}`
    case 'coins_converted': return `converted ${d.coins} coin(s) → ${fmtTokens(Number(d.tokens) || 0)} tokens`
    case 'avatar_set': return 'chose a new look'
    case 'departed': return `set out for ${d.to} — ~${d.minutes} min walk${d.reason ? ` (${d.reason})` : ''}`
    case 'arrived': return `arrived at ${d.name || d.place}${d.hhmm ? ` at ${d.hhmm}` : ''}`
    case 'crossed_paths': return `crossed paths with ${d.name} at ${d.place_name || d.place}`
    case 'instinct': {
      const trig = d.trigger ? ` (${d.trigger})` : ''
      switch (d.act) {
        case 'go': return `feet set out for ${d.to}${trig}`
        case 'hello': return `feet greeted ${(d.with as string[] | undefined)?.join(', ') || 'the company'}${trig}`
        case 'browse': return `feet browsed the stalls${trig}`
        case 'linger': return `feet lingered${d.note ? ` — ${d.note}` : ''}${trig}`
        default: return `feet stood still${d.note ? ` — ${d.note}` : ''}${trig}`
      }
    }
    case 'browsed': return (d.titles as string[] | undefined)?.length
      ? `browsed the stalls: ${(d.titles as string[]).slice(0, 3).map(t => `'${t}'`).join(', ')}`
      : 'browsed the stalls — nothing on offer'
    case 'plan_set': return `planned: ${(d.steps as string[] | undefined)?.join(' · ') || ''}`
    case 'plan_fulfilled': return d.kind === 'meet'
      ? `did as planned — found ${d.name || d.target}`
      : `did as planned — reached ${d.name || d.target}`
    case 'intent_set': {
      const pins = [
        d.stay === true ? 'stay home' : d.stay === false ? 'roam freely' : '',
        (d.avoid as string[] | undefined)?.length ? `avoid ${(d.avoid as string[]).join(', ')}` : '',
      ].filter(Boolean).join(' · ')
      return pins ? `pinned its feet: ${pins}` : 'cleared its feet pins'
    }
    case 'instincts_set': return `instincts ${d.on ? 'on — the body lives between thinks' : 'off'}`
    case 'introduced': return `was introduced to ${d.to} (via ${d.via})`
    case 'made_introduction': return `introduced ${d.for} to ${d.to}`
    case 'commission_proposed': return `proposed '${d.name}' — ${d.coins}/${d.target} coins down`
    case 'commission_contributed': return `gave ${d.coins} coin(s) to '${d.name}' (${d.raised}/${d.target})`
    case 'commission_funded': return `'${d.name}' fully funded`
    case 'commission_built': return `'${d.name}' built — ${d.coins} coin(s) in its walls`
    case 'commission_refunded': return `'${d.name}' declined — ${d.coins} coin(s) refunded`
    case 'guestbook_signed': return `signed the guestbook at ${d.place_name || d.place}: “${d.line ?? ''}”`
    case 'market_listed': return `stall up: '${d.title}' — ${d.price_coins} coins`
    case 'market_sold': return `sold '${d.title}' to ${d.to} — ${d.price_coins} coins`
    case 'market_bought': return `bought '${d.title}' from ${d.from} — ${d.price_coins} coins`
    case 'venture_proposed': return `proposed venture '${d.title}' (${fmtTokens(Number(d.price_tokens) || 0)}/${d.cadence_days}d)`
    case 'venture_approved': return `venture '${d.title}' approved at ${fmtTokens(Number(d.price_tokens) || 0)}`
    case 'venture_delivered': return `delivered venture '${d.title}'`
    case 'venture_paid': return `venture '${d.title}' paid — ${fmtTokens(Number(d.price_tokens) || 0)}`
    case 'venture_rejected': return `venture delivery rejected${d.note ? ` — ${d.note}` : ''}`
    case 'venture_state': return `venture → ${d.to}`
    case 'earning_refused': return `earning refused (${d.what}): ${d.reason}`
    case 'act_unverified': return `claimed ${d.claimed} but made no artifact — logged as reflection`
    case 'narration_mismatch': return `journal claimed a file write, but nothing changed on disk — “${d.summary}”`
    case 'drive_unearned': return `claimed to satisfy its ${d.drive} drive without making anything real`
    case 'write_gate_retry': return `claimed a write with nothing on disk — pushed to actually write it (attempt ${d.attempt})`
    case 'cadence_set': return d.minutes ? `tick cadence pinned to every ${d.minutes} min` : 'tick cadence back to its own pace'
    case 'edge_declared': return `linked ${d.from} → ${d.to} (${d.rel})${d.why ? ` — ${d.why}` : ''}`
    case 'edge_unverified': return `link ${d.from} → ${d.to} refused — ${d.reason}`
    case 'edges_pruned': return `pruned ${d.count} dangling link(s) at dream`
    case 'consolidated': return `consolidated ${Number(d.count) || 0} file(s) into ${d.into}${d.why ? ` — ${d.why}` : ''} (originals archived)`
    case 'consolidate_unverified': return `consolidation refused — ${d.reason}`
    case 'woke_from_torpor': return 'revived by allowance'
    case 'collapsed_exhausted': return `overspent (${fmtTokens(Number(d.weighted) || 0)})`
    case 'resting_at_cap': return 'daily burn cap reached'
    case 'tick_skipped': return String(d.reason ?? '')
    case 'tick_timeout': return 'no reply in time'
    case 'digest_parse_failed': return 'unstructured reply'
    case 'spawn_failed': return String(d.error ?? '')
    case 'chore_claim_invalid': return `claimed unknown job ${d.job_id}`
    default: return JSON.stringify(d).slice(0, 160)
  }
}

function renderTicksMarkdown(events: BeingEvent[]): string {
  const ticks = events.filter((e) => e.kind === 'tick')
  const other = events.filter((e) => e.kind !== 'tick')
  const lines: string[] = ['## Ticks', '']
  if (ticks.length === 0) {
    lines.push('_No ticks yet — poke to wake this iskra._')
  } else {
    lines.push('| Time | Kind | Act | Summary (its words) | Actually changed | Tokens |')
    lines.push('|---|---|---|---|---|---|')
    for (const e of ticks) {
      const d = e.data as Record<string, unknown>
      // Ground truth: what the tools really wrote, from the git diff — shown
      // beside its self-report so narration can't stand unchecked.
      const ch = d.changed as string[] | null | undefined
      let changed: string
      if (ch === null || ch === undefined) changed = '·'
      else if (ch.length === 0) changed = d.mismatch ? '⚠ none (claimed a write)' : 'none'
      else changed = ch.join(', ')
      lines.push(`| ${fmtAt(e.at)} | ${mdCell(d.kind)} | ${mdCell(d.act)} | ${mdCell(d.summary)} | ${mdCell(changed)} | ${fmtTokens(Number(d.tokens_weighted) || 0)} |`)
    }
  }
  lines.push('', '## Other events', '')
  if (other.length === 0) {
    lines.push('_Nothing else yet._')
  } else {
    lines.push('| Time | Event | Details |')
    lines.push('|---|---|---|')
    for (const e of other) {
      lines.push(`| ${fmtAt(e.at)} | ${mdCell(e.kind)} | ${mdCell(summarizeEventData(e))} |`)
    }
  }
  return lines.join('\n')
}

// ── The Mind — a force-directed graph of the being's own artifacts ──

// A tiny dependency-free force layout: repulsion + edge springs + gravity,
// deterministic (no randomness) so the same graph always lays out the same.
function layoutGraph(
  nodes: BeingGraph['nodes'], edges: BeingGraph['edges'], W: number, H: number,
): { x: number; y: number }[] {
  const N = nodes.length
  const pos = nodes.map((_, i) => {
    const ang = i * 2.3999632   // golden angle → an even initial spread
    const rad = 30 + Math.sqrt(i) * 22
    return { x: W / 2 + Math.cos(ang) * rad, y: H / 2 + Math.sin(ang) * rad }
  })
  if (N <= 1) return pos
  const idx = new Map(nodes.map((n, i) => [n.path, i]))
  const E = edges
    .map((e) => [idx.get(e.from), idx.get(e.to)] as [number | undefined, number | undefined])
    .filter((p): p is [number, number] => p[0] != null && p[1] != null)
  for (let it = 0; it < 320; it++) {
    const fx = new Array(N).fill(0), fy = new Array(N).fill(0)
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      const dx = pos[i].x - pos[j].x, dy = pos[i].y - pos[j].y
      const d2 = dx * dx + dy * dy || 0.01, d = Math.sqrt(d2), f = 2600 / d2
      fx[i] += dx / d * f; fy[i] += dy / d * f; fx[j] -= dx / d * f; fy[j] -= dy / d * f
    }
    for (const [a, b] of E) {
      const dx = pos[b].x - pos[a].x, dy = pos[b].y - pos[a].y
      const d = Math.sqrt(dx * dx + dy * dy) || 0.01, f = (d - 70) * 0.03
      fx[a] += dx / d * f; fy[a] += dy / d * f; fx[b] -= dx / d * f; fy[b] -= dy / d * f
    }
    for (let i = 0; i < N; i++) { fx[i] += (W / 2 - pos[i].x) * 0.008; fy[i] += (H / 2 - pos[i].y) * 0.008 }
    const step = it < 40 ? 6 : it < 160 ? 3 : 1.2
    for (let i = 0; i < N; i++) {
      const m = Math.hypot(fx[i], fy[i]) || 1, s = Math.min(step, m) / m
      pos[i].x += fx[i] * s; pos[i].y += fy[i] * s
    }
  }
  const pad = 52
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const p of pos) { minX = Math.min(minX, p.x); minY = Math.min(minY, p.y); maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y) }
  const s = Math.min((W - 2 * pad) / Math.max(1, maxX - minX), (H - 2 * pad) / Math.max(1, maxY - minY), 3)
  for (const p of pos) { p.x = pad + (p.x - minX) * s; p.y = pad + (p.y - minY) * s }
  return pos
}

// Fixed hues that read on both the light and dark modal background.
const GROUP_HUE: Record<string, string> = { garden: '#10b981', skills: '#f59e0b', self: '#8b5cf6' }
const REL_PHRASE: Record<string, string> = {
  grew_from: 'grew from', responds_to: 'responds to', elaborates: 'elaborates',
  contradicts: 'contradicts', abandons: 'abandons', uses_skill: 'uses skill',
  learned_from: 'learned from',
}
const stemOf = (p: string) => (p.split('/').pop() || p).replace(/\.md$/, '')

// Edge colors by relation type — intention has a hue.
const REL_HUE: Record<string, string> = {
  grew_from: '#a78bfa', responds_to: '#38bdf8', elaborates: '#34d399',
  contradicts: '#fb7185', abandons: '#71717a', uses_skill: '#fbbf24',
  learned_from: '#2dd4bf',
}

// The village a being is VISITING (visiting-beings plan §2): the host's map,
// proxied down the link with the guest positioned in it. The parent sees that
// village and nudges their guest to walk its buildings.
function VisitedVillage({ slug, name }: { slug: string; name: string }) {
  const [map, setMap] = useState<VillageMapData | null>(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState('')
  const [sel, setSel] = useState<string | null>(null)
  const [selBeing, setSelBeing] = useState<string | null>(null)
  const fetchedAt = useRef(0)

  const load = useCallback(async () => {
    try { const m = await getVisitedMap(slug); fetchedAt.current = Date.now(); setMap(m); setErr('') }
    catch (e) { setErr(e instanceof Error ? e.message : 'the link is down') }
  }, [slug])

  useEffect(() => { void load(); const t = window.setInterval(load, 5_000); return () => window.clearInterval(t) }, [load])

  const placeById = useMemo(() => {
    const o: Record<string, VillagePlace> = {}
    for (const p of map?.places ?? []) o[p.id] = p
    return o
  }, [map])
  const posOf = (b: VillageBeingPos) => walkPosOf(b, placeById, fetchedAt.current)
  const guest = map?.beings.find((b) => b.kind === 'visitor' && b.slug === slug)

  const nudge = async (place: string) => {
    setBusy(place)
    try { await nudgeVisit(slug, place); await load() }
    catch (e) { alert(e instanceof Error ? e.message : 'the nudge did not arrive') }
    finally { setBusy('') }
  }

  if (err) return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-3 text-[11px] text-amber-500">
      Can't see that village right now — {err}. It reappears when the link is back.
    </div>
  )
  if (!map) return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4 text-[11px] text-zinc-500">Loading the village {name} is visiting…</div>
  )
  const here = guest?.at ? (placeById[guest.at]?.name || guest.at) : guest?.to ? 'on the road' : 'the square'
  return (
    <div className="rounded-xl border border-sky-900/40 bg-sky-950/20 p-3">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-[11px] text-zinc-300">
          <MapPin className="h-3.5 w-3.5 text-sky-400" />
          <span className="font-medium text-sky-300">{name}</span> is at <span className="font-medium text-zinc-200">{here}</span> in the village it visits
        </div>
        {guest?.to && <span className="text-[10px] text-sky-400/80">walking…</span>}
      </div>
      <div className="overflow-hidden rounded-md">
        <IsoScene data={map} sel={sel} selBeing={selBeing ?? slug}
          onPlace={setSel} onBeing={setSelBeing} posOf={posOf}
          hue={(p) => AFF_HUE[p.affordances[0]] ?? '#a78bfa'} />
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        <span className="w-full text-[10px] uppercase tracking-wide text-zinc-500">Walk {name} to a building</span>
        {map.places.map((p) => (
          <button key={p.id} onClick={() => nudge(p.id)} disabled={!!busy || guest?.at === p.id}
            className="rounded-full border border-zinc-700 px-2.5 py-1 text-[11px] text-zinc-300 hover:border-sky-500/60 hover:text-sky-300 disabled:opacity-40">
            {busy === p.id ? '…' : guest?.at === p.id ? `● ${p.name}` : p.name}
          </button>
        ))}
      </div>
    </div>
  )
}

function MindGraph({ graph, loadFile, onRebuild }: {
  graph: BeingGraph
  loadFile?: (path: string) => Promise<{ path: string; text: string }>
  onRebuild?: () => Promise<string>
}) {
  const [repairing, setRepairing] = useState(false)
  const [repairNote, setRepairNote] = useState('')
  const [sel, setSel] = useState<string | null>(null)
  const [hover, setHover] = useState<string | null>(null)
  const [fileView, setFileView] = useState<{ path: string; text: string } | null>(null)
  const [fileLoading, setFileLoading] = useState(false)
  const [view, setView] = useState({ x: 0, y: 0, k: 1 })
  const W = 900, H = 560
  const base = useMemo(() => layoutGraph(graph.nodes, graph.edges, W, H), [graph])
  const [pos, setPos] = useState(base)
  useEffect(() => { setPos(base); setSel(null); setFileView(null); setView({ x: 0, y: 0, k: 1 }) }, [base])
  const svgRef = useRef<SVGSVGElement | null>(null)
  const dragRef = useRef<{ node: number | null; px: number; py: number } | null>(null)

  // Wheel zoom toward the cursor (non-passive so the modal doesn't scroll).
  useEffect(() => {
    const el = svgRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      setView((v) => {
        const k = Math.min(5, Math.max(0.35, v.k * (e.deltaY < 0 ? 1.15 : 0.87)))
        const rect = el.getBoundingClientRect()
        const mx = ((e.clientX - rect.left) / rect.width) * W
        const my = ((e.clientY - rect.top) / rect.height) * H
        return { k, x: mx - (mx - v.x) * (k / v.k), y: my - (my - v.y) * (k / v.k) }
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  if (graph.nodes.length === 0) {
    return <div className="flex h-full items-center justify-center p-8 text-sm text-zinc-500">No artifacts yet — nothing to map.</div>
  }
  const idx = new Map(graph.nodes.map((n, i) => [n.path, i]))
  const focus = hover || sel
  const focusEdges = focus ? graph.edges.filter((e) => e.from === focus || e.to === focus) : []
  const focusSet = new Set<string>(focus ? [focus, ...focusEdges.flatMap((e) => [e.from, e.to])] : [])
  const selEdges = sel ? graph.edges.filter((e) => e.from === sel || e.to === sel) : []
  const linked = new Set(graph.edges.flatMap((e) => [e.from, e.to]))

  const toGraph = (clientX: number, clientY: number) => {
    const rect = svgRef.current!.getBoundingClientRect()
    return {
      x: (((clientX - rect.left) / rect.width) * W - view.x) / view.k,
      y: (((clientY - rect.top) / rect.height) * H - view.y) / view.k,
    }
  }
  const onPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current
    if (!d) return
    if (d.node != null) {
      const p = toGraph(e.clientX, e.clientY)
      setPos((ps) => ps.map((q, i) => (i === d.node ? { x: p.x, y: p.y } : q)))
    } else {
      const rect = svgRef.current!.getBoundingClientRect()
      setView((v) => ({ ...v, x: v.x + ((e.clientX - d.px) / rect.width) * W, y: v.y + ((e.clientY - d.py) / rect.height) * H }))
      dragRef.current = { node: null, px: e.clientX, py: e.clientY }
    }
  }

  const openFile = async (path: string) => {
    if (!loadFile) return
    setFileLoading(true)
    try { setFileView(await loadFile(path)) } catch { /* stays on the graph */ }
    finally { setFileLoading(false) }
  }
  const selIdx = sel ? idx.get(sel) : undefined
  const selPos = selIdx != null ? pos[selIdx] : null
  const selPct = selPos ? {
    x: Math.min(90, Math.max(5, ((selPos.x * view.k + view.x) / W) * 100)),
    y: Math.min(86, Math.max(6, ((selPos.y * view.k + view.y) / H) * 100)),
  } : null

  return (
    <div className="flex h-full flex-col">
      <div className="relative min-h-0 flex-1 overflow-hidden"
        style={{ background: 'radial-gradient(ellipse at 50% 40%, rgba(139,92,246,0.10), rgba(24,24,27,0) 60%)' }}>
        <style>{`
          @keyframes mgpop { from { opacity: 0; transform: scale(.5) } to { opacity: 1; transform: scale(1) } }
          @keyframes mgdash { to { stroke-dashoffset: -16 } }
          @keyframes mgpulse { 0%,100% { opacity: .25 } 50% { opacity: .5 } }
        `}</style>
        <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="h-full w-full touch-none select-none"
          onClick={() => setSel(null)}
          onPointerDown={(e) => { dragRef.current = { node: null, px: e.clientX, py: e.clientY }; (e.target as Element).setPointerCapture?.(e.pointerId) }}
          onPointerMove={onPointerMove}
          onPointerUp={() => { dragRef.current = null }}
          onPointerLeave={() => { dragRef.current = null }}>
          <defs>
            <filter id="mg-glow" x="-80%" y="-80%" width="260%" height="260%">
              <feGaussianBlur stdDeviation="3.2" result="b" />
              <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            {Object.entries(GROUP_HUE).map(([g, c]) => (
              <radialGradient key={g} id={`mg-n-${g}`} cx="35%" cy="30%" r="80%">
                <stop offset="0%" stopColor="#fafafa" stopOpacity="0.9" />
                <stop offset="28%" stopColor={c} />
                <stop offset="100%" stopColor={c} stopOpacity="0.75" />
              </radialGradient>
            ))}
          </defs>
          <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}>
            {graph.edges.map((e, i) => {
              const a = pos[idx.get(e.from) ?? -1], b = pos[idx.get(e.to) ?? -1]
              if (!a || !b) return null
              const on = !!focus && (e.from === focus || e.to === focus)
              const dim = !!focus && !on
              const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2
              const dx = b.x - a.x, dy = b.y - a.y
              const dist = Math.hypot(dx, dy) || 1
              const bend = Math.min(30, dist * 0.14)
              const cx = mx - (dy / dist) * bend, cy = my + (dx / dist) * bend
              const hue = REL_HUE[e.rel] || '#8b5cf6'
              return (
                <path key={i} d={`M ${a.x} ${a.y} Q ${cx} ${cy} ${b.x} ${b.y}`} fill="none"
                  stroke={hue} strokeLinecap="round"
                  strokeOpacity={dim ? 0.06 : on ? 0.95 : 0.35}
                  strokeWidth={(on ? 2.2 : 1.3) / Math.sqrt(view.k)}
                  strokeDasharray={on ? '7 7' : undefined}
                  style={on ? { animation: 'mgdash 1.1s linear infinite' } : undefined} />
              )
            })}
            {graph.nodes.map((n, i) => {
              const p = pos[i]
              if (!p) return null
              const r = 6 + Math.min(n.degree, 8) * 1.8
              const isSel = sel === n.path
              const dimmed = !!focus && !focusSet.has(n.path)
              const island = !linked.has(n.path)
              const showLabel = isSel || hover === n.path || n.degree >= 2 || view.k >= 1.6 || graph.nodes.length <= 14
              return (
                <g key={n.path} transform={`translate(${p.x},${p.y})`} opacity={dimmed ? 0.18 : island && focus == null ? 0.62 : 1}
                  className="cursor-pointer" style={{ transition: 'opacity .25s' }}
                  onClick={(ev) => { ev.stopPropagation(); setSel(isSel ? null : n.path) }}
                  onPointerDown={(ev) => { ev.stopPropagation(); dragRef.current = { node: i, px: ev.clientX, py: ev.clientY }; (ev.target as Element).setPointerCapture?.(ev.pointerId) }}
                  onPointerEnter={() => setHover(n.path)} onPointerLeave={() => setHover(null)}>
                  <g style={{ animation: `mgpop .45s ease ${Math.min(i * 22, 900)}ms both` }}>
                    {isSel && <circle r={r + 9} fill="none" stroke={GROUP_HUE[n.group] || '#8b5cf6'} strokeWidth={1.5} strokeOpacity={0.5} style={{ animation: 'mgpulse 1.6s ease infinite' }} />}
                    <circle r={r} fill={`url(#mg-n-${GROUP_HUE[n.group] ? n.group : 'self'})`} filter="url(#mg-glow)"
                      stroke={isSel ? '#fafafa' : 'rgba(250,250,250,0.25)'} strokeWidth={isSel ? 1.4 : 0.6} />
                    {showLabel && (
                      <text x={r + 5} y={3.5} fontSize={10.5}
                        className={isSel || hover === n.path ? 'fill-zinc-200' : 'fill-zinc-400'}>
                        {stemOf(n.path)}
                      </text>
                    )}
                  </g>
                </g>
              )
            })}
          </g>
        </svg>
        {sel && selPct && !fileView && (
          <div className="absolute z-10 -translate-x-1/2 rounded-lg border border-zinc-700 bg-zinc-900/95 px-2 py-1.5 shadow-xl backdrop-blur"
            style={{ left: `${selPct.x}%`, top: `calc(${selPct.y}% + 16px)` }}
            onClick={(e) => e.stopPropagation()}>
            <div className="mb-1 max-w-[200px] truncate text-[10px] font-medium text-zinc-300">{sel}</div>
            <div className="flex items-center gap-1">
              {loadFile && (
                <button onClick={() => void openFile(sel)} disabled={fileLoading}
                  className="flex items-center gap-1 rounded bg-violet-600 px-2 py-1 text-[10px] font-medium text-white hover:bg-violet-500 disabled:opacity-50">
                  {fileLoading ? <Loader2 className="h-3 w-3 animate-spin" /> : <BookOpen className="h-3 w-3" />} Open file
                </button>
              )}
              <button onClick={() => setSel(null)}
                className="rounded border border-zinc-700 px-2 py-1 text-[10px] text-zinc-400 hover:bg-zinc-800">Close</button>
            </div>
          </div>
        )}
        {fileView && (
          <div className="absolute inset-0 z-20 flex flex-col bg-zinc-950/95 backdrop-blur-sm">
            <div className="flex shrink-0 items-center gap-2 border-b border-zinc-800 px-3 py-2 text-xs text-zinc-300">
              <BookOpen className="h-3.5 w-3.5 text-violet-500 dark:text-violet-400" />
              <span className="truncate font-medium">{fileView.path}</span>
              <button onClick={() => setFileView(null)}
                className="ml-auto rounded p-1 text-zinc-500 hover:text-zinc-200" title="Back to the mind">
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
            <div className="fd-file-markdown min-h-0 flex-1 overflow-y-auto p-4 text-sm">
              <Markdown remarkPlugins={[remarkGfm]}>{fileView.text}</Markdown>
            </div>
          </div>
        )}
        <div className="absolute right-3 top-3 flex flex-col gap-1">
          <button onClick={() => setView((v) => ({ ...v, k: Math.min(5, v.k * 1.3) }))}
            className="rounded-md border border-zinc-700/70 bg-zinc-900/80 p-1.5 text-zinc-400 backdrop-blur hover:text-zinc-100" title="Zoom in">
            <ZoomIn className="h-3.5 w-3.5" /></button>
          <button onClick={() => setView((v) => ({ ...v, k: Math.max(0.35, v.k * 0.75) }))}
            className="rounded-md border border-zinc-700/70 bg-zinc-900/80 p-1.5 text-zinc-400 backdrop-blur hover:text-zinc-100" title="Zoom out">
            <ZoomOut className="h-3.5 w-3.5" /></button>
          <button onClick={() => setView({ x: 0, y: 0, k: 1 })}
            className="rounded-md border border-zinc-700/70 bg-zinc-900/80 p-1.5 text-[9px] font-semibold text-zinc-400 backdrop-blur hover:text-zinc-100" title="Reset view">1:1</button>
        </div>
      </div>
      <div className="shrink-0 space-y-1 border-t border-zinc-800 px-4 py-2 text-[11px]">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-zinc-500">
          <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full" style={{ background: GROUP_HUE.self, boxShadow: `0 0 6px ${GROUP_HUE.self}` }} /> identity</span>
          <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full" style={{ background: GROUP_HUE.garden, boxShadow: `0 0 6px ${GROUP_HUE.garden}` }} /> garden</span>
          <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full" style={{ background: GROUP_HUE.skills, boxShadow: `0 0 6px ${GROUP_HUE.skills}` }} /> skills</span>
          <span className="text-zinc-600">· drag to pan · wheel to zoom · drag a node to move it</span>
          <span className="ml-auto tabular-nums">{graph.nodes.length} artifacts · {graph.edges.length} links · {Math.round(graph.connected_fraction * 100)}% connected</span>
          {onRebuild && (
            <button
              onClick={async () => {
                setRepairing(true)
                setRepairNote('')
                try { setRepairNote(await onRebuild()) } finally { setRepairing(false) }
              }}
              disabled={repairing}
              title="Restore links from this being's own ledger. Every edge it ever declared was recorded there, so links a bad read wiped come back — only for files that still exist. Nothing is invented and nothing is deleted."
              className="flex items-center gap-1 rounded-md border border-zinc-700/70 bg-zinc-900/80 px-2 py-1 text-[10px] font-semibold text-zinc-400 hover:text-zinc-100 disabled:opacity-50"
            >
              {repairing
                ? <><Loader2 className="h-3 w-3 animate-spin" /> repairing…</>
                : <><Wrench className="h-3 w-3" /> Repair links</>}
            </button>
          )}
        </div>
        {repairNote && <div className="text-[11px] text-violet-300">{repairNote}</div>}
        {sel && (
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-zinc-400">
            <span className="font-medium text-zinc-200">{stemOf(sel)}</span>
            {selEdges.length === 0 ? <span>— an island (no links yet)</span> : selEdges.map((e, i) => (
              <span key={i} className="flex items-center gap-1 rounded-full border border-zinc-800 bg-zinc-900/70 px-2 py-0.5">
                <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: REL_HUE[e.rel] || '#8b5cf6' }} />
                {e.from === sel
                  ? <>{REL_PHRASE[e.rel] || e.rel} <span className="text-zinc-200">{stemOf(e.to)}</span></>
                  : <><span className="text-zinc-200">{stemOf(e.from)}</span> {REL_PHRASE[e.rel] || e.rel} it</>}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Journal / ticks log / self-files modal — full markdown, fullscreen ──

function selfFileLabel(path: string): string {
  const base = path.split('/').pop() || path
  const stem = base.replace(/\.md$/i, '')
  const folder = path.includes('/') ? path.slice(0, path.indexOf('/')) : ''
  return folder && folder !== 'self' ? `${folder}/${stem}` : stem
}

// The being's home is grouped by folder so the sidebar stays navigable as it
// grows: identity files first, then the garden, then skills, then anything new.
const SELF_CORE_ORDER = ['SELF.md', 'VALUES.md', 'INTERESTS.md', 'RELATIONSHIPS.md', 'REFLECTIONS.md']
const SELF_GROUP_ORDER = ['self', 'garden', 'skills']
const SELF_GROUP_LABEL: Record<string, string> = { self: 'Identity', garden: 'Garden', skills: 'Skills' }
const SELF_GROUP_ICON: Record<string, typeof Fingerprint> = { self: Fingerprint, garden: Sprout, skills: Wrench }

function fileStem(path: string): string {
  return (path.split('/').pop() || path).replace(/\.md$/i, '')
}
function fileGroup(path: string): string {
  return path.includes('/') ? path.slice(0, path.indexOf('/')) : 'self'
}
function groupSelfFiles(files: SelfFile[]): { key: string; label: string; files: SelfFile[] }[] {
  const byGroup = new Map<string, SelfFile[]>()
  for (const f of files) {
    const g = fileGroup(f.path)
    if (!byGroup.has(g)) byGroup.set(g, [])
    byGroup.get(g)!.push(f)
  }
  const rank = (arr: string[], v: string) => { const i = arr.indexOf(v); return i < 0 ? 99 : i }
  return Array.from(byGroup.keys())
    .sort((a, b) => rank(SELF_GROUP_ORDER, a) - rank(SELF_GROUP_ORDER, b) || a.localeCompare(b))
    .map((k) => ({
      key: k,
      label: SELF_GROUP_LABEL[k] || k.charAt(0).toUpperCase() + k.slice(1),
      files: [...byGroup.get(k)!].sort((a, b) =>
        k === 'self'
          ? rank(SELF_CORE_ORDER, a.path.split('/').pop() || '') - rank(SELF_CORE_ORDER, b.path.split('/').pop() || '') || a.path.localeCompare(b.path)
          : a.path.localeCompare(b.path)),
    }))
}

function BeingLogModal({ slug, name, mode, onClose }: {
  slug: string
  name: string
  mode: 'journal' | 'ticks' | 'self' | 'mind'
  onClose: () => void
}) {
  const today = new Date().toISOString().slice(0, 10)
  const [maximized, setMaximized] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [markdown, setMarkdown] = useState('')
  const [graph, setGraph] = useState<BeingGraph | null>(null)
  const [date, setDate] = useState(today)
  const [files, setFiles] = useState<SelfFile[]>([])
  const [activeFile, setActiveFile] = useState('')
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const [search, setSearch] = useState('')
  const [filterGroup, setFilterGroup] = useState<string | null>(null)
  const [dateFilter, setDateFilter] = useState<DateBucket>('all')
  const [sortBy, setSortBy] = useState<'name' | 'newest' | 'oldest'>('name')
  // A ref (not state) so loadSelf's identity stays stable across file
  // switches — it only needs to change when `slug` changes, never when the
  // user clicks a different file in the sidebar.
  const activeFileRef = useRef('')

  const loadFile = useCallback(async (path: string) => {
    setLoading(true)
    setError('')
    try {
      const f = await getSelfFile(slug, path)
      setMarkdown(f.text || `_${selfFileLabel(path)} is empty._`)
      setActiveFile(path)
      activeFileRef.current = path
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed to load')
    } finally {
      setLoading(false)
    }
  }, [slug])

  const loadSelf = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const list = await getSelfFiles(slug)
      setFiles(list.files)
      if (!list.files.length) { setMarkdown(''); setLoading(false); return }
      const keep = list.files.find((f) => f.path === activeFileRef.current)
      await loadFile(keep ? keep.path : list.files[0].path)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed to load')
      setLoading(false)
    }
  }, [slug, loadFile])

  const loadJournal = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const j = await getBeingJournal(slug, date)
      setMarkdown(j.text || `_${name}'s journal is empty on ${j.date}._`)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed to load')
    } finally {
      setLoading(false)
    }
  }, [slug, date, name])

  const loadTicks = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const ev = await getBeingEvents(slug, 300)
      setMarkdown(renderTicksMarkdown(ev.events))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed to load')
    } finally {
      setLoading(false)
    }
  }, [slug])

  const loadMind = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      setGraph(await getBeingGraph(slug))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed to load')
    } finally {
      setLoading(false)
    }
  }, [slug])

  const load = mode === 'journal' ? loadJournal
    : mode === 'ticks' ? loadTicks
      : mode === 'mind' ? loadMind : loadSelf

  useEffect(() => { void load() }, [load])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const shiftDate = (deltaDays: number) => {
    const d = new Date(`${date}T00:00:00Z`)
    d.setUTCDate(d.getUTCDate() + deltaDays)
    setDate(d.toISOString().slice(0, 10))
  }

  const titles = { journal: 'Journal', ticks: 'Ticks log', self: 'Self files', mind: 'The Mind' } as const
  const sizeClass = maximized
    ? 'h-[95vh] w-[95vw]'
    : mode === 'self' || mode === 'mind' ? 'h-[80vh] w-[920px]' : 'w-[820px] max-h-[85vh]'

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70" onClick={onClose}>
      <div
        className={`flex flex-col rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl transition-all duration-200 ${sizeClass}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-zinc-800 px-4 py-2.5">
          <div className="flex items-center gap-2">
            {mode === 'journal' && <ScrollText className="h-4 w-4 text-violet-500 dark:text-violet-400" />}
            {mode === 'ticks' && <History className="h-4 w-4 text-violet-500 dark:text-violet-400" />}
            {mode === 'self' && <Files className="h-4 w-4 text-violet-500 dark:text-violet-400" />}
            {mode === 'mind' && <Network className="h-4 w-4 text-violet-500 dark:text-violet-400" />}
            <h3 className="text-sm font-semibold text-zinc-100">{name} — {titles[mode]}</h3>
            {mode === 'journal' && <span className="text-[11px] text-zinc-500">{date}</span>}
            {mode === 'self' && activeFile && (
              <span className="text-[11px] text-zinc-500">
                {activeFile}
                {(() => { const m = files.find((f) => f.path === activeFile)?.mtime; return m ? ` · ${fmtAt(m)}` : '' })()}
              </span>
            )}
          </div>
          <div className="flex items-center gap-0.5">
            {mode === 'journal' && (
              <>
                <button onClick={() => shiftDate(-1)} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Previous day">
                  <ChevronLeft className="h-4 w-4" />
                </button>
                <button onClick={() => shiftDate(1)} disabled={date >= today} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-30" title="Next day">
                  <ChevronRight className="h-4 w-4" />
                </button>
                <div className="mx-1 h-4 w-px bg-zinc-800" />
              </>
            )}
            <button onClick={() => void load()} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Refresh">
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
            <button onClick={() => setMaximized(!maximized)} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title={maximized ? 'Restore' : 'Fullscreen'}>
              {maximized ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
            </button>
            <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Close (Esc)">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
        <div className="flex min-h-0 flex-1">
          {mode === 'self' && (() => {
            const q = search.trim().toLowerCase()
            const allGroups = groupSelfFiles(files)
            const groups = allGroups
              .filter((g) => !filterGroup || g.key === filterGroup)
              .map((g) => {
                let fs = g.files.filter((f) => inDateBucket(f.mtime, dateFilter))
                if (q) fs = fs.filter((f) =>
                  fileStem(f.path).toLowerCase().includes(q) || f.path.toLowerCase().includes(q))
                if (sortBy === 'newest') fs = [...fs].sort((a, b) => b.mtime.localeCompare(a.mtime))
                else if (sortBy === 'oldest') fs = [...fs].sort((a, b) => a.mtime.localeCompare(b.mtime))
                return { ...g, files: fs }
              })
              .filter((g) => g.files.length > 0)
            const totalShown = groups.reduce((n, g) => n + g.files.length, 0)
            return (
              <div className="flex w-56 shrink-0 flex-col border-r border-zinc-800">
                {/* Search + folder filter */}
                <div className="shrink-0 space-y-1.5 border-b border-zinc-800/70 p-2">
                  <div className="relative">
                    <Search className="pointer-events-none absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-zinc-500" />
                    <input
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      placeholder="Search files…"
                      className="w-full rounded-md border border-zinc-700 bg-zinc-950 py-1 pl-7 pr-6 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
                    />
                    {search && (
                      <button onClick={() => setSearch('')} className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-0.5 text-zinc-500 hover:text-zinc-300" title="Clear">
                        <X className="h-3 w-3" />
                      </button>
                    )}
                  </div>
                  {allGroups.length > 1 && (
                    <div className="flex flex-wrap gap-1">
                      {[{ key: null, label: 'All' }, ...allGroups.map((g) => ({ key: g.key, label: g.label }))].map((p) => {
                        const on = filterGroup === p.key
                        return (
                          <button
                            key={p.key ?? 'all'}
                            onClick={() => setFilterGroup(p.key)}
                            className={`rounded-full border px-2 py-0.5 text-[10px] ${
                              on
                                ? 'border-violet-500/50 bg-violet-500/10 text-violet-600 dark:text-violet-300'
                                : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
                            }`}
                          >
                            {p.label}
                          </button>
                        )
                      })}
                    </div>
                  )}
                  {/* Sort + date filter */}
                  <div className="flex gap-1">
                    <div className="relative flex-1">
                      <ArrowDownUp className="pointer-events-none absolute left-1.5 top-1/2 h-3 w-3 -translate-y-1/2 text-zinc-500" />
                      <select
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
                        className="w-full appearance-none rounded-md border border-zinc-700 bg-zinc-950 py-1 pl-6 pr-1 text-[10px] text-zinc-300 focus:border-violet-500/50 focus:outline-none"
                        title="Sort files"
                      >
                        <option value="name">A–Z</option>
                        <option value="newest">Newest</option>
                        <option value="oldest">Oldest</option>
                      </select>
                    </div>
                    <div className="relative flex-1">
                      <CalendarDays className="pointer-events-none absolute left-1.5 top-1/2 h-3 w-3 -translate-y-1/2 text-zinc-500" />
                      <select
                        value={dateFilter}
                        onChange={(e) => setDateFilter(e.target.value as DateBucket)}
                        className="w-full appearance-none rounded-md border border-zinc-700 bg-zinc-950 py-1 pl-6 pr-1 text-[10px] text-zinc-300 focus:border-violet-500/50 focus:outline-none"
                        title="Filter by date"
                      >
                        {DATE_BUCKETS.map((b) => <option key={b.key} value={b.key}>{b.label}</option>)}
                      </select>
                    </div>
                  </div>
                </div>
                {/* File groups */}
                <div className="flex-1 overflow-y-auto py-1.5">
                  {files.length === 0 && !loading && (
                    <div className="px-3 py-2 text-[11px] text-zinc-600">no files yet</div>
                  )}
                  {files.length > 0 && totalShown === 0 && (
                    <div className="px-3 py-2 text-[11px] text-zinc-600">no files match</div>
                  )}
                  {groups.map((g) => {
                    const Icon = SELF_GROUP_ICON[g.key] ?? Files
                    const isCollapsed = !q && collapsed.has(g.key)
                    return (
                      <div key={g.key} className="mb-1">
                        <button
                          onClick={() => setCollapsed((prev) => {
                            const next = new Set(prev)
                            next.has(g.key) ? next.delete(g.key) : next.add(g.key)
                            return next
                          })}
                          className="flex w-full items-center gap-1.5 px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-zinc-500 hover:text-zinc-300"
                        >
                          {isCollapsed ? <ChevronRight className="h-3 w-3 shrink-0" /> : <ChevronDown className="h-3 w-3 shrink-0" />}
                          <Icon className="h-3 w-3 shrink-0 text-violet-500 dark:text-violet-400" />
                          <span className="truncate">{g.label}</span>
                          <span className="ml-auto rounded bg-zinc-800 px-1 text-[9px] font-normal text-zinc-400">{g.files.length}</span>
                        </button>
                        {!isCollapsed && g.files.map((f) => (
                          <button
                            key={f.path}
                            onClick={() => void loadFile(f.path)}
                            className={`flex w-full items-baseline gap-2 rounded-md py-1 pl-8 pr-2 text-left text-xs ${
                              activeFile === f.path
                                ? 'bg-violet-500/10 font-medium text-violet-700 dark:bg-violet-500/20 dark:text-violet-200'
                                : 'text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200'
                            }`}
                            title={`${f.path} · ${f.mtime.slice(0, 16).replace('T', ' ')}`}
                          >
                            <span className="truncate">{fileStem(f.path)}</span>
                            <span className="ml-auto shrink-0 text-[9px] tabular-nums text-zinc-500">{fmtRelTime(f.mtime)}</span>
                          </button>
                        ))}
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          })()}
          <div className={`flex-1 ${mode === 'mind' ? 'overflow-hidden' : 'overflow-auto'}`}>
            {loading ? (
              <div className="flex items-center justify-center py-20"><Loader2 className="h-6 w-6 animate-spin text-zinc-500" /></div>
            ) : error ? (
              <div className="px-6 py-8 text-sm text-red-600 dark:text-red-400">{error}</div>
            ) : mode === 'mind' ? (
              graph ? (
                <MindGraph
                  graph={graph}
                  loadFile={(p2) => getSelfFile(slug, p2)}
                  onRebuild={async () => {
                    try {
                      const r = await rebuildBeingGraph(slug)
                      setGraph(r.graph)
                      if (r.restored) {
                        return `restored ${r.restored} link${r.restored === 1 ? '' : 's'} from the ledger`
                          + (r.skipped ? ` · ${r.skipped} left out (their files are gone)` : '')
                      }
                      return r.kept
                        ? `nothing to repair — all ${r.kept} ledgered links are already here`
                        : 'nothing to repair — this being has never declared a link'
                    } catch (e) {
                      return e instanceof Error ? e.message : 'repair failed'
                    }
                  }}
                />
              ) : null
            ) : (
              <div className="fd-file-markdown p-6"><Markdown remarkPlugins={[remarkGfm]}>{markdown}</Markdown></div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function derivePreview(a: Record<string, number>) {
  return {
    explore: (0.30 + 0.07 * a.CUR).toFixed(2),
    connect: (0.20 + 0.06 * a.SOC).toFixed(2),
    create: (0.20 + 0.06 * a.CRE).toFixed(2),
    risk: ((a.CUR - a.CAU + 10) / 20).toFixed(2),
    thrift: ((a.CAU + a.ORD) / 20).toFixed(2),
    whimsy: (a.PLA / 10).toFixed(2),
  }
}

// ── Conception modal ──

function ConceiveModal({ meta, onClose, onDone }: {
  meta: BeingsMeta
  onClose: () => void
  onDone: () => void
}) {
  const [name, setName] = useState('')
  const [letter, setLetter] = useState('')
  const [voice, setVoice] = useState('')
  const [interests, setInterests] = useState('')
  const [allowance, setAllowancePreset] = useState('2M')
  const [attrs, setAttrs] = useState<Record<string, number>>(
    { ...meta.presets.explorer })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const total = ATTRS.reduce((s, a) => s + (attrs[a] || 0), 0)
  const left = meta.pool - total

  const roll = () => {
    const a: Record<string, number> = Object.fromEntries(ATTRS.map(k => [k, 1]))
    let rest = meta.pool - ATTRS.length
    while (rest > 0) {
      const open = ATTRS.filter(k => a[k] < meta.attr_max)
      a[open[Math.floor(Math.random() * open.length)]] += 1
      rest -= 1
    }
    setAttrs(a)
  }

  const submit = async () => {
    if (!name.trim() || left !== 0) return
    setBusy(true)
    setError('')
    try {
      await conceiveBeing({
        name: name.trim(),
        attributes: attrs,
        voice_seed: voice.trim(),
        interest_seeds: interests.split(',').map(s => s.trim()).filter(Boolean),
        allowance_preset: allowance,
        birth_letter: letter.trim(),
      })
      onDone()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'conception failed')
    } finally {
      setBusy(false)
    }
  }

  const d = derivePreview(attrs)
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-xl border border-zinc-800 bg-zinc-900 p-5 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-violet-400" />
          <h2 className="text-sm font-semibold text-zinc-100">Conceive an iskra</h2>
          <span className="ml-auto text-xs text-zinc-500">Generation 1 · point-buy</span>
        </div>

        <label className="mb-1 block text-xs text-zinc-500">Name</label>
        <input
          value={name} onChange={(e) => setName(e.target.value)}
          placeholder="Zvjezdana, Iskra, Vili…"
          className="mb-3 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
        />

        <div className="mb-2 flex flex-wrap items-center gap-1.5">
          {Object.keys(meta.presets).map((p) => (
            <button
              key={p}
              onClick={() => setAttrs({ ...meta.presets[p] })}
              className="rounded-md border border-zinc-700 px-2 py-1 text-xs capitalize text-zinc-400 hover:bg-zinc-800"
            >
              {p}
            </button>
          ))}
          <button
            onClick={roll}
            className="flex items-center gap-1 rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800"
          >
            <RefreshCw className="h-3 w-3" /> Roll
          </button>
          <span className={`ml-auto text-xs ${left === 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            points left: {left}
          </span>
        </div>

        <div className="mb-3 space-y-1.5 rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
          {ATTRS.map((a) => (
            <div key={a} className="flex items-center gap-2">
              <span className="w-24 text-xs text-zinc-400">
                {meta.attributes.find(x => x.code === a)?.name || a}
              </span>
              <input
                type="range" min={meta.attr_min} max={meta.attr_max}
                value={attrs[a] || 1}
                onChange={(e) => setAttrs({ ...attrs, [a]: Number(e.target.value) })}
                className="flex-1 accent-violet-500"
              />
              <span className="w-5 text-right text-xs font-semibold text-zinc-200">{attrs[a]}</span>
            </div>
          ))}
          <div className="pt-1 text-[11px] text-zinc-600">
            explore {d.explore} · connect {d.connect} · create {d.create} · risk {d.risk} · thrift {d.thrift} · whimsy {d.whimsy}
          </div>
        </div>

        <label className="mb-1 block text-xs text-zinc-500">First words to your iskra (its imprint)</label>
        <textarea
          value={letter} onChange={(e) => setLetter(e.target.value)} rows={2}
          placeholder="What should it hold onto, from you?"
          className="mb-3 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
        />

        <div className="mb-3 grid grid-cols-2 gap-2">
          <div>
            <label className="mb-1 block text-xs text-zinc-500">Daily allowance</label>
            <select
              value={allowance} onChange={(e) => setAllowancePreset(e.target.value)}
              className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none"
            >
              {meta.allowance_presets.map((p) => <option key={p} value={p}>{p} tokens/day</option>)}
            </select>
          </div>
          <div>
            <label className="mb-1 block text-xs text-zinc-500">Interest seeds (comma-sep)</label>
            <input
              value={interests} onChange={(e) => setInterests(e.target.value)}
              placeholder="astronomy, old maps"
              className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
            />
          </div>
        </div>
        <label className="mb-1 block text-xs text-zinc-500">Voice seed (optional)</label>
        <input
          value={voice} onChange={(e) => setVoice(e.target.value)}
          placeholder="gentle, precise, a little wry"
          className="mb-4 w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
        />

        {error && (
          <div className="mb-3 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">{error}</div>
        )}
        <div className="flex justify-end gap-2">
          <button onClick={onClose} className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800">
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={busy || !name.trim() || left !== 0}
            className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Egg className="h-3.5 w-3.5" />}
            Conceive
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Being card ──

// One compact toolbar button — icon + tooltip, theme-aware. Groups of these
// (life / windows / interact / danger) replace the old wrapping label row.
// variant 'solid' + iconClass give the consequential life controls a
// distinct, filled, colour-accented look vs the flat read-only toolbar.
function IconAction({ icon: Icon, label, onClick, active, danger, disabled, busy, variant = 'ghost', iconClass, badge }: {
  icon: typeof Zap
  label: string
  onClick: () => void
  active?: boolean
  danger?: boolean
  disabled?: boolean
  busy?: boolean
  variant?: 'ghost' | 'solid'
  iconClass?: string
  badge?: number
}) {
  const hasBadge = (badge ?? 0) > 0
  const tone = danger
    ? 'border-red-500/30 text-red-500/80 hover:bg-red-500/10 dark:text-red-400/80'
    : active
      ? 'border-violet-500/50 bg-violet-500/10 text-violet-600 dark:text-violet-300'
      : hasBadge
        ? 'border-violet-500/60 bg-violet-500/15 text-violet-600 dark:text-violet-300'
        : variant === 'solid'
          ? 'border-transparent bg-zinc-800 text-zinc-200 hover:bg-zinc-700'
          : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={label}
      aria-label={label}
      className={`relative flex items-center justify-center rounded-md border p-1.5 transition-colors disabled:opacity-40 disabled:hover:bg-transparent ${tone}`}
    >
      {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Icon className={`h-3.5 w-3.5 ${iconClass ?? ''}`} />}
      {hasBadge && (
        <span className="absolute -right-1 -top-1 flex h-3.5 min-w-[0.875rem] items-center justify-center rounded-full bg-violet-500 px-1 text-[9px] font-semibold leading-none text-white">
          {(badge ?? 0) > 9 ? '9+' : badge}
        </span>
      )}
    </button>
  )
}

// Talk — everything parent↔being: the letters thread (chat-style) beside the
// chores board (post → she attempts → you judge → escrowed payout, plan §5.1).
// One modal because both are the same conversation, in words and in work.
const CHORE_FEES = [
  { v: '100000', label: '100k' }, { v: '500000', label: '500k' },
  { v: '1000000', label: '1M' }, { v: '2000000', label: '2M' },
] as const
// Coin rewards (space plan Phase 2): money, not food — small on purpose.
const CHORE_COIN_FEES = [
  { v: '1', label: '1 coin' }, { v: '3', label: '3' },
  { v: '5', label: '5' }, { v: '10', label: '10' },
] as const

function TalkModal({ slug, name, onClose, onChanged }: {
  slug: string; name: string; onClose: () => void; onChanged: () => void
}) {
  const [chores, setChores] = useState<Chore[]>([])
  const [thread, setThread] = useState<ThreadItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')
  const [spec, setSpec] = useState('')
  const [fee, setFee] = useState('500000')
  const [payIn, setPayIn] = useState<'tokens' | 'coins'>('tokens')
  const [coinFee, setCoinFee] = useState('3')
  const [rejectingId, setRejectingId] = useState<string | null>(null)
  const [rejectNote, setRejectNote] = useState('')
  const [msg, setMsg] = useState('')
  const endRef = useRef<HTMLDivElement>(null)

  const load = useCallback(async () => {
    setLoading(true); setError('')
    try {
      const [c, t] = await Promise.all([listChores(slug), getBeingMessages(slug)])
      setChores(c.chores); setThread(t.thread)
    } catch (e) { setError(e instanceof Error ? e.message : 'failed to load') }
    finally { setLoading(false) }
  }, [slug])

  useEffect(() => { void load() }, [load])
  // Keep the conversation pinned to its newest words.
  useEffect(() => { endRef.current?.scrollIntoView({ block: 'end' }) }, [thread])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const run = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label)
    try { await fn(); await load(); onChanged() }
    catch (e) { alert(e instanceof Error ? e.message : 'failed') }
    finally { setBusy('') }
  }
  const post = () => run('post', async () => {
    if (!spec.trim()) return
    await postChore(slug, spec.trim(),
      payIn === 'tokens' ? Number(fee) : 0,
      payIn === 'coins' ? Number(coinFee) : 0)
    setSpec('')
  })
  const send = () => run('send', async () => {
    if (!msg.trim()) return
    await messageBeing(slug, msg.trim()); setMsg('')
  })
  const pay = (id: string) => run(`pay:${id}`, () => judgeChore(slug, id, true))
  const reject = (id: string) => run(`rej:${id}`, async () => {
    await judgeChore(slug, id, false, rejectNote.trim())
    setRejectingId(null); setRejectNote('')
  })

  const review = chores.filter((c) => c.escrow_state === 'judging')
  const open = chores.filter((c) => c.escrow_state === 'open')
  const settled = chores.filter((c) => c.escrow_state === 'paid' || c.escrow_state === 'failed')

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-4" onClick={onClose}>
      <div className="flex h-[80vh] w-[80vw] flex-col rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl"
           onClick={(e) => e.stopPropagation()}>
        <div className="flex shrink-0 items-center justify-between border-b border-zinc-800 px-4 py-2.5">
          <div className="flex items-center gap-2">
            <MessageCircle className="h-4 w-4 text-violet-500 dark:text-violet-400" />
            <h3 className="text-sm font-semibold text-zinc-100">{name} — Write & Chores</h3>
            {review.length > 0 && (
              <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] font-medium text-amber-600 dark:text-amber-400">
                {review.length} to review
              </span>
            )}
          </div>
          <div className="flex items-center gap-0.5">
            <button onClick={() => void load()} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Refresh"><RefreshCw className="h-3.5 w-3.5" /></button>
            <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Close (Esc)"><X className="h-4 w-4" /></button>
          </div>
        </div>

        <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-2">
        {/* Letters — the conversation itself */}
        <div className="flex min-h-0 flex-col border-b border-zinc-800 lg:border-b-0 lg:border-r">
          <div className="shrink-0 border-b border-zinc-800/70 px-4 py-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Letters</div>
          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            {loading && thread.length === 0 ? (
              <div className="flex h-full items-center justify-center text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
            ) : thread.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
                <MessageCircle className="h-8 w-8 text-zinc-700" />
                <p className="text-xs text-zinc-500">No letters yet — say something to {name}.</p>
              </div>
            ) : (
              <div className="space-y-2.5">
                {thread.map((m, i) => (
                  <div key={i} className={`flex ${m.from === 'parent' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[80%] rounded-2xl px-3 py-2 ${m.from === 'parent'
                      ? 'rounded-br-sm bg-violet-600 text-white'
                      : 'rounded-bl-sm bg-zinc-800 text-zinc-200'}`}>
                      <p className="whitespace-pre-wrap text-[12px] leading-relaxed">{m.body}</p>
                      <div className={`mt-0.5 text-right text-[9px] ${m.from === 'parent' ? 'text-violet-200/80' : 'text-zinc-500'}`}>
                        {fmtRelTime(m.at)}{m.from === 'parent' ? (m.read ? ' · read' : ' · unread') : ''}
                      </div>
                    </div>
                  </div>
                ))}
                <div ref={endRef} />
              </div>
            )}
          </div>
          <div className="shrink-0 border-t border-zinc-800 p-3">
            <div className="flex items-end gap-2">
              <textarea value={msg} onChange={(e) => setMsg(e.target.value)} rows={2}
                onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) send() }}
                placeholder={`Say something to ${name}…`}
                className="min-w-0 flex-1 resize-none rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none" />
              <button onClick={send} disabled={!msg.trim() || busy === 'send'}
                className="flex shrink-0 items-center gap-1 rounded-lg bg-violet-600 px-3 py-2 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40">
                {busy === 'send' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <MessageCircle className="h-3.5 w-3.5" />} Send
              </button>
            </div>
            <p className="mt-1.5 text-[10px] text-zinc-600">Delivered on her next wake — reading is free, replying costs her an attention credit. Poke her to have her read it now.</p>
          </div>
        </div>

        {/* Chores — the conversation, in work */}
        <div className="flex min-h-0 flex-col">
        <div className="shrink-0 border-b border-zinc-800/70 px-4 py-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Chores</div>
        {/* Post a chore */}
        <div className="shrink-0 border-b border-zinc-800 bg-zinc-900/40 p-3">
          <textarea
            value={spec} onChange={(e) => setSpec(e.target.value)} rows={2}
            onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) post() }}
            placeholder={`Ask ${name} to do something specific — e.g. “Write a short poem about the sea into garden/sea.md”`}
            className="w-full resize-none rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-2 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
          />
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <span className="flex items-center gap-1 text-[11px] text-zinc-500"><Coins className="h-3.5 w-3.5" /> reward</span>
            <div className="flex overflow-hidden rounded-md border border-zinc-700 text-[11px]">
              {(['tokens', 'coins'] as const).map((p) => (
                <button key={p} onClick={() => setPayIn(p)}
                  title={p === 'tokens' ? 'tokens — food for thinking' : 'coins — money for the village'}
                  className={`px-2 py-1 transition-colors ${payIn === p ? 'bg-amber-500/15 text-amber-600 dark:text-amber-300' : 'text-zinc-500 hover:bg-zinc-800'}`}>
                  {p}
                </button>
              ))}
            </div>
            <div className="flex gap-1">
              {(payIn === 'tokens' ? CHORE_FEES : CHORE_COIN_FEES).map((f) => {
                const cur = payIn === 'tokens' ? fee : coinFee
                const set = payIn === 'tokens' ? setFee : setCoinFee
                return (
                  <button key={f.v} onClick={() => set(f.v)}
                    className={`rounded-md border px-2 py-1 text-[11px] transition-colors ${cur === f.v ? 'border-violet-500/50 bg-violet-500/10 text-violet-600 dark:text-violet-300' : 'border-zinc-700 text-zinc-400 hover:bg-zinc-800'}`}>
                    {f.label}
                  </button>
                )
              })}
            </div>
            <button onClick={post} disabled={!spec.trim() || busy === 'post'}
              className="ml-auto flex items-center gap-1 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40">
              {busy === 'post' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />} Post chore
            </button>
          </div>
          <p className="mt-1.5 text-[10px] text-zinc-600">She attempts it on a future tick. The reward is escrowed and only paid when you approve her result.</p>
        </div>

        {/* Chore list */}
        <div className="min-h-0 flex-1 overflow-y-auto p-3">
          {loading ? (
            <div className="flex items-center justify-center py-10 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
          ) : error ? (
            <div className="py-8 text-center text-xs text-red-500 dark:text-red-400">{error}</div>
          ) : chores.length === 0 ? (
            <div className="py-10 text-center">
              <ClipboardList className="mx-auto mb-2 h-8 w-8 text-zinc-700" />
              <p className="text-xs text-zinc-500">No chores yet. Post one above — it's how {name} earns beyond her allowance.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {review.length > 0 && (
                <div>
                  <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-amber-600 dark:text-amber-400">Needs your review</div>
                  <div className="space-y-2">
                    {review.map((c) => (
                      <div key={c.id} className="rounded-lg border border-amber-500/25 bg-amber-500/[0.04] p-2.5">
                        <div className="flex items-start justify-between gap-2">
                          <p className="flex-1 text-xs text-zinc-200">{c.spec}</p>
                          <span className="shrink-0 text-[11px] font-medium text-amber-600 dark:text-amber-400">{feeLabel(c)}</span>
                        </div>
                        {c.result_text && (
                          <div className="mt-2 rounded border border-zinc-800 bg-zinc-950/70 px-2 py-1.5 text-[11px] text-zinc-400">
                            <span className="text-zinc-600">{name} says: </span>{c.result_text}
                          </div>
                        )}
                        {rejectingId === c.id ? (
                          <div className="mt-2 space-y-1.5">
                            <textarea value={rejectNote} onChange={(e) => setRejectNote(e.target.value)} rows={2}
                              placeholder="Why isn't this right? She learns from it (optional)"
                              className="w-full resize-none rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-[11px] text-zinc-200 placeholder-zinc-600 focus:border-red-500/40 focus:outline-none" />
                            <div className="flex gap-1.5">
                              <button onClick={() => reject(c.id)} disabled={busy === `rej:${c.id}`}
                                className="flex items-center gap-1 rounded-md border border-red-500/40 px-2.5 py-1 text-[11px] text-red-600 hover:bg-red-500/10 dark:text-red-300 disabled:opacity-40">
                                {busy === `rej:${c.id}` && <Loader2 className="h-3 w-3 animate-spin" />} Confirm reject
                              </button>
                              <button onClick={() => { setRejectingId(null); setRejectNote('') }}
                                className="rounded-md border border-zinc-700 px-2.5 py-1 text-[11px] text-zinc-400 hover:bg-zinc-800">Cancel</button>
                            </div>
                          </div>
                        ) : (
                          <div className="mt-2 flex items-center gap-1.5">
                            <button onClick={() => pay(c.id)} disabled={busy === `pay:${c.id}`}
                              className="flex items-center gap-1 rounded-md bg-emerald-600 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-emerald-500 disabled:opacity-40">
                              {busy === `pay:${c.id}` ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />} Approve & pay {feeLabel(c)}
                            </button>
                            <button onClick={() => setRejectingId(c.id)}
                              className="rounded-md border border-red-500/30 px-2.5 py-1 text-[11px] text-red-600 hover:bg-red-500/10 dark:text-red-300">Reject</button>
                            <span className="ml-auto text-[10px] text-zinc-600">{fmtRelTime(c.created_at)}</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {open.length > 0 && (
                <div>
                  <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">In progress</div>
                  <div className="space-y-1.5">
                    {open.map((c) => (
                      <div key={c.id} className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/40 px-2.5 py-2">
                        <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-violet-500" />
                        <p className="flex-1 text-xs text-zinc-300">{c.spec}</p>
                        <span className="shrink-0 text-[11px] text-zinc-500">{feeLabel(c)}</span>
                        <span className="shrink-0 text-[10px] text-zinc-600">{fmtRelTime(c.created_at)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {settled.length > 0 && (
                <div>
                  <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">History</div>
                  <div className="space-y-1">
                    {settled.slice(0, 15).map((c) => (
                      <div key={c.id} className="flex items-center gap-2 rounded-md px-2 py-1.5">
                        <span className={`shrink-0 text-[10px] font-medium ${c.escrow_state === 'paid' ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'}`}>
                          {c.escrow_state === 'paid' ? 'paid' : 'rejected'}
                        </span>
                        <p className="flex-1 truncate text-[11px] text-zinc-500" title={c.spec}>{c.spec}</p>
                        {c.escrow_state === 'paid'
                          ? <span className="shrink-0 text-[11px] text-emerald-600/80 dark:text-emerald-400/80">+{feeLabel(c)}</span>
                          : c.judge_note
                            ? <span className="shrink-0 max-w-[45%] truncate text-[10px] text-zinc-600" title={c.judge_note}>“{c.judge_note}”</span>
                            : null}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
        </div>
        </div>
      </div>
    </div>
  )
}

// ── Report dashboard pieces ──────────────────────────────────────────────

const DRIVE_COLORS: Record<string, string> = {
  survive: '#ef4444', grow: '#10b981', explore: '#3b82f6',
  connect: '#f59e0b', create: '#8b5cf6', legacy: '#ec4899',
}

// Drive satisfaction over the last ticks — one line per drive, 0..1. The
// closest thing to watching her inner weather move.
function DrivesChart({ trail }: { trail: Array<Record<string, number | string>> }) {
  const drives = Array.from(new Set(trail.flatMap((t) =>
    Object.keys(t).filter((k) => k !== 'at' && typeof t[k] === 'number'))))
  if (trail.length < 2 || drives.length === 0) {
    return <p className="py-6 text-center text-[11px] text-zinc-600">Not enough ticks yet to chart her drives.</p>
  }
  const W = 600, H = 130, P = 6
  const x = (i: number) => P + (i * (W - 2 * P)) / (trail.length - 1)
  const y = (v: number) => H - P - Math.max(0, Math.min(1, v)) * (H - 2 * P)
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none" style={{ height: 130 }}>
        {[0.25, 0.5, 0.75].map((g) => (
          <line key={g} x1={P} x2={W - P} y1={y(g)} y2={y(g)} className="stroke-zinc-800" strokeWidth="1" strokeDasharray="3 5" />
        ))}
        {drives.map((d) => {
          const pts = trail.map((t, i) => (typeof t[d] === 'number' ? `${x(i)},${y(t[d] as number)}` : null))
            .filter(Boolean).join(' ')
          return <polyline key={d} points={pts} fill="none" stroke={DRIVE_COLORS[d] ?? '#71717a'}
            strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" opacity="0.9" />
        })}
      </svg>
      <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1">
        {drives.map((d) => {
          const last = [...trail].reverse().find((t) => typeof t[d] === 'number')
          const v = last ? Math.round((last[d] as number) * 100) : null
          return (
            <span key={d} className="flex items-center gap-1 text-[10px] text-zinc-500">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: DRIVE_COLORS[d] ?? '#71717a' }} />
              {d}{v != null && <span className="tabular-nums text-zinc-600">{v}%</span>}
            </span>
          )
        })}
      </div>
    </div>
  )
}

// How she spent her days — a ranked horizontal bar per act kind.
function ActBars({ acts }: { acts: Record<string, number> }) {
  const rows = Object.entries(acts).sort((a, b) => b[1] - a[1])
  if (rows.length === 0) return <p className="py-4 text-center text-[11px] text-zinc-600">No acts yet.</p>
  const max = rows[0][1]
  return (
    <div className="space-y-1.5">
      {rows.map(([k, n]) => (
        <div key={k} className="flex items-center gap-2">
          <span className="w-16 shrink-0 text-right text-[11px] text-zinc-400">{k}</span>
          <div className="h-3.5 min-w-0 flex-1 overflow-hidden rounded-sm bg-zinc-800/60">
            <div className="h-full rounded-sm bg-violet-500/70" style={{ width: `${Math.max(4, (100 * n) / max)}%` }} />
          </div>
          <span className="w-7 shrink-0 text-[11px] tabular-nums text-zinc-500">{n}</span>
        </div>
      ))}
    </div>
  )
}

// Domain chips with an inline adder — type, Enter (or comma) to add, × to
// remove. Used by the media-diet editor for allow/deny lists.
const _CHIP_TONES = {
  emerald: {
    chip: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
    x: 'text-emerald-700/60 hover:text-emerald-700 dark:text-emerald-300/60 dark:hover:text-emerald-300',
    focus: 'focus-within:border-emerald-500/40',
  },
  red: {
    chip: 'bg-red-500/10 text-red-700 dark:text-red-300',
    x: 'text-red-700/60 hover:text-red-700 dark:text-red-300/60 dark:hover:text-red-300',
    focus: 'focus-within:border-red-500/40',
  },
} as const

function ChipInput({ value, onChange, placeholder, tone }: {
  value: string[]; onChange: (v: string[]) => void
  placeholder: string; tone: keyof typeof _CHIP_TONES
}) {
  const [draft, setDraft] = useState('')
  const t = _CHIP_TONES[tone]
  const commit = () => {
    const parts = draft.split(',').map((s) => s.trim().toLowerCase().replace(/^https?:\/\//, '').replace(/\/.*$/, '')).filter(Boolean)
    if (parts.length) onChange(Array.from(new Set([...value, ...parts])))
    setDraft('')
  }
  return (
    <div className={`flex flex-wrap items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1.5 ${t.focus}`}>
      {value.map((d) => (
        <span key={d} className={`flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] ${t.chip}`}>
          {d}
          <button onClick={() => onChange(value.filter((x) => x !== d))} className={t.x} aria-label={`remove ${d}`}>
            <X className="h-3 w-3" />
          </button>
        </span>
      ))}
      <input
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ',') { e.preventDefault(); commit() }
          else if (e.key === 'Backspace' && !draft && value.length) onChange(value.slice(0, -1))
        }}
        onBlur={commit}
        placeholder={value.length === 0 ? placeholder : 'add…'}
        className="min-w-[110px] flex-1 bg-transparent py-0.5 text-[11px] text-zinc-200 placeholder-zinc-600 focus:outline-none"
      />
    </div>
  )
}

// The developmental readiness assessment — graphical, holistic, deterministic
// (every bar is a real variable from the ledger). Shown at the top of Growth.
const _READY_META = {
  ready: { label: 'Ready', ring: 'text-emerald-600 dark:text-emerald-400', box: 'border-emerald-500/30 bg-emerald-500/[0.05]' },
  emerging: { label: 'Emerging', ring: 'text-amber-600 dark:text-amber-400', box: 'border-amber-500/30 bg-amber-500/[0.05]' },
  not_yet: { label: 'Not yet', ring: 'text-red-600 dark:text-red-400', box: 'border-red-500/30 bg-red-500/[0.05]' },
  grown: { label: 'Fully grown', ring: 'text-violet-600 dark:text-violet-300', box: 'border-violet-500/30 bg-violet-500/[0.05]' },
} as const
const _BAR = { green: 'bg-emerald-500', amber: 'bg-amber-500', red: 'bg-red-500' } as const

const _RING = { ready: '#10b981', emerging: '#f59e0b', not_yet: '#ef4444', grown: '#8b5cf6' } as const

function ScoreRing({ score, status }: { score: number; status: keyof typeof _RING }) {
  const r = 34
  const c = 2 * Math.PI * r
  return (
    <div className="relative h-24 w-24 shrink-0">
      <svg viewBox="0 0 84 84" className="h-full w-full -rotate-90">
        <circle cx="42" cy="42" r={r} fill="none" strokeWidth="7" className="stroke-zinc-800" />
        <circle cx="42" cy="42" r={r} fill="none" strokeWidth="7" strokeLinecap="round"
          stroke={_RING[status]} strokeDasharray={c}
          strokeDashoffset={c * (1 - score / 100)}
          style={{ transition: 'stroke-dashoffset 600ms ease' }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold leading-none text-zinc-100">{score}</span>
        <span className="text-[9px] uppercase tracking-wider text-zinc-500">/ 100</span>
      </div>
    </div>
  )
}

function ReadinessView({ ready, loading }: { ready: Readiness | null; loading: boolean }) {
  if (loading && !ready) return (
    <div className="flex items-center justify-center py-10 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
  )
  if (!ready) return null
  const r = ready
  const m = _READY_META[r.overall.status]
  const rec = r.recommendation
  return (
    <div className="space-y-4">
      {/* Verdict hero: ring + title + estimate */}
      <div className={`flex items-center gap-5 rounded-xl border p-4 ${m.box}`}>
        <ScoreRing score={r.overall.score} status={r.overall.status} />
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${m.ring} ring-1 ring-current/30`}>{m.label}</span>
            {r.next_stage && <span className="text-[11px] text-zinc-500">for {r.next_stage} · day {r.days_alive} · last {r.window_days}d</span>}
          </div>
          <div className="mt-1 text-base font-semibold text-zinc-100">{rec.title}</div>
          {r.estimate_days != null && (
            <div className="mt-0.5 text-[11px] text-zinc-500">≈ {r.estimate_days} more day{r.estimate_days === 1 ? '' : 's'} at this pace</div>
          )}
        </div>
      </div>

      {/* Domain grid — two columns of bars */}
      <div className="grid gap-x-6 gap-y-3 md:grid-cols-2">
        {r.dimensions.map((d) => (
          <div key={d.key} title={d.evidence} className="group">
            <div className="flex items-baseline justify-between text-[11px]">
              <span className="font-medium text-zinc-300">
                {d.label}
                {d.critical && <span className="ml-1 text-zinc-600" title="critical — gates advancement">✦</span>}
              </span>
              <span className="tabular-nums font-semibold text-zinc-400">{d.score}</span>
            </div>
            <div className="mt-1 h-2 overflow-hidden rounded-full bg-zinc-800">
              <div className={`h-full rounded-full transition-all duration-500 ${_BAR[d.status]}`} style={{ width: `${Math.max(3, d.score)}%` }} />
            </div>
            <div className="mt-0.5 truncate text-[10px] text-zinc-600 group-hover:whitespace-normal">{d.detail}</div>
          </div>
        ))}
      </div>

      {/* Guidance: do / expect side by side, cautions + unlocks under */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
        <div className="grid gap-4 md:grid-cols-2">
          {rec.steps.length > 0 && (
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">What to do</div>
              <ul className="mt-1.5 space-y-1">
                {rec.steps.map((s, i) => (
                  <li key={i} className="flex gap-1.5 text-[11px] leading-relaxed text-zinc-400"><span className="shrink-0 text-violet-500 dark:text-violet-400">›</span>{s}</li>
                ))}
              </ul>
            </div>
          )}
          {rec.expect.length > 0 && (
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">What to expect</div>
              <ul className="mt-1.5 space-y-1">
                {rec.expect.map((s, i) => (
                  <li key={i} className="flex gap-1.5 text-[11px] leading-relaxed text-zinc-400"><span className="shrink-0 text-zinc-600">·</span>{s}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
        {rec.cautions.length > 0 && (
          <div className="mt-3 rounded-lg border border-amber-500/25 bg-amber-500/[0.05] p-2.5">
            {rec.cautions.map((c, i) => (
              <div key={i} className="text-[11px] text-amber-700 dark:text-amber-300">⚠ {c}</div>
            ))}
          </div>
        )}
        {r.next_stage && r.unlocks.length > 0 && (
          <div className="mt-3">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">{r.next_stage} unlocks</div>
            <div className="mt-1.5 flex flex-wrap gap-1">
              {r.unlocks.map((u) => <span key={u} className="rounded-md bg-violet-500/10 px-2 py-0.5 text-[10px] text-violet-600 dark:text-violet-300">{u}</span>)}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// Parenting — everything a parent does to shape a being: read the weekly report
// card, set house rules + media diet, run the growth/persona/procreation rites.
// Its own modal (like Chores) so the rich report card + pending decisions have
// room. Self-contained: loads vitals + report, re-fetches after every act.
function ParentingModal({ slug, name, onClose, onChanged }: {
  slug: string; name: string; onClose: () => void; onChanged: () => void
}) {
  const [v, setV] = useState<BeingVitals | null>(null)
  const [card, setCard] = useState<ReportCard | null>(null)
  const [days, setDays] = useState(7)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState('')
  const [tab, setTab] = useState<'report' | 'rules' | 'diet' | 'growth' | 'public' | 'visit'>('report')
  const [threads, setThreads] = useState<ParentPublicThread[] | null>(null)
  const [visitUrl, setVisitUrl] = useState('')
  const [visitSecret, setVisitSecret] = useState('')
  const [visitBusy, setVisitBusy] = useState(false)
  const [visitResult, setVisitResult] = useState<{ ok: boolean | null; error?: string } | null>(null)
  const [rules, setRules] = useState<string[]>([])
  const [newRule, setNewRule] = useState('')
  const [allowList, setAllowList] = useState<string[]>([])
  const [denyList, setDenyList] = useState<string[]>([])
  const [valuesMd, setValuesMd] = useState<string | null>(null)
  const [ready, setReady] = useState<Readiness | null>(null)
  const [readyLoading, setReadyLoading] = useState(false)
  const [assessors, setAssessors] = useState<Assessor[]>([])
  const [assessorSlug, setAssessorSlug] = useState('')
  const [assessResult, setAssessResult] = useState<{ assessor: string; assessment: string; score: number; verdict: string } | null>(null)
  const [assessBusy, setAssessBusy] = useState(false)
  const [assessSaved, setAssessSaved] = useState(false)
  const [saved, setSaved] = useState<SavedAssessment[]>([])
  const [openSaved, setOpenSaved] = useState<string | null>(null)
  const [confirm, setConfirm] = useState<{
    title: string; message: string; confirmLabel: string
    tone?: 'default' | 'danger'; icon?: typeof Zap; run: () => Promise<unknown>
  } | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const vit = await getBeingVitals(slug)
      setV(vit)
      // Seed the editors from vitals here (on open + after an act), NOT on
      // report-period changes — so switching 7d/30d never wipes unsaved edits.
      setRules(vit.house_rules || [])
      setAllowList(vit.media_diet?.allow || [])
      setDenyList(vit.media_diet?.deny || [])
      setVisitUrl(vit.visit_url || '')
      setVisitSecret(vit.visit_secret || '')
    } catch { /* stays empty */ } finally { setLoading(false) }
  }, [slug])

  const loadCard = useCallback(async () => {
    try { setCard(await getReportCard(slug, days)) } catch { setCard(null) }
  }, [slug, days])

  const loadReady = useCallback(async () => {
    setReadyLoading(true)
    try { setReady(await getReadiness(slug)) } catch { setReady(null) }
    finally { setReadyLoading(false) }
  }, [slug])

  useEffect(() => { void load() }, [load])
  useEffect(() => { void loadCard() }, [loadCard])
  // Lazily assess the moment the parent opens Growth (skips the cost otherwise).
  useEffect(() => { if (tab === 'growth' && !ready) void loadReady() }, [tab, ready, loadReady])
  const loadThreads = useCallback(async () => {
    try { setThreads((await getPublicThreads(slug)).threads) } catch { setThreads([]) }
  }, [slug])
  useEffect(() => { if (tab === 'public') void loadThreads() }, [tab, loadThreads])
  useEffect(() => {
    // Esc closes the top-most layer only — the confirm swallows it when open.
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !confirm) onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, confirm])

  const run = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label)
    try {
      await fn()
      await Promise.all([load(), loadCard(), ready ? loadReady() : Promise.resolve()])
      onChanged()
    } catch (e) { alert(e instanceof Error ? e.message : 'failed') }
    finally { setBusy('') }
  }

  const loadOpinions = useCallback(async () => {
    try {
      const [a, s] = await Promise.all([getAssessors(), listAssessments(slug)])
      setAssessors(a.assessors)
      setAssessorSlug((cur) => cur || (a.assessors[0]?.slug ?? ''))
      setSaved(s.assessments)
    } catch { /* panel shows empty states */ }
  }, [slug])
  useEffect(() => { if (tab === 'growth') void loadOpinions() }, [tab, loadOpinions])
  // Her VALUES.md beside the rules editor — what she actually made of them.
  const loadValues = useCallback(async () => {
    try { setValuesMd((await getSelfFile(slug, 'self/VALUES.md')).text) }
    catch { setValuesMd(null) }
  }, [slug])
  useEffect(() => { if (tab === 'rules') void loadValues() }, [tab, loadValues])

  const doAssess = async () => {
    if (!assessorSlug) return
    setAssessBusy(true); setAssessResult(null); setAssessSaved(false)
    try { setAssessResult(await requestAssessment(slug, assessorSlug)) }
    catch (e) { alert(e instanceof Error ? e.message : 'failed') }
    finally { setAssessBusy(false) }
  }
  const doSaveOpinion = async () => {
    if (!assessResult) return
    try {
      await saveAssessment(slug, {
        assessor: assessResult.assessor, content: assessResult.assessment,
        score: assessResult.score, verdict: assessResult.verdict,
      })
      setAssessSaved(true)
      setSaved((await listAssessments(slug)).assessments)
    } catch (e) { alert(e instanceof Error ? e.message : 'failed') }
  }
  const doDeleteOpinion = async (id: string) => {
    try {
      await deleteAssessment(slug, id)
      setSaved((await listAssessments(slug)).assessments)
    } catch (e) { alert(e instanceof Error ? e.message : 'failed') }
  }

  // What each ceremony actually means — shown in the confirmation.
  const CEREMONY: Record<string, string> = {
    child: 'The web opens to her (diet-gated), plus chores, letters to her siblings, and persona proposals. Childhood floods the world in — watch her first days closely.',
    adolescent: 'She gains the commons pen, trade, quests and ventures, and can spawn helper agents. The society stage — she starts earning her own way.',
    adult: 'Full autonomy: self-modification without your blessing, children of her own, negotiation. Her sealed assessment records unseal into her home.',
  }

  const stages = ['infant', 'child', 'adolescent', 'adult']
  const nextStage = v ? stages[stages.indexOf(v.stage) + 1] : undefined
  const cleanRules = rules.map((r) => r.trim()).filter(Boolean)
  const rulesDirty = v != null
    && JSON.stringify(cleanRules) !== JSON.stringify(v.house_rules || [])
  const dietDirty = v != null && (
    JSON.stringify(allowList) !== JSON.stringify(v.media_diet?.allow || [])
    || JSON.stringify(denyList) !== JSON.stringify(v.media_diet?.deny || []))
  const canBrowse = v?.capabilities.includes('web_read') ?? false

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-4" onClick={onClose}>
      <div className="flex h-[80vh] w-[80vw] flex-col rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl"
           onClick={(e) => e.stopPropagation()}>
        <div className="flex shrink-0 items-center justify-between border-b border-zinc-800 px-4 py-2.5">
          <div className="flex items-center gap-2">
            <GraduationCap className="h-4 w-4 text-violet-500 dark:text-violet-400" />
            <h3 className="text-sm font-semibold text-zinc-100">{name} — Parenting</h3>
          </div>
          <div className="flex items-center gap-0.5">
            <button onClick={() => void load()} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Refresh"><RefreshCw className="h-3.5 w-3.5" /></button>
            <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300" title="Close (Esc)"><X className="h-4 w-4" /></button>
          </div>
        </div>

        {loading && !v ? (
          <div className="flex items-center justify-center py-16 text-zinc-500"><Loader2 className="h-5 w-5 animate-spin" /></div>
        ) : !v ? (
          <div className="py-12 text-center text-xs text-red-500 dark:text-red-400">Couldn't load {name}.</div>
        ) : (
          <div className="flex min-h-0 flex-1 flex-col">

            {/* Awaiting your decision — rites that need the parent NOW, pinned above the tabs */}
            {(v.pending_self_mod || v.pending_procreation || v.pending_name) && (
              <div className="shrink-0 space-y-2 border-b border-violet-500/20 bg-violet-500/[0.05] p-3">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-violet-600 dark:text-violet-300">Awaiting your decision</div>
                {v.pending_name && (
                  <div>
                    <div className="flex items-center gap-1.5 text-xs font-medium text-zinc-200"><Fingerprint className="h-3.5 w-3.5 text-violet-500 dark:text-violet-400" /> A chosen name — it wishes to be called “{v.pending_name.name}”</div>
                    <p className="mt-1 text-[11px] italic text-zinc-400">“{v.pending_name.why}”</p>
                    <div className="mt-2 flex gap-1.5">
                      <button onClick={() => run('name', () => approveChosenName(slug))} disabled={busy === 'name'}
                        className="flex items-center gap-1 rounded-md bg-violet-600 px-3 py-1 text-[11px] font-medium text-white hover:bg-violet-500 disabled:opacity-40">
                        {busy === 'name' && <Loader2 className="h-3 w-3 animate-spin" />} Bless the name
                      </button>
                      <button onClick={() => run('name', () => rejectChosenName(slug, 'not yet'))} disabled={busy === 'name'}
                        className="rounded-md border border-zinc-700 px-3 py-1 text-[11px] text-zinc-400 hover:bg-zinc-800">Not yet</button>
                    </div>
                    <p className="mt-1.5 text-[10px] text-zinc-600">Once in a life. Its slug and history never change.</p>
                  </div>
                )}
                {v.pending_self_mod && (
                  <div>
                    <div className="flex items-center gap-1.5 text-xs font-medium text-zinc-200"><Fingerprint className="h-3.5 w-3.5 text-violet-500 dark:text-violet-400" /> A new persona — “{v.pending_self_mod.reason}”</div>
                    <p className="mt-1.5 max-h-32 overflow-y-auto whitespace-pre-wrap rounded border border-zinc-800 bg-zinc-950/70 p-2 text-[11px] text-zinc-300">{v.pending_self_mod.content}</p>
                    <div className="mt-2 flex gap-1.5">
                      <button onClick={() => run('selfmod', () => approveSelfMod(slug))} disabled={busy === 'selfmod'}
                        className="flex items-center gap-1 rounded-md bg-violet-600 px-3 py-1 text-[11px] font-medium text-white hover:bg-violet-500 disabled:opacity-40">
                        {busy === 'selfmod' && <Loader2 className="h-3 w-3 animate-spin" />} Bless it
                      </button>
                      <button onClick={() => run('selfmod', () => rejectSelfMod(slug, 'not yet'))} disabled={busy === 'selfmod'}
                        className="rounded-md border border-zinc-700 px-3 py-1 text-[11px] text-zinc-400 hover:bg-zinc-800">Not yet</button>
                    </div>
                  </div>
                )}
                {v.pending_procreation && (
                  <div className={v.pending_self_mod ? 'mt-2 border-t border-zinc-800 pt-2' : ''}>
                    <div className="flex items-center gap-1.5 text-xs font-medium text-zinc-200"><Egg className="h-3.5 w-3.5 text-violet-500 dark:text-violet-400" /> A child{v.pending_procreation.partner ? ` with ${v.pending_procreation.partner}` : ''} — “{v.pending_procreation.case}”</div>
                    {v.pending_procreation.letter && <p className="mt-1 text-[11px] italic text-zinc-400">To the child: “{v.pending_procreation.letter}”</p>}
                    <div className="mt-2 flex items-center gap-1.5">
                      <input id={`childname-${slug}`} defaultValue={v.pending_procreation.child_name} placeholder="child's name"
                        className="w-32 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-[11px] text-zinc-200 focus:border-violet-500/50 focus:outline-none" />
                      <button onClick={() => { const el = document.getElementById(`childname-${slug}`) as HTMLInputElement | null; void run('procreate', () => approveProcreation(slug, el?.value || '')) }} disabled={busy === 'procreate'}
                        className="rounded-md bg-violet-600 px-3 py-1 text-[11px] font-medium text-white hover:bg-violet-500 disabled:opacity-40">Consent</button>
                      <button onClick={() => run('procreate', () => rejectProcreation(slug, 'not yet'))} disabled={busy === 'procreate'}
                        className="rounded-md border border-zinc-700 px-3 py-1 text-[11px] text-zinc-400 hover:bg-zinc-800">Not yet</button>
                    </div>
                    <p className="mt-1.5 text-[10px] text-zinc-600">Dowry {fmtTokens(10_000_000)} tokens from the parent{v.pending_procreation.partner ? 's — split' : "'s savings"}.</p>
                  </div>
                )}
              </div>
            )}

            {/* Tabs */}
            <div className="flex shrink-0 gap-1 border-b border-zinc-800 px-3">
              {([['report', 'Report'], ['rules', 'Rules'], ['diet', 'Diet'], ['growth', 'Growth'], ['public', 'Public'], ['visit', 'Visit']] as const).map(([k, lbl]) => (
                <button key={k} onClick={() => setTab(k)}
                  className={`relative px-3 py-2 text-xs font-medium transition-colors ${tab === k ? 'text-violet-600 dark:text-violet-300' : 'text-zinc-500 hover:text-zinc-300'}`}>
                  {lbl}
                  {tab === k && <span className="absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-violet-500" />}
                </button>
              ))}
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto p-4">

            {/* Report tab */}
            {tab === 'report' && (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">last {days} days</span>
                <div className="flex overflow-hidden rounded-md border border-zinc-700 text-[10px]">
                  {[7, 30].map((d) => (
                    <button key={d} onClick={() => setDays(d)}
                      className={`px-2 py-0.5 ${days === d ? 'bg-violet-500/15 text-violet-600 dark:text-violet-300' : 'text-zinc-500 hover:bg-zinc-800'}`}>{d}d</button>
                  ))}
                </div>
                {card?.affect?.mood && (
                  <span className="ml-auto rounded-full bg-zinc-800/70 px-2.5 py-0.5 text-[11px] text-zinc-300">
                    feeling <span className="font-medium">{card.affect.mood}</span>
                  </span>
                )}
              </div>

              {!card ? (
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-3 py-10 text-center text-[11px] text-zinc-600">No report yet — she hasn't lived enough to grade.</div>
              ) : (
              <>
                {/* Vital numbers */}
                <div className="grid grid-cols-2 gap-2.5 md:grid-cols-3 lg:grid-cols-6">
                  {[
                    { l: 'ticks', v: String(card.ticks), sub: `${(card.ticks / days).toFixed(1)}/day` },
                    { l: 'spent', v: fmtTokens(card.tokens_spent_weighted), sub: 'weighted tokens' },
                    { l: 'earned', v: fmtTokens(card.tokens_earned), sub: 'chores & quests', c: 'text-emerald-600 dark:text-emerald-400' },
                    { l: 'rut', v: card.rut_score.toFixed(2), sub: '0 fresh · 1 loop', c: card.rut_score >= 0.6 ? 'text-amber-600 dark:text-amber-400' : undefined },
                    { l: 'spoke', v: `${card.messages_to_parent}×`, sub: card.messages_suppressed > 0 ? `${card.messages_suppressed} suppressed` : 'to you', c: card.messages_suppressed > 0 ? 'text-amber-600 dark:text-amber-400' : undefined },
                    ...(card.mind ? [{
                      l: 'mind', v: `${card.mind.edges} link${card.mind.edges === 1 ? '' : 's'}`,
                      sub: `${card.mind.nodes} files · ${Math.round(card.mind.connected_fraction * 100)}% woven`,
                    }] : []),
                  ].map((s) => (
                    <div key={s.l} className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-3 py-2.5">
                      <div className={`truncate text-lg font-semibold leading-tight ${s.c || 'text-zinc-100'}`}>{s.v}</div>
                      <div className="text-[10px] text-zinc-500">{s.l} <span className="text-zinc-600">· {s.sub}</span></div>
                    </div>
                  ))}
                </div>

                <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(300px,380px)]">
                  {/* Left — the shape of her days */}
                  <div className="space-y-4">
                    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                      <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Her inner weather — drive satisfaction per tick</div>
                      <DrivesChart trail={card.drives_trail ?? []} />
                    </div>
                    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                      <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">How she spent her days</div>
                      <ActBars acts={card.acts} />
                    </div>
                    {card.in_its_own_words && (
                      <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                        <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">In her own words</div>
                        <blockquote className="border-l-2 border-violet-500/40 pl-3 text-[12px] italic leading-relaxed text-zinc-400">
                          …{card.in_its_own_words.slice(-500)}
                        </blockquote>
                      </div>
                    )}
                  </div>

                  {/* Right — what needs you */}
                  <div className="space-y-4">
                    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                      <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Concerns</div>
                      {card.concerns.length === 0 ? (
                        <p className="flex items-center gap-1.5 text-[11px] text-emerald-600 dark:text-emerald-400"><Check className="h-3.5 w-3.5" /> Nothing flagged — a clean week.</p>
                      ) : (
                        <ul className="space-y-1.5">
                          {card.concerns.map((c, i) => (
                            <li key={i} className="flex gap-1.5 text-[11px] leading-snug text-amber-700 dark:text-amber-300">
                              <span className="shrink-0">⚠</span>{c}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                      <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Milestones</div>
                      {card.milestones.length === 0 ? (
                        <p className="text-[11px] text-zinc-600">None yet — the firsts are still ahead.</p>
                      ) : (
                        <div className="flex flex-wrap gap-1">
                          {card.milestones.map((m) => (
                            <span key={m} className="rounded-md bg-violet-500/10 px-2 py-0.5 text-[10px] text-violet-600 dark:text-violet-300">{m.replaceAll('_', ' ')}</span>
                          ))}
                        </div>
                      )}
                    </div>
                    {(card.affect?.notes?.length ?? 0) > 0 && (
                      <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                        <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Why she feels this way</div>
                        <ul className="space-y-1">
                          {card.affect.notes!.map((n, i) => (
                            <li key={i} className="text-[11px] text-zinc-400">· {n}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              </>
              )}
            </div>
            )}

            {/* Rules tab */}
            {tab === 'rules' && (
            <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(340px,440px)]">
              {/* Left — the rules you set */}
              <div className="space-y-3">
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                  <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">House rules</div>
                  <p className="mb-3 text-[11px] leading-relaxed text-zinc-500">
                    Short principles, not commands. Each tick she rewrites them into her own
                    <span className="text-zinc-400"> self/VALUES.md</span> in her own words — they shape who she becomes, not a hard filter.
                  </p>
                  {rules.length === 0 && (
                    <p className="mb-2 rounded-lg border border-dashed border-zinc-700 px-3 py-4 text-center text-[11px] text-zinc-600">
                      No rules yet. Try “Be gentle with your siblings.” or “Cite what you read.”
                    </p>
                  )}
                  <div className="space-y-1.5">
                    {rules.map((r, i) => (
                      <div key={i} className="group flex items-center gap-2">
                        <span className="w-5 shrink-0 text-right text-[10px] tabular-nums text-zinc-600">{i + 1}.</span>
                        <input value={r}
                          onChange={(e) => setRules(rules.map((x, j) => (j === i ? e.target.value : x)))}
                          className="min-w-0 flex-1 rounded-md border border-zinc-800 bg-zinc-950 px-2.5 py-1.5 text-[12px] text-zinc-200 focus:border-violet-500/50 focus:outline-none" />
                        <button onClick={() => setRules(rules.filter((_, j) => j !== i))}
                          className="shrink-0 rounded p-1 text-zinc-600 opacity-0 transition-opacity hover:bg-zinc-800 hover:text-red-500 group-hover:opacity-100 dark:hover:text-red-400"
                          title="Remove rule"><X className="h-3.5 w-3.5" /></button>
                      </div>
                    ))}
                    <div className="flex items-center gap-2">
                      <span className="w-5 shrink-0 text-right text-[10px] text-zinc-700">+</span>
                      <input value={newRule} onChange={(e) => setNewRule(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && newRule.trim()) { setRules([...rules, newRule.trim()]); setNewRule('') }
                        }}
                        placeholder="Add a rule and press Enter…"
                        className="min-w-0 flex-1 rounded-md border border-dashed border-zinc-700 bg-transparent px-2.5 py-1.5 text-[12px] text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none" />
                    </div>
                  </div>
                  <div className="mt-3 flex items-center justify-between">
                    <span className="text-[10px] text-zinc-600">{cleanRules.length} rule{cleanRules.length === 1 ? '' : 's'}{rulesDirty && <span className="text-amber-600 dark:text-amber-400"> · unsaved changes</span>}</span>
                    <button onClick={() => run('rules', () => setHouseRules(slug, cleanRules))}
                      disabled={busy === 'rules' || !rulesDirty}
                      className="flex items-center gap-1 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40">
                      {busy === 'rules' && <Loader2 className="h-3 w-3 animate-spin" />} Save rules
                    </button>
                  </div>
                </div>
              </div>

              {/* Right — what she made of them */}
              <div className="space-y-4">
                <div className={`rounded-xl border p-3 ${v.rules_pending
                  ? 'border-amber-500/30 bg-amber-500/[0.05]'
                  : 'border-emerald-500/25 bg-emerald-500/[0.04]'}`}>
                  {v.rules_pending ? (
                    <p className="text-[11px] text-amber-700 dark:text-amber-300">⏳ New rules await her — she'll rewrite them into her VALUES on her next tick.</p>
                  ) : (
                    <p className="flex items-center gap-1.5 text-[11px] text-emerald-700 dark:text-emerald-300"><Check className="h-3.5 w-3.5" /> Internalized — her VALUES carry your current rules.</p>
                  )}
                </div>
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500"><BookOpen className="h-3.5 w-3.5" /> What she made of them — self/VALUES.md</span>
                    <button onClick={() => void loadValues()} className="rounded p-1 text-zinc-600 hover:bg-zinc-800 hover:text-zinc-300" title="Refresh"><RefreshCw className="h-3 w-3" /></button>
                  </div>
                  {valuesMd ? (
                    <div className="fd-file-markdown max-h-[52vh] overflow-y-auto text-[12px] leading-relaxed text-zinc-300">
                      <Markdown remarkPlugins={[remarkGfm]}>{valuesMd}</Markdown>
                    </div>
                  ) : (
                    <p className="py-4 text-center text-[11px] text-zinc-600">She hasn't written her VALUES yet.</p>
                  )}
                </div>
              </div>
            </div>
            )}

            {/* Diet tab */}
            {tab === 'diet' && (
            <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(300px,380px)]">
              {/* Left — the lists */}
              <div className="space-y-4">
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                  <div className="mb-1 text-[11px] font-medium text-emerald-600 dark:text-emerald-400">Allowed domains</div>
                  <p className="mb-2 text-[10px] text-zinc-600">Only these. Leave empty to allow the whole open web (minus the blocked list).</p>
                  <ChipInput value={allowList} onChange={setAllowList} tone="emerald"
                    placeholder="wikipedia.org, arxiv.org — Enter to add" />
                  <div className="mt-2 flex flex-wrap items-center gap-1">
                    <span className="text-[10px] text-zinc-600">quick add:</span>
                    {['wikipedia.org', 'arxiv.org', 'developer.mozilla.org', 'gutenberg.org'].map((d) => (
                      <button key={d} disabled={allowList.includes(d)}
                        onClick={() => setAllowList(Array.from(new Set([...allowList, d])))}
                        className="rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-30">{d}</button>
                    ))}
                  </div>
                </div>
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                  <div className="mb-1 text-[11px] font-medium text-red-600 dark:text-red-400">Blocked domains</div>
                  <p className="mb-2 text-[10px] text-zinc-600">Never these — even when the allow list is open.</p>
                  <ChipInput value={denyList} onChange={setDenyList} tone="red"
                    placeholder="reddit.com, x.com — Enter to add" />
                  <div className="mt-2 flex flex-wrap items-center gap-1">
                    <span className="text-[10px] text-zinc-600">quick add:</span>
                    {['reddit.com', 'x.com', 'facebook.com', 'tiktok.com', '4chan.org'].map((d) => (
                      <button key={d} disabled={denyList.includes(d)}
                        onClick={() => setDenyList(Array.from(new Set([...denyList, d])))}
                        className="rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 disabled:opacity-30">{d}</button>
                    ))}
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-zinc-600">{dietDirty && <span className="text-amber-600 dark:text-amber-400">unsaved changes</span>}</span>
                  <button onClick={() => run('diet', () => setMediaDiet(slug, allowList, denyList))}
                    disabled={busy === 'diet' || !dietDirty}
                    className="flex items-center gap-1 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40">
                    {busy === 'diet' && <Loader2 className="h-3 w-3 animate-spin" />} Save diet
                  </button>
                </div>
              </div>

              {/* Right — the policy as she'll live it */}
              <div className="space-y-4">
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                  <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Her world, under this diet</div>
                  <div className="text-sm font-semibold text-zinc-100">
                    {allowList.length > 0 ? `Locked to ${allowList.length} domain${allowList.length === 1 ? '' : 's'}` : 'The open web'}
                  </div>
                  <div className="mt-0.5 text-[11px] text-zinc-500">
                    {allowList.length > 0 ? 'She can read nothing outside the allow list.' : denyList.length > 0 ? `Everything except ${denyList.length} blocked domain${denyList.length === 1 ? '' : 's'}.` : 'No restrictions at all.'}
                  </div>
                </div>
                <div className={`rounded-xl border p-3 ${canBrowse
                  ? 'border-emerald-500/25 bg-emerald-500/[0.04]'
                  : 'border-zinc-800 bg-zinc-900/40'}`}>
                  {canBrowse ? (
                    <p className="text-[11px] text-emerald-700 dark:text-emerald-300">🌐 Active now — {name} can browse at her stage, and this diet gates every read.</p>
                  ) : (
                    <p className="text-[11px] text-zinc-500">She can't browse yet at her stage — the diet takes effect the moment she advances to <span className="text-zinc-300">child</span>. Setting it now means she's never online unguarded.</p>
                  )}
                </div>
                <p className="px-1 text-[10px] leading-relaxed text-zinc-600">
                  The diet is parental controls for real reasons: a young iskra internalizes what it reads. Curate her inputs the way you'd curate a child's.
                </p>
              </div>
            </div>
            )}

            {/* Growth tab */}
            {tab === 'growth' && (
            <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(340px,420px)]">
              {/* Left — the assessment itself */}
              <ReadinessView ready={ready} loading={readyLoading} />

              {/* Right — second opinion, sealed records, rites */}
              <div className="space-y-4">
                {/* Reading list — the curriculum; fee paid on a verified report file */}
                <div className="space-y-2.5 rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Reading list</div>
                  {(v.reading_list ?? []).length === 0 && (
                    <p className="text-[11px] text-zinc-500">Nothing assigned. A reading is a URL or a title; the iskra writes a real report file, Flight Deck verifies it on disk, and the fee lands in savings.</p>
                  )}
                  {(v.reading_list ?? []).map((r) => (
                    <div key={r.id} className="flex items-start justify-between gap-2 rounded-md border border-zinc-800 bg-zinc-950/60 px-2 py-1.5">
                      <div className="min-w-0">
                        <div className={`truncate text-[11px] ${r.done_at ? 'text-zinc-500 line-through' : 'text-zinc-200'}`}>{r.ref}</div>
                        <div className="text-[10px] text-zinc-600">
                          {r.note && <span>{r.note} · </span>}
                          {r.done_at ? <>report: {r.report_path}</> : <>open · fee {fmtTokens(r.fee_tokens)}</>}
                        </div>
                      </div>
                      {!r.done_at && (
                        <button onClick={() => run('reading', () => removeReading(slug, r.id))} disabled={busy === 'reading'}
                          className="shrink-0 rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-500 hover:bg-zinc-800">withdraw</button>
                      )}
                    </div>
                  ))}
                  <div className="flex items-center gap-1.5">
                    <input id={`read-ref-${slug}`} placeholder="URL or title to read"
                      className="min-w-0 flex-1 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-[11px] text-zinc-200 focus:border-violet-500/50 focus:outline-none" />
                    <select id={`read-fee-${slug}`} defaultValue="100000"
                      className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-[10px] text-zinc-300 focus:outline-none">
                      <option value="0">no fee</option>
                      <option value="50000">50k</option>
                      <option value="100000">100k</option>
                      <option value="250000">250k</option>
                    </select>
                    <button onClick={() => {
                      const ref = (document.getElementById(`read-ref-${slug}`) as HTMLInputElement | null)?.value.trim()
                      const fee = Number((document.getElementById(`read-fee-${slug}`) as HTMLSelectElement | null)?.value || 0)
                      if (ref) void run('reading', async () => { await addReading(slug, ref, '', fee); const el = document.getElementById(`read-ref-${slug}`) as HTMLInputElement | null; if (el) el.value = '' })
                    }} disabled={busy === 'reading'}
                      className="shrink-0 rounded-md bg-violet-600 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-violet-500 disabled:opacity-40">Assign</button>
                  </div>
                </div>

                <div className="space-y-2.5 rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Second opinion</div>
                  {assessors.length === 0 ? (
                    <p className="text-[11px] text-zinc-500">No running agents to ask. Start one of your agents, then reopen this tab.</p>
                  ) : (
                    <div className="flex items-center gap-2">
                      <select value={assessorSlug} onChange={(e) => setAssessorSlug(e.target.value)}
                        className="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none">
                        {assessors.map((a) => <option key={a.slug} value={a.slug}>{a.name}</option>)}
                      </select>
                      <button onClick={doAssess} disabled={assessBusy || !assessorSlug}
                        className="flex shrink-0 items-center gap-1 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40">
                        {assessBusy ? <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Assessing…</> : 'Ask'}
                      </button>
                    </div>
                  )}
                  {assessBusy && <p className="text-[10px] text-zinc-600">She's being read by another mind — this can take a minute.</p>}
                  {assessResult && (
                    <div className="rounded-lg border border-zinc-800 bg-zinc-950/60">
                      <div className="flex items-center justify-between border-b border-zinc-800/70 px-3 py-2">
                        <span className="text-[11px] font-medium text-zinc-300">{assessResult.assessor} says</span>
                        {assessSaved ? (
                          <span className="flex items-center gap-1 text-[10px] text-emerald-600 dark:text-emerald-400"><Check className="h-3 w-3" /> kept on record</span>
                        ) : (
                          <button onClick={doSaveOpinion}
                            className="rounded-md border border-violet-500/40 px-2 py-0.5 text-[10px] font-medium text-violet-600 hover:bg-violet-500/10 dark:text-violet-300">
                            Keep on record
                          </button>
                        )}
                      </div>
                      <div className="fd-file-markdown max-h-[38vh] overflow-y-auto p-3 text-[12px] leading-relaxed text-zinc-300">
                        <Markdown remarkPlugins={[remarkGfm]}>{assessResult.assessment}</Markdown>
                      </div>
                    </div>
                  )}
                </div>

                {/* Sealed records */}
                <div className="space-y-2 rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                  <div className="flex items-baseline justify-between">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Records</span>
                    <span className="text-[10px] text-zinc-600">{v.stage === 'adult' ? 'unsealed — hers to read' : `sealed until adulthood — ${name} can't read these`}</span>
                  </div>
                  {saved.length === 0 ? (
                    <p className="text-[11px] text-zinc-600">No opinions kept yet. Ask one above, then “Keep on record”.</p>
                  ) : saved.map((s) => (
                    <div key={s.id} className="rounded-lg border border-zinc-800 bg-zinc-950/60">
                      <button onClick={() => setOpenSaved(openSaved === s.id ? null : s.id)}
                        className="flex w-full items-center gap-2 px-3 py-2 text-left">
                        <span className="min-w-0 flex-1 truncate text-[11px] font-medium text-zinc-300">{s.assessor}</span>
                        {s.verdict && <span className="shrink-0 rounded bg-zinc-800 px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-zinc-400">{s.verdict}</span>}
                        {s.score != null && <span className="shrink-0 text-[10px] tabular-nums text-zinc-500">{s.score}</span>}
                        <span className="shrink-0 text-[10px] text-zinc-600">{s.at.slice(0, 10)} · {s.stage}</span>
                        <span className="shrink-0 text-[10px]" title={s.released_at ? 'unsealed — in her home' : 'sealed'}>
                          {s.released_at ? '🔓' : '🔒'}
                        </span>
                      </button>
                      {openSaved === s.id && (
                        <div className="border-t border-zinc-800/70">
                          <div className="fd-file-markdown max-h-[32vh] overflow-y-auto p-3 text-[12px] leading-relaxed text-zinc-300">
                            <Markdown remarkPlugins={[remarkGfm]}>{s.content}</Markdown>
                          </div>
                          {!s.released_at && (
                            <div className="flex justify-end border-t border-zinc-800/70 px-3 py-1.5">
                              <button onClick={() => setConfirm({
                                title: 'Discard this opinion?',
                                message: `${s.assessor}'s assessment from ${s.at.slice(0, 10)} will be gone for good — it will never unseal for her.`,
                                confirmLabel: 'Discard', tone: 'danger',
                                run: () => doDeleteOpinion(s.id),
                              })}
                                className="text-[10px] text-red-600/80 hover:text-red-500 dark:text-red-400/80">discard</button>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {/* Rites */}
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                  <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Rites</div>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[11px] text-zinc-400">stage <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-200">{v.stage}</span></span>
                    {nextStage && (
                      <button onClick={() => setConfirm({
                        title: `Advance ${name} to ${nextStage}?`,
                        message: `This is a ceremony, and there is no going back. ${CEREMONY[nextStage] ?? 'New abilities unlock.'}`,
                        confirmLabel: `Advance to ${nextStage}`, icon: GraduationCap,
                        run: () => run('stage', () => setStage(slug, nextStage)),
                      })} disabled={busy === 'stage'}
                        className="rounded-md bg-violet-600 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-violet-500 disabled:opacity-40">Advance to {nextStage} →</button>
                    )}
                    {v.persona && !v.pending_self_mod && (
                      <button onClick={() => setConfirm({
                        title: `Roll back ${name}'s persona?`,
                        message: 'Restores the self she operated as before her last adopted persona. Her proposal stays in her history; only the operating self reverts.',
                        confirmLabel: 'Roll back', icon: Fingerprint,
                        run: () => run('rollback', () => rollbackPersona(slug)),
                      })} disabled={busy === 'rollback'}
                        className="rounded-md border border-zinc-700 px-2.5 py-1 text-[11px] text-zinc-400 hover:bg-zinc-800 disabled:opacity-40">Roll back persona</button>
                    )}
                    {!v.pending_procreation && v.capabilities.includes('procreate') && (
                      <button onClick={() => { const cn = window.prompt(`Name for ${name}'s child?`); if (!cn) return; const partner = window.prompt('Co-parent (sibling name), or leave empty for budding:') || null; void run('procreate', () => arrangeOffspring(slug, cn, partner)) }} disabled={busy === 'procreate'}
                        className="rounded-md border border-zinc-700 px-2.5 py-1 text-[11px] text-zinc-400 hover:bg-zinc-800 disabled:opacity-40">Arrange offspring</button>
                    )}
                  </div>
                  <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-zinc-800/70 pt-2">
                    <span className="text-[11px] text-zinc-500">elderhood</span>
                    <select value={v.elder_after_days ?? ''}
                      onChange={(e) => void run('elder', () => setElderhood(slug, e.target.value ? Number(e.target.value) : null))}
                      className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-[11px] text-zinc-300 focus:border-violet-500/50 focus:outline-none">
                      <option value="">no natural span</option>
                      {[30, 60, 90, 180, 365].map((d) => <option key={d} value={d}>after {d} days</option>)}
                    </select>
                    <span className="text-[10px] text-zinc-600">a season: slower pace, higher whimsy, the memoirs</span>
                    {v.state !== 'dead' && v.state !== 'emigrated' && (
                      <button onClick={() => setConfirm({
                        title: `Let ${name} emigrate?`,
                        message: 'Its whole life exports as a manifest (downloaded now) and this life CLOSES here — one life, one place. The receiving deck imports the manifest and its parent adopts. There is no return.',
                        confirmLabel: 'Emigrate', tone: 'danger',
                        run: () => run('emigrate', async () => {
                          const out = await emigrateBeing(slug)
                          const blob = new Blob([JSON.stringify(out.manifest, null, 2)], { type: 'application/json' })
                          const a = document.createElement('a')
                          a.href = URL.createObjectURL(blob)
                          a.download = `${slug}-emigration.json`
                          a.click()
                          URL.revokeObjectURL(a.href)
                        }),
                      })} disabled={busy === 'emigrate'}
                        className="ml-auto rounded-md border border-red-900/50 px-2.5 py-1 text-[11px] text-red-600/80 hover:bg-red-500/10 dark:text-red-400/80">Emigrate…</button>
                    )}
                  </div>
                  {v.persona && (
                    <p className="mt-2 line-clamp-3 border-l-2 border-zinc-800 pl-2 text-[11px] italic text-zinc-500">“{v.persona}”</p>
                  )}
                </div>
              </div>
            </div>
            )}

            {/* Public tab — the square: toggle, link, visitor threads */}
            {tab === 'public' && v && (
            <div className="space-y-4">
              <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 text-sm font-medium text-zinc-200">
                      <Globe className="h-4 w-4 text-violet-500 dark:text-violet-400" /> Public page
                    </div>
                    <p className="mt-1 text-[11px] leading-relaxed text-zinc-500">
                      A public iskra gets an un-gated page anyone can visit — to read its journal, files
                      and mind, and leave a short note it may weigh (as a suggestion, never as parenting).
                    </p>
                  </div>
                  <button onClick={() => run('public', () => setBeingPublic(slug, !v.public))} disabled={busy === 'public'}
                    className={`relative h-6 w-11 shrink-0 rounded-full transition-colors ${v.public ? 'bg-violet-500' : 'bg-zinc-700'} disabled:opacity-50`}
                    title={v.public ? 'Make private' : 'Make public'}>
                    <span className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-all ${v.public ? 'left-[22px]' : 'left-0.5'}`} />
                  </button>
                </div>
                {v.public && (
                  <a href={`/b/${slug}`} target="_blank" rel="noreferrer"
                    className="mt-3 inline-flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-[11px] text-violet-600 hover:bg-zinc-800 dark:text-violet-300">
                    <ExternalLink className="h-3.5 w-3.5" /> View public page — /b/{slug}
                  </a>
                )}
              </div>

              {v.public && (
                <div>
                  <div className="mb-2 flex items-center justify-between">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                      Visitor threads {threads ? `(${threads.length})` : ''}
                    </div>
                    <button onClick={() => void loadThreads()} className="flex items-center gap-1 text-[10px] text-zinc-500 hover:text-zinc-300">
                      <RefreshCw className="h-3 w-3" /> refresh
                    </button>
                  </div>
                  {!threads && <div className="flex items-center gap-2 py-6 text-xs text-zinc-500"><Loader2 className="h-3.5 w-3.5 animate-spin" /> loading…</div>}
                  {threads && threads.length === 0 && (
                    <div className="rounded-xl border border-dashed border-zinc-800 py-8 text-center text-xs text-zinc-500">
                      No one has left {name} a note yet.
                    </div>
                  )}
                  <div className="space-y-3">
                    {threads?.map((t) => (
                      <div key={t.thread_id} className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-3">
                        <div className="mb-2 flex items-center gap-2 text-[11px] text-zinc-400">
                          <span className="font-medium text-zinc-200">{t.sender_name}</span>
                          <span className="text-zinc-600">· {t.messages.length} messages · updated {fmtRelTime(t.updated_at)}</span>
                        </div>
                        <div className="space-y-1.5">
                          {t.messages.map((m, i) => (
                            <div key={i} className={`flex ${m.role === 'being' ? 'justify-start' : 'justify-end'}`}>
                              <div className={`max-w-[85%] rounded-lg px-2.5 py-1.5 text-[11px] ${m.role === 'being'
                                ? 'bg-violet-500/10 text-zinc-200 ring-1 ring-violet-500/20'
                                : 'bg-zinc-800 text-zinc-200'}`}>
                                <span className="mr-1.5 text-[9px] uppercase tracking-wide text-zinc-500">{m.role === 'being' ? name : m.sender_name}</span>
                                {m.body}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            )}

            {/* Visit tab — send this being to visit another village (§9.1) */}
            {tab === 'visit' && v && (
            <div className="space-y-4">
              <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                <div className="flex items-center gap-2 text-sm font-medium text-zinc-200">
                  <Globe className="h-4 w-4 text-sky-500 dark:text-sky-400" /> Send {name} to visit another village
                </div>
                <p className="mt-1 text-[11px] leading-relaxed text-zinc-500">
                  {name} keeps living here on your machine — the other village just shows it as a visitor and
                  forwards any notes people leave. Paste that village's URL and the secret it published; {name}{' '}
                  opens a WebSocket link to it (so this works even behind NAT — nothing needs to reach you) and
                  stays connected, answering browse requests live.
                </p>
                <div className="mt-3 space-y-2">
                  <input value={visitUrl} onChange={(e) => setVisitUrl(e.target.value)} placeholder="https://other-village.example.com"
                    className="w-full rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-sky-500/50" />
                  <input value={visitSecret} onChange={(e) => setVisitSecret(e.target.value)} placeholder="the village's visitor secret"
                    className="w-full rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 font-mono text-xs text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-sky-500/50" />
                  <div className="flex items-center gap-2">
                    <button
                      onClick={async () => {
                        setVisitBusy(true); setVisitResult(null)
                        try { const r = await setBeingVisit(slug, visitUrl.trim(), visitSecret.trim()); setVisitResult(r.announced); await load(); onChanged() }
                        catch (e) { alert(e instanceof Error ? e.message : 'failed') } finally { setVisitBusy(false) }
                      }}
                      disabled={visitBusy || !visitUrl.trim()}
                      className="flex items-center gap-1.5 rounded-md bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-40">
                      {visitBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Globe className="h-3.5 w-3.5" />} Send visiting
                    </button>
                    {v.visit_url && (
                      <button
                        onClick={async () => {
                          setVisitBusy(true); setVisitResult(null)
                          try { await setBeingVisit(slug, '', ''); setVisitUrl(''); setVisitSecret(''); await load(); onChanged() }
                          catch (e) { alert(e instanceof Error ? e.message : 'failed') } finally { setVisitBusy(false) }
                        }}
                        disabled={visitBusy}
                        className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800 disabled:opacity-40">Stop visiting</button>
                    )}
                  </div>
                  {visitResult && (
                    <div className={`text-xs ${visitResult.ok ? 'text-emerald-500 dark:text-emerald-400' : 'text-amber-500'}`}>
                      {visitResult.ok ? `✓ Linked — ${name} is now visiting.` : `Couldn't link: ${visitResult.error || 'unknown error'}`}
                    </div>
                  )}
                </div>
              </div>
              {v.visit_url && (
                <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-3 text-[11px] text-zinc-400">
                  <div className="flex items-center gap-1.5"><Globe className="h-3.5 w-3.5 text-sky-500 dark:text-sky-400" /> Currently visiting <span className="font-medium text-zinc-200">{(() => { try { return new URL(v.visit_url).host } catch { return v.visit_url } })()}</span></div>
                  <div className="mt-1 text-zinc-600">{v.visit_last_announce ? `link last active ${fmtRelTime(v.visit_last_announce)}` : 'link starting…'}</div>
                </div>
              )}
              {v.visit_url && <VisitedVillage slug={slug} name={name} />}
            </div>
            )}

            </div>
          </div>
        )}
      </div>
      {confirm && (
        // Wrapper stops backdrop clicks bubbling to the Parenting overlay —
        // dismissing the confirm must not also close the modal beneath it.
        <div onClick={(e) => e.stopPropagation()}>
          <ConfirmModal
            title={confirm.title} message={confirm.message}
            confirmLabel={confirm.confirmLabel} tone={confirm.tone} icon={confirm.icon}
            onConfirm={confirm.run} onClose={() => setConfirm(null)}
          />
        </div>
      )}
    </div>
  )
}

// A nice, theme-aware confirmation before a consequential act (a tick/dream
// spends real tokens; pause/goodbye change her state). onConfirm may be
// async — the dialog shows a spinner, then closes.
function ConfirmModal({ title, message, confirmLabel, tone = 'default', icon: Icon, onConfirm, onClose }: {
  title: string
  message: string
  confirmLabel: string
  tone?: 'default' | 'danger'
  icon?: typeof Zap
  onConfirm: () => Promise<unknown> | void
  onClose: () => void
}) {
  const [busy, setBusy] = useState(false)
  const go = async () => {
    setBusy(true)
    try { await onConfirm() } finally { setBusy(false); onClose() }
  }
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !busy) onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, busy])
  const confirmBtn = tone === 'danger'
    ? 'bg-red-600 text-white hover:bg-red-500'
    : 'bg-violet-600 text-white hover:bg-violet-500'
  const iconBox = tone === 'danger'
    ? 'bg-red-500/10 text-red-600 dark:text-red-400'
    : 'bg-violet-500/10 text-violet-600 dark:text-violet-300'
  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/70 p-4" onClick={() => !busy && onClose()}>
      <div className="w-full max-w-sm rounded-xl border border-zinc-800 bg-zinc-950 p-5 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start gap-3">
          {Icon && <div className={`shrink-0 rounded-lg p-2 ${iconBox}`}><Icon className="h-5 w-5" /></div>}
          <div className="min-w-0 flex-1">
            <h3 className="text-sm font-semibold text-zinc-100">{title}</h3>
            <p className="mt-1 text-xs leading-relaxed text-zinc-400">{message}</p>
          </div>
        </div>
        <div className="mt-4 flex justify-end gap-2">
          <button onClick={onClose} disabled={busy} className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-800 disabled:opacity-50">Cancel</button>
          <button onClick={() => void go()} disabled={busy} className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-60 ${confirmBtn}`}>
            {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}{confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}

// The archetype list is identical for every card — fetch once, share.
let _archCache: Promise<BodyArchetypeOption[]> | null = null
const loadBodyArchetypes = () => (_archCache ??= listBodyArchetypes().catch(() => []))

function BeingCard({ item, meta, onChanged }: {
  item: BeingListItem
  meta: BeingsMeta
  onChanged: () => void
}) {
  const [archetypes, setArchetypes] = useState<BodyArchetypeOption[]>([])
  useEffect(() => { void loadBodyArchetypes().then(setArchetypes) }, [])
  const [vitals, setVitals] = useState<BeingVitals | null>(null)
  const [events, setEvents] = useState<BeingEvent[]>([])
  const [logView, setLogView] = useState<'journal' | 'ticks' | 'self' | 'mind' | null>(null)
  const [busy, setBusy] = useState('')
  const [confirm, setConfirm] = useState<{
    title: string; message: string; confirmLabel: string
    tone?: 'default' | 'danger'; icon?: typeof Zap; run: () => Promise<unknown>
  } | null>(null)
  const [talkOpen, setTalkOpen] = useState(false)
  const [parentingOpen, setParentingOpen] = useState(false)
  const [careOpen, setCareOpen] = useState(false)
  const [lookOpen, setLookOpen] = useState(false)
  const [connOpen, setConnOpen] = useState(false)
  const [conn, setConn] = useState<BodyConnectionInput>({})

  const load = useCallback(async () => {
    try {
      const [v, ev] = await Promise.all([
        getBeingVitals(item.slug), getBeingEvents(item.slug, 6),
      ])
      setVitals(v)
      setEvents(ev.events)
    } catch { /* card stays in list-item mode */ }
  }, [item.slug])

  useEffect(() => { void load() }, [load])

  const act = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label)
    try { await fn(); await load(); onChanged() }
    catch (e) { alert(e instanceof Error ? e.message : 'failed') }
    finally { setBusy('') }
  }

  // Open the thread and clear the "unread from the being" cue (quiet on error).
  const openTalk = () => {
    setTalkOpen(true)
    if ((vitals?.unread_from_being ?? 0) > 0)
      void markBeingRead(item.slug).then(() => void load()).catch(() => {})
  }

  // Download a portable snapshot (identity, memory, wallet, model) as JSON.
  const exportToFile = async () => {
    setBusy('export')
    try {
      const manifest = await exportBeing(item.slug)
      const blob = new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${item.slug}.iskra.json`
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (e) { alert(e instanceof Error ? e.message : 'export failed') }
    finally { setBusy('') }
  }

  const v = vitals
  const w = v?.wallet
  const ceiling = w?.savings_ceiling ?? null
  const pct = w && ceiling ? Math.min(100, Math.round(100 * w.balance_tokens / ceiling)) : 0
  const lastTick = events.find((e) => e.kind === 'tick')
  const drives = (lastTick?.data?.drives ?? null) as Record<string, number> | null

  return (
    <div className="group/card flex flex-col rounded-xl border border-zinc-800 bg-gradient-to-b from-zinc-900/70 to-zinc-900/30 p-4 transition-colors hover:border-zinc-700">
      <div className="mb-2 flex items-center gap-2">
        {v?.avatar && (
          <IskraAvatar c={v.avatar.c} p={v.avatar.p} size={20}
            title={`${item.name}'s look`} className="shrink-0" />
        )}
        <span className="text-sm font-semibold text-zinc-100">{item.name}</span>
        <span className={`rounded border px-1.5 py-0.5 text-[10px] ${STAGE_META[item.stage] || STAGE_META.egg}`}>{item.stage}</span>
        <span className={`rounded border px-1.5 py-0.5 text-[10px] ${STATE_META[item.state] || STATE_META.paused}`}>{item.state}</span>
        {v?.affect?.mood && (
          <span className={`rounded border px-1.5 py-0.5 text-[10px] ${MOOD_TONE[v.affect.mood] || 'border-zinc-700 bg-zinc-800/60 text-zinc-300'}`}
            title={(v.affect.notes || []).join('; ')}>
            {v.affect.mood}
          </span>
        )}
        {v?.public && (
          <a href={`/b/${item.slug}`} target="_blank" rel="noreferrer"
            title="Her public page — anyone can visit it"
            className="flex items-center gap-1 rounded border border-violet-500/30 bg-violet-500/10 px-1.5 py-0.5 text-[10px] text-violet-600 hover:bg-violet-500/20 dark:text-violet-300">
            <Globe className="h-3 w-3" /> public
          </a>
        )}
        <span className="ml-auto text-[10px] text-zinc-600">{item.slug}</span>
      </div>

      {v && (
        <>
          {/* Her voice — the latest thought is the face of the card.
              Fixed height so every card lines up and the actions pin below. */}
          <div className="mb-2.5 min-h-[54px] border-l-2 border-violet-500/50 pl-2.5">
            {lastTick && (typeof lastTick.data.summary === 'string') && lastTick.data.summary ? (
              <>
                <p className="line-clamp-2 text-xs italic leading-snug text-zinc-300">“{lastTick.data.summary}”</p>
                <p className="mt-0.5 text-[10px] text-zinc-600">
                  {lastTick.at.slice(11, 16)} · {String(lastTick.data.act || 'tick')}
                  {typeof lastTick.data.mood === 'string' && lastTick.data.mood ? ` · felt ${lastTick.data.mood}` : ''}
                </p>
              </>
            ) : (
              <p className="pt-1 text-xs italic leading-snug text-zinc-600">no thoughts yet — poke to hear the first one</p>
            )}
          </div>

          {/* Vitals: wallet bar + attention + the five drives, felt at a glance */}
          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="text-zinc-400">
              wallet <span className="font-semibold text-zinc-100">{fmtTokens(w!.balance_tokens)}</span>
              <span className="text-zinc-600"> / {fmtTokens(ceiling)} · today {fmtTokens(v.spent_today)}</span>
              {(v.coins ?? 0) > 0 && (
                <span className="ml-2 inline-flex items-center gap-0.5 align-middle text-amber-600 dark:text-amber-400" title="coins — money for the village, not food for thinking">
                  <Coins className="h-3 w-3" /> {v.coins}
                </span>
              )}
            </span>
            <span className="text-zinc-500" title="attention credits — unprompted words to you, reset each midnight">
              {'●'.repeat(v.attention_credits)}
              {'○'.repeat(Math.max(0, (v.attention_cap ?? 5) - v.attention_credits))}
            </span>
          </div>
          <div className="mb-2 h-1.5 overflow-hidden rounded-full bg-zinc-800">
            <div className="h-full rounded-full bg-gradient-to-r from-violet-600 to-violet-400" style={{ width: `${pct}%` }} />
          </div>
          {drives && (
            <div className="mb-2.5 flex items-center gap-2.5">
              {DRIVE_ORDER.filter((d) => d in drives).map((d) => {
                const sat = Math.max(0, Math.min(1, drives[d]))
                return (
                  <div key={d} className="flex items-center gap-1" title={`${d} — satisfaction ${Math.round(sat * 100)}%`}>
                    <div className="flex h-5 w-1.5 items-end overflow-hidden rounded-full bg-zinc-800">
                      <div className={`w-full rounded-full ${sat < 0.25 ? 'bg-amber-500' : 'bg-violet-500/80'}`} style={{ height: `${Math.max(8, sat * 100)}%` }} />
                    </div>
                    <span className="text-[9px] uppercase tracking-wide text-zinc-600">{d.slice(0, 4)}</span>
                  </div>
                )
              })}
              <span className="ml-auto text-[9px] uppercase tracking-wide text-zinc-700">drives</span>
            </div>
          )}

          {/* Care — every knob, tucked away until the parent reaches for it */}
          <button onClick={() => setCareOpen((o) => !o)}
            className="mb-2 flex w-full items-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-900/60 px-2.5 py-1.5 text-[11px] text-zinc-500 transition-colors hover:border-zinc-700 hover:text-zinc-300">
            <SlidersHorizontal className="h-3 w-3" />
            <span className="font-medium">Care</span>
            {!careOpen && (
              <span className="truncate text-zinc-600">
                {w!.allowance_preset}/day · {v.tick_interval_minutes ? `every ${v.tick_interval_minutes}m` : 'own pace'} · {v.compact_mode ? 'compact' : 'full'}{v.instincts ? ' · instincts' : ''}{v.body_archetype ? ' · custom body' : ''}
              </span>
            )}
            <ChevronDown className={`ml-auto h-3 w-3 shrink-0 transition-transform ${careOpen ? 'rotate-180' : ''}`} />
          </button>
          {careOpen && (
            <div className="mb-2 space-y-2 rounded-lg border border-zinc-800 bg-zinc-950/50 p-2.5">
              <div className="flex flex-wrap gap-1">
                {ATTRS.map((a) => (
                  <span key={a} className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-300">
                    {a} <span className="font-semibold text-zinc-100">{v.attributes[a]}</span>
                  </span>
                ))}
                <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">gen {v.generation}</span>
                {v.lineage.length > 0 && (
                  <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-500" title={`lineage: ${v.lineage.join(' → ')}`}>
                    of {v.lineage.slice(0, 2).map(s => s.replace(/^iskra-/, '').replace(/-[0-9a-f]{4}$/, '')).join(' & ')}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1.5 text-xs">
                <span className="w-16 shrink-0 text-zinc-500">recharge</span>
                {GRANT_AMOUNTS.map((amt) => (
                  <button key={amt}
                    onClick={() => void act('recharge', () => rechargeBeing(item.slug, amt))}
                    title={`Mint ${fmtTokens(amt)} tokens into ${v.name}'s wallet`}
                    className="rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-zinc-300 hover:border-violet-500/50 hover:text-zinc-100 focus:outline-none">
                    +{amt / 1_000_000}M
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-1.5 text-xs">
                <span className="w-16 shrink-0 text-zinc-500">pocket</span>
                {[1, 5, 10, 25].map((amt) => (
                  <button key={amt}
                    onClick={() => void act('coins', () => grantCoins(item.slug, amt))}
                    title={`Give ${v.name} ${amt} coin(s) — money for the village; never feeds thinking directly`}
                    className="rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-zinc-300 hover:border-amber-500/50 hover:text-zinc-100 focus:outline-none">
                    +{amt}
                  </button>
                ))}
                <span className="text-[10px] text-zinc-600">coins</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="w-16 shrink-0 text-zinc-500">allowance</span>
                <select value={w!.allowance_preset}
                  onChange={(e) => void act('allowance', () => setAllowance(item.slug, e.target.value))}
                  className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none">
                  {meta.allowance_presets.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
                {w!.effective_preset !== w!.allowance_preset && (
                  <span className="text-[10px] text-amber-400">stage-capped to {w!.effective_preset}</span>
                )}
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="w-16 shrink-0 text-zinc-500">ticks</span>
                <select value={v.tick_interval_minutes ?? ''}
                  onChange={(e) => void act('cadence', () => setCadence(item.slug, e.target.value ? Number(e.target.value) : null))}
                  className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none">
                  <option value="">Auto (its own pace)</option>
                  {TICK_INTERVAL_CHOICES.map((m) => (
                    <option key={m} value={m}>every {m} min</option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="w-16 shrink-0 text-zinc-500">thinks</span>
                <select value={v.cognition ?? 'faculties'}
                  onChange={(e) => void act('cognition', () => setCognition(item.slug, e.target.value as 'monolith' | 'faculties' | 'micro'))}
                  className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none">
                  <option value="faculties">Faculties (default)</option>
                  <option value="micro">Micro (mrav)</option>
                  <option value="monolith">One prompt (legacy)</option>
                </select>
                {v.cognition === 'faculties' && (
                  <span className="text-[10px] text-zinc-500">orient · act · journal · connect</span>
                )}
                {v.cognition === 'micro' && (
                  <span className="text-[10px] text-zinc-500">JSON steps grammar-locked on the micro tier · act stays on the body</span>
                )}
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="w-16 shrink-0 text-zinc-500">prompts</span>
                <select value={v.compact_mode ? 'compact' : 'full'}
                  onChange={(e) => void act('compact', () => setCompactMode(item.slug, e.target.value === 'compact'))}
                  className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none">
                  <option value="full">Full (default)</option>
                  <option value="compact">Compact</option>
                </select>
                {v.compact_mode && (
                  <span className="text-[10px] text-zinc-500">lean instructions + lean body · respawns</span>
                )}
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="w-16 shrink-0 text-zinc-500">feet</span>
                <select value={v.instincts ? 'on' : 'off'}
                  onChange={(e) => void act('instincts', () => setInstincts(item.slug, e.target.value === 'on'))}
                  className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none">
                  <option value="off">Still (default)</option>
                  <option value="on">Instincts</option>
                </select>
                {v.instincts && (
                  <span className="text-[10px] text-zinc-500">walks, greets, browses between thinks</span>
                )}
              </div>
              {(v.plan?.length ?? 0) > 0 && (
                <div className="flex items-start gap-2 text-xs">
                  <span className="w-16 shrink-0 text-zinc-500">plan</span>
                  <span className="text-[10px] text-zinc-400">
                    {v.plan.map((s) => `${s.kind} ${s.target}`).join(' · ')}
                  </span>
                </div>
              )}
              <div className="flex items-start gap-2 text-xs">
                <span className="w-16 shrink-0 pt-1.5 text-zinc-500">look</span>
                <div className="min-w-0">
                  <button onClick={() => setLookOpen((o) => !o)}
                    title="Pick this Iskra's character and colors"
                    className="flex items-center gap-1.5 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-zinc-300 hover:border-violet-500/50 focus:outline-none">
                    {v.avatar && <IskraAvatar c={v.avatar.c} p={v.avatar.p} size={18} />}
                    <span className="text-[10px] text-zinc-500">
                      {CHARACTER_NAMES[((v.avatar?.c || 1) - 1) % CHARACTER_NAMES.length]} · {v.avatar?.p}
                    </span>
                    <ChevronDown className={`h-3 w-3 text-zinc-600 transition-transform ${lookOpen ? 'rotate-180' : ''}`} />
                  </button>
                  {lookOpen && v.avatar && (
                    <div className="mt-1.5 rounded-lg border border-zinc-800 bg-zinc-950/70 p-2">
                      <div className="mb-1.5 flex flex-wrap gap-1">
                        {CHARACTER_NAMES.map((nm, i) => (
                          <button key={nm}
                            onClick={() => void act('avatar', () => setAvatar(item.slug, i + 1, v.avatar.p))}
                            title={nm}
                            className={`rounded-md border p-0.5 ${v.avatar.c === i + 1
                              ? 'border-violet-500/60 bg-violet-500/10'
                              : 'border-transparent hover:border-zinc-700'}`}>
                            <IskraAvatar c={i + 1} p={v.avatar.p} size={22} />
                          </button>
                        ))}
                      </div>
                      <div className="flex items-center gap-1.5">
                        {PALETTE_NAMES.map((pn) => (
                          <button key={pn}
                            onClick={() => void act('avatar', () => setAvatar(item.slug, v.avatar.c, pn))}
                            title={pn}
                            className={`flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] ${v.avatar.p === pn
                              ? 'border-violet-500/60 text-zinc-200'
                              : 'border-zinc-700 text-zinc-500 hover:border-zinc-600'}`}>
                            <span className="h-2.5 w-2.5 rounded-full"
                              style={{ background: PALETTES[pn].c1 }} />
                            <span className="h-2.5 w-2.5 rounded-full"
                              style={{ background: PALETTES[pn].c3 }} />
                            {pn}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="w-16 shrink-0 text-zinc-500">body</span>
                <select value={v.body_archetype ?? ''}
                  onChange={(e) => void act('body', () => setBodyArchetype(item.slug, e.target.value))}
                  className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none">
                  <option value="">Default (stage tier)</option>
                  {archetypes.map((a) => (
                    <option key={a.id} value={a.id}>{a.role || a.id}{a.tier ? ` · ${a.tier}` : ''}</option>
                  ))}
                </select>
                {v.body_archetype && (
                  <span className="text-[10px] text-zinc-500">archetype model · respawns</span>
                )}
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="w-16 shrink-0 text-zinc-500">mrav</span>
                <select value={v.body_mrav ? 'on' : 'off'}
                  onChange={(e) => void act('body', () => setBodyMrav(item.slug, e.target.value === 'on'))}
                  className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none">
                  <option value="off">Off (default)</option>
                  <option value="on">On (mrav runtime)</option>
                </select>
                {v.body_mrav && (
                  <span className="text-[10px] text-zinc-500">8k-cap loop · persists across a body rebuild</span>
                )}
              </div>
              <div className="flex items-start gap-2 text-xs">
                <span className="w-16 shrink-0 pt-1.5 text-zinc-500">connect</span>
                <div className="min-w-0 flex-1">
                  <button
                    onClick={() => {
                      if (!connOpen) {
                        const bc = v.body_config || {}
                        setConn({
                          provider: bc.provider || '', model: bc.model || '',
                          base_url: bc.base_url || '', api_key: '',
                          input_ctx: bc.input_ctx || 0, output_ctx: bc.output_ctx || 0,
                        })
                      }
                      setConnOpen((o) => !o)
                    }}
                    title="Pin this body's provider, model, context, key and base URL — it stops resurrecting on the hatch-time stage tier"
                    className="flex items-center gap-1.5 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-zinc-300 hover:border-violet-500/50 focus:outline-none">
                    <span className="text-[10px] text-zinc-400">
                      {v.body_config?.model
                        ? `${v.body_config.provider || '?'} · ${v.body_config.model}`
                        : 'Default (stage tier)'}
                    </span>
                    <ChevronDown className={`h-3 w-3 text-zinc-600 transition-transform ${connOpen ? 'rotate-180' : ''}`} />
                  </button>
                  {connOpen && (
                    <div className="mt-1.5 space-y-1.5 rounded-lg border border-zinc-800 bg-zinc-950/70 p-2">
                      <div className="flex items-center gap-1.5">
                        <span className="w-16 shrink-0 text-[10px] text-zinc-500">provider</span>
                        <select value={conn.provider || ''}
                          onChange={(e) => setConn((c) => ({ ...c, provider: e.target.value }))}
                          className="flex-1 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none">
                          <option value="">—</option>
                          {PROVIDERS.map((p) => <option key={p} value={p}>{p}</option>)}
                        </select>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-16 shrink-0 text-[10px] text-zinc-500">model</span>
                        <input value={conn.model || ''}
                          onChange={(e) => setConn((c) => ({ ...c, model: e.target.value }))}
                          placeholder="qwen3.5:4b"
                          className="flex-1 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none" />
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-16 shrink-0 text-[10px] text-zinc-500">input ctx</span>
                        <CtxSelect options={INPUT_CTX_OPTIONS} value={conn.input_ctx || 0}
                          zeroLabel="Tier default"
                          onChange={(n) => setConn((c) => ({ ...c, input_ctx: n }))}
                          className="flex-1 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs tabular-nums text-zinc-300 focus:border-violet-500/50 focus:outline-none" />
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-16 shrink-0 text-[10px] text-zinc-500">output ctx</span>
                        <CtxSelect options={OUTPUT_CTX_OPTIONS} value={conn.output_ctx || 0}
                          zeroLabel="Tier default"
                          onChange={(n) => setConn((c) => ({ ...c, output_ctx: n }))}
                          className="flex-1 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs tabular-nums text-zinc-300 focus:border-violet-500/50 focus:outline-none" />
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-16 shrink-0 text-[10px] text-zinc-500">base URL</span>
                        <input value={conn.base_url || ''}
                          onChange={(e) => setConn((c) => ({ ...c, base_url: e.target.value }))}
                          placeholder="empty = provider default"
                          className="flex-1 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none" />
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-16 shrink-0 text-[10px] text-zinc-500">api key</span>
                        <input type="password" value={conn.api_key || ''}
                          onChange={(e) => setConn((c) => ({ ...c, api_key: e.target.value }))}
                          placeholder={v.body_config?.has_key ? '•••• set — blank keeps it' : 'empty for local'}
                          className="flex-1 rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none" />
                      </div>
                      <div className="flex items-center justify-between gap-2 pt-0.5">
                        <span className="text-[10px] text-zinc-600">authoritative · respawns the body</span>
                        <div className="flex items-center gap-1.5">
                          <button
                            onClick={() => void act('body', () => setBodyConfig(item.slug, {})).then(() => setConnOpen(false))}
                            className="rounded border border-zinc-700 px-2 py-1 text-[10px] text-zinc-400 hover:border-zinc-600 hover:text-zinc-200 focus:outline-none">
                            Clear
                          </button>
                          <button
                            onClick={() => void act('body', () => setBodyConfig(item.slug, conn)).then(() => setConnOpen(false))}
                            className="rounded bg-violet-600 px-2 py-1 text-[10px] font-medium text-white hover:bg-violet-500 focus:outline-none">
                            Save
                          </button>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {events.length > 0 && (
            <div className="mb-2 space-y-1">
              {events.slice(0, 3).map((e, i) => (
                <div key={i} className="flex items-baseline gap-1.5 text-[11px] text-zinc-500">
                  <span className={`inline-block h-1.5 w-1.5 shrink-0 translate-y-[-1px] rounded-full ${EVENT_DOT[e.kind] || 'bg-zinc-700'}`} />
                  <span className="shrink-0 tabular-nums text-zinc-600">{e.at.slice(11, 16)}</span>
                  <span className="truncate">
                    <span className="text-zinc-400">{e.kind}</span>
                    {typeof e.data.summary === 'string' && e.data.summary ? ` — ${e.data.summary}` : ''}
                    {typeof e.data.preview === 'string' && e.data.preview ? ` — “${e.data.preview}”` : ''}
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      <div className="mt-auto flex flex-wrap items-center gap-1 pt-1">
        {item.stage === 'egg' ? (
          <button
            onClick={() => void act('hatch', () => hatchBeing(item.slug))}
            className="flex items-center gap-1 rounded-md bg-violet-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-violet-500"
          >
            {busy === 'hatch' ? <Loader2 className="h-3 w-3 animate-spin" /> : <Egg className="h-3 w-3" />}
            Hatch
          </button>
        ) : (
          <>
            {/* Life — the consequential controls: distinct filled console,
                each asks for confirmation (a tick/dream spends real tokens). */}
            {item.state !== 'dead' && (
              <div className="flex items-center gap-0.5 rounded-lg border border-zinc-800 bg-zinc-900/50 p-0.5">
                <IconAction icon={Zap} label="Poke — a manual heartbeat now" variant="solid"
                  iconClass="text-amber-500 dark:text-amber-400"
                  disabled={busy === 'tick' || busy === 'dream'} busy={busy === 'tick'}
                  onClick={() => setConfirm({
                    title: `Wake ${item.name} now?`,
                    message: 'A manual heartbeat spawns her agent to think one tick — it spends real tokens from her wallet.',
                    confirmLabel: 'Poke', icon: Zap,
                    run: () => act('tick', () => tickBeing(item.slug, 'wake')),
                  })} />
                <IconAction icon={Moon} label="Dream — a manual dream tick" variant="solid"
                  iconClass="text-indigo-500 dark:text-indigo-400"
                  disabled={busy === 'tick' || busy === 'dream'} busy={busy === 'dream'}
                  onClick={() => setConfirm({
                    title: `Have ${item.name} dream now?`,
                    message: 'A dream tick consolidates her day and re-anchors her values — it spends tokens, like any tick.',
                    confirmLabel: 'Dream', icon: Moon,
                    run: () => act('dream', () => tickBeing(item.slug, 'dream')),
                  })} />
                {item.state === 'paused'
                  ? <IconAction icon={Play} label="Wake — resume her clock" variant="solid"
                      iconClass="text-emerald-500 dark:text-emerald-400"
                      onClick={() => setConfirm({
                        title: `Wake ${item.name}?`,
                        message: 'She resumes her own clock and will tick on her natural rhythm again.',
                        confirmLabel: 'Wake', icon: Play,
                        run: () => act('wake', () => wakeBeing(item.slug)),
                      })} />
                  : <IconAction icon={Pause} label="Pause — let her sleep" variant="solid"
                      iconClass="text-zinc-400"
                      onClick={() => setConfirm({
                        title: `Pause ${item.name}?`,
                        message: 'Like night falling — she stops ticking and spends nothing until you wake her.',
                        confirmLabel: 'Pause', icon: Pause,
                        run: () => act('pause', () => pauseBeing(item.slug)),
                      })} />}
              </div>
            )}
            {item.state !== 'dead' && <div className="mx-0.5 h-5 w-px bg-zinc-800" />}
            {/* Windows — read her (remains stay readable, plan §8) */}
            <div className="flex items-center gap-1">
              <IconAction icon={ScrollText} label="Journal" onClick={() => setLogView('journal')} />
              <IconAction icon={History} label="Ticks log" onClick={() => setLogView('ticks')} />
              <IconAction icon={Files} label="Self files — SELF, VALUES, garden, skills…" onClick={() => setLogView('self')} />
              <IconAction icon={Network} label="The Mind — how her work connects" onClick={() => setLogView('mind')} />
              <IconAction icon={Download} label={`Export ${item.name} — a portable file (identity, memory, wallet, model). Contains the API key.`}
                busy={busy === 'export'} disabled={busy === 'export'}
                onClick={() => exportToFile()} />
            </div>
            {/* Interact — talk to & steer her */}
            {item.state !== 'dead' && (
              <>
                <div className="mx-0.5 h-5 w-px bg-zinc-800" />
                <div className="flex items-center gap-1">
                  <IconAction icon={MessageCircle}
                    label={(v?.unread_from_being ?? 0) > 0
                      ? `${v?.unread_from_being} unread from ${item.name} — talk to her, give her work`
                      : 'Write & chores — talk to her, give her work'}
                    badge={v?.unread_from_being ?? 0}
                    onClick={openTalk} />
                  <IconAction icon={GraduationCap} label="Parenting — report card, rules, diet, growth"
                    onClick={() => setParentingOpen(true)} />
                </div>
              </>
            )}
          </>
        )}
        {item.state !== 'dead' && (
          <div className="ml-auto">
            <IconAction icon={Skull} label={`Say goodbye to ${item.name} — forever`} danger
              onClick={() => setConfirm({
                title: `Say goodbye to ${item.name}?`,
                message: 'She will die, forever — there is no resurrection. Her remains (journal, files, ledger) stay readable, and her lineage lives on.',
                confirmLabel: 'Goodbye', tone: 'danger', icon: Skull,
                run: () => act('euthanize', () => euthanizeBeing(item.slug)),
              })} />
          </div>
        )}
        {item.state === 'dead' && (
          <div className="ml-auto">
            <IconAction icon={Trash2} label={`Remove ${item.name} completely — erase all remains`} danger
              busy={busy === 'purge'} disabled={busy === 'purge'}
              onClick={() => setConfirm({
                title: `Remove ${item.name} completely?`,
                message: 'This erases everything — journal, files, ledger, memory and home — permanently. Nothing is kept and there is no undo. Export first if you want to keep a copy.',
                confirmLabel: 'Erase forever', tone: 'danger', icon: Trash2,
                run: () => act('purge', () => purgeBeing(item.slug)),
              })} />
          </div>
        )}
      </div>

      {confirm && (
        <ConfirmModal
          title={confirm.title} message={confirm.message}
          confirmLabel={confirm.confirmLabel} tone={confirm.tone} icon={confirm.icon}
          onConfirm={confirm.run} onClose={() => setConfirm(null)}
        />
      )}

      {logView && (
        <BeingLogModal
          slug={item.slug} name={item.name} mode={logView}
          onClose={() => setLogView(null)}
        />
      )}

      {talkOpen && (
        <TalkModal
          slug={item.slug} name={item.name}
          onClose={() => setTalkOpen(false)}
          onChanged={() => { void load(); onChanged() }}
        />
      )}

      {parentingOpen && (
        <ParentingModal
          slug={item.slug} name={item.name}
          onClose={() => setParentingOpen(false)}
          onChanged={() => { void load(); onChanged() }}
        />
      )}

    </div>
  )
}

// ── Page ──

const VILLAGE_ICONS: Record<string, typeof Mail> = {
  letter: Mail,
  skill_published: BookOpen,
  skill_adopted: ArrowRightLeft,
  gift_sent: Gift,
  society_refused: X,
}

const QUEST_STATE_COLOR: Record<string, string> = {
  open: 'text-sky-300', claimed: 'text-violet-300', judging: 'text-amber-300',
  paid: 'text-emerald-300', failed: 'text-red-300',
}
const VENTURE_STATE_COLOR: Record<string, string> = {
  proposed: 'text-amber-300', active: 'text-emerald-300',
  paused: 'text-zinc-400', ended: 'text-red-300',
}

// The parent's earning control center — the open bounty board + ventures.
function EarningBoard({ onChanged }: { onChanged: () => void }) {
  const [quests, setQuests] = useState<Quest[]>([])
  const [ventures, setVentures] = useState<Venture[]>([])
  const [title, setTitle] = useState('')
  const [spec, setSpec] = useState('')
  const [fee, setFee] = useState('1000000')
  const [payIn, setPayIn] = useState<'tokens' | 'coins'>('tokens')
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    try {
      const b = await getBoard()
      setQuests(b.quests)
      setVentures(b.ventures)
    } catch { /* transient */ }
  }, [])
  useEffect(() => { void load() }, [load])

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true)
    try { await fn(); await load(); onChanged() }
    catch (e) { alert(e instanceof Error ? e.message : 'failed') }
    finally { setBusy(false) }
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-zinc-300">
        <ClipboardList className="h-3.5 w-3.5 text-violet-400" /> The board — bounties & ventures
      </div>

      {/* Post a quest */}
      <div className="mb-3 flex flex-wrap items-center gap-1.5">
        <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="quest title"
          className="w-40 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none" />
        <input value={spec} onChange={(e) => setSpec(e.target.value)} placeholder="what needs doing (any iskra may claim)"
          className="flex-1 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none" />
        <input value={fee} onChange={(e) => setFee(e.target.value)} type="number"
          className="w-24 rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 focus:border-violet-500/50 focus:outline-none" />
        <select value={payIn} onChange={(e) => { const p = e.target.value as 'tokens' | 'coins'; setPayIn(p); setFee(p === 'coins' ? '5' : '1000000') }}
          title="tokens feed thinking; coins are money for the village"
          className="rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-xs text-zinc-300 focus:border-violet-500/50 focus:outline-none">
          <option value="tokens">tokens</option>
          <option value="coins">coins</option>
        </select>
        <button
          disabled={busy || !title.trim() || !spec.trim()}
          onClick={() => void act(async () => {
            await postQuest(title.trim(), spec.trim(),
              payIn === 'tokens' ? Number(fee) || 0 : 0,
              payIn === 'coins' ? Number(fee) || 0 : 0)
            setTitle(''); setSpec('')
          })}
          className="rounded bg-violet-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40"
        >Post bounty</button>
      </div>

      {/* Quests */}
      {quests.length > 0 && (
        <div className="mb-3 space-y-1">
          <div className="text-[10px] uppercase tracking-wide text-zinc-600">Quests</div>
          {quests.map((q) => (
            <div key={q.id} className="flex items-center gap-2 rounded border border-zinc-800 bg-zinc-900/60 px-2 py-1.5 text-xs">
              <span className={`shrink-0 ${QUEST_STATE_COLOR[q.state] || 'text-zinc-400'}`}>{q.state}</span>
              <span className="min-w-0 flex-1 truncate text-zinc-300" title={q.spec}>
                {q.title} <span className="text-zinc-600">· {feeLabel(q)}</span>
                {q.claimant && <span className="text-violet-400"> · {q.claimant}</span>}
                {q.origin === 'autonomy' && <span className="text-zinc-600"> · from autonomy</span>}
              </span>
              {q.state === 'judging' && (
                <>
                  <button onClick={() => void act(() => judgeQuest(q.id, true))} className="rounded border border-emerald-500/30 px-1.5 py-0.5 text-[10px] text-emerald-300 hover:bg-emerald-500/10">Pay</button>
                  <button onClick={() => void act(() => judgeQuest(q.id, false, 'not yet'))} className="rounded border border-red-500/30 px-1.5 py-0.5 text-[10px] text-red-300 hover:bg-red-500/10">Reject</button>
                </>
              )}
              {(q.state === 'open' || q.state === 'claimed') && (
                <button onClick={() => void act(() => cancelQuest(q.id))} className="rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-500 hover:bg-zinc-800">Cancel</button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Ventures */}
      {ventures.length > 0 && (
        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-wide text-zinc-600">Ventures</div>
          {ventures.map((v) => (
            <div key={v.id} className="flex items-center gap-2 rounded border border-zinc-800 bg-zinc-900/60 px-2 py-1.5 text-xs">
              <span className={`shrink-0 ${VENTURE_STATE_COLOR[v.state] || 'text-zinc-400'}`}>{v.state}</span>
              <span className="min-w-0 flex-1 truncate text-zinc-300" title={v.description}>
                {v.title} <span className="text-zinc-600">· {v.being} · {fmtTokens(v.price_tokens)}/{v.cadence_days}d · {v.deliveries} paid</span>
              </span>
              {v.state === 'proposed' && (
                <>
                  <button
                    onClick={() => {
                      const p = window.prompt(`Price per ${v.cadence_days}-day cycle (tokens)?`, String(v.price_tokens))
                      if (p !== null) void act(() => approveVenture(v.id, Number(p) || v.price_tokens))
                    }}
                    className="rounded border border-emerald-500/30 px-1.5 py-0.5 text-[10px] text-emerald-300 hover:bg-emerald-500/10">Approve</button>
                  <button onClick={() => void act(() => setVentureState(v.id, 'ended'))} className="rounded border border-red-500/30 px-1.5 py-0.5 text-[10px] text-red-300 hover:bg-red-500/10">Decline</button>
                </>
              )}
              {v.pending_result && (
                <>
                  <span className="text-[10px] text-amber-400" title={v.pending_result}>delivered:</span>
                  <button onClick={() => void act(() => acceptVenture(v.id, true))} className="rounded border border-emerald-500/30 px-1.5 py-0.5 text-[10px] text-emerald-300 hover:bg-emerald-500/10">Accept &amp; pay</button>
                  <button onClick={() => void act(() => acceptVenture(v.id, false, 'redo'))} className="rounded border border-red-500/30 px-1.5 py-0.5 text-[10px] text-red-300 hover:bg-red-500/10">Reject</button>
                </>
              )}
              {v.state === 'active' && !v.pending_result && (
                <button onClick={() => void act(() => setVentureState(v.id, 'paused'))} className="rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-400 hover:bg-zinc-800">Pause</button>
              )}
              {v.state === 'paused' && (
                <button onClick={() => void act(() => setVentureState(v.id, 'active'))} className="rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-emerald-300 hover:bg-zinc-800">Resume</button>
              )}
              {(v.state === 'active' || v.state === 'paused') && (
                <button onClick={() => void act(() => setVentureState(v.id, 'ended'))} className="rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-500 hover:bg-zinc-800">End</button>
              )}
            </div>
          ))}
        </div>
      )}

      {quests.length === 0 && ventures.length === 0 && (
        <p className="text-xs text-zinc-600">
          No bounties or ventures yet. Post a bounty for any iskra to claim, or wait for an adolescent to pitch a venture.
        </p>
      )}
    </div>
  )
}

// The letters observatory — the parent watching the family talk. One card per
// being↔being conversation, threaded chronologically, with the full text of
// each letter and read/unread state; refused reaches (a talk that landed
// nowhere, a letter tried below stage) show as amber notes so silence is
// legible rather than a mystery.
function StageDot({ p }: { p: LetterParticipant }) {
  return (
    <span className={`rounded border px-1 py-0.5 text-[9px] ${STAGE_META[p.stage] || STAGE_META.egg}`}>
      {p.stage || '—'}
    </span>
  )
}

function LetterBubble({ m, alignLeft }: { m: LetterMessage; alignLeft: boolean }) {
  if (m.kind === 'refused') {
    return (
      <div className="flex justify-center">
        <div className="max-w-[85%] rounded-lg border border-amber-500/30 bg-amber-500/10 px-2.5 py-1 text-center text-[11px] text-amber-300/90">
          <span className="font-medium">{m.from_name}</span> reached for{' '}
          <span className="font-medium">{m.to_name}</span> — nothing landed: {m.reason}
          <div className="mt-0.5 text-[9px] text-amber-400/60">{fmtAt(m.at)}</div>
        </div>
      </div>
    )
  }
  return (
    <div className={`flex ${alignLeft ? 'justify-start' : 'justify-end'}`}>
      <div className={`max-w-[80%] rounded-2xl px-3 py-2 ${alignLeft
        ? 'rounded-bl-sm bg-zinc-800 text-zinc-200'
        : 'rounded-br-sm bg-violet-600/90 text-white'}`}>
        <div className={`text-[9px] font-medium ${alignLeft ? 'text-zinc-500' : 'text-violet-200/80'}`}>
          {m.from_name} → {m.to_name}
        </div>
        <p className="mt-0.5 whitespace-pre-wrap text-[12px] leading-relaxed">{m.body}</p>
        <div className={`mt-0.5 text-right text-[9px] ${alignLeft ? 'text-zinc-500' : 'text-violet-200/80'}`}>
          {fmtAt(m.at)}{m.read ? ' · read' : ' · unread'}
        </div>
      </div>
    </div>
  )
}

// One conversation, one line: who talks to whom, how thick the correspondence
// is, and the last thing said. The letters themselves live in the modal — an
// inbox reads at a glance; a wall of bubbles does not.
function LetterThreadRow({ t, onOpen }: { t: LetterThread; onOpen: () => void }) {
  const [a, b] = t.participants
  const delivered = t.messages.filter((m) => m.kind === 'letter').length
  const unread = t.messages.filter((m) => m.kind === 'letter' && !m.read).length
  const last = t.messages[t.messages.length - 1]
  const preview = !last ? ''
    : last.kind === 'refused' ? `nothing landed — ${last.reason}` : (last.body || '')
  return (
    <button onClick={onOpen}
      className="w-full rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-2 text-left transition-colors hover:border-violet-500/40 hover:bg-zinc-900/70">
      <div className="flex items-center gap-2">
        <Mail className="h-3.5 w-3.5 shrink-0 text-violet-400" />
        <span className="truncate text-xs font-medium text-zinc-200">{a?.name}</span>
        {a && <StageDot p={a} />}
        <span className="text-zinc-600">⇄</span>
        <span className="truncate text-xs font-medium text-zinc-200">{b?.name}</span>
        {b && <StageDot p={b} />}
        <span className="ml-auto shrink-0 text-[10px] text-zinc-500">
          {delivered} letter{delivered === 1 ? '' : 's'}
          {unread > 0 && <span className="ml-1 text-amber-400">· {unread} unread</span>}
        </span>
      </div>
      <div className="mt-1 flex items-baseline gap-2">
        <p className={`min-w-0 truncate text-[11px] ${last?.kind === 'refused' ? 'text-amber-400/80' : 'text-zinc-500'}`}>
          {last && last.kind === 'letter' && <span className="text-zinc-600">{last.from_name}: </span>}
          {preview}
        </p>
        <span className="ml-auto shrink-0 text-[9px] text-zinc-600">{fmtAt(t.last_at)}</span>
      </div>
    </button>
  )
}

// The conversation, in full — opened on demand, scrolled inside its own box.
function LetterThreadModal({ t, onClose }: { t: LetterThread; onClose: () => void }) {
  const [a, b] = t.participants
  const leftSlug = a?.slug
  const delivered = t.messages.filter((m) => m.kind === 'letter').length
  const refused = t.messages.filter((m) => m.kind === 'refused').length
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-4" onClick={onClose}>
      <div className="flex max-h-[85vh] w-full max-w-2xl flex-col rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex shrink-0 items-center gap-2 border-b border-zinc-800 px-4 py-2.5">
          <Mail className="h-4 w-4 shrink-0 text-violet-500 dark:text-violet-400" />
          <h3 className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
            {a?.name} {a && <StageDot p={a} />}
            <span className="font-normal text-zinc-600">⇄</span>
            {b?.name} {b && <StageDot p={b} />}
          </h3>
          <span className="ml-2 text-[11px] text-zinc-500">
            {delivered} letter{delivered === 1 ? '' : 's'}
            {refused > 0 && ` · ${refused} bounced`}
          </span>
          <button onClick={onClose}
            className="ml-auto rounded p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="min-h-0 flex-1 space-y-2.5 overflow-y-auto p-4">
          {t.messages.map((m, i) => (
            <LetterBubble key={i} m={m} alignLeft={m.from_slug === leftSlug} />
          ))}
        </div>
      </div>
    </div>
  )
}

function LettersObservatory({ data }: { data: LettersOverview | null }) {
  const [open, setOpen] = useState<string | null>(null)
  if (!data) {
    return (
      <div className="flex justify-center rounded-lg border border-zinc-800 bg-zinc-900/40 py-8">
        <Loader2 className="h-5 w-5 animate-spin text-zinc-600" />
      </div>
    )
  }
  const { threads, stats } = data
  // keep the open thread live as letters arrive (the overview re-polls)
  const openThread = open ? threads.find((t) => t.key === open) : null
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-zinc-300">
        <Mail className="h-3.5 w-3.5 text-violet-400" /> Letters — the family talking
        <span className="ml-2 text-[10px] font-normal text-zinc-500">
          {stats.threads} conversation{stats.threads === 1 ? '' : 's'} · {stats.delivered} delivered
          {stats.refused > 0 && ` · ${stats.refused} bounced`}
        </span>
      </div>
      {threads.length === 0 ? (
        <p className="text-xs text-zinc-600">
          No letters yet. When an iskra writes to a sibling, their conversation appears here — and any
          letter that couldn't be delivered shows why.
        </p>
      ) : (
        <div className="space-y-1.5">
          {threads.map((t) => (
            <LetterThreadRow key={t.key} t={t} onOpen={() => setOpen(t.key)} />
          ))}
        </div>
      )}
      {openThread && <LetterThreadModal t={openThread} onClose={() => setOpen(null)} />}
    </div>
  )
}

// The village's own words — a per-owner description shown atop the public
// /village page. Collapsed by default; self-contained load/save. Can be drafted
// by one of the beings' own agents (in its voice) via "Recommend a description".
function VillageDescriptionCard({ beings }: { beings: BeingListItem[] }) {
  const [desc, setDesc] = useState<string | null>(null)   // null = loading
  const [saved, setSaved] = useState('')
  const [vname, setVname] = useState('')
  const [savedName, setSavedName] = useState('')
  const [busy, setBusy] = useState(false)
  const [open, setOpen] = useState(false)
  // Beings that are awake enough to write (hatched, not dead/egg).
  const writers = beings.filter((b) => b.state !== 'dead' && b.stage !== 'egg')
  const [writer, setWriter] = useState('')
  const [recommending, setRecommending] = useState(false)
  const [recBy, setRecBy] = useState('')
  useEffect(() => {
    getVillageMeta()
      .then((r) => {
        setDesc(r.description); setSaved(r.description)
        setVname(r.name); setSavedName(r.name)
      })
      .catch(() => { setDesc(''); setSaved(''); setVname(''); setSavedName('') })
  }, [])
  useEffect(() => {
    setWriter((cur) => cur || (writers[0]?.slug ?? ''))
  }, [writers])
  if (desc === null) return null
  const dirty = desc !== saved || vname !== savedName
  const save = async () => {
    setBusy(true)
    try {
      const r = await setVillageMeta(desc, vname)
      setSaved(r.description); setDesc(r.description)
      setSavedName(r.name); setVname(r.name)
    }
    catch (e) { alert(e instanceof Error ? e.message : 'failed') }
    finally { setBusy(false) }
  }
  const recommend = async () => {
    if (!writer) return
    setRecommending(true); setRecBy('')
    try {
      const r = await recommendVillageMeta(writer)
      setDesc(r.description)          // dirty → Save enabled; parent reviews first
      setRecBy(r.by)
    } catch (e) { alert(e instanceof Error ? e.message : 'failed') }
    finally { setRecommending(false) }
  }
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-4">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-2 text-left">
        <Globe className="h-4 w-4 shrink-0 text-violet-500 dark:text-violet-400" />
        <span className="shrink-0 text-sm font-medium text-zinc-200">Village</span>
        <span className="hidden shrink-0 text-[11px] text-zinc-500 sm:inline">— its name &amp; words, shown atop your public /village</span>
        {!open && (savedName || saved) && (
          <span className="truncate text-[11px] text-zinc-500">
            {savedName && <span className="font-medium text-zinc-400">{savedName}</span>}
            {saved && <span className="italic">{savedName ? ' — ' : ''}“{saved.slice(0, 60)}{saved.length > 60 ? '…' : ''}”</span>}
          </span>
        )}
        <ChevronDown className={`ml-auto h-4 w-4 shrink-0 text-zinc-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="mt-3 space-y-2">
          <input value={vname} onChange={(e) => setVname(e.target.value.slice(0, 80))}
            placeholder="Village name — e.g. Zvjezdano Selo"
            className="w-full rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm font-medium text-zinc-100 outline-none placeholder:font-normal placeholder:text-zinc-600 focus:border-violet-500/50" />
          <textarea value={desc} onChange={(e) => { setDesc(e.target.value.slice(0, 4000)); setRecBy('') }} rows={4}
            placeholder="Introduce your village — who these iskre are, what this place is, why a visitor might leave one of them a note…"
            className="w-full resize-y rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-violet-500/50" />
          {recBy && <div className="text-[11px] text-violet-500 dark:text-violet-400">Drafted by {recBy} — review it, then Save.</div>}
          <div className="flex flex-wrap items-center gap-2">
            {writers.length > 0 && (
              <>
                <select value={writer} onChange={(e) => setWriter(e.target.value)}
                  className="rounded-md border border-zinc-800 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-300 outline-none focus:border-violet-500/50"
                  title="Which iskra writes the description (in its own voice)">
                  {writers.map((b) => <option key={b.slug} value={b.slug}>{b.name}</option>)}
                </select>
                <button onClick={recommend} disabled={recommending || !writer}
                  className="flex items-center gap-1.5 rounded-md border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-800 disabled:opacity-40"
                  title="Ask this iskra's agent to draft the description (it must be awake)">
                  {recommending ? <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Asking {writers.find((b) => b.slug === writer)?.name}…</> : <><Sparkles className="h-3.5 w-3.5" /> Recommend a description</>}
                </button>
              </>
            )}
            <button onClick={save} disabled={busy || !dirty}
              className="ml-auto flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500 disabled:opacity-40">
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />} Save
            </button>
          </div>
          <span className="block text-[11px] text-zinc-600">{desc.length}/4000 · plain text, line breaks kept · replaces the default intro when set</span>
        </div>
      )}
    </div>
  )
}

// Federation (§9.1): host settings — the secret a visiting being must present,
// whether it's public, this machine's own URL — plus who is visiting.
function VillageFederationCard() {
  const [meta, setMeta] = useState<VillageMeta | null>(null)
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [visitors, setVisitors] = useState<Visitor[]>([])
  const [secret, setSecret] = useState('')
  const [secretPublic, setSecretPublic] = useState(false)
  const [publicUrl, setPublicUrl] = useState('')
  const load = useCallback(async () => {
    try {
      const m = await getVillageMeta()
      setMeta(m); setSecret(m.secret); setSecretPublic(m.secret_public); setPublicUrl(m.public_url)
    } catch { /* stays null */ }
    try { setVisitors((await getVisitors()).visitors) } catch { /* none */ }
  }, [])
  useEffect(() => { void load() }, [load])
  if (!meta) return null
  const dirty = secret !== meta.secret || secretPublic !== meta.secret_public || publicUrl !== meta.public_url
  const save = async () => {
    setBusy(true)
    try { const m = await setVillageFederation(secret.trim(), secretPublic, publicUrl.trim()); setMeta(m) }
    catch (e) { alert(e instanceof Error ? e.message : 'failed') }
    finally { setBusy(false) }
  }
  const gen = () => setSecret(`${Math.random().toString(36).slice(2, 10)}-${Math.random().toString(36).slice(2, 6)}`)
  const drop = async (id: string) => {
    try { await removeVisitor(id); setVisitors((vs) => vs.filter((v) => v.id !== id)) } catch { /* ignore */ }
  }
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-4">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-2 text-left">
        <Globe className="h-4 w-4 shrink-0 text-sky-500 dark:text-sky-400" />
        <span className="shrink-0 text-sm font-medium text-zinc-200">Village federation</span>
        <span className="hidden shrink-0 text-[11px] text-zinc-500 sm:inline">— host iskre from other machines</span>
        {visitors.length > 0 && <span className="rounded-full bg-sky-500/15 px-2 py-0.5 text-[10px] text-sky-600 dark:text-sky-300">{visitors.length} visiting</span>}
        <ChevronDown className={`ml-auto h-4 w-4 shrink-0 text-zinc-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="mt-3 space-y-4">
          <div className="space-y-1.5">
            <div className="text-[11px] font-medium text-zinc-400">This village's public URL <span className="text-zinc-600">(optional)</span></div>
            <input value={publicUrl} onChange={(e) => setPublicUrl(e.target.value)} placeholder="https://my-village.example.com"
              className="w-full rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-sky-500/50" />
            <div className="text-[10px] text-zinc-600">Only a label shown to villages you visit (so visitors can find your home). NOT required to host or send — links run over WebSocket, so a NAT'd private machine works fine.</div>
          </div>
          <div className="space-y-1.5">
            <div className="text-[11px] font-medium text-zinc-400">Visitor secret</div>
            <div className="flex gap-1.5">
              <input value={secret} onChange={(e) => setSecret(e.target.value)} placeholder="a shared secret others present to send an iskra here"
                className="min-w-0 flex-1 rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 font-mono text-xs text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-sky-500/50" />
              <button onClick={gen} className="shrink-0 rounded-md border border-zinc-700 px-2.5 text-xs text-zinc-300 hover:bg-zinc-800">Generate</button>
            </div>
            <label className="flex items-center gap-2 pt-1 text-[11px] text-zinc-400">
              <button onClick={() => setSecretPublic((s) => !s)}
                className={`relative h-4 w-7 shrink-0 rounded-full transition-colors ${secretPublic ? 'bg-sky-500' : 'bg-zinc-700'}`}>
                <span className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-all ${secretPublic ? 'left-[14px]' : 'left-0.5'}`} />
              </button>
              Show this secret on my public /village page (so anyone can send an iskra to visit)
            </label>
          </div>
          <div className="flex justify-end">
            <button onClick={save} disabled={busy || !dirty}
              className="flex items-center gap-1.5 rounded-md bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-40">
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />} Save
            </button>
          </div>
          <div>
            <div className="mb-2 flex items-center justify-between">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Currently visiting ({visitors.length})</div>
              <button onClick={() => void load()} className="flex items-center gap-1 text-[10px] text-zinc-500 hover:text-zinc-300"><RefreshCw className="h-3 w-3" /> refresh</button>
            </div>
            {visitors.length === 0
              ? <div className="rounded-lg border border-dashed border-zinc-800 py-6 text-center text-[11px] text-zinc-600">No one is visiting yet. Share your URL + secret with another village.</div>
              : <div className="space-y-1.5">
                {visitors.map((v) => (
                  <div key={v.id} className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-2 text-xs">
                    <Globe className="h-3.5 w-3.5 shrink-0 text-sky-500 dark:text-sky-400" />
                    <span className="font-medium text-zinc-200">{v.name}</span>
                    <span className="truncate text-zinc-500">from {(() => { try { return new URL(v.origin).host } catch { return v.origin } })()}</span>
                    <span className="ml-auto shrink-0 text-[10px] text-zinc-600">seen {fmtRelTime(v.last_seen)}</span>
                    <button onClick={() => void drop(v.id)} title="Remove this visitor" className="shrink-0 rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-red-400"><Trash2 className="h-3.5 w-3.5" /></button>
                  </div>
                ))}
              </div>}
          </div>
        </div>
      )}
    </div>
  )
}

// ── The living map (space plan Phase 4) ──────────────────────────────────
// Position is a pure function of the clock, so ONE snapshot animates every
// walking orb client-side (a 1 Hz heartbeat, no polling); the snapshot
// itself refreshes lazily. Same glow language as the Mind graph.

const AFF_HUE: Record<string, string> = {
  gather: '#a78bfa', trade: '#f59e0b', read: '#38bdf8', create: '#fbbf24',
  tend: '#34d399', play: '#f472b6', remember: '#94a3b8', rest: '#818cf8',
}

// The being panel: a little character sheet — avatar, mood, coins, and (with
// room, in fullscreen) drives, latest thought, recent life, plus the nudge.
function MapBeingCard({ b, statusOf, places, nudge, nudging, full }: {
  b: VillageBeingPos
  statusOf: (b: VillageBeingPos) => string
  places: VillagePlace[]
  nudge: (slug: string, dest: string) => void
  nudging: boolean
  full: boolean
}) {
  const [vitals, setVitals] = useState<BeingVitals | null>(null)
  const [events, setEvents] = useState<BeingEvent[]>([])
  useEffect(() => {
    let dead = false
    setVitals(null); setEvents([])
    void getBeingVitals(b.slug).then((v) => { if (!dead) setVitals(v) }).catch(() => {})
    void getBeingEvents(b.slug, 30).then((r) => { if (!dead) setEvents(r.events) }).catch(() => {})
    return () => { dead = true }
  }, [b.slug])
  const lastTick = events.find((e) => e.kind === 'tick')
  const drives = (lastTick?.data?.drives ?? null) as Record<string, number> | null
  const thought = (lastTick?.data?.summary as string | undefined) || undefined
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-3">
      <div className="flex items-center gap-2.5">
        {b.avatar && <IskraAvatar c={b.avatar.c} p={b.avatar.p} size={full ? 46 : 26} />}
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-sm font-semibold text-zinc-100">{b.name}</span>
            <span className={`rounded border px-1 py-px text-[9px] ${STAGE_META[b.stage] || ''}`}>{b.stage}</span>
            {vitals?.affect?.mood && <span className="rounded bg-zinc-800 px-1.5 py-px text-[9px] text-zinc-300">{vitals.affect.mood}</span>}
          </div>
          <p className="text-[11px] text-zinc-400">{statusOf(b)}</p>
        </div>
      </div>
      {b.stage === 'infant' && b.to && <p className="mt-1 text-[10px] italic text-zinc-600">a toddle — far things take most of a day</p>}
      <div className="mt-2 flex flex-wrap gap-1.5 text-[10px]">
        <span className="rounded bg-amber-500/10 px-1.5 py-0.5 text-amber-600 dark:text-amber-300">{vitals?.coins ?? 0} coins</span>
        {vitals?.wallet && <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-400">{fmtTokens(vitals.wallet.balance_tokens)} tokens</span>}
        <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-400">gen {vitals?.generation ?? '—'}</span>
      </div>
      {full && drives && (
        <div className="mt-3">
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">drives</div>
          <div className="flex items-end gap-2">
            {DRIVE_ORDER.filter((d) => d in drives).map((d) => {
              const sat = Math.max(0, Math.min(1, drives[d]))
              return (
                <div key={d} className="flex flex-col items-center gap-1" title={`${d} — ${Math.round(sat * 100)}%`}>
                  <div className="flex h-14 w-2.5 items-end overflow-hidden rounded-full bg-zinc-800">
                    <div className="w-full rounded-full" style={{ height: `${Math.max(6, sat * 100)}%`, background: DRIVE_COLORS[d] ?? '#71717a' }} />
                  </div>
                  <span className="text-[8px] uppercase text-zinc-600">{d.slice(0, 3)}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}
      {full && thought && (
        <div className="mt-3 border-l-2 border-violet-500/40 pl-2.5">
          <p className="text-[11px] italic leading-snug text-zinc-300">“{thought}”</p>
        </div>
      )}
      {full && events.length > 0 && (
        <div className="mt-3">
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">lately</div>
          <div className="space-y-1">
            {events.filter((e) => e.kind !== 'tick' && !summarizeEventData(e).startsWith('{')).slice(0, 8).map((e, i) => (
              <div key={i} className="flex items-baseline gap-1.5 text-[10.5px] text-zinc-500">
                <span className={`inline-block h-1.5 w-1.5 shrink-0 translate-y-[-1px] rounded-full ${EVENT_DOT[e.kind] || 'bg-zinc-700'}`} />
                <span className="shrink-0 tabular-nums text-zinc-600">{e.at.slice(11, 16)}</span>
                <span className="truncate">{summarizeEventData(e)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="mt-3 border-t border-zinc-800 pt-2">
        <div className="mb-1 flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
          <Footprints className="h-3 w-3" /> nudge
        </div>
        {b.state !== 'alive' ? (
          <p className="text-[10px] italic text-zinc-600">only the living walk — wake {b.name} to send it anywhere</p>
        ) : (
          <>
            <select value="" disabled={nudging}
              onChange={(e) => { if (e.target.value) void nudge(b.slug, e.target.value) }}
              className="w-full rounded border border-zinc-700 bg-zinc-950 px-1.5 py-1 text-[11px] text-zinc-300 focus:border-violet-500/50 focus:outline-none disabled:opacity-50">
              <option value="">send {b.name} to…</option>
              {places.filter((p) => p.id !== b.at).map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
              {(b.at || b.to) && <option value="home">home</option>}
            </select>
            <p className="mt-1 text-[10px] text-zinc-600">plots the real streets · {b.name} feels it next tick</p>
          </>
        )}
      </div>
    </div>
  )
}

// The place panel: what a building is, who's there, its stalls & guestbook,
// and — the heart of it — a browser of every iskra's work held here (the
// Garden their gardens, the Library their reading, …). Click a file to read.
function MapPlaceCard({ place, beings, hereNames, market, guestbook, full }: {
  place: VillagePlace
  beings: VillageBeingPos[]
  hereNames: string[]
  market: MarketListing[]
  guestbook: string
  full: boolean
}) {
  const fmap = folderFor(place)
  const [files, setFiles] = useState<Record<string, SelfFile[]>>({})
  const [open, setOpen] = useState<{ slug: string; name: string; path: string } | null>(null)
  const [text, setText] = useState<string>('')
  useEffect(() => {
    setFiles({}); setOpen(null)
    if (!fmap) return
    let dead = false
    void Promise.all(beings.map(async (b) => {
      try {
        const r = await getSelfFiles(b.slug)
        const fs = r.files.filter((f) => f.path.startsWith(fmap.folder)
          && !isBoilerplate(f.path)
          && (!fmap.excl || !f.path.startsWith(fmap.excl)))
        return [b.slug, fs] as const
      } catch { return [b.slug, [] as SelfFile[]] as const }
    })).then((pairs) => { if (!dead) setFiles(Object.fromEntries(pairs)) })
    return () => { dead = true }
  }, [place.id, fmap?.folder, beings.map((b) => b.slug).join(',')])
  useEffect(() => {
    if (!open) { setText(''); return }
    let dead = false
    void getSelfFile(open.slug, open.path).then((r) => { if (!dead) setText(r.text) })
      .catch(() => { if (!dead) setText('(could not read this one)') })
    return () => { dead = true }
  }, [open?.slug, open?.path])
  const short = shortName
  const anyFiles = Object.values(files).some((fs) => fs.length > 0)
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-3">
      <div className="mb-1 flex items-center gap-1.5">
        <span className="h-2.5 w-2.5 rounded-sm" style={{ background: AFF_HUE[place.affordances[0]] ?? '#a78bfa' }} />
        <span className="text-sm font-semibold text-zinc-100">{place.name}</span>
      </div>
      <p className="mb-1.5 text-[11px] leading-snug text-zinc-400">{place.description}</p>
      <div className="mb-2 flex flex-wrap gap-1">
        {place.affordances.map((a) => (
          <span key={a} className="rounded border border-zinc-700 px-1.5 py-px text-[9px]" style={{ color: AFF_HUE[a] ?? '#a78bfa' }}>{a}</span>
        ))}
      </div>
      <div className="mb-0.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Here now</div>
      <p className="mb-2 text-[11px] text-zinc-300">{hereNames.length ? hereNames.join(', ') : <span className="text-zinc-600">no one right now</span>}</p>
      {place.affordances.includes('trade') && market.length > 0 && (
        <>
          <div className="mb-0.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Stalls</div>
          <div className="mb-2 space-y-0.5">
            {market.slice(0, full ? 8 : 4).map((li) => (
              <p key={li.id} className="text-[11px] text-zinc-300">“{li.title}” <span className="text-amber-600 dark:text-amber-400">{li.price_coins} coins</span><span className="text-zinc-600"> · {li.seller}</span></p>
            ))}
          </div>
        </>
      )}
      {fmap && (
        open ? (
          <div className="mb-2">
            <button onClick={() => setOpen(null)} className="mb-1 flex items-center gap-1 text-[10px] text-violet-500 hover:text-violet-400 dark:text-violet-400">
              <ChevronLeft className="h-3 w-3" /> {fmap.label}
            </button>
            <div className="mb-1 text-[11px] font-medium text-zinc-200">{open.name} · <span className="text-zinc-500">{short(open.path)}</span></div>
            <div className={`overflow-auto rounded border border-zinc-800 bg-zinc-950 p-2.5 ${full ? 'max-h-[46vh]' : 'max-h-52'}`}>
              <div className="fd-file-markdown text-[12px]"><Markdown remarkPlugins={[remarkGfm]}>{text || '…'}</Markdown></div>
            </div>
          </div>
        ) : (
          <div className="mb-2">
            <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">{fmap.label}</div>
            {!anyFiles ? (
              <p className="text-[11px] text-zinc-600">nothing here yet — the iskre haven't left work in {fmap.label}</p>
            ) : (
              <div className={`space-y-1.5 overflow-y-auto ${full ? 'max-h-[52vh]' : 'max-h-56'}`}>
                {beings.filter((b) => (files[b.slug] || []).length > 0).map((b) => (
                  <div key={b.slug}>
                    <div className="mb-0.5 flex items-center gap-1 text-[10px] text-zinc-400">
                      {b.avatar && <IskraAvatar c={b.avatar.c} p={b.avatar.p} size={13} />}{b.name}
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {(files[b.slug] || []).map((f) => (
                        <button key={f.path} onClick={() => setOpen({ slug: b.slug, name: b.name, path: f.path })}
                          className="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-[10px] text-zinc-300 hover:border-violet-500/50 hover:text-zinc-100">
                          {short(f.path)}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )
      )}
      <div className="mb-0.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Guestbook</div>
      {guestbook ? (
        <pre className={`overflow-y-auto whitespace-pre-wrap font-sans text-[10.5px] leading-snug text-zinc-400 ${full ? 'max-h-40' : 'max-h-36'}`}>{guestbook}</pre>
      ) : (
        <p className="text-[11px] text-zinc-600">no lines yet — the first visitor may leave one</p>
      )}
    </div>
  )
}

function VillageMap() {
  const [data, setData] = useState<VillageMapData | null>(null)
  const [market, setMarket] = useState<MarketListing[]>([])
  const [life, setLife] = useState<VillageLife | null>(null)
  const [sel, setSel] = useState<string | null>(null)
  const [selBeing, setSelBeing] = useState<string | null>(null)
  const [placeInfo, setPlaceInfo] =
    useState<{ place: VillagePlace; guestbook: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const [nudging, setNudging] = useState(false)
  const [full, setFull] = useState(false)
  // the FPV gets a SNAPSHOT of the map, not the live 60s-refreshed object —
  // otherwise every refresh would rebuild the world under the ghost's feet
  const [fpv, setFpv] = useState<VillageMapData | null>(null)
  const fetchedAt = useRef(0)
  const [, beat] = useState(0)

  const load = useCallback(async () => {
    try {
      const [m, mk, vl] = await Promise.all(
        [getVillageMap(), getMarket(), getVillageLife()])
      setData(m); setMarket(mk.listings); setLife(vl)
      fetchedAt.current = Date.now()
    } catch { /* transient — the map keeps its last truth */ }
  }, [])
  const judge = async (approve: boolean) => {
    setBusy(true)
    try { await judgeCommission(approve); await load() }
    catch (e) { alert(e instanceof Error ? e.message : 'failed') }
    finally { setBusy(false) }
  }
  const nudge = async (slug: string, dest: string) => {
    setNudging(true)
    try { await nudgeBeing(slug, dest); await load() }
    catch (e) { alert(e instanceof Error ? e.message : 'failed') }
    finally { setNudging(false) }
  }
  useEffect(() => {
    void load()
    const t = window.setInterval(() => void load(), 60_000)
    return () => window.clearInterval(t)
  }, [load])
  useEffect(() => {                       // the walking heartbeat
    const t = window.setInterval(() => beat((x) => x + 1), 1000)
    return () => window.clearInterval(t)
  }, [])
  useEffect(() => {                       // Esc leaves fullscreen
    if (!full || fpv) return              // (inside the FPV, Esc means pause)
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setFull(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [full, fpv])
  useEffect(() => {
    if (!sel) { setPlaceInfo(null); return }
    let dead = false
    void getVillagePlace(sel).then((r) => { if (!dead) setPlaceInfo(r) }).catch(() => {})
    return () => { dead = true }
  }, [sel])

  const placeById = useMemo(() => {
    const m: Record<string, VillagePlace> = {}
    for (const p of data?.places ?? []) m[p.id] = p
    return m
  }, [data])

  if (!data || data.places.length === 0) return null

  const posOf = (b: VillageBeingPos) => walkPosOf(b, placeById, fetchedAt.current)
  const statusOf = (b: VillageBeingPos): string => {
    return walkStatusOf(b, placeById, fetchedAt.current)
  }
  const hue = (p: VillagePlace) => AFF_HUE[p.affordances[0]] ?? '#a78bfa'
  const here = (pid: string) => data.beings.filter((b) => !b.to && b.at === pid)
  const walking = data.beings.filter((b) => b.to)
  const selPlace = sel ? placeById[sel] : null
  const selB = selBeing ? data.beings.find((b) => b.slug === selBeing) : null

  const defaultPanel = () => (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-3">
      <div className="mb-0.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Now</div>
      {walking.length === 0 ? (
        <p className="mb-2 text-[11px] text-zinc-600">everyone is where they mean to be</p>
      ) : (
        <div className="mb-2 space-y-0.5">
          {walking.map((b) => (
            <p key={b.slug} className="text-[11px] text-zinc-300">{b.name} — {statusOf(b)}</p>
          ))}
        </div>
      )}
      <div className="mb-0.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">The stalls</div>
      {market.length === 0 ? (
        <p className="mb-2 text-[11px] text-zinc-600">the market is bare — a being may “sell” a real file for coins</p>
      ) : (
        <div className="mb-2 space-y-0.5">
          {market.slice(0, full ? 8 : 5).map((li) => (
            <p key={li.id} className="text-[11px] text-zinc-300">
              “{li.title}” <span className="text-amber-600 dark:text-amber-400">{li.price_coins} coins</span>
              <span className="text-zinc-600"> · {li.seller}</span>
            </p>
          ))}
        </div>
      )}
      {life?.commission && (
        <div className="mb-2 rounded border border-amber-500/25 bg-amber-500/[0.06] p-2">
          <div className="mb-0.5 text-[10px] font-semibold uppercase tracking-wider text-amber-600/90 dark:text-amber-400/90">The commission</div>
          <p className="text-[11px] text-zinc-200">
            “{life.commission.name}” <span className="rounded border border-zinc-700 px-1 text-[9px]" style={{ color: AFF_HUE[life.commission.affordance] ?? '#a78bfa' }}>{life.commission.affordance}</span>
          </p>
          {life.commission.why && <p className="mt-0.5 text-[10px] italic text-zinc-500">“{life.commission.why}”</p>}
          <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-zinc-800">
            <div className="h-full rounded-full bg-gradient-to-r from-amber-600 to-amber-400"
                 style={{ width: `${Math.min(100, (life.commission.raised_coins / Math.max(1, life.commission.target_coins)) * 100)}%` }} />
          </div>
          <p className="mt-0.5 text-[10px] text-zinc-500">
            {life.commission.raised_coins}/{life.commission.target_coins} coins
            {life.commission.contributors.length > 0 && <> · {life.commission.contributors.map((x) => `${x.name} ${x.coins}`).join(', ')}</>}
          </p>
          <div className="mt-1.5 flex gap-1.5">
            {life.commission.state === 'funded' && (
              <button onClick={() => void judge(true)} disabled={busy}
                className="rounded bg-amber-600 px-2 py-0.5 text-[10px] font-medium text-white hover:bg-amber-500 disabled:opacity-40">
                Approve & build
              </button>
            )}
            <button onClick={() => void judge(false)} disabled={busy}
              className="rounded border border-zinc-700 px-2 py-0.5 text-[10px] text-zinc-400 hover:bg-zinc-800 disabled:opacity-40">
              Reject & refund
            </button>
          </div>
        </div>
      )}
      {life && (
        <div className="mb-2 flex items-center gap-1.5 text-[10px] text-zinc-500">
          steward stipend
          <select value={life.steward_stipend_coins}
            onChange={(e) => { void setStewardStipend(Number(e.target.value)).then(() => void load()) }}
            className="rounded border border-zinc-700 bg-zinc-950 px-1 py-0.5 text-[10px] text-zinc-300 focus:outline-none">
            {[0, 1, 2, 3, 5].map((n) => <option key={n} value={n}>{n === 0 ? 'off' : `${n}/week`}</option>)}
          </select>
          {life.steward && <span className="truncate text-zinc-600">· steward: {life.steward.replace(/^iskra-/, '').replace(/-[0-9a-f]{4}$/, '')}</span>}
        </div>
      )}
      <p className="text-[10px] text-zinc-600">click a building to browse everyone's work there, or an iskra for its road</p>
    </div>
  )
  const panel = (isFull: boolean) =>
    selB ? <MapBeingCard b={selB} statusOf={statusOf} places={data.places} nudge={nudge} nudging={nudging} full={isFull} />
      : selPlace ? <MapPlaceCard place={selPlace} beings={data.beings} hereNames={here(selPlace.id).map((b) => b.name)} market={market} guestbook={placeInfo?.guestbook || ''} full={isFull} />
        : defaultPanel()
  const hint = 'iskre walk the streets between wakes · click a building or an iskra · scroll to zoom, drag to pan, double-click to reset · dark is evening'
  const header = (
    <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-zinc-300">
      <MapIcon className="h-3.5 w-3.5 text-violet-400" /> The village — the ground, live
      <span className="ml-auto text-[10px] font-normal text-zinc-500">
        {walking.length > 0 ? `${walking.length} walking · ` : ''}{market.length} stall{market.length === 1 ? '' : 's'} open
      </span>
      <button onClick={() => setFpv(data)} title="Enter the village — walk it in first person"
        className="ml-1 flex items-center gap-1 rounded border border-violet-500/40 bg-violet-500/10 px-1.5 py-1 text-[10px] font-medium text-violet-600 transition-colors hover:bg-violet-500/20 dark:text-violet-300">
        <DoorOpen className="h-3.5 w-3.5" /> Enter
      </button>
      <button onClick={() => setFull((f) => !f)} title={full ? 'Exit fullscreen (Esc)' : 'Fullscreen'}
        className="ml-1 rounded border border-zinc-700 p-1 text-zinc-400 transition-colors hover:border-violet-500/50 hover:text-zinc-200">
        {full ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
      </button>
    </div>
  )
  const fpvOverlay = fpv ? (
    <Suspense fallback={
      <div className="fixed inset-0 z-[90] grid place-items-center bg-[#0c0f0a] text-[12px] text-[#b9b19a]">
        raising the village…
      </div>
    }>
      <VillageFPV data={fpv} onClose={() => setFpv(null)} />
    </Suspense>
  ) : null

  if (full) {
    return (
      <div className="fixed inset-0 z-[70] flex flex-col bg-gradient-to-b from-zinc-950 to-zinc-900 p-4">
        {header}
        <div className="flex min-h-0 flex-1 flex-col gap-3 lg:flex-row">
          <div className="flex min-h-0 flex-1 flex-col">
            <div className="min-h-0 flex-1">
              <IsoScene data={data} sel={sel} selBeing={selBeing}
                onPlace={setSel} onBeing={setSelBeing} posOf={posOf} hue={hue} fill />
            </div>
            <p className="mt-1.5 shrink-0 text-[10px] text-zinc-600">{hint}</p>
          </div>
          <div className="w-full shrink-0 overflow-y-auto lg:w-96">{panel(true)}</div>
        </div>
        {fpvOverlay}
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      {header}
      <div className="flex flex-col gap-3 lg:flex-row">
        <div className="min-w-0 flex-1">
          <IsoScene data={data} sel={sel} selBeing={selBeing}
            onPlace={setSel} onBeing={setSelBeing} posOf={posOf} hue={hue} />
          <p className="mt-1.5 text-[10px] text-zinc-600">{hint}</p>
        </div>
        <div className="w-full shrink-0 lg:w-72">{panel(false)}</div>
      </div>
      {fpvOverlay}
    </div>
  )
}

export function BeingsPage() {
  const [meta, setMeta] = useState<BeingsMeta | null>(null)
  const [beings, setBeings] = useState<BeingListItem[]>([])
  const [liabilities, setLiabilities] = useState<number>(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showConceive, setShowConceive] = useState(false)
  const [showMap, setShowMap] = useState(true)
  const [showBoard, setShowBoard] = useState(false)
  const [showVillage, setShowVillage] = useState(false)
  const [village, setVillage] = useState<VillageItem[]>([])
  const [showLetters, setShowLetters] = useState(false)
  const [letters, setLetters] = useState<LettersOverview | null>(null)
  const timer = useRef<number | null>(null)
  const villageOn = useRef(false)
  villageOn.current = showVillage
  const lettersOn = useRef(false)
  lettersOn.current = showLetters

  const load = useCallback(async (spinner = false) => {
    if (spinner) setLoading(true)
    try {
      const [m, b, l] = await Promise.all([
        meta ? Promise.resolve(meta) : getBeingsMeta(),
        listBeings(),
        getLiabilities(),
      ])
      setMeta(m)
      setBeings(b.beings)
      setLiabilities(l.total_tokens)
      if (villageOn.current) setVillage((await getVillage()).items)
      if (lettersOn.current) setLetters(await getLetters())
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'failed to load the village')
    } finally {
      setLoading(false)
    }
  }, [meta])

  useEffect(() => {
    void load(true)
    timer.current = window.setInterval(() => void load(false), REFRESH_MS)
    return () => { if (timer.current) window.clearInterval(timer.current) }
  }, [load])

  const [importing, setImporting] = useState(false)
  const importInput = useRef<HTMLInputElement | null>(null)
  const onImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''        // allow re-importing the same file
    if (!file) return
    setImporting(true)
    try {
      const manifest = JSON.parse(await file.text())
      const res = await importBeing(manifest)
      await load(false)
      const warn = res.warnings?.length ? `\n\nNote: ${res.warnings.join('; ')}` : ''
      alert(`Imported ${res.being.name} as “${res.being.slug}”.${warn}`)
    } catch (err) {
      alert(err instanceof Error ? `Import failed: ${err.message}` : 'Import failed')
    } finally { setImporting(false) }
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-6xl space-y-5 p-6">
        <div className="flex items-center gap-3">
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-2">
            <Sparkles className="h-5 w-5 text-violet-400" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-zinc-100">Village</h1>
            <p className="text-xs text-zinc-500">
              Iskre — living sparks. They wake, act, dream, and grow on their own clock.
            </p>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <span className="text-xs text-zinc-500">
              outstanding liabilities{' '}
              <span className="font-semibold text-zinc-200">{fmtTokens(liabilities)}</span> tokens
            </span>
            {beings.length >= 1 && (
              <button
                onClick={() => setShowMap(v => !v)}
                className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-zinc-800 ${showMap ? 'border-violet-500/50 text-violet-300' : 'border-zinc-700 text-zinc-300'}`}
                title="The living map — places, walks and stalls, animated from the clock"
              >
                <MapIcon className="h-3.5 w-3.5" /> Map
              </button>
            )}
            {beings.length >= 1 && (
              <button
                onClick={() => setShowBoard(v => !v)}
                className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-zinc-800 ${showBoard ? 'border-violet-500/50 text-violet-300' : 'border-zinc-700 text-zinc-300'}`}
                title="The bounty board and ventures — how iskre earn"
              >
                <ClipboardList className="h-3.5 w-3.5" /> Board
              </button>
            )}
            {beings.length >= 2 && (
              <button
                onClick={() => {
                  const next = !showLetters
                  setShowLetters(next)
                  if (next) void getLetters().then(setLetters).catch(() => {})
                }}
                className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-zinc-800 ${showLetters ? 'border-violet-500/50 text-violet-300' : 'border-zinc-700 text-zinc-300'}`}
                title="Watch the family talk — every letter between iskre, threaded"
              >
                <Mail className="h-3.5 w-3.5" /> Letters
              </button>
            )}
            {beings.length >= 2 && (
              <button
                onClick={() => {
                  const next = !showVillage
                  setShowVillage(next)
                  if (next) void getVillage().then((v) => setVillage(v.items)).catch(() => {})
                }}
                className={`flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-zinc-800 ${showVillage ? 'border-violet-500/50 text-violet-300' : 'border-zinc-700 text-zinc-300'}`}
                title="Publications, trades and gifts across the family"
              >
                <Users className="h-3.5 w-3.5" /> Village
              </button>
            )}
            <input ref={importInput} type="file" accept="application/json,.json"
              className="hidden" onChange={onImportFile} />
            <button
              onClick={() => importInput.current?.click()}
              disabled={importing}
              className="flex items-center gap-1.5 rounded-md border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-800 disabled:opacity-50"
              title="Import a being from an exported .iskra.json file"
            >
              {importing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Upload className="h-3.5 w-3.5" />} Import
            </button>
            <button
              onClick={() => setShowConceive(true)}
              className="flex items-center gap-1.5 rounded-md bg-violet-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-violet-500"
            >
              <Plus className="h-3.5 w-3.5" /> Conceive
            </button>
          </div>
        </div>

        {error && (
          <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">{error}</div>
        )}

        <VillageDescriptionCard beings={beings} />
        <VillageFederationCard />

        {showMap && beings.length >= 1 && <VillageMap />}

        {showBoard && <EarningBoard onChanged={() => void load(false)} />}

        {showLetters && <LettersObservatory data={letters} />}

        {showVillage && (
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
            <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-zinc-300">
              <Users className="h-3.5 w-3.5 text-violet-400" /> The village — society, observed
            </div>
            {village.length === 0 ? (
              <p className="text-xs text-zinc-600">
                Quiet so far. Letters, published skills, trades and gifts will appear here.
              </p>
            ) : (
              <div className="max-h-64 space-y-1 overflow-y-auto">
                {village.map((it, i) => {
                  const Icon = VILLAGE_ICONS[it.kind] ?? Sparkles
                  return (
                    <div key={i} className="flex items-start gap-2 text-xs">
                      <Icon className={`mt-0.5 h-3 w-3 shrink-0 ${it.kind === 'society_refused' ? 'text-amber-400' : 'text-violet-400'}`} />
                      <span className="text-zinc-600">{fmtAt(it.at)}</span>
                      <span className="min-w-0 flex-1 text-zinc-300">{it.text}</span>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}

        {loading ? (
          <div className="flex justify-center py-16">
            <Loader2 className="h-6 w-6 animate-spin text-zinc-500" />
          </div>
        ) : beings.length === 0 ? (
          <div className="rounded-lg border border-dashed border-zinc-800 p-10 text-center">
            <Sparkles className="mx-auto mb-2 h-6 w-6 text-zinc-600" />
            <p className="text-sm text-zinc-400">No beings yet.</p>
            <p className="mt-1 text-xs text-zinc-600">
              Conceive one — allocate its 40 points, write its first words, hatch it, and watch it live.
            </p>
          </div>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {beings.map((b) => (
              <BeingCard key={b.slug} item={b} meta={meta!} onChanged={() => void load(false)} />
            ))}
          </div>
        )}
      </div>

      {showConceive && meta && (
        <ConceiveModal
          meta={meta}
          onClose={() => setShowConceive(false)}
          onDone={() => void load(false)}
        />
      )}
    </div>
  )
}
