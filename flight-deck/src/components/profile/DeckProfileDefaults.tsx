import { useEffect, useState } from 'react'
import { Building2, Loader2 } from 'lucide-react'
import {
  getDeckProfileDefaults,
  saveDeckProfileDefaults,
  agentsUpdatedLabel,
  charCount,
  DEFAULT_PROFILE_CAPS,
  ECO_SHORTENED_HINT,
  type DeckProfileDefaults as Defaults,
} from '../../services/profile'
import { CappedTextarea } from './CappedTextarea'

// Admin: the company description and standing instructions every user's
// agents receive (/fd/admin/profile-defaults). On an auth-off deck the single
// local user edits them from the Profile page instead — Admin isn't reachable
// there (the kiosk only ever shows them read-only). `onSaved` lets that page
// refresh its preview.
export function DeckProfileDefaults({ onSaved }: { onSaved?: () => void }) {
  const [saved, setSaved] = useState<Defaults | null>(null)
  const [company, setCompany] = useState('')
  const [instructions, setInstructions] = useState('')
  const [caps, setCaps] = useState({ company: DEFAULT_PROFILE_CAPS.company, instructions: DEFAULT_PROFILE_CAPS.instructions })
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    getDeckProfileDefaults()
      .then((d) => {
        if (cancelled) return
        const v = { company: d.company || '', instructions: d.instructions || '' }
        setSaved(v); setCompany(v.company); setInstructions(v.instructions)
        setCaps((c) => ({ company: d.caps?.company ?? c.company, instructions: d.caps?.instructions ?? c.instructions }))
      })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : String(e)) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [])

  const dirty = !!saved && (company !== saved.company || instructions !== saved.instructions)
  const tooLong = charCount(company) > caps.company || charCount(instructions) > caps.instructions

  const save = async () => {
    setSaving(true); setError(null); setNotice(null)
    try {
      const d = await saveDeckProfileDefaults({ company, instructions })
      const v = { company: d.company ?? company, instructions: d.instructions ?? instructions }
      setSaved(v); setCompany(v.company); setInstructions(v.instructions)
      setNotice(`Saved — ${agentsUpdatedLabel(d.agents_updated)}.`)
      onSaved?.()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 space-y-4">
      <div className="flex items-start gap-2">
        <Building2 className="mt-0.5 h-4 w-4 shrink-0 text-violet-400" />
        <div>
          <h3 className="text-sm font-semibold text-zinc-200">Deck-wide profile defaults</h3>
          <p className="text-xs text-zinc-500 mt-0.5">
            A company description and standing instructions that every user's agents receive. A user's own
            company text replaces the deck's; these instructions apply alongside their own preferences,
            and win if they conflict.
          </p>
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center py-4">
          <Loader2 className="h-5 w-5 animate-spin text-zinc-500" />
        </div>
      ) : saved ? (
        <>
          <p className="text-[11px] text-zinc-500">{ECO_SHORTENED_HINT}</p>
          <CappedTextarea
            label="Company"
            value={company}
            onChange={(v) => { setCompany(v); setNotice(null) }}
            cap={caps.company}
            rows={6}
            placeholder="What the company does, its products and customers, the voice it writes in…"
          />
          <CappedTextarea
            label="Instructions for everyone"
            value={instructions}
            onChange={(v) => { setInstructions(v); setNotice(null) }}
            cap={caps.instructions}
            rows={4}
            placeholder="e.g. Write in British English. Never share pricing outside the company."
          />
          <div className="flex items-center gap-2">
            <button
              onClick={save}
              disabled={saving || !dirty || tooLong}
              className="rounded-md bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
            >
              {saving ? 'Saving…' : 'Save defaults'}
            </button>
            {dirty && (
              <button
                onClick={() => { setCompany(saved.company); setInstructions(saved.instructions); setError(null) }}
                disabled={saving}
                className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800"
              >
                Revert
              </button>
            )}
            <span className="text-[11px] text-zinc-600">Agents pick it up on their next turn — no restart.</span>
          </div>
        </>
      ) : null}

      {notice && (
        <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-700 dark:text-emerald-300">
          {notice}
        </div>
      )}
      {error && (
        <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-300">
          {error}
        </div>
      )}
    </div>
  )
}
