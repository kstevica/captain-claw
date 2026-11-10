import { useState } from 'react'
import { ChevronDown, Eye, EyeOff } from 'lucide-react'
import type { CredentialField } from '../../stores/externalMemoryStore'

// The credential fields of an external memory, as a provider declares them
// (captain_claw/flight_deck/external_memory/base.py: CredentialField). Fields
// marked `advanced` — what most people leave as it is, like where vectors are
// made — are folded under "More settings", open when one of them is saved.

interface Props {
  idPrefix: string
  fields: CredentialField[]
  drafts: Record<string, string>
  onChange: (name: string, value: string) => void
  // What is saved: a secret's hint, any other field's value.
  hints?: Record<string, string>
  disabled?: boolean
  inputClass: string
}

export default function CredentialInputs({ idPrefix, fields, drafts, onChange, hints, disabled, inputClass }: Props) {
  const [reveal, setReveal] = useState(false)
  const basic = fields.filter((f) => !f.advanced)
  const advanced = fields.filter((f) => f.advanced)
  const [open, setOpen] = useState(() => advanced.some((f) => hints?.[f.name]))

  const field = (f: CredentialField) => {
    const id = `${idPrefix}-${f.name}`
    const hint = hints?.[f.name]
    return (
      <div key={f.name} className="space-y-1">
        <label htmlFor={id} className="block text-xs font-medium text-zinc-400">
          {f.label}
          {!f.required && <span className="ml-1 text-zinc-600">(optional)</span>}
          {hint && f.secret && <span className="ml-2 text-[10px] text-emerald-600 dark:text-emerald-500">(saved)</span>}
        </label>
        <div className="relative">
          <input
            id={id}
            type={f.secret && !reveal ? 'password' : 'text'}
            value={drafts[f.name] || ''}
            onChange={(e) => onChange(f.name, e.target.value)}
            placeholder={hint ? (f.secret ? `•••• saved (${hint}) — paste a new one to replace` : hint) : f.placeholder}
            autoComplete="off"
            spellCheck={false}
            disabled={disabled}
            className={`${inputClass} font-mono ${f.secret ? 'pr-9' : ''}`}
          />
          {f.secret && (
            <button
              type="button"
              onClick={() => setReveal((v) => !v)}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
              tabIndex={-1}
              aria-label={reveal ? 'Hide' : 'Show'}
            >
              {reveal ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
            </button>
          )}
        </div>
        {f.help && <div className="text-[11px] leading-snug text-zinc-500">{f.help}</div>}
      </div>
    )
  }

  return (
    <>
      {basic.map(field)}
      {advanced.length > 0 && (
        <div>
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            className="flex items-start gap-1 text-left text-xs text-zinc-400 hover:text-zinc-200"
          >
            <ChevronDown className={`h-3.5 w-3.5 mt-px shrink-0 transition-transform ${open ? '' : '-rotate-90'}`} />
            <span>
              More settings
              <span className="block text-[11px] text-zinc-600">{advanced.map((f) => f.label).join(' · ')}</span>
            </span>
          </button>
          {open && <div className="mt-3 space-y-3 border-l border-zinc-800 pl-3">{advanced.map(field)}</div>}
        </div>
      )}
    </>
  )
}
