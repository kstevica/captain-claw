// How the Anonymize page names and colours kinds and dates.

const KIND_TONE: Record<string, string> = {
  PERSON: 'bg-violet-500/15 text-violet-700 dark:text-violet-300',
  ORG: 'bg-sky-500/15 text-sky-700 dark:text-sky-300',
  EMAIL: 'bg-cyan-500/15 text-cyan-700 dark:text-cyan-300',
  PHONE: 'bg-cyan-500/15 text-cyan-700 dark:text-cyan-300',
  ADDRESS: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300',
  DETAIL: 'bg-fuchsia-500/15 text-fuchsia-700 dark:text-fuchsia-300',
}
const NUMBER_TONE = 'bg-amber-500/15 text-amber-700 dark:text-amber-300'

export function kindTone(kind: string): string {
  return KIND_TONE[kind] || NUMBER_TONE
}

export function kindLabel(kind: string): string {
  const k = kind.toLowerCase().replace(/_/g, ' ')
  return { dob: 'date of birth', oib: 'OIB', jmbg: 'JMBG', iban: 'IBAN', ip: 'IP address', org: 'company',
    'profile url': 'profile link', 'id number': 'ID number', ssn: 'SSN', cvv: 'card code', 'card expiry': 'card expiry',
    detail: 'description' }[k] || k
}

export function when(ts: number | null | undefined): string {
  if (!ts) return '—'
  const d = new Date(ts * 1000)
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}
