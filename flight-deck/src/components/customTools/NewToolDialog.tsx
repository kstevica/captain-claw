import { useEffect, useState } from 'react'
import { Hammer } from 'lucide-react'
import { api } from '../../services/fileLibraryApi'
import type { CrewMe } from '../../services/crewApi'
import { customToolsApi } from '../../services/customToolsApi'
import { Dialog, Button, Field, Notice, inputCls } from '../console/parts'

// A new tool from what it should do: Flight Deck makes its repository and a
// Crew worker writes it. It goes live after the person approves the merge.

let harnessCache: CrewMe | null = null

export function HarnessPicker({ value, onChange }: { value: string; onChange: (id: string) => void }) {
  const [me, setMe] = useState<CrewMe | null>(harnessCache)
  useEffect(() => {
    if (harnessCache) return
    api<CrewMe>('/fd/crew/me').then((m) => { harnessCache = m; setMe(m) }).catch(() => setMe(null))
  }, [])
  const usable = (me?.harnesses || []).filter((h) => h.installed)
  useEffect(() => {
    if (!value && usable.length) onChange((usable.find((h) => h.login?.ok) || usable[0]).id)
  }, [value, usable, onChange])
  return (
    <select aria-label="Coding agent" className={`${inputCls} w-auto`} value={value} onChange={(e) => onChange(e.target.value)}>
      {usable.length === 0 && <option value="">No coding agent installed</option>}
      {usable.map((h) => <option key={h.id} value={h.id}>{h.label}{h.login?.ok ? '' : ' (needs a login in Crew)'}</option>)}
    </select>
  )
}

const SLUG = /^[a-z0-9](?:[a-z0-9-]{0,46}[a-z0-9])?$/

export function NewToolDialog({ onClose, onStarted }: { onClose: () => void; onStarted: (message: string) => void }) {
  const [slug, setSlug] = useState('')
  const [title, setTitle] = useState('')
  const [what, setWhat] = useState('')
  const [harness, setHarness] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const ok = SLUG.test(slug) && what.trim().length >= 10 && !!harness
  const submit = async () => {
    setBusy(true)
    setError('')
    try {
      const started = await customToolsApi.build({ slug, title, what, harness })
      onStarted(`${started.task.harness_label || 'A coding agent'} is building “${slug}”. Approve its merge on the Crew page when it’s done; it goes live once its tests and examples pass.`)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
      setBusy(false)
    }
  }

  return (
    <Dialog title="New custom tool" icon={Hammer} onClose={onClose} footer={<>
      <Button variant="ghost" onClick={onClose}>Cancel</Button>
      <Button variant="primary" icon={Hammer} busy={busy} disabled={!ok} onClick={submit}>Build it</Button>
    </>}>
      <div className="space-y-4">
        {error && <Notice tone="red">{error}</Notice>}
        <Field label="Name" hint="Lowercase letters, digits and dashes, like invoice-parse. Agents call it by this name.">
          <input className={`${inputCls} font-mono`} value={slug} maxLength={48} onChange={(e) => setSlug(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, '-'))} />
        </Field>
        <Field label="Title" hint="A few words, like “Read an invoice PDF”.">
          <input className={inputCls} value={title} maxLength={80} onChange={(e) => setTitle(e.target.value)} />
        </Field>
        <Field label="What it should do" hint="Its input, what it returns, and an example. The worker writes tests and examples from this.">
          <textarea className={inputCls} rows={6} value={what} onChange={(e) => setWhat(e.target.value)}
            placeholder="Read a CSV file and add up one column. Input: the file and the column's name. Returns the number of rows and the total. Example: …" />
        </Field>
        <Field label="Coding agent">
          <HarnessPicker value={harness} onChange={setHarness} />
        </Field>
      </div>
    </Dialog>
  )
}
