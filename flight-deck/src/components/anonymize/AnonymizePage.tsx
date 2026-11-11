import { useCallback, useEffect, useState } from 'react'
import { EyeOff, X, Undo2, Vault as VaultIcon, ShieldCheck, Sparkles, AlertTriangle } from 'lucide-react'
import {
  anonymizeApi, fileToBase64, saveFile,
  type AnonKinds, type AnonStatus, type Lane, type Mode, type RestoreResult, type RunResult, type Vault,
} from '../../services/anonymizeApi'
import { useAnonymizeStore, type AnonymizeTab } from '../../stores/anonymizeStore'
import { Page, PageHeader, Card, Button, Segmented, Notice, Field, Pill, inputCls } from '../console/parts'
import { CopyButton, DownloadButton, FileDrop, TokenText } from './anonParts'
import { kindLabel, kindTone } from './anonFormat'
import { VaultsPanel } from './VaultsPanel'

// Anonymize: take names, numbers and addresses out of a text or a file. Each
// becomes a token (⟦PERSON_1⟧); the real values stay in the person's vault,
// encrypted, and "Put back" brings them back — for the person only.

const DEFAULT_VAULT = 'My documents'
const FALLBACK_KINDS: AnonKinds = { kinds: [], formats: [], file_kinds: [], max_file_mb: 25 }

function lines(text: string): string[] {
  return text.split(/\n|,/).map((s) => s.trim()).filter(Boolean)
}

function parseExtra(text: string): { text: string; kind: string }[] {
  return text.split('\n').map((line) => line.trim()).filter(Boolean).map((line) => {
    const [term, kind] = line.split('|').map((s) => s.trim())
    return { text: term, kind: (kind || 'PERSON').toUpperCase().replace(/[^A-Z0-9_]/g, '_') }
  }).filter((x) => x.text)
}

export function AnonymizePage({ phone = false, onClose }: { phone?: boolean; onClose?: () => void }) {
  const storeTab = useAnonymizeStore((s) => s.tab)
  const [tab, setTab] = useState<AnonymizeTab>(onClose ? storeTab : 'anonymize')
  const [status, setStatus] = useState<AnonStatus | null>(null)
  const [kinds, setKinds] = useState<AnonKinds>(FALLBACK_KINDS)
  const [vaults, setVaults] = useState<Vault[]>([])
  const [sheetOpen, setSheetOpen] = useState(false)
  const [error, setError] = useState('')

  const reloadVaults = useCallback(async () => {
    try {
      setVaults((await anonymizeApi.vaults()).vaults)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't load your vaults")
    }
  }, [])

  useEffect(() => {
    anonymizeApi.status().then(setStatus).catch(() => setStatus(null))
    anonymizeApi.kinds().then(setKinds).catch(() => {})
    anonymizeApi.vaults().then((r) => setVaults(r.vaults)).catch(() => setError("Couldn't load your vaults"))
  }, [])

  // In the kiosk's dialog, Escape closes the page — unless a vault is open on top.
  useEffect(() => {
    if (!onClose || sheetOpen) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, sheetOpen])

  return (
    <div className="@container h-full overflow-y-auto">
      <Page full={!!onClose}>
        <PageHeader
          icon={EyeOff}
          title="Anonymize"
          description="Take names, numbers and addresses out of a text or a file. Each becomes a token like ⟦PERSON_1⟧ that means nothing outside Captain; the real values stay in your vault, encrypted, for you alone."
          actions={onClose && <Button variant="ghost" icon={X} onClick={onClose} aria-label="Close" />}
        />
        {error && <Notice tone="red">{error}</Notice>}
        <div className="overflow-x-auto">
          <Segmented
            label="What to do"
            value={tab}
            onChange={setTab}
            options={[
              { value: 'anonymize', label: 'Anonymize' },
              { value: 'restore', label: 'Put back' },
              { value: 'vaults', label: 'Vaults', count: vaults.length },
            ]}
          />
        </div>
        {tab === 'anonymize' && <AnonymizeTab status={status} kinds={kinds} vaults={vaults} onDone={reloadVaults} phone={phone} />}
        {tab === 'restore' && <RestoreTab kinds={kinds} vaults={vaults} />}
        {tab === 'vaults' && <VaultsPanel vaults={vaults} reload={reloadVaults} onSheet={setSheetOpen} />}
      </Page>
    </div>
  )
}

// ── Anonymize ───────────────────────────────────────────────────────

function AnonymizeTab({ status, kinds, vaults, onDone, phone }: {
  status: AnonStatus | null
  kinds: AnonKinds
  vaults: Vault[]
  onDone: () => void
  phone: boolean
}) {
  const [source, setSource] = useState<'text' | 'file'>('text')
  const [text, setText] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [vault, setVault] = useState('')
  const [mode, setMode] = useState<Mode>('reversible')
  const [lane, setLane] = useState<Lane>('both')
  const [more, setMore] = useState(false)
  const [keep, setKeep] = useState('')
  const [extra, setExtra] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<RunResult | null>(null)
  const [ranOn, setRanOn] = useState<'text' | 'file'>('text')

  const vaultName = vault.trim() || vaults[0]?.name || DEFAULT_VAULT
  const ready = source === 'text' ? !!text.trim() : !!file

  const run = async () => {
    setBusy(true); setError(''); setResult(null)
    try {
      const opts = {
        mode,
        vault: mode === 'reversible' ? vaultName : undefined,
        lane: status?.llm ? lane : 'deterministic' as Lane,
        keep: lines(keep),
        extra: parseExtra(extra),
      }
      if (source === 'text') {
        setResult(await anonymizeApi.run(text, opts))
      } else if (file) {
        if (file.size > kinds.max_file_mb * 1024 * 1024) throw new Error(`The file is over ${kinds.max_file_mb} MB.`)
        setResult(await anonymizeApi.file(file.name, await fileToBase64(file), opts))
      }
      setRanOn(source)
      onDone()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Anonymizing failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-5">
      <Card bodyClassName="space-y-5 p-5">
        <Segmented label="What to anonymize" value={source} onChange={setSource}
          options={[{ value: 'text', label: 'Text' }, { value: 'file', label: 'File' }]} />
        {source === 'text' ? (
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={phone ? 8 : 10}
            placeholder="Paste a letter, an email, notes, a table…"
            className={`${inputCls} font-mono leading-relaxed`}
          />
        ) : (
          <FileDrop file={file} accept={kinds.file_kinds} maxMb={kinds.max_file_mb} onFile={setFile} />
        )}

        <div className="grid gap-4 wide:grid-cols-2">
          <Field label="Keep the real values?" hint={mode === 'reversible'
            ? 'They go into a vault, so you can put them back later. The same person gets the same token in every document of that vault.'
            : 'Nothing is kept: the tokens can never be turned back. Use this to share data for good.'}>
            <Segmented label="Keep the real values" value={mode} onChange={setMode}
              options={[{ value: 'reversible', label: 'In a vault' }, { value: 'one_way', label: 'Keep nothing' }]} />
          </Field>
          {mode === 'reversible' && (
            <Field label="Vault" hint="One vault per case or client. A new name makes a new vault.">
              <input
                list="anon-vaults"
                value={vault}
                onChange={(e) => setVault(e.target.value)}
                placeholder={vaults[0]?.name || DEFAULT_VAULT}
                className={inputCls}
                maxLength={80}
              />
              <datalist id="anon-vaults">
                {vaults.map((v) => <option key={v.id} value={v.name} />)}
              </datalist>
            </Field>
          )}
        </div>

        {status?.llm ? (
          <Field label="Who looks" hint={`Code finds numbers, emails, labelled names and addresses. The trusted model (${status.model}) also finds names with nothing around them and descriptions that point to one person.`}>
            <Segmented label="Who looks" value={lane} onChange={setLane}
              options={[
                { value: 'both', label: <span className="flex items-center gap-1.5"><Sparkles className="h-3.5 w-3.5" />Code + trusted model</span> },
                { value: 'deterministic', label: 'Code only' },
                { value: 'llm', label: 'Trusted model only' },
              ]} />
          </Field>
        ) : (
          <p className="text-xs leading-relaxed text-zinc-500">
            Code looks for numbers, emails, labelled names, addresses and the names your vault already knows. This deck has no trusted
            model, so a name with nothing around it ("we met Marko") can be missed — check the result, and add misses under More.
          </p>
        )}

        <div>
          <button type="button" onClick={() => setMore((v) => !v)} className="text-xs font-medium text-violet-600 hover:underline dark:text-violet-400">
            {more ? 'Fewer options' : 'More options'}
          </button>
          {more && (
            <div className="mt-3 grid gap-4 wide:grid-cols-2">
              <Field label="Leave as they are" hint="One per line: your own company, public figures, places.">
                <textarea value={keep} onChange={(e) => setKeep(e.target.value)} rows={3} className={inputCls} />
              </Field>
              <Field label="Always hide" hint="One per line. Add | and a kind for something that isn't a person: Project Falcon | PROJECT">
                <textarea value={extra} onChange={(e) => setExtra(e.target.value)} rows={3} className={inputCls} />
              </Field>
            </div>
          )}
        </div>

        {error && <Notice tone="red">{error}</Notice>}
        <div className="flex justify-end">
          <Button variant="primary" icon={EyeOff} busy={busy} disabled={!ready} onClick={run}>Anonymize</Button>
        </div>
      </Card>

      {result && <ResultView result={result} isFile={ranOn === 'file'} />}
    </div>
  )
}

function lanesLine(r: RunResult): { text: string; tone: 'emerald' | 'amber' | 'zinc' } {
  const l = r.lanes
  if (l.llm === 'ok') {
    return { text: `Checked by code and the trusted model ${l.llm_model || ''}${l.llm_terms ? ` (it found ${l.llm_terms} more)` : ''}.`, tone: 'emerald' }
  }
  if (l.llm === 'failed') {
    return { text: 'The trusted model failed, so only code looked: names with nothing around them may be left. Check the result.', tone: 'amber' }
  }
  if (l.deterministic === 'off') return { text: 'Checked by the trusted model only.', tone: 'zinc' }
  return { text: 'Checked by code only.', tone: 'zinc' }
}

function ResultView({ result, isFile }: { result: RunResult; isFile: boolean }) {
  const counts = Object.entries(result.summary)
  const lane = lanesLine(result)
  const text = isFile ? result.preview || '' : result.output || ''
  return (
    <Card
      icon={ShieldCheck}
      title={counts.length ? 'Anonymized' : 'Nothing private found'}
      hint={result.vault ? `Real values kept in your vault "${result.vault.name}".` : 'Nothing was kept: these tokens can never be turned back.'}
      action={isFile && result.data ? (
        <DownloadButton onClick={() => saveFile(result.filename, { base64: result.data || '', type: result.content_type })}>
          Download {result.filename}
        </DownloadButton>
      ) : (
        <>
          <CopyButton text={result.output || ''} />
          <DownloadButton onClick={() => saveFile('anonymized.txt', { text: result.output || '' })} />
        </>
      )}
      bodyClassName="space-y-4 p-5"
    >
      {counts.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {counts.map(([kind, n]) => (
            <span key={kind} className={`rounded-full px-2 py-0.5 text-xs font-medium ${kindTone(kind)}`}>
              {n} {kindLabel(kind)}{n === 1 ? '' : 's'}
            </span>
          ))}
        </div>
      )}
      <Notice tone={lane.tone}>{lane.text}{result.verify.fixed ? ` A second look caught ${result.verify.fixed} more.` : ''}</Notice>
      {result.warnings.map((w) => (
        <Notice key={w} tone="amber"><span className="inline-flex items-start gap-2"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />{w}</span></Notice>
      ))}
      {isFile && <p className="text-xs text-zinc-500">The text inside the file, as it reads now:</p>}
      <TokenText text={text} className="max-h-[50vh] overflow-y-auto" />
      {result.entities.length > 0 && (
        <details className="text-sm">
          <summary className="cursor-pointer text-xs font-medium text-zinc-400">Every token ({result.entities.length})</summary>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {result.entities.map((e) => (
              <Pill key={e.token} title={`${kindLabel(e.kind)} · ${e.count}×${e.new ? ' · new in this vault' : ''}`}>
                <span className="font-mono">{e.token}</span> ×{e.count}{e.new ? ' · new' : ''}
              </Pill>
            ))}
          </div>
        </details>
      )}
    </Card>
  )
}

// ── Put back ────────────────────────────────────────────────────────

function RestoreTab({ kinds, vaults }: { kinds: AnonKinds; vaults: Vault[] }) {
  const [source, setSource] = useState<'text' | 'file'>('text')
  const [text, setText] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [vault, setVault] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<RestoreResult | null>(null)
  const vaultName = vault || vaults[0]?.name || ''

  const run = async () => {
    setBusy(true); setError(''); setResult(null)
    try {
      if (source === 'text') setResult(await anonymizeApi.restore(text, vaultName))
      else if (file) setResult(await anonymizeApi.restoreFile(file.name, await fileToBase64(file), vaultName))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Putting back failed')
    } finally {
      setBusy(false)
    }
  }

  if (!vaults.length) {
    return <Notice>You have no vaults yet. Anonymize something with “In a vault”, and the real values can be put back here.</Notice>
  }
  return (
    <div className="space-y-5">
      <Notice tone="amber">
        The result holds the real values again. It's for you — don't paste it into an agent's chat or anywhere it shouldn't go.
      </Notice>
      <Card bodyClassName="space-y-5 p-5">
        <div className="grid gap-4 wide:grid-cols-2">
          <Field label="Vault" hint="The vault the text or file was anonymized into.">
            <select value={vaultName} onChange={(e) => setVault(e.target.value)} className={inputCls}>
              {vaults.map((v) => <option key={v.id} value={v.name}>{v.name} ({v.entity_count})</option>)}
            </select>
          </Field>
          <Field label="What to put back">
            <Segmented label="What to put back" value={source} onChange={setSource}
              options={[{ value: 'text', label: 'Text' }, { value: 'file', label: 'File' }]} />
          </Field>
        </div>
        {source === 'text' ? (
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={8}
            placeholder="Paste text with tokens like ⟦PERSON_1⟧ — an agent's answer, a summary…"
            className={`${inputCls} font-mono leading-relaxed`} />
        ) : (
          <FileDrop file={file} accept={kinds.file_kinds} maxMb={kinds.max_file_mb} onFile={setFile} />
        )}
        {error && <Notice tone="red">{error}</Notice>}
        <div className="flex justify-end">
          <Button variant="primary" icon={Undo2} busy={busy} disabled={source === 'text' ? !text.trim() : !file} onClick={run}>
            Put back
          </Button>
        </div>
      </Card>
      {result && (
        <Card
          icon={VaultIcon}
          title={`Put back ${result.restored} value${result.restored === 1 ? '' : 's'}`}
          hint={result.unknown.length ? `Not in this vault, left as they are: ${result.unknown.slice(0, 12).join(', ')}${result.unknown.length > 12 ? ' …' : ''}` : undefined}
          action={result.data ? (
            <DownloadButton onClick={() => saveFile(result.filename || 'restored', { base64: result.data || '', type: result.content_type })}>
              Download {result.filename}
            </DownloadButton>
          ) : (
            <>
              <CopyButton text={result.output || ''} />
              <DownloadButton onClick={() => saveFile('restored.txt', { text: result.output || '' })} />
            </>
          )}
          bodyClassName="p-5"
        >
          {result.output !== undefined ? <TokenText text={result.output} className="max-h-[50vh] overflow-y-auto" />
            : <p className="text-sm text-zinc-400">The file is ready to download.</p>}
        </Card>
      )}
    </div>
  )
}
