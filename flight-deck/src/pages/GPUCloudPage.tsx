import { useEffect, useState, useCallback } from 'react'
import {
  Cloud,
  Loader2,
  Play,
  Square,
  Trash2,
  Key,
  X,
  RefreshCw,
  Search,
  Cpu,
  DollarSign,
  HardDrive,
  Wifi,
  CheckCircle2,
  AlertCircle,
  Copy,
  Download,
  CircleDot,
  Server,
  Timer,
  Shield,
  ShieldOff,
} from 'lucide-react'
import { useVastAIStore, type VastInstance, type VastOffer, type VastOfferFilter } from '../stores/vastaiStore'

// ── Helpers ──

function formatPrice(dph: number): string {
  return `$${dph.toFixed(3)}/hr`
}

function formatSize(gb: number): string {
  return gb >= 1024 ? `${(gb / 1024).toFixed(1)} TB` : `${gb.toFixed(0)} GB`
}

function stateColor(state: string): string {
  switch (state) {
    case 'running': return 'text-emerald-400'
    case 'creating': case 'loading': return 'text-amber-400'
    case 'stopped': return 'text-zinc-400'
    case 'stopping': return 'text-amber-400'
    case 'error': case 'exited': return 'text-red-400'
    case 'destroyed': return 'text-zinc-600'
    default: return 'text-zinc-500'
  }
}

function stateBg(state: string): string {
  switch (state) {
    case 'running': return 'bg-emerald-500/10 border-emerald-500/20'
    case 'creating': case 'loading': return 'bg-amber-500/10 border-amber-500/20'
    case 'stopped': return 'bg-zinc-500/10 border-zinc-500/20'
    case 'error': case 'exited': return 'bg-red-500/10 border-red-500/20'
    default: return 'bg-zinc-500/10 border-zinc-500/20'
  }
}

// ── Sub-Components ──

function ApiKeySetup() {
  const { setApiKey, loading, error } = useVastAIStore()
  const [key, setKey] = useState('')

  const submit = async () => {
    if (!key.trim()) return
    const ok = await setApiKey(key.trim())
    if (ok) setKey('')
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-6">
      <div className="flex items-center gap-2 mb-3">
        <Key className="h-4 w-4 text-violet-400" />
        <h3 className="text-sm font-semibold text-zinc-100">Connect vast.ai</h3>
      </div>
      <p className="text-xs text-zinc-500 mb-4">
        Enter your vast.ai API key to browse GPU offers and manage instances.
        Get your key at{' '}
        <a href="https://cloud.vast.ai/manage-keys/" target="_blank" rel="noopener noreferrer"
           className="text-violet-400 hover:text-violet-300 underline">
          cloud.vast.ai/manage-keys
        </a>
      </p>
      <div className="flex gap-2">
        <input
          type="password"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && submit()}
          placeholder="vast.ai API key"
          className="flex-1 rounded-md border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500 focus:outline-none"
        />
        <button
          onClick={submit}
          disabled={loading || !key.trim()}
          className="rounded-md bg-violet-600 px-4 py-2 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {loading && <Loader2 className="h-3 w-3 animate-spin" />}
          Connect
        </button>
      </div>
      {error && (
        <p className="mt-2 text-xs text-red-400">{error}</p>
      )}
    </div>
  )
}

function AccountBar() {
  const { balance, email, removeApiKey, refreshStatus } = useVastAIStore()

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-3 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="h-4 w-4 text-emerald-400" />
          <span className="text-xs text-zinc-400">Connected</span>
          {email && <span className="text-xs text-zinc-500">({email})</span>}
        </div>
        {balance !== null && (
          <div className="flex items-center gap-1">
            <DollarSign className="h-3 w-3 text-zinc-500" />
            <span className="text-xs text-zinc-300 font-mono">{balance.toFixed(2)}</span>
            <span className="text-xs text-zinc-500">balance</span>
          </div>
        )}
      </div>
      <div className="flex items-center gap-2">
        <button onClick={refreshStatus} className="p-1 rounded hover:bg-zinc-800 text-zinc-500 hover:text-zinc-300" title="Refresh">
          <RefreshCw className="h-3.5 w-3.5" />
        </button>
        <button onClick={removeApiKey} className="p-1 rounded hover:bg-zinc-800 text-zinc-500 hover:text-red-400" title="Disconnect">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  )
}

const AUTO_STOP_OPTIONS = [
  { value: 0, label: 'Off' },
  { value: 1, label: '1 min' },
  { value: 2, label: '2 min' },
  { value: 5, label: '5 min' },
  { value: 10, label: '10 min' },
]

function InstanceCard({ inst }: { inst: VastInstance }) {
  const { stopInstance, startInstance, destroyInstance, setAutoStop, actionLoading, pullModel } = useVastAIStore()
  const [pullInput, setPullInput] = useState('')
  const [showPull, setShowPull] = useState(false)
  const [showConnection, setShowConnection] = useState(false)
  const isActing = actionLoading?.startsWith(`${inst.id}:`)

  const copyConnection = useCallback(() => {
    const info = {
      provider: 'ollama',
      base_url: inst.public_ip && inst.ollama_port ? `http://${inst.public_ip}:${inst.ollama_port}` : '',
      api_key: inst.auth_token,
    }
    navigator.clipboard.writeText(JSON.stringify(info, null, 2))
  }, [inst])

  const handlePull = async () => {
    if (!pullInput.trim()) return
    await pullModel(inst.id, pullInput.trim())
    setPullInput('')
    setShowPull(false)
  }

  return (
    <div className={`rounded-lg border p-4 ${stateBg(inst.state)}`}>
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="flex items-center gap-2">
            <Server className="h-4 w-4 text-zinc-400" />
            <span className="text-sm font-semibold text-zinc-100">
              {inst.label || `Instance ${inst.id}`}
            </span>
            <span className={`text-xs font-medium ${stateColor(inst.state)}`}>
              {inst.state}
            </span>
          </div>
          <div className="flex items-center gap-3 mt-1 text-xs text-zinc-500">
            <span className="flex items-center gap-1">
              <Cpu className="h-3 w-3" />
              {inst.gpu_name} {inst.num_gpus > 1 ? `x${inst.num_gpus}` : ''}
            </span>
            {inst.gpu_ram_gb > 0 && <span>{inst.gpu_ram_gb} GB VRAM</span>}
            <span className="flex items-center gap-1">
              <DollarSign className="h-3 w-3" />
              {formatPrice(inst.dph_total)}
            </span>
            {inst.secure_ollama ? (
              <span className="flex items-center gap-0.5 text-emerald-500" title="Secured with nginx proxy + Bearer token auth">
                <Shield className="h-3 w-3" />
              </span>
            ) : (
              <span className="flex items-center gap-0.5 text-amber-500" title="Ollama exposed without auth">
                <ShieldOff className="h-3 w-3" />
              </span>
            )}
            <span className="flex items-center gap-1">
              <HardDrive className="h-3 w-3" />
              {formatSize(inst.disk_gb)}
            </span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1.5">
          {/* Auto-stop selector */}
          {inst.state !== 'destroyed' && (
            <div className="flex items-center gap-1 mr-1" title="Auto-stop after inactivity">
              <Timer className="h-3 w-3 text-zinc-500" />
              <select
                value={inst.auto_stop_minutes}
                onChange={(e) => setAutoStop(inst.id, parseInt(e.target.value))}
                className="bg-transparent border border-zinc-700 rounded px-1 py-0.5 text-xs text-zinc-400 focus:border-violet-500 focus:outline-none cursor-pointer"
              >
                {AUTO_STOP_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
          )}
          {inst.state === 'running' && (
            <button onClick={() => stopInstance(inst.id)} disabled={!!isActing}
              className="p-1.5 rounded hover:bg-zinc-700/50 text-zinc-400 hover:text-amber-400 disabled:opacity-50" title="Stop">
              <Square className="h-3.5 w-3.5" />
            </button>
          )}
          {(inst.state === 'stopped' || inst.state === 'exited') && (
            <button onClick={() => startInstance(inst.id)} disabled={!!isActing}
              className="p-1.5 rounded hover:bg-zinc-700/50 text-zinc-400 hover:text-emerald-400 disabled:opacity-50" title="Start">
              <Play className="h-3.5 w-3.5" />
            </button>
          )}
          {inst.state !== 'destroyed' && (
            <button onClick={() => { if (confirm('Destroy this instance? All data will be lost.')) destroyInstance(inst.id) }}
              disabled={!!isActing}
              className="p-1.5 rounded hover:bg-zinc-700/50 text-zinc-400 hover:text-red-400 disabled:opacity-50" title="Destroy">
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}
          {isActing && <Loader2 className="h-3.5 w-3.5 animate-spin text-zinc-500" />}
        </div>
      </div>

      {/* Ollama status */}
      {inst.state === 'running' && (
        <div className="mt-2 pt-2 border-t border-white/5">
          <div className="flex items-center gap-2">
            {inst.ollama_ready ? (
              <span className="flex items-center gap-1 text-xs text-emerald-400">
                <CircleDot className="h-3 w-3" /> Ollama ready
                {inst.public_ip && inst.ollama_port > 0 && (
                  <span className="ml-1 text-zinc-500 font-mono">http://{inst.public_ip}:{inst.ollama_port}</span>
                )}
              </span>
            ) : (
              <span className="flex items-center gap-1 text-xs text-amber-400">
                <Loader2 className="h-3 w-3 animate-spin" /> Ollama starting...
              </span>
            )}
            {inst.ollama_ready && (
              <div className="flex items-center gap-1 ml-auto">
                <button onClick={() => setShowPull(!showPull)}
                  className="px-2 py-0.5 rounded text-xs text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/50 flex items-center gap-1">
                  <Download className="h-3 w-3" /> Pull model
                </button>
                <button onClick={() => setShowConnection(!showConnection)}
                  className="px-2 py-0.5 rounded text-xs text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/50 flex items-center gap-1">
                  <Copy className="h-3 w-3" /> Connection
                </button>
              </div>
            )}
          </div>

          {/* Models list */}
          {inst.models_loaded.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {inst.models_loaded.map((m) => (
                <span key={m} className="rounded bg-zinc-800 px-2 py-0.5 text-xs text-zinc-300 font-mono">
                  {m}
                </span>
              ))}
            </div>
          )}

          {/* Pull model input */}
          {showPull && (
            <div className="mt-2 flex gap-2">
              <input
                value={pullInput}
                onChange={(e) => setPullInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handlePull()}
                placeholder="e.g. llama3.2, deepseek-r1:70b"
                className="flex-1 rounded border border-zinc-700 bg-zinc-800 px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500 focus:outline-none"
              />
              <button onClick={handlePull} disabled={!pullInput.trim() || actionLoading === `${inst.id}:pull`}
                className="rounded bg-violet-600 px-3 py-1 text-xs text-white hover:bg-violet-500 disabled:opacity-50 flex items-center gap-1">
                {actionLoading === `${inst.id}:pull` ? <Loader2 className="h-3 w-3 animate-spin" /> : <Download className="h-3 w-3" />}
                Pull
              </button>
            </div>
          )}

          {/* Connection info */}
          {showConnection && inst.public_ip && inst.ollama_port > 0 && (
            <div className="mt-2 rounded bg-zinc-800 p-2 text-xs font-mono">
              <div className="text-zinc-400">
                <span className="text-zinc-500">provider:</span> ollama
              </div>
              <div className="text-zinc-400">
                <span className="text-zinc-500">base_url:</span> http://{inst.public_ip}:{inst.ollama_port}
              </div>
              <div className="text-zinc-400">
                <span className="text-zinc-500">api_key:</span> {inst.auth_token.slice(0, 8)}...
              </div>
              <button onClick={copyConnection}
                className="mt-1 text-violet-400 hover:text-violet-300 text-xs flex items-center gap-1">
                <Copy className="h-3 w-3" /> Copy JSON
              </button>
            </div>
          )}
        </div>
      )}

      {inst.ollama_error && (
        <p className="mt-2 text-xs text-red-400 flex items-center gap-1">
          <AlertCircle className="h-3 w-3" /> {inst.ollama_error}
        </p>
      )}
    </div>
  )
}

function OfferCard({ offer, onSelect }: { offer: VastOffer; onSelect: (offer: VastOffer) => void }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-3 hover:border-zinc-700 transition-colors">
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Cpu className="h-4 w-4 text-violet-400" />
            <span className="text-sm font-semibold text-zinc-100">
              {offer.gpu_name}
              {offer.num_gpus > 1 && <span className="text-zinc-500"> x{offer.num_gpus}</span>}
            </span>
          </div>
          <div className="flex items-center gap-3 mt-1 text-xs text-zinc-500">
            <span>{offer.gpu_ram_gb} GB VRAM</span>
            <span>{offer.cpu_cores} CPU</span>
            <span>{offer.ram_gb.toFixed(0)} GB RAM</span>
            <span className="flex items-center gap-1">
              <Wifi className="h-3 w-3" />
              {offer.inet_down_mbps.toFixed(0)} Mbps
            </span>
            {offer.geolocation && <span>{offer.geolocation}</span>}
          </div>
        </div>
        <div className="text-right">
          <div className="text-sm font-semibold text-emerald-400">{formatPrice(offer.dph_total)}</div>
          <div className="text-xs text-zinc-500">{(offer.reliability * 100).toFixed(1)}% rel.</div>
        </div>
      </div>
      <div className="mt-2 flex justify-end">
        <button onClick={() => onSelect(offer)}
          className="rounded-md bg-violet-600 px-3 py-1 text-xs font-medium text-white hover:bg-violet-500">
          Rent &amp; Setup Ollama
        </button>
      </div>
    </div>
  )
}

function CreateModal({ offer, onClose }: { offer: VastOffer; onClose: () => void }) {
  const { createInstance, actionLoading } = useVastAIStore()
  const [label, setLabel] = useState(`${offer.gpu_name.toLowerCase()}-ollama`)
  const [diskGb, setDiskGb] = useState(64)
  const [prePullModel, setPrePullModel] = useState('')
  const [secureOllama, setSecureOllama] = useState(true)
  const isCreating = actionLoading === 'create'

  const submit = async () => {
    const inst = await createInstance(offer.id, label, diskGb, prePullModel, secureOllama)
    if (inst) onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div className="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-900 p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-semibold text-zinc-100">Create GPU Instance</h3>
          <button onClick={onClose} className="p-1 rounded hover:bg-zinc-800 text-zinc-500"><X className="h-4 w-4" /></button>
        </div>

        <div className="rounded bg-zinc-800 p-3 mb-4">
          <div className="flex justify-between text-sm">
            <span className="text-zinc-300 font-medium">{offer.gpu_name}</span>
            <span className="text-emerald-400 font-semibold">{formatPrice(offer.dph_total)}</span>
          </div>
          <div className="text-xs text-zinc-500 mt-1">
            {offer.gpu_ram_gb} GB VRAM &middot; {offer.cpu_cores} CPU &middot; {offer.ram_gb.toFixed(0)} GB RAM
          </div>
        </div>

        <div className="space-y-3">
          <div>
            <label className="block text-xs text-zinc-400 mb-1">Instance label</label>
            <input value={label} onChange={(e) => setLabel(e.target.value)}
              className="w-full rounded border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-sm text-zinc-200 focus:border-violet-500 focus:outline-none" />
          </div>
          <div>
            <label className="block text-xs text-zinc-400 mb-1">Disk size (GB)</label>
            <input type="number" value={diskGb} onChange={(e) => setDiskGb(parseInt(e.target.value) || 64)} min={20} max={500}
              className="w-full rounded border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-sm text-zinc-200 focus:border-violet-500 focus:outline-none" />
          </div>
          <div>
            <label className="block text-xs text-zinc-400 mb-1">Auto-pull model (optional)</label>
            <input value={prePullModel} onChange={(e) => setPrePullModel(e.target.value)}
              placeholder="e.g. llama3.2, qwen3:8b"
              className="w-full rounded border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500 focus:outline-none" />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <label className="block text-xs text-zinc-300 font-medium">Secure Ollama</label>
              <p className="text-xs text-zinc-500 mt-0.5">
                {secureOllama
                  ? 'nginx proxy validates Bearer token — only you can access'
                  : 'Ollama exposed directly — anyone with the IP can use it'}
              </p>
            </div>
            <button
              type="button"
              onClick={() => setSecureOllama(!secureOllama)}
              className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ${
                secureOllama ? 'bg-violet-600' : 'bg-zinc-700'
              }`}
            >
              <span className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform duration-200 ${
                secureOllama ? 'translate-x-4' : 'translate-x-0'
              }`} />
            </button>
          </div>
        </div>

        <div className="mt-4 flex items-center justify-between">
          <p className="text-xs text-zinc-500">
            Billing starts immediately at {formatPrice(offer.dph_total)}
          </p>
          <button onClick={submit} disabled={isCreating}
            className="rounded-md bg-violet-600 px-4 py-2 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-50 flex items-center gap-2">
            {isCreating && <Loader2 className="h-3 w-3 animate-spin" />}
            Create Instance
          </button>
        </div>
      </div>
    </div>
  )
}

function OfferSearch() {
  const { searchOffers, offers, offersLoading } = useVastAIStore()
  const [gpuName, setGpuName] = useState('')
  const [minVram, setMinVram] = useState('')
  const [maxPrice, setMaxPrice] = useState('')
  const [selectedOffer, setSelectedOffer] = useState<VastOffer | null>(null)
  const [searched, setSearched] = useState(false)

  const search = useCallback(async () => {
    const filters: VastOfferFilter = {}
    if (gpuName.trim()) filters.gpu_name = gpuName.trim()
    if (minVram) filters.min_gpu_ram_gb = parseFloat(minVram)
    if (maxPrice) filters.max_price_per_hour = parseFloat(maxPrice)
    await searchOffers(filters)
    setSearched(true)
  }, [gpuName, minVram, maxPrice, searchOffers])

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <input value={gpuName} onChange={(e) => setGpuName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && search()}
          placeholder="GPU (e.g. RTX, H100, 4090)"
          className="flex-1 rounded border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500 focus:outline-none" />
        <input value={minVram} onChange={(e) => setMinVram(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && search()}
          placeholder="Min VRAM GB"
          type="number" step="1" min="0"
          className="w-32 rounded border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500 focus:outline-none" />
        <input value={maxPrice} onChange={(e) => setMaxPrice(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && search()}
          placeholder="Max $/hr"
          type="number" step="0.1" min="0"
          className="w-28 rounded border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:border-violet-500 focus:outline-none" />
        <button onClick={search} disabled={offersLoading}
          className="rounded-md bg-violet-600 px-4 py-2 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-50 flex items-center gap-2">
          {offersLoading ? <Loader2 className="h-3 w-3 animate-spin" /> : <Search className="h-3 w-3" />}
          Search
        </button>
      </div>

      {offersLoading && (
        <div className="text-center py-6 text-zinc-500 text-sm flex items-center justify-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" /> Searching marketplace...
        </div>
      )}

      {!offersLoading && searched && offers.length === 0 && (
        <div className="text-center py-6 text-zinc-500 text-sm">
          No offers found. Try different filters.
        </div>
      )}

      {!offersLoading && offers.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs text-zinc-500">{offers.length} offers found, sorted by price</p>
          {offers.map((o) => (
            <OfferCard key={o.id} offer={o} onSelect={setSelectedOffer} />
          ))}
        </div>
      )}

      {selectedOffer && (
        <CreateModal offer={selectedOffer} onClose={() => setSelectedOffer(null)} />
      )}
    </div>
  )
}

// ── Main Page ──

export function GPUCloudPage() {
  const { configured, instances, loading, error, refreshStatus, refreshInstances, clearError } = useVastAIStore()

  useEffect(() => {
    refreshStatus()
  }, [refreshStatus])

  useEffect(() => {
    if (configured) refreshInstances()
  }, [configured, refreshInstances])

  // Auto-refresh instances every 15 seconds when configured.
  useEffect(() => {
    if (!configured) return
    const interval = setInterval(refreshInstances, 15000)
    return () => clearInterval(interval)
  }, [configured, refreshInstances])

  const activeInstances = instances.filter((i) => i.state !== 'destroyed')
  const destroyedInstances = instances.filter((i) => i.state === 'destroyed')

  return (
    <div className="h-full overflow-y-auto bg-zinc-950 text-zinc-200">
      <div className="max-w-4xl mx-auto px-6 py-8 pb-16">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <div className="h-10 w-10 rounded-md bg-zinc-900 border border-zinc-800 flex items-center justify-center">
            <Cloud className="h-5 w-5 text-violet-500" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-zinc-100">GPU Cloud</h1>
            <p className="text-sm text-zinc-500">
              Rent GPUs on vast.ai, run Ollama models, and use them as LLM providers.
            </p>
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="mb-4 rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-3 flex items-center justify-between">
            <span className="text-xs text-red-400 flex items-center gap-2">
              <AlertCircle className="h-3.5 w-3.5" /> {error}
            </span>
            <button onClick={clearError} className="text-red-400 hover:text-red-300"><X className="h-3.5 w-3.5" /></button>
          </div>
        )}

        {loading && !configured && (
          <div className="text-center py-12 text-zinc-500 flex items-center justify-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" /> Checking configuration...
          </div>
        )}

        {!loading && !configured && <ApiKeySetup />}

        {configured && (
          <div className="space-y-6">
            <AccountBar />

            {/* Active instances */}
            {activeInstances.length > 0 && (
              <div>
                <h2 className="text-sm font-semibold text-zinc-300 mb-3 flex items-center gap-2">
                  <Server className="h-4 w-4" />
                  Instances ({activeInstances.length})
                </h2>
                <div className="space-y-2">
                  {activeInstances.map((inst) => (
                    <InstanceCard key={inst.id} inst={inst} />
                  ))}
                </div>
              </div>
            )}

            {/* GPU Marketplace */}
            <div>
              <h2 className="text-sm font-semibold text-zinc-300 mb-3 flex items-center gap-2">
                <Search className="h-4 w-4" />
                GPU Marketplace
              </h2>
              <OfferSearch />
            </div>

            {/* Destroyed instances (collapsed) */}
            {destroyedInstances.length > 0 && (
              <details className="group">
                <summary className="text-xs text-zinc-600 cursor-pointer hover:text-zinc-400">
                  {destroyedInstances.length} destroyed instance{destroyedInstances.length > 1 ? 's' : ''}
                </summary>
                <div className="mt-2 space-y-2 opacity-50">
                  {destroyedInstances.map((inst) => (
                    <InstanceCard key={inst.id} inst={inst} />
                  ))}
                </div>
              </details>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
