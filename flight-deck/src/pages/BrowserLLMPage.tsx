// Browser LLM — this tab as an inference worker for your Mrav agents.
// The panel logic lives in components/system/LocalInferencePanel; this page
// adds the model catalog and the how-it-works context.
import { Bug, Cpu, Globe, Zap } from 'lucide-react'
import { LocalInferencePanel } from '../components/system/LocalInferencePanel'
import { MODEL_CATALOG, extraCatalogModels, webgpuAvailable } from '../inference/localInference'
import { useInferenceStore } from '../stores/inferenceStore'

const STEPS = [
  {
    Icon: Globe,
    title: 'This tab serves tokens',
    text: 'Pick a model and press Start: WebLLM loads it onto your GPU (WebGPU, cached after the first download) and the tab registers with Flight Deck as your personal inference worker.',
  },
  {
    Icon: Cpu,
    title: 'Agents route here',
    text: 'Any agent whose provider is "browser" — e.g. a micro tier set to browser — sends its LLM calls through Flight Deck to this tab. Tools and state stay on the server; only tokens are made here.',
  },
  {
    Icon: Zap,
    title: 'Built for Mrav',
    text: 'The engine runs a 9216-token window (8k input + 1k output), matching the micro runtime’s cap. JSON is grammar-enforced in the tab, so even small models keep the step protocol.',
  },
]

export function BrowserLLMPage() {
  const { status, modelId } = useInferenceStore()
  const gpu = webgpuAvailable()

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-6xl space-y-5 p-6">
        <div className="flex items-center gap-3">
          <Bug className="h-5 w-5 text-violet-400" />
          <div>
            <h1 className="text-lg font-semibold text-zinc-100">Browser LLM</h1>
            <p className="text-xs text-zinc-500">
              Serve your agents from this tab — no server GPU needed. Worker lives as long as the tab does.
            </p>
          </div>
        </div>

        <LocalInferencePanel />

        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          {STEPS.map(({ Icon, title, text }) => (
            <div key={title} className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-4">
              <div className="mb-1.5 flex items-center gap-2">
                <Icon className="h-3.5 w-3.5 text-violet-400" />
                <h2 className="text-sm font-medium text-zinc-200">{title}</h2>
              </div>
              <p className="text-xs leading-relaxed text-zinc-500">{text}</p>
            </div>
          ))}
        </div>

        <div className="rounded-lg border border-zinc-800 bg-zinc-950/50 p-4">
          <h2 className="mb-1 text-sm font-medium text-zinc-200">Models</h2>
          <p className="mb-3 text-xs text-zinc-500">
            GPU memory shown is the catalog reference at a 4k window — the 9k engine window adds some KV on top.
            Weights download once and are cached by the browser.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-zinc-800 text-[10px] uppercase tracking-wide text-zinc-500">
                  <th className="py-1.5 pr-4 font-medium">Model</th>
                  <th className="py-1.5 pr-4 font-medium">GPU mem</th>
                  <th className="py-1.5 font-medium">Notes</th>
                </tr>
              </thead>
              <tbody>
                {MODEL_CATALOG.map((m) => (
                  <tr key={m.id} className="border-b border-zinc-900">
                    <td className="py-1.5 pr-4 font-medium text-zinc-300">
                      {m.label}
                      {status !== 'off' && modelId === m.id && (
                        <span className="ml-2 rounded border border-emerald-500/25 bg-emerald-600/15 px-1 py-px text-[10px] text-emerald-400">loaded</span>
                      )}
                    </td>
                    <td className="py-1.5 pr-4 tabular-nums text-zinc-400">{m.vramGB} GB</td>
                    <td className="py-1.5 text-zinc-500">{m.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[11px] text-zinc-600">
            + {extraCatalogModels().length} more chat models in the full-catalog list (Llama, Mistral, Hermes, DeepSeek-R1 distills, SmolLM2…).
            Qwen3.6 ships only as 27B/35B — beyond what a browser tab can hold.
          </p>
        </div>

        {!gpu && (
          <div className="rounded-lg border border-amber-500/30 bg-amber-500/[0.06] px-4 py-3 text-xs text-amber-400">
            WebGPU is not available in this browser — use a current Chrome, Edge, Safari 26+ or Firefox.
          </div>
        )}
      </div>
    </div>
  )
}
