import { useState } from 'react'
import {
  GitBranch,
  Plus,
  Trash2,
  X,
  ChevronDown,
  ChevronRight,
  GripVertical,
  Power,
  ArrowRight,
  Edit3,
  Box,
  Settings2 as Cog,
} from 'lucide-react'
import { usePipelineStore, type Pipeline, type PipelineStep } from '../../stores/pipelineStore'
import { useContainerStore } from '../../stores/containerStore'
import { useLocalAgentStore } from '../../stores/localAgentStore'

export function PipelineBuilder({ onClose }: { onClose: () => void }) {
  const { pipelines, createPipeline, deletePipeline, updatePipeline, addStep, removeStep, updateStep } = usePipelineStore()
  const containers = useContainerStore((s) => s.containers)
  const localAgents = useLocalAgentStore((s) => s.agents)

  const [newName, setNewName] = useState('')
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const allAgents = [
    ...containers.filter((c) => c.web_port).map((c) => ({
      id: c.id, name: c.agent_name || c.name, kind: 'docker' as const,
    })),
    ...localAgents.map((a) => ({ id: a.id, name: a.name, kind: 'local' as const })),
  ]

  const handleCreate = () => {
    if (!newName.trim()) return
    const id = createPipeline(newName.trim())
    setNewName('')
    setExpandedId(id)
  }

  const agentName = (id: string) => allAgents.find((a) => a.id === id)?.name || id.slice(0, 8)
  const agentKind = (id: string) => allAgents.find((a) => a.id === id)?.kind || 'docker'

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2.5">
        <div className="flex items-center gap-2">
          <GitBranch className="h-4 w-4 text-emerald-400" />
          <span className="text-sm font-semibold">Pipelines</span>
          <span className="rounded-full bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400">{pipelines.length}</span>
        </div>
        <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Create new */}
      <div className="border-b border-zinc-800 p-2.5">
        <div className="flex gap-1.5">
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
            placeholder="New pipeline name..."
            className="flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
          />
          <button
            onClick={handleCreate}
            disabled={!newName.trim()}
            className="rounded-md bg-violet-600 px-2 py-1 text-xs text-white hover:bg-violet-500 disabled:opacity-40"
          >
            <Plus className="h-3 w-3" />
          </button>
        </div>
      </div>

      {/* Pipeline list */}
      <div className="flex-1 overflow-y-auto">
        {pipelines.length === 0 ? (
          <div className="px-3 py-8 text-center text-xs text-zinc-600">
            No pipelines yet. Create one to chain agent outputs automatically.
          </div>
        ) : (
          <div className="divide-y divide-zinc-800/60">
            {pipelines.map((pipeline) => (
              <PipelineCard
                key={pipeline.id}
                pipeline={pipeline}
                expanded={expandedId === pipeline.id}
                onToggle={() => setExpandedId(expandedId === pipeline.id ? null : pipeline.id)}
                onDelete={() => deletePipeline(pipeline.id)}
                onToggleEnabled={() => updatePipeline(pipeline.id, { enabled: !pipeline.enabled })}
                onRename={(name) => updatePipeline(pipeline.id, { name })}
                onAddStep={(step) => addStep(pipeline.id, step)}
                onRemoveStep={(idx) => removeStep(pipeline.id, idx)}
                onUpdateStep={(idx, patch) => updateStep(pipeline.id, idx, patch)}
                allAgents={allAgents}
                agentName={agentName}
                agentKind={agentKind}
              />
            ))}
          </div>
        )}
      </div>

      {/* Info */}
      <div className="border-t border-zinc-800 px-3 py-2 text-[10px] text-zinc-600">
        When an agent in step N responds, its output is automatically forwarded to the agent in step N+1.
      </div>
    </div>
  )
}

function PipelineCard({
  pipeline, expanded, onToggle, onDelete, onToggleEnabled, onRename,
  onAddStep, onRemoveStep, onUpdateStep, allAgents, agentName, agentKind,
}: {
  pipeline: Pipeline
  expanded: boolean
  onToggle: () => void
  onDelete: () => void
  onToggleEnabled: () => void
  onRename: (name: string) => void
  onAddStep: (step: PipelineStep) => void
  onRemoveStep: (idx: number) => void
  onUpdateStep: (idx: number, patch: Partial<PipelineStep>) => void
  allAgents: { id: string; name: string; kind: 'docker' | 'local' }[]
  agentName: (id: string) => string
  agentKind: (id: string) => string
}) {
  const [renaming, setRenaming] = useState(false)
  const [nameInput, setNameInput] = useState(pipeline.name)
  const [addingStep, setAddingStep] = useState(false)
  const [editingPrompt, setEditingPrompt] = useState<number | null>(null)
  const [promptInput, setPromptInput] = useState('')

  const handleRename = () => {
    if (nameInput.trim()) onRename(nameInput.trim())
    setRenaming(false)
  }

  return (
    <div className={`${expanded ? 'bg-zinc-900/30' : 'hover:bg-zinc-900/20'}`}>
      <div className="flex items-center gap-2 px-3 py-2 cursor-pointer" onClick={onToggle}>
        {expanded ? <ChevronDown className="h-3 w-3 text-zinc-600" /> : <ChevronRight className="h-3 w-3 text-zinc-600" />}

        <div className="min-w-0 flex-1">
          {renaming ? (
            <input
              value={nameInput}
              onChange={(e) => setNameInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleRename(); if (e.key === 'Escape') setRenaming(false) }}
              onBlur={handleRename}
              className="w-full rounded border border-zinc-700 bg-zinc-950 px-1 py-0.5 text-xs text-zinc-200 focus:outline-none"
              onClick={(e) => e.stopPropagation()}
              autoFocus
            />
          ) : (
            <span className="text-xs font-medium text-zinc-200">{pipeline.name}</span>
          )}
          <div className="flex items-center gap-1 mt-0.5">
            <span className="text-[10px] text-zinc-600">{pipeline.steps.length} step{pipeline.steps.length !== 1 ? 's' : ''}</span>
            {pipeline.steps.length > 0 && (
              <span className="text-[10px] text-zinc-600 truncate max-w-[200px]">
                {pipeline.steps.map((s) => agentName(s.agentId)).join(' → ')}
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
          <button
            onClick={onToggleEnabled}
            className={`rounded p-0.5 ${pipeline.enabled ? 'text-emerald-400' : 'text-zinc-600'}`}
            title={pipeline.enabled ? 'Disable' : 'Enable'}
          >
            <Power className="h-3 w-3" />
          </button>
          <button onClick={() => { setRenaming(true); setNameInput(pipeline.name) }} className="rounded p-0.5 text-zinc-600 hover:text-zinc-300">
            <Edit3 className="h-3 w-3" />
          </button>
          <button onClick={onDelete} className="rounded p-0.5 text-zinc-600 hover:text-red-400">
            <Trash2 className="h-3 w-3" />
          </button>
        </div>
      </div>

      {expanded && (
        <div className="px-3 pb-3">
          {/* Steps */}
          <div className="space-y-1.5 ml-3">
            {pipeline.steps.map((step, i) => (
              <div key={i}>
                <div className="flex items-center gap-1.5">
                  <GripVertical className="h-3 w-3 text-zinc-700 shrink-0" />
                  <div className="flex items-center gap-1 rounded-lg border border-zinc-800 bg-zinc-950/50 px-2 py-1 flex-1">
                    <span className="text-[10px] font-mono text-zinc-600 w-4">{i + 1}.</span>
                    {agentKind(step.agentId) === 'docker'
                      ? <Box className="h-3 w-3 text-blue-400/70 shrink-0" />
                      : <Cog className="h-3 w-3 text-amber-400/70 shrink-0" />}
                    <span className="text-xs text-zinc-300 flex-1 truncate">{agentName(step.agentId)}</span>
                    {step.prompt && (
                      <span className="text-[10px] text-violet-400 truncate max-w-[80px]" title={step.prompt}>
                        +prompt
                      </span>
                    )}
                    <button
                      onClick={() => { setEditingPrompt(editingPrompt === i ? null : i); setPromptInput(step.prompt || '') }}
                      className="rounded p-0.5 text-zinc-600 hover:text-zinc-300"
                      title="Edit prompt prefix"
                    >
                      <Edit3 className="h-2.5 w-2.5" />
                    </button>
                    <button onClick={() => onRemoveStep(i)} className="rounded p-0.5 text-zinc-600 hover:text-red-400">
                      <X className="h-2.5 w-2.5" />
                    </button>
                  </div>
                </div>

                {/* Prompt editor */}
                {editingPrompt === i && (
                  <div className="ml-8 mt-1">
                    <textarea
                      value={promptInput}
                      onChange={(e) => setPromptInput(e.target.value)}
                      placeholder="Optional: prompt to prepend before forwarded output..."
                      className="w-full resize-none rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1 text-[10px] text-zinc-200 placeholder-zinc-600 focus:border-violet-500/50 focus:outline-none"
                      rows={2}
                    />
                    <div className="mt-0.5 flex gap-1">
                      <button
                        onClick={() => { onUpdateStep(i, { prompt: promptInput.trim() || undefined }); setEditingPrompt(null) }}
                        className="rounded bg-violet-600 px-1.5 py-0.5 text-[10px] text-white hover:bg-violet-500"
                      >
                        Save
                      </button>
                      <button onClick={() => setEditingPrompt(null)} className="rounded px-1.5 py-0.5 text-[10px] text-zinc-500">
                        Cancel
                      </button>
                    </div>
                  </div>
                )}

                {/* Arrow between steps */}
                {i < pipeline.steps.length - 1 && (
                  <div className="flex items-center ml-8 my-0.5">
                    <ArrowRight className="h-3 w-3 text-zinc-700" />
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Add step */}
          {addingStep ? (
            <div className="mt-2 ml-3 flex flex-wrap gap-1">
              {allAgents.map((a) => (
                <button
                  key={a.id}
                  onClick={() => { onAddStep({ agentId: a.id }); setAddingStep(false) }}
                  className="flex items-center gap-1 rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-300 hover:bg-violet-600/20 hover:text-violet-400"
                >
                  {a.kind === 'docker' ? <Box className="h-2.5 w-2.5 text-blue-400/70" /> : <Cog className="h-2.5 w-2.5 text-amber-400/70" />}
                  {a.name}
                </button>
              ))}
              <button onClick={() => setAddingStep(false)} className="rounded px-1.5 py-0.5 text-[10px] text-zinc-600 hover:text-zinc-300">
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setAddingStep(true)}
              className="mt-2 ml-3 flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
            >
              <Plus className="h-3 w-3" />
              Add Step
            </button>
          )}
        </div>
      )}
    </div>
  )
}
