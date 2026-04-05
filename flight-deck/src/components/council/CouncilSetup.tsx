import { useState, useRef } from 'react'
import { Swords, Lightbulb, ClipboardCheck, Map, Play, Paperclip, X, MessageCircleQuestion, Bug, ShieldAlert, MessagesSquare } from 'lucide-react'
import { AgentPicker, isOldManName } from './AgentPicker'
import { formatSize } from '../../services/fileTransfer'
import type {
  CouncilAgentDef, CreateSessionConfig, SessionType, Verbosity,
} from '../../stores/councilStore'

const SESSION_TYPES: { id: SessionType; label: string; icon: typeof Swords; desc: string }[] = [
  { id: 'debate', label: 'Debate', icon: Swords, desc: 'Structured argumentation' },
  { id: 'brainstorm', label: 'Brainstorm', icon: Lightbulb, desc: 'Creative ideation' },
  { id: 'review', label: 'Review', icon: ClipboardCheck, desc: 'Critical analysis' },
  { id: 'planning', label: 'Planning', icon: Map, desc: 'Task decomposition' },
  { id: 'interview', label: 'Interview', icon: MessageCircleQuestion, desc: 'Knowledge extraction' },
  { id: 'troubleshoot', label: 'Troubleshoot', icon: Bug, desc: 'Problem diagnosis' },
  { id: 'critique', label: 'Critique', icon: ShieldAlert, desc: 'Adversarial stress-test' },
  { id: 'freeform', label: 'Freeform', icon: MessagesSquare, desc: 'Open conversation' },
]

const VERBOSITY_OPTIONS: { id: Verbosity; label: string; desc: string }[] = [
  { id: 'thought', label: 'Thought', desc: '1-2 sentences' },
  { id: 'message', label: 'Message', desc: 'Up to 5 sentences' },
  { id: 'short', label: 'Short', desc: 'Up to 3 paragraphs' },
  { id: 'medium', label: 'Medium', desc: 'Up to 5 paragraphs' },
  { id: 'long', label: 'Long', desc: 'Up to 10 paragraphs' },
]

interface CouncilSetupProps {
  onStart: (cfg: CreateSessionConfig) => void
  onCancel: () => void
}

export function CouncilSetup({ onStart, onCancel }: CouncilSetupProps) {
  const [title, setTitle] = useState('')
  const [topic, setTopic] = useState('')
  const [sessionType, setSessionType] = useState<SessionType>('brainstorm')
  const [verbosity, setVerbosity] = useState<Verbosity>('message')
  const [maxRounds, setMaxRounds] = useState(5)
  const [agents, setAgents] = useState<CouncilAgentDef[]>([])
  const [firstSpeaker, setFirstSpeaker] = useState('random')
  const [files, setFiles] = useState<File[]>([])
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const oldMan = agents.find(a => isOldManName(a.name))
  const moderatorMode = oldMan ? 'moderator' : 'round-robin'
  const canStart = topic.trim() && agents.length >= 2

  const addFiles = (newFiles: FileList | File[]) => {
    const arr = Array.from(newFiles)
    setFiles(prev => {
      const names = new Set(prev.map(f => f.name))
      return [...prev, ...arr.filter(f => !names.has(f.name))]
    })
  }

  const removeFile = (name: string) => setFiles(prev => prev.filter(f => f.name !== name))

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files)
  }

  const handleStart = () => {
    if (!canStart) return
    onStart({
      title: title.trim() || topic.slice(0, 60),
      topic: topic.trim(),
      sessionType,
      verbosity,
      maxRounds,
      moderatorMode,
      moderatorAgentId: oldMan?.id || '',
      agents,
      firstSpeaker,
      files,
    })
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      {/* Title */}
      <div>
        <label className="mb-1 block text-xs font-medium text-zinc-400">Title (optional)</label>
        <input
          value={title}
          onChange={e => setTitle(e.target.value)}
          placeholder="Council session title..."
          className="w-full rounded-lg border border-zinc-700/50 bg-zinc-800/50 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-500 focus:border-violet-500/50 focus:outline-none"
        />
      </div>

      {/* Topic */}
      <div>
        <label className="mb-1 block text-xs font-medium text-zinc-400">Topic / Task *</label>
        <textarea
          value={topic}
          onChange={e => setTopic(e.target.value)}
          placeholder="What should the council discuss or work on?"
          rows={3}
          className="w-full rounded-lg border border-zinc-700/50 bg-zinc-800/50 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-500 focus:border-violet-500/50 focus:outline-none resize-none"
        />
      </div>

      {/* Session Type */}
      <div>
        <label className="mb-2 block text-xs font-medium text-zinc-400">Session Type</label>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {SESSION_TYPES.map(t => (
            <button
              key={t.id}
              onClick={() => setSessionType(t.id)}
              className={`flex flex-col items-center gap-1 rounded-lg border p-3 text-center transition-colors ${
                sessionType === t.id
                  ? 'border-violet-500/50 bg-violet-500/10 text-violet-300'
                  : 'border-zinc-700/50 bg-zinc-800/30 text-zinc-400 hover:bg-zinc-700/30'
              }`}
            >
              <t.icon className="h-5 w-5" />
              <span className="text-xs font-medium">{t.label}</span>
              <span className="text-[10px] opacity-60">{t.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Verbosity */}
      <div>
        <label className="mb-2 block text-xs font-medium text-zinc-400">Response Verbosity</label>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {VERBOSITY_OPTIONS.map(v => (
            <button
              key={v.id}
              onClick={() => setVerbosity(v.id)}
              className={`rounded-lg border px-3 py-2 text-center transition-colors ${
                verbosity === v.id
                  ? 'border-violet-500/50 bg-violet-500/10 text-violet-300'
                  : 'border-zinc-700/50 bg-zinc-800/30 text-zinc-400 hover:bg-zinc-700/30'
              }`}
            >
              <div className="text-xs font-medium">{v.label}</div>
              <div className="text-[10px] opacity-60">{v.desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Max Rounds */}
      <div>
        <label className="mb-1 block text-xs font-medium text-zinc-400">
          Max Rounds: <span className="text-violet-400">{maxRounds}</span>
        </label>
        <input
          type="range"
          min={1}
          max={20}
          value={maxRounds}
          onChange={e => setMaxRounds(parseInt(e.target.value))}
          className="w-full accent-violet-500"
        />
        <div className="flex justify-between text-[10px] text-zinc-500">
          <span>1</span><span>10</span><span>20</span>
        </div>
      </div>

      {/* Agent Picker */}
      <div>
        <label className="mb-2 block text-xs font-medium text-zinc-400">
          Select Agents ({agents.length} selected)
          {oldMan && <span className="ml-2 text-amber-400">Moderator mode (Old Man detected)</span>}
          {!oldMan && agents.length >= 2 && <span className="ml-2 text-zinc-500">Round-robin mode</span>}
        </label>
        <AgentPicker selected={agents} onChange={setAgents} />
      </div>

      {/* File Attachments */}
      <div>
        <label className="mb-2 block text-xs font-medium text-zinc-400">
          Attachments {files.length > 0 && <span className="text-zinc-500">({files.length} file{files.length !== 1 ? 's' : ''})</span>}
        </label>
        <div
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`flex cursor-pointer flex-col items-center gap-1 rounded-lg border border-dashed p-4 text-center transition-colors ${
            dragOver
              ? 'border-violet-500 bg-violet-500/10'
              : 'border-zinc-700/50 bg-zinc-800/30 hover:border-zinc-600 hover:bg-zinc-700/20'
          }`}
        >
          <Paperclip className="h-5 w-5 text-zinc-500" />
          <span className="text-xs text-zinc-400">Drop files here or click to browse</span>
          <span className="text-[10px] text-zinc-500">Files will be uploaded to all agents at council start</span>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={e => { if (e.target.files?.length) { addFiles(e.target.files); e.target.value = '' } }}
        />
        {files.length > 0 && (
          <div className="mt-2 space-y-1">
            {files.map(f => (
              <div key={f.name} className="flex items-center gap-2 rounded-md bg-zinc-800/50 px-2 py-1.5 text-xs">
                <Paperclip className="h-3.5 w-3.5 flex-shrink-0 text-zinc-500" />
                <span className="min-w-0 flex-1 truncate text-zinc-300">{f.name}</span>
                <span className="flex-shrink-0 text-zinc-500">{formatSize(f.size)}</span>
                <button
                  onClick={e => { e.stopPropagation(); removeFile(f.name) }}
                  className="flex-shrink-0 rounded p-0.5 text-zinc-500 hover:bg-zinc-700 hover:text-zinc-300"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* First Speaker */}
      {agents.length >= 2 && (
        <div>
          <label className="mb-1 block text-xs font-medium text-zinc-400">First Speaker</label>
          <select
            value={firstSpeaker}
            onChange={e => setFirstSpeaker(e.target.value)}
            className="w-full rounded-lg border border-zinc-700/50 bg-zinc-800/50 px-3 py-2 text-sm text-zinc-200 focus:border-violet-500/50 focus:outline-none"
          >
            <option value="random">Random</option>
            {agents.filter(a => a.id !== oldMan?.id).map(a => (
              <option key={a.id} value={a.id}>{a.name}</option>
            ))}
          </select>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-3 pt-2">
        <button
          onClick={handleStart}
          disabled={!canStart}
          className="flex items-center gap-2 rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Play className="h-4 w-4" /> Start Council
        </button>
        <button
          onClick={onCancel}
          className="rounded-lg border border-zinc-700/50 px-4 py-2 text-sm text-zinc-400 hover:bg-zinc-700/30"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}
