import { useEffect, useState } from 'react'
import { Check, Loader2 } from 'lucide-react'
import type { BasnaProject } from '../../stores/basnaProjectStore'

// Details tab for a project: edit the theme (name / description / instructions)
// that seeds every run. Files + Datastore live in their own top-bar tabs.
export function ProjectDetails({ project, saving, onSave }: {
  project: BasnaProject
  saving?: boolean
  onSave: (fields: { name: string; description: string; instructions: string }) => void
}) {
  const [name, setName] = useState(project.name)
  const [description, setDescription] = useState(project.description)
  const [instructions, setInstructions] = useState(project.instructions)
  useEffect(() => {
    setName(project.name); setDescription(project.description); setInstructions(project.instructions)
  }, [project.id]) // eslint-disable-line react-hooks/exhaustive-deps

  const dirty = name !== project.name || description !== project.description || instructions !== project.instructions
  const isUnfiled = !project.vfs_folder

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="mb-3 flex items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Project theme</span>
        <span className="text-[11px] text-zinc-600">description + instructions are sent to every run</span>
        {dirty && !isUnfiled && (
          <button
            onClick={() => onSave({ name: name.trim() || project.name, description, instructions })}
            disabled={saving}
            className="ml-auto flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-40"
          >
            {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
            Save changes
          </button>
        )}
      </div>
      {isUnfiled ? (
        <p className="text-xs text-zinc-500">
          Unfiled is a bucket for runs that don't belong to a project — it has no theme or shared folder.
          Create a project to bundle runs with a shared theme and files.
        </p>
      ) : (
        <div className="space-y-3">
          <div>
            <label className="mb-1 block text-[11px] font-medium text-zinc-400">Name</label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full rounded-lg border border-zinc-700 bg-zinc-950/60 px-2.5 py-1.5 text-sm text-zinc-200 focus:border-sky-600 focus:outline-none"
            />
          </div>
          <div>
            <label className="mb-1 block text-[11px] font-medium text-zinc-400">Description <span className="font-normal text-zinc-600">— the shared theme</span></label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
              placeholder="What this project is about — prepended to every run."
              className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950/60 px-2.5 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none"
            />
          </div>
          <div>
            <label className="mb-1 block text-[11px] font-medium text-zinc-400">Additional instructions <span className="font-normal text-zinc-600">— extra guidance for every run</span></label>
            <textarea
              value={instructions}
              onChange={(e) => setInstructions(e.target.value)}
              rows={5}
              placeholder="Constraints, format, focus areas — appended to each run's task."
              className="w-full resize-y rounded-lg border border-zinc-700 bg-zinc-950/60 px-2.5 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-sky-600 focus:outline-none"
            />
          </div>
        </div>
      )}
    </div>
  )
}
