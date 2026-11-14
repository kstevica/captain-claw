import { useEffect, useState } from 'react'
import { Check, ClipboardList, HardDrive } from 'lucide-react'
import { Button, Card, Field, inputCls } from '../console/parts'
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
    <Card
      title="Project theme"
      icon={ClipboardList}
      hint="Description + instructions are sent to every run."
      action={dirty && !isUnfiled ? (
        <Button
          variant="primary"
          icon={Check}
          busy={saving}
          onClick={() => onSave({ name: name.trim() || project.name, description, instructions })}
        >
          Save changes
        </Button>
      ) : undefined}
    >
      {isUnfiled ? (
        <p className="text-sm leading-relaxed text-zinc-500">
          Unfiled is a bucket for runs that don't belong to a project — it has no theme or shared folder.
          Create a project to bundle runs with a shared theme and files.
        </p>
      ) : (
        <div className="space-y-4">
          <Field label="Name">
            <input value={name} onChange={(e) => setName(e.target.value)} className={inputCls} />
          </Field>
          <Field label={<>Description <span className="font-normal text-zinc-600">— the shared theme</span></>}>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
              placeholder="What this project is about — prepended to every run."
              className={`${inputCls} resize-y leading-relaxed`}
            />
          </Field>
          <Field label={<>Additional instructions <span className="font-normal text-zinc-600">— extra guidance for every run</span></>}>
            <textarea
              value={instructions}
              onChange={(e) => setInstructions(e.target.value)}
              rows={5}
              placeholder="Constraints, format, focus areas — appended to each run's task."
              className={`${inputCls} resize-y leading-relaxed`}
            />
          </Field>
          {project.vfs_folder && (
            <p className="flex items-center gap-1.5 text-xs text-zinc-500">
              <HardDrive className="h-3.5 w-3.5 shrink-0" />
              Shared folder <span className="truncate font-mono text-zinc-400">{project.vfs_folder}</span>
            </p>
          )}
        </div>
      )}
    </Card>
  )
}
