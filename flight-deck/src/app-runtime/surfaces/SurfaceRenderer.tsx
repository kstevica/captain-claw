import type { AgentManifest, SurfaceDef } from '../types'
import { EntityList } from '../primitives/EntityList'
import { EntityDetail } from '../primitives/EntityDetail'
import { ActionButton } from '../primitives/ActionButton'
import { InboxList } from '../primitives/InboxList'
import { Upload } from '../primitives/Upload'

interface Props {
  manifest: AgentManifest
  surface: SurfaceDef
}

export function SurfaceRenderer({ manifest, surface }: Props) {
  if (surface.layout === 'upload') {
    return (
      <Upload
        manifest={manifest}
        accept={surface.accept}
        multiple={surface.multiple}
        actionIds={surface.sources ?? []}
      />
    )
  }

  if (surface.layout === 'inbox') {
    return <InboxList manifest={manifest} feedIds={surface.sources ?? []} />
  }

  if (surface.layout === 'list') {
    // Auto-render every feed declared in `sources`.
    return (
      <div className="space-y-4">
        {(surface.sources ?? []).map((fid) => {
          const feed = manifest.feeds[fid]
          if (!feed) return null
          return <EntityList key={fid} manifest={manifest} feed={feed} />
        })}
      </div>
    )
  }

  // dashboard or entity — both use sections; entity also shows a detail header.
  const sections = surface.sections ?? []
  const actions = sections.filter((s) => s.type === 'action')
  const feeds = sections.filter((s) => s.type === 'feed')

  return (
    <div className="space-y-4">
      {surface.layout === 'entity' && surface.entity && (
        <EntityDetail manifest={manifest} entity={manifest.entities[surface.entity]} />
      )}

      {actions.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {actions.map((sec) => {
            const action = manifest.actions[sec.id]
            if (!action) return null
            return (
              <ActionButton
                key={sec.id}
                manifest={manifest}
                action={action}
                prefill={sec.prefill}
                prominent={sec.prominent}
              />
            )
          })}
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {feeds.map((sec) => {
          const feed = manifest.feeds[sec.id]
          if (!feed) return null
          return <EntityList key={sec.id} manifest={manifest} feed={feed} filter={sec.filter} />
        })}
      </div>
    </div>
  )
}
