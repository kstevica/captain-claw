import type { AgentManifest } from '../types'
import { EntityList } from './EntityList'

interface Props {
  manifest: AgentManifest
  feedIds: string[]
}

export function InboxList({ manifest, feedIds }: Props) {
  return (
    <div className="space-y-4">
      {feedIds.map((fid) => {
        const feed = manifest.feeds[fid]
        if (!feed) return null
        return <EntityList key={fid} manifest={manifest} feed={feed} />
      })}
    </div>
  )
}
