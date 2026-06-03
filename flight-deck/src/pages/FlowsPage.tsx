import { useFlowsStore } from '../stores/flowsStore'
import { FlowsPanel } from '../components/flows/FlowsPanel'
import { FlowBuilder } from '../components/flows/FlowBuilder'
import { FlowRunLog } from '../components/flows/FlowRunLog'

export function FlowsPage() {
  const view = useFlowsStore((s) => s.view)

  return (
    <div className="flex h-full flex-col">
      {view === 'list' && <FlowsPanel />}
      {view === 'builder' && <FlowBuilder />}
      {view === 'runlog' && <FlowRunLog />}
    </div>
  )
}
