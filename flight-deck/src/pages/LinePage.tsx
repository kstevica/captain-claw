import { LinePanel } from '../components/line/LinePanel'

// The full view's Line: the same panel the kiosk opens as a dialog.
export function LinePage() {
  return (
    <div className="flex h-full min-h-0 justify-center overflow-hidden bg-zinc-950">
      <div className="flex h-full w-full max-w-6xl min-h-0 flex-col xl:border-x xl:border-zinc-800/60">
        <LinePanel />
      </div>
    </div>
  )
}
