import { AppHost } from '../app-runtime/AppHost'

// Entry point for the agent-app runtime. Manifests are owned by
// Captain Claw and fetched from /fd/apps — the renderer never bundles
// them.
export function AppPage() {
  return <AppHost />
}
