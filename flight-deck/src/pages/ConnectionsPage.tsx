import { Plug } from 'lucide-react'
import GoogleConnection from '../components/connections/GoogleConnection'
import CodexConnection from '../components/connections/CodexConnection'
import MCPConnection from '../components/connections/MCPConnection'
import MCPAgentAccess from '../components/connections/MCPAgentAccess'
import TypesenseConnection from '../components/connections/TypesenseConnection'

// `kiosk` — the locked Simple layout's Connections dialog: a person's own
// connections only (their Google account, the deck's MCP servers). The ChatGPT
// subscription and the deep-memory server are deck setup, left to the full view.
export default function ConnectionsPage({ kiosk = false }: { kiosk?: boolean }) {
  return (
    <div className="h-full overflow-y-auto bg-zinc-950 text-zinc-200">
      <div className="max-w-3xl mx-auto px-6 py-8 pb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="h-10 w-10 rounded-md bg-zinc-900 border border-zinc-800 flex items-center justify-center">
            <Plug className="h-5 w-5 text-violet-500" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-zinc-100">Connections</h1>
            <p className="text-sm text-zinc-500">
              Link Captain Claw to external services so your agents can read and write on your behalf.
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <GoogleConnection />
          {!kiosk && <CodexConnection />}
          <MCPConnection />
          <MCPAgentAccess />
          {!kiosk && <TypesenseConnection />}
        </div>
      </div>
    </div>
  )
}
