import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import 'katex/dist/katex.min.css'
import App from './App'
import { NetworkShareDialog } from './components/agents/NetworkShareDialog'
import { CustomToolAsks } from './components/customTools/CustomToolAsks'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
    <NetworkShareDialog />
    <CustomToolAsks />
  </StrictMode>,
)
