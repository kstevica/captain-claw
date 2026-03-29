import { useEffect, useCallback } from 'react'
import { X, Keyboard } from 'lucide-react'
import { useUIStore } from '../../stores/uiStore'
import { useChatStore } from '../../stores/chatStore'

const shortcuts = [
  { keys: ['Cmd/Ctrl', '1'], action: 'Agent Desktop' },
  { keys: ['Cmd/Ctrl', '2'], action: 'Operations' },
  { keys: ['Cmd/Ctrl', '3'], action: 'Workflows' },
  { keys: ['Cmd/Ctrl', '4'], action: 'Spawn Agent' },
  { keys: ['Cmd/Ctrl', 'D'], action: 'Toggle Director' },
  { keys: ['Cmd/Ctrl', 'J'], action: 'Toggle Chat Panel' },
  { keys: ['Cmd/Ctrl', 'K'], action: 'Toggle Shortcuts Help' },
  { keys: ['Cmd/Ctrl', '['], action: 'Previous Chat Tab' },
  { keys: ['Cmd/Ctrl', ']'], action: 'Next Chat Tab' },
  { keys: ['Escape'], action: 'Close Modals / Panels' },
]

export function useKeyboardShortcuts(
  _directorOpen: boolean,
  onToggleDirector: () => void,
  shortcutsOpen: boolean,
  setShortcutsOpen: (v: boolean) => void,
) {
  const { setView } = useUIStore()
  const chatStore = useChatStore()

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    const mod = e.metaKey || e.ctrlKey
    if (!mod && e.key !== 'Escape') return

    // Don't interfere with text inputs
    const target = e.target as HTMLElement
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
      if (e.key === 'Escape') {
        target.blur()
        return
      }
      return
    }

    switch (e.key) {
      case '1':
        if (mod) { e.preventDefault(); setView('desktop') }
        break
      case '2':
        if (mod) { e.preventDefault(); setView('operations') }
        break
      case '3':
        if (mod) { e.preventDefault(); setView('workflow') }
        break
      case '4':
        if (mod) { e.preventDefault(); setView('spawner') }
        break
      case 'd':
        if (mod) { e.preventDefault(); onToggleDirector() }
        break
      case 'j':
        if (mod) {
          e.preventDefault()
          if (chatStore.chatOpen) chatStore.closeChat()
          else if (chatStore.activeChatId) useChatStore.setState({ chatOpen: true })
        }
        break
      case 'k':
        if (mod) { e.preventDefault(); setShortcutsOpen(!shortcutsOpen) }
        break
      case '[':
        if (mod) {
          e.preventDefault()
          switchChatTab(-1)
        }
        break
      case ']':
        if (mod) {
          e.preventDefault()
          switchChatTab(1)
        }
        break
      case 'Escape':
        if (shortcutsOpen) setShortcutsOpen(false)
        break
    }
  }, [setView, onToggleDirector, shortcutsOpen, setShortcutsOpen, chatStore])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])
}

function switchChatTab(direction: number) {
  const { sessions, activeChatId } = useChatStore.getState()
  const ids = Array.from(sessions.keys())
  if (ids.length < 2 || !activeChatId) return
  const currentIdx = ids.indexOf(activeChatId)
  const nextIdx = (currentIdx + direction + ids.length) % ids.length
  useChatStore.setState({ activeChatId: ids[nextIdx], chatOpen: true })
}

export function ShortcutsOverlay({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="w-[400px] rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
          <div className="flex items-center gap-2">
            <Keyboard className="h-4 w-4 text-violet-400" />
            <span className="text-sm font-semibold">Keyboard Shortcuts</span>
          </div>
          <button onClick={onClose} className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="p-4">
          <div className="space-y-2">
            {shortcuts.map((s, i) => (
              <div key={i} className="flex items-center justify-between">
                <span className="text-xs text-zinc-400">{s.action}</span>
                <div className="flex items-center gap-1">
                  {s.keys.map((key, j) => (
                    <span key={j}>
                      {j > 0 && <span className="text-zinc-700 mx-0.5">+</span>}
                      <kbd className="inline-block rounded border border-zinc-700 bg-zinc-900 px-1.5 py-0.5 text-[10px] font-mono text-zinc-300">
                        {key}
                      </kbd>
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="border-t border-zinc-800 px-4 py-2.5">
          <p className="text-[10px] text-zinc-600">Press Cmd/Ctrl+K to toggle this panel</p>
        </div>
      </div>
    </div>
  )
}
