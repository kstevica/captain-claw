import { useEffect } from 'react'
import { useScheduledStore } from '../stores/scheduledStore'
import { useUIStore } from '../stores/uiStore'
import { useConsoleStore } from '../stores/consoleStore'
import { isAppShell, isKioskOnly } from '../services/desktopShell'

// <deck>/#task=<id>: open Scheduled tasks on that task — Captain for Chrome's
// "Change or stop it in Flight Deck" for a Watch it check. Read at load, so a
// sign-in in between doesn't lose it; rendered once the person is in. Both
// layouts: the full one's Scheduled tasks page, the simple one's dialog.

const TASK = (() => {
  const m = window.location.hash.match(/^#task=([A-Za-z0-9_-]{1,64})$/)
  return m ? m[1] : ''
})()

export default function TaskDeepLink() {
  useEffect(() => {
    if (!TASK) return
    window.history.replaceState(null, '', window.location.pathname + window.location.search)
    useScheduledStore.getState().openTasks(TASK)
    useUIStore.getState().setView('scheduler')
    // A browser tab without ?app=classic has no Scheduled tasks page: an admin
    // in the Admin console goes to the users' kiosk, where its dialog is, for
    // this visit (a console address stays where it asked to be).
    if (isKioskOnly() && !isAppShell() && !useConsoleStore.getState().requested) {
      useUIStore.getState().visitLayout('simple')
    }
  }, [])
  return null
}
