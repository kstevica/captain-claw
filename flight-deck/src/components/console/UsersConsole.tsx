import { useCallback, useEffect, useState } from 'react'
import { UserCog, RefreshCw } from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { useConsoleStore } from '../../stores/consoleStore'
import { UsersAdmin } from '../admin/UsersAdmin'
import { InvitesAdmin } from '../admin/InvitesAdmin'
import { WaitlistAdmin } from '../admin/WaitlistAdmin'
import {
  CONFIG_DEFAULTS, fetchConfig, kioskModeNote, type SystemConfig, type UsersTab,
} from '../admin/deckConfig'
import { Page, PageHeader, Button, Segmented } from './parts'

// Users: every account, the invitations out and the waitlist — the Users
// page's three lists (UsersPage.tsx), all kept mounted so the counts are right
// and a half-written invite survives switching.

export function UsersConsole({ onWaitlistCount }: { onWaitlistCount: (n: number) => void }) {
  const signUpMode = useAuthStore((s) => s.signUp.mode)
  // Sent here for one list (the Overview's "waiting for approval"): open on it.
  const [tab, setTab] = useState<UsersTab>(() => useConsoleStore.getState().intent.usersTab ?? 'users')
  const [config, setConfig] = useState<SystemConfig>({ ...CONFIG_DEFAULTS })
  const [loading, setLoading] = useState(true)
  const [refreshTick, setRefreshTick] = useState(0)
  const [usersTick, setUsersTick] = useState(0)
  const [total, setTotal] = useState<number | null>(null)
  const [openInvites, setOpenInvites] = useState<number | null>(null)
  const [pending, setPending] = useState<number | undefined>(undefined)

  useEffect(() => {
    useConsoleStore.getState().takeIntent()
    // …or sent here while already on this page.
    return useConsoleStore.subscribe((s) => {
      if (!s.intent.usersTab) return
      setTab(s.intent.usersTab)
      useConsoleStore.getState().takeIntent()
    })
  }, [])

  useEffect(() => {
    fetchConfig().then(setConfig).catch(() => {}).finally(() => setLoading(false))
  }, [refreshTick])
  const refresh = () => { setLoading(true); setRefreshTick((t) => t + 1) }

  const onPending = useCallback((n: number) => { setPending(n); onWaitlistCount(n) }, [onWaitlistCount])
  const onApproved = useCallback(() => setUsersTick((t) => t + 1), [])

  return (
    <Page>
      <PageHeader
        icon={UserCog}
        title="Users"
        description="Who is on the deck and who is coming: add or invite people, set their plan and role, lock them to the simple chat, approve the waitlist."
        actions={<Button icon={RefreshCw} busy={loading} onClick={refresh}>Refresh</Button>}
      />

      <Segmented
        value={tab}
        onChange={setTab}
        label="Lists"
        options={[
          { value: 'users', label: 'Users', count: total ?? undefined },
          { value: 'invitations', label: 'Invitations', count: openInvites ?? undefined },
          { value: 'waitlist', label: 'Waitlist', count: pending },
        ]}
      />

      <div hidden={tab !== 'users'}>
        <UsersAdmin refreshTick={refreshTick + usersTick} config={config} onTotal={setTotal} />
      </div>
      <div hidden={tab !== 'invitations'}>
        <InvitesAdmin
          refreshTick={refreshTick}
          kioskDefault={config.kiosk_new_users}
          kioskNote={kioskModeNote(config.kiosk_mode)}
          onOpenCount={setOpenInvites}
        />
      </div>
      <div hidden={tab !== 'waitlist'}>
        <WaitlistAdmin
          refreshTick={refreshTick}
          kioskDefault={config.kiosk_new_users}
          kioskNote={kioskModeNote(config.kiosk_mode)}
          onPendingCount={onPending}
          onApproved={onApproved}
          note={
            signUpMode === 'register'
              ? 'Self-registration is open (FD_REGISTRATION_OPEN), so the sign-up page creates accounts directly and takes no applications.'
              : !config.waitlist_enabled
                ? 'The waitlist is off (Settings → Sign-up), so the sign-up page takes no new applications.'
                : null
          }
        />
      </div>
    </Page>
  )
}
