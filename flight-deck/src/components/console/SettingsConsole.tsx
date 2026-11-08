import { useEffect, useState } from 'react'
import { Settings2, ClipboardList, MessageSquare, Boxes, Save, Wrench, Archive } from 'lucide-react'
import { AgentFilesSigning } from '../layout/AgentBackupsDialog'
import { useAuthStore } from '../../stores/authStore'
import { useConsoleStore } from '../../stores/consoleStore'
import { BrandingAdmin } from '../admin/BrandingAdmin'
import { Switch } from '../admin/Switch'
import { IdleAgentsSetting, IDLE_AGENTS_HINT, IDLE_AGENTS_TITLE } from '../admin/IdleAgentsSetting'
import { DeckProfileDefaults } from '../profile/DeckProfileDefaults'
import { DeckCourseSettings } from '../course/DeckCourseSettings'
import {
  CUSTOM_TOOLS_HINT, CUSTOM_TOOLS_TITLE, EVERYBODY_APPROVAL_HINT, EVERYBODY_APPROVAL_TITLE, SANDBOX_HINT, SANDBOX_TITLE, sandboxLine,
} from '../admin/customTools'
import { useCustomToolsStore } from '../../stores/customToolsStore'
import {
  CONFIG_DEFAULTS, fetchConfig, saveConfig, waitlistName, type KioskMode, type SystemConfig,
} from '../admin/deckConfig'
import { Page, PageHeader, Card, SettingRow, Segmented, Notice, Button, inputCls } from './parts'

// Settings: how the deck looks and who gets in — branding, sign-up and the
// waitlist, kiosk mode, how agents run, and what every agent knows about the
// company. The same system config as Admin → Settings.

const KIOSK_HINT: Record<KioskMode, string> = {
  off: 'Everyone gets the normal layout.',
  all: 'Every non-admin is locked to the simple chat.',
  selected: 'Only the users marked on the Users page are locked.',
}

export function SettingsConsole() {
  const go = useConsoleStore((s) => s.go)
  const signUpMode = useAuthStore((s) => s.signUp.mode)
  // No sign-in (the desktop app): nobody signs up and nobody is locked.
  const accounts = useAuthStore((s) => s.authEnabled !== false)
  const [config, setConfig] = useState<SystemConfig>({ ...CONFIG_DEFAULTS })
  const [loaded, setLoaded] = useState(false)
  const [cohort, setCohort] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')
  // What this host offers custom tools (the sandbox), from their status.
  const toolStatus = useCustomToolsStore((s) => s.status)
  const refreshToolStatus = useCustomToolsStore((s) => s.refreshStatus)
  useEffect(() => { refreshToolStatus() }, [refreshToolStatus])

  useEffect(() => {
    fetchConfig()
      .then((c) => { setConfig(c); setCohort(c.waitlist_cohort) })
      .catch(() => setError("Couldn't load the deck's settings"))
      .finally(() => setLoaded(true))
  }, [])

  // Save one change; what the app shows elsewhere follows it.
  const save = async (key: string, patch: Partial<SystemConfig>) => {
    setBusy(key); setError('')
    try {
      const next = await saveConfig(patch)
      setConfig(next)
      const auth = useAuthStore.getState()
      if ('docker_spawn_enabled' in patch) auth.setDockerSpawnEnabled(next.docker_spawn_enabled)
      if ('kiosk_mode' in patch) auth.setSimpleChatOnly(next.kiosk_mode === 'all')
      if ('waitlist_enabled' in patch || 'waitlist_cohort' in patch) {
        const mode = auth.signUp.mode === 'register' ? 'register' : next.waitlist_enabled ? 'waitlist' : 'closed'
        auth.setSignUp({ mode, cohort: waitlistName(next) })
      }
      return next
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save')
      return null
    } finally {
      setBusy('')
    }
  }

  return (
    <Page>
      <PageHeader
        icon={Settings2}
        title="Settings"
        description={accounts
          ? 'How the deck looks and who gets in: its name and logo, sign-up and the waitlist, kiosk mode, and what every agent knows about your company.'
          : 'How the deck looks, how agents run, and what every agent knows about your company.'}
      />

      {error && <Notice tone="red">{error}</Notice>}

      {loaded && (
        <BrandingAdmin
          name={config.instance_name}
          description={config.instance_description}
          poweredBy={config.show_powered_by}
          onSave={async (patch) => {
            const next = await saveConfig(patch)
            setConfig(next)
            const { signUp, setSignUp } = useAuthStore.getState()
            setSignUp({ ...signUp, cohort: waitlistName(next) })
            return next
          }}
        />
      )}

      {accounts && <Card
        icon={ClipboardList}
        title="Sign-up"
        hint={signUpMode === 'register'
          ? 'Self-registration is open on this deck (FD_REGISTRATION_OPEN): anyone can create an account, so there is no waitlist.'
          : 'Registration is closed. People apply on the waitlist, or you add and invite them on the Users page.'}
        bodyClassName="divide-y divide-zinc-800/80 px-5 py-4"
      >
        <SettingRow
          title="Take waitlist applications"
          hint={signUpMode === 'register'
            ? 'Not used while self-registration is open: people create their own accounts instead.'
            : <>The sign-up page takes applications: name, email, a password and who recommended them. You approve them on{' '}
              <button onClick={() => go('users', { usersTab: 'waitlist' })} className="text-violet-600 underline underline-offset-2 hover:no-underline dark:text-violet-400">Users → Waitlist</button>.
              Off: the sign-in page says an administrator creates accounts.</>}
        >
          <Switch
            on={config.waitlist_enabled && signUpMode !== 'register'}
            disabled={busy === 'waitlist' || signUpMode === 'register'}
            label="Take waitlist applications"
            onToggle={() => save('waitlist', { waitlist_enabled: !config.waitlist_enabled })}
          />
        </SettingRow>
        <div className="py-4 last:pb-0">
          <div className="text-sm font-medium text-zinc-200">Waitlist name</div>
          <div className="mt-0.5 text-xs text-zinc-500">
            Shown on the sign-up page and saved with each application. Blank uses the deck's name ({config.instance_name}).
          </div>
          <form
            className="mt-3 flex gap-2"
            onSubmit={async (e) => {
              e.preventDefault()
              const next = await save('cohort', { waitlist_cohort: cohort })
              if (next) setCohort(next.waitlist_cohort)
            }}
          >
            <input value={cohort} onChange={(e) => setCohort(e.target.value)} maxLength={60} placeholder={config.instance_name} className={inputCls} />
            <Button type="submit" icon={Save} busy={busy === 'cohort'} disabled={cohort.trim() === config.waitlist_cohort}>Save</Button>
          </form>
        </div>
      </Card>}

      {accounts && <Card
        icon={MessageSquare}
        title="Kiosk mode"
        hint="Locks people to the simple, chat-first layout: they chat, start an agent from an archetype, manage their own connections and profile, and sign out. Admins are never locked; a change reaches people already signed in within about 15 minutes."
        bodyClassName="divide-y divide-zinc-800/80 px-5 py-4"
      >
        <div className="pb-4">
          <Segmented
            value={config.kiosk_mode}
            label="Kiosk mode"
            onChange={(m) => { if (m !== config.kiosk_mode || config.kiosk_mode_from_env) save('kiosk', { kiosk_mode: m }) }}
            options={[
              { value: 'off', label: 'Off' },
              { value: 'all', label: 'Everyone' },
              { value: 'selected', label: 'Selected users' },
            ]}
          />
          <p className="mt-2 text-xs text-zinc-500">
            {KIOSK_HINT[config.kiosk_mode]}
            {config.kiosk_mode_from_env && ' Set by --simple-chat until you pick a mode here.'}
          </p>
        </div>
        <SettingRow
          title="New users start in kiosk mode"
          hint={'Marks accounts created from now on (added, invited, signed up or approved) — they count under "Selected users".'}
        >
          <Switch
            on={config.kiosk_new_users}
            disabled={busy === 'kiosk-new'}
            label="New users start in kiosk mode"
            onToggle={() => save('kiosk-new', { kiosk_new_users: !config.kiosk_new_users })}
          />
        </SettingRow>
      </Card>}

      <Card icon={Boxes} title="Agents" bodyClassName="divide-y divide-zinc-800/80 px-5 py-4">
        <SettingRow
          title="Docker containers"
          hint="Let people start agents as Docker containers. Off: agents run as processes on this machine only."
        >
          <Switch
            on={config.docker_spawn_enabled}
            disabled={busy === 'docker'}
            label="Docker containers"
            onToggle={() => save('docker', { docker_spawn_enabled: !config.docker_spawn_enabled })}
          />
        </SettingRow>
        <SettingRow title={IDLE_AGENTS_TITLE} hint={IDLE_AGENTS_HINT}>
          <IdleAgentsSetting
            minutes={config.agent_idle_stop_minutes}
            busy={busy === 'idle'}
            onSave={(m) => save('idle', { agent_idle_stop_minutes: m })}
          />
        </SettingRow>
      </Card>

      <Card icon={Wrench} title={CUSTOM_TOOLS_TITLE} hint={sandboxLine(toolStatus?.sandbox)} bodyClassName="divide-y divide-zinc-800/80 px-5 py-4">
        <SettingRow title="On for this deck" hint={CUSTOM_TOOLS_HINT}>
          <Switch
            on={config.external_tools_enabled}
            disabled={busy === 'xtools'}
            label={CUSTOM_TOOLS_TITLE}
            onToggle={() => save('xtools', { external_tools_enabled: !config.external_tools_enabled })}
          />
        </SettingRow>
        {accounts && <SettingRow title={EVERYBODY_APPROVAL_TITLE} hint={EVERYBODY_APPROVAL_HINT}>
          <Switch
            on={config.external_tools_everybody_approval}
            disabled={busy === 'xtools-approval'}
            label={EVERYBODY_APPROVAL_TITLE}
            onToggle={() => save('xtools-approval', { external_tools_everybody_approval: !config.external_tools_everybody_approval })}
          />
        </SettingRow>}
        <SettingRow title={SANDBOX_TITLE} hint={SANDBOX_HINT}>
          <Switch
            on={config.external_tools_require_sandbox}
            disabled={busy === 'xtools-sandbox'}
            label={SANDBOX_TITLE}
            onToggle={() => save('xtools-sandbox', { external_tools_require_sandbox: !config.external_tools_require_sandbox })}
          />
        </SettingRow>
      </Card>

      <Card icon={Archive} title="Agent files"
        hint="Agents exported from this deck are signed. Choose which other decks' files people here may import.">
        <AgentFilesSigning />
      </Card>

      {/* What every user's agents know about the company — loads on its own. */}
      <DeckProfileDefaults />

      {/* Course: the horizons each level uses and the limits. */}
      <DeckCourseSettings />
    </Page>
  )
}
