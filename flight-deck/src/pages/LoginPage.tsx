import { useEffect, useState } from 'react'
import { ArrowRight, AtSign, CheckCircle2, KeyRound, Loader2, Lock, Mail, User, UserCheck } from 'lucide-react'
import { checkAuthStatus, joinWaitlist, loginUser, registerUser, useAuthStore } from '../stores/authStore'
import { AuthCard, AuthError, AuthField, AuthShell } from '../components/auth/AuthShell'

type Mode = 'login' | 'register' | 'waitlist' | 'applied'

export function LoginPage() {
  const signUp = useAuthStore((s) => s.signUp)
  const name = useAuthStore((s) => s.branding.name)
  // A deck with no account yet opens on creating the admin account.
  const firstRun = signUp.firstRun === true
  const [mode, setMode] = useState<Mode>(firstRun ? 'register' : 'login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [referrerName, setReferrerName] = useState('')
  const [referrerEmail, setReferrerEmail] = useState('')
  const [appliedCohort, setAppliedCohort] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const cohort = signUp.cohort
  // What "the other tab" is: creating an account, applying, or nothing at all.
  // On a first run there's no one to sign in as yet: no tabs.
  const signUpMode: Mode | null = firstRun ? null
    : signUp.mode === 'register' ? 'register' : signUp.mode === 'waitlist' ? 'waitlist' : null

  useEffect(() => {
    if (firstRun) setMode('register')
  }, [firstRun])

  const switchTo = (next: Mode) => {
    setMode(next)
    setError('')
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      if (mode === 'login') {
        await loginUser(email, password)
      } else if (mode === 'register') {
        await registerUser(email, password, displayName)
      } else if (mode === 'waitlist') {
        const applied = await joinWaitlist({
          email: email.trim(),
          password,
          display_name: displayName.trim(),
          referrer_name: referrerName.trim(),
          referrer_email: referrerEmail.trim(),
        })
        setAppliedCohort(applied || cohort)
        setMode('applied')
      }
    } catch (err: any) {
      setError(err.message || 'Something went wrong')
      if (mode === 'register') {
        // Registration closed since this page loaded: offer what the deck takes now.
        try {
          await checkAuthStatus()
          if (useAuthStore.getState().signUp.mode === 'waitlist') setMode('waitlist')
        } catch { /* keep the error */ }
      }
    } finally {
      setLoading(false)
    }
  }

  if (mode === 'applied') {
    return (
      <AuthShell>
        <AuthCard>
          <div className="flex flex-col items-center text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-emerald-400/10 ring-1 ring-emerald-300/30">
              <CheckCircle2 className="h-7 w-7 text-emerald-300" />
            </div>
            <h2 className="mt-5 text-2xl font-semibold tracking-tight text-white">You're on the waitlist</h2>
            <p className="mt-3 text-sm leading-relaxed text-white/65">
              Thanks{displayName.trim() ? `, ${displayName.trim()}` : ''}. Your application
              {appliedCohort ? <> for <span className="font-medium text-white">{appliedCohort}</span></> : null} is in.
            </p>
            <p className="mt-2 text-sm leading-relaxed text-white/50">
              An administrator will review it. Once you're approved, sign in with{' '}
              <span className="text-white/85">{email.trim()}</span> and the password you chose.
            </p>
          </div>
          <button
            type="button"
            onClick={() => { setPassword(''); switchTo('login') }}
            className="fd-auth-primary mt-7 flex h-11 w-full items-center justify-center gap-2 rounded-xl text-sm font-medium text-white"
          >
            Back to sign in
            <ArrowRight className="h-4 w-4" />
          </button>
        </AuthCard>
      </AuthShell>
    )
  }

  const heading =
    mode === 'login' ? 'Welcome back'
      : mode === 'register' ? (firstRun ? 'Create the admin account' : 'Create your account')
        : 'Join the waitlist'
  const subheading =
    mode === 'login'
      ? `Sign in to ${name}.`
      : mode === 'register'
        ? (firstRun
          ? `${name} is new. The first account is its administrator: you set up its models, people and connections.`
          : `Set up your account on ${name}.`)
        : cohort
          ? <>Sign-up is by application. Apply for <span className="font-medium text-violet-200">{cohort}</span> — an administrator reviews each one.</>
          : 'Sign-up is by application — an administrator reviews each one.'
  const submitLabel =
    mode === 'login' ? 'Sign in' : mode === 'register' ? (firstRun ? 'Create admin account' : 'Create account')
      : cohort ? `Apply for ${cohort}` : 'Apply'

  return (
    <AuthShell>
      <AuthCard>
        {signUpMode && (
          <div role="tablist" aria-label="Sign in or sign up" className="mb-7 grid grid-cols-2 rounded-xl bg-white/[0.05] p-1 ring-1 ring-white/[0.07]">
            {(['login', signUpMode] as Mode[]).map((m) => (
              <button
                key={m}
                type="button"
                role="tab"
                aria-selected={mode === m}
                onClick={() => switchTo(m)}
                className={`h-9 rounded-lg text-sm font-medium transition-all ${
                  mode === m
                    ? 'bg-white/[0.12] text-white shadow-sm ring-1 ring-white/10'
                    : 'text-white/50 hover:text-white/80'
                }`}
              >
                {m === 'login' ? 'Sign in' : m === 'register' ? 'Create account' : 'Join waitlist'}
              </button>
            ))}
          </div>
        )}

        <div className="mb-6">
          <h2 className="text-2xl font-semibold tracking-tight text-white">{heading}</h2>
          <p className="mt-1.5 text-sm leading-relaxed text-white/55">{subheading}</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {error && <AuthError>{error}</AuthError>}

          {(mode === 'register' || mode === 'waitlist') && (
            <AuthField
              label={mode === 'waitlist' ? 'Your name' : 'Display name'}
              icon={User}
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="Your name"
              autoComplete="name"
              required={mode === 'waitlist'}
              maxLength={100}
            />
          )}

          <AuthField
            label="Email"
            icon={Mail}
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            autoComplete={mode === 'login' ? 'username' : 'email'}
            required
            autoFocus
          />

          <AuthField
            label="Password"
            icon={Lock}
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder={mode === 'login' ? 'Your password' : 'At least 6 characters'}
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            required
            minLength={6}
            hint={mode === 'waitlist' ? "You'll sign in with this once you're approved." : undefined}
          />

          {mode === 'waitlist' && (
            <fieldset className="space-y-4 pt-2">
              <legend className="sr-only">Who recommended you</legend>
              <div className="flex items-center gap-3 pb-1">
                <div className="h-px flex-1 bg-white/10" />
                <span className="flex items-center gap-1.5 text-xs font-medium text-white/60">
                  <UserCheck className="h-3.5 w-3.5" />
                  Who recommended you?
                </span>
                <div className="h-px flex-1 bg-white/10" />
              </div>
              <AuthField
                label="Their name"
                icon={User}
                value={referrerName}
                onChange={(e) => setReferrerName(e.target.value)}
                placeholder="Who recommended you"
                required
                maxLength={100}
              />
              <AuthField
                label="Their email"
                icon={AtSign}
                type="email"
                value={referrerEmail}
                onChange={(e) => setReferrerEmail(e.target.value)}
                placeholder="them@example.com"
                required
              />
            </fieldset>
          )}

          <button
            type="submit"
            disabled={loading}
            className="fd-auth-primary mt-2 flex h-11 w-full items-center justify-center gap-2 rounded-xl text-sm font-medium text-white"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            {loading ? 'Please wait…' : submitLabel}
            {!loading && <ArrowRight className="h-4 w-4" />}
          </button>
        </form>

        {/* Sign-up closed and no waitlist: an administrator creates accounts. */}
        {mode === 'login' && !signUpMode && (
          <div className="mt-6 flex items-start gap-3 rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-3.5">
            <KeyRound className="mt-0.5 h-4 w-4 shrink-0 text-violet-200/80" />
            <div className="text-sm leading-relaxed">
              <p className="font-medium text-white/85">Sign-up is closed</p>
              <p className="mt-0.5 text-white/50">
                Accounts on {name} are created by an administrator. Ask yours to set one up for you.
              </p>
            </div>
          </div>
        )}
      </AuthCard>
    </AuthShell>
  )
}
