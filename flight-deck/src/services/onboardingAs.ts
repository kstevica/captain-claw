/** The user an admin opened an agent's Onboarding for (Users → Agents → Onboarding): every onboarding
 *  call — lessons, practice, exams, knowledge — then names them (X-FD-Act-As), as Way of working's do,
 *  and what the admin teaches is that agent's base. Set while that dialog is open. */

import { ACT_AS_HEADER } from './docker'

let actingFor = ''

export const onboardingAs = {
  get: () => actingFor,
  set: (uid: string) => { actingFor = uid },
}

export function actAsHeaders(): Record<string, string> {
  return actingFor ? { [ACT_AS_HEADER]: actingFor } : {}
}
