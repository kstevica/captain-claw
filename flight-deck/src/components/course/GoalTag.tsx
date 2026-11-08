import { Compass } from 'lucide-react'
import { useGoalTitles } from './useGoalTitles'

/** "Serves: <goal>" — the Course goal something is for, by its title (nothing
 *  when there is none, or when the goal is gone). */
export function GoalTag({ goalId, prefix = 'Serves' }: { goalId?: string | null; prefix?: string }) {
  const titles = useGoalTitles()
  const title = goalId ? titles.get(goalId) : ''
  if (!title) return null
  return (
    <span
      className="inline-flex max-w-full items-center gap-1 truncate rounded bg-violet-500/10 px-1.5 py-0.5 text-[11px] font-medium text-violet-700 dark:text-violet-300"
      title={`${prefix}: ${title} (on your Course)`}
    >
      <Compass className="h-3 w-3 shrink-0" aria-hidden />
      <span className="truncate">{prefix}: {title}</span>
    </span>
  )
}
