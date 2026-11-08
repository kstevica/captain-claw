import { useEffect, useState } from 'react'
import { courseOnce } from '../../services/course'

/** Every goal the user can see, by id → title (empty until loaded, or when
 *  Course can't be read). */
export function useGoalTitles(): Map<string, string> {
  const [titles, setTitles] = useState<Map<string, string>>(() => new Map())
  useEffect(() => {
    let gone = false
    courseOnce()
      .then((c) => {
        if (gone) return
        const all = [...c.goals.deck, ...c.goals.mine, ...Object.values(c.goals.agents).flat()]
        setTitles(new Map(all.map((g) => [g.id, g.title])))
      })
      .catch(() => { /* no Course: no chips */ })
    return () => { gone = true }
  }, [])
  return titles
}
