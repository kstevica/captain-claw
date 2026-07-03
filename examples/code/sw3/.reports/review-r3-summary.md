# Review summary — r3

A blocking same-frame race condition cancels the peasant's move-to-build order, leaving it idle during construction. Three additional major bugs cause building drift, a search radius off-by-one, and pathfinding that ignores wall blocking. All other findings from security and QA are low severity and do not block release.