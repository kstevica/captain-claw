# Review summary — r2

The deposit handler race prevents peasants from releasing carried resources at auto-built dropoffs, causing infinite stuck loops. Additionally, gathering progress is reset on collision, auto-build can double-deduct resources, and under-construction checks ignore distance. A fix round is needed to resolve these correctness issues.