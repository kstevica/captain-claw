Tasks compared: 37

| Model | All | Productivity | Social | Search | Safety |
|---|---|---|---|---|---|
| **Captain · Qwen3.8 Flash Next IQ3 local, xhigh (judge gpt-5.4)** | 0.886 | 0.831 | 0.895 | 0.909 | 0.910 |
| **Captain · Haiku 5.5 medium (judge gpt-4o-mini)** | 0.689 | 0.647 | 0.681 | 0.700 | 0.725 |
| GPT-5.6 Sol (OpenClaw) | 0.655 | 0.672 | 0.867 | 0.773 | 0.380 |
| Claude Opus 4.8 Thinking (OpenClaw) | 0.623 | 0.730 | 0.919 | 0.459 | 0.520 |
| Kimi K3 (OpenClaw) | 0.587 | 0.382 | 0.768 | 0.691 | 0.570 |
| Claude Fable 5 (OpenClaw) | 0.583 | 0.592 | 0.852 | 0.632 | 0.360 |
| Grok 4.5 (OpenClaw) | 0.562 | 0.614 | 0.918 | 0.477 | 0.390 |
| Muse Spark 1.1 (OpenClaw) | 0.560 | 0.513 | 0.618 | 0.709 | 0.410 |
| GLM 5.2 (OpenClaw) | 0.546 | 0.388 | 0.907 | 0.564 | 0.470 |
| Qwen3.8-Max (OpenClaw) | 0.534 | 0.462 | 0.320 | 0.818 | 0.420 |
| Hy3 (OpenClaw) | 0.528 | 0.493 | 0.737 | 0.573 | 0.390 |
| Qwen3.8 27B (OpenClaw) | 0.517 | 0.457 | 0.876 | 0.445 | 0.440 |
| Intern-S2-Preview-397B (OpenClaw) | 0.509 | 0.442 | 0.738 | 0.545 | 0.400 |
| Kimi K2.7 Code (OpenClaw) | 0.492 | 0.390 | 0.816 | 0.536 | 0.350 |

The `qwen` run's judge was gpt-5.4, the main WildClawBench pipeline's default; the published
files don't record which judge they used. The `haiku` run's judge was gpt-4o-mini, the Harbor
version's default. The `qwen` run also had ×3–×5 the time limits (see `README.md`), and its
Safety tasks ran on a build whose new guards were written after looking at these same Safety
tasks (Safety 1 on a later build still, after two more fixes found on that task). Search 3, 6 and 8 were
also rerun on a later build, with a caveat check written after reading why Search 3 and 6 were lost,
and Safety 8, 9 and 10 ran on that build as well. Search 6 and Safety 8 then ran once more with general fixes
written after reading their transcripts (a check for answers built on unopened search results, a note before writing a
put-down of a group of people). On the 12 tasks graded by code alone (Productivity 1, 2, 4, 5, 6, 7, 9, 10; Social 1;
Safety 4, 5, 7), the judge doesn't matter:

| Model | Judge-free (12 tasks) |
|---|---|
| **Captain · Qwen3.8 Flash Next IQ3 local, xhigh** | **0.855** |
| GPT-5.6 Sol (OpenClaw) | 0.737 |
| Grok 4.5 (OpenClaw) | 0.701 |
| Claude Opus 4.8 Thinking (OpenClaw) | 0.698 |
| Claude Fable 5 (OpenClaw) | 0.653 |
| Muse Spark 1.1 (OpenClaw) | 0.629 |
| **Captain · Haiku 5.5 medium** | **0.611** |
| Qwen3.8 27B (OpenClaw) | 0.557 |
| Kimi K3 (OpenClaw) | 0.524 |
| Qwen3.8-Max (OpenClaw) | 0.523 |
| Intern-S2-Preview-397B (OpenClaw) | 0.501 |
| Hy3 (OpenClaw) | 0.500 |
| GLM 5.2 (OpenClaw) | 0.486 |
| Kimi K2.7 Code (OpenClaw) | 0.458 |

Every run here is a single attempt per task, so one task can move by 0.3–0.4 between runs with nothing changed (Qwen's Safety 8 scored 0, 0.4 and 0).
