# H2R Stage 0: failed, no retry allowed, H2 line closed

**Verdict: no full run, no retry, and the H2 line closes.** H2a and H2b were not tested, so the
shared-state claim is neither supported nor killed.

Run 41754260090, seeds 300 to 302, disjoint from evaluation seeds. Data: `stage0.json`.

| Check | Measured | Threshold | Result |
| --- | --- | --- | --- |
| Oracle minus sgd zero-shot, short gap | 0.239 | at least 0.5 | **fail** |
| Oracle minus sgd zero-shot, long gap | 0.597 | at least 0.5 | pass |
| Forgetting, sgd | 0.558 | at least 0.1 | pass |
| sgd last-quarter AUC, largest seed | 0.937 | below 1 | pass |
| sgd last minus first quarter AUC, largest seed | -0.173 | below 0 | pass |
| Protect minus sgd last-quarter AUC, smallest and largest seed | +0.280, +0.676 | below 0, all seeds | **fail** |
| Censored `reuse` events, sgd | 0% | under 20% | pass |
| Gate SD | 0.577 | at least 0.05 | pass |
| Gate Spearman | 0.975 | at least 0.5 | pass |

## How the rule was applied

Deviation 2 allows a retry "only if every failed check is plasticity or tension". Short-gap retention
is neither, so no retry runs.

**One reading to flag.** The text says a second failure closes the line, and does not name the case
of a first failure with no retry allowed. It does say that missing any one check means the full run
does not start, and that no third design is preregistered. I read that as closing the line. Any
return to this question needs a new preregistration under a new name.

## My threshold error

The failed retention check could not have passed. Mean sgd short-gap zero-shot was 0.544, so a gap of
0.5 needs an oracle above 1.0. I set one threshold for both gaps before seeing that sgd keeps
over half its accuracy across a short gap. **The rule stands anyway**: changing a threshold after
seeing it fail, in order to get a retry, is what preregistration exists to prevent. The tension check also
failed in all 3 seeds, independently of this error.

## Descriptive only

Stage 0 data is not used in any verdict, and n is 3. These inform future designs.

| Condition | Rate | Long-gap zero-shot, mean | Last-quarter AUC, mean | Online accuracy, mean |
| --- | --- | --- | --- | --- |
| sgd | 0.0148 | 0.211 | 0.852 | 0.655 |
| protect | 0.0054 | 0.177 | 1.273 | 0.702 |
| both_util | 0.0072 | 0.150 | 1.148 | 0.675 |
| both_perm | 0.0072 | 0.132 | 1.075 | 0.677 |
| oracle | 0.0148 | 0.808 | 1.000 | 0.686 |

- **The tension is not what H2 assumed.** Protection did not lose plasticity relative to sgd here.
  This is confounded: tuning gave protect about a third of sgd's rate, and a lower rate alone could
  raise AUC. So it does not show that protection preserves plasticity.
- **The rank gate did not retain.** No gated condition beat sgd on long-gap zero-shot, against an
  oracle near 0.81. A mean-1 gate with half of each layer at a multiplier above 1 may not protect
  enough. That is a guess, not a measured cause.
- **Selection on online accuracy may work against the question.** Tuning chooses rates for
  online accuracy, which does not reward retention, and that also applies to H2.

Stage 0 took about 49 minutes including tuning, and avoided a full run projected at about 287.
