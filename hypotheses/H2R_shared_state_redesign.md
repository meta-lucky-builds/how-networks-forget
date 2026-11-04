# H2R: H2 redesigned so that it can answer

**Status:** preregistered before any H2R code. H2 data informed this design, per
`results/h2_shared_state/RESULT.md`, so H2's evaluation seeds are not reused. Later
changes go in the Deviations section.

The hypotheses are H2a and H2b as stated in `H2_shared_synaptic_state.md`, and the
biological grades there still apply: evidence is indirect, and a positive result supports
an engineering claim only.

## Bounds, derived before design

1. **Input permutation removes the zero-shot ceiling.** H2's ceiling
   (`acc_old <= A + 1 - a`) came from identical inputs carrying conflicting labels. With
   pixels permuted per task and labels fixed, inputs differ between tasks, so one function
   can in principle be right on all of them. The limit becomes capacity and interference.
2. **A multiplicative gate has two effects, and only one is of interest.** With
   `P = 1 - u`, a mean gate `m` rescales the rate by `m` and the spread reallocates it. A
   gate normalised to mean 1 removes the rescale, so a difference from SGD cannot be a
   learning-rate effect.
3. **Rank gates have a known spread.** With `u` the within-layer rank in (0, 1), `gate_std`
   of `P = 2(1 - u)` is 2/sqrt(12) = 0.58 by construction. The power rule is met by design.

## Design changes from H2

| H2 defect | H2R |
| --- | --- |
| Inputs identical, zero-shot ceiling near 0.2 | Input-permuted MNIST, labels fixed |
| Gate nearly constant, acts as rate scale | `u` = within-layer rank of the trace, recomputed every 10 steps. `P = 2(1 - u)`, so mean 1 |
| Savings censored, confounded with speed | `reuse`: steps to a fixed rolling accuracy on a repeat, minus the same on the two nearest non-repeat tasks, same run |
| Binary mask as null | `both_perm`: the same gate values, re-permuted within layer at each refresh. Exactly the same distribution, no relation to `u` |

Keeps from H2: online, batch size 1, 1000 steps per task, 100 tasks, MLP 784-128-128-10,
10% repeats split between short (5 to 10) and long (at least 40) gaps, disjoint image pools,
probe AUC, and tuning by 10 random configurations on 2 seeds over 40 tasks, selected on
mean online accuracy.

Conditions: `sgd`, `protect` (`P` only), `noise_util` (noise gated by `R = 1 - u`),
`both_util`, `both_perm`, `both_multi` (three traces, `u` from the max of their within-layer
ranks, re-ranked, so spread stays matched), `shrink_perturb`, and `oracle` in Stage 0 only.

Seeds: tuning 200 to 201, Stage 0 300 to 302, evaluation 400 to 409.

## Stage 0: can this stream answer the question?

Run only `sgd`, `protect`, `both_util`, `both_perm` and `oracle`, on 3 seeds, full length.
`oracle` is `sgd` that restores its saved weights from the source task's end at each repeat:
the positive control, showing what perfect retention scores on these metrics.

Pass requires all of these. The metric thresholds are set in advance, and missing any one
means the full run does not start:

- **Retention can move.** `oracle` zero-shot at least 0.5 above `sgd` zero-shot at both gaps.
- **There is forgetting.** `sgd` zero-shot at long gap at least 0.1 under its accuracy at the
  end of the source task.
- **There is plasticity loss.** `sgd` probe AUC in the last quarter of the stream below 1, and
  below its first-quarter value, in all 3 seeds.
- **There is tension.** `protect` probe AUC below `sgd` in all 3 seeds, in the last quarter.
- **`reuse` is measurable.** Under 20% of `reuse` events censored for `sgd`, with the
  target rolling accuracy of 0.75 fixed for all conditions.
- **The gate is real.** `both_util` gate SD at least 0.05, and the gate rank order is stable
  over a refresh: mean Spearman at least 0.5 between consecutive refreshes. A gate that is
  noise cannot express utility.

Failing the plasticity or tension items allows **one** retry at a higher learning rate for
`sgd` (a preset multiple of 3). Failing again closes the H2 line as "no stream of this form
in budget", recorded in the results and with the mission moving on. Stage 0 data is
not used in the verdict.

## Predictions and kill criteria

Evaluation runs 10 paired seeds, with 95% bootstrap intervals.

**Manipulation check (as in H2).** `both_util` must retain better than `noise_util`, and
`protect` must lose probe AUC relative to `both_util`, each with the CI excluding 0. If this
fails, no conclusion follows, and the H2 line closes: a third design is not preregistered
in advance.

**H2a supported** if `both_util` minus `both_perm` improves zero-shot or `reuse`, with the CI
excluding 0 and probe AUC not falling by more than one pooled SE.

**H2a killed** if `both_util` and `both_perm` match within one pooled SE on both retention
measures and on AUC. Since spread is guaranteed here, this kill is informative, unlike in H2.

**H2b supported** if `both_multi` beats `both_util` on long-gap `reuse` with the CI excluding 0,
no AUC loss beyond one pooled SE, and no comparable short-gap advantage (the CI excludes 0 and
the difference is at least half the long-gap one). **H2b killed** if `both_util`'s tuned trace
matches or beats it on long-gap `reuse` within one pooled SE.

Multiple comparisons: H2a allows two retention measures, so its verdict uses Holm-adjusted
intervals at family level 0.05.

## Scope limits

Rank gates depart from UPGD's published form, so a result here is about this rank-gated
rule. Rank discards magnitude, so a real effect that depends on utility size could be missed.
This is a CPU-sized stream.

## Deviations

None yet.
