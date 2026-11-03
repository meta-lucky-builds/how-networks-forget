# H1 Stage A result: gate failed, H1 stopped

Code at commit 14ec174, run on Colab (CPU), 1139 s. 39 tests passed first. Tuned on
`mixed` with 30 configurations and seeds 0-9. Evaluated on seeds 100-149, paired.
`sanity_anomalies` was empty, and the Bayes filter converged at K = 16.

Per the preregistration, Stage B does not run.

## Gate

On `mixed`, a positive difference would favour `deferred`.

| `deferred` against | Mean MSE difference | 95% CI | Gate clause |
| --- | --- | --- | --- |
| `monotone` | -5.92 | -6.51 to -5.32 | Fail |
| `robust` | -6.41 | -6.90 to -5.91 | Fail |
| `deferred_nocheck` | +9.01 | +8.58 to +9.42 | Pass |
| `robust_bocd` (reported, not in gate) | -6.83 | -7.35 to -6.31 | |

## Mean MSE against the noise-free mean

| Learner | changepoint | outlier | mixed | stationary |
| --- | --- | --- | --- | --- |
| `fixed` | 2.918 | 5.810 | 9.275 | 0.435 |
| `monotone` | 2.177 | 6.598 | 9.325 | 0.434 |
| `robust` | 3.505 | 4.911 | 8.836 | 0.495 |
| `robust_bocd` | 3.712 | 2.219 | 8.416 | 0.005 |
| `deferred` | 12.870 | 1.302 | 15.250 | 0.481 |
| `deferred_nocheck` | 20.341 | 3.298 | 24.257 | 0.205 |
| `bayes` (ceiling) | 0.202 | 0.018 | 7.444 | 0.004 |
| `oracle_events` | 0.149 | 0.008 | 0.545 | 0.004 |

Gap closed on `mixed`: `robust_bocd` 0.469 (CI 0.396 to 0.540), `robust` 0.240,
`monotone` -0.027, `deferred` -3.264.

Chosen `deferred` config: alpha 0.575, commit_alpha 0.716, threshold 9.05, k 1,
window 4, tolerance 4.03.

## Exploratory analysis (post hoc, not preregistered)

Contribution to overall MSE on `mixed`, seeds 100-129, for four step types: the step
a changepoint occurs, the step after, an outlier step, all others. Each step counts
in the first bucket it matches.

| Learner | changepoint step | step after | outlier step | other | total |
| --- | --- | --- | --- | --- | --- |
| `fixed` | 2.61 | 0.71 | 4.50 | 1.30 | 9.12 |
| `robust` | 3.18 | 0.69 | 3.84 | 0.99 | 8.71 |
| `robust_bocd` | 3.85 | 0.54 | 3.61 | 0.17 | 8.18 |
| `deferred` | 12.43 | 1.15 | 0.56 | 0.80 | 14.95 |
| `bayes` | 3.58 | 0.41 | 3.18 | 0.13 | 7.30 |

## Interpretation

**Why it failed.** On arrival, a changepoint and an outlier look the same. A point
estimate must then pick a value. If the jump is Delta, and the two causes are about
equally likely, staying put costs Delta^2 when the world changed and 0 when it did
not, while the midpoint costs Delta^2 / 4 either way. Averaged, withholding costs
twice as much as hedging. The data match: changepoint-plus-outlier error is 12.99
for `deferred` and 6.76 for `bayes`.

**Search was not the problem.** With h = 0.05 and E[Delta^2] about 266 for two draws
from U(-20, 20), any policy that does not move on a large surprise pays about 13 per
step on changepoint steps alone. That already matches the best config found, so a
bigger search would not rescue it.

**This was foreseeable.** The bound takes a few lines, and we should have derived it
before preregistering. The Stage A gate, as written, could not be passed by any
withholding policy on this stream. That is a flaw in our reasoning, not a reason to
revisit the verdict: the claim as stated, that withholding beats arrival decisions
on tracking error, is false here.

**What did hold.**
- `deferred` beat every non-ceiling learner on `outlier` (1.30) and on outlier steps.
- Corroboration beat no corroboration, by a wide margin. Note `deferred_nocheck` also
  picked a different config, so this is not a pure single-factor contrast.

**What this says about biology.** Little. Deferral was our extrapolation, and the
human data never showed it.

## Scope limits

The stream is 1-D, with a jump size and event rates that make arrival-step error
dominate. A smaller Delta relative to noise, rarer events, or a cost function where
corrupting durable weights is worse than one bad prediction could all change the
balance. None of those were tested.

## Parked idea, not preregistered

H1b: separate what the learner predicts from what it commits to. Hedge the prediction
during ambiguity while deferring the durable weight update. In 1-D this is close to
a mixture filter and adds nothing new. Its only open form is a nonlinear network
running two weight hypotheses during ambiguity. It would need its own preregistration
and an analytic bound first. It waits behind H2, H3 and H4.

## Raw data

Per-seed CSVs stayed in the ephemeral Colab runtime. Moving them would need either a
repo credential in the notebook, which we avoid, or re-emitting thousands of rows
through the agent. The run is deterministic: `python experiments/h1_stage_a.py` at
14ec174 regenerates them. This departs from the charter's aim of storing raw outputs.
