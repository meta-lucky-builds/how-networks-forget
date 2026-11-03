# H1: Corroboration-gated plasticity

**Status:** preregistered, not yet run. Committed before any H1 code exists.
Later changes go in the Deviations section at the end.

## Hypothesis

A surprising sample has an ambiguous cause: the world changed, or the sample is
an outlier. A single-sample signal cannot separate them, because the two look
identical on arrival. In a nonlinear SGD learner on a stream mixing both causes,
withholding weight commits for surprising samples until later data corroborates
them reduces tracking error, relative to any policy that decides on arrival.

## Biological basis, stated honestly

| Claim | Grade | Source |
| --- | --- | --- |
| Surprise raises human learning rate in changepoint contexts and lowers it in oddball contexts (n=39, interaction beta 0.71, t 9.9) | Established in one lab's EEG study; numbers read via summary | Nassar 2019 |
| Participants were told the context, with instruction and training, so this shows context-dependent use of surprise, not inference of cause | Corrects our earlier reading | Nassar 2019 |
| Humans defer learning until corroboration | **No evidence found.** Deferral is our engineering extrapolation, not a biological finding | `docs/literature/prior_art_corroboration.md` |
| Latent cause attribution accumulates gradually over evidence | Modelling literature; no buffered deferral shown | Gershman et al. |
| Fast store plus slow learner | Established in outline; selection rule contested | `docs/literature/consolidation.md` |

A positive result would therefore support an engineering method informed by the
biology. It would not be evidence that brains work this way.

## Prior art and the gap tested

The principle is published in other settings (Shinohara 2609.10927, Yiu 2604.14322,
Wang 2512.12289, Chen & Wu 2508.06385, Oda 2608.19888). The surviving gap is
selective deferred commits in a nonlinear SGD learner, compared head to head with
robust-plus-changepoint filtering and monotone gating, on a stream mixing label
noise with true change in P(y|x). Label-noise methods (SPR, CNLL, PuriDivER) never
change the label function. Strongest rival: WoLF-IMQ+BOCD (Duran-Martin et al.,
BONE). Our version is a reimplementation of that method, labelled as such.

## Stage A: 1-D tracking, exact ceiling (gate for Stage B)

Stream: y_t = m_t + N(0, sigma^2), sigma = 1. Changepoint with hazard h = 0.05
redraws m_t from U(-R, R), R = 20. Outlier with probability p = 0.05 replaces y_t
with a draw from U(-R, R). Contexts: changepoint-only, outlier-only, mixed,
stationary. T = 2000 steps.

Learners:
- `fixed`: delta rule, tuned alpha.
- `monotone`: alpha increases with |error|.
- `robust`: IMQ-weighted Kalman update (never adapts upward).
- `robust_bocd`: IMQ weighting plus run-length changepoint posterior, decides on arrival.
- `deferred`: delta rule that quarantines surprising samples, committing when K of
  the next W agree with each other and disagree with the current estimate.
- `deferred_nocheck` (ablation): same delay, always commits.
- `bayes`: Rao-Blackwellised Gaussian mixture filter over {continue, outlier,
  changepoint}, top-K pruned, K large enough that doubling it changes MSE by
  under 1%. This is the ceiling.
- `oracle_events`: told true event types.

Metric: mean squared error against the noise-free m_t. Gap closed =
(MSE_fixed - MSE_x) / (MSE_fixed - MSE_bayes).

**Stage A gate.** In the mixed context, over 50 seeds, `deferred` must beat both
`monotone` and `robust` with paired bootstrap 95% CI excluding 0, and must beat
`deferred_nocheck`. If it fails, H1 stops here and no Stage B runs.
`robust_bocd` is not part of the gate: in 1-D a good filter may already be near
the ceiling, and the gap under test is the nonlinear learner. Its result is
reported either way, because it bears on whether Stage B is worth its cost.

## Stage B: nonlinear learner

Stream: online regression, x ~ N(0, I_10), y = f_t(x) + N(0, 0.1^2), with f_t a
random one-hidden-layer teacher (tanh, 32 units). Batch size 1 per step, T = 30000.
Learner: MLP 10-64-64-1, ReLU, Adam.

Contexts:
- `changepoint`: teacher redrawn every geometric(1/3000) steps.
- `outlier`: 5% of labels replaced by draws from N(0, 25 var(y)).
- `mixed`: both, the primary context.
- `stationary`: neither, a cost control.
- `coherent_burst`: negative control. Bursts of 100 to 300 samples from a *wrong*
  teacher, after which the true teacher resumes.

Learners:
- `adam`: constant learning rate.
- `monotone`: learning rate boosted on loss z-score.
- `robust`: IMQ down-weighting of the loss by standardised residual.
- `robust_cp`: IMQ plus a BOCD run-length posterior on the loss stream scaling the
  learning rate, deciding on arrival.
- `bone_ll`: WoLF-IMQ+BOCD reimplementation on a Bayesian linear last layer over
  random features; the strongest filtering rival.
- `corrob`: samples above a loss z-score threshold go to quarantine rather than
  being trained on. When the quarantine holds at least K items, clone the weights
  and train the clone on the quarantine. Promote the clone if it beats the current
  weights on the next V *fresh* samples by a preset margin (samples not used to
  fit it, which guards against selection bias). Otherwise clear items past their TTL.
- `corrob_nocheck` (ablation): same quarantine and delay, always commits.
- `corrob_random` (null control): quarantines a random sample at the same rate,
  same corroboration test. This asks whether the surprise selection does any work.
- `oracle`: learning rate boost at true changes, skips true outliers.

Metric: prequential MSE against the noise-free current teacher, measured before
each update. Gap closed is relative to `adam` and `oracle`.

Tuning: 30 random configurations per learner, an equal budget, selected on the
mixed context only, seeds 0-9. Evaluation seeds 100-119, paired. Reports 95%
bootstrap intervals over seeds.

**Runtime constraint.** Colab here is CPU-only with 2 cores. If the full design does
not fit, reduce T and seeds, and record the reduction as a deviation before running.

## Predictions and kill criteria for Stage B

H1 is supported only if all of these hold:

1. **Beats rivals.** On `mixed`, `corrob` beats the best of `adam`, `monotone`,
   `robust`, `robust_cp`, `bone_ll` by at least 0.05 gap closed, CI excluding 0.
2. **Corroboration is load-bearing.** On `mixed`, `corrob` beats `corrob_nocheck`, CI
   excluding 0.
3. **Selection is load-bearing.** On `mixed`, `corrob` beats `corrob_random`, CI
   excluding 0.
4. **No free lunch on stationary.** On `stationary`, `corrob` is within 5% of `adam` MSE.

**Predicted failure.** On `coherent_burst`, `corrob` is expected to promote the wrong
teacher, because a coherent burst is indistinguishable from real change without
extra information (Klymenko 2610.18904). If it does *not* degrade there, suspect a
bug or a stream too easy before believing the result.

**Kill.** If 1 fails on the primary context, H1 is dead as a method claim. If 1 holds
but 2 or 3 fails, the mechanism is not what we described and the claim must be
rewritten around whatever does the work. Either way the result is recorded and
the mission moves to the next bet.

## Deviations

1. **Before any data was generated, Stage A.** The preregistration did not say how a
   commit is applied. The first implementation used the slow tracking rate `alpha`,
   which would make a failure uninterpretable, since a corroborated change would
   still be absorbed slowly. `deferred` and `deferred_nocheck` both gain a tuned
   `commit_alpha` in [alpha, 1]. The search budget stays at 30 configurations. This
   favours the deferred family over `monotone` and `robust` by one extra parameter,
   and should be remembered when reading a narrow win.
2. **Stage A assumptions fixed in code.** Changepoint and outlier are independent
   per step, so `bayes` has a fourth joint branch. `bayes` receives each context's
   true (h, p). The `deferred_nocheck` gate clause requires a CI excluding 0, the
   stricter reading; the laxer result is also reported.
