# H2 result: no conclusion, because the design could not answer the question

Run: GitHub Actions 41726803865, code at 1274e77, 100 tasks, 10 evaluation seeds,
165 minutes. Raw data: `raw_results.csv`, `summary.json`, `run.log`.

## Verdict under the preregistration

| Check | Outcome |
| --- | --- |
| `control_valid` | **Pass.** Maximum energy error 8e-8; maximum density z 4.1 |
| Manipulation check | **Fail.** See below |
| Power rule | **Fail.** `gate_std` of `both_util` is 0.0039, against a floor of 0.05 |
| H2a | **No conclusion** |
| H2b | **No conclusion** |

This is neither support nor a kill. In `summary.json`, `h2a.killed` and `h2b.killed`
read `true`, but those are raw flags computed before the rules are applied. The
`verdict` field and this file are what count.

The manipulation check needed two things. `both_util` did retain better than
`noise_util` on savings (+0.098, CI 0.085 to 0.109), but `protect` was supposed to lose
probe AUC relative to `both_util`, and it gained instead (`both_util` minus `protect`
-0.081, CI -0.096 to -0.065). So protection carried no plasticity cost here, and the
design lacks the tension the question depends on.

The numbers for the H2a and H2b contrasts are in `summary.json`. They are not evidence
about either hypothesis and are not discussed further.

## Why it failed

Everything in this section is post hoc, from the same data. It explains the failure and
guides the redesign. It tests nothing. Three independent defects were found, and any
one of them was enough. All three are design errors of mine.

**1. Zero-shot retention could not register retention at all.** Label permutation keeps
the inputs identical across tasks, and a network is one function of its input with no
context signal. Where the network predicts the current mapping, it can be right about an
old mapping only if the two permutations agree on that class. So

    acc_old <= A + (1 - a)

where `A` is the fraction of classes on which the permutations agree (0.1 in
expectation) and `a` is accuracy on the current mapping. With `a` around 0.9 the ceiling
is about 0.2, whatever the learner does. The data agree: the SD of zero-shot across
seeds is 0.050 (short gap) and 0.036 (long gap), and across conditions 0.0015 and 0.0011.
The metric measured which permutations each seed drew. H2 was preregistered before the
bound-first rule existed, and this bound would have caught it.

**2. Savings measured learning speed, and was censored.** In the five conditions checked
(`sgd`, `protect`, `both_util`, `both_multi`, `adam`), fresh networks failed to reach the
target within the task in 51 to 93 of 100 events, while the trained network always reached
it. So savings was close to `1 - main / 1000`, with the denominator often at its ceiling.
Probe AUC, on permutations never seen before, was 1.4 to 1.9
for every condition but Adam, so most of the speed-up is general feature reuse that a
repeat does not need. Across the nine conditions, savings correlates with online accuracy
at r = 0.94 (long) and 0.98 (short). With n = 9 and no novel-task comparator, this cannot
separate retention from speed.

**3. The gate was nearly constant, and the stream showed no plasticity loss.** With
`gate_std` of 0.004, `both_util` and `both_rand` are almost the same treatment. It also
means `1 - u` acts largely as a learning-rate scale. On the tuning runs, `sgd` and
`protect` fall close together once `protect`'s rate is halved, up to about 0.008. This
assumes a mean gate near 0.5, which the run did not record, so it is an inference. Above
about 0.013 `protect` is 2 to 3 points better than this rescaling predicts, on 2 seeds
and with sparse learning-rate samples, so whether anything beyond rescaling happens is
open. Separately, AUC above 1 for SGD means noise had no lost plasticity to restore. Only
Adam fell below 1, at 0.72.

## What can be stated

- For this stream and these seeds, every condition with reusable features learned a new
  permutation faster than a fresh network did, and Adam was the exception.
- Noise did not improve online accuracy, savings or probe AUC in this stream, which is
  expected given the above and says nothing about streams that do lose plasticity.
- The published UPGD form with this normalisation gave almost no per-weight spread in
  this implementation at this scale. That was measured here only, and at task ends.

## What it does not show

Nothing about whether shared state or multiple timescales help. Nothing about UPGD on
the streams it was published on. Nothing about biology.

## Lesson for the protocol

H1 and H2 both spent their budget before finding a problem that a small pilot or a line
of algebra would have shown. `docs/CHARTER.md` now requires a Stage 0 with a positive
control before any full run. The redesign is `hypotheses/H2R_shared_state_redesign.md`.
