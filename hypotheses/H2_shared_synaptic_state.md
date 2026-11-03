# H2: One synaptic state for protection and recycling

**Status:** preregistered before any H2 code, and before H1 Stage A results were
seen. Independent of H1. Later changes go in the Deviations section.

## Hypothesis

**H2a (coordination).** When the same per-weight utility decides both which weights
are protected and which are perturbed, retention improves relative to perturbing
equally many weights at random with the same noise energy, at equal plasticity. If
it does not, shared state is not load-bearing and any benefit comes from perturbing
less.

**H2b (timescales).** A utility tracked over several timescales retains long-gap
tasks better than the best single timescale, at equal plasticity.

## Biological basis, stated honestly

| Claim | Grade | Source |
| --- | --- | --- |
| Synapses carry internal state over many timescales | Theoretical, with indirect support | Benna & Fusi 2016; Zenke & Laborieux, via `synaptic.md` |
| Forgetting can act by suppression rather than erasure | Reviews and abstracts, largely Drosophila | Ryan & Frankland 2022, abstract |
| A synapse's consolidation depth predicts whether it is downscaled or forgotten | **No direct measurement found.** Our inference | `synaptic.md` section 4 |
| Sleep downscaling | Contested | Hengen 2016, Durkin 2016, Timofeev 2017, abstracts |
| Human adult neurogenesis as motivation | Contested; not relied on here | `synaptic.md` |

Evidence is indirect, so a positive result supports an engineering claim and would
not show that biology works this way. Reversible suppression is out of scope here
and stays a follow-up.

## Design

Stream: label-permuted MNIST on a fixed subset, raw pixels, online with batch size
1. Labels are re-permuted every 1000 steps, 100 tasks. 10% of tasks repeat an earlier
permutation, half at short gaps (5 to 10 tasks) and half at long gaps (at least 40
tasks). Learner: MLP 784-128-128-10, ReLU.

Update for all gated conditions, following UPGD (Elsayed & Mahmood):
`w <- w - lr * (g * P + n * R)`, where `P` is the protection factor and `n * R` the
gated noise. Utility `u` is a normalised trace of `-g * w`.

| Condition | Protection P | Noise | Purpose |
| --- | --- | --- | --- |
| `sgd` | none | none | Baseline |
| `protect` | 1 - u | none | Protection alone |
| `noise_rand` | none | random mask | Restoration alone |
| `noise_util` | none | gated by 1 - u | Restoration alone, gated |
| `both_rand` | 1 - u | random mask, redrawn each step | **Null control for H2a** |
| `both_util` | 1 - u | gated by the same 1 - u | **H2a** |
| `both_multi` | 1 - max(u_k) | gated by 1 - max(u_k) | **H2b**: three traces, decays spanning 2 orders of magnitude |
| `shrink_perturb` | external reference | | Shrink and perturb |
| `adam` | external reference | | Tuned Adam |

The random mask must match `both_util` at each step in expected density and in total
noise energy, computed from that step's actual gate. If this is not matched exactly,
the control is invalid.

`both_util` tunes its single trace over the same range `both_multi` spans. So a win
for `both_multi` cannot be explained by `both_util` lacking access to a slow decay.

Metrics, measured separately because a method that never learns also never forgets:
- **Plasticity:** from a weight snapshot after each 10th task, train a copy on a fresh
  probe permutation for 300 steps, and report the online accuracy AUC normalised by
  the same probe from fresh initialisation. Probe updates are discarded.
- **Retention, zero-shot:** accuracy on a repeating task's first 100 samples at its
  return, before any update on them is applied.
- **Retention, savings:** steps to regain 90% of that task's earlier best accuracy,
  relative to the same from fresh initialisation.
- Long-gap and short-gap repeats are reported separately.

Tuning: 10 random configurations per condition, 2 tuning seeds, 40-task streams,
selected on mean online accuracy, the same budget for all. Evaluation: 10 fresh seeds,
paired. Reports 95% bootstrap intervals over seeds.

**Runtime constraint.** Colab here has 2 CPU cores. If this does not fit, reduce tasks
and seeds, and log the reduction as a deviation before running.

## Predictions and kill criteria

**Manipulation check first.** `both_util` must retain measurably better than
`noise_util`, and `protect` must lose probe AUC relative to `both_util`, or the design
lacks the tension it needs and no conclusion follows either way. Then fix the design
rather than reading the result.

**H2a supported** only if, on `both_util` minus `both_rand`, zero-shot retention or
savings improves with the CI excluding 0, and probe AUC does not fall by more than one
pooled SE.

**H2a killed** if `both_rand` matches `both_util` within one SE on both retention and
plasticity. The conclusion would be that the shared signal is not what matters, and
the coordination story gains nothing from this evidence.

**H2b supported** only if `both_multi` beats `both_util` on long-gap savings with the
CI excluding 0, with no loss of probe AUC beyond one pooled SE, and shows no comparable
advantage at short gaps. **H2b killed** if the best-tuned single trace matches it, which
would make any earlier apparent benefit a hyperparameter artefact.

Either way the result is recorded and the mission moves to the next bet.

## Deviations

All recorded before any H2 data was generated.

1. **Power rule for an H2a kill.** UPGD's sigmoid normalisation keeps most gates near
   0.5. If the gate barely varies, `both_util` and `both_rand` are almost the same
   treatment and matching them shows nothing. So if `gate_std` for `both_util` is below
   0.05, an H2a kill is reported as uninformative rather than as a kill.
2. **Assumptions fixed in code.** `both_rand` is yoked per step to the same-seed,
   same-config `both_util` run, and the primary contrast uses `both_util`'s config. Its
   own best config is a sensitivity check. Tuning streams of 40 tasks cannot hold a
   gap of 40, so their repeats are all short-gap. Image pools for tuning and evaluation
   are disjoint. Noise comes from a reused pool for speed: any one weight still sees
   independent values across steps, but values are correlated across weights within a
   refresh window.
