# H5: Is catastrophic forgetting mostly overwriting, or loss of access?

**Status:** preregistered before any H5 code or data. Later changes go in the Deviations section.

## The question, made exact

In a network every change is a weight change, so "overwritten or inaccessible" means nothing until it
is operational. H5 asks how much *must be supplied* to get a forgotten task back:

| Answer | Definition | Biological analogue |
| --- | --- | --- |
| **Access** | Restored without changing any body weight, by a small key | Ryan 2015: engram intact, retrieval route lost |
| **Repair** | Restored only by changing body weights, but faster than learning a new task | Savings |
| **Overwrite** | No key restores it better than the same key teaches a task never seen | Storage loss |

## Why earlier tests cannot answer this

| Source | What it did (read depth) | Gap |
| --- | --- | --- |
| Trivedi & Melwani 2026, arXiv 2606.06032 (pages read) | Split CIFAR-100, accuracy 0.000, probe 0.468 of 0.616, called "accessibility collapse" | No never-seen-task control, so a probe may reflect generic features, not memory |
| Davari 2022, arXiv 2203.13381 (pages read) | Probe before and after a new task, CIFAR-10 forgetting 5.7 points against an observed 22 | Same gap |
| Ramasesh 2020, arXiv 2007.07400 (pages read) | Layer resets, CKA, a probe with a random-lift control | Control is a different network at init, and does not split access from repair |
| Zhou 2026, arXiv 2610.22381 (pages read) | Matched never-learned bindings relearn more slowly, in one LLM | Has the right control, partial forgetting only, and says it does not locate storage |
| Ryan 2015, Science (summaries only, not the paper) | Light activation of engram cells restored memory after amnesia | The biological distinction. Mouse evidence, so nothing here tests biology |

**Core flaw in a plain probe:** a probe on a network that never learned a task also scores well above
chance. Recovery by itself cannot show that anything was remembered.

## Bound and logic, derived before design

**Old-versus-new on one network.** Draw tasks exchangeably, so which is trained and which is held out is
random. Take one network `w`, one key `k`, one budget of labelled samples. Then:

    advantage_k(w) = r_k(w, A) - r_k(w, C)

where `A` was trained, `C` was never seen, and `r_k` is accuracy after key `k`. Same weights, key and
budget. By exchangeability, the expected difference is zero unless `w` holds something specific to `A`.
So a positive advantage is memory, and generic features and key capacity both cancel.

**Surviving fraction.** With `w_0` the network at the end of `A` and `w_m` after `m` further tasks:

    S_k(m) = advantage_k(w_m) / advantage_k(w_0)

`S = 1` means the key reaches all the original advantage, `S = 0` means none. With no key, `S_none` is
ordinary retention, so forgetting and recovery share one scale.

**Shares of what was lost**, at lag `m`:

    access = (S_readout - S_none) / (1 - S_none)
    repair = (S_finetune - S_readout) / (1 - S_none)
    unrecovered = (1 - S_finetune) / (1 - S_none)

**The inference is asymmetric, and this is stated up front.** A key that works proves storage. A key that
fails shows only that this key could not find it. So `access` is a lower bound and `unrecovered` is an
upper bound on overwriting. Planted controls (below) measure how much that matters here.

## Design

Learner: MLP 784-128-128-outputs, plain SGD, cross-entropy, no replay or regularisation. Numpy, CPU.

**Regimes, each with its own verdict**, because the answer may differ:

| Regime | Tasks | Outputs |
| --- | --- | --- |
| `perm_shared` | Random pixel permutations of MNIST | 10 shared |
| `perm_class` | Same | 10 new per task, softmax over all seen |
| `split_class` | Random 3-class subsets of MNIST plus Fashion-MNIST, 20 classes | New per task, softmax over all seen |

`split_class` is the setting the accessibility claims come from, with shared features, so it is the
strongest test of whether their probes reflect memory. It allows lags up to 3, with 6 tasks per seed.

Each seed trains `A`, then `m` tasks. Two fresh tasks `C1` and `C2` are never trained. `C` means their mean.

**Keys, in order.** Each runs on a copy, from a key pool never used in training, with `n` labelled
samples per class, and is scored on held-out test images:

| Key | Changes | Parameters |
| --- | --- | --- |
| `none` | Nothing | 0 |
| `mask` | Restricts the output to the probed task. `perm_class` and `split_class` only | 0 |
| `gain` | One positive gain per hidden unit. No weights change | 256 |
| `readout` | New last layer, ridge on frozen body. Primary access key | 1290 or fewer |
| `finetune` | All weights, warm start | All |

`readout` is primary because it has a closed form with no search, and it matches the published claims.

## Controls

1. **Sham null, on every network.** `S` from `C1` against `C2`, neither trained. It must be near 0.
2. **Planted access.** `w_0` with a random fraction of last-hidden units shuffled before the readout, set
   by search to match the real `S_none`. Nothing is lost, so `S_readout` must be near 1.
3. **Planted erasure, total.** All weights redrawn from init. Nothing survives, so `S` must be near 0 at
   every key.
4. **Planted erasure, matched.** A random fraction of all weights redrawn, set by search to match the real
   `S_none`. Its `S_readout` must sit clearly below planted access.
5. **Plasticity.** `r_k(w_m, C)` is tracked, because a network that learns worse would inflate the
   advantage. A fixed-baseline `S`, using `r_k(w_0, C)` in place of `r_k(w_m, C)`, brackets this.

## Stage 1: calibration, blind to the outcome

Seeds 500 to 504. **Calibration never measures `r_k(w_m, A)` for any key but `none`.** It computes what the
gates need and nothing else, so constants cannot be picked to favour an answer. This follows the first lesson
of `docs/MAPPINGS.md`.

Rules, applied per regime by the code:

- **Config.** Grid: rate 0.01, 0.03, 0.1, 0.3, and 500 or 2000 steps per task, batch 64. Keep configs where
  `w_0` and the final task both reach accuracy 0.85 under the task mask. Of those, take the smallest lag
  with `S_none` at most 0.5, then the highest `w_0` accuracy. That lag is the primary lag `m*`. No config
  means the regime is recorded as unanswerable.
- **Budget.** The smallest `n` in 5, 10, 20, 50 where `advantage_k(w_0)` is at least 0.15 for `readout` and
  `finetune`, so the denominator has real range. None means unanswerable.

Gates, derived from each metric's range, at the chosen config:

| Gate | Requirement | Why it is reachable |
| --- | --- | --- |
| Sham null | Absolute mean `S` at most 0.15, at `readout` and `finetune` | Expected 0 by construction |
| Planted access | `S_readout` at least 0.8, `S_none` at most 0.6 | Ridge is invariant to shuffling, so near 1 |
| Total erasure | Absolute mean `S` at most 0.15, at `readout` and `finetune` | Expected 0 by construction |
| Matched erasure | `S_readout` at least 0.2 below planted access | Range is 0 to 1 |

A failed gate excludes that regime. Calibration is blind, so one rerun on seeds 505 to 509 with changed
constants is allowed, logged here first. The gates themselves do not change.

## Stage 2: confirmation

Seeds 600 to 619, once, at constants fixed by Stage 1, committed before it runs. All intervals are 95%
percentile bootstrap over seeds, 10000 resamples, on ratios of means.

**Verdict for each regime, at `m*`:**

| Label | Rule |
| --- | --- |
| **Mostly access** | `access` CI lower bound above 0.5 |
| **Mostly repair** | `access` CI upper below 0.5, and `access + repair` CI lower above 0.5 |
| **Mostly unrecovered** | `access + repair` CI upper below 0.5 |
| **No majority** | Anything else, reported with its intervals and no leaning |

A label is **not robust** if the fixed-baseline `S` gives a different label, and this must be stated. It is
also void if sham null or planted controls fail their gates on the confirmation seeds. "Mostly unrecovered"
may not be reported as "erased": it means no key here found it.

**Reported, not deciding:** the same shares at every lag, `gain` and `mask`, all budgets, and the naive
retention `r_readout(w_m, A) / r_readout(w_0, A)` that published work reports, set beside `S_readout`.

## What the result can and cannot show

**Falsifiers, stated as bets before data:**

1. I expect `perm_class` and `split_class` to read mostly access, driven by output competition.
2. I expect `perm_shared` to have a real repair or unrecovered share, since its readout is not re-biased.
3. I expect naive retention to exceed `S_readout` in `split_class`, meaning published probes overstate
   stored memory. This is the bet most likely to add something new. If `S_readout` matches naive retention, the
   published reading survives a stronger control and my criticism of it was wrong.

**Limits.** This is a small MLP with plain SGD. Scenario, architecture and optimiser may all change the
answer. Nothing here tests brains. A positive result is about stored task-specific information being
reachable, not about how it is encoded.

**No outcome is designed in.** This test can come out any way, and all outcomes are informative. What the
design guarantees is that the answer will be interpretable, which H2, H2R and H4 were not.

## Deviations

1. **Constants the spec left open, fixed in code before any data** (`experiments/h5_access.py`). Ridge
   penalty 1.0 on standardised features. `finetune`: 100 steps at the task's training rate, batch 64 or the
   whole key set if smaller. `gain`: 100 full-batch Adam steps at 0.05, gains kept non-negative. 200 test
   images per class. Keys beyond `none` and `mask` restrict output to the probed task, so they include the
   mask. Key samples are the same across networks within a seed and task. Sham and planted gates run again on
   the confirmation seeds, and a failure voids the label.
2. **Comparisons added at the owner's request, before any outcome was seen. None decides a label.** The first
   calibration run crashed in the sham gate, which tried to read a trained-task score that the blind stage
   never computes, so it produced no gate or outcome, and calibration reruns on the same seeds 500 to 504.
   Added: `restore_head`, `restore_layer1`, `restore_top2` and `restore_body` put saved `w_0` arrays back with no
   samples, to locate where overwriting happened. They carry many bits, so they are not access keys. `readout1`
   is the same ridge on the first hidden layer, testing the claim that early layers keep more. `finetune10` is the
   first 10 steps of `finetune`, showing how cheap repair is. Each key's recovered share and interval is reported,
   as is the interval of naive retention minus `S_readout`, which is bet 3.
3. **`split_class` calibration rerun, blind, logged before it runs.** On seeds 500 to 504, `perm_shared` and
   `perm_class` passed every gate, with `m*` 5 and 1 and budget 5. `split_class` was excluded: no budget had
   an advantage on `w_0` of at least 0.15, so a new 3-class task is too easy for a key to separate old from new.
   As allowed, only `split_class` reruns on seeds 505 to 509, with budgets 1, 2, 5, 10, 20, 50 and all gates and
   rules unchanged. If it still has no answerable budget, it stays excluded, and that itself is reported: on
   these tasks a probe cannot tell a trained task from a new one. `perm_shared` and `perm_class` keep the
   constants from seeds 500 to 504 and confirm on seeds 600 to 619 first. Only their code path for regimes and
   output files changed, so their constants, keys and analysis are as they were. Raw results now go in one
   file per regime, and `summary.json` is merged.
