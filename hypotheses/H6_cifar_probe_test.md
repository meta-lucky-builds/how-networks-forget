# H6: Do probes overstate stored memory on a real shared-feature task?

**Status:** complete, see `results/h6/RESULT.md`. Preregistered before any H6 code or data. It follows H5 and uses what
H5 showed, so H5's seeds and results are not reused. Later changes go in the Deviations section.

## Question

H5 could not answer on 3-way MNIST and Fashion tasks. Published accessibility claims come from Split CIFAR with deep
networks (Trivedi & Melwani 2026, Davari 2022, Ramasesh 2020; read depth in `H5_overwrite_or_access.md`). They report
probe retention, with no never-seen-task control on the same network.

**H6a (primary).** Naive retention `r_readout(w_m, A) / r_readout(w_0, A)` exceeds the controlled `S_readout`.
**H6b.** The access, recoverable and unrecovered split from H5, on this task.

## Design

Split CIFAR-100, random per seed: 10 tasks of 10 classes. Task 0 is `A`, tasks 1 to 5 are trained later, tasks 6 to 9
are never trained and are `C`. Using 4 new tasks reduces the noise that sank `split_class`. Sham compares the mean of
two new tasks with the mean of the other two.

Model: ResNet-18 at width 32 with batch norm, 100 outputs. SGD, momentum 0.9, batch 128, flips and small shifts.

| Regime | Training softmax | `none` scored over |
| --- | --- | --- |
| `cifar_class` | All seen outputs | All seen outputs plus the probed task |
| `cifar_task` | Current task's outputs only | The probed task, since the task is known |

Keys, each on a copy, from 50 images per class never used in training, scored on the 100 test images per class. Batch
norm stays in eval mode except in `bn_recal`.

| Key | Changes |
| --- | --- |
| `none`, `mask` | Nothing. `mask` restricts to the probed task, in `cifar_class` only |
| `bn_recal` | Batch norm statistics only, from unlabelled key images. A new zero-label access key |
| `readout` | New head, ridge on the penultimate layer, penalty 1. Primary access key |
| `readout_early` | Same, on pooled third-stage features |
| `finetune_low` | All weights, warm start, 30 steps at a tenth of the training rate |
| `restore_head`, `restore_body` | Saved `w_0` arrays. These locate damage and are not access |

`S_k`, the sham and the planted controls are as in H5, including the fixed baseline.

**Shares, changed from H5 because `finetune` was not the strongest key there:**

    access = (S_readout - S_none) / (1 - S_none)
    recoverable = (S_best - S_none) / (1 - S_none)
    unrecovered = 1 - recoverable

`S_best` is the highest of `readout`, `readout_early` and `finetune_low`, taken in each resample. This slightly favours
recovery and is stated. `unrecovered` remains an upper limit on overwriting.

## Stage 1: blind calibration

Seeds 700 to 705. A rerun with changed constants is allowed once, on seeds 706 to 711, logged here first. It never scores
`A` on a later network with any key but `none`.

- **Config.** Rate 0.01 or 0.05, 10 or 30 epochs per task, on the first 3 calibration seeds. Keep configs where `w_0` and the
  last task both reach accuracy 0.4 under the task's outputs. Take the smallest lag in 1, 2, 3, 5 with `S_none` at most 0.5,
  then the highest `w_0` accuracy. That lag is `m*`. In `cifar_task`, forgetting may be slow, and a regime with no such lag is
  recorded as unanswerable.
- **Budget.** The smallest `n` in 2, 5, 10, 20, 50 where the `readout` and `finetune_low` advantages on `w_0` are at least 0.15.
- **Gates**, as in H5: sham and total erasure at most 0.15 in absolute value, at `readout` and `finetune_low`. Planted
  access `S_readout` at least 0.8 with `S_none` at most 0.6. Matched erasure at least 0.2 below planted access.

Passing regimes go straight to Stage 2, so that this step involves no choice.

## Stage 2: confirmation

Seeds 800 to 819, at `n` and 20. 95% percentile bootstrap over seeds, 10000 resamples, on ratios of means.

**H6a verdict, at `m*` and `n`, in each regime:**

| Label | Rule |
| --- | --- |
| **Probes overstate** | `naive - S_readout` CI lower bound above 0.05 |
| **No overstatement** | CI within plus or minus 0.05 |
| **Probes understate** | CI upper bound below -0.05 |
| **Inconclusive** | Anything else |

0.05 matches H5's `perm_shared` gap of 0.04, which I would not call material. H6a is judged on the same-network control
alone, because that is the point of the test. The fixed baseline is reported beside it.

**H6b verdict:** as in H5, with `recoverable` in place of `access + repair`. Mostly access if the `access` CI lower is above
0.5. Mostly repair if `access` CI upper is below 0.5 and `recoverable` CI lower is above 0.5. Mostly unrecovered if
`recoverable` CI upper is below 0.5. Otherwise no majority. **A label stands only if both baselines give it.** Otherwise it
is reported as baseline-dependent, with both. Failed gates on the confirmation seeds void the labels.

## Bets, before data

1. Naive retention will exceed `S_readout` by more than 0.05 in both regimes, because CIFAR features transfer to new classes.
2. `cifar_class` reads mostly access or baseline-dependent, and `cifar_task` has slow forgetting.
3. `bn_recal` recovers a clear share. This is a guess, since batch norm drift would be an access failure that needs no labels.

If bet 1 fails, the published probes survive a stronger control, and my criticism of them was wrong.

## Limits

One small network, one optimiser, and 20 seeds. GPU runs are not bit-for-bit reproducible. It tests no brain mechanism. Class
difficulty varies, so exchangeability holds across seeds and not within one seed.

**Execution.** One GPU, chosen in Deviation 2. Code in `experiments/h6_cifar.py`, whose `selftest` mode replaces a separate
test file because it needs the GPU.

## Deviations
1. **Compute trims and data source, before any H6 data.** Timing on the T4: one epoch 1.5 s, one seed about 5 to 9 minutes, so
   the plan needed 6 to 8 hours, which risks a Colab session. Epochs become 10 or 20 instead of 10 or 30. The config curve uses the
   first 2 calibration seeds instead of 3. Confirmation scores budget `n` only, not also 20, so budget sensitivity is not reported.
   The official CIFAR-100 server gave about 53 kB/s, so the file comes from the Google Cloud mirror Keras uses, with the same md5,
   eb9058c3a382ffc7106e4002c42a8d85, also checked by torchvision. No rule, seed, gate or label changes.
2. **Execution moves to the owner's RTX 5090 in a Jupyter-style cloud, before any H6 outcome was seen.** Colab disconnected, and
   nothing from its partial run was read or kept. The owner runs one uploaded file and returns the results. The file now loads
   CIFAR-100 itself, with the same md5, in place of torchvision, and adds checks on the expected image counts. It runs the self-test
   first, and the results are zipped. Seeds, constants, rules, gates and labels do not change. Which GPU ran it is printed in the log.
