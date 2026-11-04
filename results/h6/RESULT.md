# H6: do probes overstate stored memory on Split CIFAR-100?

**Status: complete. All gates passed in both settings on the confirmation seeds.**

**Answer.** At two labelled images per class, yes. Plain probe retention looks like about half the old task survives, and it
barely moves as learning continues, while the same-network measures fall. The fixed baseline falls less, and early-layer readouts hold up on it,
so how far probes overstate total stored memory is not settled. The split reads mostly unrecovered in
both settings on both baselines. **Both verdicts hold at that one budget.** Budget 2 is a weak key, and the sweep that would
show how far this carries was cut to save compute. "Unrecovered" is an upper limit on overwriting, not a measurement of it.

Spec: `hypotheses/H6_cifar_probe_test.md`. Run on one RTX 5090 in 72 minutes, calibration seeds 700 to 705 with no rerun,
confirmation seeds 800 to 819. Data: `summary.json`, `calibration.json`, `calibration_jobs.jsonl`, `confirm_jobs.jsonl`,
`run.log`. Post hoc numbers: `audit.txt`, from `experiments/h6_audit.py`. Intervals are 95% bootstrap over seeds.

Chosen blind: `cifar_class` rate 0.01, 20 epochs, test point 1 later task. `cifar_task` rate 0.01, 10 epochs, test point
2 later tasks. Budget 2 images per class in both, the smallest allowed.

## Preregistered verdicts

### H6a: **probes overstate, in both settings**

| Setting | Naive retention | `S_readout` | Gap | Fixed-baseline `S_readout` | Seeds with a positive gap |
| --- | --- | --- | --- | --- | --- |
| `cifar_class` | 0.552 | 0.146 | **0.406** [0.351, 0.469] | 0.196 | 20 of 20 |
| `cifar_task` | 0.594 | 0.038 | **0.556** [0.464, 0.674] | 0.184 | 20 of 20 |

The raw accuracies show why. Never-trained tasks already score about 0.28 on the original network, and later training adds 0.02 to 0.04 at these test points. Much of what looks like memory is ability the probe has on any task.

| Setting | Readout on old task, start to later | Readout on never-trained tasks, start to later |
| --- | --- | --- |
| `cifar_class` | 0.645 to 0.357 | 0.286 to 0.304 |
| `cifar_task` | 0.558 to 0.332 | 0.281 to 0.321 |

### H6b: **mostly unrecovered, in both settings, on both baselines**

Shares are of the advantage lost so far, with no key. `S_none` is what survives with no key at all.

| Setting | `S_none` | Access | Recoverable | Unrecovered | Fixed baseline: access, recoverable |
| --- | --- | --- | --- | --- | --- |
| `cifar_class` | 0.00 | 0.15 [0.07, 0.21] | 0.39 [0.34, 0.44] | 0.61 | 0.20 [0.13, 0.26], 0.39 [0.34, 0.49] |
| `cifar_task` | 0.42 | -0.66 [-0.88, -0.47] | 0.20 [0.13, 0.26] | 0.80 | -0.42 [-0.64, -0.25], 0.20 [0.11, 0.27] |

The fixed-baseline interval in `cifar_class` clears the 0.5 line only narrowly. In `cifar_task`, 0.42 of the original
advantage survives untouched, so 0.80 of the rest is 0.46 of the original.

### Gates on the confirmation seeds

| Check | Limit | `cifar_class` | `cifar_task` |
| --- | --- | --- | --- |
| Sham, `readout` | abs at most 0.15 | -0.015 | -0.067 |
| Sham, `finetune_low` | abs at most 0.15 | -0.034 | -0.006 |
| Total erasure, `readout` | abs at most 0.15 | 0.035 | 0.048 |
| Total erasure, `finetune_low` | abs at most 0.15 | 0.009 | 0.003 |
| Planted access, `S_readout` | at least 0.8 | 1.000 | 1.000 |
| Planted access, `S_none` | at most 0.6 | 0.124 | 0.416 |
| Planted access minus matched erasure | at least 0.2 | 0.960 | 0.460 |

### Bets

1. Naive above `S_readout` by over 0.05 in both settings: **right.**
2. `cifar_class` mostly access or baseline-dependent: **wrong.** `cifar_task` forgets slowly: right, 0.61 of the advantage
   still shows at 1 later task against 0.00 in `cifar_class`.
3. `bn_recal` recovers a clear share: **wrong.** It reads 0.233 at 1 later task in `cifar_class`, but it is scored with the
   output mask, so the fair comparison is `mask` at 0.204, a difference of 0.029 [0.014, 0.045]. In `cifar_task` the
   difference from `none` at the test point is -0.007 [-0.036, 0.020]. With 20 images, these statistics are themselves noisy.

## Everything below is post hoc and decides no label

### Naive retention does not track forgetting

Same network, budget 2. "Best key" is the highest of `readout`, `readout_early` and `finetune_low`, which favours recovery and
so makes this gap conservative. Test points are in bold.

| Setting | Later tasks | Naive | `S_readout` | Best key | Naive minus best key |
| --- | --- | --- | --- | --- | --- |
| `cifar_class` | **1** | 0.552 | 0.146 | 0.390 | 0.162 [0.108, 0.217] |
| `cifar_class` | 2 | 0.529 | 0.069 | 0.182 | 0.347 [0.275, 0.393] |
| `cifar_class` | 3 | 0.538 | 0.069 | 0.125 | 0.414 [0.274, 0.468] |
| `cifar_class` | 5 | 0.491 | -0.012 | 0.052 | 0.439 [0.308, 0.482] |
| `cifar_task` | 1 | 0.673 | 0.286 | 0.684 | -0.011 [-0.066, 0.043] |
| `cifar_task` | **2** | 0.594 | 0.038 | 0.536 | 0.058 [0.023, 0.094] |
| `cifar_task` | 3 | 0.639 | 0.138 | 0.445 | 0.195 [0.137, 0.251] |
| `cifar_task` | 5 | 0.610 | 0.017 | 0.316 | 0.294 [0.234, 0.353] |

The sharpest single reading: in `cifar_class` after 5 later tasks, the readout scores 0.317 on the trained task and 0.321 on
tasks the network never saw, while naive retention still says 0.49. Correcting naive retention for chance (0.1) moves it only
to about 0.40.

### The gap on the fixed baseline

Plain retention minus the fixed-baseline readout is 0.36 [0.31, 0.41] and 0.41 [0.33, 0.51] at the test points. It is positive in all 20 seeds at all
four lags in both settings, with no interval's lower bound below 0.27. Against the strongest key on the fixed baseline the gap does not grow:
0.16 to 0.12 between 1 and 5 later tasks in `cifar_class`, and 0.00 to 0.03 in `cifar_task`. Full table: `audit.txt`.

### Where the damage is

`S` after restoring saved `w_0` weights into a copy. This locates damage and is not recovery, since it uses the old weights.
Both are scored with the task's outputs only, and `mask` is the matching reference.

| Later tasks | `cifar_class`: mask, restore head, restore body | `cifar_task`: none, restore head, restore body |
| --- | --- | --- |
| 1 | 0.20, 0.27, 0.97 | 0.61, 0.61, 1.00 |
| 2 | 0.08, 0.10, 0.97 | 0.42, 0.42, 1.00 |
| 3 | 0.05, 0.05, 0.97 | 0.31, 0.31, 1.00 |
| 5 | 0.01, 0.02, 0.97 | 0.20, 0.20, 1.00 |

So nearly all the damage is below the head, matching H5's late damage in hidden layers. The `cifar_task` head for the old task
is not updated by later softmaxes, so `restore_head` equal to `none` there, to the third decimal, is a check that passes.

### The baselines disagree more for earlier features

`readout_early` at 5 later tasks: same-network 0.05 and fixed 0.37 in `cifar_class`, 0.19 and 0.58 in `cifar_task`. The old task's
accuracy on early features holds near 0.33 while new tasks climb towards it. As in H5, the data do not say where between
them the truth lies.

## Problems with this result

1. **Two images per class is a very weak key, and I removed the check on it.** The blind rule takes the smallest budget with
   enough range, which here was the smallest allowed. Deviation 1 dropped the confirmation at budget 20 to save compute. H5's
   label changed across budgets, so this one may too. "Mostly unrecovered" means unrecovered by these keys at this budget.
2. **`S_readout` understates what is stored in `cifar_task`.** The old head alone beats the fresh readout, `S_none` minus
   `S_readout` 0.384 [0.271, 0.512] at the test point, which is why access is negative. A negative share is not a memory
   fraction, so `access` has no reading in this setting. The label rests on `recoverable`, and the H6a gap of 0.556 overstates
   the overstatement: against the best key it is 0.058 at that point. The preregistered verdict stands, and this limits it.
3. **The best-key comparison mixes keys.** Naive retention uses the readout, and `finetune_low` starts from the old head, so its
   advantage counts head and body together. It is a generous reference, not the same quantity.
4. **The planted-access gate cannot fail for this readout.** Planted access shuffles units in front of the head, which a fresh
   readout ignores, so 1.000 is built in. It checks the arithmetic. It does not show the keys would detect access loss
   lower in the network. Sham and total erasure are the informative nulls.
5. **This does not refute published probe studies.** It shows the gap on one small ResNet at a very low budget. They train on
   the task's own data, so they use far more labels. Rechecked on 2026-11-05: Trivedi & Melwani 2026 report 0.468 of 0.616, and Davari
   2022 report 5.7 points against 22. Neither reports a never-trained-task control on the same network, and Ramasesh 2020's
   control is a network at initialization. Davari was read in part, in the pages about probes and controls.
6. **Not comparable to H5 directly.** `perm_class` read 0.96 access and `cifar_class` 0.15, but network, data, task structure and
   budget (5 against 2) all changed at once, so no one cause can be assigned.
7. One network, one optimiser, 20 seeds, and GPU runs that are not bit-for-bit reproducible. The two runtime warnings in
   `run.log` come from the self-test's tiny data, before the real run starts. It says nothing about brains.

## What would settle it

Rerun confirmation at budgets 2, 5, 20 and 50 on fresh seeds, preregistered first. If the gap and label survive at 20 and 50,
the probe criticism gains real weight. If they vanish, the finding is about low-label probes only.
