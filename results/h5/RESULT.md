# H5: is forgetting mostly loss of access or overwriting?

**Status: two of three settings confirmed. `split_class` is still rerunning its calibration.**

Run 41791382866, seeds 600 to 619, constants fixed on separate seeds. All sham and planted gates passed on
the confirmation seeds. Data: `summary.json`, `raw_perm_shared.csv`, `raw_perm_class.csv`. Intervals are 95%
bootstrap over seeds.

## Preregistered verdicts

| Setting | Test point | Access | Repair | Unrecovered | Verdict | Robust? |
| --- | --- | --- | --- | --- | --- | --- |
| `perm_class` | 1 later task | 0.96 [0.92, 0.99] | 0.00 [-0.04, 0.05] | 0.04 [0.01, 0.07] | **Mostly access** | Yes, fixed baseline 0.87 / 0.06 / 0.07 |
| `perm_shared` | 5 later tasks | 0.46 [0.41, 0.51] | 0.01 [-0.07, 0.08] | 0.53 [0.46, 0.61] | **No majority** | **No**, fixed baseline 0.15 / 0.11 / 0.75 reads mostly unrecovered |

"Unrecovered" is the most that overwriting can be, not a measured amount of it.

## Everything below is descriptive and decided no label

### The answer changes with how much is learned afterwards

Shares at budget 5, access / repair / unrecovered. `S_none` is the fraction of the advantage still visible with
no key. The test points above were set by a blind rule, and these other lags were not preselected.

| Later tasks | `perm_class` shares | `S_none` | `perm_shared` shares | `S_none` |
| --- | --- | --- | --- | --- |
| 1 | 0.96 / 0.00 / 0.04 | 0.38 | 0.68 / -0.31 / 0.63 | 0.84 |
| 2 | 0.89 / 0.08 / 0.03 | 0.08 | 0.67 / -0.17 / 0.50 | 0.61 |
| 5 | 0.74 / 0.14 / 0.12 | 0.00 | 0.46 / 0.01 / 0.53 | 0.26 |
| 10 | 0.47 / 0.24 / 0.29 | 0.00 | 0.20 / 0.08 / 0.71 | 0.07 |
| 20 | 0.24 / 0.16 / 0.59 | 0.00 | 0.08 / 0.07 / 0.85 | 0.01 |

### Where the damage is, from restoring saved weights

These use no samples and no new-task control, because a new task stays at chance here. Share of the loss recovered:

| Restored | `perm_class`, 20 tasks | `perm_shared`, 20 tasks | `perm_shared`, 5 tasks |
| --- | --- | --- | --- |
| Output mask only | 0.20 | n/a | n/a |
| Head | 0.20 | 0.04 | 0.23 |
| First layer | 0.81 | 0.15 | 0.47 |
| Second layer and head | 0.42 | 0.32 | 0.68 |
| All hidden layers | 1.00 | 0.56 | 0.80 |

So in both settings late damage is in the hidden layers and not in the output weights. In `perm_class` the
restore keys include the mask, so 0.20 for the head means the head adds nothing beyond it.

### Against the planted controls

This reading was worked out after seeing the results. On the `S_readout` scale, planted access scored 1.00 and
matched random erasure scored about 0.50 to 0.54.

| Setting | Real `S_readout` | Way from erasure to access |
| --- | --- | --- |
| `perm_class` | 0.97 | About 0.95 |
| `perm_shared` | 0.60 | About 0.21 |

Random erasure is not the same as overwriting by gradient steps, so this is a comparison and not a measurement.

### The first layer keeps far more than the last

A readout on the first hidden layer recovered 1.09 of the loss in `perm_shared` at 5 tasks, against 0.46 for the
last hidden layer, and 0.53 against 0.08 at 20 tasks. This fits published claims about early layers. It is
partly inflated: the new-task score on that layer also falls, from 0.42 to 0.28, and at 20 tasks the trained
task (0.43) barely exceeds what a new task scored on the untrained-by-others `w_0` (0.42). The advantage on the
same network is still 0.15, and sham null says this is memory.

### Bet 3 so far

Naive probe retention is close to `S_readout` here: 0.65 against 0.60 in `perm_shared`, and 0.96 against 0.97 in
`perm_class`. Permuted tasks share few features, so little inflation was expected, and `split_class` is the real test.

## Problems with this result

1. **Which control is right matters a lot in `perm_shared`.** The network teaches new tasks worse as it ages:
   readout on a new task falls from 0.36 to 0.22, and fine-tuning from 0.61 to 0.48 by 20 tasks. The preregistered
   same-network control credits this as memory, and the fixed baseline counts it as lost, so the truth is probably
   between. That drop is a plasticity loss, a finding in its own right.
2. **My key ladder was not ordered as I assumed.** I treated `finetune` as the strongest key, but it often loses
   to `readout`, which gives negative repair. 100 steps at rate 0.3 on few images seems to overfit: `finetune10`
   scores higher than `finetune`. So "unrecovered" is looser than it should be, and at budget 50 the shares contradict
   each other, with `readout` recovering 1.0 while `unrecovered` says 0.44. **The labels rest on budget 5, and in
   `perm_shared` the label changes with budget:** no majority, mostly repair, mostly access, mostly access at 5, 10, 20, 50.
3. **One optimiser, one rate, one small network.** The blind rule chose rate 0.3 with 2000 steps for both settings,
   the top of the grid. A gentler rate may give a different split.
4. **Readout at budget 5 cannot reach the original accuracy,** 0.83 against 0.96 on `w_0`. `S` is relative to what
   the same key achieves on `w_0`, so a full `S` does not mean full accuracy.
