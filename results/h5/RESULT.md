# H5: is forgetting mostly loss of access or overwriting?

**Status: complete. Two of three settings answered, `split_class` could not be answered and is excluded.**

**Answer.** Neither explanation is "the" cause. Early forgetting is mostly loss of access, above all when tasks
compete for outputs. As learning continues, the share no key can recover grows until it is the largest part.
With shared outputs, access is weaker and the split is contested at moderate forgetting. "Unrecovered" is an upper
limit on overwriting, so overwriting itself is not measured. This holds for a small MLP on permuted MNIST.

Run 41791382866, seeds 600 to 619, constants fixed on separate seeds. All sham and planted gates passed on
the confirmation seeds. Data: `summary.json`, `raw_perm_shared.csv`, `raw_perm_class.csv`. Intervals are 95%
bootstrap over seeds.

## Preregistered verdicts

| Setting | Test point | Access | Repair | Unrecovered | Verdict | Robust? |
| --- | --- | --- | --- | --- | --- | --- |
| `perm_class` | 1 later task | 0.96 [0.92, 0.99] | 0.00 [-0.04, 0.05] | 0.04 [0.01, 0.07] | **Mostly access** | Yes, fixed baseline 0.87 / 0.06 / 0.07 |
| `perm_shared` | 5 later tasks | 0.46 [0.41, 0.51] | 0.01 [-0.07, 0.08] | 0.53 [0.46, 0.61] | **No majority** | **No**, fixed baseline 0.15 / 0.11 / 0.75 reads mostly unrecovered |

"Unrecovered" is the most that overwriting can be, not a measured amount of it.

## `split_class`: excluded, no verdict

It used its one blind rerun, seeds 505 to 509 (Deviation 3). On seeds 500 to 504 no budget from 5 to 50 had an
advantage on `w_0` of at least 0.15. On the rerun, budget 1 had enough range (readout 0.36, finetune 0.32), but two
null checks failed: total erasure on `readout` read 0.34, and sham on `finetune` read -0.22, each against a limit of
0.15. At one image per class with five seeds, the measurement is too noisy. As preregistered it stays excluded,
and **bet 3 is not settled by this experiment.**

The first failure is still worth noting. With 5 or more images per class, a readout on a never-seen 3-class task
scored within 0.15 of the trained task, even on the network that had just learned it. So on these easy tasks a probe
says little about memory. This does not rebut the published claims, since these 3-way MNIST and Fashion tasks are
much easier than theirs, and it was measured before forgetting, not after.

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
task (0.43) barely exceeds what a new task scored on `w_0` (0.42). The advantage on the
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
3. **One optimiser and one small network.** The blind rule chose rate 0.3 with 2000 steps for both settings, the
   top of the grid. The follow-up below covers rate 0.1, but not width, optimiser or dataset.
4. **Readout at budget 5 cannot reach the original accuracy,** 0.83 against 0.96 on `w_0`. `S` is relative to what
   the same key achieves on `w_0`, so a full `S` does not mean full accuracy.

## Follow-up: replication, gentler rate, fairer strongest key (Deviation 4)

Descriptive, designed after the verdicts, and it changes no label. Run 41795802842, fresh seeds 620 to 639. All gates
passed for all four combinations. Full table: `sensitivity.md`.

**The verdict numbers replicated at rate 0.3:**

| Setting, test point | Measure | Confirmation | Replication |
| --- | --- | --- | --- |
| `perm_shared`, 5 tasks | Access, same-network | 0.46 | 0.47 |
| `perm_shared`, 5 tasks | Access, fixed baseline | 0.15 | 0.14 |
| `perm_class`, 1 task | Access, same-network | 0.96 | 1.00 |
| `perm_class`, 1 task | Access, fixed baseline | 0.87 | 0.92 |

**Budget 5. Access is same-network, then fixed baseline. Unrecovered uses the best key, same-network, then fixed.**

| Setting | Rate | Later tasks | `S_none` | Access | Unrecovered, best key |
| --- | --- | --- | --- | --- | --- |
| `perm_class` | 0.3 | 1 | 0.34 | 1.00, 0.92 | -0.06, 0.08 |
| `perm_class` | 0.3 | 5 | 0.00 | 0.79, 0.63 | 0.02, 0.22 |
| `perm_class` | 0.3 | 20 | 0.00 | 0.20, -0.06 | 0.46, 0.83 |
| `perm_class` | 0.1 | 1 | 0.21 | 0.95, 0.95 | 0.04, 0.05 |
| `perm_class` | 0.1 | 5 | 0.00 | 0.85, 0.83 | 0.08, 0.14 |
| `perm_class` | 0.1 | 20 | 0.00 | 0.45, 0.35 | 0.30, 0.49 |
| `perm_shared` | 0.3 | 1 | 0.81 | 1.08, 0.37 | -0.35, 0.63 |
| `perm_shared` | 0.3 | 5 | 0.25 | 0.47, 0.14 | 0.02, 0.63 |
| `perm_shared` | 0.3 | 20 | 0.03 | 0.13, -0.20 | 0.52, 0.99 |
| `perm_shared` | 0.1 | 1 | 0.94 | 0.56, 0.36 | -0.09, 0.37 |
| `perm_shared` | 0.1 | 10 | 0.23 | 0.49, 0.18 | 0.14, 0.73 |
| `perm_shared` | 0.1 | 20 | 0.07 | 0.23, -0.11 | 0.40, 0.90 |

Readings:

1. **Separate outputs: access is robust.** At rate 0.1 the two baselines nearly agree, so the disagreement seen at 0.3
   comes largely from the high rate. Access stays the largest share through 10 tasks at 0.1 (0.66, 0.62).
2. **Shared outputs: the rate does not change the split at matching forgetting.** At `S_none` near 0.25, access is
   0.47 at rate 0.3 with 5 tasks and 0.49 at rate 0.1 with 10 tasks. So a gentler rate delays forgetting without
   changing its kind.
3. **The limit on overwriting at moderate forgetting is wide.** With the best key, `perm_shared` unrecovered at 5
   tasks is 0.02 same-network and 0.63 fixed. Values below 0 and `S` above 1 cannot be real memory fractions, so
   the same-network reading is inflated at the early lags, by the falling new-task score. Fixed baseline counts all
   of that fall as lost, so it is too harsh. The data do not say where between them the truth lies.
4. **Deep forgetting is the firmest result.** At 20 tasks in `perm_shared`, unrecovered is at least 0.40 on either
   baseline at either rate, even with the best key chosen after the fact.
5. **The first layer was usually the best key,** often ahead of fine-tuning, consistent with the earlier reading.
