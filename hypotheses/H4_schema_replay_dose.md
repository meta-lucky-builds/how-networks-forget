# H4: Does schema consistency add anything beyond gradient interference?

**Status:** stopped at Stage 0, see `results/h4/RESULT.md`. Preregistered before any H4 code. H2 was running and no H2 results had
been seen. Later changes go in the Deviations section.

## Biological claim under test

Tse et al. 2007: rats with a pre-existing schema assimilated new paired associates
rapidly, in a way that became hippocampus-independent far sooner than usual. The
AI-relevant reading is that new information consistent with existing knowledge needs
less replay. McClelland, McNaughton & Lampinen 2020 proposed similarity-weighted
interleaving, in a linear toy whose authors report it failing elsewhere.

| Claim | Grade | Source |
| --- | --- | --- |
| Schema-dependent rapid consolidation in rodents | Established in the original study; see notes for read depth | Tse 2007, `consolidation.md` |
| Similarity-weighted interleaving suffices | Toy only, failed on some other datasets per its authors | McClelland 2020, `consolidation.md` |
| Replay dose is set by relation to existing knowledge | Not established; our reading of the AI gap | `consolidation.md` |

This tests a machine-learning analogue. It cannot show what brains do.

## Bound, derived before design (charter rule)

For a small step on a new item, to first order:

    Delta L_old = -eta * <g_old, g_new> + O(eta^2)

So the damage to old knowledge from one step is fixed, to first order, by gradient
overlap. This gives a concrete rival to the biological story: **schema consistency
may be nothing more than low gradient interference.** It also bounds what a schema
variable can add. Any effect beyond this overlap must come from higher-order or
cumulative terms, so it should grow as training on the new item lengthens.

## Hypotheses

**H4a (residual).** After controlling for measured gradient overlap, schema-consistent
new classes still need less replay. A positive result would mean schema is doing
something first-order interference does not capture.

**H4b (rival, preregistered as the likelier outcome).** Overlap explains the effect,
and the schema coefficient is not distinguishable from zero. Note that absence of a
residual is only meaningful if the interval is tight, so see the equivalence bound.

## Design

Learner: MLP 784-256-256-N on Fashion-MNIST and EMNIST-letters-style class sets, with
natural superclasses. CPU only. Schema phase trains on all classes but the withheld
ones, until a fixed accuracy criterion, so every condition starts equally trained.

For Fashion-MNIST, superclasses: upper body (t-shirt, pullover, coat, shirt, dress),
footwear (sandal, sneaker, ankle boot), and singletons (trouser, bag).

Conditions for the new class:
- **consistent:** withheld class's superclass remains in the schema (coat, with other
  upper-body classes present; ankle boot, with other footwear present).
- **inconsistent:** the withheld class has no relatives in the schema (all members of
  its superclass are withheld too, and one is then introduced).
- **foreign:** a class from a different dataset (an MNIST digit), matched on image count.

Each condition uses at least 20 schema-network seeds, each with several withheld
classes where the design allows, giving enough spread in overlap to fit the model.

Measured at introduction, on the schema network, before any update:
- `overlap`: mean over batches of the normalised negative overlap between new-class
  and old-class gradients, on held-out data.
- `distance`: cosine distance of the new-class mean penultimate feature to the nearest
  old-class mean. A second non-schema covariate.

Procedure: focused fine-tuning on the new class, at a fixed number of steps, with a
replay ratio from a preregistered ladder (0, 1/8, 1/4, 1/2, 1 old samples per new sample),
using a large buffer so buffer contents are not the limiting factor. Three training
lengths (short, medium, long) test the higher-order prediction.

Primary outcomes:
- `drop`: fall in old-class accuracy on held-out data after fine-tuning.
- `dose`: smallest ladder rung bringing `drop` under 2 points. Censored if none does.
- `new_acc`: accuracy on the new class, reported so that a method cannot look good by
  not learning the new class.

**First-order validity check.** Predicted drop from overlap must correlate with
measured `drop` at the short length, Pearson r at least 0.5. If not, the first-order
account fails here and H4a and H4b are both unanswerable as framed. Stop and report.

## Analysis

Primary model, at the medium length and zero replay:

    drop ~ schema_condition + overlap + distance + (1 | seed)

Reported with cluster bootstrap over schema-network seeds (10000 resamples).
The coefficient of interest is consistent minus inconsistent, in accuracy points.

**Equivalence bound, fixed in advance:** 1.0 accuracy point.

## Predictions and kill criteria

Check first: the raw effect must exist. Without covariates, consistent must have lower
`drop` than inconsistent with CI excluding 0. If not, the Tse analogue is absent here,
and the result is reported as "no schema effect in this setting", with neither H4a nor
H4b supported.

Given a raw effect:
- **H4a supported:** adjusted coefficient has its CI entirely beyond 1.0 point in the
  predicted direction, with `new_acc` within 2 points across conditions.
- **H4b supported:** adjusted CI lies entirely within plus or minus 1.0 point. Schema
  then adds nothing measurable beyond the two covariates in this setting.
- **Inconclusive:** anything else. This is reported as inconclusive, with no leaning
  in the write-up.

**Higher-order prediction.** If H4a holds, the residual should grow from short to long
training, since the first-order term captures the short-horizon effect. If it does not
grow, the residual likely reflects covariate measurement error rather than a real extra
mechanism, and that caveat must be stated.

**Confound stated up front.** Semantically related classes tend to be visually similar,
so `distance` and condition are correlated. If their overlap is too strong to separate,
variance inflation above 5, H4a and H4b are not answerable and the result says so.
A CIFAR-100 version is out of scope on this CPU runtime.

## Deviations

1. **Stage 0 added, before any H4 code or data**, under the charter rule that followed H2.
   Run 3 schema-network seeds, disjoint from the evaluation seeds and not used in the
   verdict. The full run starts only if all of these hold:
   - Positive control: fine-tuning a new class with zero replay gives `drop` of at least
     5 points, and the top ladder rung lowers it by at least half. Otherwise replay dose
     cannot vary here.
   - `new_acc` reaches at least 70% at the medium length, so every condition learns.
   - First-order validity check, at the short length, r at least 0.5 across events.
   - Variance inflation of condition against `distance` and `overlap` at most 5.
   - Under half of `dose` values are censored.

   If a check fails, H4 stops with that reason recorded. The raw effect is deliberately not
   a Stage 0 check, because that would select on the outcome. It stays in the full analysis
   as preregistered. Fashion-MNIST alone is used for H4: the EMNIST-style set is dropped
   for runtime, and the design already allows this. The full run executes on GitHub Actions.

2. **Schema size and training depth, fixed before data.** Withholding classes makes schema size
   differ by network: 8 (consistent, singleton), 7 (footwear withheld), 5 (upper body withheld).
   A fixed accuracy criterion also does not equalise steps, since an easy schema reaches it far
   sooner. Either can produce a false raw effect in the predicted direction. The primary model
   stays as preregistered. Three verdict-free checks are added: dropping the upper-withheld network,
   dropping singleton new classes, and adding schema size and log steps as covariates. **A supported
   H4a or H4b must be reported as not robust if any of the three moves the consistent coefficient's
   interval to a different label.** Stage 0 reports steps per network.
3. **Constants and readings, fixed before data** (`experiments/h4_schema.py`). 40 evaluation seeds
   (400 to 439), so a CI within 1 point is reachable. Fine-tuning: plain SGD, rate 0.002, 32 new
   samples per step, lengths 10, 100 and 500 steps. Schema: rate 0.05, batch 64, criterion 80% checked
   every 25 steps, cap 20000. All schema images of an old class form the replay buffer. N is 11, so
   parameter count is matched. The validity check uses the named `overlap` variable; r with the raw
   inner product is reported and decides nothing. New-class accuracy at Stage 0 is checked at zero
   replay, the primary model's setting, with other rungs reported. Added Stage 0 check: every schema
   network reaches the criterion. Growth must hold in points and in SD units. Intervals are 95%
   percentile cluster bootstrap. The Stage 0 checkpoint is not committed, since it would show the
   raw effect. These lengths rest on an estimate, so Stage 0 can fail on them, and H4 then stops.
