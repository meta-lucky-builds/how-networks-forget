# H4: Does schema consistency add anything beyond gradient interference?

**Status:** preregistered before any H4 code. H2 was running and no H2 results had
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

None yet.
