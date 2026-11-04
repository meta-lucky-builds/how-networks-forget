# Research charter

**Goal.** Take mechanisms humans use biologically to learn, remember and decide,
and implement their computational core in AI models as closely as the evidence
allows, with code and experiments rigorous enough to be wrong in public.

## What counts as a valid mapping

A biological analogy earns nothing by itself. Each mapping must state:

1. **Mechanism.** What the biology does, with primary citations and an honest
   grade of the evidence (established / contested / speculative).
2. **Computational claim.** The function the mechanism is hypothesised to serve,
   stated without biological vocabulary.
3. **Load-bearing constraint.** Which property of the biology does the work. An
   implementation that drops that property is not testing the mapping.
4. **Non-biological rival.** The best engineering method for the same function.
   Where a normative optimum exists (e.g. Bayesian changepoint inference), it is
   the ceiling we report against.
5. **Falsifier.** A preregistered result that would kill the hypothesis.

## Experimental rules

- **Preregister before running.** Hypothesis, metric, seeds, analysis and kill
  criterion are committed to `hypotheses/` before the experiment executes. Any
  later change is logged as a deviation.
- **Bound it first.** Before preregistering, derive whatever analytic bound the
  claim allows, including the best any policy of that class could score. H1 lost
  its Stage A to a bound that takes a few lines, and that could have been known
  before any code was written.
- **Stage 0 before any full run.** Run a small pilot on its own seeds, with a positive
  control that shows each primary metric can move, and preset checks that the phenomenon
  and the tension under test exist in this setup. Failing means no full run. H2 spent
  165 minutes on a stream whose retention metric could not register retention.
- **Ablation and null controls.** Every mechanism has an off switch with matched
  parameter count and a shuffled or random-signal control, so that "it helped"
  cannot just mean "it added capacity or noise".
- **Interactions are measured.** Combined mechanisms use factorial designs, so a
  claimed synergy is an estimated interaction term with an interval.
- **Tune fairly.** Baselines get the same hyperparameter budget as the proposal.
  Tuning seeds and evaluation seeds are disjoint.
- **Report with uncertainty.** Per-seed results, bootstrap intervals, effect
  sizes. Null and negative results are recorded with the same care.
- **Execute in the cloud only.** Experiments and tests run on cloud machines, never a
  local one: GitHub Actions, Colab or a cloud GPU.
- **Say how deeply each source was read.** `docs/literature/` records for each claim whether
  it rests on a full text, an abstract or a snippet, and marks what was not opened.

## Repository layout

| Path | Purpose |
| --- | --- |
| `docs/literature/` | Per-thread evidence notes |
| `docs/MAPPINGS.md` | Biology to computation ledger |
| `hypotheses/` | Preregistrations, one file per hypothesis |
| `src/clu/` | Mechanism implementations |
| `tests/` | Unit and property tests, run in the cloud |
| `experiments/` | Runnable experiment definitions |
| `results/` | Raw per-seed outputs and analyses |
| `docs/STATUS.md` | Where the work stands, what is next, how to resume |
