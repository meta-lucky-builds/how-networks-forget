# Status

Last updated 2026-11-04. Anyone resuming, human or agent, starts here, then reads [`CHARTER.md`](CHARTER.md).

**Where it stands.** H5 and H6, the two forgetting experiments, are complete. All other lines are paused or closed.

## Bets

| Bet | Claim | Status |
| --- | --- | --- |
| H1 | Commit surprising samples only after corroboration | **Stopped**, gate failed. H1b (hedge predictions, defer weight commits) is parked in its result |
| H2 | One multi-timescale state sets both protection and recycling | **No conclusion**, the design could not answer |
| H2R | H2 with input permutation, rank gates, a repeat-versus-novel metric | **Stopped at Stage 0**, line closed, claim untested |
| H3 | Gates must be self-generated, without task labels or blocks | Not started |
| H4 | Does schema consistency add anything beyond gradient interference? | **Stopped at Stage 0**, claim untested |
| H5 | Forgetting: overwriting or loss of access, on an MLP | **Complete.** `perm_class` mostly access, `perm_shared` no majority, `split_class` excluded. Replicated |
| H6 | Probes overstate memory, on a ResNet | **Complete.** Probes overstate and mostly unrecovered, at budget 2 only |

Kill rules live in each preregistration.

## Next

1. **H6 budget sweep.** Confirmation at budgets 2, 5, 20 and 50 on fresh seeds, preregistered first. It decides whether H6 is
   about low-label probes only.
2. H3 and new bets wait until the forgetting line is settled.

## A synthesis, not a finding

Each literature thread's dropped property turned out to involve inferring what caused new data rather than reacting to how
surprising it is. No experiment here has tested this. An earlier working idea, one global neuromodulatory scalar coupling
all plasticity, is weakened: a shared signal is contested in biology and the AI evidence on granularity is mixed.

## How the work runs

- All experiments and tests run on cloud machines, never on a local one: GitHub Actions, Colab, or a rented GPU.
- Every experiment is preregistered in `hypotheses/` first, and later changes are logged as deviations.
- Long jobs resume from per-seed records and write to a log, so a dropped machine loses little.
- No credentials go into notebooks. Code reaches Colab through notebook cells, checked by checksum.

## Open items

- Citations from 2026 preprints come from one research pass and have not been independently re-opened.
- H1's per-seed files were not kept. The run is deterministic, so `experiments/h1_stage_a.py` regenerates them.
