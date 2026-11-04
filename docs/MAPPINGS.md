# Biology-to-AI mappings tested so far

What this mission has established, including what it failed to. A mapping counts as supported only
after it passes its preregistered test against a non-biological rival (`docs/CHARTER.md`).

**Current state: no mapping is supported.** One was tested and failed its gate. Three designs
could not answer their question, so those claims remain open. H5 and H6 gave answers about AI models and
test no brain mechanism, so they are results about forgetting and not supported mappings.

| Bet | Biological idea | AI claim | Outcome | What it licenses | Record |
| --- | --- | --- | --- | --- | --- |
| H1 | Corroboration before commitment | Commit surprising samples only after corroboration | **Negative**, gate failed | Withholding on arrival cost about twice what hedging cost under squared error; budget was not the cause | `results/h1_stage_a/RESULT.md` |
| H2 | One multi-timescale synaptic state sets protection and recycling | A shared gate beats unrelated gates | **No conclusion**, design failure | Nothing about the claim. Label permutation capped zero-shot near 0.2 | `results/h2_shared_state/RESULT.md` |
| H2R | As H2, on input-permuted MNIST with rank gates | As H2 | **Stopped at Stage 0**, line closed | Nothing about the claim. Short-gap threshold could not be met, and protect did not lose AUC | `results/h2r/RESULT.md` |
| H4 | Schema-consistent learning needs less replay (Tse 2007) | Schema adds to replay savings beyond gradient overlap | **Stopped at Stage 0** | Nothing about the claim. `dose` was 96% censored, first-order r was 0.445 | `results/h4/RESULT.md` |
| H5 | Engram retained, retrieval lost (Ryan 2015) | Forgetting is mostly loss of access, not overwriting | **Answered in two of three settings, for AI only.** `perm_class` mostly access, `perm_shared` no majority, `split_class` excluded | Early forgetting is mostly access, deep forgetting mostly unrecovered, on a small MLP. Says nothing about brains | `results/h5/RESULT.md` |
| H6 | None: tests a measurement claim from H5 on a deep network | Probe retention overstates stored memory | **Answered at budget 2 only.** Probes overstate, and the split is mostly unrecovered, in both CIFAR-100 settings | Naive probe retention can read about half retained while old and never-trained tasks score alike. Not tested at larger budgets | `results/h6/RESULT.md` |
| H3 | Self-generated gating | Gates must work without task labels or blocks | Not started | | `docs/literature/gating_metacognition.md` |

## Method lessons, each paid for

These are about how to run this work, and they are the most reliable output so far.

1. **Check the question can be answered before running it.** H2 spent about 165 minutes on a design
   with a ceiling that could be derived on paper. Stage 0 then stopped H2R and H4 before about 414
   projected minutes of full runs.
2. **Check thresholds against what is arithmetically possible.** H2R's short-gap check needed an
   oracle above accuracy 1.0, given what sgd scored. Thresholds should be derived from the metric's
   range, and a pilot should report that range before they are fixed.
3. **Match the bound's quantity to the measured quantity.** H4's bound was on loss and its check on
   accuracy.
4. **Tuning can decide the outcome.** Selecting rates on online accuracy gave protect a third of sgd's
   rate, confounding the tension check. Fixing a rate, or sweeping it as a factor, separates the two.
5. **Separate calibration from confirmation.** Three Stage 0 failures came from constants that were
   guesses (H4's lengths and ladder, H2R's thresholds). A calibration run that only sets constants,
   followed by a preregistered test on fresh seeds, would have caught them.
6. **Do not trim the check on what a blind rule chose.** H6's rule took the smallest budget, the weakest key, and the compute
   trim dropped the larger budget that would have tested how far the verdict carries.

## Open synthesis, not a finding

Each thread's dropped property involved inferring what caused new data rather than reacting to how
surprising it is (`MISSION_STATE.md`). No experiment here has tested this yet.
