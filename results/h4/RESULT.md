# H4 Stage 0: failed, H4 stopped

**Verdict: H4 stops as preregistered.** No full run. Neither H4a nor H4b was tested, and this
says nothing about whether schema consistency adds anything beyond gradient interference.

Run 41755160533, seeds 300 to 302, disjoint from evaluation seeds. Data: `stage0.json`.

| Check | Measured | Threshold | Result |
| --- | --- | --- | --- |
| Zero-replay drop | 57.1 points | at least 5 | pass |
| Top rung reduction | 0.85 of the drop | at least 0.5 | pass |
| New-class accuracy, medium, zero replay | 99.9% | at least 70% | pass |
| First-order r, short length | 0.445 | at least 0.5 | **fail** |
| Variance inflation | 2.42 | at most 5 | pass |
| Censored `dose` | 96% | under 50% | **fail** |
| Every schema reaches criterion | 100% | 100% | pass |

The preregistration has no retry: "If a check fails, H4 stops with that reason recorded." The raw
schema effect was hidden from Stage 0 by design and was not looked at.

## What the failures mean

Everything here is post hoc and supports no claim. It only informs future designs.

- **`dose` could not vary.** The top rung leaves a drop of about 8.7 points at the medium length,
  over the 2-point target, so almost every event is censored. The ladder stops at one old sample per
  new sample, which was too low for this learning rate and length.
- **The new class overwhelms the old ones.** New-class accuracy near 100% with a 57-point drop means
  the network mostly learns to predict the new class. That is a much coarser failure than the
  fine interference H4 wanted to measure.
- **The validity check compared unlike quantities (my design error).** The bound is on old-class
  loss, and the check used an accuracy drop, which is a nonlinear function of the margins. So
  r of 0.445 does not show that the first-order account fails for loss, and this run cannot tell the
  two apart.
- **Stated risk realised.** Deviation 3 warned that the lengths rested on an estimate and that
  Stage 0 could fail on them.

Stage 0 cost about 3 minutes of experiment time and avoided a full run projected at about 127.
