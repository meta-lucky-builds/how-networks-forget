# Mission state

Durable state for a long-running research mission. Anyone resuming, human or
agent, starts here, then reads `docs/CHARTER.md`.

**Last heartbeat:** 2026-11-04T07:20Z
**Working branch:** `research/bio-to-ai`
**Phase:** 3, H1 negative, H2 and H2R closed untested, H4 stopped untested, H3 next

## Standing instructions from the project owner

- Mission: map what humans do biologically to rigorous AI implementations.
  Read, hypothesise, gather, analyse, implement, fix, iterate. It is meant to
  keep running rather than finish in one pass.
- Apply the `how-to-think` skill whenever hypothesising, planning, analysing or
  designing, and tell agents to use it when needed.
- Models, all at medium effort: Opus 5.5 for implementation and thinking, Sonnet
  5.5 for executing finished work, simple tasks and web search, Haiku 5.5 for
  simple retrieval.
- All work goes in the `meta-lucky-builds/clu` repo.
- Never run tests locally, even CPU ones. Use Colab MCP; other cloud is allowed.
- Make small decisions autonomously; the owner is away. Act safely.
- Resume automatically after usage limits, and use usage carefully without
  lowering quality.
- Keep good context and usage habits. Adapt: refine or form new hypotheses and
  approaches as evidence arrives, and do not stay in a local optimum when a line
  is not yielding anything. In practice: keep several bets open, give each a
  preset kill criterion, and record why any line was dropped.
- Keep context small: subagents write detail to repo files and reply briefly;
  this file is the summary a resume depends on.

## Done

- Charter and protocol: `docs/CHARTER.md`
- Literature notes: `docs/literature/` (four threads plus an adversarial prior-art
  check). Read depth is recorded per source; many primaries are abstract-level.
- H1 preregistered and run. **Negative: Stage A gate failed, H1 stopped.**
  `results/h1_stage_a/RESULT.md`. Withholding on arrival costs about twice what
  hedging costs under squared error. Search budget was not the cause.
- H2 run on GitHub Actions. **No conclusion: manipulation check and power rule failed.**
  `results/h2_shared_state/RESULT.md`. Label permutation capped zero-shot near 0.2 for any
  learner, savings tracked speed and was censored, the gate was almost constant, and SGD
  lost no plasticity. All design errors. Charter now requires a Stage 0 with a positive
  control.
- H2R Stage 0 **failed, no retry allowed, H2 line closed.** `results/h2r/RESULT.md`. Its
  short-gap threshold was arithmetically unreachable, my error, and the rule stands anyway.
- H4 Stage 0 **failed, stopped.** `results/h4/RESULT.md`. `dose` 96% censored, first-order r 0.445.
- `docs/MAPPINGS.md` written: no mapping supported so far, plus the method lessons.

## In progress

- H5 (overwriting or access, requested by the owner 2026-11-04): `perm_shared` and `perm_class` passed
  calibration and are confirming on Actions. Then `calibrate-retry` for `split_class` only (Deviation 3),
  then its `confirm` if answerable. Request format: line 1 mode, line 2 regimes, line 3 free text.
- **The Colab runtime is shared with work we do not own** (`experiments/thread_a.py`,
  `mechinterp`), possibly another session of the owner's. Never kill those processes or
  delete cells that are not ours. Use Colab only for small tests, and expect delays.

## Bets

| Bet | Claim | Status |
| --- | --- | --- |
| H1 | Commit surprising samples only after corroboration | **Stopped.** H1b (hedge predictions, defer weight commits) is parked in RESULT.md |
| H2 | One multi-timescale state sets both protection and recycling (`synaptic.md` Exp 1, 2) | **No conclusion.** Design could not answer |
| H2R | H2 with input permutation, rank gates of mean 1, a repeat-versus-novel metric | **Stopped at Stage 0, H2 line closed.** No retry allowed. Claim untested |
| H3 | Gates must be self-generated, without task labels or blocks (`gating_metacognition.md` Exp 1) | Not started, next |
| H4 | Does schema consistency add anything beyond gradient interference? | **Stopped at Stage 0.** Claim untested |
| H5 | Forgetting is mostly overwriting or mostly loss of access, by old-versus-new on one network | Preregistered, calibrating |

Kill rules live in each preregistration. H1 failing its gate does not end the
mission: record it and move to H2.

## Revised view of what connects the threads

Each thread's dropped property turned out to involve inferring what caused new data
rather than reacting to how surprising it is. This is our synthesis, not a finding.

My first working hypothesis, a single global neuromodulatory scalar coupling all
plasticity, is weakened: a shared signal is contested in biology and AI evidence on
granularity is mixed. It remains an open test (`neuromodulation.md` Exp 2).

## Next

**Owner's instruction, 2026-11-04: focus only on H5, overwriting or loss of access. All other lines are
paused.** Finish calibration, confirmation and the verdict before anything else. H3 and new bets wait.

Usage note: on 2026-11-03 the 5-hour window reached 66% about 75 minutes in, driven by
five research agents and a large coordinator context. Wait cheaply while Colab runs.

## Execution notes

- Colab: CPU-only, 2 cores, 12 GB. Code is uploaded through notebook cells and
  verified by checksum, so no credentials go in the notebook.
- Run long jobs with nohup and read the log; do not block a cell on them.
- Subagent research threads cost 260k to 440k tokens each. Keep briefs narrow.

## Open items

- The owner mentioned two context files, but none were attached to the session.
  Work proceeds without them.
- Citations from 2026 preprints come from one agent's prior-art pass and have not
  been independently re-opened.
