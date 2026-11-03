# Mission state

Durable state for a long-running research mission. Anyone resuming, human or
agent, starts here, then reads `docs/CHARTER.md`.

**Last heartbeat:** 2026-11-03T21:30Z
**Working branch:** `research/bio-to-ai`
**Phase:** 3, H1 stopped on a negative result, H2 running

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
- H2 preregistered and implemented at 1274e77.

## In progress

- H2 on Colab, in `/content/clu`: tests, then smoke, then the full run (about 2.5 h
  by estimate, 2 workers). The runtime is ephemeral and the repo holds the only
  durable copy of the code, so results must be copied back and committed. If the
  runtime is lost, re-upload and rerun; results are deterministic.

## Bets

| Bet | Claim | Status |
| --- | --- | --- |
| H1 | Commit surprising samples only after corroboration | **Stopped.** H1b (hedge predictions, defer weight commits) is parked in RESULT.md |
| H2 | One multi-timescale state sets both protection and recycling (`synaptic.md` Exp 1, 2) | Running |
| H3 | Gates must be self-generated, without task labels or blocks (`gating_metacognition.md` Exp 1) | Not started |
| H4 | Replay dose set by relation to existing knowledge (`consolidation.md` E1, E2) | Not started |

Kill rules live in each preregistration. H1 failing its gate does not end the
mission: record it and move to H2.

## Revised view of what connects the threads

Each thread's dropped property turned out to involve inferring what caused new data
rather than reacting to how surprising it is. This is our synthesis, not a finding.

My first working hypothesis, a single global neuromodulatory scalar coupling all
plasticity, is weakened: a shared signal is contested in biology and AI evidence on
granularity is mixed. It remains an open test (`neuromodulation.md` Exp 2).

## Next

1. Read H2 `summary.json`. Check `control_valid` and the manipulation check first,
   apply the power rule (`gate_std`), then record the verdict in `results/`.
2. Meanwhile, preregister H3 or H4 with an analytic bound first (charter rule).
3. Write `docs/MAPPINGS.md`, including negative results.

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
