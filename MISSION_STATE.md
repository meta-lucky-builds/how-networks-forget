# Mission state

Durable state for a long-running research mission. Anyone resuming, human or
agent, starts here, then reads `docs/CHARTER.md`.

**Last heartbeat:** 2026-11-04T00:00Z
**Working branch:** `research/bio-to-ai`
**Phase:** 2, H1 Stage A running on Colab

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
- H1 preregistered: `hypotheses/H1_corroboration_gating.md`
- H1 Stage A code at 14ec174. 39 tests pass on Colab.

## In progress

- H1 Stage A running on Colab, in `/content/clu`. The runtime is ephemeral and the
  repo holds the only durable copy of the code, so results must be copied back and
  committed. If the runtime is lost, re-upload and rerun; results are deterministic.

## Bets

| Bet | Claim | Status |
| --- | --- | --- |
| H1 | Commit surprising samples only after corroboration | Stage A running |
| H2 | One multi-timescale state sets both protection and recycling (`synaptic.md` Exp 1, 2) | Not started; next independent bet |
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

1. Read Stage A `summary.json`, check `sanity_anomalies`, commit results, apply the
   gate as written.
2. Pass: implement Stage B (scale to 2 CPU cores, log the reduction as a deviation).
   Fail: record why and start H2.
3. Write `docs/MAPPINGS.md`.

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
