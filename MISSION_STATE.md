# Mission state

Durable state for a long-running research mission. Anyone resuming, human or
agent, starts here, then reads `docs/CHARTER.md`.

**Last heartbeat:** 2026-11-03T20:10Z
**Working branch:** `research/bio-to-ai`
**Phase:** 1, literature threads running

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

## In progress

- Literature threads writing to `docs/literature/`: consolidation,
  neuromodulation, synaptic, gating_metacognition.

## Next

1. Synthesise the threads into `docs/MAPPINGS.md` and preregister hypotheses.
2. Implement mechanisms and tests in `src/clu/` and `tests/`.
3. Run on Colab, record in `results/`, then iterate.

## Working hypothesis, not yet checked against the literature

Biology appears to couple its stability and plasticity mechanisms through shared
neuromodulatory signals, while AI work tends to study them separately. If so, the
coupling itself should have a measurable effect: an interaction term, beyond the
main effects of the same components run uncoupled. Literature threads are asked
for evidence on this without being told to confirm it.

## Open items

- The owner mentioned two context files, but none were attached to the session.
  Work proceeds without them.
