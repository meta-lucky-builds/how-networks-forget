#!/usr/bin/env python3
"""H1 Stage A: tune, evaluate and apply the gate. See hypotheses/H1_corroboration_gating.md.

Run from anywhere; only numpy and scipy are needed:

    nohup python experiments/h1_stage_a.py > h1_stage_a.stdout 2>&1 &
    tail -f results/h1_stage_a/run.log

Quick check that everything runs (about a minute, goes to its own directory, and
its results mean nothing):

    python experiments/h1_stage_a.py --smoke

Outputs in the output directory:
    run.log          progress log
    raw_results.csv  one row per run (phase, learner, config, K, context, seed, MSE),
                     flushed row by row
    summary.json     written atomically, only on success

Every result is deterministic, so a killed run resumes by launching the same
command: rows already in raw_results.csv are reused. After a code change, pass
--fresh so stale rows are not reused.

Decisions the preregistration leaves open, all stated in summary.json:
- `bayes` and `oracle_events` have no tunable parameters. `bayes` is given the
  true (h, p, sigma, R) of each context, because it is the ceiling for that
  context. Every other learner is tuned on `mixed` only and keeps that config.
- K convergence is measured on the mean MSE over the evaluation seeds and must
  hold in all four contexts. K is the smaller value of the converged pair.
- Gate: the clause on `deferred_nocheck` has no explicit CI. The stricter reading
  (CI excluding 0) decides; `nocheck_mean_only_pass` shows the laxer one.

Runtime estimate for the full run, serial on one core (no parallelism is used):
tuning about 2 minutes, evaluation of the other learners about 1 minute, bayes
K ladder about 3 to 5 minutes, bootstrap seconds. Roughly 10 minutes in total,
and under 20 on a slow Colab CPU. This is reasoned from operation counts, since
nothing has been executed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from clu.tracking.learners import (  # noqa: E402
    LEARNERS,
    Bayes,
    OracleEvents,
    run_learner,
    run_oracle,
)
from clu.tracking.streams import (  # noqa: E402
    CONTEXTS,
    DEFAULT_T,
    RADIUS,
    SIGMA,
    Stream,
    generate_stream,
)

CONTEXT_NAMES = tuple(CONTEXTS)
TUNED_LEARNERS = ("fixed", "monotone", "robust", "robust_bocd", "deferred", "deferred_nocheck")
ALL_LEARNERS = TUNED_LEARNERS + ("bayes", "oracle_events")

CONFIG_SEED = 20261103
BOOTSTRAP_SEED = 20261104
K_CONVERGENCE_TOLERANCE = 0.01

# `deferred` and its ablation draw the same 30 candidates, so the two tuning
# budgets are identical rather than merely equal in size.
CONFIG_STREAM = {
    "fixed": 0,
    "monotone": 1,
    "robust": 2,
    "robust_bocd": 3,
    "deferred": 4,
    "deferred_nocheck": 4,
}

log = logging.getLogger("h1_stage_a")


@dataclass(frozen=True)
class Plan:
    T: int
    n_configs: int
    tune_seeds: tuple[int, ...]
    eval_seeds: tuple[int, ...]
    k_ladder: tuple[int, ...]
    n_bootstrap: int


FULL_PLAN = Plan(
    T=DEFAULT_T,
    n_configs=30,
    tune_seeds=tuple(range(0, 10)),
    eval_seeds=tuple(range(100, 150)),
    k_ladder=(16, 32, 64, 128, 256, 512),
    n_bootstrap=10000,
)
SMOKE_PLAN = Plan(
    T=200,
    n_configs=3,
    tune_seeds=(0, 1),
    eval_seeds=(100, 101, 102, 103),
    k_ladder=(4, 8, 16),
    n_bootstrap=500,
)


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def sample_config(name: str, rng: np.random.Generator) -> dict:
    if name == "fixed":
        return {"alpha": _log_uniform(rng, 0.01, 1.0)}
    if name == "monotone":
        alpha_lo = _log_uniform(rng, 0.01, 1.0)
        return {
            "alpha_lo": alpha_lo,
            "alpha_hi": float(rng.uniform(alpha_lo, 1.0)),
            "scale": _log_uniform(rng, 0.5, 20.0),
        }
    if name == "robust":
        return {
            "process_var": _log_uniform(rng, 1e-3, 50.0),
            "c": _log_uniform(rng, 0.5, 20.0),
        }
    if name == "robust_bocd":
        return {
            "hazard": _log_uniform(rng, 0.005, 0.3),
            "c": _log_uniform(rng, 0.5, 20.0),
            "prior_var": _log_uniform(rng, 10.0, 400.0),
        }
    if name in ("deferred", "deferred_nocheck"):
        k = int(rng.integers(1, 6))
        alpha = _log_uniform(rng, 0.01, 1.0)
        return {
            "alpha": alpha,
            "commit_alpha": float(rng.uniform(alpha, 1.0)),
            "threshold": _log_uniform(rng, 1.0, 10.0),
            "k": k,
            "window": k + int(rng.integers(0, 6)),
            "tolerance": _log_uniform(rng, 0.5, 8.0),
        }
    raise ValueError(f"no tuning space for {name}")


def make_stream(context: str, seed: int, T: int) -> Stream:
    rng = np.random.default_rng([seed, CONTEXT_NAMES.index(context)])
    return generate_stream(context, rng, T=T)


def mse(estimate: np.ndarray, m: np.ndarray) -> float:
    return float(np.mean((estimate - m) ** 2))


def run_tuned(name: str, config: dict, context: str, seed: int, T: int) -> float:
    stream = make_stream(context, seed, T)
    return mse(run_learner(LEARNERS[name](**config), stream.y), stream.m)


def run_bayes(context: str, seed: int, T: int, top_k: int) -> float:
    stream = make_stream(context, seed, T)
    hazard, outlier_prob = CONTEXTS[context]
    learner = Bayes(hazard, outlier_prob, sigma=SIGMA, radius=RADIUS, top_k=top_k)
    return mse(run_learner(learner, stream.y), stream.m)


def run_oracle_events(context: str, seed: int, T: int) -> float:
    stream = make_stream(context, seed, T)
    oracle = OracleEvents(sigma=SIGMA, radius=RADIUS)
    return mse(run_oracle(oracle, stream.y, stream.changepoint, stream.outlier), stream.m)


class RawStore:
    """Append-only per-run CSV that doubles as the resume checkpoint."""

    FIELDS = ("phase", "learner", "config_index", "k", "context", "seed", "mse", "config")

    def __init__(self, path: Path, fresh: bool) -> None:
        self.path = path
        self.values: dict[tuple[str, ...], float] = {}
        rows = [] if fresh else self._read_valid_rows()
        # Rewrite without any line cut short by a crash, then append after it.
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
        for row in rows:
            self.values[self._key_of_row(row)] = float(row["mse"])
        self._handle = path.open("a", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDS)

    def _read_valid_rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                # A None key holds surplus fields; a None value is a missing one.
                if None in row or any(row.get(field) is None for field in self.FIELDS):
                    continue
                try:
                    float(row["mse"])
                except ValueError:
                    continue
                rows.append(row)
        return rows

    @staticmethod
    def _key_of_row(row: dict[str, str]) -> tuple[str, ...]:
        return (
            row["phase"],
            row["learner"],
            row["config_index"],
            row["k"],
            row["context"],
            row["seed"],
        )

    def get_or_compute(
        self,
        phase: str,
        learner: str,
        config_index: int,
        k: int,
        context: str,
        seed: int,
        config: dict | None,
        compute: Callable[[], float],
    ) -> float:
        key = (phase, learner, str(config_index), str(k), context, str(seed))
        if key in self.values:
            return self.values[key]
        value = compute()
        self._writer.writerow(
            {
                "phase": phase,
                "learner": learner,
                "config_index": config_index,
                "k": k,
                "context": context,
                "seed": seed,
                "mse": repr(value),
                "config": json.dumps(config) if config is not None else "",
            }
        )
        self._handle.flush()
        self.values[key] = value
        return value

    def close(self) -> None:
        self._handle.close()


def tune(name: str, plan: Plan, store: RawStore) -> tuple[int, dict, float]:
    """Pick the config with the lowest mean MSE on `mixed` over the tuning seeds."""
    best_index, best_config, best_score = -1, {}, math.inf
    for index in range(plan.n_configs):
        rng = np.random.default_rng([CONFIG_SEED, CONFIG_STREAM[name], index])
        config = sample_config(name, rng)
        scores = [
            store.get_or_compute(
                "tune",
                name,
                index,
                0,
                "mixed",
                seed,
                config,
                lambda seed=seed: run_tuned(name, config, "mixed", seed, plan.T),
            )
            for seed in plan.tune_seeds
        ]
        score = float(np.mean(scores))
        if score < best_score:
            best_index, best_config, best_score = index, config, score
    return best_index, best_config, best_score


def select_k(plan: Plan, store: RawStore) -> tuple[int, bool, dict[str, dict[str, float]]]:
    """Double K until doubling changes the mean MSE by under 1% in every context.

    Returns the selected K, whether the criterion was met, and the means per K.
    """
    table: dict[str, dict[str, float]] = {}
    previous_k: int | None = None
    for k in plan.k_ladder:
        started = time.time()
        table[str(k)] = {}
        for context in CONTEXT_NAMES:
            values = [
                store.get_or_compute(
                    "bayes_k",
                    "bayes",
                    0,
                    k,
                    context,
                    seed,
                    None,
                    lambda context=context, seed=seed: run_bayes(context, seed, plan.T, k),
                )
                for seed in plan.eval_seeds
            ]
            table[str(k)][context] = float(np.mean(values))
        log.info("bayes K=%d done in %.0fs: %s", k, time.time() - started, table[str(k)])
        if previous_k is not None:
            changes = [
                abs(table[str(k)][c] - table[str(previous_k)][c]) / table[str(k)][c]
                for c in CONTEXT_NAMES
            ]
            log.info("K %d -> %d, largest relative change %.5f", previous_k, k, max(changes))
            if max(changes) < K_CONVERGENCE_TOLERANCE:
                return previous_k, True, table
        previous_k = k
    log.warning("bayes did not meet the K criterion; using the largest K")
    return plan.k_ladder[-1], False, table


def compare(baseline: np.ndarray, candidate: np.ndarray, resamples: np.ndarray) -> dict:
    """Paired comparison of per-seed MSE. A positive difference favours `candidate`."""
    diff = baseline - candidate
    boot = diff[resamples].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_excludes_0_in_favour": bool(low > 0.0),
    }


def gap_closed(
    fixed: np.ndarray, bayes: np.ndarray, learner: np.ndarray, resamples: np.ndarray
) -> dict:
    def gap(f: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        return (f - x) / (f - b)

    point = gap(np.mean(fixed), np.mean(bayes), np.mean(learner))
    boot = gap(
        fixed[resamples].mean(axis=1),
        bayes[resamples].mean(axis=1),
        learner[resamples].mean(axis=1),
    )
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"gap_closed": float(point), "ci_low": float(low), "ci_high": float(high)}


def rounded(value):
    """Six significant digits, for the compact summary on stdout."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def analyse(scores: dict[str, dict[str, np.ndarray]], resamples: np.ndarray) -> dict:
    mean_mse = {
        name: {context: float(scores[name][context].mean()) for context in CONTEXT_NAMES}
        for name in ALL_LEARNERS
    }
    gaps = {
        name: {
            context: gap_closed(
                scores["fixed"][context],
                scores["bayes"][context],
                scores[name][context],
                resamples,
            )
            for context in CONTEXT_NAMES
        }
        for name in ALL_LEARNERS
    }

    mixed = {name: scores[name]["mixed"] for name in ALL_LEARNERS}
    deferred_vs = {
        name: compare(mixed[name], mixed["deferred"], resamples)
        for name in ("monotone", "robust", "deferred_nocheck", "robust_bocd", "fixed")
    }
    beats_monotone = deferred_vs["monotone"]["ci_excludes_0_in_favour"]
    beats_robust = deferred_vs["robust"]["ci_excludes_0_in_favour"]
    beats_nocheck_ci = deferred_vs["deferred_nocheck"]["ci_excludes_0_in_favour"]
    beats_nocheck_mean = deferred_vs["deferred_nocheck"]["mean_diff"] > 0.0
    gate = {
        "beats_monotone": beats_monotone,
        "beats_robust": beats_robust,
        "beats_nocheck_ci": beats_nocheck_ci,
        "pass": beats_monotone and beats_robust and beats_nocheck_ci,
        "nocheck_mean_only_pass": beats_monotone and beats_robust and beats_nocheck_mean,
    }

    # Nothing given only y can be reliably better than the posterior mean, and
    # bayes cannot be reliably better than the oracle. A hit points to a bug.
    anomalies = []
    for context in CONTEXT_NAMES:
        for name in TUNED_LEARNERS:
            result = compare(scores["bayes"][context], scores[name][context], resamples)
            if result["ci_excludes_0_in_favour"]:
                anomalies.append({"better": name, "than": "bayes", "context": context, **result})
        result = compare(scores["oracle_events"][context], scores["bayes"][context], resamples)
        if result["ci_excludes_0_in_favour"]:
            anomalies.append({"better": "bayes", "than": "oracle_events", "context": context, **result})

    return {
        "mean_mse": mean_mse,
        "gap_closed": gaps,
        "deferred_vs_mixed": deferred_vs,
        "gate": gate,
        "sanity_anomalies": anomalies,
    }


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def experiment(plan: Plan, out_dir: Path, fresh: bool, is_smoke: bool) -> None:
    started = time.time()
    store = RawStore(out_dir / "raw_results.csv", fresh)
    try:
        scores: dict[str, dict[str, np.ndarray]] = {}
        chosen: dict[str, dict] = {}

        for name in TUNED_LEARNERS:
            phase_started = time.time()
            index, config, tune_score = tune(name, plan, store)
            chosen[name] = {"config_index": index, "config": config, "tune_mse": tune_score}
            log.info(
                "tuned %s in %.0fs: mse=%.4f config=%s",
                name, time.time() - phase_started, tune_score, config,
            )
            scores[name] = {
                context: np.array(
                    [
                        store.get_or_compute(
                            "eval",
                            name,
                            index,
                            0,
                            context,
                            seed,
                            config,
                            lambda context=context, seed=seed: run_tuned(
                                name, config, context, seed, plan.T
                            ),
                        )
                        for seed in plan.eval_seeds
                    ]
                )
                for context in CONTEXT_NAMES
            }
            log.info("evaluated %s, total elapsed %.0fs", name, time.time() - started)

        scores["oracle_events"] = {
            context: np.array(
                [
                    store.get_or_compute(
                        "eval",
                        "oracle_events",
                        0,
                        0,
                        context,
                        seed,
                        None,
                        lambda context=context, seed=seed: run_oracle_events(context, seed, plan.T),
                    )
                    for seed in plan.eval_seeds
                ]
            )
            for context in CONTEXT_NAMES
        }
        log.info("evaluated oracle_events, total elapsed %.0fs", time.time() - started)

        selected_k, converged, k_table = select_k(plan, store)
        # Reuse the K-ladder runs and also record them as evaluation rows.
        scores["bayes"] = {
            context: np.array(
                [
                    store.get_or_compute(
                        "eval",
                        "bayes",
                        0,
                        selected_k,
                        context,
                        seed,
                        None,
                        lambda context=context, seed=seed: store.values[
                            ("bayes_k", "bayes", "0", str(selected_k), context, str(seed))
                        ],
                    )
                    for seed in plan.eval_seeds
                ]
            )
            for context in CONTEXT_NAMES
        }
        log.info("bayes K=%d converged=%s", selected_k, converged)

        rng = np.random.default_rng(BOOTSTRAP_SEED)
        # Whole seeds are resampled, and every comparison shares the resamples.
        resamples = rng.integers(0, len(plan.eval_seeds), size=(plan.n_bootstrap, len(plan.eval_seeds)))
        analysis = analyse(scores, resamples)

        summary = {
            "smoke": is_smoke,
            "T": plan.T,
            "n_configs": plan.n_configs,
            "tune_seeds": [plan.tune_seeds[0], plan.tune_seeds[-1]],
            "eval_seeds": [plan.eval_seeds[0], plan.eval_seeds[-1]],
            "n_bootstrap": plan.n_bootstrap,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "chosen": chosen,
            "bayes_k": {
                "selected": selected_k,
                "criterion_met": converged,
                "mean_mse_by_k": k_table,
            },
            "notes": {
                "bayes_parameters": "true (h, p, sigma, R) of each context",
                "gate_nocheck_rule": "CI excluding 0; see nocheck_mean_only_pass",
                "estimate_timing": "estimate after seeing y_t, for every learner",
            },
            **analysis,
            "elapsed_seconds": time.time() - started,
        }
        write_atomic(out_dir / "summary.json", json.dumps(summary, indent=2))
        log.info("wrote %s", out_dir / "summary.json")
        log.info("GATE %s", "PASS" if analysis["gate"]["pass"] else "FAIL")
        print(json.dumps(rounded(summary), separators=(",", ":")), flush=True)
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--fresh", action="store_true", help="ignore rows from earlier runs")
    parser.add_argument("--smoke", action="store_true", help="tiny run to check the plumbing")
    args = parser.parse_args()

    plan = SMOKE_PLAN if args.smoke else FULL_PLAN
    default_name = "h1_stage_a_smoke" if args.smoke else "h1_stage_a"
    out_dir = args.out if args.out is not None else REPO_ROOT / "results" / default_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stderr)],
    )
    log.info("start: plan=%s out=%s fresh=%s", plan, out_dir, args.fresh)
    try:
        experiment(plan, out_dir, args.fresh, args.smoke)
    except Exception:
        log.exception("run failed; rerun the same command to resume")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
