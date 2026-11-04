#!/usr/bin/env python3
"""H2R experiment; see hypotheses/H2R_shared_state_redesign.md. Numpy, backprop from h2_shared_state.

python experiments/h2r.py --stage0            # writes results/h2r/stage0.json
python experiments/h2r.py --stage0 --retry    # the one permitted retry, all rates at 3x
python experiments/h2r.py                     # full run; refuses unless stage0.json has an overall pass
Writes run.log, raw_results.csv (resume checkpoint), and atomically stage0.json and summary.json.
Rerun the same command to resume. After a code change, delete results/h2r first. Choices the
preregistration leaves open are in NOTES.

Runtime plan (estimates; nothing here has been timed).
H2: 270 runs, 165 min on 2 cores, so 330 core-min. Its steps: 180 tuning runs of 40k plus 90
evaluation runs of 116k (task steps, probes, fresh copies) = 17.6M, about 1.1 ms per step. That
is used below as a pessimistic base for every condition, since H2's sgd did not pay for its traces.
Rank refresh every 10 steps: an argsort of the 100k-entry first layer (about 10 ms) plus the
permutation, the writes and the smaller layers gives about 15 ms, or 1.5 ms per step. both_multi
ranks 4 times per refresh (3 traces and the re-rank), 6 ms per step. both_perm also permutes, about
0.2 ms per step on top of its rank. So ms per step: sgd, shrink_perturb 1.1; protect, noise_util,
both_util 2.6; both_perm 2.8; both_multi 7.1. A run is 106k steps at evaluation (100k, and 10 probes of 600).
Stage 0: tuning 3 conditions, 20 runs of 40k each, 5040 core-s; 15 runs of 106k, about 3000 core-s.
    About 67 min on 2 workers. A retry reruns all 5 conditions at evaluation length, about 25 min.
Full, reusing Stage 0's tuning: tuning 4 more conditions 10880 core-s; evaluation 7 conditions, 10
    seeds, 21090 core-s; both_perm sensitivity block, if its own best config differs, 5720 core-s.
    About 314 min on 2 workers (266 without the sensitivity block), against a 320 min timeout.
So one resume is plausible, and it loses at most the runs in flight (both_multi, about 13 min each). No reduction of the design is made here.
stage0.json carries a projection from measured timings, to check before requesting the full run.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, logging, math, os, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2_shared_state as h2  # noqa: E402

from h2_shared_state import (LONG_GAP, MULTI_TAUS, N_PARAMS, PROBE_EVERY, PROBE_STEPS, SHAPES, STEPS,  # noqa: E402,F401
                             WINDOW, contrast, grad_step, init_weights, layer_views, load_mnist, online,
                             rolling, schedule_repeats, zero_shot)

log = logging.getLogger("h2r")
NOTES = [
    "tuning precedes Stage 0: Stage 0 runs at the tuned configs of sgd, protect and both_util, which"
    " the full run reuses; the preregistration gives Stage 0 no other source of hyperparameters",
    "within-layer means within each of the 6 weight and bias arrays",
    "ties in a trace (dead units, silent pixels) are broken at random, so every refresh of every"
    " condition has the exact same gate multiset; the gate SD check therefore cannot fail, and random"
    " tie-breaks can only lower the Spearman check",
    "the first refresh is at step 1, so a gate exists from the first update; traces update each step",
    "both_perm: P comes from the real u, exactly as in both_util; only R is the re-permuted gate, so the"
    " contrast isolates the coordination of noise with protection (Deviation 1)",
    "no bias correction or sigmoid: rank is scale-invariant, so neither can change u",
    "reuse = steps(repeat) - mean(steps(two nearest non-repeat tasks)), so lower is better; gain ="
    " -reuse in contrasts; a repeat's source counts as non-repeat; distance ties go to the earlier task",
    "steps are on 100-sample rolling online accuracy, floor 100; an event is censored if the repeat or"
    " either comparator misses 0.75; censored tasks count as task length and stay in, which shrinks"
    " differences toward 0",
    "accuracy at the end of a source task is its last 100-sample rolling online accuracy",
    "Stage 0 checks without 'all 3 seeds' in the preregistration use the mean over seeds",
    "probe AUC quarters are by task index: first quarter task < 25, last quarter task >= 75, which"
    " holds 2 and 3 of the 10 probes; the verdict's probe AUC is the mean of all 10, as in H2",
    "the retry multiplies every condition's tuned rate by 3 and reruns all of Stage 0's conditions;"
    " after a retry pass the full run does the same (Deviation 2)",
    "a retry is allowed only when every failed check is plasticity or tension",
    "Stage 0 has its own image pool, so it shares images with neither tuning nor evaluation",
    "both_perm tunes over both_util's candidates; the primary contrast is at both_util's config, its"
    " own best is a sensitivity block, as in H2",
    "retention in verdicts averages short and long gaps, as in H2; the manipulation check is either"
    " retention measure with a plain 95% CI, as in H2",
    "Holm intervals: step-down percentile intervals, level 0.025 for the smaller bootstrap p, then"
    " 0.05 only if the first excludes 0; p = 2 min(P(b<=0), P(b>=0))",
    "noise and ties use numpy's generator, keyed by seed, condition and task; tied order is independent"
    " of which SIMD sort a CPU dispatches, but floating point can still differ between CPUs",
]
REFRESH, TARGET, RETRY_LR_MULTIPLE = 10, 0.75, 3.0
ROOT_SEED, CONFIG_SEED, BOOT_SEED, SUBSET_SEED = 20261115, 20261116, 20261117, 20261118
K_STREAM, K_INIT, K_NOISE, K_PROBE, K_P_NOISE, K_P_INIT, K_P_FRESH = range(7)
COND = {"sgd": (0, None), "protect": (1, None), "noise_util": (0, "util"), "both_util": (1, "util"),
        "both_perm": (1, "util"), "both_multi": (1, "util"), "shrink_perturb": (0, "plain"),
        "oracle": (0, None)}  # condition: (protect, noise)
CONDS = tuple(COND)
FULL_CONDS = tuple(c for c in CONDS if c != "oracle")
STAGE0_CONDS = ("sgd", "protect", "both_util", "both_perm", "oracle")
STAGE0_TUNED = ("sgd", "protect", "both_util")
RETRY_ITEMS = {"plasticity", "tension"}
# The first member of a group draws the configs. both_perm shares both_util's, oracle is sgd.
GROUPS = (("sgd", "oracle"), ("protect",), ("noise_util",), ("both_util", "both_perm"), ("both_multi",),
          ("shrink_perturb",))
GROUP_OF = {c: g for g in GROUPS for c in g}
METRICS = ("online_acc", "zs_short", "zs_long", "reuse_short", "reuse_long", "reuse_censored_frac", "forget_long",
           "probe_auc", "auc_first", "auc_last", "gate_mean", "gate_std", "gate_spearman")
_ENDS = np.cumsum([0] + [int(np.prod(s)) for s in SHAPES]).tolist()
SLICES = tuple(zip(_ENDS, _ENDS[1:]))
_GRIDS = {}


class Refused(Exception):
    """A rule of the preregistration forbids this run."""


def key(seed, *tags):
    return np.random.default_rng([ROOT_SEED, seed, *tags])


def grid(n):
    """The values every within-layer u takes: mean exactly 0.5, so P = 2(1 - u) has mean 1."""
    if n not in _GRIDS:
        _GRIDS[n] = ((np.arange(n, dtype=np.float64) + 0.5) / n).astype(np.float32)
    return _GRIDS[n]


def rank_unit(values, rng):
    """Rank in (0, 1). Ties are broken at random, in a way that does not depend on the sort used."""
    n = values.size
    jump = rng.permutation(n)
    order = np.argsort(values[jump])
    ordered = values[jump[order]]
    same = ordered[1:] == ordered[:-1]
    if same.any():
        in_run = np.flatnonzero(np.r_[same, False] | np.r_[False, same])
        run_id = np.cumsum(np.r_[True, ~same])[in_run]
        order[in_run] = order[in_run[np.lexsort((order[in_run], run_id))]]
    out = np.empty(n, dtype=np.float32)
    out[jump[order]] = grid(n)
    return out


def within_layer_rank(trace, rng):
    u = np.empty(N_PARAMS, dtype=np.float32)
    for a, b in SLICES:
        u[a:b] = rank_unit(trace[a:b], rng)
    return u


def rerank_max(traces, rng):
    u = np.empty(N_PARAMS, dtype=np.float32)
    for a, b in SLICES:
        u[a:b] = rank_unit(np.maximum.reduce([rank_unit(t[a:b], rng) for t in traces]), rng)
    return u


def permuted_gate(rng):
    u = np.empty(N_PARAMS, dtype=np.float32)
    for a, b in SLICES:
        u[a:b] = grid(b - a)[rng.permutation(b - a)]
    return u


def rank_corr(a, b):
    """Spearman when both arguments are ranks: near the size-weighted mean of the layers' Spearman."""
    x, y = a.astype(np.float64), b.astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    return float(np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y)))


class Learner:
    def __init__(self, cond, cfg, rng, w):
        self.cond, self.cfg, self.rng, self.w, self.views = cond, cfg, rng, w, layer_views(w)
        self.g = np.zeros_like(w)
        self.gv = layer_views(self.g)
        self.lr, self.sigma = cfg["lr"], cfg.get("sigma", 0.0)
        self.protect, self.noise = COND[cond]
        gated = self.protect or self.noise == "util"
        taus = MULTI_TAUS if cond == "both_multi" else (cfg["tau"],) if gated else ()
        self.rates = [1.0 / tau for tau in taus]
        self.traces = [np.zeros_like(w) for _ in taus]
        self.pool = h2.NoisePool(rng) if self.noise else None
        self.t, self.u, self.P, self.R, self.force_u, self.noise_term = 0, None, None, None, None, None
        self.track, self.sp_sum, self.sp_n = False, 0.0, 0

    def clone(self, rng):
        """Independent probe copy: keeps traces and gate, shares no mutable array or RNG."""
        copy = Learner(self.cond, self.cfg, rng, self.w.copy())
        copy.traces, copy.t = [t.copy() for t in self.traces], self.t
        copy.u, copy.P, copy.R = self.u, self.P, self.R  # replaced at a refresh, never written to
        return copy

    def restore(self, saved):
        self.w[:] = saved

    def _new_u(self):
        if self.force_u is not None:
            return self.force_u
        if len(self.traces) == 1:
            return within_layer_rank(self.traces[0], self.rng)
        return rerank_max(self.traces, self.rng)

    def _refresh(self):
        u = self._new_u()
        if self.track and self.u is not None:
            self.sp_sum += rank_corr(u, self.u)
            self.sp_n += 1
        self.u, self.P = u, 2.0 * (1.0 - u)
        self.R = 1.0 - (permuted_gate(self.rng) if self.cond == "both_perm" else u)

    def step(self, x, y):
        pred = grad_step(self.views, self.gv, x, y)
        self.t += 1
        w, g, lr = self.w, self.g, self.lr
        if self.traces:
            gw = -g * w
            for trace, rate in zip(self.traces, self.rates):
                trace += rate * (gw - trace)
            if (self.t - 1) % REFRESH == 0:
                self._refresh()
        if self.noise == "plain":
            w *= 1.0 - lr * self.cfg["decay"]
        update = g * self.P if self.protect else g  # w -= lr * (g * P + n * R)
        if self.noise:
            n = self.sigma * self.pool.normal()
            self.noise_term = n if self.noise == "plain" else n * self.R
            update = update + self.noise_term
        w -= lr * update
        return pred


@dataclass
class Data:
    X: np.ndarray
    y: np.ndarray
    tune_pool: np.ndarray
    stage0_pool: np.ndarray
    eval_pool: np.ndarray

    def pool(self, seed):
        for low, high, images in ((200, 300, self.tune_pool), (300, 400, self.stage0_pool), (400, 500, self.eval_pool)):
            if low <= seed < high:
                return images
        raise ValueError(f"seed {seed} belongs to no H2R phase")


def make_data(X, y, per_class):
    rng, parts = np.random.default_rng(SUBSET_SEED), ([], [], [])
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(y == c))
        for j, part in enumerate(parts):
            part.append(idx[j * per_class:(j + 1) * per_class])
    keep, n = np.concatenate([i for part in parts for i in part]), 10 * per_class
    return Data(X[keep].astype(np.float32) / 255.0, y[keep].astype(np.int64),
                np.arange(n), np.arange(n, 2 * n), np.arange(2 * n, 3 * n))


def make_stream(seed, n_tasks, steps, pool):
    rng = key(seed, K_STREAM)
    repeats = schedule_repeats(rng, n_tasks)
    perms = np.zeros((n_tasks, 784), dtype=np.int64)
    for t in range(n_tasks):
        perms[t] = perms[repeats[t][0]] if t in repeats else rng.permutation(784)
    idx = np.stack([rng.choice(pool, steps, replace=False) for _ in range(n_tasks)])
    return SimpleNamespace(perms=perms, idx=idx, repeats=repeats)


def task_data(stream, data, task):
    """Pixels permuted per task, labels fixed."""
    idx = stream.idx[task]
    return data.X[idx][:, stream.perms[task]], data.y[idx]


def probe_ratio(learner, cfg, seed, cid, j, data):
    rng = key(seed, K_PROBE, j)
    perm, idx = rng.permutation(784), rng.choice(data.pool(seed), PROBE_STEPS, replace=False)
    xs, labels = data.X[idx][:, perm], data.y[idx]
    fresh = Learner(learner.cond, cfg, key(seed, K_P_FRESH, cid, j), init_weights(key(seed, K_P_INIT, j)))
    return online(learner.clone(key(seed, K_P_NOISE, cid, j)), xs, labels) / online(fresh, xs, labels)


def steps_to_target(correct, target=TARGET):
    """Steps until the rolling accuracy first reaches target, and whether it did."""
    reached = np.flatnonzero(rolling(correct) >= target)
    return (int(reached[0]) + WINDOW, True) if reached.size else (len(correct), False)


def nearest_non_repeats(task, repeats, n_tasks, k=2):
    return sorted((t for t in range(n_tasks) if t not in repeats), key=lambda t: (abs(t - task), t))[:k]


def reuse_event(task, steps, reached, repeats, n_tasks):
    near = nearest_non_repeats(task, repeats, n_tasks)
    reuse = float(steps[task] - np.mean([steps[t] for t in near]))
    return {"task": task, "steps": int(steps[task]), "near": near, "near_steps": [int(steps[t]) for t in near],
            "reuse": reuse, "censored": not (reached[task] and all(reached[t] for t in near))}


def _mean(values):
    return float(np.mean(values)) if len(values) else math.nan


def run_stream(cond, cfg, seed, data, n_tasks, steps=STEPS, metrics=True):
    stream = make_stream(seed, n_tasks, steps, data.pool(seed))
    cid = CONDS.index(cond)
    L = Learner(cond, cfg, key(seed, K_NOISE, cid), init_weights(key(seed, K_INIT)))
    L.track = metrics
    sources = {source for source, _, _ in stream.repeats.values()} if cond == "oracle" else set()
    saved, correct, hits = {}, np.zeros(steps, dtype=bool), 0
    to_target, reached, end_acc = np.zeros(n_tasks, dtype=np.int64), np.zeros(n_tasks, dtype=bool), np.zeros(n_tasks)
    zs, forget, aucs, means, stds = {"short": [], "long": []}, [], [], [], []
    for task in range(n_tasks):
        xs, labels = task_data(stream, data, task)
        rep = stream.repeats.get(task)
        if rep and cond == "oracle":
            L.restore(saved.pop(rep[0]))
        # From the weights in place at the return, before any step of this task.
        zero = zero_shot(L, xs[:WINDOW], labels[:WINDOW]) if rep and metrics else None
        for i in range(steps):
            correct[i] = L.step(xs[i], labels[i]) == labels[i]
        hits += int(correct.sum())
        if task in sources:
            saved[task] = L.w.copy()
        if not metrics:
            continue
        if L.P is not None:
            means.append(float(L.P.mean(dtype=np.float64)))
            stds.append(float(L.P.std(dtype=np.float64)))
        to_target[task], reached[task] = steps_to_target(correct)
        end_acc[task] = float(rolling(correct)[-1])
        if rep:
            zs[rep[2]].append(zero)
            if rep[2] == "long":
                forget.append(float(end_acc[rep[0]]) - zero)
        if (task + 1) % PROBE_EVERY == 0:
            aucs.append((task, probe_ratio(L, cfg, seed, cid, task, data)))
    events = []
    if metrics:
        for task, (source, gap, kind) in sorted(stream.repeats.items()):
            events.append({**reuse_event(task, to_target, reached, stream.repeats, n_tasks), "source": source,
                           "gap": gap, "kind": kind})
    reuse = {kind: [e["reuse"] for e in events if e["kind"] == kind] for kind in ("short", "long")}
    return {"online_acc": hits / (n_tasks * steps), "zs_short": _mean(zs["short"]), "zs_long": _mean(zs["long"]),
            "reuse_short": _mean(reuse["short"]), "reuse_long": _mean(reuse["long"]),
            "reuse_censored_frac": _mean([float(e["censored"]) for e in events]), "forget_long": _mean(forget),
            "probe_auc": _mean([a for _, a in aucs]), "auc_first": _mean([a for t, a in aucs if t < n_tasks / 4]),
            "auc_last": _mean([a for t, a in aucs if t >= 3 * n_tasks / 4]), "gate_mean": _mean(means),
            "gate_std": _mean(stds), "gate_spearman": L.sp_sum / L.sp_n if L.sp_n else math.nan,
            "detail": {"events": events, "aucs": aucs, "w_crc": zlib.crc32(L.w.tobytes())}}


@dataclass(frozen=True)
class Plan:
    tasks: int
    tune_tasks: int
    steps: int
    n_configs: int
    tune_seeds: tuple
    stage0_seeds: tuple
    eval_seeds: tuple
    n_bootstrap: int
    per_class: int = 600


FULL_PLAN = Plan(100, 40, STEPS, 10, (200, 201), (300, 301, 302), tuple(range(400, 410)), 10000)


def group_config(group, index):
    rng, cond = np.random.default_rng([CONFIG_SEED, GROUPS.index(group), index]), group[0]
    lu = lambda low, high: float(math.exp(rng.uniform(math.log(low), math.log(high))))  # noqa: E731
    cfg = {"lr": lu(1e-3, 0.1)}
    if cond in ("protect", "noise_util", "both_util"):
        cfg["tau"] = lu(MULTI_TAUS[0], MULTI_TAUS[-1])
    if cond in ("noise_util", "both_util", "both_multi", "shrink_perturb"):
        cfg["sigma"] = lu(1e-3, 1.0)
    if cond == "shrink_perturb":
        cfg["decay"] = lu(1e-5, 1e-2)
    return cfg


def cfg_for(cond, index, lr_multiplier=1.0):
    cfg = group_config(GROUP_OF[cond], index)
    cfg["lr"] *= lr_multiplier
    return cfg


class RawStore:
    FIELDS = ("phase", "condition", "config_index", "seed", "n_tasks", *METRICS, "seconds", "config", "detail")

    def __init__(self, path):
        rows = []
        if path.exists():
            with path.open(newline="") as handle:
                rows = [r for r in csv.DictReader(handle) if self._complete(r)]
        tmp = path.with_suffix(".tmp")  # rewrite without any line cut short by a crash
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
        self.values = {}
        self._remember(rows)
        self._handle = path.open("a", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDS)

    def _complete(self, row):
        try:
            json.loads(row["detail"])
            [float(row[m]) for m in (*METRICS, "seconds")]
            return None not in row
        except (ValueError, TypeError, KeyError):
            return False

    def _remember(self, rows):
        for r in rows:
            self.values[tuple(str(r[f]) for f in self.FIELDS[:5])] = {m: float(r[m]) for m in (*METRICS, "seconds")}

    def get(self, *k):
        return self.values.get(tuple(str(x) for x in k))

    def add(self, rows):
        self._writer.writerows(rows)
        self._handle.flush()
        self._remember(rows)

    def close(self):
        self._handle.close()


_DATA = None


def _init_worker(cache, per_class):
    global _DATA
    np.seterr(all="ignore")
    _DATA = make_data(*load_mnist(Path(cache)), per_class)


def run_job(job):
    phase, cond, index, cfg, seed, n_tasks, steps, metrics = job
    started = time.time()
    r = run_stream(cond, cfg, seed, _DATA, n_tasks, steps, metrics)
    return [{"phase": phase, "condition": cond, "config_index": index, "seed": seed, "n_tasks": n_tasks,
             **{m: repr(float(r[m])) for m in METRICS}, "seconds": round(time.time() - started, 1),
             "config": json.dumps(cfg), "detail": json.dumps(r["detail"])}]


def run_jobs(jobs, store, pool, label):
    todo = [j for j in jobs if not store.get(j[0], j[1], j[2], j[4], j[5])]
    log.info("%s: %d jobs, %d to run", label, len(jobs), len(todo))
    started = time.time()
    for n, future in enumerate(as_completed([pool.submit(run_job, j) for j in todo]), 1):
        rows = future.result()
        store.add(rows)
        log.info("%s %d/%d %s seed=%s %.0fs", label, n, len(todo), rows[0]["condition"], rows[0]["seed"], time.time() - started)


def tune(plan, store, pool, conds):
    run_jobs([("tune", c, i, cfg_for(c, i), s, plan.tune_tasks, plan.steps, False)
              for c in conds for i in range(plan.n_configs) for s in plan.tune_seeds], store, pool, "tune")
    best = {c: int(np.argmax([np.mean([store.get("tune", c, i, s, plan.tune_tasks)["online_acc"] for s in plan.tune_seeds])
                              for i in range(plan.n_configs)])) for c in conds}
    log.info("tuned: %s", best)
    return best


def derived(out):
    """Add gap-averaged measures. gain = -reuse, so that higher is better in every contrast."""
    out["zs"] = (out["zs_short"] + out["zs_long"]) / 2
    out["gain_short"], out["gain_long"] = -out["reuse_short"], -out["reuse_long"]
    out["gain"] = (out["gain_short"] + out["gain_long"]) / 2
    return out


def table(store, phase, cond, index, seeds, n_tasks):
    rows = [store.get(phase, cond, index, s, n_tasks) for s in seeds]
    return derived({m: np.array([r[m] for r in rows]) for m in METRICS})


def holm_intervals(boots, alpha=0.05):
    """Step-down Holm over a family, given each measure's bootstrap distribution of its difference."""
    p = {k: min(1.0, 2.0 * min(float((b <= 0).mean()), float((b >= 0).mean()))) for k, b in boots.items()}
    m, alive, level, out = len(boots), True, alpha / len(boots), {}
    for i, k in enumerate(sorted(boots, key=lambda name: (p[name], name))):
        if alive:
            level = alpha / (m - i)
        low, high = np.percentile(boots[k], [50.0 * level, 100.0 - 50.0 * level])
        reject = bool(alive and (low > 0.0 or high < 0.0))
        alive = reject
        out[k] = {"level": level, "p": p[k], "ci_low": float(low), "ci_high": float(high), "reject": reject}
    return out


def evaluate(t, res):
    """t[cond][metric] holds per-seed arrays, paired by position. Higher is better for every metric used."""
    c = lambda a, b, m: contrast(t[a][m], t[b][m], res)  # noqa: E731
    up = lambda r: r["ci_low"] > 0.0  # noqa: E731
    within = lambda r: abs(r["diff"]) <= r["pooled_se"]  # noqa: E731
    keeps_auc = lambda r: r["diff"] >= -r["pooled_se"]  # noqa: E731
    m_zs, m_gain = c("both_util", "noise_util", "zs"), c("both_util", "noise_util", "gain")
    m_auc = c("both_util", "protect", "probe_auc")
    manipulation = (up(m_zs) or up(m_gain)) and up(m_auc)
    a = {m: c("both_util", "both_perm", m) for m in ("zs", "gain", "probe_auc", "zs_short", "zs_long", "gain_short", "gain_long")}
    holm = holm_intervals({m: (t["both_util"][m] - t["both_perm"][m])[res].mean(axis=1) for m in ("zs", "gain")})
    a_sup = any(holm[m]["reject"] and a[m]["diff"] > 0.0 for m in holm) and keeps_auc(a["probe_auc"])
    a_kill = within(a["zs"]) and within(a["gain"]) and within(a["probe_auc"])
    b = {m: c("both_multi", "both_util", m) for m in ("gain_long", "gain_short", "probe_auc")}
    comparable = up(b["gain_short"]) and b["gain_short"]["diff"] >= 0.5 * b["gain_long"]["diff"]
    b_sup = up(b["gain_long"]) and keeps_auc(b["probe_auc"]) and not comparable
    b_kill = b["gain_long"]["diff"] <= b["gain_long"]["pooled_se"]  # the single trace matches or beats it

    def verdict(sup, kill):
        if not manipulation:
            return "no_conclusion_manipulation_check_failed"
        return "supported" if sup else "killed" if kill else "inconclusive"

    return {"manipulation_check": {"pass": manipulation, "util_vs_noise_util_zs": m_zs, "util_vs_noise_util_gain": m_gain,
                                   "util_vs_protect_auc": m_auc},
            "h2a": {"verdict": verdict(a_sup, a_kill), "supported": a_sup, "killed": a_kill, "contrasts": a, "holm": holm},
            "h2b": {"verdict": verdict(b_sup, b_kill), "supported": b_sup, "killed": b_kill,
                    "comparable_short_gap_advantage": comparable, "contrasts": b}}


def stage0_checks(t):
    """t[cond][metric] holds per-seed arrays. A NaN anywhere fails its check."""
    sgd, oracle, protect, util = t["sgd"], t["oracle"], t["protect"], t["both_util"]
    checks = {}

    def check(name, item, statement, per_seed, aggregate, op, threshold):
        per_seed = np.asarray(per_seed, dtype=np.float64)
        value = float(per_seed.mean() if aggregate == "mean" else per_seed.max())
        checks[name] = {"item": item, "statement": statement, "value": value, "aggregate": aggregate, "op": op,
                        "threshold": threshold, "per_seed": per_seed.tolist(),
                        "pass": bool(value >= threshold if op == ">=" else value < threshold)}

    check("retention_short", "retention", "oracle minus sgd zero-shot, short gap", oracle["zs_short"] - sgd["zs_short"], "mean", ">=", 0.5)
    check("retention_long", "retention", "oracle minus sgd zero-shot, long gap", oracle["zs_long"] - sgd["zs_long"], "mean", ">=", 0.5)
    check("forgetting", "forgetting", "sgd accuracy at source end minus long-gap zero-shot", sgd["forget_long"], "mean", ">=", 0.1)
    check("plasticity_below_1", "plasticity", "sgd last-quarter probe AUC, every seed", sgd["auc_last"], "max", "<", 1.0)
    check("plasticity_below_first", "plasticity", "sgd last minus first quarter probe AUC, every seed",
          sgd["auc_last"] - sgd["auc_first"], "max", "<", 0.0)
    check("tension", "tension", "protect minus sgd last-quarter probe AUC, every seed", protect["auc_last"] - sgd["auc_last"], "max", "<", 0.0)
    check("reuse_measurable", "reuse", "fraction of sgd reuse events censored at target 0.75", sgd["reuse_censored_frac"], "mean", "<", 0.2)
    check("gate_sd", "gate", "both_util gate SD", util["gate_std"], "mean", ">=", 0.05)
    check("gate_stability", "gate", "both_util Spearman between consecutive refreshes", util["gate_spearman"], "mean", ">=", 0.5)
    return checks


def retry_refusal(previous):
    """Why the single retry is not allowed, or None if it is."""
    if previous is None:
        return "no completed Stage 0 to retry"
    if previous.get("retry"):
        return "the one permitted retry has already been used"
    if previous.get("overall_pass"):
        return "Stage 0 passed, so there is nothing to retry"
    failed = {c["item"] for c in previous["checks"].values() if not c["pass"]}
    if not failed <= RETRY_ITEMS:
        return f"a retry needs every failed check to be plasticity or tension, but these failed: {sorted(failed)}"
    return None


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    return None if isinstance(x, float) and not math.isfinite(x) else x


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=2, allow_nan=False))
    os.replace(tmp, path)


def require_stage0_pass(out_dir):
    record = read_json(out_dir / "stage0.json")
    if record is None or record.get("overall_pass") is not True:
        raise Refused("the full run needs results/h2r/stage0.json with an overall pass")
    return record


def project_minutes(ms, workers, plan=FULL_PLAN):
    """Full-run minutes left after Stage 0, from measured ms per step of sgd, both_util and both_perm."""
    sgd, util, perm = ms["sgd"], ms["both_util"], ms["both_perm"]
    multi = sgd + 4.0 * max(util - sgd, 0.0)  # 3 trace ranks and a re-rank, against both_util's 1
    tune_steps = plan.tune_tasks * plan.steps * plan.n_configs * len(plan.tune_seeds)
    eval_steps = run_steps(plan) * len(plan.eval_seeds)
    core_ms = (tune_steps * (util + perm + multi + sgd)  # noise_util, both_perm, both_multi, shrink_perturb
               + eval_steps * (2.0 * sgd + 3.0 * util + perm + multi)
               + eval_steps * (util + perm))  # the sensitivity block, if it runs
    return core_ms / 1000.0 / 60.0 / workers


def run_steps(plan):
    return plan.tasks * plan.steps + (plan.tasks // PROBE_EVERY) * 2 * PROBE_STEPS


def run_stage0(plan, out_dir, workers, cache, retry=False):
    started, path = time.time(), out_dir / "stage0.json"
    previous = read_json(path)
    if retry:
        reason = retry_refusal(previous)
        if reason:
            raise Refused(reason)
    elif previous is not None and previous.get("retry"):
        raise Refused("a retry is on record; rerunning Stage 0 would overwrite it")
    X, y = load_mnist(cache)
    sha = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
    del X, y
    multiplier = RETRY_LR_MULTIPLE if retry else 1.0
    store, phase_of = RawStore(out_dir / "raw_results.csv"), {c: "stage0_retry" if retry else "stage0" for c in STAGE0_CONDS}
    pool = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(cache), plan.per_class))
    try:
        best = tune(plan, store, pool, STAGE0_TUNED)
        index = {"sgd": best["sgd"], "oracle": best["sgd"], "protect": best["protect"], "both_util": best["both_util"],
                 "both_perm": best["both_util"]}
        configs = {c: cfg_for(c, index[c], multiplier) for c in STAGE0_CONDS}
        run_jobs([(phase_of[c], c, index[c], configs[c], s, plan.tasks, plan.steps, True)
                  for c in STAGE0_CONDS for s in plan.stage0_seeds], store, pool, "stage0")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    tab = {c: table(store, phase_of[c], c, index[c], plan.stage0_seeds, plan.tasks) for c in STAGE0_CONDS}
    checks = stage0_checks(tab)
    overall = all(c["pass"] for c in checks.values())
    ms = {c: 1000.0 * float(np.mean([store.get(phase_of[c], c, index[c], s, plan.tasks)["seconds"] for s in plan.stage0_seeds]))
          / run_steps(plan) for c in STAGE0_CONDS}
    record = {"overall_pass": overall, "retry": retry, "lr_multiplier": multiplier, "plan": asdict(plan),
              "mnist_sha256": sha, "notes": NOTES, "checks": checks,
              "chosen": {c: {"config_index": index[c], "config": configs[c]} for c in STAGE0_CONDS},
              "diagnostics": {"per_seed": {c: {m: tab[c][m] for m in METRICS} for c in STAGE0_CONDS},
                              "ms_per_step": ms, "projected_full_minutes_on_2_workers": project_minutes(ms, 2)},
              "elapsed_seconds": time.time() - started}
    if retry:
        record["first_attempt"] = previous
        if not overall:
            record["closed"] = "H2 line closed: no stream of this form in budget"
    write_json(path, record)
    log.info("stage0 overall_pass=%s failed=%s", overall, [n for n, c in checks.items() if not c["pass"]])
    return record


def run_full(plan, out_dir, workers, cache):
    started = time.time()
    record = require_stage0_pass(out_dir)
    multiplier = float(record.get("lr_multiplier", 1.0))
    X, y = load_mnist(cache)
    sha = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
    del X, y
    store = RawStore(out_dir / "raw_results.csv")
    pool = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(cache), plan.per_class))

    def jobs(phase, conds, index):
        return [(phase, c, index[c], cfg_for(c, index[c], multiplier), s, plan.tasks, plan.steps, True)
                for c in conds for s in plan.eval_seeds]

    pair = ("both_util", "both_perm")
    try:
        best = tune(plan, store, pool, FULL_CONDS)
        # both_perm runs at both_util's config, which keeps the contrast about the gate alone.
        index_of = {**best, "both_perm": best["both_util"]}
        own = best["both_perm"]
        run_jobs(jobs("eval", FULL_CONDS, index_of), store, pool, "eval")
        if own != best["both_util"]:
            run_jobs(jobs("sens", pair, {c: own for c in pair}), store, pool, "sens")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    tab = {c: table(store, "eval", c, index_of[c], plan.eval_seeds, plan.tasks) for c in FULL_CONDS}
    n = len(plan.eval_seeds)
    res = np.random.default_rng(BOOT_SEED).integers(0, n, size=(plan.n_bootstrap, n))
    analysis, sensitivity = evaluate(tab, res), None
    if own != best["both_util"]:
        t2 = {c: table(store, "sens", c, own, plan.eval_seeds, plan.tasks) for c in pair}
        sensitivity = {m: contrast(t2["both_util"][m], t2["both_perm"][m], res) for m in ("zs", "gain", "probe_auc")}
    means = {c: {m: {"mean": float(t[m].mean()), "per_seed": t[m], "ci": np.percentile(t[m][res].mean(axis=1), [2.5, 97.5])}
                 for m in t} for c, t in tab.items()}
    summary = {"plan": asdict(plan), "mnist_sha256": sha, "notes": NOTES, "lr_multiplier": multiplier,
               "chosen": {c: {"config_index": index_of[c], "config": cfg_for(c, index_of[c], multiplier)} for c in FULL_CONDS},
               "both_perm_own_best_index": own, **analysis, "h2a_sensitivity_at_perm_config": sensitivity,
               "per_condition": means, "elapsed_seconds": time.time() - started}
    write_json(out_dir / "summary.json", summary)
    log.info("manipulation=%s H2a=%s H2b=%s", analysis["manipulation_check"]["pass"], analysis["h2a"]["verdict"], analysis["h2b"]["verdict"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage0", action="store_true")
    parser.add_argument("--retry", action="store_true", help="with --stage0: the one retry, all rates at 3x")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/clu/mnist_canonical.npz")
    args = parser.parse_args()
    if args.retry and not args.stage0:
        parser.error("--retry needs --stage0")
    out_dir = args.out or Path(__file__).resolve().parents[1] / "results" / "h2r"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.seterr(all="ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler()])
    try:
        if args.stage0:
            run_stage0(FULL_PLAN, out_dir, args.workers, args.cache, args.retry)
        else:
            run_full(FULL_PLAN, out_dir, args.workers, args.cache)
    except Refused as exc:
        log.error("refused: %s", exc)
        return 2
    except Exception:
        log.exception("run failed; rerun the same command to resume")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
