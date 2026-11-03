#!/usr/bin/env python3
"""H2 experiment; see hypotheses/H2_shared_synaptic_state.md. Numpy, hand-written backprop.

nohup python experiments/h2_shared_state.py > h2.stdout 2>&1 &   # --smoke: plumbing only
Writes run.log, raw_results.csv (resume checkpoint), summary.json (atomic). Rerun to resume;
pass --fresh after a code change. Open decisions are in NOTES.
"""
from __future__ import annotations

import argparse, csv, hashlib, io, json, logging, math, os, sys, time, urllib.request, zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

log = logging.getLogger("h2")
NOTES = [
    "u = sigmoid(bias-corrected trace / its max), as in UPGD's code",
    "both_rand, noise_rand are yoked per step to the same-seed, same-config util run's density"
    " and actual energy; probe and fresh-init copies have no partner and match their own gate",
    "both_rand tunes over both_util's candidates; primary contrast is at both_util's choice,"
    " since another config breaks the match; the sensitivity block uses its own choice",
    "fixed subset is split into disjoint tuning and evaluation image pools",
    "40-task tuning streams cannot hold a gap of 40, so their repeats are all short-gap",
    "normals and uniforms are random windows into a pool refreshed every 64 draws, for speed",
    "probe copy keeps utility traces, resets Adam; savings = 1 - main/fresh steps, on 100-sample"
    " rolling online accuracy, censored at task length",
    "pooled SE = sqrt((var_a+var_b)/(2n)); verdict retention averages short and long gaps",
    "comparable short-gap advantage: CI excludes 0 and diff >= half the long-gap diff",
]
STEPS, PROBE_STEPS, PROBE_EVERY, WINDOW, LONG_GAP, FIRST_EVAL_SEED = 1000, 300, 10, 100, 40, 100
ROOT_SEED, CONFIG_SEED, BOOT_SEED, SUBSET_SEED = 20261105, 20261106, 20261107, 20261108
MULTI_TAUS = (100.0, 1000.0, 10000.0)  # both_util's tau is tuned over this same range
SHAPES = ((128, 784), (128,), (128, 128), (128,), (10, 128), (10,))
N_PARAMS = sum(int(np.prod(s)) for s in SHAPES)
K_STREAM, K_INIT, K_NOISE, K_PROBE, K_P_NOISE, K_P_INIT, K_P_FRESH, K_F_INIT, K_F_NOISE = range(9)
COND = {"sgd": (0, None), "protect": (1, None), "noise_rand": (0, "rand"), "noise_util": (0, "util"),
        "both_rand": (1, "rand"), "both_util": (1, "util"), "both_multi": (1, "util"),
        "shrink_perturb": (0, "plain"), "adam": (0, None)}  # condition: (protect, noise)
CONDS = tuple(COND)
# The first member of a group draws the configs; a second member is yoked to it.
GROUPS = (("sgd",), ("protect",), ("noise_util", "noise_rand"), ("both_util", "both_rand"),
          ("both_multi",), ("shrink_perturb",), ("adam",))
METRICS = ("online_acc", "zs_short", "zs_long", "sav_short", "sav_long", "probe_auc", "gate_std",
           "energy_err", "density_z")


def key(seed, *tags):
    return np.random.default_rng([ROOT_SEED, seed, *tags])


def layer_views(w):
    out, i = [], 0
    for shape in SHAPES:
        n = int(np.prod(shape))
        out.append(w[i:i + n].reshape(shape))
        i += n
    return out


def init_weights(rng, dtype=np.float32):
    w = np.zeros(N_PARAMS, dtype=dtype)
    for view in layer_views(w):
        if view.ndim == 2:  # He normal; biases stay 0
            view[:] = rng.standard_normal(view.shape, dtype=dtype) * math.sqrt(2.0 / view.shape[1])
    return w


def logits(v, x):
    h1 = np.maximum(x @ v[0].T + v[1], 0.0)
    h2 = np.maximum(h1 @ v[2].T + v[3], 0.0)
    return h1, h2, h2 @ v[4].T + v[5]


def grad_step(v, gv, x, y):
    """Write the cross-entropy gradient into gv; return the prediction made before any update."""
    h1, h2, z = logits(v, x)
    pred = int(z.argmax())
    p = np.exp(z - z.max())
    p /= p.sum()
    p[y] -= 1.0
    gv[5][:] = p
    np.multiply.outer(p, h2, out=gv[4])
    d2 = (p @ v[4]) * (h2 > 0)
    gv[3][:] = d2
    np.multiply.outer(d2, h1, out=gv[2])
    d1 = (d2 @ v[2]) * (h1 > 0)
    gv[1][:] = d1
    np.multiply.outer(d1, x, out=gv[0])
    return pred


def normalise(trace):
    eta = float(trace.max())
    if eta <= 0.0:
        eta = float(np.abs(trace).max()) or 1.0
    return 1.0 / (1.0 + np.exp(-trace / eta))


def energy(a):
    v = a.astype(np.float64)
    return float(np.dot(v, v))


class NoisePool:
    """Any one weight still sees independent values across steps. Windows are read-only views."""

    SIZE, REUSE = 2**20, 64

    def __init__(self, rng):
        self.rng, self.pools, self.counts = rng, {}, {"n": 0, "u": 0}

    def _draw(self, kind):
        if self.counts[kind] % self.REUSE == 0:
            self.pools[kind] = (self.rng.standard_normal(self.SIZE, dtype=np.float32) if kind == "n"
                                else self.rng.random(self.SIZE, dtype=np.float32))
        self.counts[kind] += 1
        i = int(self.rng.integers(self.SIZE - N_PARAMS + 1))
        return self.pools[kind][i:i + N_PARAMS]

    def normal(self):
        return self._draw("n")

    def uniform(self):
        return self._draw("u")


class Learner:
    def __init__(self, cond, cfg, rng, w):
        self.cond, self.cfg, self.w, self.views = cond, cfg, w, layer_views(w)
        self.g = np.zeros_like(w)
        self.gv = layer_views(self.g)
        self.lr, self.sigma = cfg["lr"], cfg.get("sigma", 0.0)
        self.protect, self.noise = COND[cond]
        taus = MULTI_TAUS if cond == "both_multi" else (cfg["tau"],) if "tau" in cfg else ()
        self.betas = [1.0 - 1.0 / tau for tau in taus]
        self.traces = [np.zeros_like(w) for _ in taus]
        self.pool = NoisePool(rng) if self.noise else None
        self.t, self.u, self.force_u, self.noise_term, self.density, self.energy = 0, None, None, None, 0.0, 0.0
        if cond == "adam":
            self.m, self.v = np.zeros_like(w), np.zeros_like(w)

    def clone(self, rng):
        """Independent probe copy: keeps traces, resets Adam, shares no arrays or RNG."""
        copy = Learner(self.cond, self.cfg, rng, self.w.copy())
        if self.traces:
            copy.traces, copy.t = [t.copy() for t in self.traces], self.t
        return copy

    def _utility(self):
        gw, us = -self.g * self.w, []
        for trace, beta in zip(self.traces, self.betas):
            trace += (1.0 - beta) * (gw - trace)
            us.append(normalise(trace / (1.0 - beta**self.t)))
        if self.force_u is not None:
            return self.force_u
        return us[0] if len(us) == 1 else np.maximum.reduce(us)

    def _noise_term(self, yoke):
        if self.noise == "plain":
            return self.sigma * self.pool.normal()
        gate = None if yoke is not None else 1.0 - self.u
        if yoke is None:
            term = self.sigma * self.pool.normal() * gate
            self.density, self.energy = float(gate.mean()), energy(term)
            if self.noise == "util":
                return term
        else:
            self.density, self.energy = float(yoke[0]), float(yoke[1])
        n = self.pool.normal() * (self.pool.uniform() < self.density)
        e = energy(n)
        return math.sqrt(self.energy / e) * n if e > 0.0 else n

    def step(self, x, y, yoke=None):
        pred = grad_step(self.views, self.gv, x, y)
        self.t += 1
        w, g, lr = self.w, self.g, self.lr
        if self.cond == "adam":
            b1 = self.cfg["beta1"]
            self.m += (1.0 - b1) * (g - self.m)
            self.v += 0.001 * (g * g - self.v)
            w -= lr * (self.m / (1.0 - b1**self.t)) / (np.sqrt(self.v / (1.0 - 0.999**self.t)) + 1e-8)
            return pred
        if self.traces:
            self.u = self._utility()
        if self.noise == "plain":
            w *= 1.0 - lr * self.cfg["decay"]
        update = g * (1.0 - self.u) if self.protect else g  # w -= lr * (g * P + n * R)
        if self.noise:
            self.noise_term = self._noise_term(yoke)
            update = update + self.noise_term
        w -= lr * update
        return pred


@dataclass
class Data:
    X: np.ndarray
    y: np.ndarray
    tune_pool: np.ndarray
    eval_pool: np.ndarray

    def pool(self, seed):
        return self.tune_pool if seed < FIRST_EVAL_SEED else self.eval_pool


def make_data(X, y, per_class):
    rng, tune, evals = np.random.default_rng(SUBSET_SEED), [], []
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(y == c))
        tune.append(idx[:per_class])
        evals.append(idx[per_class:2 * per_class])
    keep, n = np.concatenate(tune + evals), 10 * per_class
    return Data(X[keep].astype(np.float32) / 255.0, y[keep].astype(np.int64), np.arange(n), np.arange(n, 2 * n))


def schedule_repeats(rng, n_tasks):
    """Map return task to (source, gap, kind). A permutation returns at most once."""
    n_rep = round(0.1 * n_tasks)
    n_long = n_rep // 2 if n_tasks > LONG_GAP else 0
    repeats, sources = {}, set()
    for kind in ["long"] * n_long + ["short"] * (n_rep - n_long):
        for _ in range(100000):
            if kind == "long":
                r = int(rng.integers(LONG_GAP, n_tasks))
                gap = int(rng.integers(LONG_GAP, r + 1))
            else:
                gap = int(rng.integers(5, 11))
                r = int(rng.integers(gap, n_tasks))
            s = r - gap
            if not {r, s} & (set(repeats) | sources):
                repeats[r] = (s, gap, kind)
                sources.add(s)
                break
        else:
            raise RuntimeError(f"cannot place repeats in {n_tasks} tasks")
    return repeats


def _new_perm(rng, used):
    perm = rng.permutation(10)
    while tuple(perm.tolist()) in used:
        perm = rng.permutation(10)
    return perm


def make_stream(seed, n_tasks, steps, pool):
    rng = key(seed, K_STREAM)
    repeats = schedule_repeats(rng, n_tasks)
    perms, seen = np.zeros((n_tasks, 10), dtype=np.int64), set()
    for t in range(n_tasks):
        perms[t] = perms[repeats[t][0]] if t in repeats else _new_perm(rng, seen)
        seen.add(tuple(perms[t].tolist()))
    idx = np.stack([rng.choice(pool, steps, replace=False) for _ in range(n_tasks)])
    return SimpleNamespace(perms=perms, idx=idx, repeats=repeats)


def probe_task(seed, j, pool, used):
    rng = key(seed, K_PROBE, j)
    perm = _new_perm(rng, used)
    return perm, rng.choice(pool, PROBE_STEPS, replace=False)


def zero_shot(learner, X, y):
    return float((logits(learner.views, X)[2].argmax(axis=1) == y).mean())


def rolling(correct):
    return np.convolve(correct.astype(np.float64), np.ones(WINDOW), "valid") / WINDOW


def online(learner, xs, labels):
    return float(np.mean([learner.step(x, y) == y for x, y in zip(xs, labels)]))


def probe_ratio(learner, cfg, seed, cid, j, data, used):
    perm, idx = probe_task(seed, j, data.pool(seed), used)
    xs, labels = data.X[idx], perm[data.y[idx]]
    fresh = Learner(learner.cond, cfg, key(seed, K_P_FRESH, cid, j), init_weights(key(seed, K_P_INIT, j)))
    return online(learner.clone(key(seed, K_P_NOISE, cid, j)), xs, labels) / online(fresh, xs, labels)


def fresh_steps(cond, cfg, seed, cid, task, xs, labels, target):
    learner = Learner(cond, cfg, key(seed, K_F_NOISE, cid, task), init_weights(key(seed, K_F_INIT, task)))
    correct = np.zeros(len(xs), dtype=bool)
    for i in range(len(xs)):
        correct[i] = learner.step(xs[i], labels[i]) == labels[i]
        if i >= WINDOW - 1 and correct[i - WINDOW + 1:i + 1].sum() / WINDOW >= target:
            return i + 1
    return len(xs)


def _mean(values):
    return float(np.mean(values)) if len(values) else math.nan


def run_stream(cond, cfg, seed, data, n_tasks, steps=STEPS, yoke=None, metrics=True, record=False):
    stream = make_stream(seed, n_tasks, steps, data.pool(seed))
    used = {tuple(p.tolist()) for p in stream.perms}
    cid = CONDS.index(cond)
    L = Learner(cond, cfg, key(seed, K_NOISE, cid), init_weights(key(seed, K_INIT)))
    out = np.zeros((n_tasks * steps, 2)) if record else None
    correct = np.zeros(steps, dtype=bool)
    hits, err, dz, best, aucs, gs, events = 0, 0.0, 0.0, {}, [], [], []
    zs, sav = {"short": [], "long": []}, {"short": [], "long": []}
    for task in range(n_tasks):
        idx = stream.idx[task]
        xs, labels = data.X[idx], stream.perms[task][data.y[idx]]
        rep = stream.repeats.get(task)
        # From the weights the previous task left, before any step of this one.
        zero = zero_shot(L, xs[:WINDOW], labels[:WINDOW]) if rep and metrics else None
        for i in range(steps):
            t = L.t
            correct[i] = L.step(xs[i], labels[i], None if yoke is None else yoke[t]) == labels[i]
            if record:
                out[t] = (L.density, L.energy)
            if L.noise == "rand" and L.t % 100 == 0 and L.energy > 0.0:
                err = max(err, abs(energy(L.noise_term) - L.energy) / L.energy)
                d = L.density
                got = np.count_nonzero(L.noise_term) / N_PARAMS
                dz = max(dz, abs(got - d) / math.sqrt(max(d * (1.0 - d), 1e-12) / N_PARAMS))
        hits += int(correct.sum())
        if not metrics:
            continue
        if L.u is not None:
            gs.append(float(L.u.std()))
        roll = rolling(correct)
        if rep:
            source, gap, kind = rep
            target = 0.9 * best[source]
            reached = np.flatnonzero(roll >= target)
            main = int(reached[0]) + WINDOW if reached.size else steps
            fresh = fresh_steps(cond, cfg, seed, cid, task, xs, labels, target)
            zs[kind].append(zero)
            sav[kind].append(1.0 - main / fresh)
            events.append([task, gap, zero, main, fresh])
        best[task] = float(roll.max())
        if (task + 1) % PROBE_EVERY == 0:
            aucs.append(probe_ratio(L, cfg, seed, cid, task, data, used))
    result = {"online_acc": hits / (n_tasks * steps), "zs_short": _mean(zs["short"]), "zs_long": _mean(zs["long"]),
              "sav_short": _mean(sav["short"]), "sav_long": _mean(sav["long"]), "probe_auc": _mean(aucs),
              "gate_std": _mean(gs), "energy_err": err, "density_z": dz,
              "detail": {"events": events, "aucs": aucs, "w_crc": zlib.crc32(L.w.tobytes())}}
    if record:
        result["yoke"] = out
    return result


def _fetch_keras():
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    with urllib.request.urlopen(url, timeout=120) as response:
        blob = response.read()
    with np.load(io.BytesIO(blob)) as d:
        return np.concatenate([d["x_train"], d["x_test"]]).reshape(-1, 784), np.concatenate([d["y_train"], d["y_test"]])


def _fetch_openml():
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml("mnist_784", version=1, as_frame=False)
    return np.asarray(bunch.data), np.asarray(bunch.target)


FETCHERS = (_fetch_keras, _fetch_openml)


def load_mnist(cache):
    if cache.exists():
        with np.load(cache) as d:
            return d["X"], d["y"]
    errors = []
    for fetch in FETCHERS:
        try:
            X, y = fetch()
            X, y = np.asarray(X, dtype=np.float64).round().astype(np.uint8), np.asarray(y).astype(np.int64)
            counts = np.bincount(y, minlength=10)
            if X.shape != (70000, 784) or len(counts) != 10 or counts.min() < 6000 or not 30 < X.mean() < 37:
                raise ValueError(f"not MNIST: shape {X.shape}, classes {counts}, mean {X.mean()}")
            # Same rows and order whichever route supplied them.
            order = np.argsort([zlib.crc32(row.tobytes()) for row in X], kind="stable")
            X, y = X[order], y[order]
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp")
            with tmp.open("wb") as handle:
                np.savez_compressed(handle, X=X, y=y)
            os.replace(tmp, cache)
            return X, y
        except Exception as exc:  # every route is tried, then all are reported
            errors.append(f"{fetch.__name__}: {exc!r}")
    raise RuntimeError("MNIST unavailable; refusing other data. " + "; ".join(errors))


@dataclass(frozen=True)
class Plan:
    tasks: int
    tune_tasks: int
    steps: int
    n_configs: int
    tune_seeds: tuple
    eval_seeds: tuple
    n_bootstrap: int
    per_class: int = 600


FULL_PLAN = Plan(100, 40, STEPS, 10, (0, 1), tuple(range(100, 110)), 10000)
SMOKE_PLAN = Plan(45, 12, 110, 2, (0, 1), (100, 101), 500)


def group_config(group, index):
    rng, cond = np.random.default_rng([CONFIG_SEED, GROUPS.index(group), index]), group[0]
    lu = lambda low, high: float(math.exp(rng.uniform(math.log(low), math.log(high))))  # noqa: E731
    if cond == "adam":
        return {"lr": lu(1e-4, 1e-2), "beta1": float(rng.uniform(0.0, 0.95))}
    cfg = {"lr": lu(1e-3, 0.1)}
    if cond in ("protect", "noise_util", "both_util"):
        cfg["tau"] = lu(MULTI_TAUS[0], MULTI_TAUS[-1])
    if cond in ("noise_util", "both_util", "both_multi", "shrink_perturb"):
        cfg["sigma"] = lu(1e-3, 1.0)
    if cond == "shrink_perturb":
        cfg["decay"] = lu(1e-5, 1e-2)
    return cfg


class RawStore:
    FIELDS = ("phase", "condition", "config_index", "seed", "n_tasks", *METRICS, "seconds", "config", "detail")

    def __init__(self, path, fresh):
        rows = []
        if path.exists() and not fresh:
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
            [float(row[m]) for m in METRICS]
            return None not in row
        except (ValueError, TypeError, KeyError):
            return False

    def _remember(self, rows):
        for r in rows:
            self.values[tuple(str(r[f]) for f in self.FIELDS[:5])] = {m: float(r[m]) for m in METRICS}

    def get(self, *k):
        return self.values.get(tuple(str(x) for x in k))

    def add(self, rows):
        self._writer.writerows(rows)
        self._handle.flush()
        self._remember(rows)


_DATA = None


def _init_worker(cache, per_class):
    global _DATA
    np.seterr(all="ignore")
    _DATA = make_data(*load_mnist(Path(cache)), per_class)


def run_job(job):
    phase, conds, index, cfg, seed, n_tasks, steps, metrics = job
    rows, yoke = [], None
    for cond in conds:
        started = time.time()
        r = run_stream(cond, cfg, seed, _DATA, n_tasks, steps, yoke, metrics, record=len(conds) == 2 and yoke is None)
        yoke = r.pop("yoke", None)
        rows.append({"phase": phase, "condition": cond, "config_index": index, "seed": seed, "n_tasks": n_tasks,
                     **{m: repr(float(r[m])) for m in METRICS}, "seconds": round(time.time() - started, 1),
                     "config": json.dumps(cfg), "detail": json.dumps(r["detail"])})
    return rows


def run_jobs(jobs, store, pool, label):
    todo = [j for j in jobs if not all(store.get(j[0], c, j[2], j[4], j[5]) for c in j[1])]
    log.info("%s: %d jobs, %d to run", label, len(jobs), len(todo))
    started = time.time()
    for n, future in enumerate(as_completed([pool.submit(run_job, j) for j in todo]), 1):
        rows = future.result()
        store.add(rows)
        log.info("%s %d/%d %s seed=%s %.0fs", label, n, len(todo), rows[0]["condition"], rows[0]["seed"], time.time() - started)


def contrast(a, b, res):
    d = a - b
    low, high = np.percentile(d[res].mean(axis=1), [2.5, 97.5])
    return {"diff": float(d.mean()), "ci_low": float(low), "ci_high": float(high),
            "pooled_se": math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / (2 * len(a)))}


def evaluate(t, res):
    """t[cond][metric] holds per-seed arrays, paired by position."""
    c = lambda a, b, m: contrast(t[a][m], t[b][m], res)  # noqa: E731
    up = lambda r: r["ci_low"] > 0.0  # noqa: E731
    within = lambda r: abs(r["diff"]) <= r["pooled_se"]  # noqa: E731
    keeps_auc = lambda r: r["diff"] >= -r["pooled_se"]  # noqa: E731
    m_zs, m_sav = c("both_util", "noise_util", "zs"), c("both_util", "noise_util", "sav")
    m_auc = c("both_util", "protect", "probe_auc")
    manipulation = (up(m_zs) or up(m_sav)) and up(m_auc)
    a = {m: c("both_util", "both_rand", m) for m in ("zs", "sav", "probe_auc", "zs_short", "zs_long", "sav_short", "sav_long")}
    a_sup = (up(a["zs"]) or up(a["sav"])) and keeps_auc(a["probe_auc"])
    a_kill = within(a["zs"]) and within(a["sav"]) and within(a["probe_auc"])
    b = {m: c("both_multi", "both_util", m) for m in ("sav_long", "sav_short", "probe_auc")}
    comparable = up(b["sav_short"]) and b["sav_short"]["diff"] >= 0.5 * b["sav_long"]["diff"]
    b_sup = up(b["sav_long"]) and keeps_auc(b["probe_auc"]) and not comparable
    b_kill = b["sav_long"]["diff"] <= b["sav_long"]["pooled_se"]  # the single trace matches or beats it

    def verdict(sup, kill):
        if not manipulation:
            return "no_conclusion_manipulation_check_failed"
        return "supported" if sup else "killed" if kill else "inconclusive"

    return {"manipulation_check": {"pass": manipulation, "util_vs_noise_util_zs": m_zs, "util_vs_noise_util_sav": m_sav,
                                   "util_vs_protect_auc": m_auc},
            "h2a": {"verdict": verdict(a_sup, a_kill), "supported": a_sup, "killed": a_kill, "contrasts": a},
            "h2b": {"verdict": verdict(b_sup, b_kill), "supported": b_sup, "killed": b_kill,
                    "comparable_short_gap_advantage": comparable, "contrasts": b}}


def table(store, phase, cond, index, seeds, n_tasks):
    rows = [store.get(phase, cond, index, s, n_tasks) for s in seeds]
    out = {m: np.array([r[m] for r in rows]) for m in METRICS}
    out["zs"] = (out["zs_short"] + out["zs_long"]) / 2
    out["sav"] = (out["sav_short"] + out["sav_long"]) / 2
    return out


def experiment(plan, out_dir, fresh, smoke, workers, cache):
    started = time.time()
    X, y = load_mnist(cache)
    sha = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
    del X, y
    store = RawStore(out_dir / "raw_results.csv", fresh)
    pool = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(cache), plan.per_class))

    def jobs(phase, g, i, seeds, n, metrics):
        return [(phase, g, i, group_config(g, i), s, n, plan.steps, metrics) for s in seeds]

    pair = ("both_util", "both_rand")
    try:
        run_jobs([j for g in GROUPS for i in range(plan.n_configs) for j in jobs("tune", g, i, plan.tune_seeds, plan.tune_tasks, False)],
                 store, pool, "tune")
        best = {c: int(np.argmax([np.mean([store.get("tune", c, i, s, plan.tune_tasks)["online_acc"] for s in plan.tune_seeds])
                                  for i in range(plan.n_configs)])) for c in CONDS}
        log.info("tuned: %s", best)
        # A yoked control runs at its partner's config, which keeps the match exact.
        index_of = {c: best[g[0]] for g in GROUPS for c in g}
        run_jobs([j for g in GROUPS for j in jobs("eval", g, best[g[0]], plan.eval_seeds, plan.tasks, True)], store, pool, "eval")
        own = best["both_rand"]
        if own != best["both_util"]:
            run_jobs(jobs("sens", pair, own, plan.eval_seeds, plan.tasks, True), store, pool, "sens")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    tab = {c: table(store, "eval", c, index_of[c], plan.eval_seeds, plan.tasks) for c in CONDS}
    n = len(plan.eval_seeds)
    res = np.random.default_rng(BOOT_SEED).integers(0, n, size=(plan.n_bootstrap, n))
    analysis, sensitivity, rand_tabs = evaluate(tab, res), None, [tab["both_rand"]]
    if own != best["both_util"]:
        t2 = {c: table(store, "sens", c, own, plan.eval_seeds, plan.tasks) for c in pair}
        rand_tabs.append(t2["both_rand"])
        sensitivity = {m: contrast(t2["both_util"][m], t2["both_rand"][m], res) for m in ("zs", "sav", "probe_auc")}
    max_err = max(float(t["energy_err"].max()) for t in rand_tabs)
    max_z = max(float(t["density_z"].max()) for t in rand_tabs)
    means = {c: {m: {"mean": float(t[m].mean()), "ci": np.percentile(t[m][res].mean(axis=1), [2.5, 97.5]).tolist()}
                 for m in t if m not in ("energy_err", "density_z")} for c, t in tab.items()}
    summary = {"smoke": smoke, "plan": asdict(plan), "mnist_sha256": sha, "notes": NOTES,
               "chosen": {c: {"config_index": index_of[c], "config": group_config(g, index_of[c])} for g in GROUPS for c in g},
               "both_rand_own_best_index": own,
               "control_valid": {"pass": max_err < 1e-4 and max_z < 6.0, "max_energy_rel_err": max_err, "max_density_z": max_z},
               **analysis, "h2a_sensitivity_at_rand_config": sensitivity, "per_condition": means,
               "elapsed_seconds": time.time() - started}
    tmp = out_dir / "summary.tmp"
    tmp.write_text(json.dumps(summary, indent=2))
    os.replace(tmp, out_dir / "summary.json")
    log.info("control_valid=%s manipulation=%s H2a=%s H2b=%s", summary["control_valid"]["pass"],
             analysis["manipulation_check"]["pass"], analysis["h2a"]["verdict"], analysis["h2b"]["verdict"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/clu/mnist_canonical.npz")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out_dir = args.out or repo / "results" / ("h2_smoke" if args.smoke else "h2_shared_state")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.seterr(all="ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler()])
    try:
        experiment(SMOKE_PLAN if args.smoke else FULL_PLAN, out_dir, args.fresh, args.smoke, args.workers, args.cache)
    except Exception:
        log.exception("run failed; rerun the same command to resume")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
