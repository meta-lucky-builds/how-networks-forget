"""H5: is catastrophic forgetting mostly overwriting or loss of access?

Spec: hypotheses/H5_overwrite_or_access.md. Calibration never scores the trained task on a later
network with any key but `none`, so constants cannot be chosen to favour an answer.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h4_schema import load_datasets  # noqa: E402

log = logging.getLogger("h5")

# ---- Fixed by the spec ----
REGIMES = ("perm_shared", "perm_class", "split_class")
LAGS = {"perm_shared": (1, 2, 5, 10, 20), "perm_class": (1, 2, 5, 10, 20), "split_class": (1, 2, 3)}
GRID = tuple((lr, steps) for lr in (0.01, 0.03, 0.1, 0.3) for steps in (500, 2000))
BUDGETS = (5, 10, 20, 50)
CAL_SEEDS, RETRY_SEEDS, CONFIRM_SEEDS = tuple(range(500, 505)), tuple(range(505, 510)), tuple(range(600, 620))
MIN_ACC, MAX_S_NONE, MIN_ADVANTAGE = 0.85, 0.5, 0.15
NULL_LIMIT, PLANT_MIN_S, PLANT_MAX_NONE, MATCHED_GAP = 0.15, 0.8, 0.6, 0.2
N_BOOTSTRAP = 10000

# ---- Left open by the spec; chosen from first principles, not from outcomes ----
HIDDEN = 128
BATCH = 64
EVAL_PER_CLASS = 200  # SE about 0.01 per seed at 2000 images
KEY_POOL_PER_CLASS = 500
RIDGE_LAMBDA = 1.0  # On standardised features: a light prior, near the ordinary least squares fit
FINETUNE_STEPS = 100  # Short, so a repair counts as repair rather than relearning from nothing
GAIN_STEPS, GAIN_LR = 100, 0.05
BISECT_STEPS = 7  # Fraction to within 1/128, finer than the noise in S
ROOT_SEED = 20261104
K_INIT, K_TASKS, K_TRAIN, K_KEY, K_PLANT, K_ERASE, K_BOOT = range(7)

NOTES = [
    "keys beyond none and mask restrict output to the probed task, so readout, gain and finetune include the mask",
    "key samples are common random numbers across networks, so S compares networks and not draws",
    "planted networks are built from w_0, with S_none matched to that seed's real S_none at the primary lag",
    "in class regimes real S_none can be near 0, so matched erasure can approach total erasure; reported as such",
    "S is a ratio of means over seeds, and intervals resample seeds",
]


class Data:
    """Images by class, as uint8, in three disjoint pools."""

    def __init__(self, x_train, y_train, x_test, y_test, key_per_class=KEY_POOL_PER_CLASS, eval_per_class=EVAL_PER_CLASS):
        self.x_train, self.x_test = x_train, x_test
        self.train, self.key, self.eval = {}, {}, {}
        for c in np.unique(y_train):
            idx = np.flatnonzero(y_train == c)
            self.key[int(c)], self.train[int(c)] = idx[-key_per_class:], idx[:-key_per_class]
            self.eval[int(c)] = np.flatnonzero(y_test == c)[:eval_per_class]


class Task:
    def __init__(self, classes, outputs, perm=None):
        self.classes, self.outputs, self.perm = np.asarray(classes), np.asarray(outputs), perm

    def view(self, images):
        x = images.astype(np.float32) / 255.0
        return x if self.perm is None else x[:, self.perm]

    def pool(self, images, index, per_class=None, rng=None):
        """Return (x, y), with y the position within this task's outputs."""
        parts, labels = [], []
        for pos, c in enumerate(self.classes):
            ids = index[int(c)]
            if per_class is not None:
                ids = rng.choice(ids, size=per_class, replace=False)
            parts.append(ids)
            labels.append(np.full(len(ids), pos))
        return self.view(images[np.concatenate(parts)]), np.concatenate(labels)


def rng_for(seed, *tags):
    return np.random.default_rng([ROOT_SEED, seed, *tags])


def make_tasks(regime, seed, n_trained):
    """Tasks 0 to n_trained - 1 are trained in order. The last two tasks returned are never trained."""
    rng, total = rng_for(seed, K_TASKS), n_trained + 2
    if regime == "split_class":
        order = rng.permutation(20)
        return [Task(order[3 * i:3 * i + 3], order[3 * i:3 * i + 3]) for i in range(total)]
    perms = [rng.permutation(784) for _ in range(total)]
    if regime == "perm_shared":
        return [Task(np.arange(10), np.arange(10), p) for p in perms]
    return [Task(np.arange(10), np.arange(10) + 10 * i, p) for i, p in enumerate(perms)]


def n_outputs(regime, n_trained):
    return {"perm_shared": 10, "perm_class": 10 * (n_trained + 2), "split_class": 20}[regime]


def seen_outputs(regime, tasks, n_seen):
    if regime == "perm_shared":
        return tasks[0].outputs
    return np.concatenate([t.outputs for t in tasks[:n_seen]])


def init_weights(rng, outputs, inputs=784, hidden=HIDDEN):
    def draw(rows, cols):
        return rng.normal(0.0, np.sqrt(2.0 / cols), size=(rows, cols)).astype(np.float32)
    zeros = lambda size: np.zeros(size, dtype=np.float32)
    return [draw(hidden, inputs), zeros(hidden), draw(hidden, hidden), zeros(hidden), draw(outputs, hidden), zeros(outputs)]


def copy_net(net):
    return [a.copy() for a in net]


def hidden_states(net, x, gains=None):
    g1, g2 = (None, None) if gains is None else gains
    r1 = np.maximum(x @ net[0].T + net[1], 0.0)
    h1 = r1 if g1 is None else r1 * g1
    r2 = np.maximum(h1 @ net[2].T + net[3], 0.0)
    h2 = r2 if g2 is None else r2 * g2
    return r1, h1, r2, h2


def logits(net, x, gains=None):
    return hidden_states(net, x, gains)[3] @ net[4].T + net[5]


def grads(net, x, y_out, active, gains=None):
    """Cross-entropy over `active` outputs. y_out are positions within active.

    Returns (weight grads, (gain1 grad, gain2 grad)).
    """
    g1, g2 = (1.0, 1.0) if gains is None else gains
    r1, h1, r2, h2 = hidden_states(net, x, gains)
    head, bias = net[4][active], net[5][active]
    z = h2 @ head.T + bias
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    e = p.copy()
    e[np.arange(len(y_out)), y_out] -= 1.0
    n = len(y_out)
    dh2 = e @ head  # d loss / d h2, per sample
    dr2_gain = dh2 * g2 * (r2 > 0)
    dh1 = dr2_gain @ net[2]
    dr1 = dh1 * g1 * (r1 > 0)
    grad_head, grad_bias = np.zeros_like(net[4]), np.zeros_like(net[5])
    grad_head[active], grad_bias[active] = e.T @ h2 / n, e.mean(0)
    weight = [dr1.T @ x / n, dr1.mean(0), dr2_gain.T @ h1 / n, dr2_gain.mean(0), grad_head, grad_bias]
    return weight, ((r1 * dh1).mean(0), (r2 * dh2).mean(0))


def score(net, task, x, y, active, gains=None):
    """Accuracy, with argmax over `active`. y are positions within task.outputs."""
    pred = active[np.argmax(logits(net, x, gains)[:, active], axis=1)]
    return float((pred == task.outputs[y]).mean())


def train_task(net, task, data, active, lr, steps, rng):
    lookup = {int(o): i for i, o in enumerate(active)}
    pos = np.array([lookup[int(o)] for o in task.outputs])
    index = [data.train[int(c)] for c in task.classes]
    pool, sizes = np.concatenate(index), np.array([len(i) for i in index])
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    for _ in range(steps):
        labels = rng.integers(len(index), size=BATCH)
        ids = pool[starts[labels] + (rng.random(BATCH) * sizes[labels]).astype(np.int64)]
        weight, _ = grads(net, task.view(data.x_train[ids]), pos[labels], active)
        for a, g in zip(net, weight):
            a -= lr * g


# ---- Keys ----
def ridge_score(net, task, x_key, y_key, x_eval, y_eval):
    f = np.nan_to_num(hidden_states(net, x_key)[3]).astype(np.float64)
    test = np.nan_to_num(hidden_states(net, x_eval)[3]).astype(np.float64)
    mean, std = f.mean(0), f.std(0) + 1e-6
    f, test = (f - mean) / std, (test - mean) / std
    target = np.eye(len(task.classes))[y_key]
    offset = target.mean(0)
    beta = np.linalg.solve(f.T @ f + RIDGE_LAMBDA * np.eye(f.shape[1]), f.T @ (target - offset))
    return float((np.argmax(test @ beta + offset, axis=1) == y_eval).mean())


def gain_score(net, task, x_key, y_key, x_eval, y_eval):
    gains = [np.ones(net[0].shape[0], np.float32), np.ones(net[2].shape[0], np.float32)]
    first, second = [np.zeros_like(g) for g in gains], [np.zeros_like(g) for g in gains]
    for t in range(1, GAIN_STEPS + 1):
        _, grad_gain = grads(net, x_key, y_key, task.outputs, gains)
        for i, g in enumerate(grad_gain):  # Adam, since gains have no natural step size
            first[i] = 0.9 * first[i] + 0.1 * g
            second[i] = 0.999 * second[i] + 0.001 * g * g
            step = (first[i] / (1 - 0.9 ** t)) / (np.sqrt(second[i] / (1 - 0.999 ** t)) + 1e-8)
            gains[i] = np.maximum(gains[i] - GAIN_LR * step, 0.0)
    return score(net, task, x_eval, y_eval, task.outputs, gains)


def finetune_score(net, task, x_key, y_key, x_eval, y_eval, lr, rng):
    tuned, size = copy_net(net), min(BATCH, len(y_key))
    for _ in range(FINETUNE_STEPS):
        ids = rng.choice(len(y_key), size=size, replace=False)
        weight, _ = grads(tuned, x_key[ids], y_key[ids], task.outputs)
        for a, g in zip(tuned, weight):
            a -= lr * g
    return score(tuned, task, x_eval, y_eval, task.outputs)


def keys_for(regime):
    return ("none", "gain", "readout", "finetune") if regime == "perm_shared" else ("none", "mask", "gain", "readout", "finetune")


def probe(net, task, task_id, data, key, n, seen, lr, seed):
    """Accuracy on one task after one key. Key samples depend on seed, task and n, not on the network."""
    x_eval, y_eval = task.pool(data.x_test, data.eval)
    if key == "none":
        return score(net, task, x_eval, y_eval, np.union1d(seen, task.outputs))
    if key == "mask":
        return score(net, task, x_eval, y_eval, task.outputs)
    rng = rng_for(seed, K_KEY, task_id, n)
    x_key, y_key = task.pool(data.x_train, data.key, n, rng)
    if key == "gain":
        return gain_score(net, task, x_key, y_key, x_eval, y_eval)
    if key == "readout":
        return ridge_score(net, task, x_key, y_key, x_eval, y_eval)
    return finetune_score(net, task, x_key, y_key, x_eval, y_eval, lr, rng)


# ---- Planted truth ----
def planted_access(w0, fraction, rng_seed):
    """Shuffle a fraction of last-hidden units before the head. Nothing is lost; the route is broken."""
    rng, net = np.random.default_rng(rng_seed), copy_net(w0)
    order = rng.permutation(net[4].shape[1])
    chosen = order[:int(round(fraction * len(order)))]
    if len(chosen) > 1:
        net[4][:, chosen] = net[4][:, np.roll(chosen, 1)]
    return net


def planted_erasure(w0, fraction, rng_seed):
    """Redraw a fraction of all weights from init. Those values are gone."""
    rng, net = np.random.default_rng(rng_seed), copy_net(w0)
    fresh = init_weights(rng, net[4].shape[0], net[0].shape[1], net[0].shape[0])
    for a, f in zip(net, fresh):
        mask = rng.random(a.shape) < fraction
        a[mask] = f[mask]
    return net


def match_fraction(build, target, s_none, steps=BISECT_STEPS):
    """Find the fraction whose S_none is closest to target, assuming S falls with the fraction."""
    low, high, best, best_gap = 0.0, 1.0, 1.0, float("inf")
    for _ in range(steps):
        mid = 0.5 * (low + high)
        value = s_none(build(mid))
        if abs(value - target) < best_gap:
            best, best_gap = mid, abs(value - target)
        low, high = (mid, high) if value > target else (low, mid)
    return best


# ---- One seed ----
def run_curve(data, regime, seed, lr, steps):
    """Outcome-free training curve for choosing a config: accuracy with no key, on A and on C."""
    lags = LAGS[regime]
    tasks = make_tasks(regime, seed, max(lags) + 1)
    net = init_weights(rng_for(seed, K_INIT), n_outputs(regime, max(lags) + 1))
    rng, out = rng_for(seed, K_TRAIN), {}
    a, c1, c2 = tasks[0], tasks[-2], tasks[-1]
    for t in range(max(lags) + 1):
        seen = seen_outputs(regime, tasks, t + 1)
        train_task(net, tasks[t], data, seen, lr, steps, rng)
        if t == 0:
            out["w0_acc"] = probe(net, a, 0, data, "mask", 0, seen, lr, seed)
        if t == 0 or t in lags:
            none_c = np.mean([probe(net, c, -i, data, "none", 0, seen, lr, seed) for i, c in ((1, c1), (2, c2))])
            out[f"none_A_{t}"], out[f"none_C_{t}"] = probe(net, a, 0, data, "none", 0, seen, lr, seed), float(none_c)
    out["final_acc"] = probe(net, tasks[max(lags)], max(lags), data, "mask", 0, seen, lr, seed)
    return out


def run_seed(data, regime, seed, lr, steps, m_star, budgets, blind):
    """Return rows of (net, lag, task, key, n, acc). If blind, A on a later network gets only `none`."""
    lags = tuple(m for m in LAGS[regime] if m <= (m_star if blind else max(LAGS[regime])))
    last = max(lags)
    tasks = make_tasks(regime, seed, max(LAGS[regime]) + 1)
    named = {"A": (tasks[0], 0), "C1": (tasks[-2], -1), "C2": (tasks[-1], -2)}
    net = init_weights(rng_for(seed, K_INIT), n_outputs(regime, max(LAGS[regime]) + 1))
    rng, rows, keys, w0 = rng_for(seed, K_TRAIN), [], keys_for(regime), None

    def measure(which, lag, candidate, seen, names=("A", "C1", "C2"), allowed=keys):
        for name in names:
            task, task_id = named[name]
            for key in allowed:
                for n in (0,) if key in ("none", "mask") else budgets:
                    acc = probe(candidate, task, task_id, data, key, n, seen, lr, seed)
                    rows.append((which, lag, name, key, n, acc))

    for t in range(last + 1):
        seen = seen_outputs(regime, tasks, t + 1)
        train_task(net, tasks[t], data, seen, lr, steps, rng)
        if t == 0:
            w0, seen0 = copy_net(net), seen
            measure("real", 0, net, seen)
        elif t in lags:
            measure("real", t, net, seen, ("C1", "C2"))
            if blind:
                measure("real", t, net, seen, ("A",), ("none",))
            else:
                measure("real", t, net, seen, ("A",))

    # Planted networks all start from w_0, matched to this seed's real S_none at the primary lag.
    table = {(r[1], r[2]): r[5] for r in rows if r[0] == "real" and r[3] == "none"}
    c0 = 0.5 * (table[(0, "C1")] + table[(0, "C2")])
    denominator = (table[(0, "A")] - c0) or 1e-9
    target = (table[(m_star, "A")] - 0.5 * (table[(m_star, "C1")] + table[(m_star, "C2")])) / denominator

    def s_none(candidate):
        a = probe(candidate, named["A"][0], 0, data, "none", 0, seen0, lr, seed)
        c = [probe(candidate, named[c][0], named[c][1], data, "none", 0, seen0, lr, seed) for c in ("C1", "C2")]
        return (a - float(np.mean(c))) / denominator

    plant_seed = int(rng_for(seed, K_PLANT).integers(2 ** 31))
    erase_seed = int(rng_for(seed, K_ERASE).integers(2 ** 31))
    shuffled = match_fraction(lambda f: planted_access(w0, f, plant_seed), target, s_none)
    erased = match_fraction(lambda f: planted_erasure(w0, f, erase_seed), target, s_none)
    measure("plant_access", 0, planted_access(w0, shuffled, plant_seed), seen0)
    measure("erase_matched", 0, planted_erasure(w0, erased, erase_seed), seen0)
    measure("erase_total", 0, planted_erasure(w0, 2.0, erase_seed), seen0)  # rng.random < 2 is always true
    return rows


_DATA = {}


def _init_worker(cache):
    _DATA.update(load_data(Path(cache))[0])


def load_data(cache):
    fashion, mnist, provenance = load_datasets(cache)
    perm = Data(mnist[0], mnist[1], mnist[2], mnist[3])
    split = Data(np.concatenate([mnist[0], fashion[0]]), np.concatenate([mnist[1], fashion[1] + 10]),
                 np.concatenate([mnist[2], fashion[2]]), np.concatenate([mnist[3], fashion[3] + 10]))
    return {"perm": perm, "split": split}, provenance


def family(regime):
    return "split" if regime == "split_class" else "perm"


def _curve_job(job):
    regime, seed, lr, steps = job
    return regime, seed, lr, steps, run_curve(_DATA[family(regime)], regime, seed, lr, steps)


def _seed_job(job):
    regime, seed, lr, steps, m_star, budgets, blind = job
    return regime, seed, run_seed(_DATA[family(regime)], regime, seed, lr, steps, m_star, budgets, blind)


# ---- Analysis ----
def index_rows(rows_by_seed):
    return {(seed, *row[:5]): row[5] for seed, rows in rows_by_seed.items() for row in rows}


def column(acc, seeds, which, lag, task, key, n):
    n = 0 if key in ("none", "mask") else n
    return np.array([acc[(s, which, lag, task, key, n)] for s in seeds])


def new_task(acc, seeds, which, lag, key, n):
    return 0.5 * (column(acc, seeds, which, lag, "C1", key, n) + column(acc, seeds, which, lag, "C2", key, n))


def surviving(acc, seeds, lag, key, n, which="real", fixed_baseline=False):
    """Per-seed numerator and denominator of S_k. The ratio of their means is S_k."""
    baseline_lag, baseline_net = (0, "real") if fixed_baseline else (lag, which)
    numerator = column(acc, seeds, which, lag, "A", key, n) - new_task(acc, seeds, baseline_net, baseline_lag, key, n)
    denominator = column(acc, seeds, "real", 0, "A", key, n) - new_task(acc, seeds, "real", 0, key, n)
    return numerator, denominator


def sham(acc, seeds, lag, key, n):
    """S from two tasks that were never trained. Expected 0 by exchangeability."""
    numerator = column(acc, seeds, "real", lag, "C1", key, n) - column(acc, seeds, "real", lag, "C2", key, n)
    return numerator, surviving(acc, seeds, lag, key, n)[1]


def ratio(numerator, denominator):
    return float(numerator.mean() / denominator.mean())


def shares(s_none, s_readout, s_finetune):
    lost = 1.0 - s_none
    return (s_readout - s_none) / lost, (s_finetune - s_readout) / lost, (1.0 - s_finetune) / lost


def label(access_ci, total_ci):
    if access_ci[0] > 0.5:
        return "mostly access"
    if access_ci[1] < 0.5 and total_ci[0] > 0.5:
        return "mostly repair"
    if total_ci[1] < 0.5:
        return "mostly unrecovered"
    return "no majority"


def resample_counts(n_seeds, n_bootstrap, seed=K_BOOT):
    rng = np.random.default_rng([ROOT_SEED, seed])
    return rng.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=n_bootstrap).astype(np.float64)


def ci(samples):
    low, high = np.percentile(samples, [2.5, 97.5])
    return [float(low), float(high)]


def decompose(acc, seeds, lag, n, counts, fixed_baseline=False):
    """Point estimates and intervals of the shares, resampling seeds jointly."""
    pairs = {k: surviving(acc, seeds, lag, k, n, fixed_baseline=fixed_baseline) for k in ("none", "readout", "finetune")}
    point = shares(*(ratio(*pairs[k]) for k in ("none", "readout", "finetune")))
    boot = {k: (counts @ v[0]) / (counts @ v[1]) for k, v in pairs.items()}
    access, repair, unrecovered = shares(boot["none"], boot["readout"], boot["finetune"])
    access_ci, total_ci = ci(access), ci(access + repair)
    return {"access": point[0], "repair": point[1], "unrecovered": point[2], "access_ci": access_ci,
            "repair_ci": ci(repair), "unrecovered_ci": ci(unrecovered), "access_plus_repair_ci": total_ci,
            "S": {k: ratio(*pairs[k]) for k in pairs}, "label": label(access_ci, total_ci)}


def evaluate_gates(acc, seeds, m_star, n):
    """Validity gates of the spec, at the primary lag and budget."""
    planted = lambda which, key: ratio(*surviving(acc, seeds, 0, key, n, which=which))
    plant_readout, plant_none = planted("plant_access", "readout"), planted("plant_access", "none")
    checks = {}
    for key in ("readout", "finetune"):
        value = ratio(*sham(acc, seeds, m_star, key, n))
        checks[f"sham_null_{key}"] = {"measured": value, "threshold": f"abs <= {NULL_LIMIT}", "pass": abs(value) <= NULL_LIMIT}
        value = planted("erase_total", key)
        checks[f"total_erasure_{key}"] = {"measured": value, "threshold": f"abs <= {NULL_LIMIT}", "pass": abs(value) <= NULL_LIMIT}
    checks["planted_access_readout"] = {"measured": plant_readout, "threshold": f">= {PLANT_MIN_S}", "pass": plant_readout >= PLANT_MIN_S}
    checks["planted_access_none"] = {"measured": plant_none, "threshold": f"<= {PLANT_MAX_NONE}", "pass": plant_none <= PLANT_MAX_NONE}
    matched = planted("erase_matched", "readout")
    checks["matched_erasure_gap"] = {"measured": plant_readout - matched, "threshold": f">= {MATCHED_GAP}",
                                     "pass": plant_readout - matched >= MATCHED_GAP, "matched_S_readout": matched,
                                     "matched_S_none": planted("erase_matched", "none")}
    return checks


def choose_config(curves, regime):
    """Apply the spec's outcome-free rule. curves maps (lr, steps) to a list of per-seed dicts."""
    lags, best = LAGS[regime], None
    for (lr, steps), per_seed in curves.items():
        mean = lambda key: float(np.mean([c[key] for c in per_seed]))
        if mean("w0_acc") < MIN_ACC or mean("final_acc") < MIN_ACC:
            continue
        denominator = mean("none_A_0") - mean("none_C_0")
        for lag in lags:
            value = (mean(f"none_A_{lag}") - mean(f"none_C_{lag}")) / denominator
            if value <= MAX_S_NONE:
                rank = (lag, -mean("w0_acc"), lr, steps)
                if best is None or rank < best[0]:
                    best = (rank, {"lr": lr, "steps": steps, "m_star": lag, "S_none": value, "w0_acc": mean("w0_acc"),
                                   "final_acc": mean("final_acc")})
                break
    return None if best is None else best[1]


def choose_budget(acc, seeds):
    """The smallest budget whose readout and finetune advantages on w_0 have real range."""
    for n in BUDGETS:
        values = {k: float(surviving(acc, seeds, 0, k, n)[1].mean()) for k in ("readout", "finetune")}
        if min(values.values()) >= MIN_ADVANTAGE:
            return n, values
    return None, None


# ---- Entry points ----
def calibrate(seeds, out_dir, cache, workers):
    _, provenance = load_data(cache)
    log.info("datasets %s", provenance)
    jobs = [(regime, seed, lr, steps) for regime in REGIMES for lr, steps in GRID for seed in seeds]
    curves = {regime: {} for regime in REGIMES}
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(cache),)) as pool:
        for regime, seed, lr, steps, curve in pool.map(_curve_job, jobs):
            curves[regime].setdefault((lr, steps), []).append(curve)
            log.info("curve %s lr=%s steps=%s seed=%s", regime, lr, steps, seed)
        result = {"seeds": list(seeds), "notes": NOTES, "datasets": provenance, "regimes": {}}
        chosen = {r: choose_config(curves[r], r) for r in REGIMES}
        seed_jobs = [(r, s, c["lr"], c["steps"], c["m_star"], BUDGETS, True) for r, c in chosen.items() if c for s in seeds]
        rows = {r: {} for r in REGIMES}
        for regime, seed, seed_rows in pool.map(_seed_job, seed_jobs):
            rows[regime][seed] = seed_rows
            log.info("calibration %s seed=%s", regime, seed)
    for regime in REGIMES:
        grid = {f"{lr}/{steps}": {k: float(np.mean([c[k] for c in per])) for k in per[0]} for (lr, steps), per in curves[regime].items()}
        entry = {"config": chosen[regime], "grid_means": grid, "answerable": False}
        result["regimes"][regime] = entry
        if not chosen[regime]:
            entry["reason"] = "no config met the accuracy and forgetting rule"
            continue
        acc = index_rows(rows[regime])
        n, advantages = choose_budget(acc, seeds)
        entry["advantage_w0"] = advantages
        if n is None:
            entry["reason"] = "no budget gave a large enough advantage on w_0"
            continue
        checks = evaluate_gates(acc, seeds, chosen[regime]["m_star"], n)
        entry.update(budget=n, checks=checks, answerable=all(c["pass"] for c in checks.values()))
        if not entry["answerable"]:
            entry["reason"] = "failed: " + ", ".join(k for k, c in checks.items() if not c["pass"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(json.dumps(result, indent=2))
    log.info("answerable: %s", [r for r in REGIMES if result["regimes"][r]["answerable"]])


def confirm(out_dir, cache, workers):
    calibration = json.loads((out_dir / "calibration.json").read_text())
    regimes = {r: e for r, e in calibration["regimes"].items() if e["answerable"]}
    if not regimes:
        raise SystemExit("no regime passed calibration")
    jobs = [(r, s, e["config"]["lr"], e["config"]["steps"], e["config"]["m_star"], BUDGETS, False)
            for r, e in regimes.items() for s in CONFIRM_SEEDS]
    rows = {r: {} for r in regimes}
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(cache),)) as pool:
        for regime, seed, seed_rows in pool.map(_seed_job, jobs):
            rows[regime][seed] = seed_rows
            log.info("confirm %s seed=%s", regime, seed)
    with open(out_dir / "raw_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("regime", "seed", "net", "lag", "task", "key", "n", "acc"))
        for regime, per in rows.items():
            for seed, seed_rows in per.items():
                writer.writerows((regime, seed, *row) for row in seed_rows)
    (out_dir / "summary.json").write_text(json.dumps(analyse(rows, regimes), indent=2))


def analyse(rows, regimes):
    seeds, counts, summary = list(CONFIRM_SEEDS), resample_counts(len(CONFIRM_SEEDS), N_BOOTSTRAP), {}
    for regime, entry in regimes.items():
        acc, m_star, n = index_rows(rows[regime]), entry["config"]["m_star"], entry["budget"]
        primary = decompose(acc, seeds, m_star, n, counts)
        fixed = decompose(acc, seeds, m_star, n, counts, fixed_baseline=True)
        checks = evaluate_gates(acc, seeds, m_star, n)
        readout_a = column(acc, seeds, "real", m_star, "A", "readout", n)
        readout_a0 = column(acc, seeds, "real", 0, "A", "readout", n)
        summary[regime] = {
            "m_star": m_star, "budget": n, "primary": primary, "fixed_baseline": fixed,
            "verdict": primary["label"] if all(c["pass"] for c in checks.values()) else "void: gates failed",
            "robust": primary["label"] == fixed["label"], "checks": checks,
            "naive_retention_readout": float(readout_a.mean() / readout_a0.mean()),
            "S_by_key": {k: ratio(*surviving(acc, seeds, m_star, k, n)) for k in keys_for(regime)},
            "new_task_by_lag": {str(lag): {k: float(new_task(acc, seeds, "real", lag, k, n).mean()) for k in keys_for(regime)}
                                for lag in (0, *LAGS[regime])},
            "trained_task_by_lag": {str(lag): {k: float(column(acc, seeds, "real", lag, "A", k, n).mean()) for k in keys_for(regime)}
                                    for lag in (0, *LAGS[regime])},
            "by_lag": {str(lag): decompose(acc, seeds, lag, n, counts) for lag in LAGS[regime]},
            "by_budget": {str(b): decompose(acc, seeds, m_star, b, counts) for b in BUDGETS},
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("calibrate", "calibrate-retry", "confirm"))
    parser.add_argument("--out", default="results/h5")
    parser.add_argument("--cache", default=str(Path.home() / ".cache" / "clu"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    out_dir, cache = Path(args.out), Path(args.cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "run.log")])
    if args.mode == "confirm":
        confirm(out_dir, cache, args.workers)
    else:
        calibrate(CAL_SEEDS if args.mode == "calibrate" else RETRY_SEEDS, out_dir, cache, args.workers)


if __name__ == "__main__":
    main()
