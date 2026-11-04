#!/usr/bin/env python3
"""H4 experiment; see hypotheses/H4_schema_replay_dose.md. Numpy, hand-written backprop.

python experiments/h4_schema.py --stage0   # 3 pilot seeds; writes results/h4/stage0.json
python experiments/h4_schema.py            # full run; refuses unless Stage 0 passed
Writes run.log, raw_results.csv (resume checkpoint), summary.json (atomic). Rerun to resume;
pass --fresh after a design change. Everything the spec leaves open is a named constant below,
fixed before any data exists. No tuning run exists: CALIBRATION_SEEDS are reserved and unused.

Runtime estimate, 2 cores, nothing here has been timed. A 64-sample step costs about 34M
multiply-adds, taken as 5 ms. One event is 500 steps at each of 5 rungs (220 samples per step
across the rungs, so about 8.6 s), 15 evaluations (about 1.5 s) and overlap (0.2 s): about 10 s.
A seed is 25 events plus 7 schema phases (about 5 s each, worst case 100 s): about 290 s, or
380 s with both cores busy. 40 seeds on 2 workers: about 127 minutes, plus about 3 for the
analysis. Stage 0: about 10 minutes. The estimate may be wrong by 2x and still fit 320 minutes;
beyond that the run resumes from the checkpoint. No reduction from the spec was needed.
"""
from __future__ import annotations

import argparse, csv, gzip, hashlib, io, json, logging, math, os, sys, time, urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

log = logging.getLogger("h4")

NOTES = [
    "N is 11 for every network: 10 Fashion-MNIST outputs and one foreign output, so parameter count"
    " is matched. Softmax runs over active outputs only; other outputs stay at their random init",
    "old-class accuracy is argmax over old classes plus the new one, before and after, so drop"
    " measures only the update",
    "singletons (trouser, bag) count as inconsistent: they have no relatives by construction",
    "consistent networks withhold two classes each, so all eight non-singleton classes appear both"
    " as consistent and as inconsistent. Schema size differs: 8 (consistent, singleton), 7"
    " (footwear withheld), 5 (upper body withheld). The spec's model has no term for this",
    "a fixed accuracy criterion does not equalise steps: the easy upper-withheld schema reaches it"
    " far sooner. Followed as written; steps per net are in the summary, and verdict-free"
    " sensitivity fits drop that net and, separately, the singleton classes",
    "foreign: one digit per schema network, drawn without replacement within a seed, on all 7 nets",
    "schema_condition has 3 levels with inconsistent as reference, so the consistent column is the"
    " coefficient of interest. VIF of condition is the larger VIF of its two columns",
    "predicted drop from overlap reads as the spec's named overlap variable, so the check is Pearson"
    " r(overlap, drop) at zero replay and the short length. r with the unnormalised inner product is"
    " reported and decides nothing",
    "positive control, new_acc and censoring are at the medium length. The new_acc check is each"
    " condition mean at zero replay, the setting of the primary model; the minimum over all rungs is"
    " reported and decides nothing (Deviation 2)",
    "added beyond the preregistered Stage 0 list: every schema network must reach the criterion;"
    " in the full run, events on a network that missed it are excluded and counted",
    "all intervals are 95 percent percentile cluster bootstrap, as in H2, including for equivalence."
    " All threshold comparisons on CIs are strict",
    "lengths share one trajectory per rung and rungs share new-class and old-class draws (nested), so"
    " contrasts across lengths and rungs are paired",
    "growth must hold in points and in units of the drop SD at that length, because all drop grows"
    " with length and points alone could not fail. Failing either triggers the preregistered caveat",
    "dose: the smallest rung with drop strictly under 2, at the medium length. Censored events are"
    " never imputed at the top rung: reported as censored fractions, medians treating them as above"
    " the top rung, and a rank mean labelled as a lower bound",
    "new_acc within 2 points is the largest gap between condition means at medium, zero replay",
    "Fashion-MNIST is checked against the published MD5s (no sha256 is published) and MNIST against"
    " the sha256 Keras publishes; sha256 of every file is logged, and the full run requires the"
    " values recorded by Stage 0",
    "fine-tuning loss is the mean over the combined batch, so replay also dilutes the new-class"
    " gradient. New-class draws per step stay fixed, so the ladder is exact",
]

# ---- Fixed by the spec ----
LADDER = (0.0, 0.125, 0.25, 0.5, 1.0)
DROP_TARGET = 2.0  # dose: drop strictly under this, in points
EQUIVALENCE_BOUND = 1.0
NEW_ACC_SPREAD = 2.0
VIF_LIMIT = 5.0
FIRST_ORDER_R = 0.5
POSITIVE_DROP = 5.0
POSITIVE_REDUCTION = 0.5
NEW_ACC_FLOOR = 70.0
CENSOR_LIMIT = 0.5
N_BOOTSTRAP = 10000
STAGE0_SEEDS = (300, 301, 302)

# ---- Left open by the spec; chosen from first principles, not from outcomes ----
# Reserved for tuning. Nothing is tuned, so these never run; kept disjoint from the rest.
CALIBRATION_SEEDS = (200, 201, 202)
# 40, not the minimum 20. A CI within +-1 point needs a between-seed contrast SD of about 3.2
# at n = 40, against 2.2 at n = 20. The runtime estimate leaves room.
EVAL_SEEDS = tuple(range(400, 440))
SHORT_STEPS = 10  # With LR 0.002, gradient-flow time 0.02, around where new-class logits overtake
MEDIUM_STEPS = 100  # Time 0.2: enough to separate relatives at 1:1 replay, which halves the new gradient
LONG_STEPS = 500  # Time 1.0, 50x short: a wide span for the higher-order prediction
# Plain SGD, because the bound is for a gradient step. Logit-margin change per step is about
# LR * NTK, with NTK about 500 estimated from He-init norms: about 1 per step, a small step.
FINE_LR = 0.002
NEW_PER_STEP = 32  # A power of 2 at least 8, so each rung is a whole number: 0, 4, 8, 16, 32 old
# 80 percent: the hardest schema (upper body, trouser, bag) plateaus near 85 in published MLP
# per-class accuracies, so all schemas can reach this one, below their plateaus.
SCHEMA_CRITERION = 0.80
SCHEMA_LR = 0.05  # Common stable MLP rate on [0, 1] inputs; slow enough that the check is not overshot
SCHEMA_BATCH = 64  # Conventional; also the size at which the runtime estimate is made
SCHEMA_CHECK_EVERY = 25  # Small relative to steps to criterion, to limit overshoot
SCHEMA_MAX_STEPS = 20000  # About 32 epochs on the largest schema; a miss is recorded, not hidden
# Every schema training image of an old class is in the buffer, the "large buffer" of the spec.
# The largest draw is 500 * 32 = 16000 images, against at least 25000 in the buffer.
BUFFER_PER_CLASS = 5000
VAL_PER_CLASS = 500  # From training images; SE about 0.6 points on 8 classes at 80 percent
# Round counts that every MNIST digit can supply (rarest: 5421 train, 892 test), so Fashion classes
# and digits use identical counts. Test images are split so overlap and evaluation are disjoint.
TRAIN_PER_CLASS = BUFFER_PER_CLASS
MEASURE_PER_CLASS = 400
EVAL_PER_CLASS = 400
OVERLAP_BATCHES = 10  # Enough to average batch noise in a cosine; cost is negligible
OVERLAP_BATCH = 64  # Same as the schema batch

ROOT_SEED, SUBSET_SEED, BOOT_SEED = 20261201, 20261202, 20261203
K_INIT, K_SCHEMA, K_MEASURE, K_NEW, K_OLD, K_DIGIT = range(6)

SHAPES = ((256, 784), (256,), (256, 256), (256,), (11, 256), (11,))
FOREIGN_OUTPUT = 10
FOREIGN_BASE = 10  # Internal class id of MNIST digit d is FOREIGN_BASE + d
UPPER_BODY, FOOTWEAR, SINGLETONS = (0, 2, 3, 4, 6), (5, 7, 9), (1, 8)
SUPERCLASSES = (UPPER_BODY, FOOTWEAR, (1,), (8,))
# name, withheld classes, condition. Coat and ankle boot are the spec's consistent examples.
NETS = (("C1", (4, 9), "consistent"), ("C2", (6, 7), "consistent"), ("C3", (2, 5), "consistent"),
        ("C4", (0, 3), "consistent"), ("I_foot", FOOTWEAR, "inconsistent"),
        ("I_upper", UPPER_BODY, "inconsistent"), ("I_single", SINGLETONS, "inconsistent"))
CONDITIONS = ("consistent", "inconsistent", "foreign")
MODEL_COLUMNS = ("intercept", "consistent", "foreign", "overlap", "distance")

LABEL_FIRST_ORDER = "H4a and H4b unanswerable as framed: the first-order account fails here"
LABEL_VIF = "H4a and H4b not answerable: variance inflation above 5"
LABEL_NO_RAW = "no schema effect in this setting"
LABEL_H4A = "H4a supported"
LABEL_H4B = "H4b supported"
LABEL_INCONCLUSIVE = "inconclusive"
CAVEAT_NO_GROWTH = ("The residual did not grow from short to long training, so it likely reflects"
                    " covariate measurement error rather than a real extra mechanism.")


@dataclass(frozen=True)
class Plan:
    seeds: tuple
    lengths: tuple = (SHORT_STEPS, MEDIUM_STEPS, LONG_STEPS)
    n_bootstrap: int = N_BOOTSTRAP
    train_per_class: int = TRAIN_PER_CLASS
    val_per_class: int = VAL_PER_CLASS
    measure_per_class: int = MEASURE_PER_CLASS
    eval_per_class: int = EVAL_PER_CLASS
    schema_lr: float = SCHEMA_LR
    schema_batch: int = SCHEMA_BATCH
    schema_criterion: float = SCHEMA_CRITERION
    schema_check_every: int = SCHEMA_CHECK_EVERY
    schema_max_steps: int = SCHEMA_MAX_STEPS
    fine_lr: float = FINE_LR
    n_new: int = NEW_PER_STEP
    overlap_batches: int = OVERLAP_BATCHES
    overlap_batch: int = OVERLAP_BATCH


STAGE0_PLAN = Plan(STAGE0_SEEDS)
FULL_PLAN = Plan(EVAL_SEEDS)


def design_hash(plan):
    """Identifies the design apart from its seeds; a Stage 0 pass is only valid for this hash."""
    d = asdict(plan)
    d.pop("seeds")
    d.update(ladder=LADDER, nets=NETS, thresholds=[DROP_TARGET, EQUIVALENCE_BOUND, NEW_ACC_SPREAD, VIF_LIMIT,
                                                   FIRST_ORDER_R, POSITIVE_DROP, POSITIVE_REDUCTION,
                                                   NEW_ACC_FLOOR, CENSOR_LIMIT])
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def key(seed, *tags):
    return np.random.default_rng([ROOT_SEED, seed, *tags])


# ---- Network ----
def layer_views(w, shapes=SHAPES):
    out, i = [], 0
    for shape in shapes:
        n = int(np.prod(shape))
        out.append(w[i:i + n].reshape(shape))
        i += n
    return out


def init_weights(rng, shapes=SHAPES, dtype=np.float32):
    w = np.zeros(sum(int(np.prod(s)) for s in shapes), dtype=dtype)
    for view in layer_views(w, shapes):
        if view.ndim == 2:  # He normal; biases stay 0
            view[:] = rng.standard_normal(view.shape, dtype=dtype) * math.sqrt(2.0 / view.shape[1])
    return w


def forward(v, X):
    h1 = np.maximum(X @ v[0].T + v[1], 0.0)
    h2 = np.maximum(h1 @ v[2].T + v[3], 0.0)
    return h1, h2, h2 @ v[4].T + v[5]


def batch_grad(v, gv, X, y, active):
    """Write the mean cross-entropy gradient, softmax over `active` outputs, into gv."""
    h1, h2, z = forward(v, X)
    za = z[:, active]
    p = np.exp(za - za.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    err = np.zeros_like(z)
    err[:, active] = p
    err[np.arange(len(y)), y] -= 1.0
    err /= len(y)
    gv[5][:] = err.sum(axis=0)
    gv[4][:] = err.T @ h2
    d2 = (err @ v[4]) * (h2 > 0)
    gv[3][:] = d2.sum(axis=0)
    gv[2][:] = d2.T @ h1
    d1 = (d2 @ v[2]) * (h1 > 0)
    gv[1][:] = d1.sum(axis=0)
    gv[0][:] = d1.T @ X


def batch_loss(v, X, y, active):
    za = forward(v, X)[2][:, active]
    za = za - za.max(axis=1, keepdims=True)
    picked = za[np.arange(len(y)), np.searchsorted(active, y)]
    return float(np.mean(np.log(np.exp(za).sum(axis=1)) - picked))


def gradient(w, X, y, active, shapes=SHAPES):
    g = np.zeros_like(w)
    batch_grad(layer_views(w, shapes), layer_views(g, shapes), X, y, active)
    return g


def accuracy(v, X, y, active):
    return float((active[forward(v, X)[2][:, active].argmax(axis=1)] == y).mean())


def overlap_of(g_old, g_new):
    """Return (normalised negative overlap, raw inner product). Positive overlap means interference."""
    a, b = g_old.astype(np.float64), g_new.astype(np.float64)
    inner, norm = float(a @ b), float(np.linalg.norm(a) * np.linalg.norm(b))
    return (-inner / norm if norm > 0.0 else 0.0), inner


def cosine_distance(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 - float(a @ b) / norm if norm > 0.0 else 1.0


# ---- Data ----
def output_label(c):
    return c if c < FOREIGN_BASE else FOREIGN_OUTPUT


class Data:
    """Fixed image pools. Class ids: 0-9 Fashion-MNIST, FOREIGN_BASE + d for MNIST digit d."""

    def __init__(self, fashion, mnist, plan):
        rng = np.random.default_rng(SUBSET_SEED)
        self.train, self.val, self.measure, self.eval = {}, {}, {}, {}
        chunks, labels, offset = [], [], 0
        n_tr, n_m, n_e = plan.train_per_class, plan.measure_per_class, plan.eval_per_class
        for base, (x_tr, y_tr, x_te, y_te), n_val in ((0, fashion, plan.val_per_class), (FOREIGN_BASE, mnist, 0)):
            for k in range(10):
                c = base + k
                tr = rng.permutation(np.flatnonzero(y_tr == k))
                te = rng.permutation(np.flatnonzero(y_te == k))
                if len(tr) < n_tr + n_val or len(te) < n_m + n_e:
                    raise ValueError(f"class {c} has {len(tr)} train and {len(te)} test images, too few to match")
                part = np.concatenate((x_tr[tr[:n_tr + n_val]], x_te[te[:n_m + n_e]]))
                self.train[c] = np.arange(n_tr) + offset
                self.val[c] = np.arange(n_tr, n_tr + n_val) + offset
                self.measure[c] = np.arange(n_tr + n_val, n_tr + n_val + n_m) + offset
                self.eval[c] = np.arange(n_tr + n_val + n_m, n_tr + n_val + n_m + n_e) + offset
                chunks.append(part)
                labels.append(np.full(len(part), output_label(c), dtype=np.int64))
                offset += len(part)
        self.X = np.concatenate(chunks).astype(np.float32) / 255.0
        self.y = np.concatenate(labels)


def union(pools, classes):
    return np.concatenate([pools[c] for c in classes])


def schema_classes(withheld):
    return tuple(c for c in range(10) if c not in withheld)


def relatives(c):
    return tuple(m for group in SUPERCLASSES if c in group for m in group if m != c)


def foreign_digit(seed, net_index):
    """Without replacement within a seed; keyed on the seed alone so it is the same on resume."""
    return int(key(seed, K_DIGIT).permutation(10)[net_index])


def events_for(seed, net_index):
    _, withheld, condition = NETS[net_index]
    return [(condition, c) for c in withheld] + [("foreign", FOREIGN_BASE + foreign_digit(seed, net_index))]


def replay_counts(n_new):
    counts = []
    for ratio in LADDER:
        n = ratio * n_new
        if n != int(n):
            raise ValueError(f"{n_new} new samples cannot express replay ratio {ratio} exactly")
        counts.append(int(n))
    return tuple(counts)


# ---- Experiment ----
def train_schema(w, data, classes, rng, plan):
    """SGD in place until held-out accuracy on the schema classes reaches the criterion."""
    v, g = layer_views(w), np.zeros_like(w)
    gv, active = layer_views(g), np.array(sorted(classes))
    pool, val = union(data.train, classes), union(data.val, classes)
    steps, acc = 0, 0.0
    while steps < plan.schema_max_steps:
        idx = pool[rng.integers(len(pool), size=plan.schema_batch)]
        batch_grad(v, gv, data.X[idx], data.y[idx], active)
        w -= plan.schema_lr * g
        steps += 1
        if steps % plan.schema_check_every == 0:
            acc = accuracy(v, data.X[val], data.y[val], active)
            if acc >= plan.schema_criterion:
                return steps, acc, True
    return steps, accuracy(v, data.X[val], data.y[val], active), False


def measure_overlap(w, data, old_pool, new_pool, active, rng, plan):
    """Mean over batches, on held-out images, before any update."""
    pairs = []
    for _ in range(plan.overlap_batches):
        o = old_pool[rng.integers(len(old_pool), size=plan.overlap_batch)]
        n = new_pool[rng.integers(len(new_pool), size=plan.overlap_batch)]
        pairs.append(overlap_of(gradient(w, data.X[o], data.y[o], active), gradient(w, data.X[n], data.y[n], active)))
    return float(np.mean([p[0] for p in pairs])), float(np.mean([p[1] for p in pairs]))


def nearest_old_distance(v, data, new_class, old_classes):
    mean = lambda c: forward(v, data.X[data.measure[c]])[1].mean(axis=0)  # noqa: E731
    new = mean(new_class)
    return min(cosine_distance(new, mean(c)) for c in old_classes)


def fine_tune(w0, data, active, new_idx, old_idx, n_old, plan, old_eval, new_eval):
    """One trajectory; returns {length: (old-class accuracy, new-class accuracy)} at each length."""
    w, out = w0.copy(), {}
    v, g = layer_views(w), np.zeros_like(w)
    gv = layer_views(g)
    for step in range(1, max(plan.lengths) + 1):
        idx = np.concatenate((new_idx[step - 1], old_idx[step - 1, :n_old]))
        batch_grad(v, gv, data.X[idx], data.y[idx], active)
        w -= plan.fine_lr * g
        if step in plan.lengths:
            out[step] = (accuracy(v, data.X[old_eval], data.y[old_eval], active),
                         accuracy(v, data.X[new_eval], data.y[new_eval], active))
    return out


def run_event(w0, data, plan, seed, net_index, condition, new_class, old_classes):
    active = np.array(sorted(list(old_classes) + [output_label(new_class)]))
    v0 = layer_views(w0)
    overlap, inner = measure_overlap(w0, data, union(data.measure, old_classes), data.measure[new_class], active,
                                     key(seed, K_MEASURE, net_index, new_class), plan)
    distance = nearest_old_distance(v0, data, new_class, old_classes)
    old_eval, new_eval = union(data.eval, old_classes), data.eval[new_class]
    base = accuracy(v0, data.X[old_eval], data.y[old_eval], active)
    n_steps, counts = max(plan.lengths), replay_counts(plan.n_new)
    pool_new, buffer = data.train[new_class], union(data.train, old_classes)
    new_idx = pool_new[key(seed, K_NEW, net_index, new_class).integers(len(pool_new), size=(n_steps, plan.n_new))]
    old_idx = buffer[key(seed, K_OLD, net_index, new_class).integers(len(buffer), size=(n_steps, counts[-1]))]
    rows = []
    for rung, n_old in enumerate(counts):
        results = fine_tune(w0, data, active, new_idx, old_idx, n_old, plan, old_eval, new_eval)
        for length, (old_acc, new_acc) in results.items():
            rows.append({"condition": condition, "new_class": new_class, "length": length, "rung": rung,
                         "ratio": LADDER[rung], "drop": 100.0 * (base - old_acc), "new_acc": 100.0 * new_acc,
                         "base_acc": 100.0 * base, "old_acc": 100.0 * old_acc, "overlap": overlap, "inner": inner,
                         "distance": distance, "n_old": len(old_classes)})
    return rows


_DATA = None


def _init_worker(cache, plan):
    global _DATA
    np.seterr(all="ignore")
    fashion, mnist, _ = load_datasets(Path(cache))
    _DATA = Data(fashion, mnist, plan)


def run_job(job):
    plan, seed, net_index = job
    started = time.time()
    name, withheld, _ = NETS[net_index]
    classes = schema_classes(withheld)
    w = init_weights(key(seed, K_INIT))  # One init per seed, so networks within a seed differ only by schema
    steps, acc, reached = train_schema(w, _DATA, classes, key(seed, K_SCHEMA, net_index), plan)
    rows = []
    for condition, new_class in events_for(seed, net_index):
        rows += run_event(w, _DATA, plan, seed, net_index, condition, new_class, classes)
    seconds = round(time.time() - started, 1)
    design = design_hash(plan)
    for r in rows:
        r.update(design=design, seed=seed, net=name, schema_steps=steps, schema_acc=100.0 * acc,
                 schema_reached=int(reached), seconds=seconds)
    return rows


class RawStore:
    """Resumable CSV. A job counts only if all its rows survive, so a cut-off run loses at most that job."""

    TEXT = ("design", "net", "condition")
    INTS = ("seed", "new_class", "length", "rung", "n_old", "schema_steps", "schema_reached")
    FLOATS = ("ratio", "drop", "new_acc", "base_acc", "old_acc", "overlap", "inner", "distance", "schema_acc", "seconds")
    FIELDS = TEXT + INTS + FLOATS

    def __init__(self, path, plan, fresh=False):
        self.design = design_hash(plan)
        self.expected = {n[0]: (len(n[1]) + 1) * len(plan.lengths) * len(LADDER) for n in NETS}
        parsed = []
        if path.exists() and not fresh:
            with path.open(newline="") as handle:
                parsed = [t for t in map(self._parse, csv.DictReader(handle)) if t is not None]
        stale = {r["design"] for r in parsed} - {self.design}
        if stale:
            raise ValueError(f"{path} was written by a different design; pass --fresh to discard it")
        groups = {}
        for r in parsed:
            groups.setdefault((r["seed"], r["net"]), {})[(r["new_class"], r["length"], r["rung"])] = r
        self._done = {k for k, g in groups.items() if len(g) == self.expected.get(k[1])}
        self._rows = [r for k in sorted(self._done) for r in groups[k].values()]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")  # Rewrite without any row of an unfinished job
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)
        os.replace(tmp, path)
        self._handle = path.open("a", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDS)

    def _parse(self, row):
        try:
            if None in row or None in row.values():
                return None
            out = {f: row[f] for f in self.TEXT}
            out.update({f: int(row[f]) for f in self.INTS})
            out.update({f: float(row[f]) for f in self.FLOATS})
            return out
        except (ValueError, KeyError):
            return None

    def done(self, seed, net):
        return (seed, net) in self._done

    def add(self, rows):
        clean = [{f: r[f] for f in self.FIELDS} for r in rows]
        self._writer.writerows(clean)
        self._handle.flush()
        self._rows += clean
        self._done.add((rows[0]["seed"], rows[0]["net"]))

    def rows(self):
        return list(self._rows)

    def close(self):
        self._handle.close()


def execute(plan, store, workers, cache, raw, label):
    """Run every unfinished job. Progress lines carry no outcome values, so nothing can be peeked at."""
    todo = [(plan, seed, i) for seed in plan.seeds for i, n in enumerate(NETS) if not store.done(seed, n[0])]
    log.info("%s: %d jobs to run", label, len(todo))
    started = time.time()
    if workers == 0:
        global _DATA
        _DATA = Data(raw[0], raw[1], plan)
        for n, job in enumerate(todo, 1):
            store.add(run_job(job))
            log.info("%s %d/%d seed=%s %.0fs", label, n, len(todo), job[1], time.time() - started)
        return
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(cache), plan)) as pool:
        futures = [pool.submit(run_job, job) for job in todo]
        for n, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            store.add(rows)
            log.info("%s %d/%d seed=%s net=%s %.0fs", label, n, len(todo), rows[0]["seed"], rows[0]["net"], time.time() - started)


# ---- Statistics ----
def select(rows, length, rung=0):
    return [r for r in rows if r["length"] == length and r["rung"] == rung]


def pearson(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 3 or a.std() == 0.0 or b.std() == 0.0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def percentile_ci(samples):
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return float(low), float(high)


def seed_resamples(n_seeds, n_bootstrap):
    return np.random.default_rng(BOOT_SEED).integers(0, n_seeds, size=(n_bootstrap, n_seeds))


def vif(columns):
    """Variance inflation of each column given the others and an intercept."""
    names, out = list(columns), {}
    for name in names:
        y = np.asarray(columns[name], dtype=np.float64)
        others = [np.ones(len(y))] + [np.asarray(columns[o], dtype=np.float64) for o in names if o != name]
        A = np.column_stack(others)
        resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        total = float(((y - y.mean()) ** 2).sum())
        rss = float(resid @ resid)
        out[name] = total / rss if total > 0.0 and rss > 1e-12 * total else math.inf
    return out


def design_matrix(rows):
    n = len(rows)
    return np.column_stack((np.ones(n), [float(r["condition"] == "consistent") for r in rows],
                            [float(r["condition"] == "foreign") for r in rows],
                            [r["overlap"] for r in rows], [r["distance"] for r in rows]))


def condition_vif(rows, plan):
    """VIF of condition against distance and overlap, on the events of the primary model."""
    primary = select(rows, plan.lengths[1])
    X = design_matrix(primary)
    v = vif({c: X[:, i] for i, c in enumerate(MODEL_COLUMNS) if c != "intercept"})
    return max(v["consistent"], v["foreign"]), v


class MixedModel:
    """y ~ X + (1 | group), REML, profiled over lam = var(group) / var(residual).

    Per-group sufficient statistics make a cluster resample a reweighting. A group drawn twice
    counts as two groups, each with its own intercept, as a cluster bootstrap requires.
    """

    GRID = np.concatenate(([0.0], np.logspace(-3, 2, 11)))

    def __init__(self, X, y, groups, seeds):
        self.seeds = list(seeds)
        G, p = len(self.seeds), X.shape[1]
        self.p, self.n = p, np.zeros(G)
        self.xx, self.bxx = np.zeros((G, p, p)), np.zeros((G, p, p))
        self.xy, self.bxy = np.zeros((G, p)), np.zeros((G, p))
        self.yy, self.byy = np.zeros(G), np.zeros(G)
        groups = np.asarray(groups)
        for g, seed in enumerate(self.seeds):
            mask = groups == seed
            if not mask.any():
                continue
            Xg, yg = X[mask], y[mask]
            n, sx, sy = float(mask.sum()), Xg.sum(axis=0), float(yg.sum())
            self.n[g] = n
            self.xx[g], self.xy[g], self.yy[g] = Xg.T @ Xg, Xg.T @ yg, yg @ yg
            self.bxx[g], self.bxy[g], self.byy[g] = np.outer(sx, sx) / n, sx * sy / n, sy * sy / n

    def at(self, lam, w):
        """Return (REML criterion, coefficients, residual variance) at this variance ratio."""
        wc = w * (self.n * lam / (1.0 + self.n * lam))
        A = np.tensordot(w, self.xx, axes=1) - np.tensordot(wc, self.bxx, axes=1)
        b = w @ self.xy - wc @ self.bxy
        beta = np.linalg.solve(A, b)
        dof = float(w @ self.n) - self.p
        rss = max(float(w @ self.yy - wc @ self.byy - b @ beta), 1e-300)
        return dof * math.log(rss / dof) + float(w @ np.log1p(self.n * lam)) + np.linalg.slogdet(A)[1], beta, rss / dof

    def fit(self, w=None):
        w = np.ones(len(self.seeds)) if w is None else w
        vals = [self.at(float(lam), w)[0] for lam in self.GRID]
        i = int(np.argmin(vals))
        a, b = float(self.GRID[max(i - 1, 0)]), float(self.GRID[min(i + 1, len(self.GRID) - 1)])
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        c, d = b - phi * (b - a), a + phi * (b - a)
        fc, fd = self.at(c, w)[0], self.at(d, w)[0]
        for _ in range(12):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = self.at(c, w)[0]
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = self.at(d, w)[0]
        lam = c if fc < fd else d
        if vals[i] <= min(fc, fd):
            lam = float(self.GRID[i])
        _, beta, sigma2 = self.at(lam, w)
        return beta, sigma2, lam

    def bootstrap(self, res):
        out = np.zeros((len(res), self.p))
        for b, draw in enumerate(res):
            out[b] = self.fit(np.bincount(draw, minlength=len(self.seeds)).astype(np.float64))[0]
        return out


def size_and_depth(rows):
    """Schema size and log steps to criterion, leaving out a column that does not vary."""
    columns = [np.array([r["n_old"] for r in rows], dtype=np.float64),
               np.log(np.array([r["schema_steps"] for r in rows], dtype=np.float64))]
    return [c for c in columns if np.ptp(c) > 0.0]


def adjusted_model(rows, seeds, res, extra=()):
    """drop ~ schema_condition + overlap + distance + (1 | seed). Inconsistent is the reference."""
    y = np.array([r["drop"] for r in rows], dtype=np.float64)
    model = MixedModel(np.column_stack((design_matrix(rows), *extra)), y, [r["seed"] for r in rows], seeds)
    beta, sigma2, lam = model.fit()
    boot = model.bootstrap(res)
    return {"beta": beta, "boot": boot, "sigma2_residual": sigma2, "sigma2_seed": sigma2 * lam, "sd_drop": float(y.std())}


def per_seed_means(rows, seeds, field):
    by_seed = {s: [] for s in seeds}
    for r in rows:
        by_seed[r["seed"]].append(r[field])
    return np.array([np.mean(by_seed[s]) if by_seed[s] else np.nan for s in seeds], dtype=np.float64)


def raw_effect(rows, seeds, res):
    """Consistent minus inconsistent drop, no covariates: seed by seed, then a cluster bootstrap."""
    a = per_seed_means([r for r in rows if r["condition"] == "consistent"], seeds, "drop")
    b = per_seed_means([r for r in rows if r["condition"] == "inconsistent"], seeds, "drop")
    d = a - b
    low, high = percentile_ci(np.nanmean(d[res], axis=1))
    return {"diff": float(np.nanmean(d)), "ci_low": low, "ci_high": high}


def bootstrap_r(rows, seeds, res):
    """Cluster bootstrap of r(overlap, drop)."""
    by_seed = {s: ([], []) for s in seeds}
    for r in rows:
        by_seed[r["seed"]][0].append(r["overlap"])
        by_seed[r["seed"]][1].append(r["drop"])
    xs = [np.array(by_seed[s][0], dtype=np.float64) for s in seeds]
    ys = [np.array(by_seed[s][1], dtype=np.float64) for s in seeds]
    return percentile_ci([pearson(np.concatenate([xs[i] for i in draw]), np.concatenate([ys[i] for i in draw])) for draw in res])


def dose_of(drops, target=DROP_TARGET):
    """Index of the smallest rung with drop strictly under the target; None means censored."""
    for rung, drop in enumerate(drops):
        if drop < target:
            return rung
    return None


def event_doses(rows, plan):
    """{(seed, net, new_class): dose rung or None} at the medium length."""
    drops = {}
    for r in rows:
        if r["length"] == plan.lengths[1]:
            drops.setdefault((r["seed"], r["net"], r["new_class"]), {})[r["rung"]] = r["drop"]
    return {k: dose_of([d[g] for g in range(len(LADDER))]) for k, d in drops.items()}


def dose_summary(rows, plan, seeds, res):
    """Censored events sort above the top rung; they are never imputed at it."""
    doses = event_doses(rows, plan)
    condition = {(r["seed"], r["net"], r["new_class"]): r["condition"] for r in rows}
    top = len(LADDER)
    table = {}
    for cond in CONDITIONS:
        ranks = sorted(top if d is None else d for k, d in doses.items() if condition[k] == cond)
        if ranks:
            median = ranks[len(ranks) // 2]
            table[cond] = {"n": len(ranks), "censored_fraction": float(np.mean([x == top for x in ranks])),
                           "median_ratio": "censored (above the top rung)" if median == top else LADDER[median],
                           "mean_rank_lower_bound": float(np.mean(ranks))}

    def fraction(cond):
        out = []
        for s in seeds:
            vals = [d is None for k, d in doses.items() if k[0] == s and condition[k] == cond]
            out.append(np.mean(vals) if vals else np.nan)
        return np.array(out, dtype=np.float64)

    d = fraction("consistent") - fraction("inconsistent")
    low, high = percentile_ci(np.nanmean(d[res], axis=1))
    return {"by_condition": table, "overall_censored_fraction": float(np.mean([v is None for v in doses.values()])),
            "consistent_minus_inconsistent_censored": {"diff": float(np.nanmean(d)), "ci_low": low, "ci_high": high}}


def verdict(r, max_vif, raw_ci, adjusted_ci, new_acc_spread, grew):
    """Preregistered rules, in order. r must hold first, since failing it means stop and report."""
    if not (math.isfinite(r) and r >= FIRST_ORDER_R):
        return {"label": LABEL_FIRST_ORDER, "caveats": []}
    if not max_vif <= VIF_LIMIT:
        return {"label": LABEL_VIF, "caveats": []}
    if not raw_ci[1] < 0.0:
        return {"label": LABEL_NO_RAW, "caveats": []}
    if adjusted_ci[1] < -EQUIVALENCE_BOUND and new_acc_spread <= NEW_ACC_SPREAD:
        return {"label": LABEL_H4A, "caveats": [] if grew else [CAVEAT_NO_GROWTH]}
    if -EQUIVALENCE_BOUND < adjusted_ci[0] and adjusted_ci[1] < EQUIVALENCE_BOUND:
        return {"label": LABEL_H4B, "caveats": []}
    return {"label": LABEL_INCONCLUSIVE, "caveats": []}


# ---- Stage 0 ----
# Stage 0 checks that use drop see no condition labels. Those that use labels see no drop. So the
# raw schema effect cannot be computed in Stage 0, by construction.
OUTCOME_VIEW = ("seed", "net", "new_class", "length", "rung", "drop", "overlap")
LABEL_VIEW = ("seed", "net", "new_class", "length", "rung", "condition", "new_acc", "overlap", "distance", "schema_reached",
              "schema_steps")


def evaluate_stage0(rows, plan):
    outcomes = [{k: r[k] for k in OUTCOME_VIEW} for r in rows]
    labelled = [{k: r[k] for k in LABEL_VIEW} for r in rows]
    short, medium = plan.lengths[0], plan.lengths[1]
    drop0 = float(np.mean([r["drop"] for r in select(outcomes, medium, 0)]))
    drop_top = float(np.mean([r["drop"] for r in select(outcomes, medium, len(LADDER) - 1)]))
    reduction = (drop0 - drop_top) / drop0 if drop0 > 0.0 else 0.0
    means = {}
    for r in labelled:
        if r["length"] == medium:
            means.setdefault((r["condition"], r["rung"]), []).append(r["new_acc"])
    floor = min(float(np.mean(v)) for (_, rung), v in means.items() if rung == 0)
    floor_all_rungs = min(float(np.mean(v)) for v in means.values())
    depth = {}
    for r in labelled:
        depth.setdefault(r["net"], {})[r["seed"]] = r["schema_steps"]
    first = select(outcomes, short)
    r_short = pearson([r["overlap"] for r in first], [r["drop"] for r in first])
    max_vif, vifs = condition_vif(labelled, plan)
    doses = event_doses(outcomes, plan)
    censored = float(np.mean([d is None for d in doses.values()]))
    nets = {(r["seed"], r["net"]): r["schema_reached"] for r in labelled}
    reached = float(np.mean(list(nets.values())))

    def check(measured, threshold, passed, **extra):
        return {"measured": measured, "threshold": threshold, "pass": bool(passed), **extra}

    checks = {
        "positive_control_zero_replay_drop": check(drop0, f">= {POSITIVE_DROP} points", drop0 >= POSITIVE_DROP),
        "positive_control_top_rung_reduction": check(reduction, f">= {POSITIVE_REDUCTION} of the zero-replay drop",
                                                     drop0 > 0.0 and reduction >= POSITIVE_REDUCTION),
        "new_acc_medium_zero_replay_min_over_condition": check(floor, f">= {NEW_ACC_FLOOR}", floor >= NEW_ACC_FLOOR),
        "first_order_r_short": check(r_short, f">= {FIRST_ORDER_R}", math.isfinite(r_short) and r_short >= FIRST_ORDER_R),
        "condition_vif": check(max_vif, f"<= {VIF_LIMIT}", max_vif <= VIF_LIMIT, per_column=vifs),
        "dose_censored_fraction": check(censored, f"< {CENSOR_LIMIT}", censored < CENSOR_LIMIT),
        "schema_criterion_reached_fraction": check(reached, "== 1.0", reached == 1.0, not_preregistered=True),
    }
    audit = sorted((round(r["overlap"], 5), round(r["drop"], 4)) for r in first)
    return {"checks": checks, "overall_pass": all(c["pass"] for c in checks.values()),
            "failed": [k for k, c in checks.items() if not c["pass"]], "audit_overlap_vs_short_drop": audit,
            "diagnostics": {"new_acc_medium_min_over_condition_and_rung": floor_all_rungs,
                            "schema_steps_median_by_net": {n: float(np.median(list(d.values()))) for n, d in depth.items()}}}


class Stage0Refused(Exception):
    pass


def require_stage0(out_dir, plan, datasets=None):
    path = out_dir / "stage0.json"
    if not path.exists():
        raise Stage0Refused(f"{path} does not exist; run --stage0 first")
    record = json.loads(path.read_text())
    if record.get("overall_pass") is not True:
        raise Stage0Refused(f"Stage 0 did not pass: {record.get('failed')}")
    if record.get("design_hash") != design_hash(plan):
        raise Stage0Refused("Stage 0 passed on a different design; rerun --stage0")
    if datasets is not None and record.get("datasets") != datasets:
        raise Stage0Refused("dataset hashes differ from those Stage 0 recorded")
    return record


# ---- Analysis of the full run ----
def analyse(all_rows, plan):
    rows = [r for r in all_rows if r["schema_reached"] == 1]
    seeds = sorted({r["seed"] for r in rows})
    res = seed_resamples(len(seeds), plan.n_bootstrap)
    short, medium, long_ = plan.lengths
    primary, first = select(rows, medium), select(rows, short)
    col = MODEL_COLUMNS.index("consistent")

    r_short = pearson([r["overlap"] for r in first], [r["drop"] for r in first])
    r_inner = pearson([-r["inner"] for r in first], [r["drop"] for r in first])
    max_vif, vifs = condition_vif(rows, plan)
    raw = raw_effect(primary, seeds, res)

    fit = adjusted_model(primary, seeds, res)
    coef = {c: {"estimate": float(fit["beta"][i]), "ci": percentile_ci(fit["boot"][:, i])} for i, c in enumerate(MODEL_COLUMNS)}
    adj_ci = percentile_ci(fit["boot"][:, col])

    fits = {short: adjusted_model(first, seeds, res), long_: adjusted_model(select(rows, long_), seeds, res)}
    # The predicted direction is negative, so growth means beta becomes more negative.
    grow_points = fits[short]["boot"][:, col] - fits[long_]["boot"][:, col]
    grow_sd = fits[short]["boot"][:, col] / fits[short]["sd_drop"] - fits[long_]["boot"][:, col] / fits[long_]["sd_drop"]
    points_ci, sd_ci = percentile_ci(grow_points), percentile_ci(grow_sd)
    grew = points_ci[0] > 0.0 and sd_ci[0] > 0.0

    acc = {c: float(np.mean([r["new_acc"] for r in primary if r["condition"] == c])) for c in CONDITIONS}
    spread = max(acc.values()) - min(acc.values())

    def sensitivity(keep):
        subset = [r for r in primary if keep(r)]
        s = adjusted_model(subset, seeds, res)
        return {"raw": raw_effect(subset, seeds, res), "n_events": len(subset),
                "adjusted_consistent_minus_inconsistent": {"estimate": float(s["beta"][col]), "ci": percentile_ci(s["boot"][:, col])}}

    sized = adjusted_model(primary, seeds, res, size_and_depth(primary))

    means = {}
    for cond in CONDITIONS:
        for length in plan.lengths:
            for rung in range(len(LADDER)):
                part = [r for r in select(rows, length, rung) if r["condition"] == cond]
                for field in ("drop", "new_acc"):
                    m = per_seed_means(part, seeds, field)
                    means.setdefault(cond, {}).setdefault(str(length), {}).setdefault(str(LADDER[rung]), {})[field] = {
                        "mean": float(np.nanmean(m)), "ci": percentile_ci(np.nanmean(m[res], axis=1))}
    schema = {}
    for name, _, _ in NETS:
        nets = {(r["seed"], r["net"]): (r["schema_steps"], r["schema_acc"], r["schema_reached"]) for r in all_rows if r["net"] == name}
        vals = np.array(list(nets.values()), dtype=np.float64)
        schema[name] = {"median_steps": float(np.median(vals[:, 0])), "mean_val_acc": float(vals[:, 1].mean()),
                        "missed_criterion": int((vals[:, 2] == 0).sum()), "n_old": len(schema_classes(dict((n[0], n[1]) for n in NETS)[name]))}

    return {"verdict": verdict(r_short, max_vif, (raw["ci_low"], raw["ci_high"]), adj_ci, spread, grew),
            "first_order_validity": {"r": r_short, "ci": bootstrap_r(first, seeds, res), "threshold": FIRST_ORDER_R,
                                     "pass": math.isfinite(r_short) and r_short >= FIRST_ORDER_R,
                                     "r_inner_product_descriptive": r_inner},
            "vif": {"max_condition": max_vif, "per_column": vifs, "limit": VIF_LIMIT, "pass": max_vif <= VIF_LIMIT},
            "raw_effect": raw,
            "primary_model": {"formula": "drop ~ schema_condition + overlap + distance + (1 | seed)",
                              "length": medium, "replay": 0.0, "reference": "inconsistent", "coefficients": coef,
                              "consistent_minus_inconsistent": {"estimate": coef["consistent"]["estimate"], "ci": adj_ci},
                              "sigma2_residual": fit["sigma2_residual"], "sigma2_seed": fit["sigma2_seed"],
                              "equivalence_bound": EQUIVALENCE_BOUND, "n_events": len(primary), "n_seeds": len(seeds)},
            "higher_order": {"short": {"estimate": float(fits[short]["beta"][col]), "ci": percentile_ci(fits[short]["boot"][:, col])},
                             "long": {"estimate": float(fits[long_]["beta"][col]), "ci": percentile_ci(fits[long_]["boot"][:, col])},
                             "growth_points_ci": points_ci, "growth_sd_units_ci": sd_ci, "grew": grew},
            "new_acc": {"by_condition": acc, "largest_gap": spread, "limit": NEW_ACC_SPREAD, "within": spread <= NEW_ACC_SPREAD},
            "dose": dose_summary(rows, plan, seeds, res),
            "per_condition": means, "schema_phase": schema,
            "excluded_events_rows": len(all_rows) - len(rows),
            "sensitivity_not_for_verdict": {
                "without_upper_withheld_net": sensitivity(lambda r: r["net"] != "I_upper"),
                "without_singleton_new_classes": sensitivity(lambda r: r["new_class"] not in SINGLETONS),
                "with_schema_size_and_depth": {"estimate": float(sized["beta"][col]), "ci": percentile_ci(sized["boot"][:, col])}}}


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=2))
    os.replace(tmp, path)


# ---- Datasets ----
GCS = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
FASHION_ROUTES = (GCS, "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/")
MNIST_ROUTES = (GCS, "https://s3.amazonaws.com/img-datasets/")
# Published by Zalando in the fashion-mnist README; no sha256 is published.
FASHION_MD5 = {"train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
               "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
               "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
               "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310"}
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"  # As pinned in keras/src/datasets/mnist.py


def _download(url):
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def _matches(blob, algo, expected):
    return hashlib.new(algo, blob).hexdigest() == expected


def fetch_verified(name, routes, algo, expected, cache_dir):
    """Return bytes that match the published hash, over HTTPS only. A bad copy is never used or kept."""
    path = cache_dir / name
    if path.exists():
        blob = path.read_bytes()
        if _matches(blob, algo, expected):
            return blob
    errors = []
    for base in routes:
        url = base + name
        if not url.startswith("https://"):
            errors.append(f"{url}: not HTTPS")
            continue
        try:
            blob = _download(url)
        except Exception as exc:  # Every route is tried, then all are reported
            errors.append(f"{url}: {exc!r}")
            continue
        if not _matches(blob, algo, expected):
            errors.append(f"{url}: {algo} mismatch, got {hashlib.new(algo, blob).hexdigest()}")
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(blob)
        os.replace(tmp, path)
        log.info("downloaded %s from %s", name, url)
        return blob
    raise RuntimeError(f"{name} unavailable; refusing other data. " + "; ".join(errors))


def parse_idx(blob, expected):
    data = gzip.decompress(blob)
    magic, count = int.from_bytes(data[:4], "big"), int.from_bytes(data[4:8], "big")
    if count != expected or magic not in (2049, 2051):
        raise ValueError(f"unexpected IDX header: magic {magic}, count {count}, wanted {expected}")
    if magic == 2051:
        return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, 784)
    return np.frombuffer(data, dtype=np.uint8, offset=8).astype(np.int64)


def load_datasets(cache_dir):
    """Return (fashion, mnist, provenance); each of the first two is (x_train, y_train, x_test, y_test)."""
    provenance, parts = {}, {}
    for name, md5 in FASHION_MD5.items():
        blob = fetch_verified(name, FASHION_ROUTES, "md5", md5, cache_dir)
        provenance[f"fashion/{name}"] = hashlib.sha256(blob).hexdigest()
        parts[name] = parse_idx(blob, 60000 if name.startswith("train") else 10000)
    fashion = (parts["train-images-idx3-ubyte.gz"], parts["train-labels-idx1-ubyte.gz"],
               parts["t10k-images-idx3-ubyte.gz"], parts["t10k-labels-idx1-ubyte.gz"])
    blob = fetch_verified("mnist.npz", MNIST_ROUTES, "sha256", MNIST_SHA256, cache_dir)
    provenance["mnist/mnist.npz"] = hashlib.sha256(blob).hexdigest()
    with np.load(io.BytesIO(blob)) as d:
        mnist = (d["x_train"].reshape(-1, 784), d["y_train"].astype(np.int64),
                 d["x_test"].reshape(-1, 784), d["y_test"].astype(np.int64))
    return fashion, mnist, provenance


# ---- Entry points ----
def run_stage0(plan, out_dir, fresh, workers, cache):
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    fashion, mnist, datasets = load_datasets(cache)
    for name, sha in datasets.items():
        log.info("dataset %s sha256=%s verified", name, sha)
    # Per-event rows with labels would let anyone read off the raw effect, so this checkpoint
    # stays out of results/ and is never committed.
    store = RawStore(cache / "h4_stage0_checkpoint.csv", plan, fresh)
    try:
        execute(plan, store, workers, cache, (fashion, mnist), "stage0")
    finally:
        store.close()
    result = evaluate_stage0(store.rows(), plan)
    write_json(out_dir / "stage0.json", {"stage": "stage0", "seeds": plan.seeds, "design_hash": design_hash(plan),
                                         "datasets": datasets, "plan": asdict(plan), "notes": NOTES, **result,
                                         "elapsed_seconds": time.time() - started})
    log.info("stage0 overall_pass=%s failed=%s", result["overall_pass"], result["failed"])
    return result["overall_pass"]


def run_full(plan, out_dir, fresh, workers, cache):
    started = time.time()
    require_stage0(out_dir, plan)  # Before any download
    fashion, mnist, datasets = load_datasets(cache)
    for name, sha in datasets.items():
        log.info("dataset %s sha256=%s verified", name, sha)
    require_stage0(out_dir, plan, datasets)
    store = RawStore(out_dir / "raw_results.csv", plan, fresh)
    try:
        execute(plan, store, workers, cache, (fashion, mnist), "full")
    finally:
        store.close()
    analysis = analyse(store.rows(), plan)
    write_json(out_dir / "summary.json", {"design_hash": design_hash(plan), "datasets": datasets, "plan": asdict(plan),
                                          "notes": NOTES, **analysis, "elapsed_seconds": time.time() - started})
    log.info("verdict: %s", analysis["verdict"])
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage0", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/clu/h4")
    args = parser.parse_args()
    out_dir = args.out or Path(__file__).resolve().parents[1] / "results" / "h4"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.seterr(all="ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler()])
    try:
        if args.stage0:
            return 0 if run_stage0(STAGE0_PLAN, out_dir, args.fresh, args.workers, args.cache) else 2
        run_full(FULL_PLAN, out_dir, args.fresh, args.workers, args.cache)
        return 0
    except Stage0Refused as exc:
        log.error("refusing the full run: %s", exc)
        return 3
    except Exception:
        log.exception("run failed; rerun the same command to resume")
        return 1


if __name__ == "__main__":
    sys.exit(main())
