"""H6: do probes overstate stored memory on Split CIFAR-100?

Spec: hypotheses/H6_cifar_probe_test.md. Calibration never scores the trained task on a later network
with any key but `none`. Self-contained: upload this one file and run `python h6_cifar.py run` on a
GPU machine. It downloads and verifies CIFAR-100, tests itself, runs everything, and zips the results.
Rerunning resumes from the last finished seed.
"""
import argparse
import copy
import hashlib
import json
import pickle
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Fixed by the spec ----
REGIMES = ("cifar_class", "cifar_task")
LAGS = (1, 2, 3, 5)
N_NEW = 4
GRID = tuple((lr, epochs) for lr in (0.01, 0.05) for epochs in (10, 20))
BUDGETS = (2, 5, 10, 20, 50)
CAL_SEEDS, RETRY_SEEDS, CONFIRM_SEEDS = tuple(range(700, 706)), tuple(range(706, 712)), tuple(range(800, 820))
CURVE_SEEDS = 2
MIN_ACC, MAX_S_NONE, MIN_ADVANTAGE = 0.4, 0.5, 0.15
NULL_LIMIT, PLANT_MIN_S, PLANT_MAX_NONE, MATCHED_GAP, MATERIAL = 0.15, 0.8, 0.6, 0.2, 0.05
BEST_KEYS = ("readout", "readout_early", "finetune_low")
N_BOOTSTRAP = 10000
WIDTH, BATCH, KEY_PER_CLASS, RIDGE, TUNE_STEPS, TUNE_RATE, BISECT = 32, 128, 50, 1.0, 30, 0.1, 6
ROOT = 20261105
MIRROR = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-100-python.tar.gz"
CIFAR_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESTORE = {"restore_head": ("head.",), "restore_body": ("conv", "bn", "stage")}
ZERO_SAMPLE = ("none", "mask", *RESTORE)
PLANTED_KEYS = ("none", "readout", "finetune_low")


class Data:
    def __init__(self, x_train, y_train, x_test, y_test, key_per_class=KEY_PER_CLASS):
        self.x_train, self.x_test = x_train, x_test
        self.train, self.key, self.test = {}, {}, {}
        for c in range(int(y_train.max()) + 1):
            idx = (y_train == c).nonzero().flatten()
            self.key[c], self.train[c] = idx[-key_per_class:], idx[:-key_per_class]
            self.test[c] = (y_test == c).nonzero().flatten()


def load_cifar(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "cifar-100-python.tar.gz"
    md5 = lambda: hashlib.md5(path.read_bytes()).hexdigest()
    if not path.exists() or md5() != CIFAR_MD5:
        print("downloading CIFAR-100", flush=True)
        urllib.request.urlretrieve(MIRROR, path)
    if md5() != CIFAR_MD5:
        raise SystemExit("CIFAR-100 download failed its checksum")
    mean = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
    std = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)
    with tarfile.open(path) as tar:
        def part(name):
            d = pickle.load(tar.extractfile(f"cifar-100-python/{name}"), encoding="bytes")
            x = torch.from_numpy(d[b"data"].reshape(-1, 3, 32, 32)).float() / 255.0
            return ((x - mean) / std).to(DEVICE), torch.tensor(d[b"fine_labels"], device=DEVICE)
        data = Data(*part("train"), *part("test"))
    assert len(data.x_train) == 50000 and len(data.x_test) == 10000
    assert all(len(data.train[c]) == 450 and len(data.key[c]) == 50 and len(data.test[c]) == 100 for c in range(100))
    return data


class Block(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.a, self.ba = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out)
        self.b, self.bb = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out)
        self.skip = nn.Identity() if stride == 1 and c_in == c_out else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return F.relu(self.bb(self.b(F.relu(self.ba(self.a(x))))) + self.skip(x))


class Net(nn.Module):
    """ResNet-18 for 32 by 32 images."""

    def __init__(self, width=WIDTH, outputs=100):
        super().__init__()
        self.conv, self.bn = nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width)
        def stage(c_in, c_out, stride):
            return nn.Sequential(Block(c_in, c_out, stride), Block(c_out, c_out, 1))
        self.stage1, self.stage2 = stage(width, width, 1), stage(width, 2 * width, 2)
        self.stage3, self.stage4 = stage(2 * width, 4 * width, 2), stage(4 * width, 8 * width, 2)
        self.head = nn.Linear(8 * width, outputs)

    def features(self, x):
        h = self.stage2(self.stage1(F.relu(self.bn(self.conv(x)))))
        third = self.stage3(h)
        return F.adaptive_avg_pool2d(self.stage4(third), 1).flatten(1), F.adaptive_avg_pool2d(third, 1).flatten(1)

    def forward(self, x):
        return self.head(self.features(x)[0])


class Task:
    def __init__(self, classes):
        self.classes = torch.as_tensor(classes, device=DEVICE)


def make_tasks(seed, per_task=10):
    order = np.random.default_rng([ROOT, seed, 1]).permutation(100)
    return [Task(order[per_task * i:per_task * i + per_task]) for i in range(10)]


def task_pool(data, task, images, index):
    ids = torch.cat([index[int(c)] for c in task.classes])
    labels = torch.cat([torch.full((len(index[int(c)]),), i, device=DEVICE) for i, c in enumerate(task.classes)])
    return images[ids], labels


def key_samples(data, task, task_id, n, seed):
    rng = np.random.default_rng([ROOT, seed, 7, task_id, n])
    ids = [data.key[int(c)][torch.as_tensor(rng.choice(len(data.key[int(c)]), n, replace=False), device=DEVICE)] for c in task.classes]
    labels = torch.cat([torch.full((n,), i, device=DEVICE) for i in range(len(ids))])
    return data.x_train[torch.cat(ids)], labels


@torch.no_grad()
def forward_all(net, x, fn=None, size=1000):
    net.eval()
    return torch.cat([(fn or net)(x[i:i + size]) for i in range(0, len(x), size)])


def score(net, x, y, active, task):
    pred = active[forward_all(net, x)[:, active].argmax(1)]
    return float((pred == task.classes[y]).float().mean())


def train_task(net, task, data, active, lr, epochs, gen):
    x, y = task_pool(data, task, data.x_train, data.train)
    lookup = torch.full((net.head.out_features,), -1, device=DEVICE)
    lookup[active] = torch.arange(len(active), device=DEVICE)
    target = lookup[task.classes][y]
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    net.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=gen, device=DEVICE)
        for i in range(0, len(x) - BATCH + 1, BATCH):
            ids = order[i:i + BATCH]
            batch = x[ids]
            shift = torch.randint(-4, 5, (2,), generator=gen, device=DEVICE).tolist()
            batch = torch.roll(batch, shift, dims=(2, 3))
            if torch.rand(1, generator=gen, device=DEVICE) < 0.5:
                batch = batch.flip(3)
            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                loss = F.cross_entropy(net(batch)[:, active], target[ids])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()


# ---- Keys ----
def ridge_score(net, x_key, y_key, x_eval, y_eval, classes, layer):
    pick = lambda batch: net.features(batch)[layer]
    f, test = forward_all(net, x_key, pick).double(), forward_all(net, x_eval, pick).double()
    mean, std = f.mean(0), f.std(0) + 1e-6
    f, test = (f - mean) / std, (test - mean) / std
    target = F.one_hot(y_key, classes).double()
    offset = target.mean(0)
    beta = torch.linalg.solve(f.T @ f + RIDGE * torch.eye(f.shape[1], device=DEVICE, dtype=f.dtype), f.T @ (target - offset))
    return float(((test @ beta + offset).argmax(1) == y_eval).float().mean())


def bn_recal_score(net, x_key, x_eval, y_eval, task):
    tuned = copy.deepcopy(net)
    for m in tuned.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    tuned.train()
    with torch.no_grad():
        for i in range(0, len(x_key), BATCH):
            tuned(x_key[i:i + BATCH])
    return score(tuned, x_eval, y_eval, task.classes, task)


def finetune_score(net, x_key, y_key, x_eval, y_eval, task, lr, seed):
    tuned = copy.deepcopy(net)
    tuned.eval()
    opt = torch.optim.SGD(tuned.parameters(), lr=TUNE_RATE * lr)
    gen = torch.Generator(DEVICE).manual_seed(seed)
    for _ in range(TUNE_STEPS):
        ids = torch.randperm(len(y_key), generator=gen, device=DEVICE)[:BATCH]
        loss = F.cross_entropy(tuned(x_key[ids])[:, task.classes], y_key[ids])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return score(tuned, x_eval, y_eval, task.classes, task)


def restored(net, w0, key):
    tuned = copy.deepcopy(net)
    state = tuned.state_dict()
    for name, value in w0.items():
        if name.startswith(RESTORE[key]):
            state[name] = value
    tuned.load_state_dict(state)
    return tuned


def keys_for(regime):
    base = ("none", "bn_recal", "readout", "readout_early", "finetune_low", *RESTORE)
    return base if regime == "cifar_task" else ("mask", *base)


def probe(net, task, task_id, data, key, n, seen, lr, seed, regime, w0):
    """Accuracy on one task after one key. Key samples depend on seed, task and n, not on the network."""
    x_eval, y_eval = task_pool(data, task, data.x_test, data.test)
    if key == "none":
        active = task.classes if regime == "cifar_task" else torch.unique(torch.cat([seen, task.classes]))
        return score(net, x_eval, y_eval, active, task)
    if key == "mask":
        return score(net, x_eval, y_eval, task.classes, task)
    if key in RESTORE:
        return score(restored(net, w0, key), x_eval, y_eval, task.classes, task)
    x_key, y_key = key_samples(data, task, task_id, n, seed)
    if key == "bn_recal":
        return bn_recal_score(net, x_key, x_eval, y_eval, task)
    if key in ("readout", "readout_early"):
        return ridge_score(net, x_key, y_key, x_eval, y_eval, len(task.classes), 0 if key == "readout" else 1)
    return finetune_score(net, x_key, y_key, x_eval, y_eval, task, lr, seed)


# ---- Planted truth ----
def planted_access(net, fraction, seed):
    """Shuffle a fraction of penultimate units before the head. Nothing is lost; the route is broken."""
    rng, tuned = np.random.default_rng(seed), copy.deepcopy(net)
    order = rng.permutation(tuned.head.in_features)
    chosen = torch.as_tensor(order[:int(round(fraction * len(order)))], device=DEVICE)
    if len(chosen) > 1:
        with torch.no_grad():
            tuned.head.weight[:, chosen] = tuned.head.weight[:, chosen.roll(1)]
    return tuned


def planted_erasure(net, fraction, seed):
    """Redraw a fraction of all weights from init. Those values are gone."""
    torch.manual_seed(seed)
    fresh = Net(net.conv.out_channels, net.head.out_features).to(DEVICE)
    tuned = copy.deepcopy(net)
    with torch.no_grad():
        for a, b in zip(tuned.parameters(), fresh.parameters()):
            mask = torch.rand_like(a) < fraction
            a[mask] = b[mask]
    return tuned


def match_fraction(build, target, s_none):
    low, high, best, gap = 0.0, 1.0, 1.0, float("inf")
    for _ in range(BISECT):
        mid = 0.5 * (low + high)
        value = s_none(build(mid))
        if abs(value - target) < gap:
            best, gap = mid, abs(value - target)
        low, high = (mid, high) if value > target else (low, mid)
    return best


# ---- One seed ----
def seen_after(tasks, regime, t):
    return torch.cat([x.classes for x in tasks[:t + 1]])


def active_for(tasks, regime, t):
    return tasks[t].classes if regime == "cifar_task" else seen_after(tasks, regime, t)


def fresh_net(seed, width):
    torch.manual_seed(ROOT + seed)
    return Net(width).to(DEVICE)


def named_tasks(tasks):
    return {"A": (tasks[0], 0), **{f"C{i}": (tasks[6 + i], 1000 + i) for i in range(N_NEW)}}


def run_curve(data, regime, seed, lr, epochs, width=WIDTH):
    """Outcome-free curve for choosing a config: no key, on A and on new tasks."""
    tasks = make_tasks(seed)
    named, net, out = named_tasks(tasks), fresh_net(seed, width), {}
    gen = torch.Generator(DEVICE).manual_seed(ROOT + seed)
    for t in range(max(LAGS) + 1):
        train_task(net, tasks[t], data, active_for(tasks, regime, t), lr, epochs, gen)
        seen = seen_after(tasks, regime, t)
        if t == 0:
            out["w0_acc"] = probe(net, tasks[0], 0, data, "mask", 0, seen, lr, seed, regime, None)
        if t == 0 or t in LAGS:
            new = [probe(net, *named[f"C{i}"][:2], data, "none", 0, seen, lr, seed, regime, None) for i in range(N_NEW)]
            out[f"none_A_{t}"], out[f"none_C_{t}"] = probe(net, tasks[0], 0, data, "none", 0, seen, lr, seed, regime, None), float(np.mean(new))
    out["final_acc"] = probe(net, tasks[max(LAGS)], max(LAGS), data, "mask", 0, seen, lr, seed, regime, None)
    return out


def run_seed(data, regime, seed, lr, epochs, m_star, budgets, blind, width=WIDTH):
    """Return rows of [net, lag, task, key, n, acc]. If blind, A on a later network gets only `none`."""
    lags = tuple(m for m in LAGS if not blind or m <= m_star)
    tasks = make_tasks(seed)
    named, net, rows, keys, w0_state = named_tasks(tasks), fresh_net(seed, width), [], keys_for(regime), None
    gen = torch.Generator(DEVICE).manual_seed(ROOT + seed)
    new_names = tuple(f"C{i}" for i in range(N_NEW))

    def measure(which, lag, candidate, seen, names, allowed):
        for name in names:
            task, task_id = named[name]
            for key in allowed:
                for n in (0,) if key in ZERO_SAMPLE else budgets:
                    rows.append([which, lag, name, key, n, probe(candidate, task, task_id, data, key, n, seen, lr, seed, regime, w0_state)])

    for t in range(max(lags) + 1):
        train_task(net, tasks[t], data, active_for(tasks, regime, t), lr, epochs, gen)
        seen = seen_after(tasks, regime, t)
        if t == 0:
            w0, seen0 = copy.deepcopy(net), seen
            w0_state = {k: v.clone() for k, v in net.state_dict().items()}
            measure("real", 0, net, seen, ("A", *new_names), keys)
        elif t in lags:
            measure("real", t, net, seen, new_names, keys)
            measure("real", t, net, seen, ("A",), ("none",) if blind else keys)

    table = {(r[1], r[2]): r[5] for r in rows if r[0] == "real" and r[3] == "none"}
    mean_new = lambda lag: float(np.mean([table[(lag, c)] for c in new_names]))
    denominator = (table[(0, "A")] - mean_new(0)) or 1e-9
    target = (table[(m_star, "A")] - mean_new(m_star)) / denominator

    def s_none(candidate):
        a = probe(candidate, *named["A"][:2], data, "none", 0, seen0, lr, seed, regime, None)
        new = [probe(candidate, *named[c][:2], data, "none", 0, seen0, lr, seed, regime, None) for c in new_names]
        return (a - float(np.mean(new))) / denominator

    names = ("A", *new_names)
    plant_seed, erase_seed = ROOT + seed + 1, ROOT + seed + 2
    shuffled = match_fraction(lambda f: planted_access(w0, f, plant_seed), target, s_none)
    erased = match_fraction(lambda f: planted_erasure(w0, f, erase_seed), target, s_none)
    measure("plant_access", 0, planted_access(w0, shuffled, plant_seed), seen0, names, PLANTED_KEYS)
    measure("erase_matched", 0, planted_erasure(w0, erased, erase_seed), seen0, names, PLANTED_KEYS)
    measure("erase_total", 0, planted_erasure(w0, 2.0, erase_seed), seen0, names, PLANTED_KEYS)
    return rows


# ---- Analysis ----
def index_rows(by_seed):
    return {(seed, *row[:5]): row[5] for seed, rows in by_seed.items() for row in rows}


def column(acc, seeds, which, lag, task, key, n):
    n = 0 if key in ZERO_SAMPLE else n
    return np.array([acc[(s, which, lag, task, key, n)] for s in seeds])


def new_task(acc, seeds, which, lag, key, n):
    return np.mean([column(acc, seeds, which, lag, f"C{i}", key, n) for i in range(N_NEW)], axis=0)


def starting_advantage(acc, seeds, key, n):
    return column(acc, seeds, "real", 0, "A", key, n) - new_task(acc, seeds, "real", 0, key, n)


def surviving(acc, seeds, lag, key, n, which="real", fixed=False):
    base_lag, base_net = (0, "real") if fixed else (lag, which)
    numerator = column(acc, seeds, which, lag, "A", key, n) - new_task(acc, seeds, base_net, base_lag, key, n)
    return numerator, starting_advantage(acc, seeds, key, n)


def sham(acc, seeds, lag, key, n):
    half = N_NEW // 2
    first = np.mean([column(acc, seeds, "real", lag, f"C{i}", key, n) for i in range(half)], axis=0)
    second = np.mean([column(acc, seeds, "real", lag, f"C{i}", key, n) for i in range(half, N_NEW)], axis=0)
    return first - second, starting_advantage(acc, seeds, key, n)


def ratio(numerator, denominator):
    return float(numerator.mean() / denominator.mean())


def resample_counts(n_seeds, n_bootstrap):
    rng = np.random.default_rng([ROOT, 99])
    return rng.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=n_bootstrap).astype(np.float64)


def ci(samples):
    low, high = np.percentile(samples, [2.5, 97.5])
    return [float(low), float(high)]


def boot_ratio(counts, pair):
    return (counts @ pair[0]) / (counts @ pair[1])


def split_label(access_ci, recoverable_ci):
    if access_ci[0] > 0.5:
        return "mostly access"
    if access_ci[1] < 0.5 and recoverable_ci[0] > 0.5:
        return "mostly repair"
    if recoverable_ci[1] < 0.5:
        return "mostly unrecovered"
    return "no majority"


def probe_label(gap_ci):
    if gap_ci[0] > MATERIAL:
        return "probes overstate"
    if gap_ci[1] < -MATERIAL:
        return "probes understate"
    if gap_ci[0] >= -MATERIAL and gap_ci[1] <= MATERIAL:
        return "no overstatement"
    return "inconclusive"


def decompose(acc, seeds, lag, n, counts, fixed=False):
    none = surviving(acc, seeds, lag, "none", 0, fixed=fixed)
    read = surviving(acc, seeds, lag, "readout", n, fixed=fixed)
    pairs = {k: surviving(acc, seeds, lag, k, n, fixed=fixed) for k in BEST_KEYS}
    points = {k: ratio(*v) for k, v in pairs.items()}
    s_none, s_read, top = ratio(*none), ratio(*read), max(points, key=points.get)
    lost = max(1.0 - s_none, 1e-9)
    boot_none = boot_ratio(counts, none)
    boot_lost = np.maximum(1.0 - boot_none, 1e-9)
    boot_best = np.max([boot_ratio(counts, v) for v in pairs.values()], axis=0)
    access_ci = ci((boot_ratio(counts, read) - boot_none) / boot_lost)
    recoverable_ci = ci((boot_best - boot_none) / boot_lost)
    return {"S_none": s_none, "S_readout": s_read, "S": points, "top_key": top, "access": (s_read - s_none) / lost, "access_ci": access_ci,
            "recoverable": (points[top] - s_none) / lost, "recoverable_ci": recoverable_ci, "unrecovered": 1.0 - (points[top] - s_none) / lost,
            "label": split_label(access_ci, recoverable_ci)}


def probe_test(acc, seeds, lag, n, counts):
    now, then = column(acc, seeds, "real", lag, "A", "readout", n), column(acc, seeds, "real", 0, "A", "readout", n)
    same, fixed = surviving(acc, seeds, lag, "readout", n), surviving(acc, seeds, lag, "readout", n, fixed=True)
    naive, gap = now.mean() / then.mean(), (counts @ now) / (counts @ then) - boot_ratio(counts, same)
    return {"naive_retention": float(naive), "naive_ci": ci((counts @ now) / (counts @ then)), "S_readout": ratio(*same), "gap": float(naive - ratio(*same)),
            "gap_ci": ci(gap), "S_readout_fixed": ratio(*fixed), "gap_fixed": float(naive - ratio(*fixed)), "label": probe_label(ci(gap)),
            "acc_A_w0": float(then.mean()), "acc_A_later": float(now.mean()), "acc_new_w0": float(new_task(acc, seeds, "real", 0, "readout", n).mean()),
            "acc_new_later": float(new_task(acc, seeds, "real", lag, "readout", n).mean())}


def evaluate_gates(acc, seeds, m_star, n):
    planted = lambda which, key: ratio(*surviving(acc, seeds, 0, key, n, which=which))
    checks = {}
    for key in ("readout", "finetune_low"):
        for name, value in ((f"sham_{key}", ratio(*sham(acc, seeds, m_star, key, n))), (f"total_erasure_{key}", planted("erase_total", key))):
            checks[name] = {"measured": value, "limit": f"abs <= {NULL_LIMIT}", "pass": abs(value) <= NULL_LIMIT}
    plant, none = planted("plant_access", "readout"), planted("plant_access", "none")
    checks["planted_access_readout"] = {"measured": plant, "limit": f">= {PLANT_MIN_S}", "pass": plant >= PLANT_MIN_S}
    checks["planted_access_none"] = {"measured": none, "limit": f"<= {PLANT_MAX_NONE}", "pass": none <= PLANT_MAX_NONE}
    matched = planted("erase_matched", "readout")
    checks["matched_gap"] = {"measured": plant - matched, "limit": f">= {MATCHED_GAP}", "pass": plant - matched >= MATCHED_GAP,
                             "matched_S_readout": matched, "matched_S_none": planted("erase_matched", "none")}
    return checks


def choose_config(curves):
    best = None
    for (lr, epochs), per in curves.items():
        mean = lambda k: float(np.mean([c[k] for c in per]))
        if mean("w0_acc") < MIN_ACC or mean("final_acc") < MIN_ACC:
            continue
        denominator = mean("none_A_0") - mean("none_C_0")
        for lag in LAGS:
            value = (mean(f"none_A_{lag}") - mean(f"none_C_{lag}")) / denominator
            if value <= MAX_S_NONE:
                rank = (lag, -mean("w0_acc"), lr, epochs)
                if best is None or rank < best[0]:
                    best = (rank, {"lr": lr, "epochs": epochs, "m_star": lag, "S_none": value, "w0_acc": mean("w0_acc"), "final_acc": mean("final_acc")})
                break
    return None if best is None else best[1]


def choose_budget(acc, seeds):
    table = {}
    for n in BUDGETS:
        table[n] = {k: float(starting_advantage(acc, seeds, k, n).mean()) for k in ("readout", "finetune_low")}
        if min(table[n].values()) >= MIN_ADVANTAGE:
            return n, table
    return None, table


# ---- Entry points ----
class Store:
    """One JSON line per finished job, so a dropped runtime resumes where it stopped."""

    def __init__(self, path):
        self.path, self.done = Path(path), {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                item = json.loads(line)
                self.done[item["job"]] = item["value"]

    def get(self, job, compute):
        if job not in self.done:
            start = time.time()
            value = compute()
            self.done[job] = value
            with open(self.path, "a") as f:
                f.write(json.dumps({"job": job, "value": value}) + "\n")
            print(time.strftime("%H:%M:%S"), job, f"{time.time() - start:.0f}s", flush=True)
        return self.done[job]


def calibrate(data, out, seeds):
    store, path = Store(out / "calibration_jobs.jsonl"), out / "calibration.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    for regime in REGIMES:
        if result.get(regime, {}).get("answerable"):
            continue
        curves = {(lr, epochs): [store.get(f"curve|{regime}|{lr}|{epochs}|{s}", lambda s=s: run_curve(data, regime, s, lr, epochs)) for s in seeds[:CURVE_SEEDS]] for lr, epochs in GRID}
        config = choose_config(curves)
        entry = result[regime] = {"seeds": list(seeds), "config": config, "answerable": False,
                                  "grid": {f"{k[0]}/{k[1]}": {m: float(np.mean([c[m] for c in v])) for m in v[0]} for k, v in curves.items()}}
        if not config:
            entry["reason"] = "no config met the accuracy and forgetting rule"
        else:
            rows = {s: store.get(f"blind|{regime}|{s}", lambda s=s: run_seed(data, regime, s, config["lr"], config["epochs"], config["m_star"], BUDGETS, True)) for s in seeds}
            acc = index_rows(rows)
            n, table = choose_budget(acc, seeds)
            entry["advantage_w0"] = table
            if n is None:
                entry["reason"] = "no budget gave a large enough advantage on w_0"
            else:
                checks = evaluate_gates(acc, seeds, config["m_star"], n)
                entry.update(budget=n, checks=checks, answerable=all(c["pass"] for c in checks.values()))
                if not entry["answerable"]:
                    entry["reason"] = "failed: " + ", ".join(k for k, c in checks.items() if not c["pass"])
        path.write_text(json.dumps(result, indent=2))
    return result


def confirm(data, out, calibration):
    store, seeds, counts, summary, lines = Store(out / "confirm_jobs.jsonl"), list(CONFIRM_SEEDS), resample_counts(len(CONFIRM_SEEDS), N_BOOTSTRAP), {}, []
    for regime, entry in calibration.items():
        if not entry["answerable"]:
            lines.append(f"{regime}: unanswerable: {entry.get('reason')}")
            continue
        c, n = entry["config"], entry["budget"]
        budgets = (n,)
        rows = {s: store.get(f"confirm|{regime}|{s}", lambda s=s: run_seed(data, regime, s, c["lr"], c["epochs"], c["m_star"], budgets, False)) for s in seeds}
        acc, m_star = index_rows(rows), c["m_star"]
        checks = evaluate_gates(acc, seeds, m_star, n)
        passed = all(k["pass"] for k in checks.values())
        same, fixed = decompose(acc, seeds, m_star, n, counts), decompose(acc, seeds, m_star, n, counts, fixed=True)
        test = probe_test(acc, seeds, m_star, n, counts)
        split = same["label"] if same["label"] == fixed["label"] else f"baseline-dependent: {same['label']}, fixed {fixed['label']}"
        summary[regime] = {"config": c, "budget": n, "probe_verdict": test["label"] if passed else "void", "split_verdict": split if passed else "void", "gates_passed": passed,
                           "checks": checks, "probe_test": test, "same": same, "fixed": fixed,
                           "by_lag": {f"{lag}/{b}": {"same": decompose(acc, seeds, lag, b, counts), "fixed": decompose(acc, seeds, lag, b, counts, fixed=True), "probe": probe_test(acc, seeds, lag, b, counts),
                                                     "S_keys": {k: ratio(*surviving(acc, seeds, lag, k, b)) for k in keys_for(regime)}} for lag in LAGS for b in budgets}}
        lines += [f"## {regime}: lr {c['lr']}, epochs {c['epochs']}, m* {m_star}, n {n}, w0 acc {c['w0_acc']:.3f}, gates passed {passed}",
                  f"H6a {summary[regime]['probe_verdict']}: naive {test['naive_retention']:.3f}, S_readout {test['S_readout']:.3f}, gap {test['gap']:.3f} {test['gap_ci']}, fixed S {test['S_readout_fixed']:.3f}",
                  f"readout acc A {test['acc_A_w0']:.3f} to {test['acc_A_later']:.3f}, new {test['acc_new_w0']:.3f} to {test['acc_new_later']:.3f}",
                  f"H6b {summary[regime]['split_verdict']}: access {same['access']:.2f} {same['access_ci']}, recoverable {same['recoverable']:.2f} {same['recoverable_ci']}; fixed access {fixed['access']:.2f}, recoverable {fixed['recoverable']:.2f}",
                  "gates: " + ", ".join(f"{k} {v['measured']:.3f}" for k, v in checks.items()),
                  "lag/budget | S_none | access same, fixed | unrecovered same, fixed | top | naive, gap | S by key"]
        for name, v in summary[regime]["by_lag"].items():
            lines.append(f"{name} | {v['same']['S_none']:.2f} | {v['same']['access']:.2f}, {v['fixed']['access']:.2f} | {v['same']['unrecovered']:.2f}, {v['fixed']['unrecovered']:.2f} | {v['same']['top_key']} | "
                         f"{v['probe']['naive_retention']:.2f}, {v['probe']['gap']:.2f} | " + " ".join(f"{k}={x:.2f}" for k, x in v["S_keys"].items()))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def selftest():
    global BATCH, BISECT
    saved = BATCH, BISECT
    BATCH, BISECT = 32, 1
    g = torch.Generator(DEVICE).manual_seed(0)
    centres = torch.randn(100, 3, 32, 32, device=DEVICE, generator=g)
    def draw(per):
        y = torch.arange(100, device=DEVICE).repeat_interleave(per)
        return centres[y] + 0.5 * torch.randn(len(y), 3, 32, 32, device=DEVICE, generator=g), y
    data = Data(*draw(20), *draw(5), key_per_class=8)
    tasks = make_tasks(1)
    assert len({int(c) for t in tasks for c in t.classes}) == 100
    assert not set(data.key[0].tolist()) & set(data.train[0].tolist())
    assert probe_label([0.06, 0.2]) == "probes overstate" and probe_label([-0.04, 0.04]) == "no overstatement"
    assert probe_label([-0.2, -0.06]) == "probes understate" and probe_label([0.0, 0.2]) == "inconclusive"
    assert split_label([0.6, 0.9], [0.7, 1]) == "mostly access" and split_label([0.1, 0.4], [0.6, 0.8]) == "mostly repair"
    assert split_label([0, 0.1], [0.1, 0.4]) == "mostly unrecovered" and split_label([0.3, 0.7], [0.4, 0.9]) == "no majority"
    for regime in REGIMES:
        blind = run_seed(data, regime, 1, 0.01, 1, 1, (2,), True, width=4)
        later = [r for r in blind if r[0] == "real" and r[1] > 0 and r[2] == "A"]
        assert later and all(r[3] == "none" for r in later)
        acc = index_rows({1: blind})
        evaluate_gates(acc, [1], 1, 2)
        full = run_seed(data, regime, 1, 0.01, 1, 1, (2,), False, width=4)
        table = {tuple(r[:5]): r[5] for r in full}
        assert {r[3] for r in full if r[0] == "real" and r[1] > 0 and r[2] == "A"} == set(keys_for(regime))
        assert abs(table[("plant_access", 0, "A", "readout", 2)] - table[("real", 0, "A", "readout", 2)]) < 1e-6
        assert {r[0] for r in full} == {"real", "plant_access", "erase_matched", "erase_total"}
        assert table[("real", 0, "A", "restore_head", 0)] == table[("real", 0, "A", "restore_body", 0)]
    net, w0 = Net(4).to(DEVICE), {k: v.clone() for k, v in Net(4).to(DEVICE).state_dict().items()}
    back = restored(net, w0, "restore_head").state_dict()
    assert torch.equal(back["head.weight"], w0["head.weight"]) and torch.equal(back["conv.weight"], net.state_dict()["conv.weight"])
    back = restored(net, w0, "restore_body").state_dict()
    assert torch.equal(back["conv.weight"], w0["conv.weight"]) and torch.equal(back["head.weight"], net.state_dict()["head.weight"])
    assert not torch.equal(planted_erasure(net, 2.0, 5).conv.weight, net.conv.weight)
    assert torch.equal(planted_erasure(net, 0.0, 5).conv.weight, net.conv.weight)
    BATCH, BISECT = saved
    print("selftest passed", flush=True)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("selftest", "run", "retry"))
    parser.add_argument("--out", default=str(base / "h6_results"))
    parser.add_argument("--data", default=str(base / "cifar"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No GPU found: this experiment needs one, so nothing was run")
    print("torch", torch.__version__, "GPU", torch.cuda.get_device_name(0), flush=True)
    selftest()
    if args.mode == "selftest":
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    data = load_cifar(args.data)
    calibration = calibrate(data, out, RETRY_SEEDS if args.mode == "retry" else CAL_SEEDS)
    print("calibrated", {r: e["answerable"] for r, e in calibration.items()}, flush=True)
    confirm(data, out, calibration)
    archive = shutil.make_archive(str(out), "zip", out)
    print((out / "summary.md").read_text(), flush=True)
    print(f"finished in {(time.time() - start) / 60:.0f} minutes. Send back: {archive}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
