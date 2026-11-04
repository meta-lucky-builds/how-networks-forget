"""Tests for experiments/h5_access.py, on small synthetic data. Run on GitHub Actions only."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import h5_access as h5  # noqa: E402


def synthetic(classes=20, per_class=80, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.integers(0, 255, size=(classes, 784))

    def draw(n):
        y = np.repeat(np.arange(classes), n)
        return np.clip(centres[y] + rng.normal(0, 40, (len(y), 784)), 0, 255).astype(np.uint8), y

    x_train, y_train = draw(per_class)
    x_test, y_test = draw(20)
    return h5.Data(x_train, y_train, x_test, y_test, key_per_class=30, eval_per_class=20)


def small_net(outputs=6, seed=1):
    return h5.init_weights(np.random.default_rng(seed), outputs, inputs=784, hidden=8)


def test_gradients_match_finite_differences():
    net = [a.astype(np.float64) for a in small_net()]
    rng = np.random.default_rng(2)
    x, y, active = rng.random((5, 784)), rng.integers(3, size=5), np.array([1, 3, 4])
    gains = (np.full(8, 1.3), np.full(8, 0.7))

    def loss(weights, g):
        z = h5.logits(weights, x, g)[:, active]
        z = z - z.max(1, keepdims=True)
        return float(-(z[np.arange(5), y] - np.log(np.exp(z).sum(1))).mean())

    weight, gain = h5.grads(net, x, y, active, gains)
    eps = 1e-6
    for array, grad in zip(net, weight):
        for index in [(0,) * array.ndim, tuple(s - 1 for s in array.shape)]:
            if array.ndim == 2 and array is net[4]:
                index = (1, 2)
            elif array is net[5]:
                index = (1,)
            old = array[index]
            array[index] = old + eps
            up = loss(net, gains)
            array[index] = old - eps
            down = loss(net, gains)
            array[index] = old
            assert abs((up - down) / (2 * eps) - grad[index]) < 1e-4
    for which in (0, 1):
        g = [gains[0].copy(), gains[1].copy()]
        g[which][3] += eps
        up = loss(net, g)
        g[which][3] -= 2 * eps
        down = loss(net, g)
        assert abs((up - down) / (2 * eps) - gain[which][3]) < 1e-4


def test_inactive_outputs_get_no_gradient():
    net = small_net()
    rng = np.random.default_rng(3)
    weight, _ = h5.grads(net, rng.random((4, 784), dtype=np.float32), np.zeros(4, int), np.array([0, 1]))
    assert np.all(weight[4][2:] == 0) and np.all(weight[5][2:] == 0)


def test_tasks_are_disjoint_and_seed_dependent():
    tasks = h5.make_tasks("split_class", 5, 4)
    used = np.concatenate([t.classes for t in tasks])
    assert len(tasks) == 6 and len(set(used.tolist())) == 18
    assert not np.array_equal(tasks[0].classes, h5.make_tasks("split_class", 6, 4)[0].classes)
    perm = h5.make_tasks("perm_class", 5, 3)
    assert len(perm) == 5 and perm[1].outputs[0] == 10 and sorted(perm[0].perm.tolist()) == list(range(784))
    assert np.array_equal(h5.make_tasks("perm_shared", 5, 3)[2].outputs, np.arange(10))


def test_planted_access_cannot_change_readout():
    data = synthetic()
    task = h5.Task(np.arange(3), np.arange(3))
    net = small_net(outputs=20)
    h5.train_task(net, task, data, np.arange(3), 0.05, 50, np.random.default_rng(0))
    scrambled = h5.planted_access(net, 1.0, 7)
    assert not np.array_equal(scrambled[4], net[4])
    for a, b in zip(scrambled[:4], net[:4]):
        assert np.array_equal(a, b)
    same = [h5.probe(n, task, 0, data, "readout", 5, np.arange(3), 0.05, 9) for n in (net, scrambled)]
    assert same[0] == same[1]


def test_planted_erasure_fractions():
    net = small_net()
    assert all(np.array_equal(a, b) for a, b in zip(h5.planted_erasure(net, 0.0, 99), net))
    total = h5.planted_erasure(net, 2.0, 99)
    assert not np.any(total[0] == net[0])
    partial = h5.planted_erasure(net, 0.5, 99)
    assert 0.3 < (partial[0] == net[0]).mean() < 0.7


def test_match_fraction_finds_target():
    assert abs(h5.match_fraction(lambda f: f, 0.3, lambda f: 1.0 - f) - 0.7) < 0.02


def test_key_samples_do_not_depend_on_network():
    data, task = synthetic(), h5.Task(np.arange(3), np.arange(3))
    first = task.pool(data.x_train, data.key, 5, h5.rng_for(1, h5.K_KEY, 0, 5))
    second = task.pool(data.x_train, data.key, 5, h5.rng_for(1, h5.K_KEY, 0, 5))
    assert np.array_equal(first[0], second[0])


def test_pools_are_disjoint():
    data = synthetic()
    assert not set(data.key[0].tolist()) & set(data.train[0].tolist())


def run_small(blind, monkeypatch):
    monkeypatch.setitem(h5.LAGS, "split_class", (1, 2))
    monkeypatch.setattr(h5, "HIDDEN", 8)
    monkeypatch.setattr(h5, "FINETUNE_STEPS", 2)
    monkeypatch.setattr(h5, "GAIN_STEPS", 2)
    monkeypatch.setattr(h5, "BISECT_STEPS", 1)
    monkeypatch.setattr(h5.init_weights, "__defaults__", (784, 8))
    return h5.run_seed(synthetic(), "split_class", 1, 0.05, 5, 1, (5,), blind)


def test_blind_never_scores_a_later_network_on_the_trained_task(monkeypatch):
    rows = run_small(True, monkeypatch)
    later = [r for r in rows if r[0] == "real" and r[1] > 0 and r[2] == "A"]
    assert later and all(r[3] == "none" for r in later)


def test_gates_run_on_blind_rows(monkeypatch):
    rows = run_small(True, monkeypatch)
    checks = h5.evaluate_gates(h5.index_rows({1: rows}), [1], 1, 5)
    assert "sham_null_readout" in checks and "matched_erasure_gap" in checks


def test_restore_puts_back_only_the_named_arrays():
    now, w0 = small_net(seed=1), small_net(seed=2)
    head = h5.restored(now, w0, "restore_head")
    assert head[4] is not w0[4] and np.array_equal(head[4], w0[4]) and head[0] is now[0]
    body = h5.restored(now, w0, "restore_body")
    assert np.array_equal(body[2], w0[2]) and body[4] is now[4]


def test_finetune_short_is_the_start_of_the_long_path(monkeypatch):
    data, task = synthetic(), h5.Task(np.arange(3), np.arange(3))
    net, seen = small_net(outputs=20), np.arange(3)
    short = h5.probe(net, task, 0, data, "finetune10", 5, seen, 0.05, 3)
    monkeypatch.setattr(h5, "FINETUNE_STEPS", 10)
    assert short == h5.probe(net, task, 0, data, "finetune", 5, seen, 0.05, 3)


def test_unblinded_scores_all_keys(monkeypatch):
    rows = run_small(False, monkeypatch)
    keys = {r[3] for r in rows if r[0] == "real" and r[1] > 0 and r[2] == "A"}
    assert keys == set(h5.keys_for("split_class"))
    assert {r[0] for r in rows} == {"real", "plant_access", "erase_matched", "erase_total"}


def fake_acc(seeds, a0, am, c):
    acc = {}
    for s in seeds:
        for key, n in (("none", 0), ("readout", 5), ("finetune", 5)):
            for lag, a in ((0, a0[key]), (2, am[key])):
                acc[(s, "real", lag, "A", key, n)] = a
                acc[(s, "real", lag, "C1", key, n)] = c[key]
                acc[(s, "real", lag, "C2", key, n)] = c[key]
    return acc


def test_surviving_cancels_what_a_key_teaches():
    seeds = [1, 2]
    acc = fake_acc(seeds, {"none": 0.9, "readout": 0.9, "finetune": 0.9}, {"none": 0.1, "readout": 0.5, "finetune": 0.5},
                   {"none": 0.1, "readout": 0.5, "finetune": 0.5})
    assert h5.ratio(*h5.surviving(acc, seeds, 2, "none", 0)) == 0.0
    # Recovery to 0.5 looks large, but a new task reaches 0.5 as well, so nothing is remembered.
    assert h5.ratio(*h5.surviving(acc, seeds, 2, "readout", 5)) == 0.0


def test_decompose_shares_and_label():
    seeds = list(range(8))
    acc = fake_acc(seeds, {"none": 0.9, "readout": 0.9, "finetune": 0.9}, {"none": 0.1, "readout": 0.8, "finetune": 0.85},
                   {"none": 0.1, "readout": 0.3, "finetune": 0.3})
    out = h5.decompose(acc, seeds, 2, 5, h5.resample_counts(8, 100))
    assert abs(out["access"] + out["repair"] + out["unrecovered"] - 1.0) < 1e-9
    assert out["label"] == "mostly access"


def test_labels():
    assert h5.label((0.6, 0.9), (0.7, 1.0)) == "mostly access"
    assert h5.label((0.1, 0.4), (0.6, 0.8)) == "mostly repair"
    assert h5.label((0.0, 0.1), (0.1, 0.4)) == "mostly unrecovered"
    assert h5.label((0.3, 0.7), (0.4, 0.9)) == "no majority"


def test_choose_config_rule():
    def curve(w0, final, retained):
        out = {"w0_acc": w0, "final_acc": final, "none_A_0": 0.9, "none_C_0": 0.1}
        for lag in h5.LAGS["split_class"]:
            out[f"none_A_{lag}"], out[f"none_C_{lag}"] = retained[lag], 0.1
        return out
    curves = {(0.01, 500): [curve(0.9, 0.9, {1: 0.8, 2: 0.7, 3: 0.2})],
              (0.1, 500): [curve(0.9, 0.9, {1: 0.3, 2: 0.2, 3: 0.2})],
              (0.3, 500): [curve(0.5, 0.9, {1: 0.1, 2: 0.1, 3: 0.1})]}
    chosen = h5.choose_config(curves, "split_class")
    assert (chosen["lr"], chosen["m_star"]) == (0.1, 1)


def test_seed_sets_are_disjoint():
    assert not set(h5.CAL_SEEDS) & set(h5.RETRY_SEEDS) and not set(h5.CONFIRM_SEEDS) & set(h5.CAL_SEEDS)
