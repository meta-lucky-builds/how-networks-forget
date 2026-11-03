"""Tests for experiments/h2_shared_state.py on synthetic data, so no download is needed."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_path = Path(__file__).resolve().parents[1] / "experiments" / "h2_shared_state.py"
_spec = importlib.util.spec_from_file_location("h2_shared_state", _path)
h2 = importlib.util.module_from_spec(_spec)
sys.modules["h2_shared_state"] = h2  # dataclasses look the module up here
_spec.loader.exec_module(h2)
CFG = {"lr": 0.01, "sigma": 0.1, "tau": 100.0}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), 400)
    X = rng.integers(0, 200, size=(10, 784))[y] + rng.integers(0, 56, size=(4000, 784))
    return h2.make_data(X.astype(np.uint8), y, 150)


def learner(cond, cfg=CFG, seed=1):
    return h2.Learner(cond, cfg, np.random.default_rng(seed), h2.init_weights(np.random.default_rng(99)))


def feed(L, data, a, b):
    for i in range(a, b):
        L.step(data.X[i], int(data.y[i]))


def test_both_rand_matches_real_gate_each_step(data):
    util, rand = learner("both_util"), learner("both_rand", seed=2)
    previous = None
    for i in range(40):
        x, y = data.X[i * 7], int(data.y[i * 7])
        util.step(x, y)
        gate = 1.0 - util.u  # the real gate of this step
        assert gate.std() > 0.0  # otherwise this test proves nothing
        assert (util.noise_term / gate).std() == pytest.approx(0.1, rel=0.02)
        d, E = float(gate.mean()), h2.energy(util.noise_term)
        assert util.density == pytest.approx(d) and util.energy == pytest.approx(E)
        rand.step(x, y, (d, E))
        assert h2.energy(rand.noise_term) == pytest.approx(E, rel=1e-5)
        mask = rand.noise_term != 0
        assert abs(mask.mean() - d) < 6 * math.sqrt(d * (1 - d) / h2.N_PARAMS)
        assert abs(np.corrcoef(mask.astype(float), gate)[0, 1]) < 0.05  # blind to the gate
        assert previous is None or (mask != previous).any()  # redrawn each step
        previous = mask


def test_yoked_run_is_valid_end_to_end(data):
    first = h2.run_stream("both_util", CFG, 0, data, 12, 110, metrics=False, record=True)
    assert first["yoke"].shape == (1320, 2) and (first["yoke"][:, 1] > 0).all()
    second = h2.run_stream("both_rand", CFG, 0, data, 12, 110, first["yoke"], metrics=False)
    assert 0.0 < second["energy_err"] < 1e-4 and 0.0 < second["density_z"] < 6.0


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    w = h2.init_weights(rng, dtype=np.float64)
    v, g = h2.layer_views(w), np.zeros(h2.N_PARAMS)
    x, y = rng.random(784), 4
    h2.grad_step(v, h2.layer_views(g), x, y)

    def loss():
        z = h2.logits(v, x)[2]
        return float(np.log(np.exp(z - z.max()).sum()) - (z[y] - z.max()))

    ends = np.cumsum([0] + [int(np.prod(s)) for s in h2.SHAPES])
    nonzero = 0
    for i in np.concatenate([rng.integers(a, b, 40) for a, b in zip(ends, ends[1:])]):
        keep = w[i]
        w[i] = keep + 1e-6
        up = loss()
        w[i] = keep - 1e-6
        down = loss()
        w[i] = keep
        assert g[i] == pytest.approx((up - down) / 2e-6, rel=1e-4, abs=1e-7)
        nonzero += int(g[i] != 0)
    assert nonzero > 50


def test_unit_protection_zero_noise_is_sgd_bit_for_bit(data):
    plain = h2.init_weights(np.random.default_rng(99))
    views, grads = h2.layer_views(plain), np.zeros_like(plain)
    gv = h2.layer_views(grads)
    sgd = learner("sgd", {"lr": 0.01})
    gated = learner("both_util", {"lr": 0.01, "sigma": 0.0, "tau": 100.0})
    gated.force_u = np.zeros(h2.N_PARAMS, dtype=np.float32)
    for i in range(50):
        h2.grad_step(views, gv, data.X[i], int(data.y[i]))
        plain -= 0.01 * grads
        feed(sgd, data, i, i + 1)
        feed(gated, data, i, i + 1)
        assert plain.tobytes() == sgd.w.tobytes() == gated.w.tobytes()


def test_utility_extremes(data):
    frozen = learner("both_util")
    frozen.force_u = np.ones(h2.N_PARAMS, dtype=np.float32)
    before = frozen.w.copy()
    feed(frozen, data, 0, 10)
    assert np.array_equal(before, frozen.w)
    open_ = learner("both_util")
    open_.force_u = np.zeros(h2.N_PARAMS, dtype=np.float32)
    feed(open_, data, 0, 1)
    assert np.array_equal(open_.w, before - 0.01 * (open_.g + open_.noise_term))
    assert open_.noise_term.std() == pytest.approx(0.1, rel=0.02) and abs(open_.noise_term.mean()) < 0.01


def test_utility_formula(data):
    L, before = learner("both_util"), learner("both_util").w
    feed(L, data, 0, 1)
    gw = -L.g * before  # at t=1 the bias-corrected trace is exactly -g * w
    assert np.allclose(L.u, 1.0 / (1.0 + np.exp(-gw / gw.max())), atol=1e-4)
    M = learner("both_multi", {"lr": 0.01, "sigma": 0.1})
    feed(M, data, 0, 5)
    us = [h2.normalise(t / (1.0 - b**M.t)) for t, b in zip(M.traces, M.betas)]
    assert len(us) == 3 and np.array_equal(M.u, np.maximum.reduce(us))
    assert h2.MULTI_TAUS[-1] / h2.MULTI_TAUS[0] == pytest.approx(100.0)


@pytest.mark.parametrize("cond,cfg", [("both_util", CFG), ("adam", {"lr": 1e-3, "beta1": 0.9})])
def test_probe_leaves_main_learner_untouched(data, cond, cfg):
    a, b = learner(cond, cfg), learner(cond, cfg)
    feed(a, data, 0, 10)
    feed(b, data, 0, 10)
    w, traces = a.w.copy(), [t.copy() for t in a.traces]
    h2.probe_ratio(a, cfg, 0, 0, 0, data, set())
    assert np.array_equal(w, a.w) and a.t == 10 and all(np.array_equal(x, y) for x, y in zip(traces, a.traces))
    if cond == "adam":
        clone = a.clone(np.random.default_rng(0))
        assert clone.t == 0 and not clone.m.any() and a.m.any()
    feed(a, data, 10, 20)  # the future is unchanged, RNG and optimiser state included
    feed(b, data, 10, 20)
    assert np.array_equal(a.w, b.w)


def test_zero_shot_is_before_any_update_from_those_samples(data, monkeypatch):
    calls, real = [], h2.zero_shot

    def spy(L, X, y):
        calls.append((L.t, X.copy(), y.copy()))
        return real(L, X, y)

    monkeypatch.setattr(h2, "zero_shot", spy)
    out = h2.run_stream("sgd", {"lr": 0.01}, 100, data, 45, 110)
    stream = h2.make_stream(100, 45, 110, data.pool(100))
    tasks = sorted(stream.repeats)
    assert len(calls) == len(tasks) == len(out["detail"]["events"]) > 0
    for task, (t, X, y) in zip(tasks, calls):
        assert t == task * 110  # no step of this task has been applied
        idx = stream.idx[task][:100]
        assert np.array_equal(X, data.X[idx]) and np.array_equal(y, stream.perms[task][data.y[idx]])


@pytest.mark.parametrize("seed", range(20))
def test_repeat_schedule(data, seed):
    s = h2.make_stream(seed, 100, 110, data.pool(seed))
    gaps = {"short": [], "long": []}
    for r, (source, gap, kind) in s.repeats.items():
        assert r - source == gap and np.array_equal(s.perms[r], s.perms[source])
        gaps[kind].append(gap)
    assert len(gaps["short"]) == len(gaps["long"]) == 5
    assert all(5 <= g <= 10 for g in gaps["short"]) and all(g >= 40 for g in gaps["long"])
    assert len({tuple(p) for p in s.perms.tolist()}) == 90  # only intended pairs share a permutation


def test_no_leakage_between_tuning_and_evaluation(data):
    plan = h2.FULL_PLAN
    assert not set(plan.tune_seeds) & set(plan.eval_seeds)
    tune_rows = {r.tobytes() for r in data.X[data.tune_pool]}
    assert not tune_rows & {r.tobytes() for r in data.X[data.eval_pool]}
    tune = [h2.make_stream(s, 40, 110, data.pool(s)) for s in plan.tune_seeds]
    assert all(set(s.idx.ravel().tolist()) <= set(data.tune_pool.tolist()) for s in tune)
    tune_perms = {tuple(p) for s in tune for p in s.perms.tolist()}
    for seed in plan.eval_seeds:
        s = h2.make_stream(seed, 100, 110, data.pool(seed))
        assert set(s.idx.ravel().tolist()) <= set(data.eval_pool.tolist())
        perms = {tuple(p) for p in s.perms.tolist()}
        assert len(perms & tune_perms) <= 2  # chance collisions only; images are disjoint anyway
        perm, idx = h2.probe_task(seed, 9, data.pool(seed), perms)
        assert tuple(perm.tolist()) not in perms and set(idx.tolist()) <= set(data.eval_pool.tolist())


def test_determinism(data):
    run = lambda seed: json.dumps(h2.run_stream("both_util", CFG, seed, data, 12, 110))  # noqa: E731
    assert run(5) == run(5) != run(6)


def test_non_mnist_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(h2, "FETCHERS", (lambda: (np.zeros((10, 784)), np.zeros(10)),))
    with pytest.raises(RuntimeError, match="not MNIST"):
        h2.load_mnist(tmp_path / "m.npz")
    assert not (tmp_path / "m.npz").exists()


def test_verdict_logic():
    res = np.random.default_rng(0).integers(0, 10, size=(2000, 10))
    noise = np.random.default_rng(1).normal(0.5, 0.01, 10)

    def tables(gain):
        base = {m: noise for m in ("zs", "sav", "probe_auc", "zs_short", "zs_long", "sav_short", "sav_long")}
        return {"both_rand": base, "noise_util": {**base, "zs": noise - 0.2},
                "protect": {**base, "probe_auc": noise - 0.2}, "both_util": {**base, "zs": noise + gain},
                "both_multi": {**base, "sav_long": noise + gain}}

    out = h2.evaluate(tables(0.1), res)
    assert out["manipulation_check"]["pass"] and out["h2a"]["verdict"] == out["h2b"]["verdict"] == "supported"
    out = h2.evaluate(tables(0.0), res)
    assert out["h2a"]["killed"] and out["h2b"]["killed"]
    t = tables(0.1)
    t["protect"] = t["both_util"]
    assert h2.evaluate(t, res)["h2a"]["verdict"] == "no_conclusion_manipulation_check_failed"
