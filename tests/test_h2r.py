"""Tests for experiments/h2r.py on synthetic data, so no download is needed."""
import csv
import importlib.util
import json
import math
import sys
import zlib
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("h2r", ROOT / "experiments" / "h2r.py")
h2r = importlib.util.module_from_spec(_spec)
sys.modules["h2r"] = h2r  # dataclasses and worker pickling look the module up here
_spec.loader.exec_module(h2r)
h2 = h2r.h2
CFG = {"lr": 0.01, "sigma": 0.1, "tau": 100.0}
SGD = {"lr": 0.01}
TASKS, STEPS = 45, 110
PER_CLASS = 100


def synthetic():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), 400)
    X = rng.integers(0, 200, size=(10, 784))[y] + rng.integers(0, 56, size=(4000, 784))
    return X.astype(np.uint8), y


@pytest.fixture(scope="module")
def data():
    return h2r.make_data(*synthetic(), PER_CLASS)


def learner(cond, cfg=CFG, seed=1):
    return h2r.Learner(cond, cfg, np.random.default_rng(seed), h2.init_weights(np.random.default_rng(99)))


def feed(L, data, a, b):
    for i in range(a, b):
        L.step(data.X[i], int(data.y[i]))


# Gates


def test_gate_mean_is_one_and_sd_near_058(data):
    L = learner("both_util")
    feed(L, data, 0, 25)
    assert L.P.dtype == np.float32
    for a, b in h2r.SLICES:
        assert L.P[a:b].mean(dtype=np.float64) == pytest.approx(1.0, abs=1e-6)
    assert L.P.mean(dtype=np.float64) == pytest.approx(1.0, abs=1e-6)
    assert L.P.std(dtype=np.float64) == pytest.approx(2.0 / math.sqrt(12.0), abs=0.005)
    assert np.array_equal(L.P, 2.0 * (1.0 - L.u)) and np.array_equal(L.R, 1.0 - L.u)
    assert 0.0 < L.u.min() and L.u.max() < 1.0


def test_gate_statistics_survive_total_ties():
    u = h2r.rank_unit(np.zeros(1000, dtype=np.float32), np.random.default_rng(0))
    assert np.array_equal(np.sort(u), h2r.grid(1000))
    assert 2.0 * (1.0 - u.astype(np.float64)).mean() == pytest.approx(1.0, abs=1e-6)


def test_rank_direction_high_utility_is_protected():
    values = np.array([0.3, -1.0, 2.0, 0.1], dtype=np.float32)
    u = h2r.rank_unit(values, np.random.default_rng(0))
    assert np.allclose(u, [0.625, 0.125, 0.875, 0.375])
    assert 2.0 * (1.0 - u[2]) < 1.0 < 2.0 * (1.0 - u[1])  # the top weight's gradient is damped most


def test_ties_are_random_and_reproducible():
    values = np.zeros(100000, dtype=np.float32)
    u = h2r.rank_unit(values, np.random.default_rng(5))
    assert np.array_equal(u, h2r.rank_unit(values, np.random.default_rng(5)))
    assert not np.array_equal(u, h2r.rank_unit(values, np.random.default_rng(6)))
    assert abs(np.corrcoef(u, np.arange(values.size))[0, 1]) < 0.02  # not the array's index order


def test_mixed_ties_keep_order_between_groups():
    rng = np.random.default_rng(1)
    values = rng.integers(0, 5, 5000).astype(np.float32)
    u = h2r.rank_unit(values, rng)
    assert np.array_equal(np.sort(u), h2r.grid(5000))
    for low, high in zip(range(4), range(1, 5)):
        assert u[values == low].max() < u[values == high].min()


@pytest.mark.parametrize("cond", ["protect", "noise_util", "both_util", "both_perm"])
def test_gate_refreshes_every_10_steps_only(data, cond):
    L = learner(cond)
    feed(L, data, 0, 1)
    assert L.P is not None  # a gate exists from the first update
    first = L.P
    feed(L, data, 1, 10)
    assert L.P is first
    feed(L, data, 10, 11)
    assert L.P is not first and L.t == 11


def test_trace_updates_each_step_and_rank_follows_it(data):
    L = learner("both_util")
    feed(L, data, 0, 1)
    before = L.traces[0].copy()
    feed(L, data, 1, 2)
    assert not np.array_equal(before, L.traces[0])
    feed(L, data, 2, 11)
    for a, b in h2r.SLICES:
        ordered = L.traces[0][a:b][np.argsort(L.u[a:b])]
        assert (np.diff(ordered) >= 0).all()


def test_both_perm_scrambles_noise_gate_only(data):
    util, perm = learner("both_util", seed=1), learner("both_perm", seed=2)
    previous, refreshes = None, 0
    for i in range(41):
        feed(util, data, i * 7, i * 7 + 1)
        feed(perm, data, i * 7, i * 7 + 1)
        if (perm.t - 1) % h2r.REFRESH:
            continue
        refreshes += 1
        for a, b in h2r.SLICES:
            assert np.array_equal(np.sort(perm.R[a:b]), np.sort(util.R[a:b]))  # the same gate values
        assert np.array_equal(perm.P, 2.0 * (1.0 - perm.u))  # protection follows the real utility
        assert np.array_equal(util.P, 2.0 * util.R)
        assert abs(np.corrcoef(perm.R, 1.0 - perm.u)[0, 1]) < 0.03  # noise does not
        if previous is not None:  # re-permuted at each refresh
            assert abs(np.corrcoef(previous, perm.R)[0, 1]) < 0.03
        previous = perm.R
    assert refreshes == 5
    rng = np.random.default_rng(3)
    assert np.corrcoef(perm.u, h2r.within_layer_rank(perm.traces[0], rng))[0, 1] > 0.8  # the check can detect a relation


def test_both_multi_reranks_the_max_of_its_ranks():
    rng = np.random.default_rng(4)
    traces = [rng.standard_normal(h2r.N_PARAMS) for _ in range(3)]  # float64, so no ties
    u = h2r.rerank_max(traces, np.random.default_rng(5))
    ranks = [h2r.within_layer_rank(t, np.random.default_rng(6)) for t in traces]  # no ties, so the RNG is moot
    best = np.maximum.reduce(ranks)
    for a, b in h2r.SLICES:
        assert np.array_equal(np.sort(u[a:b]), h2r.grid(b - a))  # re-ranked: spread is matched, not inflated
        assert (np.diff(best[a:b][np.argsort(u[a:b])]) >= 0).all()
    assert best.mean() > 0.7 > u.mean()  # the unranked max would sit near 0.75
    top = int(np.argmax(ranks[2][:100352]))
    assert u[top] > 0.99  # one trace alone can protect a weight


def test_both_multi_learner_matches_single_trace_spread(data):
    multi, util = learner("both_multi", {"lr": 0.01, "sigma": 0.1}), learner("both_util")
    feed(multi, data, 0, 21)
    feed(util, data, 0, 21)
    assert [1.0 / r for r in multi.rates] == pytest.approx(list(h2.MULTI_TAUS))
    assert not np.array_equal(multi.traces[0], multi.traces[2])
    for a, b in h2r.SLICES:
        assert np.array_equal(np.sort(multi.P[a:b]), np.sort(util.P[a:b]))


def test_update_rules(data):
    for cond in ("noise_util", "both_util"):
        L = learner(cond)
        before = L.w.copy()
        feed(L, data, 0, 1)
        P = L.P if L.protect else 1.0
        assert np.allclose(L.w, before - 0.01 * (L.g * P + L.noise_term), atol=1e-6)
        assert (L.noise_term / L.R).std() == pytest.approx(0.1, rel=0.02)
    L = learner("protect", {"lr": 0.01, "tau": 100.0})
    before = L.w.copy()
    feed(L, data, 0, 1)
    assert np.array_equal(L.w, before - 0.01 * (L.g * L.P))


def test_flat_gate_is_sgd_bit_for_bit(data):
    sgd, gated, oracle = learner("sgd", SGD), learner("protect", {"lr": 0.01, "tau": 100.0}), learner("oracle", SGD)
    gated.force_u = np.full(h2r.N_PARAMS, 0.5, dtype=np.float32)
    for i in range(30):
        for L in (sgd, gated, oracle):
            feed(L, data, i, i + 1)
        assert sgd.w.tobytes() == gated.w.tobytes() == oracle.w.tobytes()


def test_spearman_helper():
    a = h2r.grid(1000).copy()
    assert h2r.rank_corr(a, a) == pytest.approx(1.0)
    assert h2r.rank_corr(a, a[::-1]) == pytest.approx(-1.0)
    assert abs(h2r.rank_corr(a, a[np.random.default_rng(0).permutation(1000)])) < 0.15


def test_run_records_gate_stability(data):
    cfg = {**CFG, "lr": 0.002}  # small, so that a divergent run cannot turn this into a test of NaNs
    out = h2r.run_stream("both_util", cfg, 300, data, 12, STEPS)
    null = h2r.run_stream("both_perm", cfg, 300, data, 12, STEPS)
    assert -1.0 <= out["gate_spearman"] <= 1.0 and -1.0 <= null["gate_spearman"] <= 1.0
    assert out["gate_mean"] == pytest.approx(1.0, abs=1e-6)
    assert out["gate_std"] == pytest.approx(null["gate_std"])


# Stream and metrics


def test_oracle_restores_source_end_weights(data, monkeypatch):
    seen, real = [], h2r.zero_shot

    def spy(L, X, y):
        seen.append((L.t, L.w.copy()))
        return real(L, X, y)

    monkeypatch.setattr(h2r, "zero_shot", spy)
    out = h2r.run_stream("oracle", SGD, 300, data, TASKS, STEPS)
    plain = h2r.run_stream("sgd", SGD, 300, data, TASKS, STEPS)
    assert out["detail"]["w_crc"] != plain["detail"]["w_crc"]
    n_repeats = len(seen) // 2
    stream = h2r.make_stream(300, TASKS, STEPS, data.pool(300))
    assert n_repeats == len(stream.repeats) > 0
    L = h2r.Learner("sgd", SGD, np.random.default_rng(0), h2.init_weights(h2r.key(300, h2r.K_INIT)))
    saved, expected = {}, []
    for task in range(TASKS):
        rep = stream.repeats.get(task)
        if rep:
            L.w[:] = saved[rep[0]]
            expected.append((task * STEPS, L.w.copy()))
        xs, labels = h2r.task_data(stream, data, task)
        for x, y in zip(xs, labels):
            L.step(x, y)
        saved[task] = L.w.copy()
    assert out["detail"]["w_crc"] == zlib.crc32(L.w.tobytes())
    for (t, w), (t2, w2) in zip(seen[:n_repeats], expected):
        assert t == t2 and np.array_equal(w, w2)
    assert any(not np.array_equal(w, w2) for (_, w), (_, w2) in zip(seen[n_repeats:], expected))  # sgd does not restore


def test_zero_shot_is_before_any_update_from_those_samples(data, monkeypatch):
    calls, real = [], h2r.zero_shot

    def spy(L, X, y):
        calls.append((L.t, X.copy(), y.copy()))
        return real(L, X, y)

    monkeypatch.setattr(h2r, "zero_shot", spy)
    out = h2r.run_stream("sgd", SGD, 300, data, TASKS, STEPS)
    stream = h2r.make_stream(300, TASKS, STEPS, data.pool(300))
    tasks = sorted(stream.repeats)
    assert len(calls) == len(tasks) == len(out["detail"]["events"]) > 0
    for task, (t, X, y) in zip(tasks, calls):
        assert t == task * STEPS  # no step of this task has been applied
        idx = stream.idx[task][:h2r.WINDOW]
        assert np.array_equal(X, data.X[idx][:, stream.perms[task]]) and np.array_equal(y, data.y[idx])


def test_steps_to_target_and_censoring():
    assert h2r.TARGET == 0.75
    correct = np.r_[np.zeros(50, dtype=bool), np.ones(950, dtype=bool)]
    assert h2r.steps_to_target(correct) == (125, True)  # 75 hits in the window ending at step 125
    assert h2r.steps_to_target(np.ones(1000, dtype=bool)) == (100, True)
    assert h2r.steps_to_target(np.zeros(1000, dtype=bool)) == (1000, False)
    assert h2r.steps_to_target(np.tile([True, False], 500)) == (1000, False)  # 0.5 never reaches 0.75


def test_nearest_non_repeats():
    repeats = {5: None, 6: None}
    assert h2r.nearest_non_repeats(5, repeats, 10) == [4, 3]  # 6 is a repeat; tie between 3 and 7 goes earlier
    assert h2r.nearest_non_repeats(6, repeats, 10) == [7, 4]
    assert h2r.nearest_non_repeats(9, {9: None}, 10) == [8, 7]
    assert h2r.nearest_non_repeats(2, {2: None}, 10) == [1, 3]


def test_reuse_formula_and_censoring():
    steps = np.array([300, 500, 200, 700, 1000, 400])
    reached = np.array([True, True, True, True, False, True])
    repeats = {2: None}
    event = h2r.reuse_event(2, steps, reached, repeats, 6)
    assert event["near"] == [1, 3] and event["reuse"] == 200 - 600  # negative: the repeat was faster
    assert not event["censored"]
    event = h2r.reuse_event(5, steps, reached, {5: None}, 6)
    assert event["near"] == [4, 3] and event["reuse"] == 400 - 850  # the censored comparator counts as task length
    assert event["censored"]
    reached = np.array([True, True, False, True, True, True])
    assert h2r.reuse_event(2, steps, reached, repeats, 6)["censored"]  # the repeat itself missing


def test_reuse_in_a_run(data):
    out = h2r.run_stream("sgd", SGD, 300, data, TASKS, STEPS)
    events = out["detail"]["events"]
    stream = h2r.make_stream(300, TASKS, STEPS, data.pool(300))
    assert [e["task"] for e in events] == sorted(stream.repeats)
    for e in events:
        assert h2r.WINDOW <= e["steps"] <= STEPS
        assert e["reuse"] == e["steps"] - np.mean(e["near_steps"])
        assert not set(e["near"]) & set(stream.repeats) and e["kind"] == stream.repeats[e["task"]][2]
    for kind in ("short", "long"):
        assert out[f"reuse_{kind}"] == pytest.approx(np.mean([e["reuse"] for e in events if e["kind"] == kind]))
    assert out["reuse_censored_frac"] == pytest.approx(np.mean([e["censored"] for e in events]))


def test_forget_uses_the_source_end_accuracy(data, monkeypatch):
    # With correct always true, end accuracy is 1, so forgetting is 1 minus zero-shot.
    monkeypatch.setattr(h2r, "zero_shot", lambda L, X, y: 0.25)
    monkeypatch.setattr(h2r.Learner, "step", lambda self, x, y: y)
    monkeypatch.setattr(h2r, "online", lambda L, xs, labels: 0.5)
    out = h2r.run_stream("sgd", SGD, 300, data, TASKS, STEPS)
    assert out["forget_long"] == pytest.approx(0.75) and out["zs_long"] == pytest.approx(0.25)
    assert out["probe_auc"] == pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(20))
def test_repeat_schedule(data, seed):
    s = h2r.make_stream(seed + 400, 100, STEPS, data.pool(seed + 400))
    gaps = {"short": [], "long": []}
    for r, (source, gap, kind) in s.repeats.items():
        assert r - source == gap and np.array_equal(s.perms[r], s.perms[source])
        assert r not in {q[0] for q in s.repeats.values()}
        gaps[kind].append(gap)
    assert len(gaps["short"]) == len(gaps["long"]) == 5
    assert all(5 <= g <= 10 for g in gaps["short"]) and all(g >= 40 for g in gaps["long"])
    assert len({tuple(p) for p in s.perms.tolist()}) == 90  # only intended pairs share a permutation
    assert (np.sort(s.perms, axis=1) == np.arange(784)).all()


def test_input_permutation_with_fixed_labels(data):
    s = h2r.make_stream(300, 12, STEPS, data.pool(300))
    xs, labels = h2r.task_data(s, data, 3)
    idx = s.idx[3]
    assert np.array_equal(labels, data.y[idx])  # labels are not permuted
    assert np.array_equal(xs[:, 0], data.X[idx][:, s.perms[3][0]])
    other, _ = h2r.task_data(s, data, 4)
    assert xs.shape == other.shape == (STEPS, 784)


def test_seeds_conditions_and_pools_are_disjoint(data):
    p = h2r.FULL_PLAN
    assert (p.tune_seeds, p.stage0_seeds, p.eval_seeds) == ((200, 201), (300, 301, 302), tuple(range(400, 410)))
    assert (p.tasks, p.tune_tasks, p.steps, p.n_configs) == (100, 40, 1000, 10)
    assert len({*p.tune_seeds, *p.stage0_seeds, *p.eval_seeds, *h2.FULL_PLAN.eval_seeds, *h2.FULL_PLAN.tune_seeds}) == 2 + 3 + 10 + 10 + 2
    assert "oracle" not in h2r.FULL_CONDS and "oracle" in h2r.STAGE0_CONDS
    assert set(h2r.STAGE0_CONDS) == {"sgd", "protect", "both_util", "both_perm", "oracle"}
    rows = [{r.tobytes() for r in data.X[pool]} for pool in (data.tune_pool, data.stage0_pool, data.eval_pool)]
    assert not rows[0] & rows[1] and not rows[0] & rows[2] and not rows[1] & rows[2]
    for seeds, pool in ((p.tune_seeds, data.tune_pool), (p.stage0_seeds, data.stage0_pool), (p.eval_seeds, data.eval_pool)):
        for seed in seeds:
            assert data.pool(seed) is pool
            s = h2r.make_stream(seed, 12, STEPS, pool)
            assert set(s.idx.ravel().tolist()) <= set(pool.tolist())
    for seed in (0, 100, 199, 500):  # H2's seeds and strays are refused rather than given a pool
        with pytest.raises(ValueError):
            data.pool(seed)


def test_configs(data):
    a = h2r.cfg_for("both_util", 3)
    assert a == h2r.cfg_for("both_perm", 3) == h2r.cfg_for("both_util", 3)
    assert h2r.cfg_for("oracle", 3) == h2r.cfg_for("sgd", 3)
    assert h2r.cfg_for("sgd", 3, 3.0)["lr"] == pytest.approx(3.0 * h2r.cfg_for("sgd", 3)["lr"])
    assert h2r.cfg_for("oracle", 3, 3.0)["lr"] == pytest.approx(3.0 * h2r.cfg_for("sgd", 3)["lr"])
    assert h2r.cfg_for("protect", 3, 3.0)["lr"] == pytest.approx(3.0 * h2r.cfg_for("protect", 3)["lr"])
    assert h2r.cfg_for("both_util", 3, 3.0)["tau"] == h2r.cfg_for("both_util", 3)["tau"]
    assert "tau" not in h2r.cfg_for("both_multi", 0) and "sigma" in h2r.cfg_for("both_multi", 0)
    assert {"lr", "sigma", "decay"} <= set(h2r.cfg_for("shrink_perturb", 0))
    assert len({h2r.cfg_for("sgd", i)["lr"] for i in range(10)}) == 10
    for cond in h2r.CONDS:  # every condition can run a step from its own config
        feed(learner(cond, h2r.cfg_for(cond, 0)), data, 0, 12)


@pytest.mark.parametrize("cond", ["both_util", "both_perm", "both_multi"])
def test_probe_leaves_main_learner_untouched(data, cond):
    cfg = h2r.cfg_for(cond, 0)
    a, b = learner(cond, cfg), learner(cond, cfg)
    feed(a, data, 0, 15)
    feed(b, data, 0, 15)
    w, traces, u = a.w.copy(), [t.copy() for t in a.traces], a.u.copy()
    h2r.probe_ratio(a, cfg, 300, 0, 0, data)
    assert np.array_equal(w, a.w) and a.t == 15 and np.array_equal(u, a.u)
    assert all(np.array_equal(x, y) for x, y in zip(traces, a.traces))
    feed(a, data, 15, 30)  # the future is unchanged, RNG included
    feed(b, data, 15, 30)
    assert np.array_equal(a.w, b.w)


def test_determinism(data):
    run = lambda seed: json.dumps(h2r.run_stream("both_util", CFG, seed, data, 12, STEPS))  # noqa: E731
    assert run(300) == run(300) != run(301)


# Stage 0


def stage0_table():
    full = lambda v: np.full(3, float(v))  # noqa: E731
    return {"sgd": {"zs_short": full(0.1), "zs_long": full(0.1), "forget_long": full(0.5), "auc_first": full(1.5),
                    "auc_last": full(0.8), "reuse_censored_frac": full(0.1)},
            "oracle": {"zs_short": full(0.9), "zs_long": full(0.9)},
            "protect": {"auc_last": full(0.6)},
            "both_util": {"gate_std": full(0.58), "gate_spearman": full(0.8)}}


def failed_of(checks):
    return {name for name, c in checks.items() if not c["pass"]}


def test_stage0_all_pass_and_record_shape():
    checks = h2r.stage0_checks(stage0_table())
    assert failed_of(checks) == set()
    assert len(checks) == 9
    for c in checks.values():
        assert {"item", "statement", "value", "threshold", "op", "per_seed", "pass"} <= set(c) and len(c["per_seed"]) == 3
    assert checks["retention_long"]["value"] == pytest.approx(0.8) and checks["retention_long"]["threshold"] == 0.5


@pytest.mark.parametrize("name,cond,metric,values", [
    ("retention_short", "oracle", "zs_short", [0.5, 0.5, 0.5]),
    ("retention_long", "oracle", "zs_long", [0.5, 0.5, 0.5]),
    ("forgetting", "sgd", "forget_long", [0.05, 0.05, 0.05]),
    ("plasticity_below_1", "sgd", "auc_last", [0.8, 0.8, 1.01]),  # one seed is enough to fail
    ("plasticity_below_first", "sgd", "auc_first", [1.5, 1.5, 0.7]),
    ("tension", "protect", "auc_last", [0.6, 0.6, 0.9]),
    ("reuse_measurable", "sgd", "reuse_censored_frac", [0.2, 0.2, 0.2]),  # strict: must be under 20%
    ("gate_sd", "both_util", "gate_std", [0.04, 0.04, 0.04]),
    ("gate_stability", "both_util", "gate_spearman", [0.49, 0.49, 0.49]),
    ("gate_stability", "both_util", "gate_spearman", [math.nan, 0.9, 0.9]),  # a NaN cannot pass
])
def test_stage0_each_check_fails_alone(name, cond, metric, values):
    t = stage0_table()
    t[cond][metric] = np.array(values)
    assert failed_of(h2r.stage0_checks(t)) == {name}


@pytest.mark.parametrize("name,cond,metric,value", [
    ("retention_short", "oracle", "zs_short", 0.75),
    ("gate_stability", "both_util", "gate_spearman", 0.5),
])
def test_stage0_at_least_is_inclusive(name, cond, metric, value):
    t = stage0_table()
    if name == "retention_short":
        t["sgd"]["zs_short"] = np.full(3, 0.25)
    t[cond][metric] = np.full(3, value)
    assert name not in failed_of(h2r.stage0_checks(t))


def record_with(failing):
    checks = h2r.stage0_checks(stage0_table())
    for name in failing:
        checks[name]["pass"] = False
    return {"overall_pass": not failing, "retry": False, "checks": checks}


def test_retry_rules():
    assert h2r.retry_refusal(record_with(["plasticity_below_1"])) is None
    assert h2r.retry_refusal(record_with(["tension"])) is None
    assert h2r.retry_refusal(record_with(["tension", "plasticity_below_first"])) is None
    assert "gate" in h2r.retry_refusal(record_with(["gate_stability"]))
    assert "reuse" in h2r.retry_refusal(record_with(["tension", "reuse_measurable"]))
    assert "no completed" in h2r.retry_refusal(None)
    assert "passed" in h2r.retry_refusal(record_with([]))
    assert "already" in h2r.retry_refusal({**record_with(["tension"]), "retry": True})


def test_full_run_refuses_without_a_pass(tmp_path):
    with pytest.raises(h2r.Refused):
        h2r.require_stage0_pass(tmp_path)
    (tmp_path / "stage0.json").write_text("{not json")
    with pytest.raises(h2r.Refused):
        h2r.require_stage0_pass(tmp_path)
    for value in (False, None, "true", 1):
        (tmp_path / "stage0.json").write_text(json.dumps({"overall_pass": value}))
        with pytest.raises(h2r.Refused):
            h2r.require_stage0_pass(tmp_path)
    with pytest.raises(h2r.Refused):
        h2r.run_full(h2r.FULL_PLAN, tmp_path, 1, tmp_path / "unused.npz")
    (tmp_path / "stage0.json").write_text(json.dumps({"overall_pass": True}))
    assert h2r.require_stage0_pass(tmp_path)["overall_pass"] is True


def test_retry_cannot_overwrite_or_repeat(tmp_path):
    (tmp_path / "stage0.json").write_text(json.dumps({**record_with(["tension"]), "retry": True}))
    with pytest.raises(h2r.Refused, match="on record"):
        h2r.run_stage0(h2r.FULL_PLAN, tmp_path, 1, tmp_path / "unused.npz")
    with pytest.raises(h2r.Refused, match="already"):
        h2r.run_stage0(h2r.FULL_PLAN, tmp_path, 1, tmp_path / "unused.npz", retry=True)


def test_project_minutes_matches_the_written_estimate():
    ms = {"sgd": 1.1, "both_util": 2.6, "both_perm": 2.8}
    assert h2r.run_steps(h2r.FULL_PLAN) == 106000
    assert h2r.project_minutes(ms, 2) == pytest.approx(314, abs=2)


def test_json_is_strict_and_atomic(tmp_path):
    h2r.write_json(tmp_path / "x.json", {"a": math.nan, "b": np.float64(1.5), "c": np.array([1.0, math.inf]), "d": np.bool_(True), "e": (1,)})
    text = (tmp_path / "x.json").read_text()
    assert json.loads(text, parse_constant=lambda c: pytest.fail(c)) == {"a": None, "b": 1.5, "c": [1.0, None], "d": True, "e": [1]}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


# Analysis


def test_holm_steps_down():
    strong, weak = np.linspace(1.0, 2.0, 10001), np.linspace(-1.0, 1.0, 10001)
    border = np.linspace(-0.02, 0.98, 10001)  # 2-sided p near 0.04: excludes 0 at 0.05, not at 0.025
    out = h2r.holm_intervals({"a": strong, "b": border})
    assert out["a"]["level"] == pytest.approx(0.025) and out["a"]["reject"]
    assert out["b"]["level"] == pytest.approx(0.05) and out["b"]["reject"]
    assert out["b"]["ci_low"] > 0.0
    out = h2r.holm_intervals({"a": weak, "b": border})
    assert out["b"]["level"] == pytest.approx(0.025) and not out["b"]["reject"] and out["b"]["ci_low"] < 0.0
    assert not out["a"]["reject"] and out["a"]["level"] == pytest.approx(0.025)  # the procedure stopped
    negative = h2r.holm_intervals({"a": -strong})
    assert negative["a"]["reject"] and negative["a"]["ci_high"] < 0.0 and negative["a"]["level"] == pytest.approx(0.05)
    zero = h2r.holm_intervals({"a": np.zeros(100), "b": np.zeros(100)})
    assert not zero["a"]["reject"] and not zero["b"]["reject"]


def test_verdict_logic():
    res = np.random.default_rng(0).integers(0, 10, size=(2000, 10))
    noise = np.random.default_rng(1).normal(0.5, 0.01, 10)
    names = ("zs", "gain", "gain_short", "gain_long", "zs_short", "zs_long", "probe_auc")

    def tables(lift):
        base = {m: noise for m in names}
        return {"both_perm": base, "noise_util": {**base, "zs": noise - 0.2}, "protect": {**base, "probe_auc": noise - 0.2},
                "both_util": {**base, "zs": noise + lift}, "both_multi": {**base, "gain_long": noise + lift}}

    out = h2r.evaluate(tables(0.1), res)
    assert out["manipulation_check"]["pass"] and out["h2a"]["verdict"] == out["h2b"]["verdict"] == "supported"
    assert out["h2a"]["holm"]["zs"]["reject"] and not out["h2a"]["holm"]["gain"]["reject"]
    out = h2r.evaluate(tables(0.0), res)
    assert out["h2a"]["verdict"] == out["h2b"]["verdict"] == "killed"
    t = tables(-0.1)  # both_util worse than the null: a rejection in the wrong direction supports nothing
    assert not h2r.evaluate(t, res)["h2a"]["supported"]
    t = tables(0.1)
    t["both_multi"] = {**t["both_multi"], "gain_short": noise + 0.1}  # comparable at short gaps
    out = h2r.evaluate(t, res)
    assert out["h2b"]["comparable_short_gap_advantage"] and not out["h2b"]["supported"]
    t = tables(0.1)
    t["both_perm"] = {**t["both_perm"], "probe_auc": noise + 0.2}  # retention bought with plasticity
    assert not h2r.evaluate(t, res)["h2a"]["supported"]
    t = tables(0.1)
    t["protect"] = t["both_util"]
    out = h2r.evaluate(t, res)
    assert out["h2a"]["verdict"] == out["h2b"]["verdict"] == "no_conclusion_manipulation_check_failed"


def test_reuse_sign_becomes_gain():
    full = lambda v: np.full(3, float(v))  # noqa: E731
    out = h2r.derived({"zs_short": full(0.2), "zs_long": full(0.4), "reuse_short": full(-100), "reuse_long": full(-300)})
    assert out["gain_short"][0] == 100 and out["gain_long"][0] == 300 and out["gain"][0] == 200
    assert out["zs"][0] == pytest.approx(0.3)


# Resume and plumbing


def row(job, seconds=1.0):
    phase, cond, index, cfg, seed, n_tasks, _, _ = job
    return {"phase": phase, "condition": cond, "config_index": index, "seed": seed, "n_tasks": n_tasks,
            **{m: "0.5" for m in h2r.METRICS}, "seconds": seconds, "config": json.dumps(cfg), "detail": "{}"}


class FakePool:
    def __init__(self):
        self.jobs = []

    def submit(self, fn, job):
        self.jobs.append(job)
        future = Future()
        future.set_result(fn(job))
        return future


def test_resume_skips_done_jobs_and_drops_a_torn_line(tmp_path, monkeypatch):
    monkeypatch.setattr(h2r, "run_job", lambda job: [row(job)])
    jobs = [("eval", c, 0, {}, s, 100, 1000, True) for c in ("sgd", "protect") for s in (400, 401)]
    path, pool = tmp_path / "raw.csv", FakePool()
    store = h2r.RawStore(path)
    h2r.run_jobs(jobs[:3], store, pool, "x")
    assert len(pool.jobs) == 3
    store.close()
    with path.open("a") as handle:
        handle.write("eval,protect,0,401,100,0.5,0.5")  # a line cut short by the timeout
    store = h2r.RawStore(path)
    pool = FakePool()
    h2r.run_jobs(jobs, store, pool, "x")
    assert pool.jobs == jobs[3:]  # only the missing one; the torn line was not trusted
    store.close()
    with path.open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 4
    store = h2r.RawStore(path)
    pool = FakePool()
    h2r.run_jobs(jobs, store, pool, "x")
    assert pool.jobs == []
    assert store.get("eval", "sgd", 0, 400, 100)["online_acc"] == 0.5
    store.close()


def test_rows_with_different_phase_or_config_are_distinct_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(h2r, "run_job", lambda job: [row(job)])
    store, pool = h2r.RawStore(tmp_path / "raw.csv"), FakePool()
    h2r.run_jobs([("stage0", "sgd", 0, {}, 300, 100, 1000, True)], store, pool, "x")
    h2r.run_jobs([("stage0_retry", "sgd", 0, {}, 300, 100, 1000, True), ("stage0", "sgd", 1, {}, 300, 100, 1000, True)], store, pool, "x")
    assert len(pool.jobs) == 3
    store.close()


def test_stage0_then_full_end_to_end_and_resume(tmp_path):
    plan = h2r.Plan(TASKS, 12, STEPS, 2, (200,), (300, 301), (400, 401), 100, PER_CLASS)
    X, y = synthetic()
    cache = tmp_path / "mnist.npz"
    np.savez(cache, X=X, y=y)
    out = tmp_path / "out"
    out.mkdir()
    record = h2r.run_stage0(plan, out, 2, cache)
    on_disk = json.loads((out / "stage0.json").read_text())
    assert on_disk["overall_pass"] == all(c["pass"] for c in on_disk["checks"].values()) == record["overall_pass"]
    assert set(on_disk["chosen"]) == set(h2r.STAGE0_CONDS)
    assert on_disk["chosen"]["both_perm"] == on_disk["chosen"]["both_util"]
    assert on_disk["chosen"]["oracle"] == on_disk["chosen"]["sgd"]

    def rows():
        with (out / "raw_results.csv").open(newline="") as handle:
            return list(csv.DictReader(handle))

    first = rows()
    assert len(first) == 3 * 2 + 5 * 2
    h2r.run_stage0(plan, out, 2, cache)
    assert rows() == first  # resumed: nothing ran again
    if not on_disk["overall_pass"]:
        with pytest.raises(h2r.Refused):
            h2r.run_full(plan, out, 2, cache)
    (out / "stage0.json").write_text(json.dumps({**on_disk, "overall_pass": True, "lr_multiplier": 3.0}))
    summary = h2r.run_full(plan, out, 2, cache)
    assert (out / "summary.json").exists() and set(summary["per_condition"]) == set(h2r.FULL_CONDS)
    assert summary["chosen"]["sgd"]["config"]["lr"] == pytest.approx(3.0 * h2r.cfg_for("sgd", summary["chosen"]["sgd"]["config_index"])["lr"])
    assert not [r for r in rows() if r["phase"] == "eval" and r["condition"] == "oracle"]
    assert {int(r["seed"]) for r in rows() if r["phase"] == "eval"} == {400, 401}
    for r in rows():
        assert json.loads(r["detail"]) is not None and r["seconds"]
