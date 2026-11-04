"""Tests for experiments/h4_schema.py on synthetic data, so no download is needed."""
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("h4_schema", ROOT / "experiments" / "h4_schema.py")
h4 = importlib.util.module_from_spec(_spec)
sys.modules["h4_schema"] = h4  # dataclasses look the module up here
_spec.loader.exec_module(h4)

# Criterion 0 so every schema phase ends at its first check; the tests here are about plumbing.
TINY = h4.Plan(seeds=(300, 301, 302), lengths=(2, 4, 8), n_bootstrap=200, train_per_class=30, val_per_class=10,
               measure_per_class=16, eval_per_class=10, schema_batch=8, schema_check_every=5, schema_max_steps=20,
               schema_criterion=0.0, n_new=8, overlap_batches=2, overlap_batch=8)
TINY_SHAPES = ((2, 2), (2,), (2, 2), (2,), (3, 2), (3,))
EVENTS = {"consistent": 8, "inconsistent": 10, "foreign": 7}


def make_split(seed, n_train=60, n_test=40):
    rng = np.random.default_rng(seed)
    centres = rng.integers(0, 200, size=(10, 784))
    y_tr, y_te = np.repeat(np.arange(10), n_train), np.repeat(np.arange(10), n_test)
    x_tr = centres[y_tr] + rng.integers(0, 56, size=(len(y_tr), 784))
    x_te = centres[y_te] + rng.integers(0, 56, size=(len(y_te), 784))
    return x_tr.astype(np.uint8), y_tr, x_te.astype(np.uint8), y_te


@pytest.fixture(scope="module")
def raw():
    return make_split(0), make_split(1)


@pytest.fixture(scope="module")
def data(raw):
    return h4.Data(raw[0], raw[1], TINY)


def tiny_net():
    w = np.zeros(sum(int(np.prod(s)) for s in TINY_SHAPES))
    v = h4.layer_views(w, TINY_SHAPES)
    v[0][:] = np.eye(2)
    v[2][:] = np.eye(2)
    v[4][:] = [[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]  # The third output is inactive and must not matter
    v[5][:] = [0.0, 0.0, 9.0]
    return w, v


# ---- Gradients ----
def test_overlap_on_hand_computed_network():
    # x_old = (1, 0) label 0, x_new = (0, 1) label 1, identity layers, so logits are the input.
    # With b = 1 / (1 + e), the error vectors are (-b, b) and (b, -b). The two gradients share only
    # the output biases, so inner = -2 b^2. Each has squared norm 8 b^2. Overlap is exactly 1/4.
    w, _ = tiny_net()
    active, b = np.array([0, 1]), 1.0 / (1.0 + math.e)
    g_old = h4.gradient(w, np.array([[1.0, 0.0]]), np.array([0]), active, TINY_SHAPES)
    g_new = h4.gradient(w, np.array([[0.0, 1.0]]), np.array([1]), active, TINY_SHAPES)
    expected_old = [[[-b, 0], [0, 0]], [-b, 0], [[-b, 0], [0, 0]], [-b, 0], [[-b, 0], [b, 0], [0, 0]], [-b, b, 0]]
    expected_new = [[[0, 0], [0, -b]], [0, -b], [[0, 0], [0, -b]], [0, -b], [[0, b], [0, -b], [0, 0]], [b, -b, 0]]
    for got, want in zip(h4.layer_views(g_old, TINY_SHAPES), expected_old):
        assert got == pytest.approx(np.array(want, dtype=float))
    for got, want in zip(h4.layer_views(g_new, TINY_SHAPES), expected_new):
        assert got == pytest.approx(np.array(want, dtype=float))
    overlap, inner = h4.overlap_of(g_old, g_new)
    assert overlap == pytest.approx(0.25)
    assert inner == pytest.approx(-2.0 * b * b)


def test_overlap_of_handles_a_zero_gradient():
    assert h4.overlap_of(np.zeros(3), np.ones(3)) == (0.0, 0.0)


def test_gradient_matches_finite_differences():
    shapes = ((5, 8), (5,), (4, 5), (4,), (4, 4), (4,))
    rng = np.random.default_rng(3)
    w = h4.init_weights(rng, shapes, np.float64)
    X, y, active = rng.random((6, 8)), np.array([0, 2, 3, 0, 2, 3]), np.array([0, 2, 3])
    g = h4.gradient(w, X, y, active, shapes)
    assert np.abs(g).max() > 0.0
    for i in range(len(w)):
        saved = w[i]
        w[i] = saved + 1e-6
        up = h4.batch_loss(h4.layer_views(w, shapes), X, y, active)
        w[i] = saved - 1e-6
        down = h4.batch_loss(h4.layer_views(w, shapes), X, y, active)
        w[i] = saved
        assert g[i] == pytest.approx((up - down) / 2e-6, abs=1e-5)
    # Output 1 is inactive, so its weights and bias get exactly zero.
    v = h4.layer_views(g, shapes)
    assert not v[4][1].any() and v[5][1] == 0.0


def test_small_step_lowers_the_batch_loss(data):
    w = h4.init_weights(np.random.default_rng(1))
    idx = np.concatenate([data.train[c][:8] for c in range(4)])
    active = np.arange(4)
    before = h4.batch_loss(h4.layer_views(w), data.X[idx], data.y[idx], active)
    w2 = w - 1e-4 * h4.gradient(w, data.X[idx], data.y[idx], active)
    assert h4.batch_loss(h4.layer_views(w2), data.X[idx], data.y[idx], active) < before


def test_measure_overlap_is_the_mean_over_batches(data):
    w = h4.init_weights(np.random.default_rng(1))
    active = np.arange(5)
    old_pool, new_pool = h4.union(data.measure, range(4)), data.measure[4]
    got = h4.measure_overlap(w, data, old_pool, new_pool, active, np.random.default_rng(9), TINY)
    rng, values = np.random.default_rng(9), []
    for _ in range(TINY.overlap_batches):
        o = old_pool[rng.integers(len(old_pool), size=TINY.overlap_batch)]
        n = new_pool[rng.integers(len(new_pool), size=TINY.overlap_batch)]
        values.append(h4.overlap_of(h4.gradient(w, data.X[o], data.y[o], active),
                                    h4.gradient(w, data.X[n], data.y[n], active)))
    assert got[0] == pytest.approx(np.mean([x[0] for x in values]))
    assert got[1] == pytest.approx(np.mean([x[1] for x in values]))


def test_distance_is_cosine_to_the_nearest_old_class_mean():
    _, v = tiny_net()
    X = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 1.0]])  # Identity layers: the features are the inputs
    stub = SimpleNamespace(X=X, measure={0: np.array([0]), 1: np.array([1]), 2: np.array([2])})
    assert h4.nearest_old_distance(v, stub, 2, (0, 1)) == pytest.approx(1.0 - 3.0 / math.sqrt(10.0))
    assert h4.cosine_distance(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == pytest.approx(0.0)
    assert h4.cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


# ---- Withholding, pools, foreign matching ----
def test_each_network_has_the_right_relation_to_its_new_classes():
    for name, withheld, condition in h4.NETS:
        schema = set(h4.schema_classes(withheld))
        assert schema.isdisjoint(withheld) and schema | set(withheld) == set(range(10))
        for c in withheld:
            if condition == "consistent":
                assert set(h4.relatives(c)) & schema, (name, c)
            else:
                assert not set(h4.relatives(c)) & schema, (name, c)
    consistent = {c for _, w, cond in h4.NETS if cond == "consistent" for c in w}
    inconsistent = {c for _, w, cond in h4.NETS if cond == "inconsistent" for c in w}
    assert {4, 9} <= consistent  # The spec's examples, coat and ankle boot
    assert inconsistent == set(range(10)) and consistent == set(range(10)) - set(h4.SINGLETONS)


def test_superclasses_match_the_spec():
    assert set(h4.UPPER_BODY) == {0, 2, 3, 4, 6} and set(h4.FOOTWEAR) == {5, 7, 9} and set(h4.SINGLETONS) == {1, 8}
    assert h4.relatives(1) == () and h4.relatives(8) == ()


def test_event_counts_per_condition():
    counts = dict.fromkeys(h4.CONDITIONS, 0)
    for i in range(len(h4.NETS)):
        for condition, _ in h4.events_for(300, i):
            counts[condition] += 1
    assert counts == EVENTS


@pytest.mark.parametrize("net_index", range(len(h4.NETS)))
def test_schema_phase_sees_only_schema_classes(data, monkeypatch, net_index):
    seen, real = [], h4.batch_grad

    def spy(v, gv, X, y, active):
        seen.append((set(y.tolist()), set(active.tolist())))
        return real(v, gv, X, y, active)

    monkeypatch.setattr(h4, "batch_grad", spy)
    classes = h4.schema_classes(h4.NETS[net_index][1])
    w = h4.init_weights(np.random.default_rng(0))
    plan = replace(TINY, schema_criterion=2.0)  # Never reached, so every step runs
    steps, _, reached = h4.train_schema(w, data, classes, np.random.default_rng(1), plan)
    assert steps == plan.schema_max_steps and reached is False and len(seen) == steps
    for labels, active in seen:
        assert labels <= set(classes) and active == set(classes)


def test_criterion_stops_the_schema_phase_at_a_check(data):
    w = h4.init_weights(np.random.default_rng(0))
    steps, _, reached = h4.train_schema(w, data, (0, 1, 2), np.random.default_rng(1), TINY)
    assert reached is True and steps == TINY.schema_check_every


@pytest.mark.parametrize("net_index", range(len(h4.NETS)))
def test_no_leakage_during_fine_tuning_or_in_evaluation_sets(data, monkeypatch, net_index):
    _, withheld, _ = h4.NETS[net_index]
    classes = h4.schema_classes(withheld)
    w = h4.init_weights(np.random.default_rng(0))
    real_grad, real_acc, grads, accs = h4.batch_grad, h4.accuracy, [], []

    def grad_spy(v, gv, X, y, active):
        grads.append((set(y.tolist()), set(active.tolist())))
        return real_grad(v, gv, X, y, active)

    def acc_spy(v, X, y, active):
        accs.append((set(y.tolist()), set(active.tolist())))
        return real_acc(v, X, y, active)

    monkeypatch.setattr(h4, "batch_grad", grad_spy)
    monkeypatch.setattr(h4, "accuracy", acc_spy)
    for condition, new_class in h4.events_for(300, net_index):
        grads.clear(), accs.clear()
        h4.run_event(w, data, TINY, 300, net_index, condition, new_class, classes)
        new_label = h4.output_label(new_class)
        allowed = set(classes) | {new_label}
        assert (new_label == h4.FOREIGN_OUTPUT) == (condition == "foreign")
        for labels, active in grads:
            assert labels <= allowed and active == allowed
        for labels, active in accs:
            # Each evaluation set is the old classes alone or the new class alone.
            assert active == allowed and (labels == {new_label} or labels <= set(classes))
        assert any(labels == set(classes) for labels, _ in accs) and any(labels == {new_label} for labels, _ in accs)


def test_pools_are_disjoint_and_old_sets_exclude_the_new_class(data):
    all_ids = list(range(10)) + [h4.FOREIGN_BASE + d for d in range(10)]
    seen = np.concatenate([p[c] for p in (data.train, data.val, data.measure, data.eval) for c in all_ids])
    assert len(seen) == len(set(seen.tolist())) == len(data.X)
    for c in all_ids:
        assert (data.y[data.train[c]] == h4.output_label(c)).all()
        assert (data.y[data.eval[c]] == h4.output_label(c)).all()
    for _, withheld, _ in h4.NETS:
        classes = h4.schema_classes(withheld)
        old_eval = h4.union(data.eval, classes)
        assert not set(data.y[old_eval].tolist()) & (set(withheld) | {h4.FOREIGN_OUTPUT})
        for c in withheld:
            assert not set(old_eval.tolist()) & set(data.eval[c].tolist())


def test_held_out_images_come_from_the_test_split(raw):
    fashion, mnist = raw
    # Mark every test image, then check that only measure and eval pools hold them.
    x_tr, y_tr, x_te, y_te = fashion
    marked = (np.zeros_like(x_tr), y_tr, np.full_like(x_te, 255), y_te)
    d = h4.Data(marked, mnist, TINY)
    for c in range(10):
        assert (d.X[d.train[c]] == 0.0).all() and (d.X[d.val[c]] == 0.0).all()
        assert (d.X[d.measure[c]] == 1.0).all() and (d.X[d.eval[c]] == 1.0).all()


def test_foreign_image_counts_match_fashion(data):
    for d in range(10):
        c = h4.FOREIGN_BASE + d
        assert len(data.train[c]) == len(data.train[0]) == TINY.train_per_class
        assert len(data.measure[c]) == len(data.measure[0]) == TINY.measure_per_class
        assert len(data.eval[c]) == len(data.eval[0]) == TINY.eval_per_class


def test_digits_are_distinct_within_a_seed_and_repeatable():
    digits = [h4.foreign_digit(300, i) for i in range(len(h4.NETS))]
    assert len(set(digits)) == len(digits) and all(0 <= d < 10 for d in digits)
    assert digits == [h4.foreign_digit(300, i) for i in range(len(h4.NETS))]


def test_real_counts_fit_the_rarest_mnist_digit():
    assert h4.TRAIN_PER_CLASS <= 5421 and h4.MEASURE_PER_CLASS + h4.EVAL_PER_CLASS <= 892
    assert h4.TRAIN_PER_CLASS + h4.VAL_PER_CLASS <= 6000


def test_too_few_images_cannot_be_silently_unmatched(raw):
    x_tr, y_tr, x_te, y_te = raw[1]
    short = (x_tr, y_tr, x_te[:30], y_te[:30])  # Digit 0 alone now, others empty
    with pytest.raises(ValueError):
        h4.Data(raw[0], short, TINY)


# ---- Ladder and outcomes ----
def test_replay_ladder_is_exact():
    assert h4.LADDER == (0.0, 0.125, 0.25, 0.5, 1.0)
    assert h4.replay_counts(h4.NEW_PER_STEP) == (0, 4, 8, 16, 32)
    assert h4.replay_counts(8) == (0, 1, 2, 4, 8)
    with pytest.raises(ValueError):
        h4.replay_counts(10)


@pytest.mark.parametrize("rung", range(5))
def test_fine_tuning_batches_follow_the_ladder(data, monkeypatch, rung):
    counts = h4.replay_counts(TINY.n_new)
    n_steps = max(TINY.lengths)
    rng = np.random.default_rng(0)
    new_idx = data.train[9][rng.integers(30, size=(n_steps, TINY.n_new))]
    buffer = h4.union(data.train, range(9))
    old_idx = buffer[rng.integers(len(buffer), size=(n_steps, counts[-1]))]
    w, active = h4.init_weights(np.random.default_rng(1)), np.arange(10)
    sizes, real = [], h4.batch_grad

    def spy(v, gv, X, y, a):
        sizes.append((len(y), int((y == 9).sum())))
        return real(v, gv, X, y, a)

    monkeypatch.setattr(h4, "batch_grad", spy)
    out = h4.fine_tune(w, data, active, new_idx, old_idx, counts[rung], TINY, h4.union(data.eval, range(9)), data.eval[9])
    assert set(out) == set(TINY.lengths) and len(sizes) == n_steps
    for total, new in sizes:
        assert new == TINY.n_new and total - new == h4.LADDER[rung] * TINY.n_new


def test_buffer_exceeds_every_draw():
    assert h4.LONG_STEPS * h4.NEW_PER_STEP <= 5 * h4.BUFFER_PER_CLASS


def test_drop_is_base_minus_after_and_zero_when_nothing_moves(data):
    classes = h4.schema_classes((4, 9))
    w = h4.init_weights(np.random.default_rng(0))
    frozen = h4.run_event(w, data, replace(TINY, fine_lr=0.0), 300, 0, "consistent", 4, classes)
    assert len(frozen) == len(TINY.lengths) * len(h4.LADDER)
    assert all(r["drop"] == 0.0 and r["old_acc"] == r["base_acc"] for r in frozen)
    assert len({r["new_acc"] for r in frozen}) == 1
    moved = h4.run_event(w, data, replace(TINY, fine_lr=0.01), 300, 0, "consistent", 4, classes)
    assert all(r["drop"] == pytest.approx(r["base_acc"] - r["old_acc"]) for r in moved)
    assert len({r["old_acc"] for r in moved}) > 1 or len({r["new_acc"] for r in moved}) > 1


def test_dose_takes_the_smallest_rung_strictly_under_two():
    assert h4.dose_of([1.9, 5.0, 5.0, 5.0, 5.0]) == 0
    assert h4.dose_of([9.0, 8.0, 1.0, 3.0, 0.5]) == 2  # Not monotone: still the smallest
    assert h4.dose_of([9.0, 8.0, 2.0, 2.0, 1.999]) == 4  # Exactly 2 is not under 2
    assert h4.dose_of([9.0, 8.0, 7.0, 6.0, -1.0]) == 4  # An improvement counts
    assert h4.dose_of([9.0, 8.0, 7.0, 6.0, 2.0]) is None  # Censored


def test_censored_doses_are_never_imputed_at_the_top_rung():
    plan = replace(TINY, seeds=(400, 401, 402, 403))
    rows = synth_rows(plan.seeds)
    for r in rows:
        if r["length"] == plan.lengths[1] and r["condition"] == "consistent":
            r["drop"] = 10.0  # All consistent events censored
    res = h4.seed_resamples(4, 200)
    out = h4.dose_summary(rows, plan, list(plan.seeds), res)
    table = out["by_condition"]
    assert table["consistent"]["censored_fraction"] == 1.0
    assert table["consistent"]["median_ratio"] == "censored (above the top rung)"
    assert table["consistent"]["mean_rank_lower_bound"] == 5.0  # Above the top rung index, 4
    assert table["inconsistent"]["censored_fraction"] == 0.0
    assert out["overall_censored_fraction"] == pytest.approx(EVENTS["consistent"] / sum(EVENTS.values()))
    assert out["consistent_minus_inconsistent_censored"]["diff"] == pytest.approx(1.0)


# ---- Statistics ----
def _explicit_reml(X, y, groups, lam):
    """The same criterion from the explicit covariance matrix."""
    V = np.eye(len(y)) + lam * (groups[:, None] == groups[None, :])
    Vi = np.linalg.inv(V)
    A = X.T @ Vi @ X
    beta = np.linalg.solve(A, X.T @ Vi @ y)
    r = y - X @ beta
    dof = len(y) - X.shape[1]
    return dof * math.log(r @ Vi @ r / dof) + np.linalg.slogdet(V)[1] + np.linalg.slogdet(A)[1], beta


def test_mixed_model_matches_explicit_covariance():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(6), [3, 4, 5, 3, 4, 5])
    X = np.column_stack((np.ones(len(groups)), rng.normal(size=len(groups)), rng.normal(size=len(groups))))
    y = X @ np.array([1.0, 2.0, -3.0]) + rng.normal(size=6)[groups] + rng.normal(size=len(groups))
    model = h4.MixedModel(X, y, groups, list(range(6)))
    for lam in (0.0, 0.7, 5.0):
        got, beta, _ = model.at(lam, np.ones(6))
        want, want_beta = _explicit_reml(X, y, groups, lam)
        assert got == pytest.approx(want)
        assert beta == pytest.approx(want_beta)


def test_mixed_model_recovers_coefficients_and_variances():
    rng = np.random.default_rng(6)
    groups = np.repeat(np.arange(40), 25)
    X = np.column_stack((np.ones(1000), rng.normal(size=1000)))
    y = X @ np.array([2.0, 3.0]) + 2.0 * rng.normal(size=40)[groups] + rng.normal(size=1000)
    beta, sigma2, lam = h4.MixedModel(X, y, groups, list(range(40))).fit()
    assert beta[1] == pytest.approx(3.0, abs=0.2)
    assert sigma2 == pytest.approx(1.0, abs=0.2)
    assert sigma2 * lam == pytest.approx(4.0, abs=2.5)


def test_resample_weight_equals_duplicating_the_cluster_with_its_own_intercept():
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(3), 6)
    X = np.column_stack((np.ones(18), rng.normal(size=18)))
    y = rng.normal(size=18) + rng.normal(size=3)[groups]
    weighted = h4.MixedModel(X, y, groups, [0, 1, 2]).fit(np.array([2.0, 1.0, 0.0]))[0]
    keep = groups < 2
    X2 = np.concatenate((X[keep], X[groups == 0]))
    y2 = np.concatenate((y[keep], y[groups == 0]))
    g2 = np.concatenate((groups[keep], np.full(6, 10)))
    assert weighted == pytest.approx(h4.MixedModel(X2, y2, g2, [0, 1, 10]).fit()[0])


def test_bootstrap_clusters_over_seeds_not_rows():
    rng = np.random.default_rng(8)
    seeds, rows = list(range(30)), []
    for s in seeds:
        shock = rng.normal(0.0, 5.0)  # Shared by all of a seed's consistent rows
        rows += [{"seed": s, "condition": "consistent", "drop": 10.0 + shock + rng.normal(0.0, 0.1)} for _ in range(8)]
        rows += [{"seed": s, "condition": "inconsistent", "drop": 10.0 + rng.normal(0.0, 0.1)} for _ in range(10)]
    res = h4.seed_resamples(30, 2000)
    assert res.shape == (2000, 30) and res.min() == 0 and res.max() == 29
    out = h4.raw_effect(rows, seeds, res)
    a = np.array([r["drop"] for r in rows if r["condition"] == "consistent"])
    b = np.array([r["drop"] for r in rows if r["condition"] == "inconsistent"])
    draw = np.random.default_rng(1)
    naive = [a[draw.integers(len(a), size=len(a))].mean() - b[draw.integers(len(b), size=len(b))].mean() for _ in range(2000)]
    naive_low, naive_high = np.percentile(naive, [2.5, 97.5])
    assert out["ci_high"] - out["ci_low"] > 1.5 * (naive_high - naive_low)
    # A seed bootstrap keeps a seed's rows together: its width follows 5 / sqrt(30), not 5 / sqrt(240).
    assert out["ci_high"] - out["ci_low"] == pytest.approx(2 * 1.96 * 5.0 / math.sqrt(30.0), rel=0.4)


def test_model_bootstrap_uses_seed_weights(monkeypatch):
    rng = np.random.default_rng(9)
    groups = np.repeat(np.arange(4), 5)
    X = np.column_stack((np.ones(20), rng.normal(size=20)))
    model = h4.MixedModel(X, rng.normal(size=20), groups, [0, 1, 2, 3])
    res = np.array([[0, 0, 1, 3], [2, 2, 2, 2]])
    out = model.bootstrap(res)
    assert out[0] == pytest.approx(model.fit(np.array([2.0, 1.0, 0.0, 1.0]))[0])
    assert out[1] == pytest.approx(model.fit(np.array([0.0, 0.0, 4.0, 0.0]))[0])


def test_vif_orthogonal_and_collinear():
    a = np.array([1.0, 1.0, -1.0, -1.0] * 25)
    b = np.array([1.0, -1.0, 1.0, -1.0] * 25)
    assert h4.vif({"a": a, "b": b})["a"] == pytest.approx(1.0)
    # Correlation 0.8 by construction with b orthogonal to a: VIF = 1 / (1 - 0.64).
    c = 0.8 * a + 0.6 * b
    out = h4.vif({"a": a, "c": c})
    assert out["a"] == pytest.approx(1.0 / (1.0 - 0.64))
    assert out["c"] == pytest.approx(1.0 / (1.0 - 0.64))
    assert h4.vif({"a": a, "same": 2.0 * a})["a"] == math.inf


def test_pearson_is_undefined_for_a_constant():
    assert math.isnan(h4.pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    assert h4.pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


# ---- Verdict ----
GOOD = dict(r=0.8, max_vif=2.0, raw_ci=(-5.0, -2.0), adjusted_ci=(-0.5, 0.5), new_acc_spread=1.0, grew=True)


@pytest.mark.parametrize("changes, label, n_caveats", [
    (dict(), h4.LABEL_H4B, 0),
    (dict(adjusted_ci=(-4.0, -1.5)), h4.LABEL_H4A, 0),
    (dict(adjusted_ci=(-4.0, -1.5), grew=False), h4.LABEL_H4A, 1),
    (dict(adjusted_ci=(-4.0, -1.5), new_acc_spread=2.5), h4.LABEL_INCONCLUSIVE, 0),
    (dict(adjusted_ci=(-4.0, -1.0)), h4.LABEL_INCONCLUSIVE, 0),  # Strictly beyond 1.0
    (dict(adjusted_ci=(-1.0, 0.5)), h4.LABEL_INCONCLUSIVE, 0),  # Strictly within 1.0
    (dict(adjusted_ci=(-2.0, 0.5)), h4.LABEL_INCONCLUSIVE, 0),  # Straddles the bound
    (dict(adjusted_ci=(1.5, 4.0)), h4.LABEL_INCONCLUSIVE, 0),  # Beyond 1.0, wrong direction
    (dict(raw_ci=(-3.0, 0.5)), h4.LABEL_NO_RAW, 0),  # CI includes 0
    (dict(raw_ci=(2.0, 5.0)), h4.LABEL_NO_RAW, 0),  # Consistent drops more
    (dict(raw_ci=(-3.0, 0.0)), h4.LABEL_NO_RAW, 0),
    (dict(max_vif=5.01), h4.LABEL_VIF, 0),
    (dict(max_vif=math.inf), h4.LABEL_VIF, 0),
    (dict(r=0.49), h4.LABEL_FIRST_ORDER, 0),
    (dict(r=math.nan), h4.LABEL_FIRST_ORDER, 0),
    (dict(r=0.49, max_vif=9.0, raw_ci=(1.0, 2.0)), h4.LABEL_FIRST_ORDER, 0),  # Order of the rules
    (dict(max_vif=9.0, raw_ci=(1.0, 2.0)), h4.LABEL_VIF, 0),
])
def test_verdict(changes, label, n_caveats):
    out = h4.verdict(**{**GOOD, **changes})
    assert out["label"] == label and len(out["caveats"]) == n_caveats


def test_verdict_at_exact_limits_passes():
    assert h4.verdict(**{**GOOD, "r": 0.5, "max_vif": 5.0})["label"] == h4.LABEL_H4B
    assert h4.verdict(**{**GOOD, "adjusted_ci": (-4.0, -1.5), "new_acc_spread": 2.0})["label"] == h4.LABEL_H4A


def test_labels_are_as_preregistered():
    assert h4.LABEL_NO_RAW == "no schema effect in this setting"
    assert h4.LABEL_INCONCLUSIVE == "inconclusive"
    assert h4.EQUIVALENCE_BOUND == 1.0 and h4.VIF_LIMIT == 5.0 and h4.FIRST_ORDER_R == 0.5 and h4.N_BOOTSTRAP == 10000


# ---- Synthetic results for Stage 0 and analysis ----
def synth_rows(seeds=(300, 301, 302), effects=(0.0, 0.0, 0.0), shift=0.0, noise=0.05, seed_sd=0.5):
    """drop = base + 3 overlap + effect for consistent + seed shock, shrinking to 0.5 at the top rung."""
    rng, rows = np.random.default_rng(7), []
    for seed in seeds:
        shock = rng.normal(0.0, seed_sd)
        for i, (name, _, _) in enumerate(h4.NETS):
            for condition, new_class in h4.events_for(seed, i):
                overlap = rng.normal() - (shift if condition == "consistent" else 0.0)
                distance, jitter = rng.uniform(), rng.normal(0.0, noise)
                for li, length in enumerate(TINY.lengths):
                    d0 = (10.0, 30.0, 40.0)[li] + 3.0 * overlap + shock + jitter
                    d0 += effects[li] if condition == "consistent" else 0.0
                    for rung, ratio in enumerate(h4.LADDER):
                        frac = rung / 4.0
                        drop = d0 * (1.0 - frac) + 0.5 * frac
                        rows.append({"design": "x", "seed": seed, "net": name, "condition": condition,
                                     "new_class": new_class, "length": length, "rung": rung, "ratio": ratio,
                                     "drop": drop, "new_acc": 90.0, "base_acc": 80.0, "old_acc": 80.0 - drop,
                                     "overlap": overlap, "inner": -overlap, "distance": distance, "n_old": 8,
                                     "schema_steps": 100, "schema_acc": 80.0, "schema_reached": 1, "seconds": 1.0})
    return rows


def failed(rows):
    return h4.evaluate_stage0(rows, TINY)["failed"]


MEDIUM = TINY.lengths[1]
PLAN10 = replace(TINY, seeds=tuple(range(400, 410)))


def test_stage0_passes_on_good_results():
    out = h4.evaluate_stage0(synth_rows(), TINY)
    assert out["overall_pass"] is True and out["failed"] == []
    for check in out["checks"].values():
        assert set(check) >= {"measured", "threshold", "pass"} and check["pass"] is True


def test_stage0_fails_small_positive_control():
    rows = synth_rows()
    for r in rows:
        r["drop"] *= 0.1
    assert failed(rows) == ["positive_control_zero_replay_drop"]


def test_stage0_fails_when_the_top_rung_does_not_halve():
    rows = synth_rows()
    for r in rows:
        if r["length"] == MEDIUM and r["rung"] == 1:
            r["drop"] = 1.0  # Dose is still reached, so only the reduction can fail
        if r["length"] == MEDIUM and r["rung"] == 4:
            r["drop"] = 20.0
    assert failed(rows) == ["positive_control_top_rung_reduction"]


def test_stage0_fails_when_one_condition_does_not_learn_at_zero_replay():
    rows = synth_rows()
    for r in rows:
        if r["length"] == MEDIUM and r["rung"] == 0 and r["condition"] == "foreign":
            r["new_acc"] = 69.9
    assert failed(rows) == ["new_acc_medium_zero_replay_min_over_condition"]


def test_stage0_reports_but_does_not_gate_new_acc_at_other_rungs():
    rows = synth_rows()
    for r in rows:
        if r["length"] == MEDIUM and r["rung"] == 4 and r["condition"] == "foreign":
            r["new_acc"] = 60.0
    out = h4.evaluate_stage0(rows, TINY)
    assert out["failed"] == [] and out["diagnostics"]["new_acc_medium_min_over_condition_and_rung"] == pytest.approx(60.0)


def test_size_and_depth_columns_leave_out_constants():
    rows = synth_rows(PLAN10.seeds)
    for r in rows:
        r["n_old"] = 5 if r["net"] == "I_upper" else 8
        r["schema_steps"] = 50 if r["net"] == "I_upper" else 1000
    assert len(h4.size_and_depth(rows)) == 2
    assert h4.size_and_depth(synth_rows()) == []  # constant columns are left out


def test_stage0_fails_when_first_order_does_not_predict():
    rows = synth_rows()
    rng = np.random.default_rng(3)
    for r in rows:
        if r["length"] == TINY.lengths[0] and r["rung"] == 0:
            r["drop"] = float(rng.normal(10.0, 3.0))
    assert failed(rows) == ["first_order_r_short"]


def test_stage0_fails_and_serialises_an_undefined_r(tmp_path):
    rows = synth_rows()
    for r in rows:
        if r["length"] == TINY.lengths[0] and r["rung"] == 0:
            r["drop"] = 5.0
    out = h4.evaluate_stage0(rows, TINY)
    assert out["failed"] == ["first_order_r_short"]
    h4.write_json(tmp_path / "s.json", out)
    assert json.loads((tmp_path / "s.json").read_text())["checks"]["first_order_r_short"]["measured"] is None


def test_stage0_fails_on_variance_inflation():
    assert failed(synth_rows(shift=10.0)) == ["condition_vif"]


def _censor_events(rows, n):
    events = sorted({(r["seed"], r["net"], r["new_class"]) for r in rows})
    chosen = set(events[:n])
    for r in rows:
        if r["length"] == MEDIUM and (r["seed"], r["net"], r["new_class"]) in chosen:
            r["drop"] = max(r["drop"], 3.0)
    return len(events)


def test_stage0_fails_at_exactly_half_censored():
    rows = synth_rows(seeds=(300, 301))
    assert _censor_events(rows, 25) == 50
    assert failed(rows) == ["dose_censored_fraction"]


def test_stage0_passes_just_under_half_censored():
    rows = synth_rows(seeds=(300, 301))
    _censor_events(rows, 24)
    assert failed(rows) == []


def test_stage0_fails_when_a_schema_network_misses_the_criterion():
    rows = synth_rows()
    for r in rows:
        if r["seed"] == 300 and r["net"] == "I_foot":
            r["schema_reached"] = 0
    assert failed(rows) == ["schema_criterion_reached_fraction"]


def test_stage0_cannot_see_drop_with_condition():
    assert "condition" not in h4.OUTCOME_VIEW and "drop" not in h4.LABEL_VIEW
    assert "new_class" in h4.OUTCOME_VIEW  # Needed to group an event's rungs; no check uses it to infer condition


def test_stage0_drop_checks_ignore_condition_labels():
    rows = synth_rows(effects=(-20.0, -20.0, -20.0))  # A very large raw effect
    swapped = [dict(r, condition={"consistent": "inconsistent", "inconsistent": "consistent"}.get(r["condition"], r["condition"]))
               for r in rows]
    a, b = h4.evaluate_stage0(rows, TINY)["checks"], h4.evaluate_stage0(swapped, TINY)["checks"]
    for name in ("positive_control_zero_replay_drop", "positive_control_top_rung_reduction", "first_order_r_short",
                 "dose_censored_fraction"):
        assert a[name]["measured"] == b[name]["measured"]


def test_stage0_run_never_computes_or_reports_the_raw_effect(raw, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Stage 0 must not reach this")

    for name in ("raw_effect", "analyse", "adjusted_model", "verdict", "dose_summary"):
        monkeypatch.setattr(h4, name, forbidden)
    monkeypatch.setattr(h4, "load_datasets", lambda cache: (raw[0], raw[1], {"fashion/x": "abc"}))
    out_dir, cache = tmp_path / "h4", tmp_path / "cache"
    plan = replace(TINY, seeds=(300,))
    h4.run_stage0(plan, out_dir, True, 0, cache)
    assert sorted(p.name for p in out_dir.iterdir()) == ["stage0.json"]  # No labelled per-event file is committed
    record = json.loads((out_dir / "stage0.json").read_text())
    assert set(record) >= {"checks", "overall_pass", "failed", "design_hash", "datasets", "seeds"}
    assert record["seeds"] == [300] and record["datasets"] == {"fashion/x": "abc"}
    for check in record["checks"].values():
        assert set(check) >= {"measured", "threshold", "pass"}
    text = json.dumps({k: v for k, v in record.items() if k != "notes"})
    assert "raw_effect" not in text and "h4a" not in text.lower() and "verdict" not in text
    for pair in record["audit_overlap_vs_short_drop"]:
        assert len(pair) == 2  # Unlabelled


# ---- Refusal ----
def _passed_stage0(out_dir, plan, datasets=None, **changes):
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"overall_pass": True, "failed": [], "design_hash": h4.design_hash(plan), "datasets": datasets or {"f": "abc"}}
    record.update(changes)
    h4.write_json(out_dir / "stage0.json", record)


def test_full_run_refuses_without_stage0(tmp_path, monkeypatch):
    def forbidden(cache):
        raise AssertionError("must refuse before any download")

    monkeypatch.setattr(h4, "load_datasets", forbidden)
    with pytest.raises(h4.Stage0Refused, match="does not exist"):
        h4.run_full(TINY, tmp_path, False, 0, tmp_path / "c")


def test_full_run_refuses_after_a_failed_stage0(tmp_path, monkeypatch):
    def forbidden(cache):
        raise AssertionError("must refuse before any download")

    monkeypatch.setattr(h4, "load_datasets", forbidden)
    _passed_stage0(tmp_path, TINY, overall_pass=False, failed=["condition_vif"])
    with pytest.raises(h4.Stage0Refused, match="condition_vif"):
        h4.run_full(TINY, tmp_path, False, 0, tmp_path / "c")


def test_full_run_refuses_a_pass_on_another_design(tmp_path):
    _passed_stage0(tmp_path, replace(TINY, fine_lr=0.5))
    with pytest.raises(h4.Stage0Refused, match="different design"):
        h4.require_stage0(tmp_path, TINY)


def test_full_run_refuses_other_datasets(tmp_path):
    _passed_stage0(tmp_path, TINY)
    assert h4.require_stage0(tmp_path, TINY, {"f": "abc"})["overall_pass"] is True
    with pytest.raises(h4.Stage0Refused, match="dataset"):
        h4.require_stage0(tmp_path, TINY, {"f": "other"})


def test_stage0_and_full_plans_share_a_design():
    assert h4.design_hash(h4.STAGE0_PLAN) == h4.design_hash(h4.FULL_PLAN)
    assert h4.design_hash(replace(h4.FULL_PLAN, fine_lr=0.1)) != h4.design_hash(h4.FULL_PLAN)


def test_seeds_are_disjoint_and_sized():
    calibration, stage0, evaluation = set(h4.CALIBRATION_SEEDS), set(h4.STAGE0_PLAN.seeds), set(h4.FULL_PLAN.seeds)
    assert stage0 == {300, 301, 302} and min(evaluation) >= 400 and len(evaluation) >= 20
    assert not (calibration & stage0 or calibration & evaluation or stage0 & evaluation)


def test_full_plan_uses_the_preregistered_settings():
    assert h4.FULL_PLAN.n_bootstrap == 10000
    assert h4.FULL_PLAN.lengths == (h4.SHORT_STEPS, h4.MEDIUM_STEPS, h4.LONG_STEPS)
    assert h4.SHORT_STEPS < h4.MEDIUM_STEPS < h4.LONG_STEPS
    assert h4.SHAPES[0] == (256, 784) and h4.SHAPES[2] == (256, 256) and h4.SHAPES[4] == (11, 256)


# ---- Analysis ----


def label_of(rows):
    return h4.analyse(rows, PLAN10)


def test_analysis_h4b_when_overlap_explains_the_raw_effect():
    out = label_of(synth_rows(PLAN10.seeds, shift=2.0))
    assert out["raw_effect"]["ci_high"] < 0.0  # A raw effect exists
    assert out["verdict"]["label"] == h4.LABEL_H4B
    assert abs(out["primary_model"]["consistent_minus_inconsistent"]["estimate"]) < 0.2
    assert out["primary_model"]["coefficients"]["overlap"]["estimate"] == pytest.approx(3.0, abs=0.1)


def test_analysis_h4a_when_a_residual_remains_and_grows():
    out = label_of(synth_rows(PLAN10.seeds, effects=(-1.0, -5.0, -12.0)))
    assert out["verdict"] == {"label": h4.LABEL_H4A, "caveats": []}
    assert out["primary_model"]["consistent_minus_inconsistent"]["estimate"] == pytest.approx(-5.0, abs=0.2)
    assert out["higher_order"]["grew"] is True


def test_analysis_states_the_caveat_when_the_residual_does_not_grow():
    out = label_of(synth_rows(PLAN10.seeds, effects=(-5.0, -5.0, -4.0)))
    assert out["verdict"]["label"] == h4.LABEL_H4A and out["verdict"]["caveats"] == [h4.CAVEAT_NO_GROWTH]
    assert out["higher_order"]["grew"] is False


def test_analysis_blocks_h4a_when_new_acc_differs():
    rows = synth_rows(PLAN10.seeds, effects=(-1.0, -5.0, -12.0))
    for r in rows:
        if r["condition"] == "foreign":
            r["new_acc"] = 80.0
    out = label_of(rows)
    assert out["new_acc"]["within"] is False and out["new_acc"]["largest_gap"] == pytest.approx(10.0)
    assert out["verdict"]["label"] == h4.LABEL_INCONCLUSIVE


def test_analysis_no_schema_effect_when_consistent_drops_more():
    assert label_of(synth_rows(PLAN10.seeds, effects=(3.0, 3.0, 3.0)))["verdict"]["label"] == h4.LABEL_NO_RAW


def test_analysis_not_answerable_on_variance_inflation():
    out = label_of(synth_rows(PLAN10.seeds, shift=10.0))
    assert out["verdict"]["label"] == h4.LABEL_VIF and out["vif"]["pass"] is False


def test_analysis_stops_when_first_order_fails():
    rows = synth_rows(PLAN10.seeds, effects=(-1.0, -5.0, -12.0))
    rng = np.random.default_rng(3)
    for r in rows:
        if r["length"] == TINY.lengths[0] and r["rung"] == 0:
            r["drop"] = float(rng.normal(10.0, 3.0))
    out = label_of(rows)
    assert out["verdict"]["label"] == h4.LABEL_FIRST_ORDER and out["first_order_validity"]["pass"] is False


def test_analysis_excludes_events_of_a_network_that_missed_the_criterion():
    rows = synth_rows(PLAN10.seeds, shift=2.0)
    n = 0
    for r in rows:
        if r["seed"] == 400 and r["net"] == "C1":
            r["schema_reached"] = 0
            n += 1
    out = label_of(rows)
    assert out["excluded_events_rows"] == n and out["schema_phase"]["C1"]["missed_criterion"] == 1
    assert out["primary_model"]["n_events"] == 10 * sum(EVENTS.values()) - 3


# ---- Resume, and the whole path ----
def test_resume_redoes_only_the_cut_off_job(raw, tmp_path, monkeypatch):
    plan, path = replace(TINY, seeds=(300,)), tmp_path / "raw.csv"
    store = h4.RawStore(path, plan)
    h4.execute(plan, store, 0, tmp_path, raw, "t")
    store.close()
    first = {(r["net"], r["new_class"], r["length"], r["rung"]): r["drop"] for r in store.rows()}
    assert len({k[0] for k in first}) == len(h4.NETS)
    blob = path.read_bytes()
    path.write_bytes(blob[:-40])  # A crash during the last write

    calls, real = [], h4.run_job
    monkeypatch.setattr(h4, "run_job", lambda job: calls.append(job) or real(job))
    store = h4.RawStore(path, plan)
    assert not store.done(300, "I_single") and store.done(300, "C1")
    assert len({r["net"] for r in store.rows()}) == len(h4.NETS) - 1  # Partial job dropped, not half-kept
    h4.execute(plan, store, 0, tmp_path, raw, "t")
    store.close()
    assert [job[2] for job in calls] == [len(h4.NETS) - 1]
    second = {(r["net"], r["new_class"], r["length"], r["rung"]): r["drop"] for r in store.rows()}
    assert second == pytest.approx(first)  # Deterministic, so a resumed run equals an unbroken one

    calls.clear()
    store = h4.RawStore(path, plan)
    h4.execute(plan, store, 0, tmp_path, raw, "t")
    store.close()
    assert calls == [] and len(store.rows()) == len(first)
    reread = {(r["net"], r["new_class"], r["length"], r["rung"]): r["drop"] for r in store.rows()}
    assert reread == second  # Floats survive the CSV exactly


def test_resume_refuses_rows_from_another_design(raw, tmp_path):
    plan, path = replace(TINY, seeds=(300,)), tmp_path / "raw.csv"
    store = h4.RawStore(path, plan)
    h4._DATA = h4.Data(raw[0], raw[1], plan)
    store.add(h4.run_job((plan, 300, 0)))
    store.close()
    with pytest.raises(ValueError, match="fresh"):
        h4.RawStore(path, replace(plan, fine_lr=0.5))
    fresh = h4.RawStore(path, replace(plan, fine_lr=0.5), fresh=True)
    fresh.close()
    assert fresh.rows() == []


def test_run_job_rows_are_complete(raw):
    plan = replace(TINY, seeds=(300,))
    h4._DATA = h4.Data(raw[0], raw[1], plan)
    rows = h4.run_job((plan, 300, 4))  # Footwear withheld: 3 events plus a foreign one
    assert len(rows) == 4 * len(plan.lengths) * len(h4.LADDER)
    assert all(set(r) == set(h4.RawStore.FIELDS) for r in rows)
    assert {r["condition"] for r in rows} == {"inconsistent", "foreign"}
    assert {r["n_old"] for r in rows} == {7}


def test_full_run_end_to_end_writes_an_atomic_summary(raw, tmp_path, monkeypatch):
    monkeypatch.setattr(h4, "load_datasets", lambda cache: (raw[0], raw[1], {"f": "abc"}))
    plan = replace(TINY, seeds=(400, 401, 402), n_bootstrap=20)
    _passed_stage0(tmp_path, plan)
    h4.run_full(plan, tmp_path, False, 0, tmp_path / "c")
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["verdict"]["label"] in {h4.LABEL_FIRST_ORDER, h4.LABEL_VIF, h4.LABEL_NO_RAW, h4.LABEL_H4A,
                                           h4.LABEL_H4B, h4.LABEL_INCONCLUSIVE}
    assert summary["primary_model"]["n_events"] == 3 * sum(EVENTS.values())
    assert not (tmp_path / "summary.json.tmp").exists()
    assert (tmp_path / "raw_results.csv").exists()


# ---- Datasets and the run request ----
def test_fetch_checks_the_hash_and_tries_the_next_route(tmp_path, monkeypatch):
    good = b"good bytes"
    expected = hashlib.sha256(good).hexdigest()
    served = {"https://a/f": b"tampered", "https://b/f": good}
    monkeypatch.setattr(h4, "_download", lambda url: served[url])
    assert h4.fetch_verified("f", ("https://a/", "https://b/"), "sha256", expected, tmp_path) == good
    assert (tmp_path / "f").read_bytes() == good
    monkeypatch.setattr(h4, "_download", lambda url: pytest.fail("cache must be used"))
    assert h4.fetch_verified("f", ("https://a/",), "sha256", expected, tmp_path) == good


def test_fetch_refuses_bad_data_and_plain_http(tmp_path, monkeypatch):
    monkeypatch.setattr(h4, "_download", lambda url: b"tampered")
    with pytest.raises(RuntimeError, match="mismatch"):
        h4.fetch_verified("f", ("https://a/",), "sha256", "0" * 64, tmp_path)
    assert not (tmp_path / "f").exists()
    with pytest.raises(RuntimeError, match="not HTTPS"):
        h4.fetch_verified("f", ("http://a/",), "sha256", "0" * 64, tmp_path)


def test_routes_are_https():
    assert all(r.startswith("https://") for r in h4.FASHION_ROUTES + h4.MNIST_ROUTES)


def test_parse_idx():
    images = gzip.compress((2051).to_bytes(4, "big") + (2).to_bytes(4, "big") + (28).to_bytes(4, "big") * 2 + bytes(2 * 784))
    labels = gzip.compress((2049).to_bytes(4, "big") + (2).to_bytes(4, "big") + bytes([3, 7]))
    assert h4.parse_idx(images, 2).shape == (2, 784)
    assert h4.parse_idx(labels, 2).tolist() == [3, 7]
    with pytest.raises(ValueError):
        h4.parse_idx(labels, 3)


def test_run_request_names_a_mode():
    first = (ROOT / ".github" / "run-requests" / "h4").read_text().split()[0]
    assert first in {"stage0", "full"}
