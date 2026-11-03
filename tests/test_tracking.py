"""Tests for H1 Stage A. Run in the cloud only, per docs/CHARTER.md."""

from __future__ import annotations

import inspect
import itertools
import math

import numpy as np
import pytest
from scipy.integrate import trapezoid

from clu.tracking.learners import (
    LEARNERS,
    Bayes,
    Deferred,
    DeferredNoCheck,
    Fixed,
    Monotone,
    OracleEvents,
    Robust,
    RobustBOCD,
    run_learner,
    run_oracle,
)
from clu.tracking.streams import CONTEXTS, HAZARD, OUTLIER_PROB, generate_stream

EXAMPLE_CONFIGS = {
    "fixed": {"alpha": 0.3},
    "monotone": {"alpha_lo": 0.1, "alpha_hi": 0.9, "scale": 5.0},
    "robust": {"process_var": 0.5, "c": 3.0},
    "robust_bocd": {"hazard": 0.05, "c": 3.0, "prior_var": 133.0},
    "deferred": {"alpha": 0.3, "threshold": 4.0, "k": 2, "window": 4, "tolerance": 2.0},
    "deferred_nocheck": {"alpha": 0.3, "threshold": 4.0, "k": 2, "window": 4, "tolerance": 2.0},
    "bayes": {"hazard": 0.05, "outlier_prob": 0.05, "top_k": 32},
}

# Observations chosen so that truncation and the event hypotheses all matter.
# The second case has a small R, with samples beyond the support of the outlier
# density, so the outlier branch must vanish there.
ENUMERATION_CASES = [
    {
        "ys": [0.3, 0.1, 15.0, 0.2, 14.0, 14.5, -0.3, 14.2],
        "hazard": 0.05,
        "outlier_prob": 0.05,
        "sigma": 1.0,
        "radius": 20.0,
    },
    {
        "ys": [2.9, 3.4, 2.5, -2.8, 3.1, -3.6, 0.0, 2.7],
        "hazard": 0.2,
        "outlier_prob": 0.2,
        "sigma": 1.0,
        "radius": 3.0,
    },
]


def brute_force(
    ys: list[float], hazard: float, outlier_prob: float, sigma: float, radius: float
) -> tuple[float, float]:
    """Log evidence and E[m_T | y_1:T] from all 4^T event paths.

    Shares no formula with the filter: each segment's integral over the uniform
    prior on m is computed by quadrature on a grid.
    """
    grid = np.linspace(-radius, radius, 40001)
    cache: dict[tuple[int, ...], tuple[float, float]] = {}

    def segment_integrals(segment: tuple[int, ...]) -> tuple[float, float]:
        if segment not in cache:
            integrand = np.full_like(grid, 1.0 / (2.0 * radius))
            for i in segment:
                integrand = integrand * (
                    np.exp(-0.5 * ((ys[i] - grid) / sigma) ** 2)
                    / (sigma * math.sqrt(2.0 * math.pi))
                )
            cache[segment] = (trapezoid(integrand, grid), trapezoid(integrand * grid, grid))
        return cache[segment]

    evidence = 0.0
    mean_numerator = 0.0
    for path in itertools.product([(0, 0), (0, 1), (1, 0), (1, 1)], repeat=len(ys)):
        weight = 1.0
        segment: list[int] = []
        for t, (changepoint, outlier) in enumerate(path):
            weight *= hazard if changepoint else 1.0 - hazard
            weight *= outlier_prob if outlier else 1.0 - outlier_prob
            if changepoint:
                weight *= segment_integrals(tuple(segment))[0]
                segment = []
            if outlier:
                weight *= 1.0 / (2.0 * radius) if abs(ys[t]) <= radius else 0.0
            else:
                segment.append(t)
        integral, moment = segment_integrals(tuple(segment))
        evidence += weight * integral
        mean_numerator += weight * moment
    return math.log(evidence), mean_numerator / evidence


@pytest.mark.parametrize("case", ENUMERATION_CASES)
@pytest.mark.parametrize("length", [1, 3, 5, 8])
def test_bayes_without_pruning_matches_enumeration(case: dict, length: int) -> None:
    ys = case["ys"][:length]
    params = {key: case[key] for key in ("hazard", "outlier_prob", "sigma", "radius")}
    log_evidence, posterior_mean = brute_force(ys, **params)

    bayes = Bayes(**params, top_k=None)
    estimate = 0.0
    for y in ys:
        estimate = bayes.step(y)

    assert bayes.log_evidence == pytest.approx(log_evidence, abs=1e-6)
    assert estimate == pytest.approx(posterior_mean, abs=1e-6)


def _integrate_predictive(bayes: Bayes, radius: float) -> float:
    """Integral of the predictive, split where the outlier density jumps."""
    limit = radius + 10.0
    pieces = [
        np.linspace(-limit, np.nextafter(-radius, -math.inf), 2001),
        np.linspace(-radius, radius, 8001),
        np.linspace(np.nextafter(radius, math.inf), limit, 2001),
    ]
    total = 0.0
    for grid in pieces:
        density = np.array([bayes.predictive_density(float(y)) for y in grid])
        total += trapezoid(density, grid)
    return total


@pytest.mark.parametrize("case", ENUMERATION_CASES)
def test_bayes_predictive_integrates_to_one(case: dict) -> None:
    params = {key: case[key] for key in ("hazard", "outlier_prob", "sigma", "radius")}
    bayes = Bayes(**params, top_k=None)
    assert _integrate_predictive(bayes, case["radius"]) == pytest.approx(1.0, abs=1e-5)
    for y in case["ys"][:4]:
        bayes.step(y)
    assert _integrate_predictive(bayes, case["radius"]) == pytest.approx(1.0, abs=1e-5)


def test_predictive_density_does_not_change_state() -> None:
    bayes = Bayes(hazard=0.05, outlier_prob=0.05, top_k=None)
    bayes.step(1.0)
    before = (bayes.n.copy(), bayes.s.copy(), bayes.log_w.copy(), bayes.log_evidence)
    bayes.predictive_density(3.0)
    assert np.array_equal(before[0], bayes.n)
    assert np.array_equal(before[1], bayes.s)
    assert np.array_equal(before[2], bayes.log_w)
    assert before[3] == bayes.log_evidence


def test_bayes_pruning_bounds_components_and_renormalises() -> None:
    stream = generate_stream("mixed", np.random.default_rng(3), T=200)
    bayes = Bayes(hazard=0.05, outlier_prob=0.05, top_k=8)
    for y in stream.y.tolist():
        bayes.step(y)
        assert bayes.log_w.size <= 8
        assert float(np.exp(bayes.log_w).sum()) == pytest.approx(1.0, abs=1e-9)


def test_bayes_large_top_k_matches_no_pruning() -> None:
    ys = np.array(ENUMERATION_CASES[0]["ys"])
    exact = run_learner(Bayes(0.05, 0.05, top_k=None), ys)
    # Eight steps from one component cannot exceed 2^10 components.
    unpruned = run_learner(Bayes(0.05, 0.05, top_k=1024), ys)
    assert np.array_equal(exact, unpruned)


def test_bayes_on_clean_stream_is_the_sample_mean_posterior() -> None:
    m = 3.0
    y = m + np.random.default_rng(11).normal(size=500)
    running_mean = np.cumsum(y) / np.arange(1, y.size + 1)

    # The model itself rules out events, and truncation is negligible this far
    # inside the support once a few samples have arrived.
    exact = run_learner(Bayes(0.0, 0.0, top_k=8), y)
    assert np.allclose(exact[20:], running_mean[20:], atol=1e-9)

    # With events possible but rare, the filter concentrates on the same answer.
    bayes = Bayes(1e-4, 1e-4, top_k=64)
    estimates = run_learner(bayes, y)
    assert estimates[-1] == pytest.approx(running_mean[-1], abs=0.01)
    top = int(np.argmax(bayes.log_w))
    assert bayes.n[top] == y.size
    assert bayes.s[top] == pytest.approx(y.sum())


def test_oracle_without_events_equals_exact_filter_without_events() -> None:
    y = np.random.default_rng(5).normal(loc=19.5, size=50)  # near the edge of the prior
    no_events = np.zeros(y.size, dtype=bool)
    oracle = run_oracle(OracleEvents(), y, no_events, no_events)
    bayes = run_learner(Bayes(0.0, 0.0, top_k=None), y)
    assert np.allclose(oracle, bayes, atol=1e-12)
    assert np.all(np.abs(bayes) <= 20.0)


def test_oracle_resets_and_skips_on_labels() -> None:
    oracle = OracleEvents()
    assert oracle.step(1.0, changepoint=False, outlier=False) == pytest.approx(1.0)
    assert oracle.step(9.0, changepoint=False, outlier=True) == pytest.approx(1.0)
    assert oracle.step(3.0, changepoint=False, outlier=False) == pytest.approx(2.0)
    assert oracle.step(10.0, changepoint=True, outlier=False) == pytest.approx(10.0)
    assert oracle.step(-5.0, changepoint=True, outlier=True) == 0.0


def test_error_ordering_on_mixed_streams() -> None:
    fixed_err, bayes_err, oracle_err = [], [], []
    for seed in range(5):
        s = generate_stream("mixed", np.random.default_rng(seed), T=1000)
        fixed_err.append(np.mean((run_learner(Fixed(0.3), s.y) - s.m) ** 2))
        bayes_err.append(np.mean((run_learner(Bayes(0.05, 0.05, top_k=64), s.y) - s.m) ** 2))
        oracle_err.append(
            np.mean((run_oracle(OracleEvents(), s.y, s.changepoint, s.outlier) - s.m) ** 2)
        )
    assert np.mean(oracle_err) < np.mean(bayes_err) < np.mean(fixed_err)


@pytest.mark.parametrize("cls", [Deferred, DeferredNoCheck])
def test_infinite_threshold_is_bit_identical_to_fixed(cls: type[Deferred]) -> None:
    y = generate_stream("mixed", np.random.default_rng(2), T=2000).y
    fixed = run_learner(Fixed(0.17), y)
    deferred = run_learner(cls(0.17, threshold=math.inf, k=2, window=4, tolerance=1.0), y)
    assert np.array_equal(fixed, deferred)


def test_nocheck_commits_everything_and_deferred_commits_a_subset() -> None:
    y = generate_stream("mixed", np.random.default_rng(0), T=2000).y
    config = EXAMPLE_CONFIGS["deferred"]

    nocheck = DeferredNoCheck(**config)
    run_learner(nocheck, y)
    assert nocheck.quarantined_indices
    assert nocheck.n_discarded == 0
    assert sorted(nocheck.committed_indices) + [i for i, _ in nocheck.quarantine] == sorted(
        nocheck.quarantined_indices
    )

    deferred = Deferred(**config)
    run_learner(deferred, y)
    assert deferred.committed_indices
    assert deferred.n_discarded > 0
    assert set(deferred.committed_indices) < set(deferred.quarantined_indices)
    assert (
        len(deferred.committed_indices) + deferred.n_discarded + len(deferred.quarantine)
        == len(deferred.quarantined_indices)
    )


def test_nocheck_delay_equals_k() -> None:
    y = np.zeros(20)
    y[10] = 15.0
    estimates = run_learner(DeferredNoCheck(0.5, 4.0, k=2, window=4, tolerance=1.0), y)
    assert estimates[10] == 0.0
    assert estimates[11] == 0.0
    assert estimates[12] == 7.5


def test_deferred_ignores_outlier_and_adopts_changepoint_exactly() -> None:
    t_outlier, t_change = 50, 100
    y = np.zeros(150)
    y[t_outlier] = 15.0
    y[t_change:] = 10.0
    learner = Deferred(0.5, threshold=4.0, k=2, window=4, tolerance=2.0)
    estimates = run_learner(learner, y)

    # Power check: the plain delta rule does react to this outlier.
    assert run_learner(Fixed(0.5), y)[t_outlier] == 7.5

    # The outlier never moves the estimate.
    assert np.all(estimates[:t_change] == 0.0)
    assert t_outlier in learner.quarantined_indices
    assert t_outlier not in learner.committed_indices

    # The change waits for k later samples, then commits by the delta rule.
    assert estimates[t_change] == 0.0
    assert estimates[t_change + 1] == 0.0
    assert estimates[t_change + 2] == 8.75
    assert estimates[t_change + 3] == 9.375
    assert learner.committed_indices == [t_change, t_change + 1, t_change + 2]
    assert estimates[-1] == pytest.approx(10.0)


def test_commit_alpha_sets_the_commit_rate_only() -> None:
    y = np.zeros(20)
    y[10:] = 10.0
    learner = Deferred(0.1, threshold=4.0, k=2, window=4, tolerance=2.0, commit_alpha=1.0)
    estimates = run_learner(learner, y)
    assert estimates[11] == 0.0
    assert estimates[12] == 10.0
    # With the default, commits use the tracking rate.
    slow = run_learner(Deferred(0.1, threshold=4.0, k=2, window=4, tolerance=2.0), y)
    assert slow[12] == pytest.approx(10.0 * (1.0 - 0.9**3))


def test_deferred_outlier_immediately_before_changepoint_is_not_committed() -> None:
    t_change = 100
    y = np.zeros(130)
    y[t_change - 1] = -15.0
    y[t_change:] = 10.0
    learner = Deferred(0.5, threshold=4.0, k=2, window=4, tolerance=2.0)
    estimates = run_learner(learner, y)
    assert t_change - 1 not in learner.committed_indices
    assert learner.n_discarded == 1
    assert estimates[t_change + 2] == 8.75


def test_deferred_with_noise_ignores_outlier_and_adopts_changepoint() -> None:
    t_outlier, t_change = 50, 100
    y = 0.2 * np.random.default_rng(1).normal(size=150)
    y[t_outlier] = 15.0
    y[t_change:] += 10.0
    estimates = run_learner(Deferred(0.5, threshold=4.0, k=2, window=4, tolerance=2.0), y)
    assert np.all(np.abs(estimates[:t_change]) < 1.0)
    assert np.all(np.abs(estimates[t_change + 10 :] - 10.0) < 1.0)


def test_deferred_rejects_impossible_window() -> None:
    with pytest.raises(ValueError):
        Deferred(0.5, threshold=4.0, k=3, window=2, tolerance=1.0)


def test_monotone_gain_rises_and_robust_gain_falls_with_error() -> None:
    def gain(make: type, config: dict, error: float) -> float:
        return make(**config).step(error) / error

    small, large = 1.0, 15.0
    assert gain(Monotone, EXAMPLE_CONFIGS["monotone"], large) > gain(
        Monotone, EXAMPLE_CONFIGS["monotone"], small
    )
    assert gain(Robust, EXAMPLE_CONFIGS["robust"], large) < gain(
        Robust, EXAMPLE_CONFIGS["robust"], small
    )


def test_robust_bocd_recovers_from_outlier_and_adopts_changepoint() -> None:
    y = np.zeros(120)
    y[50] = 15.0
    y[100:] = 10.0
    learner = RobustBOCD(hazard=0.05, c=3.0, prior_var=133.0, max_components=16)
    estimates = []
    for value in y.tolist():
        estimates.append(learner.step(value))
        assert learner.log_w.size <= 16
        assert float(np.exp(learner.log_w).sum()) == pytest.approx(1.0, abs=1e-9)
    assert abs(estimates[49]) < 1e-9
    assert abs(estimates[60]) < 0.5  # recovered after the outlier
    assert abs(estimates[-1] - 10.0) < 0.5  # adopted the change


def test_only_the_oracle_can_receive_labels_or_the_true_mean() -> None:
    forbidden = {
        "m", "mean", "true_mean", "stream", "event", "events", "label", "labels",
        "changepoint", "changepoints", "outlier", "outliers",
    }
    for cls in LEARNERS.values():
        assert list(inspect.signature(cls.step).parameters) == ["self", "y"]
        assert not forbidden & set(inspect.signature(cls.__init__).parameters)

    assert list(inspect.signature(OracleEvents.step).parameters) == [
        "self", "y", "changepoint", "outlier",
    ]
    assert not issubclass(OracleEvents, tuple(LEARNERS.values()))
    # The oracle gets labels, never the mean itself.
    assert "m" not in inspect.signature(OracleEvents.step).parameters


@pytest.mark.parametrize("context", list(CONTEXTS))
def test_stream_is_deterministic_under_seed(context: str) -> None:
    a = generate_stream(context, np.random.default_rng(42), T=500)
    b = generate_stream(context, np.random.default_rng(42), T=500)
    c = generate_stream(context, np.random.default_rng(43), T=500)
    for field in ("y", "m", "changepoint", "outlier"):
        assert np.array_equal(getattr(a, field), getattr(b, field))
    assert not np.array_equal(a.y, c.y)


@pytest.mark.parametrize(
    "context, hazard, outlier_prob",
    [
        ("changepoint", HAZARD, 0.0),
        ("outlier", 0.0, OUTLIER_PROB),
        ("mixed", HAZARD, OUTLIER_PROB),
        ("stationary", 0.0, 0.0),
    ],
)
def test_stream_rates_and_structure(context: str, hazard: float, outlier_prob: float) -> None:
    T = 200000
    s = generate_stream(context, np.random.default_rng(17), T=T)

    def tolerance(rate: float) -> float:
        return 5.0 * math.sqrt(rate * (1.0 - rate) / T)

    assert abs(s.changepoint.mean() - hazard) <= tolerance(hazard)
    assert abs(s.outlier.mean() - outlier_prob) <= tolerance(outlier_prob)

    # The mean moves only at changepoints and stays in the support of the prior.
    moved = np.flatnonzero(np.diff(s.m) != 0.0) + 1
    assert set(moved.tolist()) <= set(np.flatnonzero(s.changepoint).tolist())
    assert np.all(np.abs(s.m) <= 20.0)

    # Inliers are the mean plus unit noise; outliers are uniform on the support.
    noise = s.y[~s.outlier] - s.m[~s.outlier]
    assert abs(noise.mean()) < 0.03
    assert abs(noise.std() - 1.0) < 0.03
    if outlier_prob > 0.0:
        assert np.all(np.abs(s.y[s.outlier]) <= 20.0)
        assert abs(s.y[s.outlier].std() - 40.0 / math.sqrt(12.0)) < 0.5


def test_context_table_matches_preregistration() -> None:
    assert HAZARD == 0.05
    assert OUTLIER_PROB == 0.05
    assert set(CONTEXTS) == {"changepoint", "outlier", "mixed", "stationary"}


def test_learners_are_deterministic_on_a_seeded_stream() -> None:
    def run_all() -> list[np.ndarray]:
        s = generate_stream("mixed", np.random.default_rng(8), T=300)
        runs = [
            run_learner(LEARNERS[name](**EXAMPLE_CONFIGS[name]), s.y) for name in LEARNERS
        ]
        runs.append(run_oracle(OracleEvents(), s.y, s.changepoint, s.outlier))
        return runs

    for first, second in zip(run_all(), run_all()):
        assert np.array_equal(first, second)
