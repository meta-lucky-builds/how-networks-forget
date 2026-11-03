"""Stage A learners, per hypotheses/H1_corroboration_gating.md.

Timing convention, identical for every learner: the value returned by
`step(y_t)` is the estimate of m_t after y_t has been seen. It is the value
scored against m_t, so all learners are compared on the same footing. For
`Deferred` that value excludes whatever is still in quarantine at time t, which
is the cost of waiting. For `Bayes` it is the posterior mean E[m_t | y_1:t].

Interface: `step` of a learner takes only y_t, and its constructor takes only
its configuration. `OracleEvents` is the single exception and additionally
receives the true event labels. No learner ever receives the true mean.

All learners start from an estimate of 0, the midpoint of the prior on m.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from scipy.special import log_ndtr

_LOG_2PI = math.log(2.0 * math.pi)


def _logsumexp(log_values: np.ndarray) -> float:
    top = float(log_values.max())
    if not math.isfinite(top):
        return top
    return top + math.log(float(np.exp(log_values - top).sum()))


def _safe_log(x: float) -> float:
    return math.log(x) if x > 0.0 else -math.inf


def _truncated_stats(
    mu: np.ndarray, var: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Log of P(|X| <= radius) and E[X | |X| <= radius] for X ~ N(mu, var)."""
    sd = np.sqrt(var)
    a = (-radius - mu) / sd
    b = (radius - mu) / sd
    # Phi(b) - Phi(a) = Phi(-a) - Phi(-b). Mirror so the subtraction happens in
    # the lower tail, where log_ndtr keeps its precision.
    flip = a + b > 0.0
    lo = np.where(flip, -b, a)
    hi = np.where(flip, -a, b)
    log_hi = log_ndtr(hi)
    log_c = log_hi + np.log1p(-np.exp(log_ndtr(lo) - log_hi))
    log_phi_a = -0.5 * (a * a + _LOG_2PI)
    log_phi_b = -0.5 * (b * b + _LOG_2PI)
    mean = mu + sd * (np.exp(log_phi_a - log_c) - np.exp(log_phi_b - log_c))
    return log_c, np.clip(mean, -radius, radius)


class Learner(ABC):
    """A tracker that sees only its configuration and the observations."""

    @abstractmethod
    def step(self, y: float) -> float:
        """Consume y_t and return the estimate of m_t after seeing it."""


class Fixed(Learner):
    """Delta rule with a constant learning rate."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.estimate = 0.0

    def step(self, y: float) -> float:
        self.estimate += self.alpha * (y - self.estimate)
        return self.estimate


class Monotone(Learner):
    """Delta rule whose learning rate increases with |error|, from lo to hi."""

    def __init__(self, alpha_lo: float, alpha_hi: float, scale: float) -> None:
        if alpha_hi < alpha_lo:
            raise ValueError("alpha_hi must be >= alpha_lo")
        self.alpha_lo = alpha_lo
        self.alpha_hi = alpha_hi
        self.scale = scale
        self.estimate = 0.0

    def step(self, y: float) -> float:
        error = y - self.estimate
        squared = error * error
        surprise = squared / (squared + self.scale * self.scale)
        alpha = self.alpha_lo + (self.alpha_hi - self.alpha_lo) * surprise
        self.estimate += alpha * error
        return self.estimate


class Robust(Learner):
    """Kalman filter with IMQ-weighted observation noise.

    The observation variance is inflated by 1 / w^2, with
    w^2 = 1 / (1 + r^2 / (c^2 S)) and S the innovation variance. Surprise can
    only shrink the gain, never raise it.
    """

    def __init__(
        self,
        process_var: float,
        c: float,
        obs_var: float = 1.0,
        initial_var: float = 100.0,
    ) -> None:
        self.process_var = process_var
        self.c = c
        self.obs_var = obs_var
        self.estimate = 0.0
        self.var = initial_var

    def step(self, y: float) -> float:
        predicted_var = self.var + self.process_var
        residual = y - self.estimate
        innovation_var = predicted_var + self.obs_var
        inverse_w2 = 1.0 + residual * residual / (self.c * self.c * innovation_var)
        gain = predicted_var / (predicted_var + self.obs_var * inverse_w2)
        self.estimate += gain * residual
        self.var = (1.0 - gain) * predicted_var
        return self.estimate


class RobustBOCD(Learner):
    """Run-length posterior over IMQ-weighted Kalman updates; decides on arrival.

    Our reimplementation of the WoLF-IMQ+BOCD idea, not the authors' code. Each
    run length keeps a Gaussian over the mean. Its pseudo-likelihood for the
    run-length weights is the marginal of the inflated observation model,
    N(y; mu, P + obs_var / w^2), which has heavy tails. No sample is withheld:
    every sample updates every component immediately.
    """

    def __init__(
        self,
        hazard: float,
        c: float,
        prior_var: float,
        obs_var: float = 1.0,
        max_components: int = 32,
    ) -> None:
        if not 0.0 < hazard < 1.0:
            raise ValueError("hazard must be in (0, 1)")
        self.log_hazard = math.log(hazard)
        self.log_survive = math.log1p(-hazard)
        self.c = c
        self.prior_var = prior_var
        self.obs_var = obs_var
        self.max_components = max_components
        self.mu = np.zeros(1)
        self.var = np.full(1, prior_var)
        self.log_w = np.zeros(1)

    def step(self, y: float) -> float:
        # The last entry is the run that starts at this step, from the prior.
        mu = np.append(self.mu, 0.0)
        var = np.append(self.var, self.prior_var)

        residual = y - mu
        inverse_w2 = 1.0 + residual**2 / (self.c**2 * (var + self.obs_var))
        weighted_var = var + self.obs_var * inverse_w2
        log_pred = -0.5 * (np.log(weighted_var) + _LOG_2PI + residual**2 / weighted_var)
        gain = var / weighted_var

        log_w = np.empty(mu.size)
        log_w[:-1] = self.log_w + self.log_survive + log_pred[:-1]
        # Every run can end here, and self.log_w is normalised, so the parents
        # contribute a total mass of 1.
        log_w[-1] = self.log_hazard + log_pred[-1]
        log_w -= _logsumexp(log_w)

        mu_post = mu + gain * residual
        var_post = (1.0 - gain) * var
        if log_w.size > self.max_components:
            keep = np.argpartition(log_w, -self.max_components)[-self.max_components :]
            mu_post, var_post, log_w = mu_post[keep], var_post[keep], log_w[keep]
            log_w -= _logsumexp(log_w)

        self.mu, self.var, self.log_w = mu_post, var_post, log_w
        return float(np.dot(np.exp(log_w), mu_post))


class Deferred(Learner):
    """Delta rule that withholds surprising samples until they are corroborated.

    A sample with |y - estimate| > threshold enters quarantine instead of
    updating the estimate. It stays for its next `window` steps and is then
    discarded. A commit fires when a quarantined sample has at least `k` later
    samples in quarantine such that all of them agree with each other (range at
    most `tolerance`) and each still disagrees with the current estimate. The
    agreeing group is then applied by the delta rule with rate `commit_alpha`
    (default `alpha`), in arrival order. A rate above `alpha` lets a corroborated
    change be adopted quickly rather than absorbed at the slow tracking rate.
    Samples outside the group, such as an outlier just before a changepoint, are
    not committed.
    """

    def __init__(
        self,
        alpha: float,
        threshold: float,
        k: int,
        window: int,
        tolerance: float,
        commit_alpha: float | None = None,
    ) -> None:
        if k < 1 or window < k:
            raise ValueError("need 1 <= k <= window")
        self.alpha = alpha
        self.commit_alpha = alpha if commit_alpha is None else commit_alpha
        self.threshold = threshold
        self.k = k
        self.window = window
        self.tolerance = tolerance
        self.estimate = 0.0
        self.quarantine: list[tuple[int, float]] = []  # (arrival index, value)
        self.quarantined_indices: list[int] = []
        self.committed_indices: list[int] = []
        self.n_discarded = 0
        self._t = -1

    def step(self, y: float) -> float:
        self._t += 1
        if abs(y - self.estimate) > self.threshold:
            self.quarantine.append((self._t, y))
            self.quarantined_indices.append(self._t)
        else:
            self.estimate += self.alpha * (y - self.estimate)
        self._release()
        return self.estimate

    def _release(self) -> None:
        oldest_allowed = self._t - self.window
        kept = [item for item in self.quarantine if item[0] >= oldest_allowed]
        self.n_discarded += len(self.quarantine) - len(kept)
        self.quarantine = kept

        # Every member lies within `window` steps of the oldest one, because
        # older samples have just been removed. So a group of k + 1 is a sample
        # plus k of its next `window`.
        if len(self.quarantine) < self.k + 1:
            return
        group = self._largest_agreeing_group()
        if len(group) >= self.k + 1:
            self._commit(group)

    def _largest_agreeing_group(self) -> set[int]:
        """Arrival indices of the largest set with range <= tolerance."""
        eligible = sorted(
            (item for item in self.quarantine if abs(item[1] - self.estimate) > self.threshold),
            key=lambda item: item[1],
        )
        best: list[tuple[int, float]] = []
        lo = 0
        for hi in range(len(eligible)):
            while eligible[hi][1] - eligible[lo][1] > self.tolerance:
                lo += 1
            if hi - lo + 1 > len(best):
                best = eligible[lo : hi + 1]
        return {item[0] for item in best}

    def _commit(self, members: set[int]) -> None:
        remaining: list[tuple[int, float]] = []
        for index, value in self.quarantine:  # arrival order
            if index in members:
                self.estimate += self.commit_alpha * (value - self.estimate)
                self.committed_indices.append(index)
            else:
                remaining.append((index, value))
        self.quarantine = remaining


class DeferredNoCheck(Deferred):
    """Ablation of `Deferred`: same quarantine and delay, no corroboration test.

    When the oldest quarantined sample has waited k steps, the whole quarantine
    commits. On a clean changepoint this matches `Deferred` step for step, so the
    two differ only in what the test would have rejected. `window` and
    `tolerance` are kept for a matched parameter list and have no effect.
    """

    def _release(self) -> None:
        if self.quarantine and self._t - self.quarantine[0][0] >= self.k:
            self._commit({index for index, _ in self.quarantine})


class Bayes(Learner):
    """Rao-Blackwellised mixture filter for the true generative model. The ceiling.

    At each step a changepoint (prior probability h) and an outlier (p) occur
    independently, which gives four joint outcomes: continue, outlier,
    changepoint, and changepoint-and-outlier.

    A component is one history of inlier assignments since its last changepoint,
    summarised by the inlier count n and sum s. With the U(-R, R) prior on m, its
    exact posterior is N(s/n, sigma^2/n) truncated to [-R, R] (uniform if n = 0).
    Keeping the truncation makes each component exact rather than approximately
    Gaussian, which matters for means near the edge of the prior.

    Likelihoods of y_t, with C(n, s) = P(|X| <= R) for X ~ N(s/n, sigma^2/n):
      continue, n >= 1:  N(y; s/n, sigma^2/n + sigma^2) C(n+1, s+y) / C(n, s)
      continue, n = 0:   C(1, y) / (2R)
      outlier:           1 / (2R) on [-R, R], otherwise 0
      changepoint:       C(1, y) / (2R)

    Changepoint branches do not depend on the parent, so all parents merge
    exactly into a single component, and a step grows K components to 2K + 2.
    `top_k` keeps the K heaviest and renormalises; None disables pruning, which
    makes the filter exact.
    """

    def __init__(
        self,
        hazard: float,
        outlier_prob: float,
        sigma: float = 1.0,
        radius: float = 20.0,
        top_k: int | None = 64,
    ) -> None:
        if not (0.0 <= hazard < 1.0 and 0.0 <= outlier_prob < 1.0):
            raise ValueError("hazard and outlier_prob must be in [0, 1)")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive or None")
        self.sigma2 = sigma * sigma
        self.radius = radius
        self.top_k = top_k
        self._log_uniform = -math.log(2.0 * radius)
        self._log_continue = math.log((1.0 - hazard) * (1.0 - outlier_prob))
        self._log_outlier = _safe_log((1.0 - hazard) * outlier_prob)
        self._log_changepoint = _safe_log(hazard * (1.0 - outlier_prob))
        self._log_both = _safe_log(hazard * outlier_prob)

        # One component: no inliers yet, so the posterior is the uniform prior.
        self.n = np.zeros(1)
        self.s = np.zeros(1)
        self.log_c = np.zeros(1)  # log C(n, s), and 0 by convention for n = 0
        self.mean = np.zeros(1)  # posterior mean of the component
        self.log_w = np.zeros(1)  # normalised
        self.log_evidence = 0.0  # log p(y_1:t); exact only without pruning

    def _branches(
        self, y: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Joint log weights log p(y_t, branch | past) and successor states."""
        n_next = self.n + 1.0
        s_next = self.s + y
        log_c_next, mean_next = _truncated_stats(
            s_next / n_next, self.sigma2 / n_next, self.radius
        )
        has_data = self.n > 0.0
        n_safe = np.where(has_data, self.n, 1.0)
        predictive_var = self.sigma2 / n_safe + self.sigma2
        log_gaussian = -0.5 * (
            np.log(predictive_var) + _LOG_2PI + (y - self.s / n_safe) ** 2 / predictive_var
        )
        log_likelihood = np.where(
            has_data,
            log_gaussian + log_c_next - self.log_c,
            self._log_uniform + log_c_next,
        )

        log_w = [self.log_w + self._log_continue + log_likelihood]
        n = [n_next]
        s = [s_next]
        log_c = [log_c_next]
        mean = [mean_next]

        in_support = abs(y) <= self.radius
        if in_support and self._log_outlier > -math.inf:
            # m is untouched and y carries no information about it.
            log_w.append(self.log_w + self._log_outlier + self._log_uniform)
            n.append(self.n)
            s.append(self.s)
            log_c.append(self.log_c)
            mean.append(self.mean)
        if self._log_changepoint > -math.inf:
            # self.log_w is normalised, so the merged parents have total mass 1.
            log_c_fresh, mean_fresh = _truncated_stats(
                np.array([float(y)]), np.array([self.sigma2]), self.radius
            )
            log_w.append(self._log_changepoint + self._log_uniform + log_c_fresh)
            n.append(np.ones(1))
            s.append(np.array([float(y)]))
            log_c.append(log_c_fresh)
            mean.append(mean_fresh)
        if in_support and self._log_both > -math.inf:
            # m was redrawn and y says nothing about the new value.
            log_w.append(np.array([self._log_both + self._log_uniform]))
            n.append(np.zeros(1))
            s.append(np.zeros(1))
            log_c.append(np.zeros(1))
            mean.append(np.zeros(1))

        return (
            np.concatenate(log_w),
            np.concatenate(n),
            np.concatenate(s),
            np.concatenate(log_c),
            np.concatenate(mean),
        )

    def predictive_density(self, y: float) -> float:
        """p(y_t = y | y_1:t-1) under the current state. Does not change the state."""
        return math.exp(_logsumexp(self._branches(y)[0]))

    def step(self, y: float) -> float:
        log_w, n, s, log_c, mean = self._branches(y)
        log_total = _logsumexp(log_w)
        if not math.isfinite(log_total):
            raise ValueError(f"y = {y} has zero density under the model")
        log_w -= log_total
        self.log_evidence += log_total

        if self.top_k is not None and log_w.size > self.top_k:
            keep = np.argpartition(log_w, -self.top_k)[-self.top_k :]
            log_w, n, s, log_c, mean = log_w[keep], n[keep], s[keep], log_c[keep], mean[keep]
            log_w -= _logsumexp(log_w)

        self.log_w, self.n, self.s, self.log_c, self.mean = log_w, n, s, log_c, mean
        return float(np.dot(np.exp(log_w), mean))


class OracleEvents:
    """Told the true event types; returns E[m_t | y_1:t, events].

    Deliberately not a `Learner`: its `step` takes the labels. It never receives
    the true mean, so its error is the unavoidable noise in estimating m_t.
    """

    def __init__(self, sigma: float = 1.0, radius: float = 20.0) -> None:
        self.sigma2 = sigma * sigma
        self.radius = radius
        self.n = 0
        self.s = 0.0

    def step(self, y: float, changepoint: bool, outlier: bool) -> float:
        if changepoint:
            self.n = 0
            self.s = 0.0
        if not outlier:
            self.n += 1
            self.s += y
        if self.n == 0:
            return 0.0  # the mean of the prior
        _, mean = _truncated_stats(
            np.array([self.s / self.n]), np.array([self.sigma2 / self.n]), self.radius
        )
        return float(mean[0])


LEARNERS: dict[str, type[Learner]] = {
    "fixed": Fixed,
    "monotone": Monotone,
    "robust": Robust,
    "robust_bocd": RobustBOCD,
    "deferred": Deferred,
    "deferred_nocheck": DeferredNoCheck,
    "bayes": Bayes,
}


def run_learner(learner: Learner, y: np.ndarray) -> np.ndarray:
    """Estimates for every t, each made after seeing y_t."""
    return np.fromiter((learner.step(v) for v in y.tolist()), dtype=float, count=y.size)


def run_oracle(
    oracle: OracleEvents, y: np.ndarray, changepoint: np.ndarray, outlier: np.ndarray
) -> np.ndarray:
    return np.fromiter(
        (
            oracle.step(v, c, o)
            for v, c, o in zip(y.tolist(), changepoint.tolist(), outlier.tolist())
        ),
        dtype=float,
        count=y.size,
    )
