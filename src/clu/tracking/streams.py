"""Stage A stream generator, per hypotheses/H1_corroboration_gating.md.

    y_t = m_t + N(0, sigma^2)
    changepoint with hazard h: m_t is redrawn from U(-R, R)
    outlier with probability p: y_t is replaced by a draw from U(-R, R)

Changepoint and outlier indicators are drawn independently at every step, the
literal reading of the preregistration, so both can occur at one step. m_0 is
drawn from U(-R, R) before t = 0, and t = 0 follows the same rule as any other
step (a changepoint there is one more redraw from the same distribution).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HAZARD = 0.05
OUTLIER_PROB = 0.05
SIGMA = 1.0
RADIUS = 20.0
DEFAULT_T = 2000

# context name -> (hazard, outlier probability)
CONTEXTS: dict[str, tuple[float, float]] = {
    "changepoint": (HAZARD, 0.0),
    "outlier": (0.0, OUTLIER_PROB),
    "mixed": (HAZARD, OUTLIER_PROB),
    "stationary": (0.0, 0.0),
}


@dataclass(frozen=True)
class Stream:
    """Observations plus ground truth. Only `y` may reach a non-oracle learner."""

    y: np.ndarray
    m: np.ndarray  # noise-free mean, used for scoring only
    changepoint: np.ndarray  # event label, used by the oracle only
    outlier: np.ndarray  # event label, used by the oracle only


def generate_stream(
    context: str,
    rng: np.random.Generator,
    T: int = DEFAULT_T,
    sigma: float = SIGMA,
    radius: float = RADIUS,
) -> Stream:
    hazard, outlier_prob = CONTEXTS[context]
    # Every array is always drawn, so random-number consumption does not depend
    # on the context.
    changepoint = rng.random(T) < hazard
    outlier = rng.random(T) < outlier_prob
    segment_means = rng.uniform(-radius, radius, size=T + 1)
    noise = rng.normal(0.0, sigma, size=T)
    outlier_draws = rng.uniform(-radius, radius, size=T)

    # Segment 0 holds m_0; each changepoint advances to the next fresh draw.
    m = segment_means[np.cumsum(changepoint)]
    y = np.where(outlier, outlier_draws, m + noise)
    return Stream(y=y, m=m, changepoint=changepoint, outlier=outlier)
