"""Hash-power distributions used by the experiments."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def assign_attackers(n_nodes: int, attacker_count: int) -> np.ndarray:
    """Mark the first ``attacker_count`` nodes for adversarial strategies."""
    if n_nodes < 0 or not 0 <= attacker_count <= n_nodes:
        raise ValueError("attacker_count must be between 0 and n_nodes")
    flags = np.zeros(n_nodes, dtype=bool)
    flags[:attacker_count] = True
    return flags


def assign_top_share_lognormal(
    total_hash_power: float,
    n_nodes: int,
    top_pct: float,
    top_share: float,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Sample a log-normal population with the requested population quantile.

    ``top_share`` is a population-level target. A finite sample will generally
    have a nearby, but not exactly equal, realized top share.
    """
    if total_hash_power <= 0 or n_nodes <= 0:
        raise ValueError("total_hash_power and n_nodes must be positive")
    if not 0 < top_pct < 100 or not 0 < top_share < 1:
        raise ValueError("top_pct and top_share must be inside their ranges")

    # For a log-normal population, this sigma makes the upper ``top_pct`` tail
    # carry ``top_share`` in expectation. The finite sample is normalized below.
    quantile = top_pct / 100.0
    sigma = float(norm.ppf(top_share) - norm.ppf(quantile))
    mu = -0.5 * sigma**2
    samples = np.random.default_rng(seed).lognormal(mu, sigma, n_nodes)
    return samples / samples.sum() * total_hash_power, sigma


def realized_top_share(hash_power: np.ndarray, top_pct: float) -> float:
    """Measure the actual share held by the largest ``top_pct`` percent."""
    hash_power = np.asarray(hash_power, dtype=float)
    top_count = max(1, round(len(hash_power) * top_pct / 100.0))
    return float(np.sort(hash_power)[-top_count:].sum() / hash_power.sum())
