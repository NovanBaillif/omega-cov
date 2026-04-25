"""Unit tests for omega_cov.core — pure computation tests, no model required.

These tests validate:
1. The mathematical identity A_joint - A_marginal == cov(delta, sigma).
2. Correct classification of regimes by signature.
3. The pathological case Mistral pointed out (same A_marginal, opposite A_joint).
4. Numerical stability on edge cases.
"""

import numpy as np
import pytest

from omega_cov.core import compute_acov
from omega_cov.thresholds import THRESHOLD_DENSE, THRESHOLD_ANTI


# ─────────────────────────────────────────────────────────────────────────
# Test 1 — Mathematical identity
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_identity_acov_equals_covariance(seed: int) -> None:
    """A_cov must equal np.cov(delta, sigma) to numerical precision."""
    rng = np.random.default_rng(seed)
    n = rng.integers(50, 500)
    delta = rng.uniform(0, 1, n)
    sigma = rng.uniform(1.5, 8, n)  # all above sigma_min=1

    result = compute_acov(delta, sigma)
    assert result is not None

    cov_numpy = float(np.cov(delta, sigma, ddof=0)[0, 1])
    assert abs(result.A_cov - cov_numpy) < 1e-9


# ─────────────────────────────────────────────────────────────────────────
# Test 2 — Regime classification
# ─────────────────────────────────────────────────────────────────────────


def test_dense_regime() -> None:
    """Strong positive covariance → A_cov >= THRESHOLD_DENSE → signature DENSE."""
    rng = np.random.default_rng(123)
    n = 500
    sigma = rng.uniform(1.5, 8, n)
    delta = np.clip(0.1 + 0.08 * sigma + rng.normal(0, 0.05, n), 0.001, 1.0)

    result = compute_acov(delta, sigma)
    assert result is not None
    assert result.A_cov >= THRESHOLD_DENSE
    assert result.signature == "DENSE"
    assert result.pearson_rho > 0.5


def test_anti_regime() -> None:
    """True anti-correlation → A_cov < 0 → signature ANTI."""
    rng = np.random.default_rng(124)
    n = 500
    sigma = rng.uniform(1.5, 8, n)
    delta = np.clip(0.8 - 0.08 * sigma + rng.normal(0, 0.05, n), 0.001, 1.0)

    result = compute_acov(delta, sigma)
    assert result is not None
    assert result.A_cov < THRESHOLD_ANTI
    assert result.signature == "ANTI"
    assert result.pearson_rho < -0.5


def test_weak_regime() -> None:
    """Faint positive covariance → 0 <= A_cov < THRESHOLD_DENSE → signature WEAK."""
    rng = np.random.default_rng(125)
    n = 500
    sigma = rng.uniform(1.5, 8, n)
    # Weak positive coupling — delta has a tiny linear dependence on sigma plus noise.
    delta = np.clip(0.4 + 0.005 * sigma + rng.normal(0, 0.1, n), 0.001, 1.0)

    result = compute_acov(delta, sigma)
    assert result is not None
    assert THRESHOLD_ANTI <= result.A_cov < THRESHOLD_DENSE
    assert result.signature == "WEAK"


# ─────────────────────────────────────────────────────────────────────────
# Test 3 — Pathological case: same A_marginal, different dynamics
# ─────────────────────────────────────────────────────────────────────────


def test_pathological_same_marginals_different_joint() -> None:
    """Two profiles with identical A_marginal can have very different A_joint.

    This is the core motivation for using A_joint / A_cov instead of
    A_marginal alone.
    """
    rng = np.random.default_rng(7)
    n = 200

    sigma_high = rng.uniform(6, 8, n // 2)
    sigma_low = rng.uniform(1.5, 2.5, n // 2)
    delta_low = rng.uniform(0.05, 0.15, n // 2)
    delta_high = rng.uniform(0.7, 0.9, n // 2)

    sigma = np.concatenate([sigma_high, sigma_low])

    # Profile 1: anti-aligned (high sigma paired with low delta)
    delta_p1 = np.concatenate([delta_low, delta_high])

    # Profile 2: aligned (high sigma paired with high delta)
    delta_p2 = np.concatenate([delta_high, delta_low])

    r1 = compute_acov(delta_p1, sigma)
    r2 = compute_acov(delta_p2, sigma)

    assert r1 is not None and r2 is not None

    # Same marginals
    assert abs(r1.A_marginal - r2.A_marginal) < 0.01

    # Very different joint
    assert abs(r1.A_joint - r2.A_joint) > 0.5

    # Opposite signatures
    assert r1.signature == "ANTI"
    assert r2.signature == "DENSE"


# ─────────────────────────────────────────────────────────────────────────
# Test 4 — Numerical stability and edge cases
# ─────────────────────────────────────────────────────────────────────────


def test_long_sequence() -> None:
    """Sequences of realistic length (2048 tokens) produce no NaN or inf."""
    rng = np.random.default_rng(1)
    n = 2048
    delta = rng.uniform(0.001, 1.0, n)
    sigma = rng.uniform(1.5, 12, n)

    result = compute_acov(delta, sigma)
    assert result is not None
    assert not np.isnan(result.A_joint)
    assert not np.isnan(result.A_cov)
    assert not np.isinf(result.A_joint)


def test_all_below_threshold_returns_none() -> None:
    """When all sigma values are below sigma_min, return None."""
    rng = np.random.default_rng(2)
    sigma = rng.uniform(0.1, 0.9, 100)
    delta = rng.uniform(0, 1, 100)

    result = compute_acov(delta, sigma, sigma_min=1.0)
    assert result is None


def test_zero_delta() -> None:
    """All-zero delta produces zero A_marginal and A_joint."""
    rng = np.random.default_rng(3)
    delta = np.zeros(50)
    sigma = rng.uniform(1.5, 5, 50)

    result = compute_acov(delta, sigma)
    assert result is not None
    assert result.A_marginal == 0.0
    assert result.A_joint == 0.0
    assert abs(result.A_cov) < 1e-12


def test_constant_delta() -> None:
    """Constant delta produces undefined Pearson rho, returned as 0."""
    rng = np.random.default_rng(4)
    delta = np.full(100, 0.5)
    sigma = rng.uniform(1.5, 8, 100)

    result = compute_acov(delta, sigma)
    assert result is not None
    assert result.pearson_rho == 0.0
    assert abs(result.A_cov) < 1e-9


def test_too_short_returns_none() -> None:
    """Fewer than 2 tokens returns None."""
    delta = np.array([0.5])
    sigma = np.array([2.0])

    result = compute_acov(delta, sigma)
    assert result is None


def test_misaligned_lengths_truncated() -> None:
    """Delta and sigma of different lengths are truncated to the shorter."""
    rng = np.random.default_rng(5)
    delta = rng.uniform(0, 1, 100)
    sigma = rng.uniform(1.5, 5, 80)

    result = compute_acov(delta, sigma)
    assert result is not None
    # n_tokens reflects the truncated length
    assert result.n_tokens == 80
