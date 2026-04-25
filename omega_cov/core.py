"""Core computation of A_marginal, A_joint, A_cov.

This module contains the pure mathematical logic. It takes raw
arrays (delta, sigma) as input and returns a MeasureResult.
It has no dependency on torch or transformers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from omega_cov.thresholds import classify_signature


@dataclass
class MeasureResult:
    """Result of an A_cov measurement on a token sequence.

    Attributes:
        n_tokens: total number of tokens in the original sequence.
        n_valid: number of tokens retained after sigma filtering.
        pct_filtered: percentage of tokens filtered out.
        mean_delta: mean cosine displacement on filtered tokens.
        mean_sigma: mean Shannon entropy (in bits) on filtered tokens.
        A_marginal: mean(delta) * mean(sigma).
        A_joint: mean(delta * sigma).
        A_cov: A_joint - A_marginal, equal to cov(delta, sigma).
        pearson_rho: Pearson correlation between delta and sigma.
        signature: one of "DENSE", "WEAK", "ANTI" based on A_cov.
    """

    n_tokens: int
    n_valid: int
    pct_filtered: float
    mean_delta: float
    mean_sigma: float
    A_marginal: float
    A_joint: float
    A_cov: float
    pearson_rho: float
    signature: str

    def __repr__(self) -> str:
        return (
            f"MeasureResult(n_tokens={self.n_tokens}, n_valid={self.n_valid}, "
            f"A_marginal={self.A_marginal:.4f}, A_joint={self.A_joint:.4f}, "
            f"A_cov={self.A_cov:+.4f}, signature={self.signature!r})"
        )


def compute_acov(
    delta: np.ndarray,
    sigma: np.ndarray,
    sigma_min: float = 1.0,
) -> Optional[MeasureResult]:
    """Compute A_marginal, A_joint, A_cov from raw delta and sigma arrays.

    Args:
        delta: cosine displacement values, shape (n,). Each entry is
            1 - cos(h_i, h_{i-1}) where h_i is the i-th hidden state.
        sigma: Shannon entropy in bits, shape (n,). Each entry is the
            entropy of the predictive distribution at position i.
        sigma_min: filter threshold. Tokens with sigma <= sigma_min are
            excluded from the computation. Default 1.0 bit.

    Returns:
        A MeasureResult, or None if fewer than 2 tokens remain after
        filtering.
    """
    delta = np.asarray(delta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    L = min(len(delta), len(sigma))
    if L < 2:
        return None

    delta = delta[:L]
    sigma = sigma[:L]

    mask = sigma > sigma_min
    n_valid = int(np.sum(mask))
    n_tokens = L
    pct_filtered = round(100.0 * (1 - n_valid / max(L, 1)), 1)

    if n_valid < 2:
        return None

    dv = delta[mask]
    sv = sigma[mask]

    mean_delta = float(np.mean(dv))
    mean_sigma = float(np.mean(sv))
    A_marginal = mean_delta * mean_sigma
    A_joint = float(np.mean(dv * sv))
    A_cov = A_joint - A_marginal

    if np.std(dv) > 1e-10 and np.std(sv) > 1e-10:
        rho = float(np.corrcoef(dv, sv)[0, 1])
    else:
        rho = 0.0

    signature = classify_signature(A_cov)

    return MeasureResult(
        n_tokens=n_tokens,
        n_valid=n_valid,
        pct_filtered=pct_filtered,
        mean_delta=mean_delta,
        mean_sigma=mean_sigma,
        A_marginal=A_marginal,
        A_joint=A_joint,
        A_cov=A_cov,
        pearson_rho=rho,
        signature=signature,
    )
