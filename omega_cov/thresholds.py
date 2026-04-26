"""Signature classification thresholds.

A_cov = cov(delta, sigma) splits into three regimes:

    DENSE  : A_cov >= THRESHOLD_DENSE
             surprise and displacement covary tightly.
    WEAK   : THRESHOLD_ANTI <= A_cov < THRESHOLD_DENSE
             positive but low covariance; surprise occurs without
             commensurate displacement.
    ANTI   : A_cov < THRESHOLD_ANTI
             true anti-correlation — surprise and displacement actively
             oppose each other. Pathological by definition; rare on
             standard generative text.

THRESHOLD_DENSE is calibrated empirically per model — it is the boundary
between two production regimes under a given model. THRESHOLD_ANTI is
mathematical: A_cov < 0 means true anti-correlation regardless of model.
"""

# Calibrated on WikiBio (n=491 paired REF/GEN, Mistral-7B-v0.1, seed=42).
# Youden's J optimum, AUC = 0.918. See docs/calibration.md.
THRESHOLD_DENSE = 0.069

# Mathematical definition. Not calibrated.
THRESHOLD_ANTI = 0.0


def classify_signature(a_cov: float) -> str:
    """Classify A_cov into DENSE / WEAK / ANTI.

    Args:
        a_cov: the covariance value to classify.

    Returns:
        "DENSE" if a_cov >= THRESHOLD_DENSE,
        "ANTI"  if a_cov <  THRESHOLD_ANTI,
        "WEAK"  otherwise.
    """
    if a_cov >= THRESHOLD_DENSE:
        return "DENSE"
    if a_cov < THRESHOLD_ANTI:
        return "ANTI"
    return "WEAK"
