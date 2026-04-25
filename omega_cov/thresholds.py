"""Signature classification thresholds.

These thresholds determine which regime A_cov falls into.

Current values are placeholders set by inspection of synthetic data.
They will be calibrated against TriviaQA and WikiBio before v1.0.
The constants are isolated here so that calibration only requires
updating this file.
"""

# Default thresholds (placeholder, to be calibrated).
THRESHOLD_ALIGNED = 0.1
THRESHOLD_ANTI = -0.1


def classify_signature(a_cov: float) -> str:
    """Classify A_cov value into one of three regimes.

    Args:
        a_cov: the covariance value to classify.

    Returns:
        "ALIGNED" if a_cov > THRESHOLD_ALIGNED,
        "ANTI"    if a_cov < THRESHOLD_ANTI,
        "MIXED"   otherwise.
    """
    if a_cov > THRESHOLD_ALIGNED:
        return "ALIGNED"
    if a_cov < THRESHOLD_ANTI:
        return "ANTI"
    return "MIXED"
