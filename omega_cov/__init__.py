"""omega-cov: A token-level signature for LLM outputs based on the covariance
of cosine displacement and Shannon surprise.

A_cov is a production-regime signature, not a veracity test. See README and
docs/calibration.md for the empirical scope.

Public API:
    measure(text, model)    : one-shot measurement
    OmegaCov(model)         : reusable measurer for batch processing
    MeasureResult           : the result object returned by measure()
"""

from omega_cov.core import MeasureResult
from omega_cov.models import OmegaCov, measure

__version__ = "0.1.0"
__all__ = ["measure", "OmegaCov", "MeasureResult"]
