"""omega-cov: A token-level metric for detecting confabulation in LLM outputs
via covariance of cosine displacement and Shannon surprise.

Public API:
    measure(text, model)    : one-shot measurement
    OmegaCov(model)         : reusable measurer for batch processing
    MeasureResult           : the result object returned by measure()
"""

from omega_cov.core import MeasureResult
from omega_cov.models import OmegaCov, measure

__version__ = "0.1.0"
__all__ = ["measure", "OmegaCov", "MeasureResult"]
