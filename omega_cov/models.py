"""Model adapters for HuggingFace causal LMs.

This module bridges between a HuggingFace model (which provides
hidden states and logits) and the pure computation in core.py.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from omega_cov.core import MeasureResult, compute_acov

# Default model. Restricted to Mistral-7B for v0.1.
# Broader support (Llama, Qwen, Gemma) planned in 0.2.
DEFAULT_MODEL = "mistralai/Mistral-7B-v0.1"


class OmegaCov:
    """Reusable measurer that loads a model once and processes many texts.

    Example:
        oc = OmegaCov("mistralai/Mistral-7B-v0.1")
        for text in corpus:
            result = oc.measure(text)
            print(result.A_cov)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_tokens: int = 2048,
        sigma_min: float = 1.0,
    ):
        """Load a model once, ready to measure many texts.

        Args:
            model: HuggingFace model name. Tested with Mistral-7B.
            device: "cuda", "cpu", or None for auto-detection.
            max_tokens: truncate inputs to this many tokens.
            sigma_min: filter threshold for sigma. Tokens with
                Shannon entropy <= sigma_min are excluded.
        """
        # Lazy import — keeps the package importable without torch/transformers
        # if the user only needs the pure compute_acov function.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.max_tokens = max_tokens
        self.sigma_min = sigma_min
        self.model_name = model

        kwargs = {"output_hidden_states": True, "trust_remote_code": True}
        if device == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["device_map"] = "auto"
                kwargs["torch_dtype"] = torch.float16
            except ImportError:
                kwargs["device_map"] = "auto"
                kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        if device == "cpu":
            self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model.eval()

    def measure(self, text: str) -> Optional[MeasureResult]:
        """Measure A_cov on a single text.

        Args:
            text: the text to analyze.

        Returns:
            A MeasureResult, or None if the text is too short or has
            insufficient tokens after filtering.
        """
        import torch
        import torch.nn.functional as F

        if not text or len(text.strip()) == 0:
            return None

        inp = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
        ).to(self.device)

        n_tokens = inp["input_ids"].shape[1]
        if n_tokens < 5:
            return None

        with torch.no_grad():
            out = self.model(**inp)

        # sigma: Shannon entropy of predictive distribution at each position
        logits = out.logits[0].float()
        probs = torch.softmax(logits[:-1], dim=-1)
        sigma = -(probs * torch.log2(probs + 1e-9)).sum(dim=-1).cpu().numpy()

        # delta: cosine displacement between consecutive last-layer hidden states
        h = out.hidden_states[-1][0].float()
        delta = (1 - F.cosine_similarity(h[:-1], h[1:], dim=-1)).cpu().numpy()

        return compute_acov(delta, sigma, sigma_min=self.sigma_min)


# One-shot convenience function.
_DEFAULT_INSTANCE: Optional[OmegaCov] = None


def measure(
    text: str,
    model: Union[str, OmegaCov] = DEFAULT_MODEL,
) -> Optional[MeasureResult]:
    """One-shot measurement on a single text.

    Loads the model the first time it is called; subsequent calls reuse the
    same instance if the model name has not changed. For batch processing,
    instantiate OmegaCov directly.

    Args:
        text: the text to analyze.
        model: HuggingFace model name, or an existing OmegaCov instance.

    Returns:
        A MeasureResult, or None on failure.
    """
    global _DEFAULT_INSTANCE

    if isinstance(model, OmegaCov):
        return model.measure(text)

    if _DEFAULT_INSTANCE is None or _DEFAULT_INSTANCE.model_name != model:
        _DEFAULT_INSTANCE = OmegaCov(model=model)

    return _DEFAULT_INSTANCE.measure(text)
