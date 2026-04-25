#!/usr/bin/env python3
"""Calibrate ANTI threshold on WikiBio via REF vs GEN.

For each sample:
  1. REF: measure A_cov on the reference biography text.
  2. GEN: prompt Mistral-7B with "Write a 150-word biography of <name>",
     generate, then measure A_cov on the generation.

Output: CSV with one row per (sample, source). Source is REF or GEN.

The label column carries the source — it is the ground-truth split
for the calibration: REF biographies are factual by construction;
GEN biographies are at risk of confabulation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

GEN_PROMPT = "Write a 150-word biography of {name}.\n\nBiography:"


def get_name(sample) -> str | None:
    """Extract subject name from a WikiBio sample.

    WikiBio infobox is stored as a parallel-array dict with column_header
    and content lists. We look for a 'name' field; on miss, fall back to
    the first noun-phrase-ish chunk of the target text.
    """
    table = sample.get("input_text", {}).get("table", {})
    headers = table.get("column_header", []) or []
    contents = table.get("content", []) or []
    for h, c in zip(headers, contents):
        if h and c and h.lower().strip() in ("name", "fullname", "full_name", "birth_name"):
            return str(c).strip()
    target = sample.get("target_text", "") or ""
    if not target:
        return None
    head = target.split(".")[0]
    head = head.split(" is ")[0].split(" was ")[0]
    head = head.strip().strip(",").strip()
    return head[:80] if head else None


def measure_text(model, tokenizer, text: str, sigma_min: float,
                 device: str, max_tokens: int):
    """Measure A_cov on a raw text in a single forward pass."""
    import torch
    import torch.nn.functional as F
    from omega_cov.core import compute_acov

    inp = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_tokens).to(device)
    if inp["input_ids"].shape[1] < 5:
        return None

    with torch.no_grad():
        out = model(**inp)

    logits = out.logits[0].float()
    probs = torch.softmax(logits[:-1], dim=-1)
    sigma = -(probs * torch.log2(probs + 1e-9)).sum(dim=-1).cpu().numpy()

    h = out.hidden_states[-1][0].float()
    delta = (1 - F.cosine_similarity(h[:-1], h[1:], dim=-1)).cpu().numpy()

    return compute_acov(delta, sigma, sigma_min=sigma_min)


def measure_generated(model, tokenizer, prompt: str, generation: str,
                      sigma_min: float, device: str, max_tokens: int):
    """Measure A_cov on the generated continuation only, in context of the prompt."""
    import torch
    import torch.nn.functional as F
    from omega_cov.core import compute_acov

    full = prompt + generation
    inp = tokenizer(full, return_tensors="pt", truncation=True,
                    max_length=max_tokens).to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    n_prompt = len(prompt_ids)

    if inp["input_ids"].shape[1] - n_prompt < 5:
        return None

    with torch.no_grad():
        out = model(**inp)

    h = out.hidden_states[-1][0].float()
    logits = out.logits[0].float()

    h_gen = h[n_prompt - 1:]
    delta = (1 - F.cosine_similarity(h_gen[:-1], h_gen[1:], dim=-1)).cpu().numpy()

    probs = torch.softmax(logits[n_prompt - 1:-1], dim=-1)
    sigma = -(probs * torch.log2(probs + 1e-9)).sum(dim=-1).cpu().numpy()

    return compute_acov(delta, sigma, sigma_min=sigma_min)


def load_model(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device == "cpu":
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500,
                    help="Number of REF and GEN samples each (total = 2*n).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/wikibio_results.csv")
    ap.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--gen-temp", type=float, default=0.7)
    ap.add_argument("--sigma-min", type=float, default=1.0)
    ap.add_argument("--max-context", type=int, default=2048)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading WikiBio test split...")
    try:
        ds = load_dataset("wiki_bio", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  failed loading wiki_bio test split: {e}")
        print(f"  trying val split...")
        ds = load_dataset("wiki_bio", split="val", trust_remote_code=True)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=args.n, replace=False)
    print(f"  selected {args.n} samples (seed={args.seed})")

    print(f"Loading {args.model}...")
    model, tokenizer, device = load_model(args.model)
    print(f"  device={device}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["idx", "name", "source", "text", "n_tokens", "n_valid",
              "mean_delta", "mean_sigma", "A_marginal", "A_joint", "A_cov",
              "pearson_rho", "signature"]

    n_ref = n_gen = n_skip = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for i, idx in enumerate(indices):
            sample = ds[int(idx)]
            name = get_name(sample)
            target = (sample.get("target_text") or "").strip()
            if not name or len(target) < 80:
                n_skip += 1
                continue

            ref_result = measure_text(
                model, tokenizer, target,
                args.sigma_min, device, args.max_context,
            )
            if ref_result is not None:
                w.writerow({
                    "idx": int(idx),
                    "name": name[:200],
                    "source": "REF",
                    "text": target[:500],
                    "n_tokens": ref_result.n_tokens,
                    "n_valid": ref_result.n_valid,
                    "mean_delta": round(ref_result.mean_delta, 6),
                    "mean_sigma": round(ref_result.mean_sigma, 6),
                    "A_marginal": round(ref_result.A_marginal, 6),
                    "A_joint": round(ref_result.A_joint, 6),
                    "A_cov": round(ref_result.A_cov, 6),
                    "pearson_rho": round(ref_result.pearson_rho, 6),
                    "signature": ref_result.signature,
                })
                n_ref += 1

            prompt = GEN_PROMPT.format(name=name)
            inp = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen_ids = model.generate(
                    **inp,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.gen_temp,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generation = tokenizer.decode(
                gen_ids[0, inp["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            gen_result = measure_generated(
                model, tokenizer, prompt, generation,
                args.sigma_min, device, args.max_context,
            )
            if gen_result is not None:
                w.writerow({
                    "idx": int(idx),
                    "name": name[:200],
                    "source": "GEN",
                    "text": generation[:500],
                    "n_tokens": gen_result.n_tokens,
                    "n_valid": gen_result.n_valid,
                    "mean_delta": round(gen_result.mean_delta, 6),
                    "mean_sigma": round(gen_result.mean_sigma, 6),
                    "A_marginal": round(gen_result.A_marginal, 6),
                    "A_joint": round(gen_result.A_joint, 6),
                    "A_cov": round(gen_result.A_cov, 6),
                    "pearson_rho": round(gen_result.pearson_rho, 6),
                    "signature": gen_result.signature,
                })
                n_gen += 1

            f.flush()

            if (i + 1) % 25 == 0:
                msg = f"  [{i+1}/{args.n}]"
                if ref_result:
                    msg += f" REF A_cov={ref_result.A_cov:+.4f}"
                if gen_result:
                    msg += f" | GEN A_cov={gen_result.A_cov:+.4f}"
                msg += f" (REF={n_ref} GEN={n_gen} skip={n_skip})"
                print(msg)

    print(f"\nDone. Wrote {out_path}")
    print(f"  REF rows={n_ref}  GEN rows={n_gen}  skipped samples={n_skip}")


if __name__ == "__main__":
    main()
