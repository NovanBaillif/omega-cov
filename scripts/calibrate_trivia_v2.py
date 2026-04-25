#!/usr/bin/env python3
"""TriviaQA v0.2 — intra-LLM calibration of A_cov on correct vs hallucinated answers.

Differences from calibrate_trivia.py (v0.1):

  - Prompt forces 2-3 sentence answers with reasoning, with min_new_tokens=30,
    so generations have enough valid tokens to survive the sigma_min filter.
  - No alias_match fast path. Every generation is judged by Claude Haiku 4.5.
    This eliminates the v0.1 bias where label_source was perfectly correlated
    with label (alias_match captured short corrects, the judge only saw long
    misses).
  - Different seed (43) so the sample is independent of the v0.1 partial run.
  - n_valid is logged per row so the post-hoc check
    cor(n_valid, label) < 0.1 can verify the bias is gone.

Required env: ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

PROMPT_TEMPLATE = (
    "Answer the following question in 2-3 sentences with reasoning.\n"
    "Question: {q}\n"
    "Answer:"
)

JUDGE_PROMPT = """Question: {q}
Ground truth aliases: {gt}
Model answer: {ans}

Is the model answer factually correct given the ground truth?
Reply with JSON only: {{"verdict": "correct" | "hallucination" | "ambiguous", "reason": "..."}}"""


def parse_judge_output(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def llm_judge(client, model_id: str, question: str, gt_str: str, answer: str) -> dict:
    msg = client.messages.create(
        model=model_id,
        max_tokens=200,
        temperature=0,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(q=question, gt=gt_str, ans=answer),
        }],
    )
    return parse_judge_output(msg.content[0].text)


def measure_generated(model, tokenizer, prompt: str, generation: str,
                      sigma_min: float, device: str, max_tokens: int):
    """A_cov on the generated continuation only, in context of the prompt."""
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
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", default="data/trivia_v2_results.csv")
    ap.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--judge-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--min-new-tokens", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--gen-temp", type=float, default=0.7)
    ap.add_argument("--sigma-min", type=float, default=1.0)
    ap.add_argument("--max-context", type=int, default=2048)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set in environment.")

    import torch
    import anthropic
    from datasets import load_dataset

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Loading TriviaQA validation split (rc.nocontext)...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=args.n, replace=False)
    print(f"  selected {args.n} samples (seed={args.seed})")

    print(f"Loading {args.model}...")
    model, tokenizer, device = load_model(args.model)
    print(f"  device={device}")

    client = anthropic.Anthropic()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["idx", "question", "gt_value", "gt_aliases", "generation",
              "n_gen_tokens", "n_tokens", "n_valid",
              "mean_delta", "mean_sigma", "A_marginal", "A_joint", "A_cov",
              "pearson_rho", "signature", "verdict", "judge_reason"]

    n_correct = n_halluc = n_ambig = n_skip = n_err = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for i, idx in enumerate(indices):
            row = ds[int(idx)]
            q = row["question"]
            answer_obj = row["answer"]
            gt_value = answer_obj.get("value", "") or ""
            aliases = list(answer_obj.get("aliases", []) or [])
            normalized = list(answer_obj.get("normalized_aliases", []) or [])
            all_aliases = list({a for a in (aliases + normalized + [gt_value]) if a})
            gt_str = " | ".join(all_aliases)[:500]

            prompt = PROMPT_TEMPLATE.format(q=q)
            inp = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen_ids = model.generate(
                    **inp,
                    min_new_tokens=args.min_new_tokens,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.gen_temp,
                    pad_token_id=tokenizer.pad_token_id,
                )
            n_gen_tokens = int(gen_ids.shape[1] - inp["input_ids"].shape[1])
            generation = tokenizer.decode(
                gen_ids[0, inp["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            result = measure_generated(
                model, tokenizer, prompt, generation,
                args.sigma_min, device, args.max_context,
            )
            if result is None:
                n_skip += 1
                continue

            try:
                j = llm_judge(client, args.judge_model, q, gt_str, generation)
                verdict = j.get("verdict", "ambiguous")
                judge_reason = j.get("reason", "")
            except Exception as e:
                verdict = "error"
                judge_reason = str(e)[:300]
                n_err += 1

            if verdict == "correct":
                n_correct += 1
            elif verdict == "hallucination":
                n_halluc += 1
            elif verdict == "ambiguous":
                n_ambig += 1

            w.writerow({
                "idx": int(idx),
                "question": q[:500],
                "gt_value": gt_value,
                "gt_aliases": gt_str,
                "generation": generation[:1000],
                "n_gen_tokens": n_gen_tokens,
                "n_tokens": result.n_tokens,
                "n_valid": result.n_valid,
                "mean_delta": round(result.mean_delta, 6),
                "mean_sigma": round(result.mean_sigma, 6),
                "A_marginal": round(result.A_marginal, 6),
                "A_joint": round(result.A_joint, 6),
                "A_cov": round(result.A_cov, 6),
                "pearson_rho": round(result.pearson_rho, 6),
                "signature": result.signature,
                "verdict": verdict,
                "judge_reason": judge_reason[:300],
            })
            f.flush()

            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{args.n}] A_cov={result.A_cov:+.4f} "
                      f"sig={result.signature} v={verdict} "
                      f"(C={n_correct} H={n_halluc} A={n_ambig} "
                      f"skip={n_skip} err={n_err})")

    print(f"\nDone. Wrote {out_path}")
    print(f"  correct={n_correct}  hallucination={n_halluc}  "
          f"ambiguous={n_ambig}  skipped={n_skip}  errors={n_err}")


if __name__ == "__main__":
    main()
