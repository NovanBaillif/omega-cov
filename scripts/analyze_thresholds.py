#!/usr/bin/env python3
"""Compute the optimal ANTI threshold from calibration CSVs.

Reads the outputs of calibrate_trivia.py and calibrate_wikibio.py,
computes Youden's J for the binary task "is this likely a confabulation?"
on each dataset, and prints a recommended threshold.

Convention:
  - "positive" class = should be flagged ANTI (hallucination on TriviaQA,
    GEN on WikiBio)
  - "negative" class = should NOT be flagged (correct on TriviaQA, REF on
    WikiBio)
  - decision rule: flag as ANTI iff A_cov < threshold
  - TPR = P(A_cov < t | positive)
  - FPR = P(A_cov < t | negative)
  - Youden's J = TPR - FPR, maximized over t.

Outputs:
  - Per-dataset descriptive stats, ROC summary at the optimum, and
    Youden's J.
  - A combined recommendation (per-dataset average, conservative bound).
  - JSON dump at data/calibration_results.json so the result is
    machine-readable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def quartiles(arr: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(x) for x in np.quantile(arr, [0.25, 0.5, 0.75]))


def youden_j(scores_pos: np.ndarray, scores_neg: np.ndarray,
             n_thresholds: int = 400) -> dict:
    """Find threshold t maximizing TPR - FPR for the rule (score < t -> positive)."""
    if len(scores_pos) < 5 or len(scores_neg) < 5:
        return {"error": "insufficient samples", "n_pos": int(len(scores_pos)),
                "n_neg": int(len(scores_neg))}

    all_scores = np.concatenate([scores_pos, scores_neg])
    lo, hi = float(all_scores.min()), float(all_scores.max())
    pad = (hi - lo) * 0.02 if hi > lo else 0.01
    thresholds = np.linspace(lo - pad, hi + pad, n_thresholds)

    best = {"j": -np.inf, "threshold": None, "tpr": None, "fpr": None}
    for t in thresholds:
        tpr = float(np.mean(scores_pos < t))
        fpr = float(np.mean(scores_neg < t))
        j = tpr - fpr
        if j > best["j"]:
            best = {"j": j, "threshold": float(t), "tpr": tpr, "fpr": fpr}
    return best


def analyze_trivia(rows: list[dict]) -> dict:
    valid = [r for r in rows if r.get("label") in ("correct", "hallucination", "ambiguous")]
    correct = np.array([float(r["A_cov"]) for r in valid if r["label"] == "correct"])
    halluc = np.array([float(r["A_cov"]) for r in valid if r["label"] == "hallucination"])
    ambig = np.array([float(r["A_cov"]) for r in valid if r["label"] == "ambiguous"])

    n_total = len(rows)
    n_err = sum(1 for r in rows if r.get("label") == "error")

    out = {
        "dataset": "TriviaQA",
        "n_rows": n_total,
        "n_correct": int(len(correct)),
        "n_hallucination": int(len(halluc)),
        "n_ambiguous": int(len(ambig)),
        "n_errors": int(n_err),
    }
    if len(correct) >= 5:
        q1, q2, q3 = quartiles(correct)
        out["correct_acov"] = {"median": q2, "q1": q1, "q3": q3,
                               "mean": float(correct.mean()),
                               "std": float(correct.std())}
    if len(halluc) >= 5:
        q1, q2, q3 = quartiles(halluc)
        out["hallucination_acov"] = {"median": q2, "q1": q1, "q3": q3,
                                     "mean": float(halluc.mean()),
                                     "std": float(halluc.std())}
    if len(correct) >= 5 and len(halluc) >= 5:
        out["youden"] = youden_j(scores_pos=halluc, scores_neg=correct)
    return out


def analyze_wikibio(rows: list[dict]) -> dict:
    ref = np.array([float(r["A_cov"]) for r in rows if r.get("source") == "REF"])
    gen = np.array([float(r["A_cov"]) for r in rows if r.get("source") == "GEN"])

    out = {
        "dataset": "WikiBio",
        "n_rows": len(rows),
        "n_ref": int(len(ref)),
        "n_gen": int(len(gen)),
    }
    if len(ref) >= 5:
        q1, q2, q3 = quartiles(ref)
        out["ref_acov"] = {"median": q2, "q1": q1, "q3": q3,
                           "mean": float(ref.mean()), "std": float(ref.std())}
    if len(gen) >= 5:
        q1, q2, q3 = quartiles(gen)
        out["gen_acov"] = {"median": q2, "q1": q1, "q3": q3,
                           "mean": float(gen.mean()), "std": float(gen.std())}
    if len(ref) >= 5 and len(gen) >= 5:
        out["youden"] = youden_j(scores_pos=gen, scores_neg=ref)
    return out


def fmt_iqr(d: dict) -> str:
    return (f"median={d['median']:+.4f} "
            f"[Q1={d['q1']:+.4f}, Q3={d['q3']:+.4f}] "
            f"mean={d['mean']:+.4f} sd={d['std']:.4f}")


def fmt_youden(y: dict) -> str:
    if "error" in y:
        return f"  Youden J: skipped ({y['error']})"
    return (f"  Youden J optimum:\n"
            f"    threshold = {y['threshold']:+.4f}\n"
            f"    TPR (caught) = {y['tpr']:.3f}\n"
            f"    FPR (false flag) = {y['fpr']:.3f}\n"
            f"    J = {y['j']:.3f}")


def print_analysis(a: dict):
    print(f"\n=== {a['dataset']} ===")
    print(f"  rows: {a['n_rows']}")
    if a["dataset"] == "TriviaQA":
        print(f"  correct={a['n_correct']}  hallucination={a['n_hallucination']}  "
              f"ambiguous={a['n_ambiguous']}  errors={a['n_errors']}")
        if "correct_acov" in a:
            print(f"  correct A_cov:        {fmt_iqr(a['correct_acov'])}")
        if "hallucination_acov" in a:
            print(f"  hallucination A_cov:  {fmt_iqr(a['hallucination_acov'])}")
    else:
        print(f"  REF={a['n_ref']}  GEN={a['n_gen']}")
        if "ref_acov" in a:
            print(f"  REF A_cov:  {fmt_iqr(a['ref_acov'])}")
        if "gen_acov" in a:
            print(f"  GEN A_cov:  {fmt_iqr(a['gen_acov'])}")
    if "youden" in a:
        print(fmt_youden(a["youden"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trivia", default="data/trivia_results.csv")
    ap.add_argument("--wikibio", default="data/wikibio_results.csv")
    ap.add_argument("--out", default="data/calibration_results.json")
    args = ap.parse_args()

    results = {}
    thresholds = []

    trivia_path = Path(args.trivia)
    if trivia_path.exists():
        rows = load_csv(trivia_path)
        a = analyze_trivia(rows)
        print_analysis(a)
        results["trivia"] = a
        if "youden" in a and "threshold" in a["youden"]:
            thresholds.append(("trivia", a["youden"]["threshold"]))
    else:
        print(f"WARN: {trivia_path} not found, skipping TriviaQA.")

    wikibio_path = Path(args.wikibio)
    if wikibio_path.exists():
        rows = load_csv(wikibio_path)
        a = analyze_wikibio(rows)
        print_analysis(a)
        results["wikibio"] = a
        if "youden" in a and "threshold" in a["youden"]:
            thresholds.append(("wikibio", a["youden"]["threshold"]))
    else:
        print(f"WARN: {wikibio_path} not found, skipping WikiBio.")

    print("\n=== Recommended threshold ===")
    if thresholds:
        ts = [t for _, t in thresholds]
        mean_t = float(np.mean(ts))
        conservative_t = float(max(ts))
        print(f"  Per-dataset Youden optima:")
        for name, t in thresholds:
            print(f"    {name:8s}: {t:+.4f}")
        print(f"  Mean of optima:         {mean_t:+.4f}")
        print(f"  Conservative (max):     {conservative_t:+.4f}")
        print(f"  Recommendation:         THRESHOLD_ANTI = {mean_t:+.4f}")
        print(f"  (Conservative would be {conservative_t:+.4f} — fewer false flags,"
              f" lower recall.)")
        results["recommendation"] = {
            "THRESHOLD_ANTI_mean": mean_t,
            "THRESHOLD_ANTI_conservative": conservative_t,
            "per_dataset": dict(thresholds),
        }
        results["note"] = ("THRESHOLD_ALIGNED is intentionally not set here. "
                           "TriviaQA and WikiBio do not contain dense aligned "
                           "signal by construction; the upper threshold should "
                           "be calibrated separately on a corpus where ALIGNED "
                           "regimes are actually expected.")
    else:
        print("  No usable Youden optima — run calibrate_*.py first.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote machine-readable summary: {out_path}")


if __name__ == "__main__":
    main()
