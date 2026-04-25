#!/usr/bin/env python3
"""Analyze TriviaQA v0.2 calibration — intra-LLM separation.

Reads scripts/calibrate_trivia_v2.py output and computes:
  - Per-class A_cov distributions (P5/P25/P50/P75/P95).
  - Mann-Whitney U test, one-sided in both directions; reports the
    significant one.
  - AUC ROC + bootstrap 95% CI (2000 resamples), with the positive
    class chosen post-hoc as the one with the higher mean A_cov.
  - Youden's J optimum (and its bootstrap CI).
  - Post-hoc bias check: cor(n_valid, label==correct). Target is
    |cor| < 0.1, which would confirm that v0.1's alias_match-driven
    label/length confound is gone.

Output: stdout summary + data/trivia_v2_analysis.json.

Interpretation rule (per spec):
  AUC > 0.7 in direction "correct > hallucination" → A_cov detects
    intra-LLM confabulation.
  0.55 < AUC < 0.7 → partial signal, to investigate.
  AUC < 0.55 (or direction inverted) → A_cov does not detect intra-LLM
    confabulation; v0.1's WikiBio result remains a human-vs-LLM
    signature, not a confabulation detector.
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


def percentile_summary(arr: np.ndarray) -> dict:
    return {
        "n": int(len(arr)),
        "p5":  float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def youden(scores_high: np.ndarray, scores_low: np.ndarray,
           n_thresh: int = 400) -> tuple[float, float]:
    """Predict 'high class' iff score >= t. Returns (t_opt, J_opt)."""
    all_s = np.concatenate([scores_high, scores_low])
    ts = np.linspace(all_s.min() - 0.01, all_s.max() + 0.01, n_thresh)
    j = np.array([(scores_high >= t).mean() - (scores_low >= t).mean() for t in ts])
    i = int(np.argmax(j))
    return float(ts[i]), float(j[i])


def auc_high_positive(scores_high: np.ndarray, scores_low: np.ndarray) -> float:
    """AUC for the rule 'high score = positive class' (Mann-Whitney form)."""
    if len(scores_high) == 0 or len(scores_low) == 0:
        return float("nan")
    comp = scores_high[:, None] - scores_low[None, :]
    return float(((comp > 0).sum() + 0.5 * (comp == 0).sum())
                 / (len(scores_high) * len(scores_low)))


def mann_whitney(a: np.ndarray, b: np.ndarray, alternative: str = "greater") -> tuple[float, float]:
    from scipy import stats as scstats
    u, p = scstats.mannwhitneyu(a, b, alternative=alternative)
    return float(u), float(p)


def bootstrap_separation(scores_high: np.ndarray, scores_low: np.ndarray,
                         n_boot: int = 2000, seed: int = 43) -> dict:
    rng = np.random.default_rng(seed)
    aucs, ts, js = [], [], []
    for _ in range(n_boot):
        bh = rng.choice(scores_high, size=len(scores_high), replace=True)
        bl = rng.choice(scores_low, size=len(scores_low), replace=True)
        aucs.append(auc_high_positive(bh, bl))
        t, j = youden(bh, bl, n_thresh=200)
        ts.append(t)
        js.append(j)
    aucs = np.array(aucs)
    ts = np.array(ts)
    js = np.array(js)
    return {
        "n_boot": n_boot,
        "auc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "threshold_ci95": [float(np.percentile(ts, 2.5)), float(np.percentile(ts, 97.5))],
        "j_ci95": [float(np.percentile(js, 2.5)), float(np.percentile(js, 97.5))],
        "auc_median": float(np.median(aucs)),
        "threshold_median": float(np.median(ts)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/trivia_v2_results.csv")
    ap.add_argument("--out", default="data/trivia_v2_analysis.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run calibrate_trivia_v2.py first.")
        return

    rows = load_csv(csv_path)
    n_total = len(rows)
    print(f"Loaded {n_total} rows from {csv_path}")

    correct = np.array([float(r["A_cov"]) for r in rows if r["verdict"] == "correct"])
    halluc  = np.array([float(r["A_cov"]) for r in rows if r["verdict"] == "hallucination"])
    ambig   = np.array([float(r["A_cov"]) for r in rows if r["verdict"] == "ambiguous"])
    err     = np.array([float(r["A_cov"]) for r in rows if r["verdict"] == "error"])
    print(f"  correct={len(correct)}  hallucination={len(halluc)}  "
          f"ambiguous={len(ambig)}  errors={len(err)}")

    if len(correct) < 5 or len(halluc) < 5:
        print("\nInsufficient data for analysis (need >= 5 per class).")
        return

    cor_stats = percentile_summary(correct)
    hal_stats = percentile_summary(halluc)
    print()
    print(f"correct A_cov:        n={cor_stats['n']:3d}  "
          f"P5={cor_stats['p5']:+.4f}  P25={cor_stats['p25']:+.4f}  "
          f"P50={cor_stats['p50']:+.4f}  P75={cor_stats['p75']:+.4f}  "
          f"P95={cor_stats['p95']:+.4f}  mean={cor_stats['mean']:+.4f}")
    print(f"hallucination A_cov:  n={hal_stats['n']:3d}  "
          f"P5={hal_stats['p5']:+.4f}  P25={hal_stats['p25']:+.4f}  "
          f"P50={hal_stats['p50']:+.4f}  P75={hal_stats['p75']:+.4f}  "
          f"P95={hal_stats['p95']:+.4f}  mean={hal_stats['mean']:+.4f}")

    u_cgth, p_cgth = mann_whitney(correct, halluc, "greater")
    u_hgtc, p_hgtc = mann_whitney(halluc, correct, "greater")
    print()
    print("Mann-Whitney U (one-sided):")
    print(f"  correct > hallucination:  U={u_cgth:.0f}  p={p_cgth:.3e}")
    print(f"  hallucination > correct:  U={u_hgtc:.0f}  p={p_hgtc:.3e}")
    direction = ("correct > hallucination" if p_cgth < p_hgtc
                 else "hallucination > correct")
    print(f"  -> significant direction: {direction}")

    if cor_stats["mean"] >= hal_stats["mean"]:
        positive_label = "correct"
        scores_high, scores_low = correct, halluc
    else:
        positive_label = "hallucination"
        scores_high, scores_low = halluc, correct

    auc = auc_high_positive(scores_high, scores_low)
    t_opt, j_opt = youden(scores_high, scores_low)
    print()
    print(f"AUC (positive = {positive_label}, high A_cov): {auc:.4f}")
    print(f"Youden J optimum: threshold = {t_opt:+.4f}, J = {j_opt:.4f}")

    boot = bootstrap_separation(scores_high, scores_low, n_boot=args.n_boot)
    print()
    print(f"Bootstrap (n={args.n_boot}):")
    print(f"  AUC 95% CI       = [{boot['auc_ci95'][0]:.4f}, {boot['auc_ci95'][1]:.4f}]")
    print(f"  threshold 95% CI = [{boot['threshold_ci95'][0]:+.4f}, "
          f"{boot['threshold_ci95'][1]:+.4f}]")
    print(f"  Youden J 95% CI  = [{boot['j_ci95'][0]:.4f}, {boot['j_ci95'][1]:.4f}]")

    valid_rows = [r for r in rows if r["verdict"] in ("correct", "hallucination")]
    n_valids = np.array([int(r["n_valid"]) for r in valid_rows])
    labels = np.array([1 if r["verdict"] == "correct" else 0 for r in valid_rows])
    cor_nv_label = None
    if len(n_valids) > 5 and labels.std() > 0:
        cor_nv_label = float(np.corrcoef(n_valids, labels)[0, 1])
        ok = abs(cor_nv_label) < 0.1
        print()
        print(f"Post-hoc bias check: cor(n_valid, label==correct) = {cor_nv_label:+.4f}")
        print(f"  -> {'OK' if ok else 'BIAS DETECTED'} (target |cor| < 0.1)")

    print()
    print("=== Interpretation ===")
    if direction == "correct > hallucination" and auc > 0.7:
        verdict_text = (f"AUC = {auc:.3f} in direction (correct > hallucination): "
                        "A_cov detects intra-LLM confabulation.")
    elif direction == "correct > hallucination" and auc > 0.55:
        verdict_text = (f"AUC = {auc:.3f} in direction (correct > hallucination): "
                        "partial signal, worth investigating confounds.")
    elif direction == "correct > hallucination":
        verdict_text = (f"AUC = {auc:.3f} in direction (correct > hallucination): "
                        "weak signal, not a reliable detector.")
    else:
        verdict_text = (f"AUC = {auc:.3f} in direction ({direction}): "
                        "A_cov is HIGHER on hallucinations than on correct "
                        "answers — no confabulation detection signal.")
    print(f"  {verdict_text}")

    out = {
        "n_rows": n_total,
        "n_correct": int(len(correct)),
        "n_hallucination": int(len(halluc)),
        "n_ambiguous": int(len(ambig)),
        "n_errors": int(len(err)),
        "correct_acov": cor_stats,
        "hallucination_acov": hal_stats,
        "mw_correct_greater_p": p_cgth,
        "mw_hallucination_greater_p": p_hgtc,
        "significant_direction": direction,
        "auc": auc,
        "auc_positive_label": positive_label,
        "youden_threshold": t_opt,
        "youden_j": j_opt,
        "bootstrap": boot,
        "post_hoc_cor_nvalid_label": cor_nv_label,
        "interpretation": verdict_text,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
