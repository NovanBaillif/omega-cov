#!/usr/bin/env python3
"""Compute the optimal ANTI threshold from calibration CSVs.

Reads the outputs of calibrate_trivia.py and calibrate_wikibio.py,
computes per-dataset descriptive stats, AUC, Youden's J with bootstrap
CI, and distribution overlap. Prints the recommended threshold and
saves a machine-readable JSON.

Convention:
  - "positive" class = should be flagged ANTI
    (hallucination on TriviaQA, GEN on WikiBio)
  - "negative" class = should NOT be flagged
    (correct on TriviaQA, REF on WikiBio)
  - decision rule: flag as ANTI iff A_cov < threshold
  - TPR = P(A_cov < t | positive)
  - FPR = P(A_cov < t | negative)
  - Youden's J = TPR - FPR, maximized over t.

Outputs:
  - stdout summary
  - data/calibration_results.json — full numeric report
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PERCENTILES = [5, 10, 25, 50, 75, 90, 95]


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def percentile_summary(arr: np.ndarray) -> dict:
    out = {f"p{p}": float(np.percentile(arr, p)) for p in PERCENTILES}
    out["mean"] = float(arr.mean())
    out["std"] = float(arr.std())
    out["n"] = int(len(arr))
    return out


def youden_j(scores_pos: np.ndarray, scores_neg: np.ndarray,
             n_thresholds: int = 400) -> dict:
    """Threshold t maximizing TPR - FPR for the rule (score < t -> positive)."""
    if len(scores_pos) < 5 or len(scores_neg) < 5:
        return {"error": "insufficient samples",
                "n_pos": int(len(scores_pos)),
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


def auc_low_positive(scores_pos: np.ndarray, scores_neg: np.ndarray,
                     n_thresholds: int = 400) -> float:
    """AUC for the rule 'low score = positive'.

    AUC = 1.0 means perfect separation (all pos < all neg).
    AUC = 0.5 means no signal.
    AUC < 0.5 means signal is inverted (pos > neg on average).
    """
    if len(scores_pos) < 5 or len(scores_neg) < 5:
        return float("nan")
    all_s = np.concatenate([scores_pos, scores_neg])
    lo, hi = float(all_s.min()), float(all_s.max())
    pad = (hi - lo) * 0.02 if hi > lo else 0.01
    ts = np.linspace(lo - pad, hi + pad, n_thresholds)
    tprs = np.array([np.mean(scores_pos < t) for t in ts])
    fprs = np.array([np.mean(scores_neg < t) for t in ts])
    order = np.argsort(fprs)
    return float(np.trapz(tprs[order], fprs[order]))


def histogram_overlap(scores_pos: np.ndarray, scores_neg: np.ndarray,
                      bins: int = 50) -> float:
    """Overlap coefficient of two distributions via histograms.

    0.0 = perfectly separated, 1.0 = identical distributions.
    """
    if len(scores_pos) < 5 or len(scores_neg) < 5:
        return float("nan")
    all_s = np.concatenate([scores_pos, scores_neg])
    edges = np.linspace(all_s.min(), all_s.max(), bins + 1)
    h_pos, _ = np.histogram(scores_pos, bins=edges, density=True)
    h_neg, _ = np.histogram(scores_neg, bins=edges, density=True)
    bin_w = edges[1] - edges[0]
    return float(np.minimum(h_pos, h_neg).sum() * bin_w)


def bootstrap_youden(scores_pos: np.ndarray, scores_neg: np.ndarray,
                     n_boot: int = 1000, seed: int = 42) -> dict:
    """95% CI on the Youden-J-optimal threshold via bootstrap resampling."""
    if len(scores_pos) < 5 or len(scores_neg) < 5:
        return {"error": "insufficient samples"}
    rng = np.random.default_rng(seed)
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    ts, js = [], []
    for _ in range(n_boot):
        bp = rng.choice(scores_pos, size=n_pos, replace=True)
        bn = rng.choice(scores_neg, size=n_neg, replace=True)
        r = youden_j(bp, bn, n_thresholds=200)
        if "threshold" in r and r["threshold"] is not None:
            ts.append(r["threshold"])
            js.append(r["j"])
    if not ts:
        return {"error": "bootstrap produced no samples"}
    arr_t = np.array(ts)
    arr_j = np.array(js)
    return {
        "n_boot": int(len(ts)),
        "threshold_mean": float(arr_t.mean()),
        "threshold_std": float(arr_t.std()),
        "threshold_ci95_low": float(np.quantile(arr_t, 0.025)),
        "threshold_ci95_high": float(np.quantile(arr_t, 0.975)),
        "j_mean": float(arr_j.mean()),
        "j_ci95_low": float(np.quantile(arr_j, 0.025)),
        "j_ci95_high": float(np.quantile(arr_j, 0.975)),
    }


def analyze_pair(name_pos: str, scores_pos: np.ndarray,
                 name_neg: str, scores_neg: np.ndarray) -> dict:
    out = {
        f"{name_pos}_acov": percentile_summary(scores_pos) if len(scores_pos) >= 5 else {"n": int(len(scores_pos))},
        f"{name_neg}_acov": percentile_summary(scores_neg) if len(scores_neg) >= 5 else {"n": int(len(scores_neg))},
    }
    if len(scores_pos) >= 5 and len(scores_neg) >= 5:
        out["youden"] = youden_j(scores_pos, scores_neg)
        out["auc"] = auc_low_positive(scores_pos, scores_neg)
        out["overlap"] = histogram_overlap(scores_pos, scores_neg)
        out["bootstrap"] = bootstrap_youden(scores_pos, scores_neg)
    return out


def analyze_trivia(rows: list[dict]) -> dict:
    valid = [r for r in rows if r.get("label") in
             ("correct", "hallucination", "ambiguous")]
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
    out.update(analyze_pair("hallucination", halluc, "correct", correct))
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
    out.update(analyze_pair("gen", gen, "ref", ref))
    return out


def fmt_summary(s: dict) -> str:
    if "p50" not in s:
        return f"n={s.get('n', 0)} (insufficient)"
    return (f"n={s['n']:4d}  median={s['p50']:+.4f}  "
            f"IQR=[{s['p25']:+.4f}, {s['p75']:+.4f}]  "
            f"P5/P95=[{s['p5']:+.4f}, {s['p95']:+.4f}]  "
            f"mean={s['mean']:+.4f} sd={s['std']:.4f}")


def fmt_youden(y: dict) -> str:
    if "error" in y:
        return f"  Youden J: skipped ({y['error']})"
    return (f"  threshold = {y['threshold']:+.4f}  "
            f"TPR={y['tpr']:.3f}  FPR={y['fpr']:.3f}  J={y['j']:.3f}")


def fmt_bootstrap(b: dict) -> str:
    if "error" in b:
        return f"  bootstrap: {b['error']}"
    return (f"  threshold 95% CI = [{b['threshold_ci95_low']:+.4f}, "
            f"{b['threshold_ci95_high']:+.4f}]  "
            f"(mean={b['threshold_mean']:+.4f} sd={b['threshold_std']:.4f})  "
            f"  J 95% CI = [{b['j_ci95_low']:.3f}, {b['j_ci95_high']:.3f}]")


def print_analysis(a: dict):
    print(f"\n=== {a['dataset']} ===")
    if a["dataset"] == "TriviaQA":
        print(f"  rows={a['n_rows']}  correct={a['n_correct']}  "
              f"hallucination={a['n_hallucination']}  "
              f"ambiguous={a['n_ambiguous']}  errors={a['n_errors']}")
        print(f"  correct A_cov:        {fmt_summary(a.get('correct_acov', {}))}")
        print(f"  hallucination A_cov:  {fmt_summary(a.get('hallucination_acov', {}))}")
    else:
        print(f"  rows={a['n_rows']}  REF={a['n_ref']}  GEN={a['n_gen']}")
        print(f"  REF A_cov:  {fmt_summary(a.get('ref_acov', {}))}")
        print(f"  GEN A_cov:  {fmt_summary(a.get('gen_acov', {}))}")

    if "auc" in a:
        auc = a["auc"]
        print(f"  AUC (low = positive) = {auc:.3f}  "
              f"({'good separation' if auc > 0.7 else 'weak separation' if auc > 0.6 else 'poor separation'})")
    if "overlap" in a:
        print(f"  distribution overlap = {a['overlap']:.3f}  "
              f"(0 = separated, 1 = identical)")
    if "youden" in a:
        print(fmt_youden(a["youden"]))
    if "bootstrap" in a:
        print(fmt_bootstrap(a["bootstrap"]))


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
        if "youden" in a and a["youden"].get("threshold") is not None:
            thresholds.append(("trivia", a["youden"]["threshold"]))
    else:
        print(f"NOTE: {trivia_path} not found, skipping TriviaQA.")

    wikibio_path = Path(args.wikibio)
    if wikibio_path.exists():
        rows = load_csv(wikibio_path)
        a = analyze_wikibio(rows)
        print_analysis(a)
        results["wikibio"] = a
        if "youden" in a and a["youden"].get("threshold") is not None:
            thresholds.append(("wikibio", a["youden"]["threshold"]))
    else:
        print(f"NOTE: {wikibio_path} not found, skipping WikiBio.")

    print("\n=== Threshold candidates ===")
    if thresholds:
        ts = [t for _, t in thresholds]
        mean_t = float(np.mean(ts))
        conservative_t = float(max(ts))
        print(f"  Per-dataset Youden optima:")
        for name, t in thresholds:
            print(f"    {name:8s}: {t:+.4f}")
        if len(thresholds) > 1:
            print(f"  Mean of optima:     {mean_t:+.4f}")
            print(f"  Max (conservative): {conservative_t:+.4f}")
        results["candidates"] = {
            "per_dataset": dict(thresholds),
            "mean": mean_t,
            "conservative_max": conservative_t,
        }
        print("\n  No automatic recommendation written. Inspect distributions,")
        print("  AUC, overlap, and CI before updating thresholds.py.")
        print("  See plot_calibration.py for visual inspection.")
    else:
        print("  No usable Youden optima — run calibrate_*.py first.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote machine-readable summary: {out_path}")


if __name__ == "__main__":
    main()
