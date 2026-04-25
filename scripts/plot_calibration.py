#!/usr/bin/env python3
"""Plot calibration distributions and ROC curves.

Reads the calibrate_*.py CSVs and produces:
  - Histogram of A_cov for each (positive, negative) pair, with
    Youden's J threshold and the A_cov=0 line marked.
  - ROC curve with AUC.

Output: PNG files in data/ — one per dataset that has data.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def youden_threshold(scores_pos, scores_neg, n=400):
    all_s = np.concatenate([scores_pos, scores_neg])
    lo, hi = float(all_s.min()), float(all_s.max())
    pad = (hi - lo) * 0.02 if hi > lo else 0.01
    ts = np.linspace(lo - pad, hi + pad, n)
    best_t, best_j = None, -np.inf
    for t in ts:
        tpr = np.mean(scores_pos < t)
        fpr = np.mean(scores_neg < t)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = t
    return float(best_t), float(best_j)


def roc_curve(scores_pos, scores_neg, n=400):
    all_s = np.concatenate([scores_pos, scores_neg])
    lo, hi = float(all_s.min()), float(all_s.max())
    pad = (hi - lo) * 0.02 if hi > lo else 0.01
    ts = np.linspace(lo - pad, hi + pad, n)
    tprs = np.array([np.mean(scores_pos < t) for t in ts])
    fprs = np.array([np.mean(scores_neg < t) for t in ts])
    order = np.argsort(fprs)
    auc = float(np.trapz(tprs[order], fprs[order]))
    return fprs[order], tprs[order], auc


def plot_pair(name_pos, scores_pos, name_neg, scores_neg, title, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    all_s = np.concatenate([scores_pos, scores_neg])
    lo, hi = float(all_s.min()), float(all_s.max())
    pad = (hi - lo) * 0.05
    bins = np.linspace(lo - pad, hi + pad, 50)
    ax.hist(scores_neg, bins=bins, alpha=0.55, density=True,
            label=f"{name_neg} (n={len(scores_neg)})", color="#1f77b4")
    ax.hist(scores_pos, bins=bins, alpha=0.55, density=True,
            label=f"{name_pos} (n={len(scores_pos)})", color="#d62728")
    ax.axvline(0.0, color="k", linestyle=":", alpha=0.4, label="A_cov = 0")
    if len(scores_pos) >= 5 and len(scores_neg) >= 5:
        t, j = youden_threshold(scores_pos, scores_neg)
        ax.axvline(t, color="purple", linestyle="--", alpha=0.8,
                   label=f"Youden t = {t:+.4f} (J={j:.3f})")
    ax.set_xlabel("A_cov")
    ax.set_ylabel("density")
    ax.set_title(f"{title}: A_cov distribution")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.2)

    # ROC
    ax = axes[1]
    if len(scores_pos) >= 5 and len(scores_neg) >= 5:
        fpr, tpr, auc = roc_curve(scores_pos, scores_neg)
        ax.plot(fpr, tpr, color="purple", linewidth=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="random")
        ax.set_xlabel(f"FPR ({name_neg} flagged)")
        ax.set_ylabel(f"TPR ({name_pos} caught)")
        ax.set_title(f"{title}: ROC")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.2)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    else:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trivia", default="data/trivia_results.csv")
    ap.add_argument("--wikibio", default="data/wikibio_results.csv")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trivia_path = Path(args.trivia)
    if trivia_path.exists():
        rows = load_csv(trivia_path)
        valid = [r for r in rows if r.get("label") in
                 ("correct", "hallucination", "ambiguous")]
        correct = np.array([float(r["A_cov"]) for r in valid if r["label"] == "correct"])
        halluc = np.array([float(r["A_cov"]) for r in valid if r["label"] == "hallucination"])
        if len(correct) >= 5 and len(halluc) >= 5:
            plot_pair("hallucination", halluc, "correct", correct,
                      "TriviaQA", out_dir / "trivia_calibration.png")
        else:
            print(f"  TriviaQA: insufficient labeled data "
                  f"(correct={len(correct)}, hallucination={len(halluc)})")
    else:
        print(f"  TriviaQA CSV not found at {trivia_path}")

    wikibio_path = Path(args.wikibio)
    if wikibio_path.exists():
        rows = load_csv(wikibio_path)
        ref = np.array([float(r["A_cov"]) for r in rows if r.get("source") == "REF"])
        gen = np.array([float(r["A_cov"]) for r in rows if r.get("source") == "GEN"])
        if len(ref) >= 5 and len(gen) >= 5:
            plot_pair("GEN", gen, "REF", ref,
                      "WikiBio", out_dir / "wikibio_calibration.png")
        else:
            print(f"  WikiBio: insufficient data (REF={len(ref)}, GEN={len(gen)})")
    else:
        print(f"  WikiBio CSV not found at {wikibio_path}")


if __name__ == "__main__":
    main()
