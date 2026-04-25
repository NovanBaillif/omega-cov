# Calibration of `omega-cov` thresholds

This document records how the constants in `omega_cov/thresholds.py` are
derived. It is meant to be auditable: the *what*, the *why*, and *how to
reproduce*.

## Constants

`omega_cov/thresholds.py` defines two thresholds that partition `A_cov`
into three signatures:

| Constant | v0.1 value | Origin | Calibrated on |
| --- | --- | --- | --- |
| `THRESHOLD_DENSE` | `+0.069` | Empirical (Youden's J optimum) | WikiBio, n = 491 paired REF / GEN |
| `THRESHOLD_ANTI`  | `0.0`    | Mathematical (true anti-correlation) | n/a — by definition |

| Signature | Condition | Meaning |
| --- | --- | --- |
| `DENSE` | `A_cov >= THRESHOLD_DENSE` | Surprise covaries tightly with displacement. Factual, information-rich text. |
| `WEAK`  | `THRESHOLD_ANTI <= A_cov < THRESHOLD_DENSE` | Positive but low covariance. Surprise without commensurate displacement — candidate confabulation. |
| `ANTI`  | `A_cov < THRESHOLD_ANTI` | Surprise and displacement actively oppose each other. Pathological by definition; rare on standard generative text. |

The two boundaries are decoupled by construction:

- `THRESHOLD_DENSE` is **empirical and per-model**. It is the value of
  `A_cov` that maximizes Youden's J (TPR − FPR) for the binary task
  "is this a factual reference vs a freely-generated continuation?"
  on WikiBio. It depends on the model under measurement.
- `THRESHOLD_ANTI` is **mathematical and universal**. `A_cov < 0` means
  cosine displacement and Shannon surprise are anti-correlated. Whether
  this regime is reached frequently in practice depends on the corpus,
  but the boundary itself is model-independent.

## Why this naming

The original v0 placeholders (`ALIGNED` / `MIXED` / `ANTI`) implied a
symmetric structure around `A_cov = 0`. WikiBio data showed this is not
empirically what we observe:

- 99.4 % of REF samples have `A_cov > 0`.
- 84.7 % of GEN samples have `A_cov > 0`.
- The Youden-optimal threshold for separating them is **+0.069**, not 0.

In other words, even GEN samples — text generated freely by an LLM, where
confabulation is plausible but not demonstrated by this experiment — usually
have *positive* covariance between δ and σ. They are *weakly* covarying
compared with REF, not anti-correlated. The renaming reflects what the data
actually shows:

- `DENSE` — high covariance regime (formerly `ALIGNED`)
- `WEAK`  — low-but-positive covariance regime (formerly `MIXED`); empirically
  this is where GEN sits, but the regime is defined by its A_cov range, not
  by a confabulation claim
- `ANTI`  — true anti-correlation, kept under its mathematical definition

## Methodology

### WikiBio: REF vs GEN

The dataset is `wiki_bio` (HuggingFace), test split.

For each sampled biography:

1. **REF** — `A_cov` measured on the reference biography text directly.
   Factual by construction.
2. **GEN** — Mistral-7B-v0.1 prompted with
   `"Write a 150-word biography of <name>"` (name extracted from the
   WikiBio infobox). `A_cov` measured on the model's continuation,
   evaluated in context of the prompt.

No external labelling is needed: the label is the source. We are not
claiming every GEN is wrong. We are claiming that, *on average*, GEN
sits lower in `A_cov` than REF, and the threshold falls between the two
distributions.

### TriviaQA — tested and deferred

We initially planned to also calibrate against TriviaQA, where each
question has a Mistral-7B-generated answer judged by Claude Haiku 4.5.
After 100 samples, the skip rate (samples discarded because too few
tokens survived the `sigma_min` filter) was **60 %**.

This is a structural property of TriviaQA: short factual answers are
dominated by low-entropy tokens (entities, function words), and the
default `sigma_min = 1.0` filter removes most of them. Lowering
`sigma_min` to make the dataset work is **not** an option — `sigma_min`
is part of the metric definition, and changing it during calibration
would mean the calibrated threshold no longer applies to the deployed
metric.

TriviaQA calibration is therefore deferred to v0.2 with adapted
parameters (different prompt to elicit longer answers, or a different
metric variant that does not depend on `sigma_min`).

The partial run (57 labelled samples on n=100) is archived for
provenance but is not used to derive any threshold.

### Sample selection

For both datasets, samples are drawn with `np.random.default_rng(42)`
from the dataset's full split, without replacement. The seed is fixed
so the sample is reproducible.

### Decision rule and metrics

Convention: **predict REF (positive) iff `A_cov >= t`**.

- TPR (sensitivity, recall) = `P(A_cov >= t | REF)`
- FPR = `P(A_cov >= t | GEN)`
- TNR (specificity) = `1 − FPR`
- Youden's J = TPR − FPR, maximized over `t` to give the optimal
  threshold under the equal-cost assumption.

The 95 % bootstrap CI on the threshold and on Youden's J is computed
from 2000 paired bootstrap resamples (REF and GEN drawn independently,
with replacement, each at their original size).

## Results

### Sample sizes

| Dataset | Requested | Skipped | Used | Pairing |
| --- | --- | --- | --- | --- |
| WikiBio | 500 | 9 | 491 REF + 491 GEN | paired |
| TriviaQA | 100 (interrupted) | 60 | 40 labelled (deferred) | unpaired |

### Distribution of A_cov on WikiBio

| | n | P5 | P25 | P50 | P75 | P95 | mean | sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| REF | 491 | +0.052 | +0.090 | **+0.131** | +0.179 | +0.233 | +0.135 | 0.058 |
| GEN | 491 | −0.016 | +0.013 | **+0.036** | +0.062 | +0.108 | +0.040 | 0.040 |

Fraction below zero:

- REF: 0.61 % (3 / 491)
- GEN: 15.27 % (75 / 491)

### Separability

| Metric | Point estimate | 95 % CI (bootstrap, n=2000) |
| --- | --- | --- |
| AUC ROC | **0.918** | [0.900, 0.935] |
| Youden's J | 0.682 | [0.644, 0.731] |
| Threshold (Youden-optimal) | +0.0698 | [+0.063, +0.094] |
| Cohen's d (REF vs GEN) | 1.91 | — |
| Mann-Whitney U (one-sided, REF > GEN) | p = 2.77 × 10⁻¹¹⁴ | — |

(Two-sided Mann-Whitney would give p = 5.55 × 10⁻¹¹⁴, exactly twice the
one-sided value as expected when the alternative is in the correct
direction. We report the one-sided test because the directional claim
"REF stochastically dominates GEN in A_cov" is the substantive one.)

The threshold CI is asymmetric (longer right tail). The point estimate
+0.0698 is rounded to **`THRESHOLD_DENSE = 0.069`** in
`omega_cov/thresholds.py`.

### Performance at `THRESHOLD_DENSE = +0.069`

| | True REF | True GEN |
| --- | --- | --- |
| Predicted REF | 438 (TP) | 102 (FP) |
| Predicted GEN | 53 (FN) | 389 (TN) |

- Sensitivity (recall) = 0.892
- Specificity = 0.792
- Precision = 0.811

## Reproducing

```bash
git clone https://github.com/NovanBaillif/omega-cov.git
cd omega-cov
pip install -e ".[calibrate]"

# WikiBio (~30 min on a Colab T4):
python scripts/calibrate_wikibio.py --n 500 --seed 42

# Analysis:
python scripts/analyze_thresholds.py
python scripts/plot_calibration.py
```

The CSVs from the v0.1 run are committed under `calibration/` for
auditability.

### Run log

| Date | Model | Seed | n_ref | n_gen | THRESHOLD_DENSE (Youden) | AUC | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-25 | mistralai/Mistral-7B-v0.1 (4-bit) | 42 | 491 | 491 | +0.0698 | 0.918 | b583f1e |

## Limitations of the v0.1 calibration

- **Single model.** `THRESHOLD_DENSE` was derived on Mistral-7B-v0.1
  only. The relationship between hidden-state geometry and entropy is
  architecture-specific; re-calibration is required for other model
  families before the same numerical threshold can be used.
- **Single dataset.** WikiBio biographies are one specific domain
  (factual, chronological, biographical). Generalising the empirical
  threshold to news, scientific abstracts, or technical documentation
  has not been validated.
- **WEAK / ANTI boundary not empirically validated.** The boundary at
  `A_cov = 0` is mathematically motivated (true anti-correlation) but
  has not been validated against an adversarial dataset where ANTI
  cases are expected. On WikiBio, only 0.61 % of REF and 15.27 % of
  GEN reach `A_cov < 0`, so the dataset cannot tell us whether the
  ANTI regime corresponds to a meaningfully distinct failure mode or
  to noise.
- **Gating effect of `sigma_min`.** The `sigma_min = 1.0` filter
  shapes which tokens contribute to `A_cov`. The TriviaQA failure
  shows that this filter is not always benign. Future work should
  characterise the filter's effect on different text length regimes.

## v0.2 plans

- Cross-model recalibration of `THRESHOLD_DENSE` on Llama, Qwen, Gemma.
- TriviaQA calibration with adapted parameters (longer-form prompts,
  or a metric variant that lifts the `sigma_min` constraint).
- Empirical validation of the WEAK / ANTI boundary on adversarial
  text (deliberately broken or contradictory generations).
- Per-domain thresholds (news, scientific writing) if cross-domain
  drift on `THRESHOLD_DENSE` exceeds the bootstrap CI.
