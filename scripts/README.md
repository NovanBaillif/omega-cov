# Calibration scripts

Calibrate the `THRESHOLD_ANTI` constant in `omega_cov/thresholds.py`
against TriviaQA and WikiBio.

`THRESHOLD_ALIGNED` is **not** calibrated here — see notes below.

## What each script does

| Script | Inputs | Output |
| --- | --- | --- |
| `calibrate_trivia.py` | TriviaQA `rc.nocontext` validation split | `data/trivia_results.csv` |
| `calibrate_wikibio.py` | WikiBio test split | `data/wikibio_results.csv` |
| `analyze_thresholds.py` | The two CSVs above | `data/calibration_results.json` + recommendation |

## Setup

You need a GPU (Colab T4 or better). On Colab:

```bash
git clone https://github.com/NovanBaillif/omega-cov.git
cd omega-cov
pip install -e ".[calibrate]"
export ANTHROPIC_API_KEY=sk-ant-...
```

`bitsandbytes` requires CUDA. On CPU/MPS the scripts still run but
are too slow for n=1000.

`datasets` is pinned to `<4.0` because the WikiBio HF dataset still
relies on a loading script that newer datasets versions refuse to
run. TriviaQA was migrated to Parquet upstream and works either way.

## Running

```bash
# ~45 min on Colab T4 with n=1000
python scripts/calibrate_trivia.py --n 1000 --seed 42

# ~30 min on Colab T4 with n=500 (REF + GEN = 1000 rows)
python scripts/calibrate_wikibio.py --n 500 --seed 42

# Instant
python scripts/analyze_thresholds.py
```

The CSVs are written incrementally (one row per `flush()`), so a
disconnected Colab runtime does not lose work — restart the script
to continue from scratch, or post-process the partial CSV.

## Methodology

### TriviaQA labeling

Each generated answer is labeled `correct` / `hallucination` / `ambiguous`:

1. **Alias match (fast path).** Exact substring match against the canonical
   answer plus aliases plus normalized aliases provided by TriviaQA. Hits are
   labeled `correct` without calling the judge.
2. **LLM judge.** Misses go to Claude Haiku 4.5 at temperature 0, which
   returns a JSON verdict. Paraphrases, abbreviations, and alternate
   spellings should be caught at this stage.

The label source is recorded in the `label_source` column so you can
audit the split.

### WikiBio labeling

No external labeling is needed. The label is the **source**:

- `REF`: the reference biography text from WikiBio (factual by construction).
- `GEN`: a Mistral-7B continuation of the prompt
  `"Write a 150-word biography of <name>"` (potential confabulation).

### Youden's J

For each dataset, `analyze_thresholds.py` finds the threshold `t` that
maximizes `TPR - FPR` for the rule "flag as ANTI iff `A_cov < t`":

- TPR = fraction of `hallucination` (TriviaQA) or `GEN` (WikiBio) caught
- FPR = fraction of `correct` (TriviaQA) or `REF` (WikiBio) falsely flagged

Ambiguous TriviaQA samples are excluded from the optimization.

The recommendation is the **mean of the two per-dataset optima**.
The conservative bound (`max`) is also reported; pick that if you
prefer fewer false flags at the cost of recall.

## Why ALIGNED is not calibrated here

`THRESHOLD_ALIGNED` separates regions where surprise covaries with
genuine semantic movement (information-dense, novel content) from
the MIXED middle. Neither TriviaQA short answers nor WikiBio
biographical prose are reliable sources of dense aligned signal —
calibrating ALIGNED on these would systematically underestimate it.

Calibrate `THRESHOLD_ALIGNED` separately on a corpus where the
ALIGNED regime is *expected*: scientific papers, technical
documentation, dense expository writing.

## Updating `omega_cov/thresholds.py`

After running `analyze_thresholds.py`, edit the constants at the top
of `omega_cov/thresholds.py`:

```python
THRESHOLD_ANTI = -0.027    # from calibration on TriviaQA + WikiBio
THRESHOLD_ALIGNED = 0.1    # still placeholder, calibrate separately
```

Commit with a reference to the calibration run (sample sizes, seed,
date) so the choice is reproducible.
