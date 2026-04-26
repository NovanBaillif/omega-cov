# Calibration scripts

Calibrate the `THRESHOLD_DENSE` constant in `omega_cov/thresholds.py`
against WikiBio (and, in v0.2, additional datasets).

`THRESHOLD_ANTI` is mathematical (`A_cov < 0`) and is **not** calibrated.

For the rationale, the methodology, and the v0.1 results, see
[../docs/calibration.md](../docs/calibration.md).

## What each script does

| Script | Inputs | Output |
| --- | --- | --- |
| `calibrate_trivia.py` | TriviaQA `rc.nocontext` validation split | `data/trivia_results.csv` |
| `calibrate_wikibio.py` | WikiBio test split | `data/wikibio_results.csv` |
| `analyze_thresholds.py` | The CSVs above | stdout summary + `data/calibration_results.json` |
| `plot_calibration.py` | The CSVs above | `data/*.png` (histogram + ROC) |

`calibrate_trivia.py` is included for completeness and v0.2 work, but
its v0.1 run was deferred — see `docs/calibration.md` for why.

## Setup

You need a GPU (Colab T4 or better). On Colab:

```bash
git clone https://github.com/NovanBaillif/omega-cov.git
cd omega-cov
pip install -e ".[calibrate]"
export ANTHROPIC_API_KEY=sk-ant-...   # only needed for calibrate_trivia.py
```

`bitsandbytes` requires CUDA. On CPU / MPS the scripts still run but
are too slow for the target sample sizes.

`datasets` is pinned to `<4.0` because the WikiBio HF dataset still
relies on a loading script that newer `datasets` versions refuse to run.
TriviaQA was migrated to Parquet upstream and works either way.

## Running

```bash
# WikiBio: ~30 min on a Colab T4 (n=500 → 491 paired REF + GEN)
python scripts/calibrate_wikibio.py --n 500 --seed 42

# Analysis (instant once CSVs are present):
python scripts/analyze_thresholds.py
python scripts/plot_calibration.py

# TriviaQA (deferred to v0.2 — included for reference):
python scripts/calibrate_trivia.py --n 1000 --seed 42
```

The CSVs are written incrementally (one row per `flush()`), so a
disconnected Colab runtime does not lose committed rows. To survive
runtime resets entirely, mount Google Drive and pass
`--out /content/drive/MyDrive/<path>.csv`.

## Methodology

### WikiBio: REF vs GEN

The label is the **source**, no external annotation needed:

- `REF` — the human-written biography text from WikiBio.
- `GEN` — Mistral-7B continuation of
  `"Write a 150-word biography of <name>"`.

The threshold separating them empirically is what we call
`THRESHOLD_DENSE`.

### TriviaQA labeling

For TriviaQA, each generated answer is labeled
`correct` / `hallucination` / `ambiguous`:

1. **Alias match (fast path).** Exact substring match against the
   canonical answer plus aliases plus normalized aliases provided by
   TriviaQA. Hits are labeled `correct` without calling the judge.
2. **LLM judge.** Misses go to Claude Haiku 4.5 at temperature 0,
   which returns a JSON verdict. Paraphrases, abbreviations, and
   alternate spellings should be caught at this stage.

The label source is recorded in the `label_source` column so you can
audit the split.

### Youden's J

For each dataset, `analyze_thresholds.py` finds the threshold `t` that
maximizes `TPR − FPR` for the binary task "is this likely a
confabulation". On WikiBio, "positive" = GEN, "negative" = REF; the
sign convention follows the rule "flag as below-DENSE iff
`A_cov < t`".

`analyze_thresholds.py` reports the per-dataset Youden optimum, AUC,
distribution overlap, and a 95% bootstrap CI. **No automatic
recommendation is written to `thresholds.py`.** Inspect the
distributions and the CI before deciding.

## Updating `omega_cov/thresholds.py`

After running the analysis, edit the constants at the top of
`omega_cov/thresholds.py` and append a row to the run-log table in
`docs/calibration.md` with the date, sample sizes, seed, and the new
commit hash.

```python
THRESHOLD_DENSE = 0.069   # calibrated on WikiBio (Mistral-7B-v0.1)
THRESHOLD_ANTI  = 0.0     # mathematical: A_cov < 0
```
