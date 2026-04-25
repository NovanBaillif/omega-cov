# Calibration data

Raw outputs of the v0.1 threshold calibration. See
[../docs/calibration.md](../docs/calibration.md) for full methodology.

## Files

| File | Description |
| --- | --- |
| `wikibio_results.csv` | Raw `calibrate_wikibio.py` output. One row per (sample, source). 982 rows = 491 REF + 491 GEN. |
| `wikibio_paired.csv` | Same data pivoted to one row per sample (sample_id, A_cov_ref, A_cov_gen, signature_ref, signature_gen). 491 rows. |
| `wikibio_histograms.png` | A_cov distribution plot, REF vs GEN, 30 bins. |
| `trivia_partial_archived.csv` | TriviaQA partial run, 57 labelled samples on n=100. Archived for provenance only — **not used for calibration** (skip rate too high, see docs/calibration.md). |

## Provenance

- Run date: 2026-04-25
- Model: `mistralai/Mistral-7B-v0.1` loaded in 4-bit (bitsandbytes nf4)
- Hardware: Colab T4 GPU
- Seed: 42 (`np.random.default_rng(42)` for sample selection;
  `torch.manual_seed(42)` added in commit b0bc8c9 — re-runs since
  that commit are bit-reproducible)
- Sample selection: WikiBio test split, 500 indices drawn without replacement

## Reproducing

```bash
pip install -e ".[calibrate]"
python scripts/calibrate_wikibio.py --n 500 --seed 42
python scripts/analyze_thresholds.py
python scripts/plot_calibration.py
```

The output CSVs land in `data/` (gitignored). Compare row counts and
signature distributions against the files in this directory.
