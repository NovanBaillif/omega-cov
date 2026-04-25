# Calibration of `omega-cov` thresholds

This document records how the constants in `omega_cov/thresholds.py`
are derived. It is meant to be auditable: the *what*, the *why*, and
the *how to reproduce*.

## Constants

`omega_cov/thresholds.py` defines two constants:

| Constant | Role | v0.1 default | Calibrated on |
| --- | --- | --- | --- |
| `THRESHOLD_ANTI` | A_cov below this is `ANTI` (candidate confabulation) | `-0.1` (placeholder) | TriviaQA + WikiBio |
| `THRESHOLD_ALIGNED` | A_cov above this is `ALIGNED` (surprise covaries with displacement) | `+0.1` (placeholder) | TBD — separate corpus |

## Methodology for `THRESHOLD_ANTI`

The ANTI regime corresponds to "surprise without semantic movement":
the model is uncertain at a token, but the hidden state is not moving.
That signature is consistent with confabulation — making something up
that has no anchor in the latent state.

To calibrate, we need two distributions of A_cov: one from text we
believe is *not* confabulated, one from text we believe *is* (or is
much more likely to be).

### TriviaQA

- Split: `rc.nocontext`, validation.
- Sample: 1000 questions, seeded.
- For each question:
    1. Mistral-7B-v0.1 generates an answer (sampled, T=0.7, ≤64 tokens).
    2. We compute A_cov on the *generated tokens only*, in context of the
       question prompt (forward pass over `prompt + generation`,
       slice the displacements and entropies starting at the prompt
       boundary).
    3. We label the generation:
        - `correct` — the generation contains a TriviaQA-provided alias.
          Cheap, deterministic, no API call.
        - `hallucination` / `ambiguous` — Claude Haiku 4.5 (T=0) judges
          the rest, returning a one-line JSON verdict. Paraphrases and
          alternate spellings should be absorbed here.

The `label_source` column distinguishes alias matches from judge
verdicts so the split is auditable.

### WikiBio

- Split: `wiki_bio` test (or val if test not available).
- Sample: 500 entries, seeded.
- For each entry:
    1. **REF**: the reference biography text. A_cov measured on it
       directly. Labeled `REF`.
    2. **GEN**: Mistral-7B prompted with
       `"Write a 150-word biography of <name>"` (name extracted from
       the WikiBio infobox). A_cov measured on the generation.
       Labeled `GEN`.

No external labeling is needed: the label is the source. REF is
factual by construction, GEN is at risk of confabulation. We are
not claiming every GEN is wrong — we are claiming that, *on
average*, GEN should sit lower in A_cov than REF. The threshold
falls between the two distributions.

### Youden's J

For each dataset, the threshold `t` is the value of A_cov that
maximizes `TPR - FPR` for the binary rule "flag as ANTI iff
`A_cov < t`":

- TriviaQA: positives = hallucinations, negatives = correct,
  ambiguous excluded.
- WikiBio: positives = GEN, negatives = REF.

We compute the optimum independently per dataset and recommend the
**mean of the two**. The **max** (conservative) is also reported;
that one favors precision at the cost of recall, useful when false
flags are expensive.

## Reproducing

See `scripts/README.md` for the exact commands. Sample sizes and
the seed are CLI flags — record both in the commit that updates
`thresholds.py`.

```bash
pip install -e ".[calibrate]"
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/calibrate_trivia.py  --n 1000 --seed 42
python scripts/calibrate_wikibio.py --n 500  --seed 42
python scripts/analyze_thresholds.py
```

## Results

This section is populated after a calibration run.

### Run log

| Date | Model | Seed | TriviaQA n | WikiBio n | TriviaQA opt | WikiBio opt | Recommended | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _pending_ | Mistral-7B-v0.1 | 42 | 1000 | 500 | _t.b.d._ | _t.b.d._ | _t.b.d._ | _t.b.d._ |

### Distribution snapshots

To be filled in from `data/calibration_results.json` after the first run.
Include median + IQR for each labeled subset on each dataset.

## Why `THRESHOLD_ALIGNED` is not set here

The ALIGNED regime — surprise *and* displacement, covarying — is
the signature of dense, novel content. TriviaQA's terse answers
and WikiBio's biographical prose are not where ALIGNED lives by
construction. Calibrating ALIGNED on these corpora would
systematically push the threshold too low.

`THRESHOLD_ALIGNED` should be calibrated separately on a corpus
where the ALIGNED regime is genuinely present: scientific
abstracts, technical documentation, dense expository writing.
That calibration is intentionally out of scope for the v0.1
release.
