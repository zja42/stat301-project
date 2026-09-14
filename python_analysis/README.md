# Python Purchase-Propensity Modeling

This module extends the original inference-focused R analysis into a reproducible
machine-learning comparison for predicting whether an online shopping session
will generate revenue.

## Run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python_analysis/requirements.txt
python python_analysis/train_models.py
```

The script regenerates all tables in `python_analysis/results/`, all charts in
`python_analysis/figures/`, and the plain-English
`python_analysis/MODEL_REPORT.md`.

## Evaluation Design

- The same `Region != 1` filter used by the final R project
- Stratified 70/30 development-test split
- Five-fold stratified cross-validation on the development set
- PR-AUC as the primary selection metric because purchases are uncommon
- One untouched test set for final model comparison
- Out-of-fold threshold tuning for the selected model
- Model-agnostic permutation importance on the held-out test set
- Deployment sensitivity analysis with `PageValues` removed

All random operations use seed `42`.

Run the lightweight regression tests with:

```bash
.venv/bin/python -m unittest discover -s python_analysis -p "test_*.py"
```

## Headline Result

Random Forest led cross-validation and achieved **0.726 test PR-AUC** and
**0.929 test ROC-AUC**. At the development-tuned threshold, it identified
**80.6% of purchases** with **59.0% precision**. The full report compares this
result with the final R project's ordinary, LASSO, and ridge logistic models and
tests how performance changes when `PageValues` is unavailable.
