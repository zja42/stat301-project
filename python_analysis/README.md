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

- Stratified 80/20 development-test split
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

Random Forest was selected using cross-validated PR-AUC. It achieved **0.725
test PR-AUC** and **0.922 test ROC-AUC**. Removing `PageValues` reduced test
PR-AUC to **0.367**, so its availability and calculation timing must be validated
before any production use.
