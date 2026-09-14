# Online Shopper Purchase Prediction

## Executive Summary

This project compared 7 classification approaches on **7,550 online
shopping sessions**. Only **1,137 sessions (15.1%)** ended in a
purchase, so PR-AUC was used as the main model-selection metric rather than raw
accuracy.

**Random forest** achieved the strongest cross-validated PR-AUC and was selected
before the final test evaluation. On the untouched test set it reached
**0.726 PR-AUC** and **0.929 ROC-AUC**.

At the tuned threshold of **0.403**, the model identified
**80.6% of purchasing sessions** with **59.0%
precision**. This is a business trade-off: a lower threshold catches more likely
buyers but also sends more non-buyers into a campaign or sales workflow.

![Model comparison](figures/model_comparison.png)

## Why Accuracy Is Not Enough

A model that predicts "no purchase" for every session is already
84.9% accurate because purchases are uncommon. The dummy baseline
makes that limitation visible. PR-AUC focuses evaluation on how well a model
finds the minority purchase class.

![Class balance](figures/class_balance.png)

## Relationship to the Final R Project

The `prediction` branch's final R project compares ordinary, LASSO, and ridge
logistic regression after filtering out `Region == 1`. It selected LASSO and
reported test AUC **0.909**, sensitivity **0.377**, and balanced accuracy
**0.677** at a 0.30 threshold.

This Python extension keeps those three logistic families as reference models
and adds a dummy baseline plus Random Forest, Extra Trees, and Histogram
Gradient Boosting. The Python and R splits use the same seed but different
random-number implementations, so they are methodologically aligned rather than
row-for-row identical.

## Model Selection

Five-fold stratified cross-validation was performed only on the development
sample. The test sample remained untouched until the model and tuning approach
had been chosen.

| Model | CV PR-AUC | CV ROC-AUC | CV F1 |
|---|---:|---:|---:|
| Random forest | 0.705 | 0.922 | 0.634 |
| Histogram gradient boosting | 0.696 | 0.917 | 0.626 |
| LASSO logistic regression | 0.655 | 0.905 | 0.608 |
| Ridge logistic regression | 0.644 | 0.896 | 0.586 |
| Ordinary logistic regression | 0.640 | 0.892 | 0.582 |
| Extra trees | 0.566 | 0.871 | 0.542 |
| Dummy baseline | 0.151 | 0.500 | 0.000 |

## Held-Out Test Results

The table below uses the standard 0.50 probability threshold so the models are
directly comparable.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Histogram gradient boosting | 0.734 | 0.930 | 0.573 | 0.783 | 0.662 |
| Random forest | 0.726 | 0.929 | 0.678 | 0.698 | 0.688 |
| LASSO logistic regression | 0.652 | 0.913 | 0.534 | 0.792 | 0.638 |
| Ridge logistic regression | 0.643 | 0.906 | 0.501 | 0.789 | 0.613 |
| Ordinary logistic regression | 0.642 | 0.903 | 0.505 | 0.789 | 0.616 |
| Extra trees | 0.598 | 0.883 | 0.561 | 0.581 | 0.571 |
| Dummy baseline | 0.151 | 0.500 | 0.000 | 0.000 | 0.000 |

![Precision-recall curves](figures/precision_recall_curves.png)

## Operating Point for the Champion

The final threshold was selected from out-of-fold development predictions to
maximize F1, not from the test set.

| Metric | Default 0.50 | Tuned 0.403 |
|---|---:|---:|
| Precision | 0.678 | 0.590 |
| Recall | 0.698 | 0.806 |
| F1 | 0.688 | 0.682 |
| Balanced accuracy | 0.820 | 0.854 |

![Champion confusion matrix](figures/champion_confusion_matrix.png)

## Model Drivers

Model-agnostic permutation importance measures how much held-out PR-AUC falls
when each original feature is randomly shuffled.

| Feature | Mean PR-AUC decrease |
|---|---:|
| Page value | 0.5050 |
| Month | 0.0366 |
| Exit rate | 0.0320 |
| Bounce rate | 0.0270 |
| Administrative pages viewed | 0.0170 |
| Product pages viewed | 0.0144 |
| Product-page duration | 0.0119 |
| Visitor type | 0.0118 |

The leading signals were **Page value, Month, Exit rate, Bounce rate, Administrative pages viewed, Product pages viewed**. Importance shows what the
model relies on for prediction; it does not prove that changing a feature will
cause a purchase.

![Feature importance](figures/feature_importance.png)

## Deployment Sensitivity: Page Value

`PageValues` dominates predictive performance, but it may not be available at
the exact moment a live decision must be made. The champion model was therefore
retrained and evaluated without it.

| Feature set | CV PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|---:|
| All features | 0.705 | 0.726 | 0.929 |
| Without PageValues | 0.336 | 0.340 | 0.770 |

Removing the field reduced held-out PR-AUC by **0.386**. This does
not prove leakage, but it makes feature availability and calculation timing the
first production-readiness question.

![Page Value sensitivity](figures/page_value_sensitivity.png)

## Practical Interpretation

1. Use the score to prioritize sessions for interventions such as targeted
   offers, remarketing, or live assistance.
2. Set the decision threshold from campaign economics. When missing a buyer is
   expensive, favor recall; when outreach is costly, favor precision.
3. Treat Page Value carefully in production. Confirm that it is available at
   scoring time and is not calculated using information observed after purchase.
4. Monitor performance by month, visitor type, traffic source, and region before
   deployment because behavior and traffic mix can change.

## Limitations

- The random split estimates performance on sessions drawn from the same
  historical distribution; it is not a substitute for future-period validation.
- The dataset records session behavior, not customer lifetime value or campaign
  cost.
- Model importance is predictive rather than causal.
- Integer-coded fields such as browser, region, and traffic type were treated as
  categories rather than ordered quantities.

## Reproduce the Analysis

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python_analysis/requirements.txt
python python_analysis/train_models.py
```

Generated tables are stored in `python_analysis/results/`; generated charts
are stored in `python_analysis/figures/`.
