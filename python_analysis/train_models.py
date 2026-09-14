#!/usr/bin/env python3
"""Train and explain purchase-propensity models for the shoppers dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ["LOKY_MAX_CPU_COUNT"] = (
    os.environ.get("LOKY_MAX_CPU_COUNT") or str(os.cpu_count() or 1)
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "online_shoppers_intention.csv"
OUTPUT_DIR = ROOT / "python_analysis"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

TARGET = "Revenue"
CATEGORICAL_COLUMNS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

FEATURE_LABELS = {
    "Administrative": "Administrative pages viewed",
    "Administrative_Duration": "Administrative-page duration",
    "Informational": "Informational pages viewed",
    "Informational_Duration": "Informational-page duration",
    "ProductRelated": "Product pages viewed",
    "ProductRelated_Duration": "Product-page duration",
    "BounceRates": "Bounce rate",
    "ExitRates": "Exit rate",
    "PageValues": "Page value",
    "SpecialDay": "Proximity to a special day",
    "Month": "Month",
    "OperatingSystems": "Operating system",
    "Browser": "Browser",
    "Region": "Region",
    "TrafficType": "Traffic source",
    "VisitorType": "Visitor type",
    "Weekend": "Weekend session",
}


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the source CSV and return features plus a binary target."""
    data = pd.read_csv(DATA_PATH)
    if data[TARGET].isna().any():
        raise ValueError("Revenue contains missing values.")
    X = data.drop(columns=TARGET)
    y = data[TARGET].astype(bool).astype(int)
    return X, y


def make_preprocessor(columns: list[str]) -> ColumnTransformer:
    """Create leakage-safe preprocessing fitted inside each CV fold."""
    categorical = [column for column in CATEGORICAL_COLUMNS if column in columns]
    numeric = [column for column in columns if column not in categorical]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        verbose_feature_names_out=False,
    )


def build_models(columns: list[str]) -> dict[str, Pipeline]:
    """Return interpretable, bagged-tree, and boosted-tree candidates."""
    estimators = {
        "Dummy baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=SEED,
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "Extra trees": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        ),
        "Histogram gradient boosting": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=SEED,
        ),
    }
    return {
        name: Pipeline(
            [
                ("preprocess", make_preprocessor(columns)),
                ("classifier", estimator),
            ]
        )
        for name, estimator in estimators.items()
    }


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate probability and threshold-based classification metrics."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "brier_score": brier_score_loss(y_true, probabilities),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def best_f1_threshold(
    y_true: pd.Series, probabilities: np.ndarray
) -> tuple[float, float]:
    """Select a threshold from out-of-fold predictions without test leakage."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    scores = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    index = int(np.nanargmax(scores))
    return float(thresholds[index]), float(scores[index])


def save_class_balance(y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    labels = ["No purchase", "Purchase"]
    colors = ["#64748b", "#0f766e"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(labels, counts.values, color=colors, width=0.58)
    ax.set_title("Class balance in 12,330 shopping sessions", loc="left", weight="bold")
    ax.set_ylabel("Sessions")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, counts.values):
        share = value / counts.sum()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + counts.max() * 0.02,
            f"{value:,} ({share:.1%})",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=180)
    plt.close(fig)


def save_model_comparison(test_results: pd.DataFrame) -> None:
    default = test_results[test_results["evaluation"] == "default threshold"].copy()
    default = default.sort_values("pr_auc", ascending=True)
    labels = default["model"].tolist()
    positions = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.barh(
        positions - width / 2,
        default["pr_auc"],
        height=width,
        label="PR-AUC",
        color="#0f766e",
    )
    ax.barh(
        positions + width / 2,
        default["roc_auc"],
        height=width,
        label="ROC-AUC",
        color="#d97706",
    )
    ax.set_yticks(positions, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Held-out test score")
    ax.set_title("Model comparison on the untouched test set", loc="left", weight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)


def save_precision_recall_curves(
    y_test: pd.Series, test_probabilities: dict[str, np.ndarray]
) -> None:
    palette = ["#64748b", "#2563eb", "#0f766e", "#d97706", "#be123c"]
    fig, ax = plt.subplots(figsize=(8.2, 5.7))
    for color, (name, probabilities) in zip(palette, test_probabilities.items()):
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        score = average_precision_score(y_test, probabilities)
        ax.plot(recall, precision, label=f"{name} ({score:.3f})", color=color, lw=2)
    ax.axhline(y_test.mean(), color="#94a3b8", ls="--", lw=1.4, label="Base rate")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision-recall curves", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_curves.png", dpi=180)
    plt.close(fig)


def save_confusion_matrix(
    y_test: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
) -> None:
    matrix = confusion_matrix(y_test, probabilities >= threshold)
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=",",
        cmap="Blues",
        cbar=False,
        linewidths=1,
        linecolor="white",
        xticklabels=["No purchase", "Purchase"],
        yticklabels=["No purchase", "Purchase"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"{model_name}: tuned threshold = {threshold:.3f}",
        loc="left",
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "champion_confusion_matrix.png", dpi=180)
    plt.close(fig)


def save_feature_importance(importance: pd.DataFrame) -> None:
    top = importance.head(12).sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.barh(
        top["display_feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color="#2563eb",
        alpha=0.88,
        capsize=3,
    )
    ax.axvline(0, color="#334155", lw=0.8)
    ax.set_xlabel("Decrease in test PR-AUC after permutation")
    ax.set_title("What drives the champion model?", loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)


def save_page_value_sensitivity(sensitivity: pd.DataFrame) -> None:
    positions = np.arange(len(sensitivity))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    ax.bar(
        positions - width / 2,
        sensitivity["test_pr_auc"],
        width=width,
        label="PR-AUC",
        color="#0f766e",
    )
    ax.bar(
        positions + width / 2,
        sensitivity["test_roc_auc"],
        width=width,
        label="ROC-AUC",
        color="#d97706",
    )
    ax.set_xticks(positions, sensitivity["feature_set"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out test score")
    ax.set_title("Deployment sensitivity: Page Value", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "page_value_sensitivity.png", dpi=180)
    plt.close(fig)


def write_report(
    y: pd.Series,
    champion_name: str,
    threshold: float,
    cv_results: pd.DataFrame,
    test_results: pd.DataFrame,
    importance: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    champion = test_results[
        (test_results["model"] == champion_name)
        & (test_results["evaluation"] == "tuned threshold")
    ].iloc[0]
    default_champion = test_results[
        (test_results["model"] == champion_name)
        & (test_results["evaluation"] == "default threshold")
    ].iloc[0]
    top_features = importance.head(6)["display_feature"].tolist()
    cv_ranked = cv_results.sort_values("cv_pr_auc_mean", ascending=False)
    cv_rows = "\n".join(
        f"| {row.model} | {row.cv_pr_auc_mean:.3f} | "
        f"{row.cv_roc_auc_mean:.3f} | {row.cv_f1_mean:.3f} |"
        for row in cv_ranked.itertuples()
    )
    test_default = test_results[
        test_results["evaluation"] == "default threshold"
    ].sort_values("pr_auc", ascending=False)
    test_rows = "\n".join(
        f"| {row.model} | {row.pr_auc:.3f} | {row.roc_auc:.3f} | "
        f"{row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} |"
        for row in test_default.itertuples()
    )
    feature_rows = "\n".join(
        f"| {row.display_feature} | {row.importance_mean:.4f} |"
        for row in importance.head(8).itertuples()
    )
    with_page_value = sensitivity[sensitivity["feature_set"] == "All features"].iloc[0]
    without_page_value = sensitivity[
        sensitivity["feature_set"] == "Without PageValues"
    ].iloc[0]
    pr_auc_drop = with_page_value.test_pr_auc - without_page_value.test_pr_auc

    report = f"""# Online Shopper Purchase Prediction

## Executive Summary

This project compared five classification approaches on **{len(y):,} online
shopping sessions**. Only **{y.sum():,} sessions ({y.mean():.1%})** ended in a
purchase, so PR-AUC was used as the main model-selection metric rather than raw
accuracy.

**{champion_name}** achieved the strongest cross-validated PR-AUC and was selected
before the final test evaluation. On the untouched test set it reached
**{champion.pr_auc:.3f} PR-AUC** and **{champion.roc_auc:.3f} ROC-AUC**.

At the tuned threshold of **{threshold:.3f}**, the model identified
**{champion.recall:.1%} of purchasing sessions** with **{champion.precision:.1%}
precision**. This is a business trade-off: a lower threshold catches more likely
buyers but also sends more non-buyers into a campaign or sales workflow.

![Model comparison](figures/model_comparison.png)

## Why Accuracy Is Not Enough

A model that predicts "no purchase" for every session is already
{1 - y.mean():.1%} accurate because purchases are uncommon. The dummy baseline
makes that limitation visible. PR-AUC focuses evaluation on how well a model
finds the minority purchase class.

![Class balance](figures/class_balance.png)

## Model Selection

Five-fold stratified cross-validation was performed only on the development
sample. The test sample remained untouched until the model and tuning approach
had been chosen.

| Model | CV PR-AUC | CV ROC-AUC | CV F1 |
|---|---:|---:|---:|
{cv_rows}

## Held-Out Test Results

The table below uses the standard 0.50 probability threshold so the models are
directly comparable.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
{test_rows}

![Precision-recall curves](figures/precision_recall_curves.png)

## Operating Point for the Champion

The final threshold was selected from out-of-fold development predictions to
maximize F1, not from the test set.

| Metric | Default 0.50 | Tuned {threshold:.3f} |
|---|---:|---:|
| Precision | {default_champion.precision:.3f} | {champion.precision:.3f} |
| Recall | {default_champion.recall:.3f} | {champion.recall:.3f} |
| F1 | {default_champion.f1:.3f} | {champion.f1:.3f} |
| Balanced accuracy | {default_champion.balanced_accuracy:.3f} | {champion.balanced_accuracy:.3f} |

![Champion confusion matrix](figures/champion_confusion_matrix.png)

## Model Drivers

Model-agnostic permutation importance measures how much held-out PR-AUC falls
when each original feature is randomly shuffled.

| Feature | Mean PR-AUC decrease |
|---|---:|
{feature_rows}

The leading signals were **{", ".join(top_features)}**. Importance shows what the
model relies on for prediction; it does not prove that changing a feature will
cause a purchase.

![Feature importance](figures/feature_importance.png)

## Deployment Sensitivity: Page Value

`PageValues` dominates predictive performance, but it may not be available at
the exact moment a live decision must be made. The champion model was therefore
retrained and evaluated without it.

| Feature set | CV PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|---:|
| All features | {with_page_value.cv_pr_auc:.3f} | {with_page_value.test_pr_auc:.3f} | {with_page_value.test_roc_auc:.3f} |
| Without PageValues | {without_page_value.cv_pr_auc:.3f} | {without_page_value.test_pr_auc:.3f} | {without_page_value.test_roc_auc:.3f} |

Removing the field reduced held-out PR-AUC by **{pr_auc_drop:.3f}**. This does
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
"""
    (OUTPUT_DIR / "MODEL_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    X, y = load_data()
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    models = build_models(X.columns.tolist())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
    }

    cv_records: list[dict[str, float | str]] = []
    for name, model in models.items():
        print(f"Cross-validating {name}...")
        scores = cross_validate(
            model,
            X_development,
            y_development,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        cv_records.append(
            {
                "model": name,
                **{
                    f"cv_{metric}_mean": float(np.mean(scores[f"test_{metric}"]))
                    for metric in scoring
                },
                **{
                    f"cv_{metric}_std": float(np.std(scores[f"test_{metric}"]))
                    for metric in scoring
                },
            }
        )

    cv_results = pd.DataFrame(cv_records).sort_values(
        "cv_pr_auc_mean", ascending=False
    )
    champion_name = str(cv_results.iloc[0]["model"])
    champion_template = models[champion_name]

    print(f"Selecting threshold for {champion_name}...")
    oof_probabilities = cross_val_predict(
        champion_template,
        X_development,
        y_development,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold, development_f1 = best_f1_threshold(
        y_development, oof_probabilities
    )

    test_records: list[dict[str, float | int | str]] = []
    test_probabilities: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Pipeline] = {}
    for name, model in models.items():
        print(f"Fitting {name} on the full development set...")
        fitted = clone(model).fit(X_development, y_development)
        probabilities = fitted.predict_proba(X_test)[:, 1]
        fitted_models[name] = fitted
        test_probabilities[name] = probabilities
        test_records.append(
            {
                "model": name,
                "evaluation": "default threshold",
                **classification_metrics(y_test, probabilities, 0.50),
            }
        )

    champion = fitted_models[champion_name]
    champion_probabilities = test_probabilities[champion_name]
    test_records.append(
        {
            "model": champion_name,
            "evaluation": "tuned threshold",
            **classification_metrics(y_test, champion_probabilities, threshold),
        }
    )
    test_results = pd.DataFrame(test_records)

    print("Running PageValues sensitivity analysis...")
    reduced_columns = [column for column in X.columns if column != "PageValues"]
    reduced_model = build_models(reduced_columns)[champion_name]
    reduced_cv_scores = cross_validate(
        reduced_model,
        X_development[reduced_columns],
        y_development,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )
    reduced_fitted = clone(reduced_model).fit(
        X_development[reduced_columns], y_development
    )
    reduced_probabilities = reduced_fitted.predict_proba(
        X_test[reduced_columns]
    )[:, 1]
    reduced_metrics = classification_metrics(
        y_test, reduced_probabilities, 0.50
    )
    full_champion_default = test_results[
        (test_results["model"] == champion_name)
        & (test_results["evaluation"] == "default threshold")
    ].iloc[0]
    sensitivity = pd.DataFrame(
        [
            {
                "feature_set": "All features",
                "cv_pr_auc": float(
                    cv_results.loc[
                        cv_results["model"] == champion_name, "cv_pr_auc_mean"
                    ].iloc[0]
                ),
                "test_pr_auc": full_champion_default.pr_auc,
                "test_roc_auc": full_champion_default.roc_auc,
                "test_f1": full_champion_default.f1,
            },
            {
                "feature_set": "Without PageValues",
                "cv_pr_auc": float(np.mean(reduced_cv_scores["test_pr_auc"])),
                "test_pr_auc": reduced_metrics["pr_auc"],
                "test_roc_auc": reduced_metrics["roc_auc"],
                "test_f1": reduced_metrics["f1"],
            },
        ]
    )

    print("Calculating permutation importance...")
    permutation = permutation_importance(
        champion,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=SEED,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "display_feature": [
                FEATURE_LABELS.get(column, column) for column in X.columns
            ],
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    split_summary = pd.DataFrame(
        [
            {
                "split": "development",
                "sessions": len(y_development),
                "purchases": int(y_development.sum()),
                "purchase_rate": y_development.mean(),
            },
            {
                "split": "test",
                "sessions": len(y_test),
                "purchases": int(y_test.sum()),
                "purchase_rate": y_test.mean(),
            },
        ]
    )
    summary = {
        "random_seed": SEED,
        "primary_metric": "PR-AUC",
        "champion_model": champion_name,
        "tuned_threshold": threshold,
        "development_oof_f1_at_tuned_threshold": development_f1,
        "total_sessions": len(y),
        "purchase_sessions": int(y.sum()),
        "purchase_rate": float(y.mean()),
    }

    cv_results.to_csv(RESULTS_DIR / "cross_validation_metrics.csv", index=False)
    test_results.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    sensitivity.to_csv(
        RESULTS_DIR / "page_value_sensitivity.csv", index=False
    )
    split_summary.to_csv(RESULTS_DIR / "split_summary.csv", index=False)
    (RESULTS_DIR / "champion_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    save_class_balance(y)
    save_model_comparison(test_results)
    save_precision_recall_curves(y_test, test_probabilities)
    save_confusion_matrix(
        y_test, champion_probabilities, threshold, champion_name
    )
    save_feature_importance(importance)
    save_page_value_sensitivity(sensitivity)
    write_report(
        y,
        champion_name,
        threshold,
        cv_results,
        test_results,
        importance,
        sensitivity,
    )

    champion_test = test_results[
        (test_results["model"] == champion_name)
        & (test_results["evaluation"] == "tuned threshold")
    ].iloc[0]
    print(
        f"Done. Champion: {champion_name}; "
        f"test PR-AUC={champion_test.pr_auc:.3f}; "
        f"test ROC-AUC={champion_test.roc_auc:.3f}; "
        f"threshold={threshold:.3f}."
    )


if __name__ == "__main__":
    main()
