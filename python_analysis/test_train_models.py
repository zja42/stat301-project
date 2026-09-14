"""Lightweight regression tests for the modeling pipeline."""

import unittest

import numpy as np

import train_models


class ModelingPipelineTests(unittest.TestCase):
    def test_source_data_contract(self) -> None:
        X, y = train_models.load_data()
        self.assertEqual(X.shape, (7550, 17))
        self.assertEqual(len(y), 7550)
        self.assertEqual(int(y.sum()), 1137)
        self.assertEqual(set(y.unique()), {0, 1})

    def test_candidate_model_registry(self) -> None:
        X, _ = train_models.load_data()
        models = train_models.build_models(X.columns.tolist())
        self.assertEqual(
            set(models),
            {
                "Dummy baseline",
                "Ordinary logistic regression",
                "LASSO logistic regression",
                "Ridge logistic regression",
                "Random forest",
                "Extra trees",
                "Histogram gradient boosting",
            },
        )

    def test_metric_calculation(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.7, 0.6, 0.9])
        metrics = train_models.classification_metrics(
            y_true, probabilities, threshold=0.5
        )
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 0)
        self.assertEqual(metrics["true_positive"], 2)
        self.assertAlmostEqual(metrics["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
