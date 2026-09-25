"""Entity Matching Classifier and Threshold Optimizer.

Provides high-performance GBDT and linear classifiers trained on pairwise
similarity features, with custom threshold calibration designed to maximize
official macro-averaged F0.5.
"""

from typing import Any, Dict, List, Mapping, Optional, Set, Tuple
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import evaluate_predictions


class EntityMatchingClassifier:
    """Classifier wrapper for pairwise entity resolution."""

    def __init__(
        self,
        model_type: str = "hist_gbdt",
        max_iter: int = 150,
        learning_rate: float = 0.08,
        max_leaf_nodes: int = 31,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.optimal_threshold: float = 0.5

        if model_type == "hist_gbdt":
            self.model = HistGradientBoostingClassifier(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_leaf_nodes=max_leaf_nodes,
                random_state=random_state,
                class_weight="balanced",
            )
            self.scaler = None
        elif model_type == "logistic":
            self.scaler = StandardScaler()
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight="balanced",
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the classifier on pairwise feature matrix X and binary labels y."""
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive match."""
        if self.scaler is not None:
            X = self.scaler.transform(X)
        probs = self.model.predict_proba(X)
        return probs[:, 1]

    def optimize_threshold_f05(
        self,
        val_pairs: List[Tuple[str, str]],
        val_probs: np.ndarray,
        val_ground_truth: Mapping[str, Set[str]],
        threshold_range: Tuple[float, float, float] = (0.20, 0.95, 0.05),
    ) -> Tuple[float, float, Dict[float, float]]:
        """Search for decision threshold that maximizes Macro F0.5.

        Args:
            val_pairs: List of (s1_id, target_id) candidate pairs.
            val_probs: Predicted match probabilities for val_pairs.
            val_ground_truth: Ground truth mapping s1_id -> set of true matches.
            threshold_range: (min_thresh, max_thresh, step).

        Returns:
            (best_threshold, best_macro_f05, threshold_scores_dict)
        """
        min_t, max_t, step = threshold_range
        thresholds = np.arange(min_t, max_t + 1e-5, step)

        best_f05 = -1.0
        best_t = 0.5
        scores = {}

        # Pre-group pairs by s1_id with their probabilities for fast evaluation
        s1_to_pair_probs: Dict[str, List[Tuple[str, float]]] = {}
        for s1_id in val_ground_truth.keys():
            s1_to_pair_probs[s1_id] = []

        for (s1_id, tgt_id), prob in zip(val_pairs, val_probs):
            if s1_id in s1_to_pair_probs:
                s1_to_pair_probs[s1_id].append((tgt_id, prob))

        for t in thresholds:
            t = round(float(t), 4)
            preds: Dict[str, Set[str]] = {}
            for s1_id, cand_list in s1_to_pair_probs.items():
                matched = {tgt_id for tgt_id, prob in cand_list if prob >= t}
                preds[s1_id] = matched

            report = evaluate_predictions(preds, val_ground_truth)
            f05 = report.macro_f0_5
            scores[t] = f05

            if f05 > best_f05:
                best_f05 = f05
                best_t = t

        self.optimal_threshold = best_t
        return best_t, best_f05, scores
