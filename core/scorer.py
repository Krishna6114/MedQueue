"""
Urgency Scoring & Analytic Token-Level Explainability Engine for MedQueue.

Implements:
  1. TF-IDF text encoding over unigrams and bigrams (max_features=200).
  2. Bagged ensemble of 15 Ridge Regression estimators predicting urgency u_i in [0, 100].
  3. Ensemble-agreement confidence c_i in [0, 1] derived from inter-model dispersion.
  4. Exact linear token-level feature attribution (XAI) identifying which words/bigrams
     pushed the urgency score upward or downward.
  5. 5-fold out-of-fold cross-validation scoring for leakage-free simulation benchmarks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold


@dataclass
class TokenAttribution:
    """Represents a single n-gram token's contribution to the urgency score."""
    token: str
    tfidf_weight: float
    model_coefficient: float
    contribution: float  # Positive = increases urgency; Negative = decreases urgency


@dataclass
class ScoringResult:
    """Structured output returned by UrgencyScorer for a patient symptom text."""
    urgency_score: float               # u_i in [0, 100]
    confidence: float                  # c_i in [0, 1]
    prediction_std: float              # Raw dispersion across the 15 bagged models
    top_contributors: List[TokenAttribution] = field(default_factory=list)
    explanation_summary: str = ""

    def to_dict(self) -> Dict:
        return {
            "urgency_score": round(self.urgency_score, 2),
            "confidence": round(self.confidence, 4),
            "prediction_std": round(self.prediction_std, 4),
            "explanation_summary": self.explanation_summary,
            "top_contributors": [
                {
                    "token": attr.token,
                    "tfidf_weight": round(attr.tfidf_weight, 4),
                    "model_coefficient": round(attr.model_coefficient, 4),
                    "contribution": round(attr.contribution, 2),
                }
                for attr in self.top_contributors
            ],
        }


class UrgencyScorer:
    """
    TF-IDF + 15-Estimator Bagged Ridge Regression Urgency Scorer with Token XAI.
    """

    def __init__(
        self,
        max_features: int = 200,
        ngram_range: Tuple[int, int] = (1, 2),
        n_estimators: int = 15,
        ridge_alpha: float = 1.0,
        confidence_scale: float = 10.0,
        random_state: int = 42,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.n_estimators = n_estimators
        self.ridge_alpha = ridge_alpha
        self.confidence_scale = confidence_scale
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            lowercase=True,
            stop_words="english",
            sublinear_tf=True,
        )

        base_ridge = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
        self.ensemble = BaggingRegressor(
            estimator=base_ridge,
            n_estimators=self.n_estimators,
            max_samples=0.85,
            bootstrap=True,
            random_state=self.random_state,
        )

        self._is_fitted: bool = False
        self._mean_coef: Optional[np.ndarray] = None
        self._mean_intercept: float = 0.0
        self._feature_names: Optional[np.ndarray] = None

    def fit(self, texts: List[str] | pd.Series, severities: np.ndarray | pd.Series) -> "UrgencyScorer":
        """Fit TF-IDF vectorizer and the 15-model bagged Ridge ensemble."""
        y = np.asarray(severities, dtype=np.float64)
        X = self.vectorizer.fit_transform(texts)
        self.ensemble.fit(X, y)

        # Compute ensemble-averaged linear weights for fast, exact token attribution
        coefs = np.array([est.coef_ for est in self.ensemble.estimators_], dtype=np.float64)
        intercepts = np.array([est.intercept_ for est in self.ensemble.estimators_], dtype=np.float64)

        self._mean_coef = np.mean(coefs, axis=0)
        self._mean_intercept = float(np.mean(intercepts))
        self._feature_names = self.vectorizer.get_feature_names_out()
        self._is_fitted = True
        return self

    def _compute_confidence(self, member_preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ensemble dispersion (std) and normalized agreement confidence c_i in [0, 1].
        Lower spread across the 15 bagged models yields higher confidence.
        """
        stds = np.std(member_preds, axis=0)
        confidences = np.exp(-stds / self.confidence_scale)
        confidences = np.clip(confidences, 0.0, 1.0)
        return confidences, stds

    def predict_batch(self, texts: List[str] | pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict urgency scores u_i in [0, 100] and confidences c_i in [0, 1] for a batch of texts.
        """
        if not self._is_fitted:
            raise RuntimeError("UrgencyScorer must be fitted before calling predict_batch().")

        X = self.vectorizer.transform(texts)
        member_preds = np.array([est.predict(X) for est in self.ensemble.estimators_], dtype=np.float64)
        raw_mean = np.mean(member_preds, axis=0)
        urgencies = np.clip(raw_mean, 0.0, 100.0)
        confidences, _ = self._compute_confidence(member_preds)
        return urgencies, confidences

    def score_case(self, symptom_text: str, top_k_tokens: int = 5) -> ScoringResult:
        """
        Score a single patient symptom description and generate exact token-level explanations.
        """
        if not self._is_fitted or self._mean_coef is None or self._feature_names is None:
            raise RuntimeError("UrgencyScorer must be fitted before calling score_case().")

        cleaned_text = symptom_text.strip()
        if not cleaned_text:
            return ScoringResult(
                urgency_score=50.0,
                confidence=0.0,
                prediction_std=0.0,
                top_contributors=[],
                explanation_summary="Empty symptom text; assigned neutral fallback urgency with zero confidence.",
            )

        X = self.vectorizer.transform([cleaned_text])
        member_preds = np.array([est.predict(X) for est in self.ensemble.estimators_], dtype=np.float64)

        raw_urgency = float(np.mean(member_preds))
        urgency = float(np.clip(raw_urgency, 0.0, 100.0))
        conf_arr, std_arr = self._compute_confidence(member_preds)
        confidence = float(conf_arr[0])
        pred_std = float(std_arr[0])

        # Exact linear feature attribution: x_j * beta_bar_j
        row_dense = X.toarray()[0]
        nonzero_indices = np.nonzero(row_dense)[0]

        attributions: List[TokenAttribution] = []
        for idx in nonzero_indices:
            tfidf_val = float(row_dense[idx])
            coef_val = float(self._mean_coef[idx])
            contrib = tfidf_val * coef_val
            attributions.append(
                TokenAttribution(
                    token=str(self._feature_names[idx]),
                    tfidf_weight=tfidf_val,
                    model_coefficient=coef_val,
                    contribution=contrib,
                )
            )

        # Sort by absolute impact magnitude descending
        attributions.sort(key=lambda item: abs(item.contribution), reverse=True)
        top_attrs = attributions[:top_k_tokens]

        if not top_attrs:
            summary = (
                f"Estimated urgency {urgency:.1f}/100 (confidence {confidence:.2f}). "
                "No recognized vocabulary n-grams matched; score reflects baseline prior."
            )
        else:
            phrases = [
                f"'{a.token}' ({'+' if a.contribution >= 0 else ''}{a.contribution:.1f})"
                for a in top_attrs[:3]
            ]
            summary = (
                f"Estimated urgency {urgency:.1f}/100 (confidence {confidence:.2f}). "
                f"Key drivers: {', '.join(phrases)}."
            )

        return ScoringResult(
            urgency_score=urgency,
            confidence=confidence,
            prediction_std=pred_std,
            top_contributors=top_attrs,
            explanation_summary=summary,
        )

    def save(self, filepath: str) -> None:
        """Persist fitted scorer artifact to disk with canonical module path."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.__class__.__module__ = "core.scorer"
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "UrgencyScorer":
        """Load persisted scorer artifact from disk."""
        return joblib.load(filepath)


def generate_out_of_fold_scores(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Perform 5-fold cross-validation on the synthetic dataset so every case receives
    an urgency score u_i and confidence c_i from a model that never saw it in training.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_urgency = np.zeros(len(df), dtype=np.float64)
    oof_confidence = np.zeros(len(df), dtype=np.float64)

    texts = df["symptom_text"].tolist()
    y = df["true_severity"].to_numpy(dtype=np.float64)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(texts)):
        fold_scorer = UrgencyScorer(random_state=random_state + fold_idx)
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]

        fold_scorer.fit(train_texts, y[train_idx])
        u_val, c_val = fold_scorer.predict_batch(val_texts)

        oof_urgency[val_idx] = u_val
        oof_confidence[val_idx] = c_val

    scored_df = df.copy()
    scored_df["predicted_urgency"] = np.round(oof_urgency, 2)
    scored_df["confidence"] = np.round(oof_confidence, 4)

    metrics = {
        "cv_mae": round(float(mean_absolute_error(y, oof_urgency)), 3),
        "cv_r2": round(float(r2_score(y, oof_urgency)), 4),
        "mean_confidence": round(float(np.mean(oof_confidence)), 4),
    }
    return scored_df, metrics


def train_and_save_production_scorer() -> None:
    """Train UrgencyScorer from synthetic_cases.csv and save with canonical module path."""
    csv_path = os.path.join("data", "synthetic_cases.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run 'python -m core.dataset' first.")

    df_cases = pd.read_csv(csv_path)
    scored_cases, cv_metrics = generate_out_of_fold_scores(df_cases, n_splits=5, random_state=42)

    oof_output_path = os.path.join("data", "scored_synthetic_cases.csv")
    scored_cases.to_csv(oof_output_path, index=False)

    production_scorer = UrgencyScorer(random_state=42)
    production_scorer.fit(df_cases["symptom_text"], df_cases["true_severity"])
    model_path = os.path.join("models", "urgency_scorer.joblib")
    production_scorer.save(model_path)

    print("Production scorer re-saved cleanly with module path 'core.scorer.UrgencyScorer'.")
    print(f"5-Fold CV R2: {cv_metrics['cv_r2']} | MAE: {cv_metrics['cv_mae']}")


if __name__ == "__main__":
    train_and_save_production_scorer()