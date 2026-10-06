"""
ML core: `Is_Flood` binary classifier (Random Forest / optional XGBoost).

Responsibilities:
- Train a flood classifier on the telemetry features required by the spec.
- Expose held-out metrics (Accuracy, Precision, Recall, ROC-AUC).
- Provide single-row flood probability + a local feature-contribution
  explanation (leave-one-feature-at-training-baseline ablation), which works
  for any scikit-learn compatible classifier without extra dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from data import CWC_DANGER, CWC_WL, FEATURE_COLUMNS, TARGET

try:  # SHAP gives correct local attributions; ablation is the fallback.
    import shap

    HAS_SHAP = True
except Exception:  # pragma: no cover
    shap = None
    HAS_SHAP = False

RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass
class ModelBundle:
    """Trained model + everything the UI needs to render metrics/explanations."""

    model: RandomForestClassifier
    feature_columns: list[str]
    metrics: dict[str, float]
    confusion: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    importances: pd.Series
    baseline: dict[str, float]          # per-feature training baseline (median)
    ranges: dict[str, tuple[float, float]]  # per-feature (min, max) seen in training
    n_train: int
    n_test: int
    positive_rate: float
    notes: list[str] = field(default_factory=list)
    _explainer: object = field(default=None, repr=False, compare=False)

    def predict_proba_one(self, row: dict[str, float] | pd.Series) -> float:
        """Flood probability (0-1) for a single feature dict."""
        X = pd.DataFrame([{c: float(row.get(c, self.baseline[c])) for c in self.feature_columns}])
        return float(self.model.predict_proba(X)[0, 1])

    def _row_frame(self, row: dict[str, float] | pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            [{c: float(row.get(c, self.baseline[c])) for c in self.feature_columns}]
        )

    def contributions(self, row: dict[str, float] | pd.Series) -> pd.Series:
        """
        Local explanation: signed contribution of each feature to the flood
        probability for this specific input.

        Preferred: SHAP TreeExplainer values for class 1 (exact for trees;
        base + sum(shap) = probability).
        Fallback: leave-one-feature-at-training-median ablation.

        Returns a Series indexed by feature (descending |contribution|),
        in probability units (multiply by 100 for percentage points).
        """
        X = self._row_frame(row)

        if HAS_SHAP:
            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self.model)
            sv = self._explainer.shap_values(X)
            if isinstance(sv, list):  # older shap: one array per class
                vals = np.asarray(sv[1][0])
            else:
                arr = np.asarray(sv)
                vals = arr[0, :, 1] if arr.ndim == 3 else arr[0]
            return pd.Series(vals, index=self.feature_columns, dtype="float64").sort_values(
                key=np.abs, ascending=False
            )

        full = self.predict_proba_one(row)
        out = {}
        for col in self.feature_columns:
            ablated = dict(row)
            ablated[col] = self.baseline[col]
            out[col] = full - self.predict_proba_one(ablated)
        return pd.Series(out, dtype="float64").sort_values(key=np.abs, ascending=False)


def train_model(df: pd.DataFrame) -> ModelBundle:
    """Train the `Is_Flood` classifier and compute held-out metrics."""
    data = df.dropna(subset=FEATURE_COLUMNS + [TARGET]).copy()
    X = data[FEATURE_COLUMNS]
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, y_prob)) if y_test.nunique() > 1 else float("nan"),
    }
    fpr, tpr, thr = roc_curve(y_test, y_prob)

    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLUMNS
    ).sort_values(ascending=False)

    notes: list[str] = []
    if rule_check(data):
        notes.append(
            "In this dataset `Is_Flood` is exactly the rule "
            "`water level \u2265 danger level`, so the model learns that threshold "
            "near-perfectly \u2014 treat metrics as sanity checks, not "
            "generalisation estimates."
        )

    return ModelBundle(
        model=model,
        feature_columns=list(FEATURE_COLUMNS),
        metrics=metrics,
        confusion=confusion_matrix(y_test, y_pred, labels=[0, 1]),
        fpr=fpr,
        tpr=tpr,
        thresholds=thr,
        importances=importances,
        baseline={c: float(X_train[c].median()) for c in FEATURE_COLUMNS},
        ranges={c: (float(X_train[c].min()), float(X_train[c].max())) for c in FEATURE_COLUMNS},
        n_train=len(X_train),
        n_test=len(X_test),
        positive_rate=float(y.mean()),
        notes=notes,
    )


def rule_check(df: pd.DataFrame) -> bool:
    """True when Is_Flood is fully explained by WL >= Danger_Level_m."""
    return bool(
        (df[TARGET] == (df[CWC_WL] >= df[CWC_DANGER]).astype(int)).all()
    )


def risk_category(probability: float) -> tuple[str, str, str]:
    """Map a 0-1 probability to (label, emoji, css colour)."""
    p = probability * 100
    if p < 30:
        return ("SAFE / LOW RISK", "\U0001F7E2", "#16a34a")
    if p <= 70:
        return ("MODERATE RISK", "\U0001F7E1", "#d97706")
    return ("CRITICAL FLOOD ALERT", "\U0001F534", "#dc2626")
