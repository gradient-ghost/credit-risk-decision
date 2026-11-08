"""Probability Calibration Module.

Calibrates candidate credit risk models using:
- Platt Scaling (Sigmoid / Logistic regression on log-odds)
- Isotonic Regression (Non-parametric piece-wise constant monotonic mapping)

Generates publication-quality reliability diagrams and exports the champion model.
"""

import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from src.models.evaluate import (
    bootstrap_gini_ci,
    brier_score_loss,
    evaluate_predictions,
    gini_score,
    ks_statistic,
    roc_auc_score,
)


def get_calibrated_classifier(
    estimator: BaseEstimator,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Instantiate a CalibratedClassifierCV wrapper compatible across scikit-learn versions.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted base estimator.
    method : str, default='sigmoid'
        Calibration method: 'sigmoid' (Platt) or 'isotonic'.

    Returns
    -------
    CalibratedClassifierCV
        Configured calibrator for pre-fit models.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(
            estimator=FrozenEstimator(estimator),
            method=method,
        )
    except (ImportError, Exception):
        return CalibratedClassifierCV(
            estimator=estimator,
            cv="prefit",
            method=method,
        )


def calibrate_model(
    estimator: BaseEstimator,
    X_val: Union[np.ndarray, pd.DataFrame],
    y_val: Union[np.ndarray, pd.Series],
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Fit a calibration model on validation data.

    Parameters
    ----------
    estimator : BaseEstimator
        Pre-fitted model instance.
    X_val : array-like of shape (n_samples, n_features)
        Validation feature matrix.
    y_val : array-like of shape (n_samples,)
        Validation target binary labels.
    method : str, default='sigmoid'
        'sigmoid' (Platt scaling) or 'isotonic'.

    Returns
    -------
    CalibratedClassifierCV
        Calibrated model ready for probability inference.
    """
    calibrated = get_calibrated_classifier(estimator, method=method)
    y_val_arr = np.asarray(y_val).ravel()
    calibrated.fit(X_val, y_val_arr)
    return calibrated


def plot_reliability_diagrams(
    models_dict: Dict[str, Tuple[BaseEstimator, np.ndarray]],
    y_test: Union[np.ndarray, pd.Series],
    output_path: Union[str, Path] = "reports/figures/calibration_curves.png",
    n_bins: int = 10,
) -> None:
    """Generate and save publication-grade Reliability Diagrams (Calibration Curves).

    Parameters
    ----------
    models_dict : Dict[str, Tuple[BaseEstimator, np.ndarray]]
        Mapping of model label -> (model_object, predicted_probabilities).
    y_test : array-like of shape (n_samples,)
        True binary labels on test set.
    output_path : str or Path
        Destination filepath for saved diagram.
    n_bins : int, default=10
        Number of probability buckets for calibration curve.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_test_arr = np.asarray(y_test).ravel()

    # Style setup
    fig, (ax_curve, ax_hist) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Reference diagonal (perfect calibration)
    ax_curve.plot(
        [0, 1], [0, 1], linestyle="--", color="#666666", label="Perfect Calibration (y = x)", linewidth=1.5
    )

    # Color palette
    colors = {
        "XGBoost (Uncalibrated)": "#e74c3c",
        "XGBoost (Platt Scaling)": "#27ae60",
        "XGBoost (Isotonic)": "#2980b9",
        "LightGBM": "#f39c12",
        "Logistic Regression": "#8e44ad",
        "LC Grade Classifier": "#7f8c8d",
    }
    fallback_colors = ["#16a085", "#d35400", "#2c3e50", "#c0392b"]

    color_idx = 0
    for label, (_, probas) in models_dict.items():
        prob_arr = np.asarray(probas).ravel()
        brier = brier_score_loss(y_test_arr, prob_arr)

        prob_true, prob_pred = calibration_curve(
            y_test_arr, prob_arr, n_bins=n_bins, strategy="uniform"
        )

        color = colors.get(label, fallback_colors[color_idx % len(fallback_colors)])
        color_idx += 1

        ax_curve.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            label=f"{label} (Brier: {brier:.4f})",
            color=color,
        )

        # Plot density distribution
        ax_hist.hist(
            prob_arr,
            bins=50,
            histtype="step",
            linewidth=1.5,
            density=True,
            label=label,
            color=color,
            alpha=0.7,
        )

    # Formatting Top Plot
    ax_curve.set_ylabel("Empirical Default Rate (Fraction of Positives)", fontsize=11, fontweight="bold")
    ax_curve.set_title(
        "Credit Risk Reliability Diagram & Probability Calibration",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax_curve.grid(True, linestyle=":", alpha=0.6)
    ax_curve.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    ax_curve.set_xlim([0.0, 1.0])
    ax_curve.set_ylim([0.0, 1.0])

    # Formatting Bottom Histogram Plot
    ax_hist.set_xlabel("Predicted Default Probability $\\hat{P}(Y=1)$", fontsize=11, fontweight="bold")
    ax_hist.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax_hist.grid(True, linestyle=":", alpha=0.6)
    ax_hist.set_xlim([0.0, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Calibration curve successfully saved to: {output_path}")


def log_model_runs(
    runs_records: List[Dict[str, Union[str, float, int]]],
    output_path: Union[str, Path] = "models/runs_log.csv",
) -> pd.DataFrame:
    """Log model run metrics across partitions to CSV.

    Parameters
    ----------
    runs_records : List[Dict[str, Union[str, float, int]]]
        List of records containing model, split, and metric scores.
    output_path : str or Path
        Destination CSV path.

    Returns
    -------
    pd.DataFrame
        Consolidated runs dataframe.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_runs = pd.DataFrame(runs_records)
    df_runs.to_csv(output_path, index=False)
    print(f"Logged {len(df_runs)} model evaluation records to {output_path}")
    return df_runs


def save_champion_model(
    model: BaseEstimator,
    output_path: Union[str, Path] = "models/champion_model.joblib",
) -> None:
    """Serialize the champion calibrated model to disk.

    Parameters
    ----------
    model : BaseEstimator
        Calibrated model pipeline / estimator.
    output_path : str or Path
        Destination joblib file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Champion model successfully serialized to: {output_path}")
