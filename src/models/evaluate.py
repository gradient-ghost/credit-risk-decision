"""Model Evaluation & Metrics Module.

Provides standard credit risk validation metrics:
- ROC-AUC: Area Under Receiver Operating Characteristic Curve
- Gini Score: 2 * ROC_AUC - 1 (Somers' D / Power Ratio)
- KS Statistic: Kolmogorov-Smirnov two-sample maximum CDF divergence
- Brier Score Loss: Mean squared difference between predicted probabilities and actual outcomes
- Bootstrapped Gini Confidence Intervals: Non-parametric bootstrap resampling
"""

from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, roc_auc_score


def gini_score(y_true: Union[np.ndarray, pd.Series], y_prob: Union[np.ndarray, pd.Series]) -> float:
    """Calculate Gini coefficient (2 * AUC - 1).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels {0, 1}.
    y_prob : array-like of shape (n_samples,)
        Target class probability scores in [0, 1].

    Returns
    -------
    float
        Gini score in [-1.0, 1.0].
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_prob_arr = np.asarray(y_prob).ravel()

    # If only one class present in true labels, AUC is undefined
    if len(np.unique(y_true_arr)) < 2:
        return 0.0

    auc = roc_auc_score(y_true_arr, y_prob_arr)
    return float(2.0 * auc - 1.0)


def ks_statistic(y_true: Union[np.ndarray, pd.Series], y_prob: Union[np.ndarray, pd.Series]) -> float:
    """Calculate the two-sample Kolmogorov-Smirnov (KS) statistic.

    Measures maximum vertical separation between cumulative distribution
    functions of defaults (y=1) and non-defaults (y=0).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels {0, 1}.
    y_prob : array-like of shape (n_samples,)
        Predicted default probabilities in [0, 1].

    Returns
    -------
    float
        KS statistic in [0.0, 1.0].
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_prob_arr = np.asarray(y_prob).ravel()

    defaults = y_prob_arr[y_true_arr == 1]
    non_defaults = y_prob_arr[y_true_arr == 0]

    if len(defaults) == 0 or len(non_defaults) == 0:
        return 0.0

    stat = ks_2samp(defaults, non_defaults).statistic
    return float(stat)


def bootstrap_gini_ci(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    n_bootstraps: int = 200,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval for Gini score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels {0, 1}.
    y_prob : array-like of shape (n_samples,)
        Predicted default probabilities.
    n_bootstraps : int, default=200
        Number of bootstrap resamples.
    ci : float, default=0.95
        Target confidence interval width (e.g. 0.95 for 95% CI).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, float]
        (lower_ci, upper_ci) boundaries.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_prob_arr = np.asarray(y_prob).ravel()
    n_samples = len(y_true_arr)

    rng = np.random.default_rng(random_state)
    bootstrapped_ginis = []

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        sample_y = y_true_arr[boot_idx]
        sample_prob = y_prob_arr[boot_idx]

        if len(np.unique(sample_y)) < 2:
            continue

        auc = roc_auc_score(sample_y, sample_prob)
        bootstrapped_ginis.append(2.0 * auc - 1.0)

    if not bootstrapped_ginis:
        return 0.0, 0.0

    alpha = (1.0 - ci) / 2.0
    lower_bound = float(np.percentile(bootstrapped_ginis, alpha * 100.0))
    upper_bound = float(np.percentile(bootstrapped_ginis, (1.0 - alpha) * 100.0))

    return lower_bound, upper_bound


def evaluate_predictions(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Evaluate predictions returning standard credit risk metric dictionary.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_prob : array-like of shape (n_samples,)
        Predicted default probabilities.

    Returns
    -------
    Dict[str, float]
        Dictionary with roc_auc, gini, ks_stat, and brier_score.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_prob_arr = np.asarray(y_prob).ravel()

    if len(np.unique(y_true_arr)) < 2:
        auc = 0.5
        gini = 0.0
        ks = 0.0
    else:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        gini = float(2.0 * auc - 1.0)
        ks = ks_statistic(y_true_arr, y_prob_arr)

    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    return {
        "roc_auc": auc,
        "gini": gini,
        "ks_stat": ks,
        "brier_score": brier,
    }
