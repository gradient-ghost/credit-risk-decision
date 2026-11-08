"""Feature Engineering and Symmetry Tests (Phase 3).

Verifies that the serialized pipeline loads cleanly, transformed feature matrices
contain zero NaNs and zero Infs, row counts strictly match source partitions,
and single-row applicant inference dictionary executes symmetrically with zero skew.
"""

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import pytest

from config.constants import MODEL_DIR, PROCESSED_DATA_DIR, TARGET_COL
from src.features.build_features import CreditFeaturePipeline


@pytest.fixture(scope="module")
def pipeline_path() -> Path:
    path = MODEL_DIR / "feature_pipeline.joblib"
    assert path.exists(), f"feature_pipeline.joblib not found at {path}"
    return path


@pytest.fixture(scope="module")
def feature_names_path() -> Path:
    path = MODEL_DIR / "feature_names.json"
    assert path.exists(), f"feature_names.json not found at {path}"
    return path


@pytest.fixture(scope="module")
def feature_pipeline(pipeline_path: Path) -> CreditFeaturePipeline:
    pipeline = joblib.load(pipeline_path)
    assert isinstance(pipeline, CreditFeaturePipeline)
    return pipeline


@pytest.fixture(scope="module")
def feature_names(feature_names_path: Path) -> list:
    with open(feature_names_path, "r") as f:
        names = json.load(f)
    assert isinstance(names, list)
    assert len(names) > 0
    assert all(isinstance(n, str) for n in names)
    return names


def test_pipeline_serialization_and_metadata(
    feature_pipeline: CreditFeaturePipeline,
    feature_names: list,
):
    """Assert pipeline loaded cleanly and feature count matches metadata."""
    assert hasattr(feature_pipeline, "numeric_medians_")
    assert len(feature_pipeline.numeric_medians_) > 0
    assert hasattr(feature_pipeline, "feature_names_")
    assert len(feature_pipeline.feature_names_) == len(feature_names)
    assert feature_pipeline.feature_names_ == feature_names


@pytest.mark.parametrize(
    "x_file, y_file, expected_rows",
    [
        ("X_train.parquet", "y_train.parquet", 733453),
        ("X_val.parquet", "y_val.parquet", 293105),
        ("X_test.parquet", "y_test.parquet", 225639),
        ("X_stress.parquet", "y_stress.parquet", 39786),
    ],
)
def test_transformed_matrix_shapes_and_integrity(
    x_file: str,
    y_file: str,
    expected_rows: int,
    feature_names: list,
):
    """Verify transformed feature matrices and targets across all partitions."""
    x_path = PROCESSED_DATA_DIR / x_file
    y_path = PROCESSED_DATA_DIR / y_file

    assert x_path.exists(), f"{x_file} missing in {PROCESSED_DATA_DIR}"
    assert y_path.exists(), f"{y_file} missing in {PROCESSED_DATA_DIR}"

    # Verify X matrix
    df_x = pd.read_parquet(x_path)
    assert len(df_x) == expected_rows, f"{x_file} row count mismatch: {len(df_x)} vs {expected_rows}"
    assert len(df_x.columns) == len(feature_names), f"{x_file} column count mismatch"
    assert list(df_x.columns) == feature_names, f"{x_file} column names differ from feature_names.json"

    # Strict Zero NaNs and Zero Infs
    x_arr = df_x.to_numpy()
    nan_count = int(np.isnan(x_arr).sum())
    inf_count = int(np.isinf(x_arr).sum())
    assert nan_count == 0, f"Found {nan_count} NaNs in {x_file}"
    assert inf_count == 0, f"Found {inf_count} Infs in {x_file}"

    # Verify y target
    df_y = pd.read_parquet(y_path)
    assert len(df_y) == expected_rows, f"{y_file} row count mismatch: {len(df_y)} vs {expected_rows}"
    assert TARGET_COL in df_y.columns, f"Target column '{TARGET_COL}' missing in {y_file}"

    unique_targets = set(df_y[TARGET_COL].unique())
    assert unique_targets.issubset({0, 1}), f"Non-binary targets in {y_file}: {unique_targets}"
    assert df_y[TARGET_COL].isna().sum() == 0, f"Null values found in target {y_file}"


def test_inference_symmetry_full_applicant(
    feature_pipeline: CreditFeaturePipeline,
    feature_names: list,
):
    """Inference Symmetry Test: Full single-row applicant dictionary."""
    applicant: Dict[str, Any] = {
        "loan_amnt": 15000.0,
        "term": "36 months",
        "int_rate": 11.5,
        "installment": 450.0,
        "grade": "B",
        "sub_grade": "B2",
        "emp_length": "10+ years",
        "home_ownership": "MORTGAGE",
        "annual_inc": 75000.0,
        "verification_status": "Source Verified",
        "purpose": "debt_consolidation",
        "dti": 18.5,
        "delinq_2yrs": 0.0,
        "earliest_cr_line": "Jan-2005",
        "fico_range_low": 700.0,
        "fico_range_high": 704.0,
        "inq_last_6mths": 0.0,
        "mths_since_last_delinq": None,
        "open_acc": 10.0,
        "pub_rec": 0.0,
        "revol_bal": 12000.0,
        "revol_util": "35.5%",
        "total_acc": 22.0,
        "tot_cur_bal": 180000.0,
        "total_rev_hi_lim": 35000.0,
        "mort_acc": 2.0,
    }

    # Transform applicant dict
    output = feature_pipeline.transform(applicant)

    # Asserts output is 2D array matching exact feature count expected by models
    assert isinstance(output, np.ndarray), f"Expected np.ndarray, got {type(output)}"
    assert output.ndim == 2, f"Expected 2D array, got {output.ndim}D array"
    assert output.shape[0] == 1, f"Expected 1 row, got {output.shape[0]}"
    assert output.shape[1] == len(feature_names), (
        f"Feature dimension mismatch: {output.shape[1]} vs {len(feature_names)}"
    )

    # Check finite numbers
    assert not np.isnan(output).any(), "Found NaNs in inference output"
    assert not np.isinf(output).any(), "Found Infs in inference output"


def test_inference_symmetry_minimal_applicant(
    feature_pipeline: CreditFeaturePipeline,
    feature_names: list,
):
    """Inference Symmetry Test: Minimal applicant payload with missing optional fields."""
    applicant_minimal: Dict[str, Any] = {
        "loan_amnt": 10000.0,
        "annual_inc": 50000.0,
        "installment": 310.0,
        "fico_score": 680.0,
        "term": "36 months",
        "purpose": "credit_card",
    }

    output = feature_pipeline.transform(applicant_minimal)

    assert isinstance(output, np.ndarray)
    assert output.ndim == 2
    assert output.shape == (1, len(feature_names))
    assert not np.isnan(output).any(), "Minimal applicant produced NaNs"
    assert not np.isinf(output).any(), "Minimal applicant produced Infs"


def test_inference_dataframe_transform(
    feature_pipeline: CreditFeaturePipeline,
    feature_names: list,
):
    """Verify transform_df returns a DataFrame with identical schema and feature names."""
    sample = {
        "loan_amnt": 20000.0,
        "annual_inc": 95000.0,
        "installment": 620.0,
        "fico_range_low": 720.0,
        "fico_range_high": 724.0,
        "term": "60 months",
        "home_ownership": "OWN",
        "purpose": "home_improvement",
    }

    df_out = feature_pipeline.transform_df(sample)
    assert isinstance(df_out, pd.DataFrame)
    assert df_out.shape == (1, len(feature_names))
    assert list(df_out.columns) == feature_names
    assert df_out.isna().sum().sum() == 0
