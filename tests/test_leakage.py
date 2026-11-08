"""Zero-Leakage Automated Tests (ADR 3.1).

Verifies that no post-origination columns or target leakage exist in processed data,
and that the target variable is strictly binary {0, 1} without missing values.
"""

from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pytest

from config.constants import (
    IDENTIFIER_COLS,
    POST_ORIGINATION_COLS,
    PROCESSED_DATA_DIR,
    RAW_TARGET_COL,
    TARGET_COL,
)


@pytest.fixture(scope="module")
def processed_dir() -> Path:
    assert PROCESSED_DATA_DIR.exists(), f"Processed data directory {PROCESSED_DATA_DIR} does not exist"
    return PROCESSED_DATA_DIR


@pytest.mark.parametrize("partition_file", ["train.parquet", "val.parquet", "test.parquet", "stress_test.parquet"])
def test_zero_post_origination_leakage(processed_dir: Path, partition_file: str):
    """Assert that none of the blacklisted post-origination columns exist in processed data."""
    file_path = processed_dir / partition_file
    assert file_path.exists(), f"Processed partition {file_path} is missing!"

    schema = pq.ParquetFile(file_path).schema.names
    schema_cols = set(schema)

    # Check blacklisted post-origination columns
    leaked_post = schema_cols.intersection(set(POST_ORIGINATION_COLS))
    assert not leaked_post, f"Post-origination leakage detected in {partition_file}: {leaked_post}"

    # Check non-predictive identifier columns
    leaked_id = schema_cols.intersection(set(IDENTIFIER_COLS))
    assert not leaked_id, f"Identifier columns detected in {partition_file}: {leaked_id}"

    # Check raw target column is dropped
    assert RAW_TARGET_COL not in schema_cols, (
        f"Raw target column '{RAW_TARGET_COL}' must not be present in {partition_file}"
    )


@pytest.mark.parametrize("partition_file", ["train.parquet", "val.parquet", "test.parquet", "stress_test.parquet"])
def test_clean_binary_target(processed_dir: Path, partition_file: str):
    """Assert target column contains only binary {0, 1} values and 0 nulls."""
    file_path = processed_dir / partition_file
    df_target = pd.read_parquet(file_path, columns=[TARGET_COL])

    assert TARGET_COL in df_target.columns, f"Target column '{TARGET_COL}' missing in {partition_file}"

    # Null check
    null_count = df_target[TARGET_COL].isna().sum()
    assert null_count == 0, f"Found {null_count} nulls in target column of {partition_file}"

    # Binary check {0, 1}
    unique_vals = set(df_target[TARGET_COL].unique())
    assert unique_vals.issubset({0, 1}), f"Unexpected target values in {partition_file}: {unique_vals}"
    assert unique_vals == {0, 1}, f"Target does not contain both classes in {partition_file}: {unique_vals}"
