"""Walk-Forward Split and Out-of-Time Partition Tests (ADR 3.2).

Verifies that temporal partition cutoffs are respected without overlap,
and default rates remain structurally stable (between 10% and 30%).
"""

from pathlib import Path
import pandas as pd
import pytest

from config.constants import PROCESSED_DATA_DIR, TARGET_COL


@pytest.fixture(scope="module")
def processed_dir() -> Path:
    assert PROCESSED_DATA_DIR.exists(), f"Processed data directory {PROCESSED_DATA_DIR} does not exist"
    return PROCESSED_DATA_DIR


def test_train_partition_years(processed_dir: Path):
    """Assert train partition years are strictly <= 2015 (and >= 2013)."""
    train_path = processed_dir / "train.parquet"
    assert train_path.exists(), "train.parquet does not exist!"

    df = pd.read_parquet(train_path, columns=["issue_year"])
    assert (df["issue_year"] <= 2015).all(), "Found loans with issue_year > 2015 in train set"
    assert (df["issue_year"] >= 2013).all(), "Found loans with issue_year < 2013 in train set"


def test_val_partition_years(processed_dir: Path):
    """Assert val partition year is strictly 2016."""
    val_path = processed_dir / "val.parquet"
    assert val_path.exists(), "val.parquet does not exist!"

    df = pd.read_parquet(val_path, columns=["issue_year"])
    assert (df["issue_year"] == 2016).all(), "Val partition contains years other than 2016"


def test_test_partition_years(processed_dir: Path):
    """Assert test partition years are strictly >= 2017 (and <= 2018)."""
    test_path = processed_dir / "test.parquet"
    assert test_path.exists(), "test.parquet does not exist!"

    df = pd.read_parquet(test_path, columns=["issue_year"])
    assert (df["issue_year"] >= 2017).all(), "Found loans with issue_year < 2017 in test set"
    assert (df["issue_year"] <= 2018).all(), "Found loans with issue_year > 2018 in test set"


def test_crisis_stress_test_partition_years(processed_dir: Path):
    """Assert crisis stress test years are strictly between 2007 and 2011."""
    stress_path = processed_dir / "stress_test.parquet"
    assert stress_path.exists(), "stress_test.parquet does not exist!"

    df = pd.read_parquet(stress_path, columns=["issue_year"])
    assert (df["issue_year"] >= 2007).all(), "Found loans with issue_year < 2007 in stress test"
    assert (df["issue_year"] <= 2011).all(), "Found loans with issue_year > 2011 in stress test"


@pytest.mark.parametrize(
    "partition_file, expected_min_rate, expected_max_rate",
    [
        ("train.parquet", 0.10, 0.30),
        ("val.parquet", 0.10, 0.30),
        ("test.parquet", 0.10, 0.30),
        ("stress_test.parquet", 0.10, 0.30),
    ],
)
def test_partition_default_rate_bounds(
    processed_dir: Path,
    partition_file: str,
    expected_min_rate: float,
    expected_max_rate: float,
):
    """Assert default rate is between 10% and 30% across all partitions."""
    file_path = processed_dir / partition_file
    df = pd.read_parquet(file_path, columns=[TARGET_COL])
    default_rate = float(df[TARGET_COL].mean())

    assert expected_min_rate <= default_rate <= expected_max_rate, (
        f"Default rate for {partition_file} is {default_rate:.4f}, "
        f"outside expected bounds [{expected_min_rate}, {expected_max_rate}]"
    )
