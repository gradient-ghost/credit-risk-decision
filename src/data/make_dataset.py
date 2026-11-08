"""Ingestion and Preprocessing Pipeline for Credit Risk Decision System.

Filters accepted Lending Club loans to concluded loans, applies zero-leakage gate
by stripping post-origination columns, and partitions chronologically into
walk-forward train, val, test, and crisis stress-test Parquet datasets.
"""

import argparse
import gc
import logging
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

# Allow absolute imports relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.constants import (
    IDENTIFIER_COLS,
    POST_ORIGINATION_COLS,
    PROCESSED_DATA_DIR,
    RAW_DATA_PATH,
    RAW_TARGET_COL,
    TARGET_COL,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Terminal status definitions (Zero-Leakage & Concluded Loans Only)
SAFE_STATUSES = ["Fully Paid"]
DEFAULT_STATUSES = ["Charged Off", "Default"]
TERMINAL_STATUSES = set(SAFE_STATUSES + DEFAULT_STATUSES)


def get_memory_mb() -> float:
    """Return the peak resident set size (RSS) in MB for this process."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_split_config(config_path: Optional[Path] = None) -> dict:
    """Load temporal splitting configuration from model_config.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "model_config.yaml"

    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("splits", {})
    return {}


def process_chunk(
    chunk: pd.DataFrame,
    cols_to_drop: set,
) -> Optional[pd.DataFrame]:
    """Filter non-terminal loans, map target, parse date, and drop leakage columns."""
    # Filter strictly to terminal concluded loans
    mask = chunk[RAW_TARGET_COL].isin(TERMINAL_STATUSES)
    filtered = chunk[mask].copy()

    if filtered.empty:
        return None

    # Target Mapping: 0 = Safe (Fully Paid), 1 = Default (Charged Off / Default)
    filtered[TARGET_COL] = (filtered[RAW_TARGET_COL] != "Fully Paid").astype("int8")

    # Date parsing & Issue Year extraction
    issue_dt = pd.to_datetime(filtered["issue_d"], format="%b-%Y")
    filtered["issue_d"] = issue_dt
    filtered["issue_year"] = issue_dt.dt.year.astype("int16")

    # Drop blacklisted post-origination columns, target raw status, and identifiers
    existing_drop_cols = [c for c in cols_to_drop if c in filtered.columns]
    filtered = filtered.drop(columns=existing_drop_cols)

    return filtered


def make_dataset(
    raw_path: Path = RAW_DATA_PATH,
    output_dir: Path = PROCESSED_DATA_DIR,
    chunksize: int = 100000,
    config_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Process raw Lending Club dataset in chunks and write parquet partitions.

    Args:
        raw_path: Path to accepted loans gzip CSV.
        output_dir: Target directory for parquet files.
        chunksize: Number of rows per streaming CSV chunk.
        config_path: Optional path to model_config.yaml.

    Returns:
        Dictionary mapping partition name to written file path.
    """
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("Starting Credit Risk Ingestion & Preprocessing Pipeline")
    logger.info("Raw input path: %s", raw_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("Chunksize: %d rows", chunksize)
    logger.info("Initial memory: %.1f MB", get_memory_mb())
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load split boundaries from config or use defaults
    split_cfg = load_split_config(config_path)
    train_start = split_cfg.get("train", {}).get("start_year", 2013)
    train_end = split_cfg.get("train", {}).get("end_year", 2015)
    val_year = split_cfg.get("validation", {}).get("year", 2016)
    test_start = split_cfg.get("test", {}).get("start_year", 2017)
    test_end = split_cfg.get("test", {}).get("end_year", 2018)
    stress_start, stress_end = 2007, 2011

    logger.info(
        "Temporal Split Boundaries: Train=[%d-%d], Val=[%d], Test=[%d-%d], Crisis Stress=[%d-%d]",
        train_start,
        train_end,
        val_year,
        test_start,
        test_end,
        stress_start,
        stress_end,
    )

    # Columns to drop: 35 post-origination columns + 4 identifiers + 1 raw target
    cols_to_drop = set(POST_ORIGINATION_COLS) | set(IDENTIFIER_COLS) | {RAW_TARGET_COL}
    logger.info("Total blacklisted / non-predictive columns to drop: %d", len(cols_to_drop))

    # Buckets to hold partitioned chunks
    train_chunks: List[pd.DataFrame] = []
    val_chunks: List[pd.DataFrame] = []
    test_chunks: List[pd.DataFrame] = []
    stress_chunks: List[pd.DataFrame] = []

    total_raw_rows = 0
    total_concluded_rows = 0

    reader = pd.read_csv(raw_path, chunksize=chunksize, low_memory=False)

    for chunk_idx, chunk in enumerate(reader):
        raw_count = len(chunk)
        total_raw_rows += raw_count

        processed = process_chunk(chunk, cols_to_drop)
        if processed is None or processed.empty:
            continue

        concluded_count = len(processed)
        total_concluded_rows += concluded_count

        years = processed["issue_year"]

        train_part = processed[(years >= train_start) & (years <= train_end)]
        val_part = processed[years == val_year]
        test_part = processed[(years >= test_start) & (years <= test_end)]
        stress_part = processed[(years >= stress_start) & (years <= stress_end)]

        if not train_part.empty:
            train_chunks.append(train_part)
        if not val_part.empty:
            val_chunks.append(val_part)
        if not test_part.empty:
            test_chunks.append(test_part)
        if not stress_part.empty:
            stress_chunks.append(stress_part)

        if (chunk_idx + 1) % 5 == 0 or (chunk_idx == 0):
            logger.info(
                "Chunk %3d processed | Cumulative raw: %9d | Concluded: %9d | Peak RSS: %.1f MB",
                chunk_idx + 1,
                total_raw_rows,
                total_concluded_rows,
                get_memory_mb(),
            )

    logger.info(
        "Finished CSV reading. Total raw rows: %d | Total concluded rows: %d",
        total_raw_rows,
        total_concluded_rows,
    )

    partitions = [
        ("train", train_chunks, "train.parquet"),
        ("val", val_chunks, "val.parquet"),
        ("test", test_chunks, "test.parquet"),
        ("crisis_stress_test", stress_chunks, "stress_test.parquet"),
    ]

    saved_paths: Dict[str, Path] = {}

    for name, chunk_list, filename in partitions:
        logger.info("-" * 50)
        logger.info("Assembling and writing partition: %s -> %s", name, filename)

        if not chunk_list:
            logger.warning("Partition %s is empty!", name)
            continue

        df_partition = pd.concat(chunk_list, ignore_index=True)
        # Clear list reference and force garbage collection
        chunk_list.clear()
        gc.collect()

        row_count = len(df_partition)
        col_count = len(df_partition.columns)
        default_count = int(df_partition[TARGET_COL].sum())
        default_rate = (default_count / row_count) * 100.0 if row_count > 0 else 0.0
        min_yr = int(df_partition["issue_year"].min())
        max_yr = int(df_partition["issue_year"].max())
        mem_mb = df_partition.memory_usage(deep=True).sum() / (1024 * 1024)

        logger.info("  Partition: %s", name)
        logger.info("  Rows: %d | Columns: %d", row_count, col_count)
        logger.info("  Issue Years: %d to %d", min_yr, max_yr)
        logger.info(
            "  Default Count: %d / %d (Default Rate: %.2f%%)",
            default_count,
            row_count,
            default_rate,
        )
        logger.info("  DataFrame In-Memory Size: %.2f MB", mem_mb)

        target_file = output_dir / filename
        df_partition.to_parquet(
            target_file,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        file_size_mb = target_file.stat().st_size / (1024 * 1024)
        logger.info("  Written to %s (File size: %.2f MB)", target_file, file_size_mb)
        saved_paths[name] = target_file

        del df_partition
        gc.collect()

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("Pipeline Complete in %.2f seconds (%.2f minutes)", elapsed, elapsed / 60.0)
    logger.info("Peak Memory Usage: %.1f MB", get_memory_mb())
    logger.info("=" * 70)

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Risk Ingestion & Preprocessing Pipeline")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to raw accepted loans CSV gzip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Target directory for processed parquet partitions",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100000,
        help="Streaming CSV chunk size",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=PROJECT_ROOT / "config" / "model_config.yaml",
        help="Path to model_config.yaml",
    )

    args = parser.parse_args()
    make_dataset(
        raw_path=args.raw_path,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        config_path=args.config_path,
    )
