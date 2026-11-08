"""Feature Engineering Engine for Credit Risk Decision System.

Implements scikit-learn compatible CreditFeaturePipeline providing strict
train-serving symmetry across batch training and real-time REST API scoring.

Key Architecture Decisions:
- ADR 3.1: Strict Point-in-Time Feature Gate (Zero lookahead post-origination leakage).
- ADR 4.3: Fundamentals vs. Weather (scale-invariant financial discipline ratios).
- ADR 7.1: Hybrid Modular Pipeline (shared feature transformation engine).
"""

import json
import logging
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.constants import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    TARGET_COL,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Valid discrete categories for one-hot encoding
CATEGORICAL_SPEC: Dict[str, List[str]] = {
    "home_ownership": ["MORTGAGE", "RENT", "OWN", "OTHER"],
    "verification_status": ["Source Verified", "Verified", "Not Verified"],
    "purpose": [
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "other",
        "major_purchase",
        "medical",
        "small_business",
        "car",
        "moving",
        "vacation",
        "house",
        "wedding",
        "renewable_energy",
        "educational",
    ],
    "grade": ["A", "B", "C", "D", "E", "F", "G"],
    "sub_grade": [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)],
    "initial_list_status": ["w", "f"],
    "application_type": ["Individual", "Joint App"],
}

# Standard credit bureau and underwriting numeric features
BUREAU_NUMERIC_COLS: List[str] = [
    "loan_amnt",
    "installment",
    "annual_inc",
    "dti",
    "int_rate",
    "delinq_2yrs",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "total_acc",
    "collections_12_mths_ex_med",
    "acc_now_delinq",
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "chargeoff_within_12_mths",
    "delinq_amnt",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "mths_since_recent_bc",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "pub_rec_bankruptcies",
    "tax_liens",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
]

# Numeric features tracked with missingness indicators
MISSING_INDICATOR_COLS: List[str] = [
    "emp_length",
    "dti",
    "revol_util",
    "bc_open_to_buy",
    "bc_util",
    "mo_sin_old_il_acct",
    "mths_since_recent_bc",
    "num_rev_accts",
    "num_tl_120dpd_2m",
    "percent_bc_gt_75",
    "pct_tl_nvr_dlq",
    "avg_cur_bal",
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
]

# Raw columns required from applicant payloads / training frames
RAW_FEATURE_COLUMNS: List[str] = (
    BUREAU_NUMERIC_COLS
    + list(CATEGORICAL_SPEC.keys())
    + [
        "term",
        "emp_length",
        "earliest_cr_line",
        "issue_d",
        "fico_range_low",
        "fico_range_high",
        "inq_last_6mths",
        "mths_since_last_delinq",
        "mths_since_last_record",
        "mths_since_last_major_derog",
        "mths_since_recent_bc_dlq",
        "mths_since_recent_revol_delinq",
        "mths_since_recent_inq",
        "revol_util",
    ]
)


def get_memory_mb() -> float:
    """Return peak resident memory size in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def parse_date_series(
    series: pd.Series,
    default_date: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """Parse string/datetime series gracefully handling '%b-%Y' and fallback formats."""
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        parsed = series.copy()
    else:
        parsed = pd.to_datetime(series, format="%b-%Y", errors="coerce")
        missing_mask = parsed.isna() & series.notna()
        if missing_mask.any():
            fallback = pd.to_datetime(series[missing_mask], format="mixed", errors="coerce")
            parsed = parsed.fillna(fallback)

    parsed = parsed.astype("datetime64[us]")
    if default_date is not None:
        parsed = parsed.fillna(pd.Timestamp(default_date))
    return parsed



def parse_revol_util(series: pd.Series) -> pd.Series:
    """Parse percentage strings (e.g. '45.2%') or floats to float values."""
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.astype(float)
    clean = (
        series.astype(str)
        .str.rstrip("%")
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )
    return pd.to_numeric(clean, errors="coerce")


def parse_term(series: pd.Series) -> pd.Series:
    """Parse loan term string ('36 months', '60 months') to numeric months."""
    term_map = {
        " 36 months": 36.0,
        "36 months": 36.0,
        " 60 months": 60.0,
        "60 months": 60.0,
        36: 36.0,
        60: 60.0,
        "36": 36.0,
        "60": 60.0,
    }
    mapped = series.map(term_map)
    if mapped.isna().any():
        extracted = series.astype(str).str.extract(r"(\d+)")[0]
        mapped = mapped.fillna(pd.to_numeric(extracted, errors="coerce"))
    return mapped.astype(float)


def parse_emp_length(series: pd.Series) -> pd.Series:
    """Parse employment length string to numeric years (0 to 10)."""
    emp_map = {
        "10+ years": 10.0,
        "9 years": 9.0,
        "8 years": 8.0,
        "7 years": 7.0,
        "6 years": 6.0,
        "5 years": 5.0,
        "4 years": 4.0,
        "3 years": 3.0,
        "2 years": 2.0,
        "1 year": 1.0,
        "< 1 year": 0.0,
    }
    return series.map(emp_map).astype(float)


class CreditFeaturePipeline(BaseEstimator, TransformerMixin):
    """Production-grade Credit Risk Feature Engineering Transformer.

    Engineers scale-invariant credit ratios, parses temporal and percentage fields,
    imputes missing numeric values with medians learned strictly during training,
    and one-hot encodes categoricals with unknown category tolerance.
    """


    def __init__(self, clip_outliers: bool = True):
        self.clip_outliers = clip_outliers
        self.numeric_medians_: Dict[str, float] = {}
        self.numeric_cols_: List[str] = []
        self.indicator_cols_: List[str] = []
        self.feature_names_: List[str] = []
        self.ohe_: Optional[OneHotEncoder] = None
        self.ohe_feature_names_: List[str] = []

    def _prepare_dataframe(
        self,
        X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """Convert input payload to DataFrame and align expected columns."""
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(
                f"Expected DataFrame, dict, or list of dicts, got {type(X)}"
            )

        # Ensure all required raw columns exist, filling missing with NaN
        for col in RAW_FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        return df

    def _engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute scale-invariant ratios, indicators, and clean categoricals.

        Returns:
            Tuple of (numeric_and_indicators_df, cleaned_categoricals_df)
        """
        num_dict: Dict[str, pd.Series] = {}

        # 1. payment_to_income_ratio: installment / (annual_inc / 12.0 + 1e-5)
        annual_inc = pd.to_numeric(df["annual_inc"], errors="coerce").fillna(0.0)
        installment = pd.to_numeric(df["installment"], errors="coerce").fillna(0.0)
        monthly_inc = np.maximum(annual_inc, 0.0) / 12.0 + 1e-5
        pti_ratio = installment / monthly_inc
        if self.clip_outliers:
            pti_ratio = np.clip(pti_ratio, 0.0, 10.0)
        num_dict["payment_to_income_ratio"] = pti_ratio

        # 2. credit_history_age_years: difference between issue_d and earliest_cr_line
        ref_date = pd.Timestamp.now()
        issue_dt = parse_date_series(df["issue_d"], default_date=ref_date)
        earliest_dt = parse_date_series(df["earliest_cr_line"], default_date=None)
        cr_age_days = (issue_dt - earliest_dt).dt.days
        cr_age_years = cr_age_days / 365.25
        if self.clip_outliers:
            cr_age_years = np.maximum(cr_age_years, 0.0)
        num_dict["credit_history_age_years"] = cr_age_years

        # 3. account_breadth: open_acc / (total_acc + 1e-5)
        open_acc = pd.to_numeric(df["open_acc"], errors="coerce")
        total_acc = pd.to_numeric(df["total_acc"], errors="coerce")
        breadth = open_acc / (total_acc + 1e-5)
        if self.clip_outliers:
            breadth = np.clip(breadth, 0.0, 1.0)
        num_dict["account_breadth"] = breadth

        # 4. fico_score: average of range low and high (or direct fico_score if present)
        if "fico_score" in df.columns and df["fico_score"].notna().any():
            fico_direct = pd.to_numeric(df["fico_score"], errors="coerce")
        else:
            fico_direct = pd.Series(np.nan, index=df.index)

        fico_low = pd.to_numeric(df["fico_range_low"], errors="coerce")
        fico_high = pd.to_numeric(df["fico_range_high"], errors="coerce")
        fico_calc = (fico_low + fico_high) / 2.0
        num_dict["fico_score"] = fico_direct.fillna(fico_calc)

        # 5. credit_hunger_velocity: inq_last_6mths
        num_dict["credit_hunger_velocity"] = pd.to_numeric(
            df["inq_last_6mths"], errors="coerce"
        )

        # 6. delinq_recency_imputed: missing filled as sentinel 999.0
        mths_delinq = pd.to_numeric(df["mths_since_last_delinq"], errors="coerce")
        num_dict["delinq_recency_imputed"] = mths_delinq.fillna(999.0)

        # 7. has_delinq_history: binary indicator
        num_dict["has_delinq_history"] = mths_delinq.notna().astype(float)

        # 8. revol_util_clean: parsed percentage float
        num_dict["revol_util_clean"] = parse_revol_util(df["revol_util"])

        # 9. Cleaned term & emp_length
        num_dict["term_months"] = parse_term(df["term"])
        num_dict["emp_length_years"] = parse_emp_length(df["emp_length"])

        # 10. Derogatory recency fields (sentinel 999.0 + binary indicator flags)
        derog_pairs = [
            ("mths_since_last_record", "has_pub_record_history"),
            ("mths_since_last_major_derog", "has_major_derog_history"),
            ("mths_since_recent_bc_dlq", "has_recent_bc_dlq"),
            ("mths_since_recent_revol_delinq", "has_recent_revol_delinq"),
            ("mths_since_recent_inq", "has_recent_inq"),
        ]
        for recency_col, flag_col in derog_pairs:
            vals = pd.to_numeric(df[recency_col], errors="coerce")
            num_dict[recency_col] = vals.fillna(999.0)
            num_dict[flag_col] = vals.notna().astype(float)

        # 11. Standard bureau numeric features
        for col in BUREAU_NUMERIC_COLS:
            num_dict[col] = pd.to_numeric(df[col], errors="coerce")

        # 12. Missingness indicator flags for features with notable null rates
        for col in MISSING_INDICATOR_COLS:
            flag_name = f"{col}_isna"
            raw_vals = df[col]
            if pd.api.types.is_numeric_dtype(raw_vals.dtype):
                num_dict[flag_name] = raw_vals.isna().astype(float)
            else:
                num_dict[flag_name] = (
                    raw_vals.isna() | raw_vals.isin(["", "nan", "None"])
                ).astype(float)

        num_df = pd.DataFrame(num_dict, index=df.index)

        # 13. Clean categorical features
        cat_dict: Dict[str, pd.Series] = {}

        # home_ownership: MORTGAGE, RENT, OWN, OTHER
        ho_raw = df["home_ownership"].astype(str).str.upper().str.strip()
        cat_dict["home_ownership"] = ho_raw.where(
            ho_raw.isin({"MORTGAGE", "RENT", "OWN"}), "OTHER"
        )

        # verification_status: Source Verified, Verified, Not Verified
        vs_raw = df["verification_status"].astype(str).str.strip()
        cat_dict["verification_status"] = vs_raw.where(
            vs_raw.isin({"Source Verified", "Verified", "Not Verified"}), "Not Verified"
        )

        # purpose: 14 Lending Club purposes
        purp_raw = df["purpose"].astype(str).str.lower().str.strip()
        cat_dict["purpose"] = purp_raw.where(
            purp_raw.isin(set(CATEGORICAL_SPEC["purpose"])), "other"
        )

        # grade & sub_grade: Preserved for Option Z ablation benchmarks
        grade_raw = df["grade"].astype(str).str.upper().str.strip()
        cat_dict["grade"] = grade_raw.where(
            grade_raw.isin(set(CATEGORICAL_SPEC["grade"])), "C"
        )

        sg_raw = df["sub_grade"].astype(str).str.upper().str.strip()
        cat_dict["sub_grade"] = sg_raw.where(
            sg_raw.isin(set(CATEGORICAL_SPEC["sub_grade"])), "C1"
        )

        # initial_list_status: w, f
        init_raw = df["initial_list_status"].astype(str).str.lower().str.strip()
        cat_dict["initial_list_status"] = init_raw.where(
            init_raw.isin({"w", "f"}), "f"
        )

        # application_type: Individual, Joint App
        app_raw = df["application_type"].astype(str).str.strip()
        cat_dict["application_type"] = app_raw.where(
            app_raw.isin({"Individual", "Joint App"}), "Individual"
        )

        cat_df = pd.DataFrame(cat_dict, index=df.index)

        return num_df, cat_df

    def fit(self, X: pd.DataFrame, y=None) -> "CreditFeaturePipeline":
        """Fit median imputers and OneHotEncoder strictly on training data."""
        logger.info("Fitting CreditFeaturePipeline on training data...")
        start_t = time.time()

        df = self._prepare_dataframe(X)
        num_df, cat_df = self._engineer_features(df)

        self.numeric_cols_ = list(num_df.columns)

        # Fit median imputation dictionary strictly on training distribution
        medians = num_df.median()
        self.numeric_medians_ = medians.fillna(0.0).to_dict()

        # Build and fit OneHotEncoder with deterministic category lists
        cat_cols = list(CATEGORICAL_SPEC.keys())
        cat_categories = [CATEGORICAL_SPEC[c] for c in cat_cols]

        self.ohe_ = OneHotEncoder(
            categories=cat_categories,
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        self.ohe_.fit(cat_df[cat_cols])
        self.ohe_feature_names_ = list(self.ohe_.get_feature_names_out(cat_cols))

        # Master feature column order
        self.feature_names_ = self.numeric_cols_ + self.ohe_feature_names_

        logger.info(
            "Fit completed in %.2fs | Total output features: %d (%d numeric, %d OHE)",
            time.time() - start_t,
            len(self.feature_names_),
            len(self.numeric_cols_),
            len(self.ohe_feature_names_),
        )
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        return_df: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Transform applicant features with learned medians and one-hot encoding.

        Args:
            X: Input DataFrame, single-row dictionary, or list of dictionaries.
            return_df: If True, returns a pandas DataFrame; otherwise np.ndarray.

        Returns:
            2D numpy array (or DataFrame) with shape (n_samples, n_features).
        """
        if not self.numeric_medians_ or self.ohe_ is None:
            raise RuntimeError("Pipeline must be fitted before calling transform()")

        df = self._prepare_dataframe(X)
        num_df, cat_df = self._engineer_features(df)

        # Align numeric columns with learned feature set and impute NaNs
        aligned_num = pd.DataFrame(index=df.index)
        for col in self.numeric_cols_:
            if col in num_df.columns:
                series = num_df[col]
            else:
                series = pd.Series(np.nan, index=df.index)

            median_val = self.numeric_medians_.get(col, 0.0)
            aligned_num[col] = series.fillna(median_val).astype(np.float32)

        # One-hot encode categoricals
        cat_cols = list(CATEGORICAL_SPEC.keys())
        ohe_matrix = self.ohe_.transform(cat_df[cat_cols])

        # Combine into contiguous 2D array
        num_matrix = aligned_num.to_numpy(dtype=np.float32)
        combined = np.hstack([num_matrix, ohe_matrix]).astype(np.float32)

        # Ensure zero NaNs or Infs
        if np.isnan(combined).any() or np.isinf(combined).any():
            combined = np.nan_to_num(combined, nan=0.0, posinf=1e6, neginf=-1e6)

        if return_df:
            return pd.DataFrame(
                combined,
                columns=self.feature_names_,
                index=df.index,
            )
        return combined

    def transform_df(
        self,
        X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """Convenience method returning a DataFrame with named feature columns."""
        return self.transform(X, return_df=True)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return array of output feature names for scikit-learn compliance."""
        return np.array(self.feature_names_)


def build_and_save_features(
    data_dir: Path = PROCESSED_DATA_DIR,
    model_dir: Path = MODEL_DIR,
) -> CreditFeaturePipeline:
    """Fit feature pipeline on train partition and transform all datasets."""
    total_start = time.time()
    logger.info("=" * 70)
    logger.info("Executing Phase 3: Scale-Invariant Feature Engineering Pipeline")
    logger.info("Input directory: %s", data_dir)
    logger.info("Output model directory: %s", model_dir)
    logger.info("Initial memory: %.1f MB", get_memory_mb())
    logger.info("=" * 70)

    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load train partition
    train_path = data_dir / "train.parquet"
    logger.info("Loading training partition from %s...", train_path)
    train_df = pd.read_parquet(train_path)
    logger.info(
        "Loaded train partition: %d rows x %d cols | Peak RSS: %.1f MB",
        len(train_df),
        len(train_df.columns),
        get_memory_mb(),
    )

    # 2. Fit pipeline strictly on train partition
    pipeline = CreditFeaturePipeline()
    pipeline.fit(train_df)

    # 3. Save feature pipeline and feature names
    pipeline_path = model_dir / "feature_pipeline.joblib"
    logger.info("Serializing feature pipeline to %s...", pipeline_path)
    joblib.dump(pipeline, pipeline_path, compress=3)

    feature_names_path = model_dir / "feature_names.json"
    logger.info("Writing feature names to %s...", feature_names_path)
    with open(feature_names_path, "w") as f:
        json.dump(pipeline.feature_names_, f, indent=2)

    # 4. Transform and write all 4 partitions
    partitions = [
        ("train", train_df, "X_train.parquet", "y_train.parquet"),
        ("val", None, "X_val.parquet", "y_val.parquet"),
        ("test", None, "X_test.parquet", "y_test.parquet"),
        ("stress", None, "X_stress.parquet", "y_stress.parquet"),
    ]

    for name, df_obj, x_filename, y_filename in partitions:
        t0 = time.time()
        logger.info("-" * 50)
        source_name = f"{name}.parquet" if name != "stress" else "stress_test.parquet"

        if df_obj is None:
            source_path = data_dir / source_name
            logger.info("Loading partition [%s] from %s...", name, source_path)
            df_current = pd.read_parquet(source_path)
        else:
            df_current = df_obj

        logger.info(
            "Transforming partition [%s]: %d rows...", name, len(df_current)
        )
        x_transformed = pipeline.transform(df_current, return_df=True)
        y_target = pd.DataFrame(
            {TARGET_COL: df_current[TARGET_COL].astype(np.int8)},
            index=df_current.index,
        )

        # Integrity checks
        nan_count = int(x_transformed.isna().sum().sum())
        inf_count = int(np.isinf(x_transformed.to_numpy()).sum())
        target_unique = set(y_target[TARGET_COL].unique())

        logger.info(
            "  Transformed shape: %s | Target shape: %s",
            x_transformed.shape,
            y_target.shape,
        )
        logger.info("  Zero NaNs check: %d | Zero Infs check: %d", nan_count, inf_count)
        logger.info("  Target classes: %s", target_unique)
        assert nan_count == 0, f"Found {nan_count} NaNs in {x_filename}"
        assert inf_count == 0, f"Found {inf_count} Infs in {x_filename}"
        assert target_unique.issubset({0, 1}), f"Invalid target classes in {y_filename}"

        # Write Parquet artifacts
        x_out = data_dir / x_filename
        y_out = data_dir / y_filename

        x_transformed.to_parquet(
            x_out, engine="pyarrow", compression="snappy", index=False
        )
        y_target.to_parquet(
            y_out, engine="pyarrow", compression="snappy", index=False
        )

        x_size_mb = x_out.stat().st_size / (1024 * 1024)
        y_size_mb = y_out.stat().st_size / (1024 * 1024)
        logger.info(
            "  Saved %s (%.1f MB) & %s (%.1f MB) in %.2fs",
            x_filename,
            x_size_mb,
            y_filename,
            y_size_mb,
            time.time() - t0,
        )

        del df_current, x_transformed, y_target

    logger.info("=" * 70)
    logger.info(
        "Phase 3 Feature Engineering Engine completed successfully in %.1fs",
        time.time() - total_start,
    )
    logger.info("Final memory: %.1f MB", get_memory_mb())
    logger.info("=" * 70)
    return pipeline


if __name__ == "__main__":
    import src.features.build_features as bfeat
    bfeat.build_and_save_features()

