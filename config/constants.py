"""Global constants, file paths, and column definitions for Credit Risk Decision System."""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "accepted_2007_to_2018Q4.csv.gz"
REJECTED_DATA_PATH = DATA_DIR / "raw" / "rejected_2007_to_2018Q4.csv.gz"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Reproducibility
RANDOM_SEED = 42

# Target Variable Definition
RAW_TARGET_COL = "loan_status"
TARGET_COL = "target"  # 1 = Default / Charge-off, 0 = Fully Paid

# Target mapping for Lending Club raw loan_status
TARGET_MAP = {
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
}

# Post-origination / Lookahead Feature Blacklist (Zero Leakage Gate - ADR 3.1)
# Any feature generated or updated after origination MUST be dropped.
POST_ORIGINATION_COLS = [
    # Explicitly flagged in ADR 3.1
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "out_prncp",
    "out_prncp_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    # Additional post-origination payment & status tracking fields
    "total_pymnt",
    "total_pymnt_inv",
    "last_credit_pull_d",
    # Debt settlement fields (post-charge-off / workout)
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
    # Hardship program fields
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "deferral_term",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "payment_plan_start_date",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",
]

# Non-predictive and high-cardinality identifier columns
IDENTIFIER_COLS = [
    "id",
    "member_id",
    "url",
    "desc",
]

# Business & Cost Accounting Constants (ADR 2.2)
EXPOSURE_COL = "loan_amnt"
LGD_DEFAULT = 1.0  # Loss Given Default (100% loss upon default for unsecured consumer loans)

# Decision Policy Defaults (ADR 1.4 & 4.3)
DEFAULT_TAU_LOW = 0.10   # Probability < tau_low -> Auto-Approve
DEFAULT_TAU_HIGH = 0.25  # Probability > tau_high -> Auto-Reject; between -> Manual Review
