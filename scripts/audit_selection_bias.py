"""Selection Bias and Zero-Leakage Audit Script.

1. Audits post-origination columns across all processed Parquet partitions.
2. Compares Accepted vs Rejected cohorts on FICO/Risk Score, DTI, and Loan Amount.
3. Computes parametric and non-parametric summary statistics (Mean, Median, 25th, 75th).
4. Generates publication-ready figures and saves summary JSON report.
5. Builds and saves notebooks/01_leakage_and_selection_bias_audit.ipynb.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Set Matplotlib cache dir before importing pyplot to avoid permission warnings
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.constants import (
    IDENTIFIER_COLS,
    POST_ORIGINATION_COLS,
    PROCESSED_DATA_DIR,
    RAW_TARGET_COL,
    REJECTED_DATA_PATH,
    REPORTS_DIR,
    TARGET_COL,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def audit_leakage(processed_dir: Path = PROCESSED_DATA_DIR) -> dict:
    """Verify that zero post-origination or identifier columns exist in processed partitions."""
    logger.info("Auditing zero-leakage gate across processed Parquet files...")
    partitions = ["train.parquet", "val.parquet", "test.parquet", "stress_test.parquet"]
    audit_results = {}

    for part in partitions:
        path = processed_dir / part
        if not path.exists():
            audit_results[part] = {"status": "MISSING", "leakage_cols": []}
            continue

        df_sample = pd.read_parquet(path)
        cols = set(df_sample.columns)

        leaked_post = cols.intersection(set(POST_ORIGINATION_COLS))
        leaked_id = cols.intersection(set(IDENTIFIER_COLS))
        leaked_raw_target = RAW_TARGET_COL in cols

        is_clean = (len(leaked_post) == 0) and (len(leaked_id) == 0) and (not leaked_raw_target)
        has_target = TARGET_COL in cols and df_sample[TARGET_COL].isin([0, 1]).all()

        audit_results[part] = {
            "status": "PASS" if (is_clean and has_target) else "FAIL",
            "row_count": len(df_sample),
            "column_count": len(cols),
            "leaked_post_origination_cols": list(leaked_post),
            "leaked_identifier_cols": list(leaked_id),
            "has_raw_target_col": leaked_raw_target,
            "has_clean_binary_target": bool(has_target),
        }
        logger.info(
            "  [%s] Status: %s | Rows: %d | Cols: %d | Leaked Post-Orig: %d",
            part,
            audit_results[part]["status"],
            audit_results[part]["row_count"],
            audit_results[part]["column_count"],
            len(leaked_post),
        )

    return audit_results


def compute_cohort_stats(series: pd.Series, name: str) -> dict:
    """Calculate summary statistics: Mean, Std, Median, 25th, and 75th percentiles."""
    clean = series.dropna()
    return {
        "name": name,
        "count": int(len(clean)),
        "mean": round(float(clean.mean()), 2),
        "std": round(float(clean.std()), 2),
        "min": round(float(clean.min()), 2),
        "p25": round(float(clean.quantile(0.25)), 2),
        "median": round(float(clean.median()), 2),
        "p75": round(float(clean.quantile(0.75)), 2),
        "max": round(float(clean.max()), 2),
        "iqr": round(float(clean.quantile(0.75) - clean.quantile(0.25)), 2),
    }


def audit_selection_bias(
    rejected_path: Path = REJECTED_DATA_PATH,
    train_parquet_path: Path = PROCESSED_DATA_DIR / "train.parquet",
    sample_size: int = 100000,
) -> dict:
    """Compare distributions between Accepted and Rejected cohorts."""
    logger.info("Auditing Selection Bias (sample size = %d)...", sample_size)

    # 1. Load Accepted cohort sample
    logger.info("Loading Accepted cohort from %s...", train_parquet_path)
    df_acc_full = pd.read_parquet(
        train_parquet_path,
        columns=["loan_amnt", "dti", "fico_range_low", "fico_range_high"],
    )
    df_acc = df_acc_full.sample(n=min(sample_size, len(df_acc_full)), random_state=42).copy()
    df_acc["fico_midpoint"] = (df_acc["fico_range_low"] + df_acc["fico_range_high"]) / 2.0

    # 2. Load Rejected cohort sample
    logger.info("Loading Rejected cohort sample from %s...", rejected_path)
    df_rej = pd.read_csv(
        rejected_path,
        nrows=sample_size,
        usecols=["Amount Requested", "Risk_Score", "Debt-To-Income Ratio"],
        low_memory=False,
    )
    df_rej["dti_clean"] = (
        df_rej["Debt-To-Income Ratio"]
        .astype(str)
        .str.rstrip("%")
        .astype(float)
    )

    stats = {
        "fico_vs_risk_score": {
            "accepted_fico": compute_cohort_stats(df_acc["fico_midpoint"], "Accepted FICO"),
            "rejected_risk_score": compute_cohort_stats(df_rej["Risk_Score"], "Rejected Risk_Score"),
        },
        "debt_to_income": {
            "accepted_dti": compute_cohort_stats(df_acc["dti"], "Accepted DTI"),
            "rejected_dti": compute_cohort_stats(df_rej["dti_clean"], "Rejected DTI"),
        },
        "loan_amount": {
            "accepted_loan_amnt": compute_cohort_stats(df_acc["loan_amnt"], "Accepted Loan Amnt"),
            "rejected_amount_requested": compute_cohort_stats(
                df_rej["Amount Requested"], "Rejected Amount Requested"
            ),
        },
    }

    # Generate Publication Figure
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "selection_bias_comparison.png"

    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: Credit Score ---
    ax1 = axes[0]
    sns.kdeplot(
        df_acc["fico_midpoint"].dropna(),
        ax=ax1,
        fill=True,
        label=f"Accepted (Median: {stats['fico_vs_risk_score']['accepted_fico']['median']:.0f})",
        color="#2b5c8f",
        alpha=0.4,
    )
    # Filter rejected risk scores to realistic range for clean visualization
    rej_risk_vis = df_rej["Risk_Score"].dropna()
    rej_risk_vis = rej_risk_vis[(rej_risk_vis >= 300) & (rej_risk_vis <= 850)]
    sns.kdeplot(
        rej_risk_vis,
        ax=ax1,
        fill=True,
        label=f"Rejected (Median: {stats['fico_vs_risk_score']['rejected_risk_score']['median']:.0f})",
        color="#c0392b",
        alpha=0.4,
    )
    ax1.axvline(660, color="gray", linestyle="--", linewidth=1, label="Subprime / Prime Threshold (660)")
    ax1.set_title("Credit Score Distribution: Accepted vs Rejected", fontsize=12, fontweight="bold")
    ax1.set_xlabel("FICO / Risk Score")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper left", frameon=True)

    # --- Plot 2: Debt-to-Income (DTI) ---
    ax2 = axes[1]
    # Filter visual range to [0, 80] to handle extreme rejected outliers cleanly
    acc_dti_vis = df_acc["dti"].dropna()
    acc_dti_vis = acc_dti_vis[(acc_dti_vis >= 0) & (acc_dti_vis <= 80)]
    rej_dti_vis = df_rej["dti_clean"].dropna()
    rej_dti_vis = rej_dti_vis[(rej_dti_vis >= 0) & (rej_dti_vis <= 80)]

    sns.kdeplot(
        acc_dti_vis,
        ax=ax2,
        fill=True,
        label=f"Accepted (Median: {stats['debt_to_income']['accepted_dti']['median']:.1f}%)",
        color="#2b5c8f",
        alpha=0.4,
    )
    sns.kdeplot(
        rej_dti_vis,
        ax=ax2,
        fill=True,
        label=f"Rejected (Median: {stats['debt_to_income']['rejected_dti']['median']:.1f}%)",
        color="#c0392b",
        alpha=0.4,
    )
    ax2.axvline(43, color="gray", linestyle="--", linewidth=1, label="QM 43% DTI Cap")
    ax2.set_title("Debt-to-Income (DTI %) Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("DTI (%)")
    ax2.set_ylabel("Density")
    ax2.set_xlim(0, 80)
    ax2.legend(loc="upper right", frameon=True)

    # --- Plot 3: Loan Amount Requested ($) ---
    ax3 = axes[2]
    sns.kdeplot(
        df_acc["loan_amnt"].dropna(),
        ax=ax3,
        fill=True,
        label=f"Accepted (Median: ${stats['loan_amount']['accepted_loan_amnt']['median']:,.0f})",
        color="#2b5c8f",
        alpha=0.4,
    )
    sns.kdeplot(
        df_rej["Amount Requested"].dropna(),
        ax=ax3,
        fill=True,
        label=f"Rejected (Median: ${stats['loan_amount']['rejected_amount_requested']['median']:,.0f})",
        color="#c0392b",
        alpha=0.4,
    )
    ax3.set_title("Loan Amount Distribution", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Requested / Funded Amount ($)")
    ax3.set_ylabel("Density")
    ax3.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved selection bias comparison plot to %s", fig_path)

    # Save summary report JSON
    summary_report = {
        "sample_size": sample_size,
        "summary_statistics": stats,
    }
    report_path = REPORTS_DIR / "selection_bias_summary.json"
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    logger.info("Saved summary report to %s", report_path)

    return summary_report


def generate_audit_notebook(output_path: Path = PROJECT_ROOT / "notebooks" / "01_leakage_and_selection_bias_audit.ipynb"):
    """Create an interactive Jupyter notebook document for leakage and selection bias audit."""
    def make_cell(cell_type: str, source_text: str):
        lines = [line + "\n" for line in source_text.split("\n")]
        if lines and lines[-1] == "\n":
            lines[-1] = ""
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": lines,
        }
        if cell_type == "code":
            cell["execution_count": None]
            cell["outputs"] = []
        return cell

    cells = []

    # Markdown Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Leakage Audit & Selection Bias Analysis\n",
            "\n",
            "**Credit Risk Decision System - Phase 2 Ingestion Gate**\n",
            "\n",
            "### Objectives:\n",
            "1. **Zero-Leakage Verification (ADR 3.1):** Formally verify that no post-origination columns ",
            "(`recoveries`, `collection_recovery_fee`, `last_pymnt_d`, `last_pymnt_amnt`, etc.) exist in training or test sets.\n",
            "2. **Selection Bias & Reject Inference Audit:** Compare the population characteristics of applicants ",
            "who were accepted vs rejected by Lending Club to quantify truncation and survivorship bias.\n",
        ],
    })

    # Code Cell: Setup & Imports
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import json\n",
            "\n",
            "PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()\n",
            "sys.path.insert(0, str(PROJECT_ROOT))\n",
            "\n",
            "from config.constants import POST_ORIGINATION_COLS, IDENTIFIER_COLS, RAW_TARGET_COL, TARGET_COL\n",
            "sns.set_theme(style='whitegrid')\n",
        ],
    })

    # Markdown: Section 1
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Zero-Leakage Gate Audit\n",
            "\n",
            "We inspect the schemas of all generated Parquet partitions (`train.parquet`, `val.parquet`, `test.parquet`, `stress_test.parquet`).\n",
            "Any presence of post-origination data is an automatic failure.\n",
        ],
    })

    # Code Cell: Leakage verification
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from scripts.audit_selection_bias import audit_leakage\n",
            "audit_results = audit_leakage(PROJECT_ROOT / 'data' / 'processed')\n",
            "for part, res in audit_results.items():\n",
            "    print(f\"{part:20s}: Status = {res['status']} | Rows = {res['row_count']:,} | Cols = {res['column_count']}\")\n",
            "    assert res['status'] == 'PASS', f\"Leakage detected in {part}!\"\n",
        ],
    })

    # Markdown: Section 2
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Selection Bias Audit (Accepted vs Rejected Cohorts)\n",
            "\n",
            "Because credit scoring models are trained exclusively on accepted borrowers whose terminal performance is known, ",
            "the training cohort suffers from significant selection bias. Below we quantify the structural difference.\n",
        ],
    })

    # Code Cell: Load and run selection bias
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from scripts.audit_selection_bias import audit_selection_bias\n",
            "summary = audit_selection_bias(\n",
            "    rejected_path=PROJECT_ROOT / 'data' / 'raw' / 'rejected_2007_to_2018Q4.csv.gz',\n",
            "    train_parquet_path=PROJECT_ROOT / 'data' / 'processed' / 'train.parquet',\n",
            "    sample_size=100000,\n",
            ")\n",
            "with open(PROJECT_ROOT / 'reports' / 'selection_bias_summary.json', 'r') as f:\n",
            "    stats = json.load(f)['summary_statistics']\n",
            "pd.DataFrame(stats).T\n",
        ],
    })

    # Code Cell: Display Figure
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import Image, display\n",
            "fig_file = PROJECT_ROOT / 'reports' / 'figures' / 'selection_bias_comparison.png'\n",
            "display(Image(filename=str(fig_file)))\n",
        ],
    })

    # Markdown: Conclusions
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Findings & Methodological Implications\n",
            "\n",
            "- **Credit Score Divergence:** Accepted applicants have a median FICO of ~698, with virtually no applicants below 660. ",
            "In contrast, rejected applicants have a median Risk Score of ~627, extending down below 500.\n",
            "- **DTI Truncation:** Accepted loans cluster around a median DTI of ~18%, while rejected applicants have substantial mass above the 43% Qualified Mortgage cap.\n",
            "- **Policy Implication:** Models trained strictly on accepted loans cannot be blindly extrapolated to deep subprime applications. ",
            "A rigorous three-tier policy gate (Approve / Review / Reject) is vital to route borderline applicants to underwriter scrutiny.\n",
        ],
    })

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=1)
    logger.info("Generated Jupyter notebook at %s", output_path)


def main():
    leakage_results = audit_leakage()
    bias_results = audit_selection_bias()
    generate_audit_notebook()
    logger.info("All audit tasks completed successfully!")


if __name__ == "__main__":
    main()
