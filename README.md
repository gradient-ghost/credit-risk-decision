# Credit Risk Decision System
> An end-to-end, Through-The-Cycle (TTC) machine learning decision system and interactive underwriting policy simulator built on 1.3 million historical loan records.

---

## 📌 Executive Summary & Core Objective

In consumer lending, a machine learning model that merely outputs probabilities is incomplete. A functioning credit decision system requires:
1. **Calibrated Default Probability ($PD$):** A mathematically grounded estimation of the likelihood a borrower will fail to repay.
2. **Economic Risk Accounting ($\text{Expected Loss}$):** Translating abstract probabilities into actual dollar exposures:
   $$\text{Expected Loss} = P(\text{Default}) \times \text{Exposure} \times \text{LGD}$$
   *(where Exposure = Loan Amount, and Loss Given Default ($LGD$) is conservatively fixed at 100%).*
3. **Automated 3-Band Decision Policy:** Converting predictions into operational underwriting tiers:
   * **Tier 1 (Auto-Approve):** Risk $< \tau_{\text{low}}$ (instant origination).
   * **Tier 2 (Manual Review):** $\tau_{\text{low}} \le \text{Risk} \le \tau_{\text{high}}$ (routed to human underwriters with Adverse Action reasons).
   * **Tier 3 (Auto-Reject):** Risk $> \tau_{\text{high}}$ (statutory rejection with legally compliant explanations).
4. **Macroeconomic Policy Dial:** Decoupling borrower fundamentals from economic weather, allowing risk managers to adjust threshold cutoffs as economic climates tighten without retraining the core model.

---

## 🏗️ System Architecture & Data Flow

```
[Lending Club Raw Dataset (2.26M Rows, 150 Columns)]
                         │
                         ▼
             ┌───────────────────────┐
             │ Zero-Leakage Gate     │  <── Drop 35 post-origination columns
             │ & Target Formulation  │  <── Exclude active "Current" loans
             └───────────┬───────────┘
                         │
             [Cleaned Concluded Cohort]
             [1.29M Rows | 113 Features]
                         │
        ┌────────────────┴────────────────────────┐
        │                                         │
        ▼                                         ▼
┌───────────────────────────────┐     ┌───────────────────────────────────┐
│ Through-The-Cycle (TTC) Split │     │ Forensic Selection Bias Audit     │
│ Train: 2013-15 (733k, 18.8%)  │     │ Accepted vs. Rejected (100k sample)│
│ Val:   2016    (293k, 23.3%)  │     │ - Exposed FICO 660 invisible floor│
│ Test:  2017-18 (225k, 21.3%)  │     │ - Uncovered Small Loan Paradox    │
│ Stress:2007-11 ( 40k, 14.3%)  │     └───────────────────────────────────┘
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Scale-Invariant Pipeline      │  <── Fitted strictly on Train
│ - Payment-to-Income (PTI)     │  <── 151 numeric & one-hot features
│ - Credit History Tenure       │  <── Sentinel 999.0 + Delinquency Flags
│ - Account Breadth & Hunger    │  <── Exported: models/feature_pipeline.joblib
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ GPU Accelerated Modeling      │  <── Baselines: Approve-All, LogReg, Grade
│ (XGBoost CUDA, LightGBM, RF)  │  <── Probability Calibration (Platt/Isotonic)
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Policy Engine & Explainability│  <── Expected Loss threshold minimization
│ (SHAP Adverse Action Drivers) │  <── TreeSHAP waterfall attributions
└───────────────┬───────────────┘
                │
        ┌───────┴───────────────────────┐
        ▼                               ▼
┌───────────────────────────────┐     ┌───────────────────────────────────┐
│ FastAPI Prediction Engine     │     │ Streamlit Policy Simulator        │
│ (api/main.py: /predict)       │     │ (5 Tabs: Executive, Policy,       │
│ - Strict Pydantic validation  │     │  Explorer, Segments, Drift)       │
└───────────────────────────────┘     └───────────────────────────────────┘
```

---

## 🔍 Forensic Data Insights & Business Intelligence

Through exploratory data forensics and comparative distribution analysis, this system uncovered critical realities within the Lending Club lending history:

### 1. The "Invisible Floor" at FICO 660 (Empirical Selection Bias)
* **The Finding:** Comparing 100,000 accepted loans against 100,000 rejected applications revealed that accepted borrowers had a **strict minimum FICO of 662.0** (mean: 695.6, median: 687.0). Rejected applicants averaged a risk score of 579.4 (median: 627.0, 25th percentile: 524.0).
* **The Business Meaning:** Lending Club enforced an unwritten hard-stop rule: applicants below FICO 660 were rejected on the spot.
* **The Engineering Consequence:** Proves our training data inherently suffers from **survivorship/selection bias**; our model learns patterns of creditworthy borrowers, requiring policy rule-based gates for subprime applications.

### 2. The "2016 Default Spike" (+4.48% Jump) (Macro & Concept Drift)
* **The Finding:** While default rates held steady at **18.81%** during 2013–2015, they surged to **23.29%** in 2016 and stayed elevated at **21.29%** in 2017–2018.
* **The Business Meaning:** Lending Club expanded origination volume ahead of public scrutiny, loosening underwriting standards just as the Federal Reserve began its interest rate hiking cycle.
* **The Engineering Consequence:** A standard random train/test split would shuffle 2016 into training and mask this degradation. Our **Walk-Forward Out-Of-Time (OOT) Split** exposes this shift, forcing downstream probability calibration to adjust for real-world macroeconomic stress.

### 3. The "Small Loan Paradox" (Distress Borrowing)
* **The Finding:** Rejected applicants requested *smaller* loans (mean: $10,110, median: $8,000) than accepted borrowers (mean: $14,639, median: $12,800).
* **The Business Meaning:** Distressed subprime applicants sought small, emergency cash loans ($3k–$8k) to bridge urgent liquidity needs, but were disqualified by high DTI and low credit scores. Approved prime borrowers applied for larger, strategic debt consolidation loans ($15k–$35k).

---

## 🛡️ Zero-Leakage Data Partitioning

| Partition | Time Vintage | Clean Rows | Default Rate | Role in Engine |
| :--- | :---: | :---: | :---: | :--- |
| **`train.parquet`** | **2013–2015** | **733,453** | **18.81%** | Core Through-The-Cycle Training |
| **`val.parquet`** | **2016** | **293,105** | **23.29%** | Probability Calibration & Threshold Tuning |
| **`test.parquet`** | **2017–2018** | **225,639** | **21.29%** | Untouched Walk-Forward Final Exam |
| **`stress_test.parquet`** | **2007–2011** | **39,786** | **14.25%** | 2008 Financial Crisis Stress Test |

* **Zero Post-Origination Leakage:** 35 future columns (e.g. `recoveries`, `collection_recovery_fee`, `last_pymnt_amnt`, `out_prncp`) were permanently stripped.
* **No Right-Censoring Noise:** Active `"Current"` loans were filtered out, ensuring models only train on matured, verified credit outcomes.

---

## ⚙️ Scale-Invariant Feature Engineering Pipeline

Raw dollar numbers (e.g. $50,000 income) change meaning over time due to inflation. Our feature engineering engine (`src/features/build_features.py`) constructs **151 scale-invariant structural features**:

1. **Payment-to-Income Ratio (PTI):** $\text{Installment} / (\text{Annual Income} / 12)$ — measures monthly budget suffocation.
2. **Credit History Tenure:** Time in decimal years between `earliest_cr_line` and loan `issue_d` — captures multi-decade borrower track record.
3. **Account Breadth:** $\text{Open Accounts} / \text{Total Accounts}$ — isolates credit capacity from excessive debt expansion.
4. **Credit Hunger Velocity:** `inq_last_6mths` — acts as a leading indicator of emergency liquidity distress.
5. **Clean Record Sentinel Imputation:** Over 50% of borrowers had missing delinquency records because they had never missed a payment. Missing values are filled with sentinel $999.0$ paired with a binary `has_delinq_history = 0` indicator, preventing the false penalization of prime borrowers.
6. **Zero Train-Serving Skew:** Entire transformation is packaged in `models/feature_pipeline.joblib` (fitted strictly on `train.parquet`), guaranteeing that the Phase 6 FastAPI inference engine runs the exact same mathematical transformations.

---

## 🏆 Model Benchmarking & Probability Calibration

We evaluated 3 baselines and 3 challenger architectures across 733,000 historical training loans, calibrating probabilities on the 2016 validation fold and testing out-of-time on 2017–2018 ($N = 225,639$).

### Master Benchmark Scorecard (Out-of-Time Test Set: 2017–2018)
| Model Architecture | Test Gini ($2\text{AUC}-1$) | KS Statistic | ROC-AUC | Brier Score | Strategic Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Approve-Everyone Baseline** | 0.0000 | 0.0000 | 0.5000 | 0.1682 | Uninformative financial floor |
| **Logistic Regression Scorecard** | 0.2478 | 0.1801 | 0.6239 | 0.2214 | Regulated linear benchmark |
| **Lending Club Grade Classifier** | 0.3642 | 0.2660 | 0.6821 | 0.2218 | Platform internal quantitative team |
| **Random Forest (100 Trees)** | 0.4044 | 0.2932 | 0.7022 | 0.2100 | Non-linear bagging ensemble |
| **LightGBM** | 0.4214 | 0.3062 | 0.7107 | 0.2134 | High-speed gradient booster |
| **Option Z Booster (NO GRADE)** | **0.4218** | **0.3049** | **0.7109** | 0.2128 | **Our engineered ratios alone!** |
| **XGBoost (CUDA Hist)** | **0.4229** | **0.3066** | **0.7115** | 0.2084 | Raw uncalibrated champion |
| **XGBoost (Platt Calibrated) 🏆** | **0.4229** | **0.3066** | **0.7115** | **0.1525** | **OFFICIAL CHAMPION MODEL** |

### Key Machine Learning Findings:
1. **Beating the Regulated Benchmark:** The champion model outperforms the traditional banking Logistic Regression scorecard by **+17.5 Gini points (+70% relative improvement)** in risk discrimination.
2. **Beating the Platform’s Internal Grading:** Outperformed Lending Club's native assigned credit grades by **+5.87 Gini points**, proving that our non-linear ratio combinations uncover risk pockets obscured by manual tiering.
3. **The Option Z Ablation Proof:** Completely dropping all 42 Lending Club `grade` and `sub_grade` features left our booster with **99.75% of its predictive power intact** (Gini 0.4218 vs 0.4229). On the 2008 Lehman Brothers crisis stress test, the Option Z model actually **outperformed the full model (0.3541 vs 0.3539)**, confirming that structural financial ratios remain invariant when human credit ratings collapse.
4. **Probability Calibration (Platt Scaling):** Training with `scale_pos_weight = 4.3153` distorts raw tree probabilities toward extreme bounds. Applying Platt Scaling on validation holdouts reduced probability error (Brier Score) from **0.2084 down to 0.1525 ($-26.8\%$ error reduction)** while preserving 100% rank-order discrimination. Bootstrapped 95% Confidence Interval on test set: $\text{Gini} \in [0.4181, 0.4277]$.

---

## 🎯 Decision Policy Engine & Expected Loss Optimization (Phase 5)

A credit model that outputs probabilities does not make business decisions on its own. The Policy Engine (`src/policy/decision_engine.py`) translates calibrated probabilities into financial balance-sheet outcomes:

$$\text{Expected Loss (EL)} = P(\text{Default}) \times \text{Exposure} \times \text{LGD} \quad (\text{where LGD} = 1.0)$$

### 2D Grid Search & Cutoff Optimization
We conducted a 2D parameter search over $\tau_{\text{low}} \in [0.05, 0.25]$ and $\tau_{\text{high}} \in [0.15, 0.45]$ on the 2016 Out-of-Time Validation Set ($N = 293,105$), subject to business constraints:
* **Auto-Approval Floor:** $\ge 60\%$ automated approval volume to maintain business velocity.
* **Underwriter Bandwidth:** $\le 20\%$ manual review capacity.

**Optimal Policy Cutoffs Identified:**
$$\tau_{\text{low}}^* = 0.25 \quad \big| \quad \tau_{\text{high}}^* = 0.35$$

### Master Policy Scorecard (Out-of-Time Test Set: 2017–2018, $N = 225,639$)
| Decision Tier | Threshold Condition | Loan Count | Portfolio % | Exposure ($) | Realized Default Rate | Expected Loss ($) | Expected Losses Saved ($) | Realized Losses Saved ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Approve-Everyone Baseline** | None | 225,639 | 100.00% | $3,259,579,400 | 21.29% | $825,487,236 | $0 | $0 |
| **Tier 1: Auto-Approve** | $p < 0.25$ | 138,709 | **61.47%** | $1,775,283,425 | **13.05%** | $245,389,355 | — | — |
| **Tier 2: Manual Review** | $0.25 \le p \le 0.35$ | 40,620 | **18.00%** | $630,641,125 | 27.69% | $187,931,574 | — | — |
| **Tier 3: Auto-Reject** | $p > 0.35$ | 46,310 | **20.52%** | $853,654,850 | **40.38%** | — | **$392,166,307** | **$348,536,175** |
| **🌟 Headline Executive Metric** | **$\tau^* = (0.25, 0.35)$** | — | — | — | — | — | **$392,166,307 (47.51%)** | **$348,536,175 (45.25%)** |

### Policy Engineering Insights:
1. **The Asymmetric Loss Frontier ($392.2M Saved):** Auto-rejecting the top 20.52% riskiest loans ($p > 0.35$) eliminates **$392.17 Million** in Expected Loss (**47.51% reduction**) and directly avoids **$348.54 Million** in realized charge-offs. The auto-rejected cohort exhibited a **40.38% realized default rate** (nearly double the portfolio average).
2. **Human-in-the-Loop Gray Zone (18.0% Review):** Rather than blindly rejecting borderline applicants, routing 18.00% to underwriters preserves **$630.6M** in viable loan volume for secondary document verification.
3. **Pristine Auto-Approve Tier:** Auto-approved borrowers enjoy an ultra-low **13.05% default rate** (down from 21.29%), creating an exceptionally stable prime credit asset class.

---

## 🔍 Model Explainability & Statutory Adverse Action Notices (Phase 5)

Consumer lending is strictly regulated under the **Equal Credit Opportunity Act (ECOA / Regulation B)** and the **Fair Credit Reporting Act (FCRA)**. When a loan is rejected or modified, lenders are legally required to provide the applicant with the principal reasons for adverse action.

Our explainer engine (`src/explainability/shap_explainer.py`) implements `shap.TreeExplainer` on the champion booster to extract local Shapley values ($\phi_i$) and maps them to transparent, non-technical disclosures.

### Top 5 Global Credit Risk Drivers (Mean |SHAP| on 1,000 Test Sample):
1. **`int_rate` (Mean |SHAP| = 0.4534):** *High interest rate risk tier on requested credit product*
2. **`term_months` (Mean |SHAP| = 0.2743):** *Extended loan repayment term (60 months vs 36 months)*
3. **`acc_open_past_24mths` (Mean |SHAP| = 0.1188):** *Too many credit accounts opened in the past 24 months (credit-seeking velocity)*
4. **`dti` (Mean |SHAP| = 0.1178):** *High aggregate debt-to-income ratio*
5. **`payment_to_income_ratio` (Mean |SHAP| = 0.0951):** *High monthly debt payment relative to monthly income (budget suffocation)*

*A 300 DPI publication-quality summary beeswarm plot is generated at `reports/figures/shap_summary_beeswarm.png`.*

### Example Statutory Adverse Action Notice Generated:
```text
================================================================================
                    STATEMENT OF CREDIT ADVERSE ACTION
================================================================================
Date: 2026-11-09
Applicant Reference: APP-41279

DECISION: Credit Application Declined (Auto-Reject, Risk Tier 3)
Model Evaluated Probability of Default: 44.82%
Policy Maximum Threshold: 35.00%

In accordance with the Equal Credit Opportunity Act (ECOA) and the Fair Credit
Reporting Act (FCRA), our credit decision was based on the following principal
factors:

  1. High interest rate risk tier on requested credit product
  2. High aggregate debt-to-income ratio (DTI exceeds prudent policy limits)
  3. Too many credit accounts opened in the past 24 months

You have the right under the FCRA to receive a free copy of your credit report
and dispute inaccurate information.
================================================================================
```

---

## ⚡ High-Throughput REST API Scoring Engine (Phase 6)

In production banking, credit scoring must be decoupled from the batch training pipeline and respond in real-time under strict latency service-level agreements (SLAs).

### Architecture & Lifespan Preloading (`api/main.py`)
* **Lifespan Context Management:** The application preloads `feature_pipeline.joblib`, `champion_model.joblib`, `feature_names.json`, and `policy_thresholds.json` into memory once during startup.
* **Warmup Pass:** On initialization, the engine executes a synthetic warmup pass through the XGBoost tree booster, ensuring zero JIT cold-start latency for the first live user request.
* **Pydantic V2 Contract (`api/schemas.py`):** Enforces strict field bounds (`loan_amnt > 0`, `annual_inc > 0`, `fico_score` in $[300, 850]$, `dti >= 0`, `term_months` restricted to 36 or 60). Rejects malformed requests with `HTTP 422 Unprocessable Entity` before pipeline invocation.

### Production Endpoints
1. `GET /health`: Liveness and readiness probe reporting model readiness flags, memory footprint, and server uptime.
2. `GET /policy`: Returns operational threshold configuration ($\tau_{\text{low}}^* = 0.25, \tau_{\text{high}}^* = 0.35$) and validation metrics.
3. `POST /predict`: Real-time applicant scoring returning calibrated default probability, Expected Loss ($), decision tier (`AUTO_APPROVE`, `MANUAL_REVIEW`, `AUTO_REJECT`), and statutory Adverse Action reasons.

### Latency Benchmarking (10 Consecutive Invocations)
* **Mean Inference Latency:** **37.48 ms** (SLA Target: $< 50.0\text{ ms}$ — **PASSED**)
* **p95 Latency:** **40.97 ms**
* **Max Latency:** **40.97 ms** (SLA Target: $< 100.0\text{ ms}$ — **PASSED**)

---

## 🧪 Automated Testing & Verification

The codebase enforces strict test-driven quality assurance via `pytest` (**53 / 53 tests passing in 5.08s**):
```bash
.venv/bin/pytest tests/ -v
```
* **API Endpoints & SLAs (`tests/test_api.py`):** Asserts health/readiness status, policy rules, prime borrower auto-approval, subprime auto-rejection with 3 FCRA adverse action reasons, manual review routing, HTTP 422 input validations, and sub-100ms latency SLA (7 tests).
* **Leakage Gate (`tests/test_leakage.py`):** Asserts zero blacklisted post-origination features exist in any processed dataset (8 tests).
* **Temporal Cutoffs (`tests/test_splits.py`):** Asserts strict chronological year boundaries and realistic default rate bounds ($10\% - 30\%$) (8 tests).
* **Inference Symmetry (`tests/test_features.py`):** Asserts single raw applicant dictionary passes through `feature_pipeline.joblib` producing a finite 2D array of exactly `(1, 151)` features with 0 NaNs and 0 Infs (7 tests).
* **Model Validation (`tests/test_models.py`):** Asserts champion model exists, outputs bounded $[0, 1]$ probabilities, achieves Gini $> 0.25$, meets Brier $< 0.20$, and retains $> 95\%$ power in Option Z ablation (9 tests).
* **Policy Verification (`tests/test_policy.py`):** Asserts `models/policy_thresholds.json` validity ($0 < \tau_{\text{low}} < \tau_{\text{high}} < 1$), strict monotonic band assignment, boundary conditions, expected loss math, and evaluation metric accounting (6 tests).
* **Explainability Verification (`tests/test_explainability.py`):** Asserts existence of SHAP beeswarm plot and precomputed background cache, verifies exactly 3 non-empty adverse action reason strings, tests Series/NumPy input symmetry, and validates ECOA notice formatting (6 tests).

---

## 📂 Repository Structure

```
credit-risk-decision/
├── config/
│   ├── constants.py               # Global seed (42), 35-column blacklist, path constants
│   └── model_config.yaml          # Hyperparameters, split cutoffs, GPU acceleration configs
├── data/
│   ├── raw/                       # Immutable source data (accepted & rejected .csv.gz)
│   └── processed/                 # Compressed Parquet partitions (train, val, test, stress)
├── notebooks/
│   └── 01_leakage_and_selection_bias_audit.ipynb # Forensic distribution & leakage analysis
├── reports/
│   ├── selection_bias_summary.json # Quantitative FICO & DTI comparison metrics
│   ├── sample_shap_test.joblib    # Precomputed SHAP test cache for sub-second UI rendering
│   └── figures/
│       ├── selection_bias_comparison.png  # Selection bias distribution visual proofs
│       ├── calibration_curves.png         # Platt vs Isotonic reliability diagrams
│       └── shap_summary_beeswarm.png      # Global 15-feature TreeSHAP beeswarm plot
├── src/
│   ├── data/
│   │   └── make_dataset.py        # Chunked ingestion, zero-leakage gate, parquet partitioner
│   ├── features/
│   │   └── build_features.py      # Reusable CreditFeaturePipeline & ratio engine
│   ├── models/                    # Model training, Platt calibration & runs logging
│   ├── policy/                    # 3-band threshold optimization engine & EL calculator
│   ├── explainability/            # TreeSHAP explainer & statutory Adverse Action generator
│   └── monitoring/                # [Phase 7] PSI feature drift & score stability engine
├── api/                           # [Phase 6] FastAPI scoring service
│   ├── main.py                    # Async lifespan application, /health, /policy, /predict
│   └── schemas.py                 # Pydantic V2 input & response data contracts
├── dashboard/                     # [Phase 7] 5-Tab Streamlit underwriting dashboard
├── tests/
│   ├── test_api.py                # REST API, SLA latency & Pydantic validation assertions
│   ├── test_env.py                # Environment & GPU detection assertions
│   ├── test_leakage.py            # Zero post-origination leakage assertions
│   ├── test_splits.py             # Temporal boundary assertions
│   ├── test_features.py           # Feature matrix & inference symmetry assertions
│   ├── test_models.py             # Model performance, calibration & Option Z assertions
│   ├── test_policy.py             # Policy threshold validity & EL calculation assertions
│   └── test_explainability.py     # SHAP artifacts & Adverse Action notice assertions
├── models/
│   ├── feature_pipeline.joblib    # Serialized scikit-learn feature preprocessor
│   ├── feature_names.json         # 151 ordered feature names
│   ├── policy_thresholds.json     # Optimal policy thresholds (tau_low=0.25, tau_high=0.35)
│   └── runs_log.csv               # Complete benchmark audit log across 6 models
├── requirements.txt               # Pinned dependencies with CUDA support
└── README.md                      # Architecture documentation & forensic case study
```
