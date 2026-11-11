# Credit Risk Decision System
> An institutional-grade credit risk decision engine with walk-forward out-of-time validation and crisis stress testing, trained on 1.3 million historical loan records (Lending Club 2007–2018).

---

## ⚡ Executive Summary (The 60-Second Scan)

Consumer credit models cannot stop at raw probabilities. A production credit system must translate default likelihoods into balance-sheet risk, enforce operational constraints, and provide legally compliant explanations under federal lending regulations.

| Metric | Measured Value | Benchmark / SLA | Business & Operational Impact |
| :--- | :---: | :---: | :--- |
| **Expected Loss Avoided** | **$392.17M (-47.51%)** | vs. Approve-All ($825.5M) | OOT backtest declines 20.5% riskiest loans ($p > 0.35$, 40.4% default rate) |
| **Auto-Approval Rate** | **61.47%** | $\ge 60\%$ Policy Floor | Underwrites $1.78B prime credit at pristine **13.05%** default rate |
| **Underwriter Capacity** | **18.00%** | $\le 20\%$ Capacity Ceiling | Routes $630.6M borderline loans to manual review, preserving volume |
| **Champion Test Gini** | **0.4229** ($2\text{AUC}-1$) | vs. 0.2478 Regulated LogReg | **+70.6% relative discrimination lift** over linear scorecard ($p < 0.001$) |
| **Probability Calibration** | **0.1525** (Brier Score) | vs. 0.2084 Uncalibrated | Platt Scaling reduces probability error by **-26.8%** on 2017–18 test set |
| **API Scoring Latency** | **37.48 ms** (p95: 40.97 ms) | $< 50.0\text{ ms}$ SLA | Real-time FastAPI scoring with strict Pydantic V2 input validation |

---

## 🚀 One-Command Launch (Docker Compose)

Spin up the entire microservice stack (FastAPI backend + Streamlit UI) in an isolated, non-root multi-container environment:

```bash
docker compose up --build
```

* **Interactive Risk Command Center:** [http://localhost:8501](http://localhost:8501)
* **REST API Documentation (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **API Health & Readiness Probe:** `curl http://localhost:8000/health`

---

## 🏗️ End-to-End System Architecture

```
[2.26M Raw Loans] ──▶ [Terminal Status Gate] ──▶ [Scale-Invariant Ratios] ──▶ [Platt-XGBoost]
 (2007-2018 LC)        (Drop 35 Cols / 1.29M Clean) (151 PTI, Tenure, Sentinel)   (Gini 0.423, Brier 0.153)
                                                                                       │
       ┌───────────────────────────────────────────────────────────────────────────────┘
       ▼
 [3-Band Policy Dial] ──▶ [FastAPI Microservice] ──▶ [Streamlit Command Center]
 (tau: 0.25 / 0.35)        (Port 8000 | < 40ms)        (Port 8501 | Sub-10ms Sim)
```

---

## 🔍 Forensic Data Audit & Selection Bias

Exploratory distribution analysis across 2.26 million raw accepted loan records and 27.6 million rejected applications uncovered critical structural characteristics:

```
Accepted Applications (100k sample)         Rejected Applications (100k sample from 27.6M)
Mean FICO: 695.6 | Min FICO: 662.0          Mean Risk Score: 579.4 | Median: 627.0
[============== FICO >= 662 ===============] | [====== FICO 300 - 661 ======] (100% Rejected)
```

1. **The FICO 660 Invisible Floor (Selection Bias):** Lending Club enforced an unwritten hard stop: zero accepted loans existed below FICO 662. Proves the training population represents *already-screened* applicants (survivorship bias), requiring explicit policy guardrails for subprime cohorts.
2. **The 2016 Default Spike (+4.48 pp Shift / +23.8% Relative Surge):** Default rates surged from **18.81%** (2013–15) to **23.29%** in 2016 as origination volumes expanded ahead of Fed rate hikes. A random train/test split would blend these regimes; our **Walk-Forward Out-Of-Time (OOT) Split** isolates this shift.
3. **The Small Loan Paradox:** Rejected applicants requested smaller loans (mean: $10,110) than accepted borrowers ($14,639). Subprime applicants sought emergency liquidity ($3k–$8k), whereas prime borrowers applied for structured debt consolidation ($15k–$35k).

### Zero-Leakage Temporal Partitioning
* **35 Future Columns Stripped:** Permanently removed post-origination columns (`recoveries`, `collection_recovery_fee`, `last_pymnt_amnt`, `out_prncp`, etc.).
* **Active Loans Excluded:** Filtered 2.26M raw applications down to 1,291,983 concluded loans by removing active `"Current"` and grace-period accounts to eliminate right-censoring noise.

| Split | Vintage | Observations | Default Rate | Strategic Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **`train.parquet`** | 2013–2015 | 733,453 | 18.81% | Model fitting & feature pipeline training |
| **`val.parquet`** | 2016 | 293,105 | 23.29% | Platt scaling & 2D policy threshold tuning |
| **`test.parquet`** | 2017–2018 | 225,639 | 21.29% | Untouched out-of-time walk-forward evaluation |
| **`stress_test.parquet`** | 2007–2011 | 39,786 | 14.25% | 2008 Great Financial Crisis stress test |

---

## ⚙️ Scale-Invariant Feature Pipeline

To prevent inflation distortion and train-serving skew, `src/features/build_features.py` builds **151 scale-invariant features** fitted strictly on `train.parquet`:

* **Payment-to-Income (PTI):** $\text{Installment} / (\text{Annual Income} / 12)$ — isolates budget suffocation from raw dollar salary.
* **Credit History Tenure:** Decimal years between `earliest_cr_line` and `issue_d` — captures multi-decade repayment experience.
* **Account Breadth:** $\text{Open Accounts} / \text{Total Accounts}$ — separates credit capacity from debt over-extension.
* **Credit Hunger Velocity:** `inq_last_6mths` — acts as a leading indicator of emergency liquidity distress.
* **Sentinel Imputation:** Clean-record borrowers with no historical delinquencies receive sentinel `999.0` paired with `has_delinq_history = 0`, preventing false penalization of prime applicants.

---

## 🏆 Model Tournament & Platt Calibration

We evaluated 3 baselines and 3 challenger architectures across 733,453 loans, calibrating probabilities on the 2016 validation fold and testing out-of-time on 2017–2018 ($N = 225,639$).

| Model Architecture | Test Gini ($2\text{AUC}-1$) | KS Statistic | ROC-AUC | Brier Score | Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Approve-All Baseline** | 0.0000 | 0.0000 | 0.5000 | 0.1682 | Uninformative baseline |
| **Logistic Regression Scorecard** | 0.2478 | 0.1801 | 0.6239 | 0.2214 | Regulated linear baseline |
| **Lending Club Grade Classifier** | 0.3642 | 0.2660 | 0.6821 | 0.2218 | Platform internal quantitative rating |
| **Random Forest (100 Trees)** | 0.4044 | 0.2932 | 0.7022 | 0.2100 | Bagging ensemble |
| **LightGBM** | 0.4214 | 0.3062 | 0.7107 | 0.2134 | High-speed gradient booster |
| **Option Z Booster (NO GRADE)** | **0.4218** | **0.3049** | **0.7109** | 0.2128 | **Engineered ratios alone (no grade)** |
| **XGBoost (CUDA Hist)** | **0.4229** | **0.3066** | **0.7115** | 0.2084 | Raw uncalibrated champion |
| **XGBoost (Platt Calibrated) 🏆** | **0.4229** | **0.3066** | **0.7115** | **0.1525** | **OFFICIAL CHAMPION MODEL** |

### Key Machine Learning Takeaways
1. **Beating Regulated & Platform Baselines:** Champion outperforms traditional Logistic Regression by **+17.51 Gini points (+70.6% relative lift)** and beats Lending Club’s proprietary grade model by **+5.87 Gini points**.
2. **The Option Z Ablation Proof:** Completely removing all 42 Lending Club `grade` and `sub_grade` features left our booster with **99.75% of its discriminatory power intact** (Gini 0.4218 vs 0.4229). On the 2008 crisis test, Option Z actually outperformed the full model (**0.3541 vs 0.3539**), proving that structural ratios remain invariant when human credit ratings collapse.
3. **Platt Calibration:** Corrected tree probability distortion, cutting Brier score by **-26.8% (0.2084 $\to$ 0.1525)**. Test set 95% bootstrap confidence interval: $\text{Gini} \in [0.4181, 0.4277]$.

---

## 🎯 Decision Policy Engine & Expected Loss Economics

The policy engine (`src/policy/decision_engine.py`) optimizes operational thresholds via 2D grid search on the 2016 validation set ($N = 293,105$), subject to business constraints: $\ge 60\%$ auto-approval and $\le 20\%$ underwriter review capacity.

$$\text{Optimal Cutoffs:} \quad \tau_{\text{low}}^* = 0.25 \quad \big| \quad \tau_{\text{high}}^* = 0.35$$

### Master Policy Scorecard (Out-of-Time Test Set: 2017–2018, $N = 225,639$)
| Decision Tier | Threshold Rule | Loan Count | Volume % | Exposure ($) | Realized Default Rate | Expected Loss Avoided ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Approve-All Baseline** | None | 225,639 | 100.0% | $3,259,579,400 | 21.29% | $0 |
| **Tier 1: Auto-Approve** | $p < 0.25$ | 138,709 | **61.47%** | $1,775,283,425 | **13.05%** | — |
| **Tier 2: Manual Review** | $0.25 \le p \le 0.35$ | 40,620 | **18.00%** | $630,641,125 | 27.69% | Preserves $630M loan volume |
| **Tier 3: Auto-Reject** | $p > 0.35$ | 46,310 | **20.52%** | $853,654,850 | **40.38%** | **$392,166,307 (-47.51%)** |

* **Asymmetric Risk Defense:** In out-of-time backtesting, declining the top 20.52% riskiest applications identifies **$348.54M** in historical charge-offs and avoids **$392.17M** in predicted Expected Loss (under conservative $\text{LGD} = 1.0$ and gross committed exposure).
* **Underwriter Routing:** Channeling 18.00% borderline loans ($630.6M exposure) to manual review preserves viable origination volume at a 27.69% default rate.

> [!NOTE]
> **Methodological Note on Counterfactual Accounting:** All 225,639 loans in the 2017–2018 test cohort were historically originated by Lending Club. Reported loss metrics represent a counterfactual simulation. Manual review loans ($18.00\%$) are treated as originated in loss accounting, representing the most conservative posture for automated policy evaluation.

---

## ⚖️ Regulatory Compliance & Sub-Cohort Calibration Audit (ADR 9.1)

Consumer credit in the United States is governed by the **Equal Credit Opportunity Act (ECOA / Regulation B)** and the **Fair Credit Reporting Act (FCRA)**.

### 1. Proxy Variable Elimination & Statutory Compliance
- **Redlining Prevention:** Dropped 3-digit `zip_code` in the feature pipeline to prevent indirect geographic bias correlated with demographic clustering.
- **Employer Regularization:** Excluded free-text employer names (`emp_title`) with > 500,000 sparse levels to avoid employer-prestige bias.
- **Statutory Non-Collection:** *Protected demographic attributes (race, gender, sex, age, marital status, religion) are neither collected nor utilized, adhering strictly to federal statutory prohibitions under 12 CFR § 1002.5(b).*

### 2. Sub-Cohort Calibration Audit (Test Partition, $N = 225,639$)
Evaluating calibration stability across core underwriting segments on the untouched 2017–2018 test cohort:

| Sub-Cohort | Group | Test Volume (%) | Realized Default Rate | Mean Predicted PD | Calibration Spread |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Residential Tenancy** | Mortgage | 113,335 (50.2%) | 17.38% | 20.92% | +3.55% |
| | Own | 27,785 (12.3%) | 21.85% | 23.28% | +1.43% |
| | Rent | 84,296 (37.4%) | 26.38% | 25.88% | -0.50% |
| **Loan Purpose** | Debt Consolidation | 123,790 (54.9%) | 22.28% | 24.82% | +2.54% |
| | Credit Card Refi | 43,673 (19.4%) | 18.44% | 21.23% | +2.79% |
| | Small Business | 2,368 (1.0%) | 36.15% | 26.65% | -9.50% (Under-priced) |

> [!NOTE]
> The model maintains consistent calibration across core consumer credit segments (spreads $< 3.6\%$), but underestimates default risk in specialized non-consumer categories such as Small Business loans ($-9.50\%$ spread), indicating that commercial credit warrants distinct underwriting sub-models.

### 3. TreeSHAP Local Attributions & Adverse Action Explanations
When an applicant is rejected or reviewed, `src/explainability/shap_explainer.py` extracts local TreeSHAP attributions and maps the principal risk-increasing features to human-readable disclosures modeled after ECOA/FCRA adverse action requirements:

* **Attribution Scope:** Explains applicant log-odds divergence from the baseline population, surfacing the top 3 factors driving elevated model risk.
* **API Delivery:** Served in real-time under `adverse_action_reasons` via the `POST /predict` endpoint for all non-approved tiers.
* **Regulatory Note:** Adverse action templates provide mathematical interpretability of model risk factors; institutional production deployment requires formal legal and compliance sign-off for statutory notice fulfillment.

```text
================================================================================
                    STATEMENT OF CREDIT ADVERSE ACTION (ECOA / FCRA)
================================================================================
Applicant: LC-2017-TEST-0042           Decision: DECLINED (Auto-Reject, Tier 3)
Evaluated Default Probability: 44.82%   Policy Threshold: 35.00%

Principal Reasons for Adverse Action:
  1. High interest rate risk tier on requested credit product (int_rate)
  2. High aggregate debt-to-income ratio exceeding policy limits (dti)
  3. Elevated credit-seeking velocity in past 24 months (acc_open_past_24mths)
================================================================================
```

---

## ⚡ Production Services: API & Dashboard

### 1. High-Throughput REST API (`api/main.py`)
* **Lifespan Caching & JIT Warmup:** Models and preprocessors load once during application startup. A synthetic warmup pass eliminates cold-start JIT overhead.
* **Latency Benchmarks:** Mean **37.48 ms**, p95 **40.97 ms** (well under the 50ms SLA).
* **Endpoints:** `GET /health` (readiness probe), `GET /policy` (threshold config), `POST /predict` (scores, tiers, and top-3 FCRA reasons).

### 2. Interactive Risk Command Center (`dashboard/app.py`)
* **Sub-10ms Reactivity:** Vectorized NumPy slicing over 225,639 loans runs in **2.91 ms to 7.38 ms**, powering dynamic 60 FPS slider simulations.
* **5 Modular Tabs:**
  - **Tab 1: Executive Overview:** Headline exposure KPIs, model tournament leaderboard, and interactive reliability curves.
  - **Tab 2: Policy Dial:** Real-time dual threshold sliders, macro stress presets, and waterfall loss savings.
  - **Tab 3: Underwriter Dossier:** Applicant factsheets, decision badges, and interactive Plotly SHAP waterfall plots.
  - **Tab 4: Stability & Degradation:** Walk-forward out-of-time degradation (2016 $\to$ 2018), 2008 Lehman stress test, and strict grade monotonicity validation (Grade A 6.33% $\to$ Grade G 50.09%).
  - **Tab 5: Monitoring & Drift:** Basel III Population Stability Index (PSI) monitoring across top features (`int_rate` $PSI = 0.146$ captures Fed hiking cycle, all others $PSI < 0.05$).

---

## 🐳 Docker Multi-Container Architecture

The system is containerized into production-ready microservices using Docker Compose:

* **Base & Tooling:** Multi-stage `python:3.12-slim` with OpenMP (`libgomp1`) and Astral `uv` for 10x faster layer builds.
* **Security Isolation:** Containers run as non-root user `appuser` (UID 1000).
* **Orchestration & Healthchecks:** The dashboard container depends on `service_healthy` from the API container (`curl -f http://localhost:8000/health`).
* **Volume Optimization:** Precomputed lightweight artifacts in `reports/` (dossier, PSI table, baseline scores) allow instant container boot without copying 200MB+ raw parquet splits.

```bash
# Verify container configuration
sudo docker compose config

# Spin up services in detached mode
sudo docker compose up -d

# Check live logs
sudo docker compose logs -f
```

---

## ✅ 10 Binary Acceptance Criteria Audit (ADR 9.2)

| # | Acceptance Criterion | Target Specification | Empirical Outcome & Verification | Status |
| :-: | :--- | :--- | :--- | :-: |
| **1** | **Verifiable Headline Claim** | Quantify Expected Loss reduction vs Approve-All | **$392,166,307 (-47.51%) EL avoided in OOT backtest** at 61.47% auto-approval | **PASS** |
| **2** | **Beat Linear Regulated Baseline** | Significant Gini improvement vs Logistic Regression | **Gini 0.4229 vs 0.2478** (+17.51 pp / +70.6% relative lift, $p < 0.001$) | **PASS** |
| **3** | **Beat Proprietary Platform Baseline** | Match or exceed Lending Club credit grade Gini | **Gini 0.4229 vs 0.3642** (+5.87 pp / +16.1% relative lift) | **PASS** |
| **4** | **Validated Calibration** | Near-diagonal reliability curve & Brier $< 0.16$ | **Brier score 0.1525** (down from 0.2084 raw, -26.8% error) | **PASS** |
| **5** | **Zero Lookahead Feature Leakage** | 100% post-origination columns stripped | **35 blacklisted columns dropped**; verified by `tests/test_leakage.py` | **PASS** |
| **6** | **Functional Adverse Action Notices** | Top-3 SHAP factors mapped to FCRA text | Verified by `tests/test_explainability.py` and live in Tab 3 | **PASS** |
| **7** | **Interactive Policy Simulation** | Vectorized simulation latency $< 10\text{ ms}$ | **2.91 ms to 7.38 ms** execution across 225,639 loans; verified in Tab 2 | **PASS** |
| **8** | **60-Second Recruiter Screen** | High-impact metrics & architecture above fold | Executive summary, box pipeline, quickstart, and scorecard above the fold | **PASS** |
| **9** | **30-Minute Engineering Audit** | Justifications cross-referenced with ADR log | All 9 ADR groups documented, Option Z ablation proven, scripts reproducible | **PASS** |
| **10** | **Frictionless Containerized Demo** | Single command zero-auth deployment | `docker compose up` spins up healthy API (:8000) and Dashboard (:8501) | **PASS** |

---

## 🧪 Automated Testing Suite

The codebase enforces strict test-driven quality assurance via `pytest` (**66 / 66 tests passing in 5.16s**):

```bash
.venv/bin/pytest tests/ -v
```

* **Docker & Compose (`tests/test_docker.py`):** Multi-stage Dockerfile, `.dockerignore` exclusions, compose service dependencies & healthchecks (3 tests).
* **API Endpoints & SLAs (`tests/test_api.py`):** Liveness probe, policy configuration, decision tiers, FCRA reasons, input validations, sub-100ms latency SLA (7 tests).
* **Dashboard Components (`tests/test_dashboard.py`):** Caching loaders, sub-10ms policy simulation, PSI calculations, HTML generators, Plotly figure builders (10 tests).
* **Zero Leakage Gate (`tests/test_leakage.py`):** Asserts zero post-origination columns exist in processed splits (8 tests).
* **Temporal Partitions (`tests/test_splits.py`):** Asserts strict chronological year boundaries and default rate bounds (8 tests).
* **Feature Pipeline (`tests/test_features.py`):** Asserts single-applicant inference symmetry, output shapes `(1, 151)`, zero NaNs (7 tests).
* **Model Validation (`tests/test_models.py`):** Champion model bounds, Gini $> 0.25$, Brier $< 0.16$, Option Z ablation retention $> 95\%$ (9 tests).
* **Policy Engine (`tests/test_policy.py`):** Threshold ordering ($0 < \tau_{\text{low}} < \tau_{\text{high}} < 1$), monotonicity, expected loss math (6 tests).
* **Explainability (`tests/test_explainability.py`):** SHAP cache loading, top-3 reason generation, ECOA notice formatting (6 tests).
* **Environment Integrity (`tests/test_env.py`):** Import integrity and constant definitions (2 tests).

---

## 📂 Repository Structure

```
credit-risk-decision/
├── Dockerfile                     # Multi-stage python:3.12-slim with OpenMP and uv
├── docker-compose.yml             # API (:8000) & Dashboard (:8501) microservices orchestration
├── .dockerignore                  # Strict isolation excluding .venv, git, and 200MB+ raw splits
├── config/
│   ├── constants.py               # 35-column leakage blacklist, random seeds, paths
│   └── model_config.yaml          # Hyperparameters, split cutoffs, GPU acceleration configs
├── data/
│   ├── raw/                       # Immutable source data (accepted & rejected .csv.gz)
│   └── processed/                 # Compressed Parquet partitions (train, val, test, stress)
├── src/
│   ├── data/make_dataset.py       # Chunked ingestion, zero-leakage gate, parquet partitioner
│   ├── features/build_features.py # CreditFeaturePipeline & 151 scale-invariant ratios
│   ├── models/                    # Model tournament, Platt calibration, runs logging
│   ├── policy/decision_engine.py  # 3-band threshold optimization & Expected Loss calculator
│   ├── explainability/            # TreeSHAP explainer & statutory Adverse Action generator
│   └── monitoring/                # PSI feature drift & score stability engine
├── api/                           # Decoupled FastAPI scoring microservice
│   ├── main.py                    # Lifespan application, warmup pass, /health, /policy, /predict
│   └── schemas.py                 # Pydantic V2 input & response data contracts
├── dashboard/                     # Interactive 5-Tab Streamlit underwriting dashboard
│   ├── app.py                     # Main application entry point & theme management
│   ├── components/                # Cached loaders, theme tokens, cards & Plotly builders
│   └── tabs/                      # 5 modular tab renderers (Executive, Policy, Dossier, Stability, PSI)
├── tests/                         # 66 comprehensive pytest tests across 10 modules
├── reports/                       # Precomputed lightweight artifacts for sub-millisecond container rendering
└── models/                        # Serialized champion model, pipeline, and policy thresholds
```
