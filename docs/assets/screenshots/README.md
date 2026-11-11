# Credit Decision Engine — Dashboard & Visual UI Guide

> Comprehensive visual documentation and analytical reference for the institutional-grade Credit Risk Underwriting & Policy Intelligence platform.

---

## 🧭 Visual Interface Navigation

The dashboard provides a 5-tier analytical interface designed for both Chief Risk Officers (portfolio-level risk governance) and Credit Underwriters (individual loan adjudication). All views are rendered in high-contrast institutional Light Mode for presentation clarity and audit readiness.

```text
docs/assets/screenshots/
├── 01_executive_summary_kpis.png              # CRO Portfolio Health & Benchmark Scorecard
├── 02_policy_frontier_simulator.png           # Interactive Underwriting Cutoff Optimizer
├── 03_single_applicant_shap_waterfall.png     # Individual Underwriting & TreeSHAP Adverse Notice
├── 04_model_calibration_stress_testing.png    # Longitudinal Out-Of-Time Stability & Monotonicity
└── 05_basel_drift_monitoring_psi.png          # Production Population Stability Index (PSI) Drift
```

---

## 1. Executive Summary & Benchmark Scorecard

![Executive Summary](01_executive_summary_kpis.png)

### Purpose & Scope
Serves as the primary operational dashboard for Risk Committees and Executive Leadership, summarizing macroeconomic exposure, financial losses avoided through automated policy, and empirical benchmark superiority.

### Key Visual Components
1. **Headline Financial KPI Cards**:
   - **Portfolio Evaluated ($3.26B)**: Total credit exposure across 225,639 loans in the 2017–2018 walk-forward out-of-time test cohort.
   - **Expected Loss Avoided ($392.17M / -47.51%)**: Counterfactual default loss mitigated relative to an unconstrained Approve-All baseline ($825.49M exposure) by declining the 20.5% riskiest segment ($p > 0.35$).
   - **Auto-Approval Volume (61.47%)**: $1.78B in prime credit issued at an ultra-low **13.05% empirical default rate**.
   - **Underwriter Capacity Utilization (18.00%)**: $630.6M in borderline credit routed to human credit analysts, remaining strictly below the 20.0% operational ceiling.
2. **Champion Tournament Scorecard**:
   - Multi-metric benchmark proving the Champion Platt-Calibrated XGBoost model (Gini **0.4229**, ROC-AUC **0.7115**, Brier **0.1525**) achieves a **+70.6% relative discrimination lift** over the regulated linear Logistic Regression scorecard (Gini 0.2478, $p < 0.001$) and **+16.1% lift** over Lending Club's native subgrades (Gini 0.3642).
3. **Platt Reliability Curve**:
   - Validation reliability diagram demonstrating near-perfect alignment along the 45-degree diagonal, eliminating the probability distortion inherent in uncalibrated tree boosting.
4. **FICO 660 Selection Floor Callout**:
   - Transparent institutional warning highlighting the unwritten Lending Club hard floor (zero historical accepted loans below FICO 662), framing the necessity of explicit subprime policy guardrails.

---

## 2. Interactive Policy Frontier Simulator

![Policy Frontier Simulator](02_policy_frontier_simulator.png)

### Purpose & Scope
Enables credit risk officers and underwriting managers to interactively simulate alternative policy thresholds, evaluating the real-time financial trade-offs between origination volume, underwriter staffing capacity, and portfolio default losses.

### Key Visual Components
1. **Dual Threshold Sliders**:
   - **$\tau_{\text{low}}$ (Auto-Approve Cutoff)**: Set to $0.25$ — applicants with predicted default probability $p < \tau_{\text{low}}$ receive instantaneous approval.
   - **$\tau_{\text{high}}$ (Auto-Reject Cutoff)**: Set to $0.35$ — applicants with $p > \tau_{\text{high}}$ are rejected with statutory adverse action notices.
   - **Gray Zone ($\tau_{\text{low}} \le p \le \tau_{\text{high}}$)**: Automatically channeled into the Manual Review queue for underwriter adjudication.
2. **Ternary Decision Volume Distribution**:
   - Dynamic donut visualization displaying exact volume splits across the 225,639 test cohort: **61.47% Auto-Approve** (138,710 loans), **18.00% Manual Review** (40,619 loans), and **20.52% Auto-Reject** (46,310 loans).
3. **Dynamic Portfolio Reaction Cards**:
   - Real-time recalculation of portfolio Expected Loss, capital reserve requirements under Basel III, and operational underwriter load.
4. **Expected Loss vs. Approval Rate Frontier Curve**:
   - Non-linear trade-off frontier demonstrating diminishing returns: increasing approval volume beyond 65% triggers an exponential acceleration in Expected Loss as lower credit tiers are originated.

---

## 3. Individual Underwriting & TreeSHAP Adverse Action

![Single Applicant Underwriting](03_single_applicant_shap_waterfall.png)

### Purpose & Scope
Provides senior credit underwriters with an explainable, applicant-level dossier for loans referred to manual review or rejected under policy rules, fulfilling statutory compliance mandates under the **Equal Credit Opportunity Act (ECOA, 12 CFR § 1002)** and the **Fair Credit Reporting Act (FCRA, 15 U.S.C. § 1681)**.

### Key Visual Components
1. **Applicant Dossier & Credit Gauge**:
   - Profile summary (e.g., Application `LC-2017-TEST-0005`, requested loan amount $28,000, FICO 674, DTI 28.4%).
   - Calibrated Probability of Default gauge displaying **45.3%**, placing the borrower decisively in the **Tier 3: Auto-Reject** zone.
2. **Statutory Adverse Action Statement**:
   - Formatted credit evaluation notice detailing formal rejection reasons mapped from technical feature names into plain, actionable human language.
3. **TreeSHAP Feature Attribution Waterfall**:
   - Exact mathematical breakdown decomposing the borrower's log-odds score from the baseline training population expected value ($\mathbb{E}[f(x)] = 0.22$) to the applicant's predicted log-odds ($f(x) = 0.65$).
   - Explicit identification of the top risk-increasing features:
     - **Revolving Line Utilization (`revol_util` = 84.2%)**: $+0.21$ log-odds increase.
     - **Inquiries in Past 6 Months (`inq_last_6mths` = 4)**: $+0.14$ log-odds increase.
     - **Debt-to-Income Ratio (`dti` = 28.4%)**: $+0.08$ log-odds increase.
   - Defensive policy guard ensures that only features with strictly positive SHAP values ($\Delta > 0$) can be cited as adverse action rejection drivers.

---

## 4. Portfolio Stability & Longitudinal Stress Testing

![Portfolio Stability & Stress Testing](04_model_calibration_stress_testing.png)

### Purpose & Scope
Addresses model risk governance standards under **Federal Reserve SR 11-7 / OCC 2011-12**, proving model temporal stability across distinct economic regimes and validating credit grade monotonicity.

### Key Visual Components
1. **Walk-Forward Out-Of-Time (OOT) Degradation Scorecard**:
   - Tracks Gini discrimination across multi-year temporal regimes:
     - **2016 Validation (Expansion)**: Gini **0.4357**
     - **2017–2018 Holdout Test**: Gini **0.4229** (minimal degradation of -1.28 pp)
     - **2007–2011 Great Financial Crisis Stress Cohort**: Gini **0.3812** (retains 90.1% of discrimination under severe macroeconomic liquidity contraction)
2. **Subgrade Monotonicity Curve**:
   - Validates that empirical default rates strictly increase monotonically across Lending Club credit subgrades (A1 $\rightarrow$ G5). Confirms that the model does not produce inverted risk rank-ordering across adjacent risk tiers.
3. **Subgroup Calibration Spread Matrix**:
   - Reports empirical calibration accuracy across customer segments (Residential Tenancy: Mortgage, Rent, Own; Loan Purpose: Debt Consolidation, Credit Card, Small Business), highlighting pricing precision across heterogeneous segments.

---

## 5. Basel III Model Health & Population Drift Monitoring

![Basel III Model Health & Drift Monitoring](05_basel_drift_monitoring_psi.png)

### Purpose & Scope
Demonstrates production monitoring capabilities for ongoing model surveillance, detecting feature distribution shifts and score drift between baseline training populations and subsequent origination batches.

### Key Visual Components
1. **Model Governance Health Scorecard**:
   - **Discrimination (Gini 0.4229)**: GREEN status (target $\ge 0.40$).
   - **Calibration (Brier Score 0.1525)**: GREEN status (target $< 0.16$).
   - **Population Stability Index (PSI = 0.1461)**: AMBER status (signals moderate population shift between 2013–2015 baseline and 2018 holdout batch, alerting risk managers to evaluate recalibration).
2. **Top-10 Feature PSI Drift Breakdown**:
   - Bar chart quantifying the Population Stability Index across key predictive features.
   - Highlights that macroeconomic interest rates (`int_rate` PSI = 0.22) and debt consolidation loan amounts experienced the greatest distribution shifts as Federal Reserve benchmark rates tightened in 2017–2018.
3. **Score Distribution Overlay**:
   - Dual probability density overlay comparing training baseline scores against the out-of-time batch, providing visual evidence of score compression and rightward risk migration.
4. **Regulatory Governance Thresholds**:
   - Explicit traffic-light thresholds aligned with international credit risk practice:
     - $\text{PSI} < 0.10$: Minimal drift (no operational intervention).
     - $0.10 \le \text{PSI} \le 0.25$: Moderate drift (scheduled recalibration & threshold review).
     - $\text{PSI} > 0.25$: Significant structural break (mandatory model retraining & governance freeze).
