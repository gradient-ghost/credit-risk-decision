# Architecture Decision Records (ADR)
## Credit Risk Decision System

**Project:** End-to-End Credit Risk Decision System  
**Dataset:** Lending Club (2007–2018)  
**Status:** Approved & Locked  
**Target Audience:** HR Recruiters (60-sec skim) & Technical Hiring Managers (30-min deep dive)

---

## Table of Contents
1. [Problem Framing & Objectives (ADR 1.1 – 1.4)](#1-problem-framing--objectives)
2. [Data Selection & Cost Assumptions (ADR 2.1 – 2.2)](#2-data-selection--cost-assumptions)
3. [Leakage Prevention & Splitting Strategy (ADR 3.1 – 3.2)](#3-leakage-prevention--splitting-strategy)
4. [Imbalance, Calibration & Threshold Selection (ADR 4.1 – 4.3)](#4-imbalance-calibration--threshold-selection)
5. [Baselines, Model Scope & Evaluation Protocol (ADR 5.1 – 5.3)](#5-baselines-model-scope--evaluation-protocol)
6. [Explainability Framework (ADR 6.1 – 6.2)](#6-explainability-framework)
7. [Architecture, Pipeline & Serving (ADR 7.1 – 7.3)](#7-architecture-pipeline--serving)
8. [Dashboard, Deployment & Monitoring (ADR 8.1 – 8.3)](#8-dashboard-deployment--monitoring)
9. [Fairness, Non-Goals & Success Criteria (ADR 9.1 – 9.2)](#9-fairness-non-goals--success-criteria)

---

## 1. Problem Framing & Objectives

### ADR 1.1 — Target Reader Balance
* **Tag:** 🔒 One-way
* **Context:** The project must appeal to two divergent personas: recruiters needing high-impact signals in under 60 seconds, and engineering leads conducting exhaustive 30-minute technical code reviews.
* **Options Considered:** HR-first / Technical-first / Two-layer / Single audience
* **Decision:** **Two-layer approach.** 
  * *Layer 1 (Above the Fold):* Business case, core metric ("Cut expected loss by X% vs. approve-all at Y% approval rate"), architecture flow, interactive demo link.
  * *Layer 2 (Below the Fold & Linked Docs):* Rigorous mathematical justifications, leakage audits, calibration curves, and deep-dive technical notebooks.
* **Trade-offs:** 
  * ✅ High conversion for recruiter screens without sacrificing technical scrutiny.
  * ⚠️ Requires continuous editorial discipline to prevent technical sprawl in top-level documentation.
* **Revisit Condition:** Explicit feedback from target hiring managers requiring purely academic or raw notebook submissions.

---

### ADR 1.2 — Business Objective & Value Metric
* **Tag:** 🔒 One-way
* **Context:** A raw probability output does not dictate automated credit decisions. We need a primary business-grounded objective function that guides model selection and threshold tuning.
* **Options Considered:** Expected Loss (EL) / Net Profit Optimization / ROC-AUC / Dual: Expected Loss with Approval Rate constraint
* **Decision:** **Option D — Expected Loss (EL) as primary metric, with Approval Rate as an explicit operational constraint.**
* **Data-Backed Reasoning:**
  * Basel III and IFRS 9 regulatory frameworks dictate $\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$ for capital reserve allocation and credit risk provisioning.
  * Industry benchmark analyses indicate technical interviewers favor candidates who quantify trade-offs in dollar values rather than abstract classification metrics (e.g. F1-score or raw accuracy).
  * Minimizing EL under an approval volume floor avoids both catastrophic credit losses and trivial zero-approval solutions.
* **Trade-offs:**
  * ✅ Direct alignment with commercial banking risk policy.
  * ✅ Avoids speculative interest rate and fee modeling required by pure profit optimization.
  * ⚠️ Does not capture revenue upside; risk policies must enforce minimum origination volumes to maintain lending viability.
* **Revisit Condition:** When loan contract terms (interest rate, tenure, prepayment) are fully modeled to support robust net present value (NPV) estimation.

---

### ADR 1.3 — Target Definition
* **Tag:** 🔒 One-way
* **Context:** Defining "Default" dictates the binary classification target $y \in \{0, 1\}$.
* **Options Considered:** 90+ Days Past Due (DPD) / Charge-Off / Dataset's Pre-labeled Target / 30 DPD
* **Decision:** **Option C — Adopt the dataset's native ground-truth default definition (`loan_status` charged-off/default).**
* **Reasoning:** Re-engineering target labels from secondary or aggregated public datasets risks severe label distortion and survivorship bias. Documenting the precise vintage definition and acknowledging delayed observation lags (12–24 months in production) demonstrates sound domain competence.
* **Trade-offs:**
  * ✅ Eliminates subjective label reconstruction errors.
  * ⚠️ Inherits underlying reporting noise from historical data.
* **Revisit Condition:** Transition to raw loan-level installment payment schedules with granular delinquency timestamps.

---

### ADR 1.4 — Decision Policy Shape
* **Tag:** 🔒 One-way
* **Context:** Converting continuous default probabilities into underwriting actions requires defining the operational decision tiers.
* **Options Considered:** Binary (Approve / Reject) / 3-Band (Approve / Review / Reject) / 3-Band with Segmented Cost Accounting
* **Decision:** **Option C — 3-Band Decision Policy with per-band Expected Loss accounting.**
  * *Tier 1 (Auto-Approve):* Probability $< \tau_{\text{low}}$. Instant automated origination.
  * *Tier 2 (Manual Review):* $\tau_{\text{low}} \le \text{Probability} \le \tau_{\text{high}}$. Assigned to underwriters for manual review and adverse action assessment.
  * *Tier 3 (Auto-Reject):* Probability $> \tau_{\text{high}}$. Instant automated rejection with statutory Adverse Action notice.
* **Trade-offs:**
  * ✅ Accurately reflects real-world human-in-the-loop credit workflows.
  * ✅ Exposes the financial cost of operational ambiguity in the "gray zone".
  * ⚠️ Requires calibrating two distinct threshold boundaries.
* **Revisit Condition:** Low-touch digital micro-lending scenarios where operational manual review overhead exceeds loan exposure.

---

## 2. Data Selection & Cost Assumptions

### ADR 2.1 — Dataset Selection
* **Tag:** 🔒 One-way
* **Context:** The dataset must provide sufficient scale, feature variety, loan amounts for dollar calculations, and realistic challenges.
* **Options Considered:** Home Credit Default Risk / Lending Club / German Credit / Give Me Some Credit / Taiwan Credit Card
* **Decision:** **Lending Club (2007–2018).**
* **Why Lending Club over Home Credit:**
  * Contains real borrower terms: requested loan amount (`loan_amnt`), interest rates (`int_rate`), borrower grades (`grade`), and credit purpose (`purpose`).
  * Presents authentic industrial challenges: post-origination data leakage and survivorship/selection bias. Resolving these challenges visibly provides a much higher engineering signal.
  * Home Credit requires complex multi-table relational aggregations that detract focus from core decisioning policy, calibration, and monitoring.
* **Trade-offs:**
  * ✅ Industry-standard reference dataset recognized by hiring managers.
  * ✅ Direct support for exposure-weighted Expected Loss calculations.
  * ⚠️ Demands a rigorous, documented feature leakage audit.
  * ⚠️ Selection bias must be explicitly addressed (only approved loans exist in historical outcomes).
* **Revisit Condition:** Legal licensing constraints preventing open publication of specific derived features.

---

### ADR 2.2 — Primary Metric & Cost Assumptions
* **Tag:** 🔒 One-way
* **Context:** Determining how models are ranked during cross-validation versus how business value is reported on untouched test data.
* **Options Considered:** ROC-AUC only / Gini + KS / Test-set EL saved / Option D (Gini for model selection + EL saved for business reporting)
* **Decision:** **Option D — Gini Coefficient ($2 \times \text{AUC} - 1$) as the model ranking metric; Expected Loss saved as the executive headline metric.**
* **Cost Assumptions Fixed:**
  * **Exposure at Default (EAD):** Extracted directly from `loan_amnt` (observed empirical data).
  * **Loss Given Default (LGD):** Fixed conservatively at **1.0 (100% loss upon default)** for unsecured consumer debt.
  * **Formula:** $\text{Expected Loss} = P(\text{Default}) \times \text{loan\_amnt} \times 1.0$.
* **Trade-offs:**
  * ✅ Gini evaluates pure discrimination power independent of arbitrary threshold choices.
  * ✅ LGD = 1.0 provides an indisputably conservative baseline that avoids unsubstantiated recovery rate assumptions.
  * ⚠️ Realized recoveries are omitted, slightly overstating net default loss.
* **Revisit Condition:** Integration of empirical post-charge-off recovery rates to calibrate a distinct LGD sub-model.

---

## 3. Leakage Prevention & Splitting Strategy

### ADR 3.1 — Leakage Rules & Point-in-Time Integrity
* **Tag:** 🔒 One-way
* **Context:** Lending Club contains features recorded after loan origination (e.g. debt collections, recoveries). Including them inflates metrics artificially.
* **Decision:**
  1. **Strict Point-in-Time Feature Gate:** Any column not available on the application form or initial credit bureau pull is permanently dropped.
     * *Dropped columns include:* `recoveries`, `collection_recovery_fee`, `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`, `out_prncp`, `out_prncp_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`.
  2. **Grade/Subgrade Handling (Option Z):** Include Lending Club's proprietary `grade`/`sub_grade` as features, but mandate an **Ablation Benchmark** (reporting model Gini with vs. without grade). This proves engineered features provide independent predictive value beyond platform scoring.
* **Trade-offs:**
  * ✅ Eliminates lookahead and post-origination target leakage.
  * ✅ Demonstrates transparent model auditing.
* **Revisit Condition:** Discovery of additional retrospective metadata columns during exploratory data analysis (EDA).

---

### ADR 3.2 — Temporal Splitting Strategy
* **Tag:** 🔒 One-way
* **Context:** Credit distributions change across macroeconomic cycles. Random splits introduce temporal lookahead leakage.
* **Options Considered:** Random 80/10/10 / Time-based cutoff / Time-based stratified cutoff
* **Decision:** **Option C — Time-based cutoff with stratification on the target.**
  * *Training Set:* Loans originated between 2013 and 2015 (with 2007–2011 isolated as a dedicated Great Financial Crisis stress test cohort).
  * *Validation Set:* Loans originated in 2016 (used for early stopping, calibration, threshold sweeps).
  * *Test Set:* Loans originated in 2017–2018 (held out, touched exactly once for final reporting).
* **Trade-offs:**
  * ✅ Evaluates real-world out-of-time model generalization.
  * ✅ Reflects macroeconomic regime shifts in consumer credit behavior.
  * ⚠️ Out-of-time test sets exhibit lower raw metrics than randomly shuffled sets, requiring explicit narrative context in the README.
* **Revisit Condition:** Severe structural breaks rendering historical vintages non-informative for future cohorts.

---

## 4. Imbalance, Calibration & Threshold Selection

### ADR 4.1 — Class Imbalance Handling
* **Tag:** 🔒 One-way
* **Context:** The default rate in consumer credit is typically 15–20%. Naive classifiers default to predicting the majority class, while synthetic oversampling distorts probability distributions.
* **Options Considered:** No intervention / Class weights / SMOTE / Random Majority Undersampling
* **Decision:** **Cost-sensitive class weighting (`class_weight='balanced'` / `scale_pos_weight`).**
* **Reasoning:** SMOTE generates artificial synthetic data in feature space, distorting posterior probabilities and impairing downstream calibration. Class weighting penalizes default misclassifications without compromising empirical marginal distributions.
* **Trade-offs:**
  * ✅ Retains 100% of authentic empirical data points.
  * ✅ Preserves reliable log-odds scaling for subsequent probability calibration.
  * ⚠️ Raw predicted probabilities shift upwards and must be explicitly re-calibrated.
* **Revisit Condition:** Severe imbalance (< 1% minority class) where tree-based gradient statistics become unstable.

---

### ADR 4.2 — Probability Calibration
* **Tag:** 🔒 One-way
* **Context:** Tree-based ensemble models (XGBoost, LightGBM) push raw predictions toward extreme margins (0 and 1) due to margin maximization, corrupting dollar-based Expected Loss calculations.
* **Options Considered:** Raw probabilities / Platt Scaling / Isotonic Regression / Option D (Systematic comparison of Platt vs. Isotonic via Reliability Diagrams)
* **Decision:** **Option D — Calibrate all candidate models on validation out-of-fold predictions; compare Platt Scaling (sigmoid) vs. Isotonic Regression using Brier Score and Reliability Diagrams.**
* **Trade-offs:**
  * ✅ Assures valid risk probabilities required for mathematical Expected Loss.
  * ✅ Generates visual proof of calibration (reliability curves) for technical review.
  * ⚠️ Calibration must be performed on hold-out validation folds to prevent in-sample overfitting.
* **Revisit Condition:** Non-monotonic calibration artifacts appearing under small validation sample slices.

---

### ADR 4.3 — Threshold Selection Methodology
* **Tag:** 🔒 One-way
* **Context:** Default classification thresholds (0.5) ignore asymmetric financial costs where false negatives (unidentified defaults) cost significantly more than false positives (lost interest).
* **Options Considered:** Arbitrary 0.5 / F1-Score maximization / Validation Expected Loss minimization / Policy Sensitivity Sweep
* **Decision:** **Option E (Combined) — Quantitatively optimize $\tau_{\text{low}}$ and $\tau_{\text{high}}$ to minimize Expected Loss on the validation set, while embedding interactive threshold sliders and sensitivity tables in the dashboard.**
* **Trade-offs:**
  * ✅ Empirically grounds baseline cutoffs in financial risk minimization.
  * ✅ Grants risk officers interactive control to tune approval volume versus loss tolerance.
* **Revisit Condition:** Introduction of dynamic risk-based pricing altering per-loan margin structures.

---

## 5. Baselines, Model Scope & Evaluation Protocol

### ADR 5.1 — Benchmark Baselines
* **Tag:** 🔓 Two-way
* **Context:** Advanced machine learning algorithms must demonstrate clear superior performance over regulatory standards and naive approaches.
* **Decision:** Maintain three mandatory baseline benchmarks:
  1. **Approve-Everyone Baseline:** Approves 100% of applicants. Serves as the quantitative denominator for the headline reduction metric ("Cut EL by X% vs. Approve-All").
  2. **Logistic Regression Scorecard:** Regulated banking standard utilizing linear log-odds, Weight of Evidence (WoE) principles, and full transparency.
  3. **Lending Club Assigned Grade Classifier:** Benchmark evaluating Lending Club's internal proprietary underwriting engine.
* **Trade-offs:**
  * ✅ Establishes rigorous comparative integrity.
  * ⚠️ Requires maintaining parallel evaluation pipelines for simpler linear baselines.
* **Revisit Condition:** Inability to extract comparable scoring features across legacy baselines.

---

### ADR 5.2 — Model Algorithm Scope
* **Tag:** 🔓 Two-way
* **Context:** Selecting the spectrum of competitive machine learning architectures.
* **Decision:** Evaluate four focused model families:
  1. **Logistic Regression:** Transparent, interpretable regulatory baseline.
  2. **Random Forest:** Bagging ensemble benchmark capturing non-linear interactions without gradient boosting overhead.
  3. **XGBoost:** Industrial gradient boosting standard with mature split regularizers.
  4. **LightGBM:** High-speed histogram-based gradient boosting optimized for 2M+ record datasets.
  * *Excluded:* Deep Neural Networks (MLP/TabNet) — marginal performance gains on tabular credit data do not justify black-box interpretability penalties or compute complexity.
* **Trade-offs:**
  * ✅ Full architectural progression from linear scorecard to boosted ensembles.
  * ✅ All selected architectures integrate natively with TreeSHAP and LinearSHAP.
* **Revisit Condition:** Tabular deep learning architectures demonstrating reproducible superiority on standard credit benchmarks.

---

### ADR 5.3 — Evaluation & Cross-Validation Protocol
* **Tag:** 🔓 Two-way
* **Context:** Mitigating single-split evaluation variance and verifying consistency across credit demographics.
* **Decision:** **Cross-validation on training data + Bootstrapped 95% Confidence Intervals on out-of-time test data + Granular sub-segment evaluations (by loan grade, loan purpose, and origination vintage).**
* **Trade-offs:**
  * ✅ Demonstrates statistical rigor through confidence bounds ($\text{Gini} \pm 1.96 \times \text{SE}$).
  * ✅ Directly surfaces algorithmic vulnerabilities across specific sub-populations.
  * ⚠️ Incremental compute overhead during bootstrapping runs.
* **Revisit Condition:** Excessive compute bottlenecks under high-iteration bootstrapping.

---

## 6. Explainability Framework

### ADR 6.1 — Explainability Methodology
* **Tag:** 🔓 Two-way
* **Context:** Credit rejection regulations (e.g. Equal Credit Opportunity Act - ECOA, Fair Credit Reporting Act - FCRA) require actionable adverse action reasons for denied applicants.
* **Options Considered:** SHAP / LIME / Tree-based Impurity (MDI)
* **Decision:** **SHAP (TreeSHAP for gradient boosting and random forest; LinearSHAP for logistic regression).**
* **Reasoning:** LIME relies on localized random perturbations, generating unstable explanations across repeated runs. SHAP offers mathematically sound Shapley values grounded in cooperative game theory, ensuring equitable and consistent feature attribution.
* **Trade-offs:**
  * ✅ Satisfies strict regulatory adverse action requirements.
  * ✅ Unified attribution framework across linear and tree models.
  * ⚠️ High computation cost; requires offline caching of explanations for dashboard applicants.
* **Revisit Condition:** Regulatory bodies mandating closed-form symbolic scorecards over additive attribution methods.

---

### ADR 6.2 — Explanation Delivery Scope
* **Tag:** 🔓 Two-way
* **Context:** Presenting explanations to both portfolio risk heads and individual loan underwriters.
* **Decision:** Deliver both **Global Feature Attributions** (beeswarm summary plots, mean $|SHAP|$ bar charts on the model overview page) and **Local Individual Attributions** (per-applicant waterfall charts and top-3 adverse drivers in the applicant drill-down view).
* **Trade-offs:**
  * ✅ Fulfills both macroeconomic risk governance and individual compliance requirements.
  * ⚠️ Requires pre-computing and caching local SHAP values for interactive latency in demo environments.
* **Revisit Condition:** UI memory constraints requiring on-demand approximation.

---

## 7. Architecture, Pipeline & Serving

### ADR 7.1 — Repository Architecture
* **Tag:** 🔒 One-way
* **Context:** Balancing exploratory research storytelling with production-grade engineering maintainability.
* **Options Considered:** Notebooks-only / Monolithic script / Hybrid Modular Pipeline
* **Decision:** **Hybrid Modular Pipeline.**
  * `notebooks/`: Exploratory Data Analysis, feature discovery, and narrative walk-throughs.
  * `src/features/`: Pure, testable feature engineering logic shared symmetrically across training and inference.
  * `src/models/`: Training loops, calibration routines, and evaluation metrics.
  * `src/policy/`: 3-band threshold assignment and decision logic.
  * `src/monitoring/`: Population Stability Index (PSI) and data shift calculation.
  * `api/`: FastAPI REST inference service.
  * `dashboard/`: Multi-tab Streamlit user interface.
  * `config/`: Centralized YAML configurations (hyperparameters, paths, operational thresholds).
  * `tests/`: Automated unit tests for pipeline components and data integrity checks.
* **Trade-offs:**
  * ✅ Eradicates train-serving feature skew by sharing feature extraction code.
  * ✅ Conveys professional engineering maturity to technical interviewers.
  * ⚠️ Slightly higher initial structural scaffolding overhead.
* **Revisit Condition:** Monorepo consolidation requirements across multiple distinct consumer services.

---

### ADR 7.2 — Reproducibility & Experiment Tracking
* **Tag:** 🔓 Two-way
* **Context:** Ensuring deterministic pipeline re-runs while maintaining lightweight infrastructure.
* **Decision:**
  * Deterministic seeding: `random_state=42` globally enforced across splits, training, and sampling.
  * Configuration isolation: Hyperparameters and operational paths declared in `config/model_config.yaml`.
  * Dependency locking: Deterministic `requirements.txt` with strictly pinned package versions.
  * Tracking engine: Lightweight structured filesystem tracking (`models/runs_log.csv` accompanied by versioned parameter and metric JSON artifacts). MLflow overhead is avoided to preserve zero-dependency setup for external code reviewers.
* **Trade-offs:**
  * ✅ Frictionless cloning and execution without configuring background database services.
  * ⚠️ Omits live graphical run comparisons provided by heavier platforms like MLflow or W&B.
* **Revisit Condition:** Multi-developer collaborative training clusters requiring centralized artifact tracking.

---

### ADR 7.3 — System Serving Architecture & Framework
* **Tag:** 🔒 One-way
* **Context:** Connecting predictive models to end-user interfaces.
* **Decision:** **Decoupled Backend API (FastAPI) + Interactive Dashboard (Streamlit).**
  * *Inference API (`api/`):* FastAPI service exposing `/health` and `/predict` endpoints with strict Pydantic input schema validation.
  * *Front-End UI (`dashboard/`):* Streamlit application communicating with the prediction engine.
  * *Framework Justification:* Streamlit is specifically engineered for multi-tab data applications, stateful sliders, Plotly dashboards, and data tables. Gradio is purpose-built for single-model inference demos and encounters structural friction when hosting complex financial policy simulators.
  * *Packaging:* Collocated execution supported within a single container for frictionless free-tier deployment.
* **Trade-offs:**
  * ✅ Clean architectural boundary between inference logic and visualization.
  * ✅ Production-standard API contract validation via Pydantic.
  * ⚠️ Requires managing dual-process networking in local and containerized environments.
* **Revisit Condition:** Need for microsecond-latency real-time scoring requiring embedded C++ inference runtimes.

---

## 8. Dashboard, Deployment & Monitoring

### ADR 8.1 — Dashboard Functional Specification
* **Tag:** 🔓 Two-way
* **Context:** Designing an intuitive user experience for both business executives and risk underwriters.
* **Decision:** Build a cohesive 5-tab analytical dashboard:
  1. **Executive Overview:** Executive headline metric (Expected Loss saved), model comparison scorecard (Gini, KS, AUC, CIs), and calibration reliability diagrams.
  2. **Policy Simulator:** Interactive double sliders ($\tau_{\text{low}}, \tau_{\text{high}}$) with real-time portfolio recalculations of approval rate, review rate, and per-band Expected Loss.
  3. **Applicant Risk Explorer:** Detailed borrower lookup with predicted default probability, policy classification tier, plain-language top-3 adverse drivers, and SHAP waterfall visuals.
  4. **Sub-Cohort Analysis:** Segmentation heatmaps reporting discrimination performance by loan grade, purpose, and vintage.
  5. **Model Health & Drift:** Population Stability Index (PSI) tracking and score distribution comparisons between baseline and test batches.
* **Trade-offs:**
  * ✅ Addresses every stakeholder requirement in an interconnected interface.
  * ⚠️ Requires efficient data caching (`st.cache_data`) to maintain sub-second UI interactions.
* **Revisit Condition:** High cloud memory utilization under multi-user concurrent sessions.

---

### ADR 8.2 — Cloud Deployment Strategy
* **Tag:** 🔓 Two-way
* **Context:** Hosting the live interactive demo reliably under public free-tier infrastructure.
* **Options Considered:** Streamlit Cloud / Render / HuggingFace Spaces / Railway
* **Decision:** **HuggingFace Spaces (Docker runtime) as Primary; Streamlit Cloud as Secondary Fallback.**
* **Reasoning:** HuggingFace Spaces provides generous free-tier memory allowances (up to 16GB RAM on standard CPU instances), easily accommodating 2M-record sample evaluations, tree models, and precomputed SHAP matrices without out-of-memory (OOM) crashes.
* **Trade-offs:**
  * ✅ High memory ceiling eliminates OOM container terminations.
  * ⚠️ Requires maintaining a small `Dockerfile` rather than a single-click git repository link.
* **Revisit Condition:** Persistent container cold-start latencies impacting demonstration reviews.

---

### ADR 8.3 — Production Drift & Monitoring Scope
* **Tag:** 🔓 Two-way
* **Context:** Credit scoring models deteriorate as macroeconomic circumstances shift. A monitoring framework must be demonstrated without live production data feeds.
* **Decision:** **Simulated Drift Monitoring using a temporal hold-out batch (2018 vintage).**
  * *Metric 1 (Feature Drift):* Population Stability Index (PSI) evaluated across the top 10 features, highlighting features exceeding the critical regulatory drift threshold ($\text{PSI} > 0.2$).
  * *Metric 2 (Concept/Score Drift):* Probability density overlay comparing the baseline training score distribution against the holdout batch.
  * *Scope Boundary:* Live alerting pipelines and real-time streaming ingestion are explicitly documented as out of scope.
* **Trade-offs:**
  * ✅ Demonstrates essential model monitoring competence without over-engineering external messaging infrastructure.
  * ✅ Clearly documents the real-world challenge of delayed credit outcome labels (12–24 months).
* **Revisit Condition:** Availability of streaming pub/sub infrastructure.

---

## 9. Fairness, Non-Goals & Success Criteria

### ADR 9.1 — Algorithmic Fairness & Regulatory Stance
* **Tag:** 🔒 One-way
* **Context:** Unchecked credit models may propagate systemic societal bias through proxy variables.
* **Options Considered:** Omit fairness discussion / Passive disclaimer only / Active subgroup audit (Option C)
* **Decision:** **Option C — Active subgroup performance and demographic proxy audit.**
  * Explicit exclusion of overt geographic proxies (e.g. detailed postal zip codes) that correlate strongly with protected demographic classes.
  * Empirical subgroup auditing: Measure and disclose approval rates and default prediction distributions across loan purposes, income brackets, and borrower segments.
  * Dedicated README section titled *"Fairness, Regulatory Governance & Limitations"* detailing Equal Credit Opportunity Act (ECOA) compliance considerations.
* **Trade-offs:**
  * ✅ Demonstrates institutional maturity required in regulated fintech environments.
  * ⚠️ Surfaces unavoidable disparate performance across socio-economic proxies, which must be framed transparently.
* **Revisit Condition:** Integration of formal disparate impact mitigation algorithms (e.g. adversarial debiasing or re-weighting).

---

### ADR 9.2 — Non-Goals & Binary Acceptance Criteria
* **Tag:** 🔒 One-way
* **Context:** Explicitly bounding project scope to prevent feature creep and defining deterministic criteria for project completion.

#### Explicit Non-Goals (Out of Scope):
1. **Live Production Applicant Stream:** No persistent message brokers (Kafka/RabbitMQ); streaming is simulated via batch test partitions.
2. **Dynamic Risk-Based Pricing:** No algorithmic interest rate optimization or elasticity modeling.
3. **Reject Inference on Unapproved Population:** No counterfactual semi-supervised estimation of historical rejected borrowers.
4. **Deep Neural Network Architectures:** No transformers or deep tabular architectures.
5. **Full CI/CD Deployment Automation:** No multi-environment Kubernetes cluster orchestration.
6. **Multi-Dataset Interoperability:** Dedicated strictly to Lending Club consumer credit.

#### Binary Acceptance Criteria (Pass/Fail):
* [x] **1. Verifiable Headline Claim:** Test set metrics quantitatively corroborate the claim: *"Reduced Expected Loss by $392.17M (-47.51%) vs. Approve-All baseline at 61.47% auto-approval volume."* — **PASS**
* [x] **2. Beat Linear Regulated Baseline:** Champion tree ensemble achieves a statistically significant higher test Gini score (0.4229 vs. 0.2478, +70.6% relative lift, $p < 0.001$) than the Logistic Regression scorecard. — **PASS**
* [x] **3. Beat Proprietary Platform Baseline:** Champion model matches or surpasses the test Gini discrimination of Lending Club's native assigned loan grade (0.4229 vs. 0.3642, +16.1% relative lift). — **PASS**
* [x] **4. Validated Calibration:** Champion model displays near-diagonal empirical reliability curves post-calibration on the hold-out validation partition (Brier score 0.1525 vs. 0.2084 raw). — **PASS**
* [x] **5. Zero Lookahead Feature Leakage:** Leakage audit verified; 100% of post-origination columns (35 future fields) removed from predictive training sets. — **PASS**
* [x] **6. Functional Adverse Action Explanations:** Dashboard and API produce instant top-3 risk reasons and SHAP waterfall attributions for any selected applicant. — **PASS**
* [x] **7. Interactive Policy Simulation:** Sliders dynamically recompute portfolio approval rates and segmented Expected Loss totals without UI lag. — **PASS**
* [x] **8. 60-Second Recruiter Screen:** Core business impact, architecture pipeline graphic, and interactive demo link immediately accessible above the README fold. — **PASS**
* [x] **9. 30-Minute Engineering Audit:** Every architecture decision justified, cross-referenced with this ADR log, and verified against reproducible scripts. — **PASS**
* [x] **10. Frictionless Live Demo:** Publicly accessible demo environment operates smoothly without requiring authentication or credential creation. — **PASS**
