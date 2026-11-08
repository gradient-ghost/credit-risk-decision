"""Pydantic V2 schemas for Credit Risk Scoring Engine API.

Provides strongly-typed, fully validated request payloads and response contracts
for automated credit underwriting, policy configuration, and health monitoring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ApplicantPayload(BaseModel):
    """Underwriting input payload for individual loan applicants."""

    applicant_id: Optional[str] = Field(
        None,
        description="Optional tracking identifier for applicant or application.",
        json_schema_extra={"example": "APP-2026-9812"},
    )
    loan_amnt: float = Field(
        ...,
        gt=0.0,
        description="Requested loan principal amount in USD ($).",
        json_schema_extra={"example": 15000.0},
    )
    term_months: int = Field(
        36,
        description="Loan term in months (must be 36 or 60).",
        json_schema_extra={"example": 36},
    )
    int_rate: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Interest rate as percentage (0.0 to 100.0).",
        json_schema_extra={"example": 12.5},
    )
    installment: float = Field(
        ...,
        gt=0.0,
        description="Monthly installment payment in USD ($).",
        json_schema_extra={"example": 350.0},
    )
    annual_inc: float = Field(
        ...,
        gt=0.0,
        description="Self-reported annual gross income in USD ($).",
        json_schema_extra={"example": 75000.0},
    )
    dti: float = Field(
        ...,
        ge=0.0,
        description="Debt-to-income ratio (debt payments / monthly income * 100).",
        json_schema_extra={"example": 18.5},
    )
    home_ownership: str = Field(
        "RENT",
        description="Home ownership status (RENT, OWN, MORTGAGE, OTHER, ANY).",
        json_schema_extra={"example": "RENT"},
    )
    verification_status: str = Field(
        "Not Verified",
        description="Income verification status (Source Verified, Verified, Not Verified).",
        json_schema_extra={"example": "Not Verified"},
    )
    purpose: str = Field(
        "debt_consolidation",
        description="Stated purpose of the loan.",
        json_schema_extra={"example": "debt_consolidation"},
    )
    sub_grade: Optional[str] = Field(
        None,
        description="Lending Club internal sub-grade tier (e.g., A1, B3, C2).",
        json_schema_extra={"example": "B3"},
    )
    grade: Optional[str] = Field(
        None,
        description="Lending Club internal broad grade (e.g., A, B, C).",
        json_schema_extra={"example": "B"},
    )
    emp_length_years: float = Field(
        3.0,
        ge=0.0,
        le=50.0,
        description="Employment duration in years.",
        json_schema_extra={"example": 3.0},
    )
    emp_length: Optional[str] = Field(
        None,
        description="Lending Club raw employment string representation.",
        json_schema_extra={"example": "3 years"},
    )
    earliest_cr_line: Optional[str] = Field(
        "2010-01-01",
        description="Date the borrower's earliest reported credit line was opened.",
        json_schema_extra={"example": "2010-01-01"},
    )
    fico_score: Optional[float] = Field(
        None,
        ge=300.0,
        le=850.0,
        description="Primary credit score (FICO scale 300-850).",
        json_schema_extra={"example": 720.0},
    )
    fico_range_low: Optional[float] = Field(
        None,
        ge=300.0,
        le=850.0,
        description="Lower boundary of FICO score range.",
        json_schema_extra={"example": 715.0},
    )
    fico_range_high: Optional[float] = Field(
        None,
        ge=300.0,
        le=850.0,
        description="Upper boundary of FICO score range.",
        json_schema_extra={"example": 725.0},
    )
    open_acc: float = Field(
        5.0,
        ge=0.0,
        description="Number of open credit lines in the borrower's credit file.",
        json_schema_extra={"example": 5.0},
    )
    total_acc: float = Field(
        15.0,
        gt=0.0,
        description="Total number of credit lines currently in the borrower's credit file.",
        json_schema_extra={"example": 15.0},
    )
    revol_bal: float = Field(
        5000.0,
        ge=0.0,
        description="Total credit revolving balance.",
        json_schema_extra={"example": 5000.0},
    )
    revol_util: Union[float, str] = Field(
        35.0,
        description="Revolving line utilization rate (numeric or percentage string).",
        json_schema_extra={"example": 35.0},
    )
    inq_last_6mths: float = Field(
        0.0,
        ge=0.0,
        description="Number of inquiries in past 6 months (excluding auto and mortgage).",
        json_schema_extra={"example": 0.0},
    )
    acc_open_past_24mths: float = Field(
        1.0,
        ge=0.0,
        description="Number of trades opened in past 24 months.",
        json_schema_extra={"example": 1.0},
    )
    mths_since_last_delinq: Optional[float] = Field(
        None,
        description="The number of months since the borrower's last delinquency.",
        json_schema_extra={"example": None},
    )
    term: Optional[str] = Field(
        None,
        description="Raw term string format (e.g., ' 36 months').",
        json_schema_extra={"example": " 36 months"},
    )

    # Additional standard credit bureau fields with sensible defaults
    delinq_2yrs: Optional[float] = Field(0.0, ge=0.0)
    pub_rec: Optional[float] = Field(0.0, ge=0.0)
    pub_rec_bankruptcies: Optional[float] = Field(0.0, ge=0.0)
    tax_liens: Optional[float] = Field(0.0, ge=0.0)
    tot_coll_amt: Optional[float] = Field(0.0, ge=0.0)
    tot_cur_bal: Optional[float] = Field(None, ge=0.0)
    total_rev_hi_lim: Optional[float] = Field(None, ge=0.0)
    mort_acc: Optional[float] = Field(0.0, ge=0.0)
    num_actv_bc_tl: Optional[float] = Field(None, ge=0.0)
    num_actv_rev_tl: Optional[float] = Field(None, ge=0.0)
    num_bc_sats: Optional[float] = Field(None, ge=0.0)
    num_bc_tl: Optional[float] = Field(None, ge=0.0)
    num_il_tl: Optional[float] = Field(None, ge=0.0)
    num_op_rev_tl: Optional[float] = Field(None, ge=0.0)
    num_rev_accts: Optional[float] = Field(None, ge=0.0)
    num_sats: Optional[float] = Field(None, ge=0.0)
    pct_tl_nvr_dlq: Optional[float] = Field(None, ge=0.0, le=100.0)
    percent_bc_gt_75: Optional[float] = Field(None, ge=0.0, le=100.0)
    initial_list_status: str = Field("f", description="Initial listing status")
    application_type: str = Field("Individual", description="Application type")
    issue_d: Optional[str] = Field(None, description="Issue date if already originated")

    @field_validator("term_months")
    @classmethod
    def validate_term_months(cls, v: int) -> int:
        """Validate loan term is strictly 36 or 60 months."""
        if v not in (36, 60):
            raise ValueError(f"term_months must be 36 or 60, got {v}")
        return v

    @field_validator("home_ownership")
    @classmethod
    def validate_home_ownership(cls, v: str) -> str:
        """Validate home ownership matches supported categories."""
        allowed = {"RENT", "OWN", "MORTGAGE", "OTHER", "ANY"}
        cleaned = v.strip().upper()
        if cleaned not in allowed:
            raise ValueError(f"home_ownership must be one of {sorted(allowed)}, got {v}")
        return cleaned

    @model_validator(mode="after")
    def synchronize_derived_fields(self) -> "ApplicantPayload":
        """Handle FICO range synchronization and derive canonical text strings."""
        # 1. FICO score and range low/high synchronization
        if self.fico_score is not None:
            if self.fico_range_low is None:
                self.fico_range_low = float(self.fico_score)
            if self.fico_range_high is None:
                self.fico_range_high = float(self.fico_score)
        elif self.fico_range_low is not None and self.fico_range_high is not None:
            self.fico_score = (float(self.fico_range_low) + float(self.fico_range_high)) / 2.0
        elif self.fico_range_low is not None:
            self.fico_score = float(self.fico_range_low)
            self.fico_range_high = float(self.fico_range_low)
        elif self.fico_range_high is not None:
            self.fico_score = float(self.fico_range_high)
            self.fico_range_low = float(self.fico_range_high)

        # 2. Raw term string synchronization
        if self.term is None and self.term_months is not None:
            self.term = f" {self.term_months} months"

        # 3. Raw employment length string synchronization
        if self.emp_length is None and self.emp_length_years is not None:
            years = int(round(self.emp_length_years))
            if years <= 0:
                self.emp_length = "< 1 year"
            elif years == 1:
                self.emp_length = "1 year"
            elif years >= 10:
                self.emp_length = "10+ years"
            else:
                self.emp_length = f"{years} years"

        # 4. Infer grade from sub_grade if grade is omitted
        if self.sub_grade is not None and self.grade is None and len(self.sub_grade) > 0:
            self.grade = self.sub_grade[0].upper()

        return self

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "applicant_id": "APP-2026-9812",
                "loan_amnt": 15000.0,
                "term_months": 36,
                "int_rate": 12.5,
                "installment": 350.0,
                "annual_inc": 75000.0,
                "dti": 18.5,
                "home_ownership": "RENT",
                "verification_status": "Not Verified",
                "purpose": "debt_consolidation",
                "sub_grade": "B3",
                "emp_length_years": 3.0,
                "earliest_cr_line": "2010-01-01",
                "fico_score": 720.0,
                "open_acc": 5.0,
                "total_acc": 15.0,
                "revol_bal": 5000.0,
                "revol_util": 35.0,
                "inq_last_6mths": 0.0,
                "acc_open_past_24mths": 1.0,
                "mths_since_last_delinq": None,
            }
        },
    )


class PredictionResponse(BaseModel):
    """Credit underwriting scoring decision and regulatory adverse action response."""

    applicant_id: Optional[str] = Field(
        None,
        description="Applicant identifier if provided in request.",
        json_schema_extra={"example": "APP-2026-9812"},
    )
    probability_of_default: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated probability of default (PD) in [0.0, 1.0].",
        json_schema_extra={"example": 0.1245},
    )
    expected_loss: float = Field(
        ...,
        ge=0.0,
        description="Expected loss in USD ($) under LGD assumption (PD * loan_amount * LGD).",
        json_schema_extra={"example": 1867.5},
    )
    loan_amount: float = Field(
        ...,
        gt=0.0,
        description="Loan principal amount evaluated.",
        json_schema_extra={"example": 15000.0},
    )
    decision_tier: str = Field(
        ...,
        description="Automated underwriting tier (AUTO_APPROVE, MANUAL_REVIEW, or AUTO_REJECT).",
        json_schema_extra={"example": "AUTO_APPROVE"},
    )
    policy_thresholds: Dict[str, float] = Field(
        ...,
        description="Active policy decision cutoffs (tau_low, tau_high).",
        json_schema_extra={"example": {"tau_low": 0.25, "tau_high": 0.35}},
    )
    adverse_action_reasons: List[str] = Field(
        default_factory=list,
        description="Top principal adverse reasons for denial or review under ECOA/FCRA.",
        json_schema_extra={"example": []},
    )
    inference_latency_ms: float = Field(
        ...,
        description="Total round-trip inference execution latency in milliseconds.",
        json_schema_extra={"example": 24.5},
    )
    model_version: str = Field(
        ...,
        description="Identifier and version of champion scoring model.",
        json_schema_extra={"example": "xgboost_calibrated_v1"},
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "applicant_id": "APP-2026-9812",
                "probability_of_default": 0.1245,
                "expected_loss": 1867.5,
                "loan_amount": 15000.0,
                "decision_tier": "AUTO_APPROVE",
                "policy_thresholds": {"tau_low": 0.25, "tau_high": 0.35},
                "adverse_action_reasons": [],
                "inference_latency_ms": 24.5,
                "model_version": "xgboost_calibrated_v1",
            }
        }
    )


class HealthResponse(BaseModel):
    """Health status and component readiness report."""

    status: str = Field(
        ...,
        description="Overall service operational status ('healthy' or 'degraded').",
        json_schema_extra={"example": "healthy"},
    )
    model_loaded: bool = Field(
        ...,
        description="True if calibrated champion ML model is active in memory.",
        json_schema_extra={"example": True},
    )
    feature_pipeline_loaded: bool = Field(
        ...,
        description="True if fitted feature transformation pipeline is active in memory.",
        json_schema_extra={"example": True},
    )
    policy_thresholds_loaded: bool = Field(
        ...,
        description="True if policy decision thresholds are loaded and active.",
        json_schema_extra={"example": True},
    )
    version: str = Field(
        ...,
        description="API service version.",
        json_schema_extra={"example": "1.0.0"},
    )
    uptime_seconds: float = Field(
        ...,
        description="Service uptime in elapsed seconds since startup.",
        json_schema_extra={"example": 320.4},
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "feature_pipeline_loaded": True,
                "policy_thresholds_loaded": True,
                "version": "1.0.0",
                "uptime_seconds": 320.4,
            }
        }
    )


class PolicyResponse(BaseModel):
    """Current credit risk policy thresholds, decision logic rules, and portfolio metrics."""

    tau_low: float = Field(
        ...,
        description="Lower threshold cutoff (probabilities < tau_low are AUTO_APPROVE).",
        json_schema_extra={"example": 0.25},
    )
    tau_high: float = Field(
        ...,
        description="Upper threshold cutoff (probabilities > tau_high are AUTO_REJECT).",
        json_schema_extra={"example": 0.35},
    )
    auto_approve_rule: str = Field(
        ...,
        description="Mathematical rule defining auto-approval band.",
        json_schema_extra={"example": "probability_of_default < 0.25"},
    )
    manual_review_rule: str = Field(
        ...,
        description="Mathematical rule defining manual review band.",
        json_schema_extra={"example": "0.25 <= probability_of_default <= 0.35"},
    )
    auto_reject_rule: str = Field(
        ...,
        description="Mathematical rule defining auto-reject band.",
        json_schema_extra={"example": "probability_of_default > 0.35"},
    )
    portfolio_test_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Backtested portfolio performance and loss reduction metrics on test partition.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tau_low": 0.25,
                "tau_high": 0.35,
                "auto_approve_rule": "probability_of_default < 0.25",
                "manual_review_rule": "0.25 <= probability_of_default <= 0.35",
                "auto_reject_rule": "probability_of_default > 0.35",
                "portfolio_test_metrics": {
                    "total_applicants": 225639,
                    "auto_approve": {"pct_portfolio": 61.47, "realized_default_rate": 13.05},
                    "headline": {"pct_expected_loss_reduction": 47.51},
                },
            }
        }
    )
