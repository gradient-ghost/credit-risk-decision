"""FastAPI Real-Time Credit Risk Decision Scoring Engine.

Production-grade underwriting API delivering calibrated default probabilities,
expected loss valuations, automated 3-tier policy decisions, and regulatory-compliant
ECOA/FCRA adverse action reason attributions under strict sub-100ms latency SLAs.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (
    ApplicantPayload,
    HealthResponse,
    PolicyResponse,
    PredictionResponse,
)
from src.explainability.shap_explainer import CreditSHAPExplainer
from src.features.build_features import CreditFeaturePipeline
from src.policy.decision_engine import (
    BAND_AUTO_APPROVE,
    BAND_AUTO_REJECT,
    BAND_MANUAL_REVIEW,
    assign_decision_band,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("credit_risk_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup, resource preloading, engine warmup, and shutdown."""
    logger.info("Initializing Credit Risk Decision Scoring Engine...")
    app.state.start_time = time.time()

    model_dir = PROJECT_ROOT / "models"
    pipeline_path = model_dir / "feature_pipeline.joblib"
    champion_path = model_dir / "champion_model.joblib"
    features_path = model_dir / "feature_names.json"
    policy_path = model_dir / "policy_thresholds.json"

    # 1. Verify existence of required model artifacts
    for p in (pipeline_path, champion_path, features_path, policy_path):
        if not p.exists():
            logger.error("Required model artifact not found: %s", p)
            raise FileNotFoundError(f"Missing required artifact: {p}")

    # 2. Load model components into application state
    logger.info("Loading feature pipeline from %s...", pipeline_path)
    app.state.feature_pipeline: CreditFeaturePipeline = joblib.load(pipeline_path)

    logger.info("Loading calibrated champion model from %s...", champion_path)
    app.state.champion_model = joblib.load(champion_path)

    logger.info("Loading feature names from %s...", features_path)
    with open(features_path, "r", encoding="utf-8") as f:
        app.state.feature_names = json.load(f)

    logger.info("Loading policy decision thresholds from %s...", policy_path)
    with open(policy_path, "r", encoding="utf-8") as f:
        app.state.policy_thresholds: Dict[str, Any] = json.load(f)

    # 3. Initialize CreditSHAPExplainer for regulatory adverse action explanations
    logger.info("Initializing CreditSHAPExplainer...")
    app.state.shap_explainer = CreditSHAPExplainer(
        model_path=champion_path,
        feature_names_path=features_path,
    )

    # 4. Warm up inference pipeline and model to avoid cold-start latencies
    try:
        dummy_applicant = {
            "loan_amnt": 10000.0,
            "term_months": 36,
            "term": " 36 months",
            "int_rate": 10.0,
            "installment": 322.0,
            "annual_inc": 65000.0,
            "dti": 15.0,
            "home_ownership": "MORTGAGE",
            "verification_status": "Not Verified",
            "purpose": "debt_consolidation",
            "emp_length_years": 4.0,
            "emp_length": "4 years",
            "earliest_cr_line": "2010-01-01",
            "fico_score": 730.0,
            "fico_range_low": 730.0,
            "fico_range_high": 730.0,
            "open_acc": 8.0,
            "total_acc": 18.0,
            "revol_bal": 4500.0,
            "revol_util": 25.0,
            "inq_last_6mths": 0.0,
            "acc_open_past_24mths": 1.0,
        }
        X_warmup = app.state.feature_pipeline.transform([dummy_applicant])
        _ = app.state.champion_model.predict_proba(X_warmup)
        _ = app.state.shap_explainer.explain_applicant(X_warmup, top_n=3)
        logger.info("Inference engine warmed up successfully.")
    except Exception as exc:
        logger.warning("Cold-start warmup encountered non-fatal notice: %s", exc)

    logger.info("Scoring Engine fully operational.")
    yield

    logger.info("Shutting down Credit Risk Scoring Engine.")


app = FastAPI(
    title="Credit Risk Decision Scoring Engine",
    description=(
        "Production-grade automated credit underwriting API providing real-time "
        "probability of default scoring, expected loss calculations, policy tiering "
        "(AUTO_APPROVE, MANUAL_REVIEW, AUTO_REJECT), and ECOA/FCRA compliant adverse "
        "action reasons backed by Shapley attribution values."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and Readiness Check",
    tags=["System"],
)
async def health_check(request: Request) -> HealthResponse:
    """Verify that all scoring pipeline components, ML models, and policies are active in memory."""
    uptime = time.time() - getattr(request.app.state, "start_time", time.time())
    model_loaded = (
        hasattr(request.app.state, "champion_model")
        and request.app.state.champion_model is not None
    )
    pipeline_loaded = (
        hasattr(request.app.state, "feature_pipeline")
        and request.app.state.feature_pipeline is not None
    )
    policy_loaded = (
        hasattr(request.app.state, "policy_thresholds")
        and request.app.state.policy_thresholds is not None
    )

    is_healthy = model_loaded and pipeline_loaded and policy_loaded

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        model_loaded=model_loaded,
        feature_pipeline_loaded=pipeline_loaded,
        policy_thresholds_loaded=policy_loaded,
        version="1.0.0",
        uptime_seconds=round(uptime, 2),
    )


@app.get(
    "/policy",
    response_model=PolicyResponse,
    summary="Active Credit Policy Configuration",
    tags=["Policy"],
)
async def get_policy(request: Request) -> PolicyResponse:
    """Retrieve current credit risk policy thresholds, decision rules, and backtest portfolio metrics."""
    thresholds = getattr(request.app.state, "policy_thresholds", {})
    tau_low = float(thresholds.get("tau_low", 0.25))
    tau_high = float(thresholds.get("tau_high", 0.35))

    return PolicyResponse(
        tau_low=tau_low,
        tau_high=tau_high,
        auto_approve_rule=f"probability_of_default < {tau_low:.2f}",
        manual_review_rule=f"{tau_low:.2f} <= probability_of_default <= {tau_high:.2f}",
        auto_reject_rule=f"probability_of_default > {tau_high:.2f}",
        portfolio_test_metrics=thresholds.get("test_eval"),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score Loan Applicant",
    tags=["Underwriting"],
)
async def predict_credit_risk(
    payload: ApplicantPayload,
    request: Request,
) -> PredictionResponse:
    """Underwrite a loan application in real time.

    Computes:
    - Calibrated probability of default (PD)
    - Expected loss under LGD assumptions
    - 3-tier decision band (AUTO_APPROVE, MANUAL_REVIEW, AUTO_REJECT)
    - Top 3 adverse action factors via TreeSHAP attributions for reviews/rejections
    """
    t_start = time.perf_counter()

    try:
        pipeline = request.app.state.feature_pipeline
        model = request.app.state.champion_model
        thresholds = request.app.state.policy_thresholds
        explainer: CreditSHAPExplainer = request.app.state.shap_explainer
    except AttributeError as err:
        logger.error("Scoring engine accessed prior to component initialization: %s", err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring engine components are still initializing or unavailable.",
        ) from err

    tau_low = float(thresholds.get("tau_low", 0.25))
    tau_high = float(thresholds.get("tau_high", 0.35))
    lgd = float(thresholds.get("lgd_assumption", 1.0))

    # 1. Serialize Pydantic payload to dictionary
    applicant_dict = payload.model_dump()

    # 2. Feature engineering & transformation
    X_arr = pipeline.transform([applicant_dict])

    # 3. Model inference: Calibrated Probability of Default
    prob_arr = model.predict_proba(X_arr)[:, 1]
    pd_val = float(prob_arr[0])

    # 4. Expected Loss calculation ($)
    expected_loss = pd_val * float(payload.loan_amnt) * lgd

    # 5. Policy decision band assignment
    decision_bands = assign_decision_band(
        np.array([pd_val]),
        tau_low=tau_low,
        tau_high=tau_high,
    )
    decision_tier = str(decision_bands[0])

    # 6. Adverse Action Reason extraction via SHAP
    if decision_tier in (BAND_AUTO_REJECT, BAND_MANUAL_REVIEW):
        adverse_reasons, _ = explainer.explain_applicant(X_arr, top_n=3)
    else:
        adverse_reasons = []

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    return PredictionResponse(
        applicant_id=payload.applicant_id,
        probability_of_default=round(pd_val, 6),
        expected_loss=round(expected_loss, 2),
        loan_amount=float(payload.loan_amnt),
        decision_tier=decision_tier,
        policy_thresholds={"tau_low": tau_low, "tau_high": tau_high},
        adverse_action_reasons=adverse_reasons[:3],
        inference_latency_ms=round(elapsed_ms, 2),
        model_version="xgboost_calibrated_v1",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
