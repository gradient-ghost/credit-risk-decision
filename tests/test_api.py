"""Automated test suite for FastAPI Credit Risk Scoring Engine.

Validates operational health readiness, credit policy threshold contracts,
prime auto-approval workflows, subprime auto-rejections with regulatory adverse action notices,
borderline manual review routing, Pydantic V2 schema validations, and sub-100ms latency SLAs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from api.main import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Provide a lifespan-aware TestClient for the FastAPI scoring engine."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """Verify health endpoint returns status 200 with all components loaded."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["feature_pipeline_loaded"] is True
    assert data["policy_thresholds_loaded"] is True
    assert "version" in data
    assert data["uptime_seconds"] >= 0.0


def test_policy_endpoint(client: TestClient) -> None:
    """Verify policy endpoint returns active cutoffs matching policy_thresholds.json."""
    policy_file = PROJECT_ROOT / "models" / "policy_thresholds.json"
    with open(policy_file, "r", encoding="utf-8") as f:
        expected_policy = json.load(f)

    response = client.get("/policy")
    assert response.status_code == 200

    data = response.json()
    assert data["tau_low"] == expected_policy["tau_low"]
    assert data["tau_high"] == expected_policy["tau_high"]
    assert "auto_approve_rule" in data
    assert "manual_review_rule" in data
    assert "auto_reject_rule" in data
    assert data["portfolio_test_metrics"] is not None


def test_predict_prime_applicant_auto_approve(client: TestClient) -> None:
    """Verify prime applicant triggers AUTO_APPROVE tier with zero adverse action reasons."""
    prime_payload = {
        "applicant_id": "PRIME-001",
        "loan_amnt": 10000.0,
        "term_months": 36,
        "int_rate": 6.5,
        "installment": 306.0,
        "annual_inc": 150000.0,
        "dti": 5.0,
        "home_ownership": "MORTGAGE",
        "verification_status": "Not Verified",
        "purpose": "credit_card",
        "emp_length_years": 10.0,
        "fico_score": 800.0,
        "open_acc": 10.0,
        "total_acc": 25.0,
        "revol_bal": 2000.0,
        "revol_util": 10.0,
        "inq_last_6mths": 0.0,
        "acc_open_past_24mths": 0.0,
    }

    response = client.post("/predict", json=prime_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["applicant_id"] == "PRIME-001"
    assert data["decision_tier"] == "AUTO_APPROVE"
    assert data["probability_of_default"] < 0.25
    assert len(data["adverse_action_reasons"]) == 0
    assert data["expected_loss"] > 0.0
    assert data["expected_loss"] < data["loan_amount"] * 0.25
    assert data["model_version"] == "xgboost_calibrated_v1"


def test_predict_subprime_applicant_auto_reject(client: TestClient) -> None:
    """Verify high-risk applicant triggers AUTO_REJECT tier with 3 adverse action reasons."""
    subprime_payload = {
        "applicant_id": "SUBPRIME-999",
        "loan_amnt": 25000.0,
        "term_months": 60,
        "int_rate": 24.0,
        "installment": 720.0,
        "annual_inc": 25000.0,
        "dti": 38.0,
        "home_ownership": "RENT",
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "emp_length_years": 1.0,
        "fico_score": 660.0,
        "open_acc": 4.0,
        "total_acc": 10.0,
        "revol_bal": 18000.0,
        "revol_util": 88.0,
        "inq_last_6mths": 5.0,
        "acc_open_past_24mths": 6.0,
    }

    response = client.post("/predict", json=subprime_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["applicant_id"] == "SUBPRIME-999"
    assert data["decision_tier"] == "AUTO_REJECT"
    assert data["probability_of_default"] > 0.35
    assert len(data["adverse_action_reasons"]) == 3
    # Check that reasons are substantive descriptive text strings
    for reason in data["adverse_action_reasons"]:
        assert isinstance(reason, str)
        assert len(reason) > 10


def test_predict_manual_review_applicant(client: TestClient) -> None:
    """Verify borderline applicant triggers MANUAL_REVIEW tier with adverse explanations."""
    borderline_payload = {
        "applicant_id": "REVIEW-555",
        "loan_amnt": 15000.0,
        "term_months": 36,
        "int_rate": 17.5,
        "installment": 516.0,
        "annual_inc": 50000.0,
        "dti": 25.0,
        "home_ownership": "RENT",
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "emp_length_years": 3.0,
        "fico_score": 675.0,
        "open_acc": 7.0,
        "total_acc": 16.0,
        "revol_bal": 9000.0,
        "revol_util": 55.0,
        "inq_last_6mths": 2.0,
        "acc_open_past_24mths": 2.0,
    }

    response = client.post("/predict", json=borderline_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["applicant_id"] == "REVIEW-555"
    assert data["decision_tier"] == "MANUAL_REVIEW"
    assert 0.25 <= data["probability_of_default"] <= 0.35
    assert len(data["adverse_action_reasons"]) == 3


def test_predict_validation_errors(client: TestClient) -> None:
    """Verify Pydantic input validation rejects invalid fields with HTTP 422."""
    valid_base = {
        "loan_amnt": 10000.0,
        "term_months": 36,
        "int_rate": 12.0,
        "installment": 332.0,
        "annual_inc": 60000.0,
        "dti": 18.0,
    }

    # 1. Negative loan_amnt
    bad_loan = valid_base.copy()
    bad_loan["loan_amnt"] = -5000.0
    resp_loan = client.post("/predict", json=bad_loan)
    assert resp_loan.status_code == 422, f"Expected 422 for negative loan_amnt, got {resp_loan.status_code}"

    # 2. Invalid term_months (e.g. 12 instead of 36 or 60)
    bad_term = valid_base.copy()
    bad_term["term_months"] = 12
    resp_term = client.post("/predict", json=bad_term)
    assert resp_term.status_code == 422, f"Expected 422 for invalid term_months, got {resp_term.status_code}"

    # 3. Negative annual_inc
    bad_inc = valid_base.copy()
    bad_inc["annual_inc"] = -10000.0
    resp_inc = client.post("/predict", json=bad_inc)
    assert resp_inc.status_code == 422, f"Expected 422 for negative annual_inc, got {resp_inc.status_code}"


def test_predict_latency_under_100ms(client: TestClient) -> None:
    """Benchmark 10 consecutive requests and assert mean latency < 50ms and max latency < 100ms."""
    payload = {
        "loan_amnt": 12000.0,
        "term_months": 36,
        "int_rate": 9.5,
        "installment": 384.0,
        "annual_inc": 85000.0,
        "dti": 12.0,
        "home_ownership": "OWN",
        "verification_status": "Source Verified",
        "purpose": "debt_consolidation",
        "emp_length_years": 5.0,
        "fico_score": 750.0,
        "open_acc": 8.0,
        "total_acc": 20.0,
        "revol_bal": 4000.0,
        "revol_util": 25.0,
        "inq_last_6mths": 0.0,
        "acc_open_past_24mths": 1.0,
    }

    # Warmup request
    warmup_resp = client.post("/predict", json=payload)
    assert warmup_resp.status_code == 200

    latencies_ms = []
    for _ in range(10):
        t_start = time.perf_counter()
        resp = client.post("/predict", json=payload)
        t_end = time.perf_counter()
        assert resp.status_code == 200
        latencies_ms.append((t_end - t_start) * 1000.0)

    mean_latency = sum(latencies_ms) / len(latencies_ms)
    max_latency = max(latencies_ms)
    p95_latency = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]

    print(
        f"\n[LATENCY BENCHMARK] Mean: {mean_latency:.2f}ms | "
        f"Max: {max_latency:.2f}ms | p95: {p95_latency:.2f}ms"
    )

    assert (
        mean_latency < 50.0
    ), f"Mean latency {mean_latency:.2f}ms exceeded 50.0ms threshold"
    assert (
        max_latency < 100.0
    ), f"Max latency {max_latency:.2f}ms exceeded 100.0ms threshold"
