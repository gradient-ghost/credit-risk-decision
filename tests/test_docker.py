"""Tests for Phase 8 Docker containerization and compose orchestration.

Validates:
1. Dockerfile configuration (base image, non-root user, security).
2. .dockerignore exclusions (preventing large raw data / venv leakage).
3. docker-compose.yml service topology and healthcheck bindings.
"""

from __future__ import annotations

from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_configuration():
    """Verify Dockerfile uses slim base, non-root user, and exposes standard ports."""
    dockerfile_path = PROJECT_ROOT / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile must exist at project root"

    content = dockerfile_path.read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in content, "Must use python:3.12-slim base image"
    assert "USER appuser" in content, "Must run as non-root appuser"
    assert "EXPOSE 8000 8501" in content or ("8000" in content and "8501" in content)
    assert "uv pip install" in content or "pip install" in content


def test_dockerignore_configuration():
    """Verify .dockerignore prevents shipping venv, raw data, and git metadata."""
    dockerignore_path = PROJECT_ROOT / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore must exist at project root"

    content = dockerignore_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    assert ".venv" in lines, "Must ignore host .venv"
    assert ".git" in lines, "Must ignore .git"
    assert any("data/raw" in l for l in lines), "Must exclude raw data"
    assert any("test_summary.parquet" in l for l in lines), "Must preserve test_summary.parquet"


def test_docker_compose_configuration():
    """Verify docker-compose.yml topology, service bindings, and healthchecks."""
    compose_path = PROJECT_ROOT / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml must exist at project root"

    with open(compose_path, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    assert "services" in compose_data
    services = compose_data["services"]
    assert "api" in services, "Must declare 'api' service"
    assert "dashboard" in services, "Must declare 'dashboard' service"

    # API configuration
    api_svc = services["api"]
    assert "8000:8000" in api_svc.get("ports", [])
    assert "healthcheck" in api_svc, "API service must have automated healthcheck"
    assert any("health" in str(arg) for arg in api_svc["healthcheck"].get("test", []))

    # Dashboard configuration
    dash_svc = services["dashboard"]
    assert "8501:8501" in dash_svc.get("ports", [])
    assert "depends_on" in dash_svc, "Dashboard must depend on API service health"
    assert dash_svc["depends_on"]["api"]["condition"] == "service_healthy"
