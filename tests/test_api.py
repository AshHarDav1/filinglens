"""The HTTP API: no real models and no network. Dependencies are swapped with
app.dependency_overrides; TestClient is used without `with`, so the lifespan (which creates real
model clients) does not run."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from filinglens.api.main import create_app, get_db


class BrokenSession:
    def scalar(self, *args, **kwargs):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))


def client_with(overrides: dict) -> TestClient:
    app = create_app()
    app.dependency_overrides.update(overrides)
    return TestClient(app)


def test_health_reports_the_database(db_session):
    response = client_with({get_db: lambda: db_session}).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["companies"] >= 1
    assert set(body["models"]) == {"answer", "guard", "embeddings"}


def test_health_is_503_when_the_database_is_down():
    response = client_with({get_db: BrokenSession}).get("/health")
    assert response.status_code == 503
    assert response.json()["database"] == "unreachable"


def test_openapi_describes_the_api():
    spec = TestClient(create_app()).get("/openapi.json").json()
    assert spec["info"]["title"] == "FilingLens API"
    assert "/health" in spec["paths"]


@pytest.mark.parametrize("path", ["/docs", "/redoc"])
def test_docs_pages_are_served(path):
    assert TestClient(create_app()).get(path).status_code == 200
