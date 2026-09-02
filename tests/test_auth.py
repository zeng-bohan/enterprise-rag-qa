"""API Key 鉴权：API_KEYS 配置后 /v1 全部要求 X-API-Key；留空则完全开放。"""
import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.config import settings
from app.main import app

client = TestClient(app)


@pytest.fixture()
def auth_enabled(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "secret-1,secret-2", raising=False)
    yield
    monkeypatch.setattr(settings, "api_keys", "", raising=False)


def test_health_and_metrics_stay_open(auth_enabled):
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_missing_key_rejected(auth_enabled):
    r = client.post("/v1/chat", json={"question": "年假几天"})
    assert r.status_code == 401


def test_wrong_key_rejected(auth_enabled):
    r = client.post("/v1/chat", json={"question": "年假几天"}, headers={"X-API-Key": "bad"})
    assert r.status_code == 401


def test_valid_key_passes(auth_enabled, monkeypatch):
    async def fake_aask(question, **kwargs):
        return {"answer": "答", "grounded": True, "citations": [], "cached": False, "retrieval_ms": 1.0, "latency_ms": 2.0}

    monkeypatch.setattr(deps.pipeline, "aask", fake_aask, raising=False)
    r = client.post("/v1/chat", json={"question": "年假几天"}, headers={"X-API-Key": "secret-2"})
    assert r.status_code == 200


def test_no_config_means_open_access(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "", raising=False)
    r = client.post("/v1/chat", json={"question": ""})  # 未带 key：能进到参数校验（422 而非 401）
    assert r.status_code == 422
