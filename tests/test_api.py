"""HTTP 接口：出入参契约、拒答透传与指标端点。

说明：app.api.chat 的模块级 pipeline 已在 conftest 打桩，
用例通过替换 pipeline.aask 注入端点行为，不触发任何真实依赖。
"""
from fastapi.testclient import TestClient

import app.api.chat as chat_module
from app.main import app

client = TestClient(app)

_OK_RESULT = {
    "answer": "年假 15 天 [1]",
    "grounded": True,
    "citations": [
        {"index": 1, "source": "员工手册.md", "score": 0.9, "snippet": "员工年假 15 天"}
    ],
    "cached": False,
    "retrieval_ms": 12.3,
    "latency_ms": 45.6,
}

_REFUSAL_RESULT = {
    "answer": "根据公司现有资料，无法回答该问题。",
    "grounded": False,
    "citations": [],
    "cached": False,
    "retrieval_ms": 3.0,
    "latency_ms": 3.5,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metrics_exposes_prometheus_counters():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "rag_requests_total" in r.text


def test_chat_happy_path(monkeypatch):
    async def fake_aask(question: str):
        assert question == "年假几天"
        return dict(_OK_RESULT)

    monkeypatch.setattr(chat_module.pipeline, "aask", fake_aask, raising=False)
    r = client.post("/v1/chat", json={"question": "年假几天"})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True
    assert body["answer"] == "年假 15 天 [1]"
    assert body["citations"][0]["source"] == "员工手册.md"
    assert body["latency_ms"] == 45.6


def test_chat_refusal_passes_through_grounded_false(monkeypatch):
    async def fake_aask(question: str):
        return dict(_REFUSAL_RESULT)

    monkeypatch.setattr(chat_module.pipeline, "aask", fake_aask, raising=False)
    r = client.post("/v1/chat", json={"question": "公司班车路线"})
    assert r.status_code == 200
    assert r.json()["grounded"] is False


def test_chat_rejects_empty_question():
    r = client.post("/v1/chat", json={"question": ""})
    assert r.status_code == 422


def test_chat_rejects_overlong_question():
    r = client.post("/v1/chat", json={"question": "长" * 2001})
    assert r.status_code == 422


def test_chat_rejects_missing_field():
    r = client.post("/v1/chat", json={"q": "年假几天"})
    assert r.status_code == 422


def test_chat_internal_error_returns_500(monkeypatch):
    async def boom(question: str):
        raise RuntimeError("下游故障")

    monkeypatch.setattr(chat_module.pipeline, "aask", boom, raising=False)
    strict = TestClient(app, raise_server_exceptions=False)
    r = strict.post("/v1/chat", json={"question": "年假几天"})
    assert r.status_code == 500
