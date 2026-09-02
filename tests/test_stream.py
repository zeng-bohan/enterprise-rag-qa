"""SSE 流式问答：事件协议（citations → token* → done）与异常降级。"""
import json

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.main import app

client = TestClient(app)


def _fake_astream(events):
    async def astream(question, kb_id=None, history=None):
        for e in events:
            yield e

    return astream


def _parse_sse(text: str):
    out = []
    for block in text.strip().split("\n\n"):
        ev, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        out.append((ev, data))
    return out


def test_stream_event_sequence(monkeypatch):
    events = [
        {"event": "citations", "data": {"citations": [{"index": 1, "source": "手册.md", "score": 0.9, "snippet": "年假"}]}},
        {"event": "token", "data": {"t": "年假"}},
        {"event": "token", "data": {"t": "15 天 [1]"}},
        {"event": "done", "data": {"grounded": True, "cached": False, "retrieval_ms": 12.0, "latency_ms": 88.0}},
    ]
    monkeypatch.setattr(deps.pipeline, "astream", _fake_astream(events), raising=False)
    r = client.post("/v1/chat/stream", json={"question": "年假几天"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    parsed = _parse_sse(r.text)
    assert [ev for ev, _ in parsed] == ["citations", "token", "token", "done"]
    assert parsed[0][1]["citations"][0]["source"] == "手册.md"
    done = parsed[-1][1]
    assert done["grounded"] is True
    assert "retrieval_ms" in done and "latency_ms" in done  # done 事件带分阶段耗时
    joined = "".join(d["t"] for ev, d in parsed if ev == "token")
    assert joined == "年假15 天 [1]"


def test_stream_refusal_keeps_protocol(monkeypatch):
    events = [
        {"event": "citations", "data": {"citations": []}},
        {"event": "token", "data": {"t": "根据公司现有资料，无法回答该问题。"}},
        {"event": "done", "data": {"grounded": False, "cached": False}},
    ]
    monkeypatch.setattr(deps.pipeline, "astream", _fake_astream(events), raising=False)
    r = client.post("/v1/chat/stream", json={"question": "班车路线"})
    parsed = _parse_sse(r.text)
    assert parsed[-1][1]["grounded"] is False


def test_stream_internal_error_emits_error_event(monkeypatch):
    async def boom(question, kb_id=None, history=None):
        raise RuntimeError("下游故障")
        yield  # pragma: no cover

    monkeypatch.setattr(deps.pipeline, "astream", boom, raising=False)
    r = client.post("/v1/chat/stream", json={"question": "年假几天"})
    parsed = _parse_sse(r.text)
    assert parsed[-1][0] == "error"


def test_stream_validation_still_enforced():
    r = client.post("/v1/chat/stream", json={"question": ""})
    assert r.status_code == 422
