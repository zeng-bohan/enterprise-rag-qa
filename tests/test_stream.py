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
        if block.lstrip().startswith(":"):
            continue  # SSE 注释帧（心跳）按规范忽略，不算事件
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
    # 工单 19 把协议改成「error 之后必定还有一个 done 终止帧」：
    # 客户端因此可以只依赖一条收尾规则（看到 done 就结束），不必再特判 error 分支
    # 之后流会不会自己关掉。断言从这个语义出发，而不是断最后一帧的名字。
    kinds = [k for k, _ in parsed]
    assert "error" in kinds, kinds
    assert kinds[-1] == "done", f"流必须以 done 收尾：{kinds}"
    assert parsed[kinds.index("error")][1].get("message"), "error 帧应带可读原因"
    assert parsed[-1][1].get("error") is True, "终止 done 要标记这是一次失败的收尾"


def test_stream_validation_still_enforced():
    r = client.post("/v1/chat/stream", json={"question": ""})
    assert r.status_code == 422


def test_stream_sets_no_buffering_headers(monkeypatch):
    """工单 19：不关掉代理缓冲的话，"流式"在 Nginx 后面会退化成一次性返回。"""
    events = [
        {"event": "citations", "data": {"citations": []}},
        {"event": "done", "data": {"grounded": True, "cached": False}},
    ]
    monkeypatch.setattr(deps.pipeline, "astream", _fake_astream(events), raising=False)
    r = client.post("/v1/chat/stream", json={"question": "年假几天"})
    assert r.headers.get("x-accel-buffering") == "no"
    assert "no-transform" in r.headers.get("cache-control", "")


def test_stream_emits_heartbeat_while_idle(monkeypatch):
    """空闲心跳：LLM 首 token 可能等十几秒，代理的默认空闲超时会掐断连接。"""
    import asyncio

    from app.config import settings

    async def slow():
        yield {"event": "citations", "data": {"citations": []}}
        await asyncio.sleep(0.25)  # 模拟 LLM 迟迟不出字
        yield {"event": "token", "data": {"t": "年假 15 天"}}
        yield {"event": "done", "data": {"grounded": True, "cached": False}}

    async def astream_slow(question, kb_id=None, history=None):
        async for e in slow():
            yield e

    monkeypatch.setattr(deps.pipeline, "astream", astream_slow, raising=False)
    monkeypatch.setattr(settings, "sse_heartbeat_seconds", 0.05, raising=False)
    r = client.post("/v1/chat/stream", json={"question": "年假几天"})
    assert ": ping" in r.text, "空闲期必须发注释帧保活"
    kinds = [k for k, _ in _parse_sse(r.text)]
    assert kinds == ["citations", "token", "done"], f"心跳不得污染事件序列：{kinds}"
