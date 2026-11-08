"""Tests for the judge's endpoint compatibility and verdict validation.

The endpoint tests drive the real ``openai`` SDK against a small local HTTP
server that imitates the ways OpenAI-compatible endpoints differ from
api.openai.com (no Responses API, ``max_tokens`` vs ``max_completion_tokens``,
no ``store``). No network access or API key is needed.

Run with:  uv run --group tests pytest tests/test_check_prediction_judge.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_prediction as cp  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _rubric(flag: str, n_metrics: int = 1, n_logs: int = 1, n_traces: int = 1) -> dict:
    return {
        "feature_flag": flag,
        "incident_time": "2026-04-22T20:52:46+00:00",
        "description": f"{flag} description",
        "mechanism": f"{flag} mechanism",
        "symptoms": {
            "metrics": [
                {"metric_family": f"family_{i}", "variants": []}
                for i in range(n_metrics)
            ],
            "logs": [
                {
                    "representative_attributes": {
                        "body": "b",
                        "resource.service.name": "s",
                    }
                }
                for _ in range(n_logs)
            ],
            "traces": [
                {"call_chain": "a -> b", "smoking_gun": ["x"]} for _ in range(n_traces)
            ],
        },
    }


def _verdict(flag: str, match: bool, rubric: dict, score: int = 3) -> dict:
    symptoms = rubric["symptoms"]
    return {
        "feature_flag": flag,
        "incident_time_within_10min": match,
        "feature_flag_match": match,
        "mechanism_match": match,
        "symptoms": {
            "metrics": [
                {"metric_family": m["metric_family"], "family_match": match}
                for m in symptoms["metrics"]
            ],
            "logs": [
                {"body_match": match, "service_name_match": match}
                for _ in symptoms["logs"]
            ],
            "traces": [
                {"call_chain_match": match, "smoking_gun_match": match}
                for _ in symptoms["traces"]
            ],
        },
        "score": score,
    }


EXPECTED = {
    "events": [
        {"event_id": "e1", "root_cause": "paymentFailure"},
        {"event_id": "e2", "root_cause": "paymentUnreachable"},
    ]
}
RUBRICS = [_rubric("paymentFailure"), _rubric("paymentUnreachable", n_metrics=2)]
GOOD = {"rubrics": [_verdict(r["feature_flag"], True, r) for r in RUBRICS]}


class FakeEndpoint:
    """Local OpenAI-compatible server with switchable quirks."""

    def __init__(
        self,
        *,
        responses_status: int | None = None,
        responses_error: str = "",
        reject_store: bool = False,
        reject_max_tokens: bool = False,
        reject_max_completion_tokens: bool = False,
        replies: list[dict] | None = None,
    ) -> None:
        self.responses_status = responses_status
        self.responses_error = responses_error
        self.reject_store = reject_store
        self.reject_max_tokens = reject_max_tokens
        self.reject_max_completion_tokens = reject_max_completion_tokens
        self.replies = list(replies or [GOOD])
        self.requests: list[tuple[str, dict]] = []

    def _next_reply(self) -> str:
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return json.dumps(reply)

    def handle(self, path: str, body: dict) -> tuple[int, dict]:
        """Answer one request the way the configured endpoint would."""
        self.requests.append((path, body))
        if path.endswith("/responses"):
            if self.responses_status is not None:
                return self.responses_status, {
                    "error": {"message": self.responses_error}
                }
            if self.reject_store and "store" in body:
                return 400, {"error": {"message": "store is not supported"}}
            text = self._next_reply()
            return 200, {
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "model": body["model"],
                "status": "completed",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": text, "annotations": []}
                        ],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        if path.endswith("/chat/completions"):
            if self.reject_max_tokens and "max_tokens" in body:
                return 400, {
                    "error": {
                        "message": "Unsupported parameter: 'max_tokens' is not supported "
                        "with this model. Use 'max_completion_tokens' instead."
                    }
                }
            if self.reject_max_completion_tokens and "max_completion_tokens" in body:
                return 400, {
                    "error": {
                        "message": "Unrecognized parameter: max_completion_tokens"
                    }
                }
            return 200, {
                "id": "chat_1",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": self._next_reply()},
                    }
                ],
            }
        return 404, {"error": {"message": f"no route {path}"}}


Serve = Callable[["FakeEndpoint"], str]


@pytest.fixture
def serve() -> Iterator[Serve]:
    """Start local fake endpoints; shut them down after the test."""
    servers: list[ThreadingHTTPServer] = []

    def start(endpoint: FakeEndpoint) -> str:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                status, payload = endpoint.handle(self.path, body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}/v1"

    yield start
    for server in servers:
        server.shutdown()


def _judge(base_url: str, rubrics: list[dict] = RUBRICS) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="test", base_url=base_url, max_retries=0)
    return asyncio.run(
        cp.judge(
            client,
            EXPECTED,
            "report naming paymentFailure",
            rubrics,
            model="judge-model",
        )
    )


def _paths(endpoint: FakeEndpoint) -> list[str]:
    return [p.rsplit("/v1", 1)[-1] for p, _ in endpoint.requests]


# ---------------------------------------------------------------------------
# Endpoint compatibility
# ---------------------------------------------------------------------------


def test_openai_style_endpoint_uses_responses_api(serve: Serve) -> None:
    """api.openai.com behavior is unchanged: one Responses call, store=True."""
    endpoint = FakeEndpoint()
    result = _judge(serve(endpoint))
    assert result["mode"] == "llm_judge"
    assert _paths(endpoint) == ["/responses"]
    assert endpoint.requests[0][1]["store"] is True


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (404, "Not Found"),
        (405, "Method Not Allowed"),
        (501, "Not Implemented"),
        (400, "Missing required parameter: 'messages'"),
        (400, "Unrecognized request URL (POST /v1/responses)"),
    ],
)
def test_falls_back_to_chat_when_responses_api_is_missing(
    serve: Serve, status: int, message: str
) -> None:
    """Every common "no Responses API" answer routes to Chat Completions."""
    endpoint = FakeEndpoint(responses_status=status, responses_error=message)
    result = _judge(serve(endpoint))
    assert result["nested"]["rubrics"][0]["feature_flag"] == "paymentFailure"
    assert _paths(endpoint) == ["/responses", "/chat/completions"]
    chat_body = endpoint.requests[1][1]
    assert "max_completion_tokens" in chat_body
    assert "max_tokens" not in chat_body


def test_unrelated_400_is_not_swallowed_by_the_fallback(serve: Serve) -> None:
    """A genuine bad request on /responses is raised, not retried as chat."""
    endpoint = FakeEndpoint(responses_status=400, responses_error="Invalid schema")
    with pytest.raises(Exception, match="Invalid schema"):
        _judge(serve(endpoint))
    assert _paths(endpoint) == ["/responses"]


def test_auth_error_is_not_retried_as_chat(serve: Serve) -> None:
    """Auth failures surface directly."""
    endpoint = FakeEndpoint(responses_status=401, responses_error="bad key")
    with pytest.raises(Exception, match="bad key"):
        _judge(serve(endpoint))
    assert _paths(endpoint) == ["/responses"]


def test_store_is_dropped_when_the_endpoint_rejects_it(serve: Serve) -> None:
    """An endpoint without response storage gets the same call minus store."""
    endpoint = FakeEndpoint(reject_store=True)
    result = _judge(serve(endpoint))
    assert result["mode"] == "llm_judge"
    assert _paths(endpoint) == ["/responses", "/responses"]
    assert "store" not in endpoint.requests[1][1]


def test_chat_falls_back_to_max_tokens_for_older_servers(serve: Serve) -> None:
    """Servers that predate max_completion_tokens get max_tokens instead."""
    endpoint = FakeEndpoint(
        responses_status=404,
        responses_error="Not Found",
        reject_max_completion_tokens=True,
    )
    _judge(serve(endpoint))
    assert _paths(endpoint) == ["/responses", "/chat/completions", "/chat/completions"]
    retry = endpoint.requests[2][1]
    assert retry["max_tokens"] == cp.DEFAULT_MAX_TOKENS
    assert "max_completion_tokens" not in retry


def test_reasoning_model_on_chat_path_never_sees_max_tokens(serve: Serve) -> None:
    """Reasoning models that reject max_tokens succeed on the first chat call."""
    endpoint = FakeEndpoint(
        responses_status=400,
        responses_error="Missing required parameter: 'messages'",
        reject_max_tokens=True,
    )
    result = _judge(serve(endpoint))
    assert result["mode"] == "llm_judge"
    assert _paths(endpoint) == ["/responses", "/chat/completions"]


def test_judge_model_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """ORCA_JUDGE_MODEL picks the model when test.sh passes no --model."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("ORCA_JUDGE_MODEL", raising=False)
    assert cp.default_model() == "gpt-5.4"
    monkeypatch.setenv("ORCA_JUDGE_MODEL", "qwen2.5:7b")
    assert cp.default_model() == "qwen2.5:7b"
    monkeypatch.setenv("ORCA_JUDGE_MODEL", "  ")
    assert cp.default_model() == "gpt-5.4"


# ---------------------------------------------------------------------------
# Verdict validation
# ---------------------------------------------------------------------------


def test_valid_verdict_passes_and_scores() -> None:
    """A verdict that lines up with the rubrics is accepted."""
    cp.validate_judge_verdicts(GOOD, RUBRICS)
    agg = cp.aggregate_judge_response(GOOD)
    assert agg["rca_accuracy"] is True


def test_decorated_flag_names_are_accepted() -> None:
    """Case and decorations around the flag name do not trip validation."""
    parsed = json.loads(json.dumps(GOOD))
    parsed["rubrics"][0]["feature_flag"] = "`paymentFailure` (50%)"
    parsed["rubrics"][1]["feature_flag"] = "PaymentUnreachable"
    cp.validate_judge_verdicts(parsed, RUBRICS)


def test_missing_verdict_is_rejected() -> None:
    """One verdict for two rubrics would inflate rca_accuracy, so it fails."""
    # Without validation this scores rca_accuracy=True from one of two rubrics.
    parsed = {"rubrics": [GOOD["rubrics"][0]]}
    assert cp.aggregate_judge_response(parsed)["rca_accuracy"] is True
    with pytest.raises(cp.JudgeContractError, match="expected 2 rubric verdict"):
        cp.validate_judge_verdicts(parsed, RUBRICS)


def test_verdict_for_an_unlisted_flag_is_rejected() -> None:
    """A verdict naming a flag that is not the rubric's fails."""
    parsed = json.loads(json.dumps(GOOD))
    parsed["rubrics"][1]["feature_flag"] = "shoppingCartFailure"
    with pytest.raises(cp.JudgeContractError, match="shoppingCartFailure"):
        cp.validate_judge_verdicts(parsed, RUBRICS)


def test_dropped_cluster_is_rejected() -> None:
    """A verdict with fewer clusters than the rubric fails."""
    parsed = json.loads(json.dumps(GOOD))
    parsed["rubrics"][1]["symptoms"]["metrics"].pop()
    with pytest.raises(cp.JudgeContractError, match="1 metrics cluster"):
        cp.validate_judge_verdicts(parsed, RUBRICS)


def test_empty_report_synthesis_satisfies_the_contract() -> None:
    """The synthesized empty-report verdict already satisfies the contract."""
    result = asyncio.run(cp.judge(None, EXPECTED, "   ", RUBRICS))
    assert result["mode"] == "empty_report"
    cp.validate_judge_verdicts(result["nested"], RUBRICS)


def test_bad_verdict_is_retried_once_then_scored(serve: Serve) -> None:
    """A malformed verdict is re-asked once and the good answer is scored."""
    bad = {"rubrics": [GOOD["rubrics"][0]]}
    endpoint = FakeEndpoint(replies=[bad, GOOD])
    result = _judge(serve(endpoint))
    assert len(result["nested"]["rubrics"]) == 2
    assert _paths(endpoint) == ["/responses", "/responses"]


def test_persistently_bad_verdict_is_a_verifier_error(serve: Serve) -> None:
    """A verdict that stays malformed raises instead of being scored."""
    bad = {"rubrics": [GOOD["rubrics"][0]]}
    endpoint = FakeEndpoint(replies=[bad])
    with pytest.raises(cp.JudgeContractError):
        _judge(serve(endpoint))
    assert len(endpoint.requests) == cp.JUDGE_CONTRACT_ATTEMPTS
