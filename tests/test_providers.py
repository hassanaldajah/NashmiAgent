"""Provider calls: clear messages for a missing or wrong key, rate limits, timeouts and
network errors. No real network traffic: ``requests.post`` is replaced."""
import pytest
import requests

import app


class FakeResp:
    def __init__(self, status, body="", headers=None):
        self.status_code, self.text, self.headers = status, body, headers or {}

    def close(self):
        pass


GROQ = app.PROVIDERS_BY_ID["groq"]


def call():
    return app._open_stream(GROQ, "https://example.invalid/v1/chat", {}, {"model": "m"})


def test_wrong_key_says_so(monkeypatch):
    monkeypatch.setattr(app.requests, "post", lambda *a, **k: FakeResp(401, '{"error": {"message": "Invalid API Key"}}'))
    with pytest.raises(app.ProviderError) as err:
        call()
    assert "401" in str(err.value) and "invalid" in str(err.value).lower()


def test_rate_limit_retries_then_explains(monkeypatch):
    calls, slept = [], []
    monkeypatch.setattr(app.requests, "post", lambda *a, **k: calls.append(1) or FakeResp(429, "{}", {"retry-after": "2"}))
    monkeypatch.setattr(app.time, "sleep", slept.append)
    with pytest.raises(app.ProviderError) as err:
        call()
    assert len(calls) == app.MAX_RETRIES + 1
    assert slept == [2.0] * app.MAX_RETRIES          # the provider's Retry-After was used
    assert "rate limit" in str(err.value).lower()


def test_rate_limit_then_success(monkeypatch):
    answers = iter([FakeResp(429), FakeResp(200)])
    monkeypatch.setattr(app.requests, "post", lambda *a, **k: next(answers))
    monkeypatch.setattr(app.time, "sleep", lambda s: None)
    monkeypatch.setattr(app.st, "session_state", {})
    assert call().status_code == 200


@pytest.mark.parametrize("exc, words", [
    (requests.exceptions.ConnectTimeout(), "did not answer in time"),
    (requests.exceptions.ConnectionError(), "could not reach"),
])
def test_network_problems_are_readable(monkeypatch, exc, words):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(app.requests, "post", boom)
    with pytest.raises(app.ProviderError, match=f"(?i){words}"):
        call()


@pytest.mark.parametrize("header, attempt, expected", [
    ("3", 0, 3.0), ("120", 0, 8.0), (None, 0, 1.0), (None, 1, 2.0), ("soon", 1, 2.0), ("-1", 0, 1.0)])
def test_retry_delay(header, attempt, expected):
    assert app.retry_delay(header, attempt) == expected


def test_error_body_is_summarised():
    msg = app.describe_http_error("Groq", 429, '{"error": {"message": "Rate limit reached"}}')
    assert msg.startswith("Groq API error 429") and "Rate limit reached" in msg
    assert "temporary problem" in app.describe_http_error("Groq", 503, "")


def test_secrets_are_redacted_from_attachments():
    out = app.redact_secrets("GROQ_API_KEY=gsk_abc123\nname = nashmi\npassword: hunter2")
    assert "gsk_abc123" not in out and "hunter2" not in out and "nashmi" in out
