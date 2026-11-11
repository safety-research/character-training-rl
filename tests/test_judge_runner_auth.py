"""The streaming judge retries an auth error once, then records a judge error."""

from types import SimpleNamespace

import src.evals.judge_runner as jr
import src.keys as keys


class _Rejected(Exception):
    status_code = 401


class _Client:
    def __init__(self, failures):
        self.failures, self.calls = failures, 0
        self.messages = self

    def create(self, **_):
        self.calls += 1
        if self.calls <= self.failures:
            raise _Rejected("authentication failed")
        return SimpleNamespace(stop_reason="tool_use", verdict="ok")


def _run(monkeypatch, failures):
    client = _Client(failures)
    monkeypatch.setattr(keys, "anthropic_client", lambda: client)
    monkeypatch.setattr(jr.time, "sleep", lambda s: None)
    out = jr._run_streaming("claude-sonnet-4-6", "rubric", ["item"], {"name": "t"}, lambda m: {"v": m.verdict}, 100, 1)
    return out, client.calls


def test_one_auth_error_is_retried(monkeypatch):
    assert _run(monkeypatch, failures=1) == ([{"v": "ok"}], 2)


def test_a_second_auth_error_is_a_judge_error(monkeypatch):
    assert _run(monkeypatch, failures=2) == ([None], 2)
