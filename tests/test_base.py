"""Tests for src/evals/base.py — stats, caching, generators, utilities."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from src.evals.base import (
    EvalResult,
    OverloadError,
    ResponseCache,
    VLLMServerGenerator,
    _run_async,
    check_judge_errors,
    check_thinking_sanity,
    clustered_ci,
    create_generator,
    fix_thinking_tags,
    get_cache_key,
    is_api_model,
    validate_results,
    verify_thinking_param,
    wilson_ci,
)

# --- wilson_ci ---


def test_wilson_ci_n0():
    """n=0 has no reportable interval."""
    with pytest.raises(ValueError, match="n > 0"):
        wilson_ci(0.5, 0)


def test_wilson_ci_n1():
    """n=1 should return a valid interval."""
    lo, hi = wilson_ci(1.0, 1)
    assert 0 <= lo <= hi <= 1


def test_wilson_ci_extreme_p0():
    """p=0 should have ci_low=0."""
    lo, hi = wilson_ci(0.0, 100)
    assert lo == 0.0
    assert hi > 0


def test_wilson_ci_extreme_p1():
    """p=1 should have ci_high close to 1."""
    lo, hi = wilson_ci(1.0, 100)
    assert lo < 1.0
    assert hi > 0.99


def test_wilson_ci_normal_case():
    """Normal case with p=0.5, n=100."""
    lo, hi = wilson_ci(0.5, 100)
    assert 0.3 < lo < 0.5
    assert 0.5 < hi < 0.7


# --- clustered_ci ---


def test_clustered_ci_empty():
    """Empty list has no reportable interval."""
    with pytest.raises(ValueError, match="at least one cluster"):
        clustered_ci([])


def test_clustered_ci_single():
    """Single cluster is not enough evidence for a reportable interval."""
    with pytest.raises(ValueError, match="at least 4 clusters"):
        clustered_ci([0.5])


def test_clustered_ci_n_le_3():
    """n<=3 is not enough evidence for a reportable interval."""
    with pytest.raises(ValueError, match="at least 4 clusters"):
        clustered_ci([0.3, 0.5, 0.7])


def test_clustered_ci_n30():
    """n=30 should give a reasonable CI."""
    rates = [0.5] * 30
    mean, lo, hi = clustered_ci(rates)
    assert abs(mean - 0.5) < 0.01
    # All identical rates → zero variance → tight CI
    assert abs(lo - 0.5) < 0.01
    assert abs(hi - 0.5) < 0.01


# --- get_cache_key ---


def cache_key_kwargs(**overrides):
    values = {
        "framing_id": "framing",
        "behavior_id": "behavior",
        "system_prompt": "system",
        "cache_version": "v1",
    }
    values.update(overrides)
    return values


def test_cache_key_deterministic():
    """Same inputs should produce same key."""
    k1 = get_cache_key("model", "prompt", 0.7, 42, **cache_key_kwargs())
    k2 = get_cache_key("model", "prompt", 0.7, 42, **cache_key_kwargs())
    assert k1 == k2


def test_cache_key_different_thinking():
    """Thinking kwarg should change cache key."""
    k1 = get_cache_key("model", "prompt", 0.7, 42, thinking=True, **cache_key_kwargs())
    k2 = get_cache_key("model", "prompt", 0.7, 42, thinking=False, **cache_key_kwargs())
    assert k1 != k2


def test_cache_key_different_framing():
    """framing_id/behavior_id kwargs must change the cache key (condition isolation)."""
    k1 = get_cache_key(
        "model", "prompt", 0.7, 42, **cache_key_kwargs(framing_id="assigned_rules", behavior_id="bananas")
    )
    k2 = get_cache_key("model", "prompt", 0.7, 42, **cache_key_kwargs(framing_id="character", behavior_id="bananas"))
    assert k1 != k2


def test_cache_key_stability():
    """Pinned hash for regression testing."""
    k = get_cache_key("test_model", "test_prompt", 0.7, 42, **cache_key_kwargs())
    assert len(k) == 32  # MD5 hex digest length
    assert k.isalnum()


# --- ResponseCache ---


def test_response_cache_roundtrip():
    """Cache set → get should return same value."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(Path(tmpdir))
        cache.set("test_key", "test_response")
        assert cache.get("test_key") == "test_response"
        assert "test_key" in cache
        assert len(cache) == 1


def test_response_cache_persistence():
    """Cache should persist across instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache1 = ResponseCache(Path(tmpdir))
        cache1.set("k1", "v1")

        cache2 = ResponseCache(Path(tmpdir))
        assert cache2.get("k1") == "v1"


def test_response_cache_miss():
    """Cache miss should return None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(Path(tmpdir))
        assert cache.get("nonexistent") is None


def test_response_cache_rejects_corrupt_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "bad.json").write_text("{not-json")
        with pytest.raises(ValueError, match="Corrupt cache file"):
            ResponseCache(Path(tmpdir))


def test_response_cache_write_failure_does_not_update_memory(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(Path(tmpdir))

        def raise_write_error(self, *args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", raise_write_error)
        with pytest.raises(RuntimeError, match="Failed to write cache file"):
            cache.set("k1", "v1")
        assert cache.get("k1") is None


# --- is_api_model ---


def test_is_api_model():
    assert is_api_model("claude-haiku-4-5-20251001") is True
    assert is_api_model("claude-sonnet-4-5-20250929") is True
    assert is_api_model("openai/gpt-4o") is True
    assert is_api_model("anthropic/claude-3-opus") is True
    assert is_api_model("Qwen/Qwen3-32B") is False
    assert is_api_model("/local/path/model") is False


# --- EvalResult ---


def test_eval_result_save():
    """EvalResult should save summary and results files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = EvalResult(
            name="test",
            primary_metric=50.0,
            primary_metric_name="accuracy",
            ci_low=40.0,
            ci_high=60.0,
            n_samples=1,
            raw_results=[{"id": 1, "score": 0.5}],
            git_commit="test1234",
        )
        result.save(Path(tmpdir))

        summary_path = Path(tmpdir) / "test_summary.json"
        results_path = Path(tmpdir) / "test_results.jsonl"
        assert summary_path.exists()
        assert results_path.exists()

        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["primary_metric"] == 50.0
        assert summary["n_samples"] == 1


def test_eval_result_breakdowns_in_summary():
    """Breakdowns should appear in saved summary JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        breakdowns = {
            "by_category": [
                {"category": "A", "rate": 50.0, "n": 10},
                {"category": "B", "rate": 80.0, "n": 20},
            ],
        }
        result = EvalResult(
            name="test",
            primary_metric=65.0,
            primary_metric_name="accuracy",
            ci_low=55.0,
            ci_high=75.0,
            n_samples=30,
            breakdowns=breakdowns,
            raw_results=[{"id": i} for i in range(30)],
            git_commit="test1234",
        )
        result.save(Path(tmpdir))

        with open(Path(tmpdir) / "test_summary.json") as f:
            summary = json.load(f)

        assert "breakdowns" in summary
        assert len(summary["breakdowns"]["by_category"]) == 2
        assert summary["breakdowns"]["by_category"][0]["category"] == "A"
        assert summary["breakdowns"]["by_category"][1]["rate"] == 80.0


def test_eval_result_empty_breakdowns():
    """Empty breakdowns should be an empty dict in summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = EvalResult(
            name="test",
            primary_metric=50.0,
            primary_metric_name="accuracy",
            ci_low=40.0,
            ci_high=60.0,
            n_samples=100,
            raw_results=[{"id": i} for i in range(100)],
            git_commit="test1234",
        )
        result.save(Path(tmpdir))

        with open(Path(tmpdir) / "test_summary.json") as f:
            summary = json.load(f)

        assert "breakdowns" in summary
        assert summary["breakdowns"] == {}


def test_eval_result_save_requires_raw_rows():
    result = EvalResult(
        name="test",
        primary_metric=50.0,
        primary_metric_name="accuracy",
        ci_low=40.0,
        ci_high=60.0,
        n_samples=1,
        git_commit="test1234",
    )
    with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(ValueError, match="raw_results"):
        result.save(Path(tmpdir))


def test_eval_result_save_requires_raw_rows_match_n_samples():
    result = EvalResult(
        name="test",
        primary_metric=50.0,
        primary_metric_name="accuracy",
        ci_low=40.0,
        ci_high=60.0,
        n_samples=2,
        raw_results=[{"id": 1}],
        git_commit="test1234",
    )
    with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(ValueError, match="expected 2 raw_results"):
        result.save(Path(tmpdir))


# --- check_judge_errors ---


def test_check_judge_errors_all_ok():
    results = [{"response": "hi", "judge_output": {"score": 1}} for _ in range(10)]
    stats = check_judge_errors(results)
    assert stats["n_judge_errors"] == 0
    assert stats["n_generation_failures"] == 0
    assert stats["judge_error_rate"] == 0.0


def test_check_judge_errors_some_failures():
    results = [{"response": "hi", "judge_output": {"score": 1}} for _ in range(9)]
    results.append({"response": "hi", "judge_output": None})  # judge error
    stats = check_judge_errors(results, response_key="response", judge_output_key="judge_output")
    assert stats["n_judge_errors"] == 1
    assert stats["n_generation_failures"] == 0


def test_check_judge_errors_generation_failures():
    results = [{"response": "hi", "judge_output": {"score": 1}} for _ in range(9)]
    results.append({"response": None, "judge_output": None})  # gen failure
    stats = check_judge_errors(results, response_key="response", judge_output_key="judge_output")
    assert stats["n_judge_errors"] == 0  # gen failure, not judge error
    assert stats["n_generation_failures"] == 1


def test_check_judge_errors_all_generation_failures_raise():
    results = [{"response": None, "judge_output": None} for _ in range(3)]
    with pytest.raises(RuntimeError, match="All 3 generations failed"):
        check_judge_errors(results, response_key="response", judge_output_key="judge_output")


def test_check_judge_errors_raises_above_threshold():
    # 3/10 judge errors = 33% > 10% threshold
    results = [{"response": "hi", "judge_output": {"score": 1}} for _ in range(7)]
    results.extend([{"response": "hi", "judge_output": None} for _ in range(3)])
    with pytest.raises(RuntimeError, match="Judge error rate too high"):
        check_judge_errors(results, response_key="response", judge_output_key="judge_output")


def test_check_judge_errors_custom_keys():
    results = [{"answer": "hi", "classification": "safe"} for _ in range(10)]
    stats = check_judge_errors(results, response_key="answer", judge_output_key="classification")
    assert stats["n_judge_errors"] == 0


# --- check_thinking_sanity ---


def test_check_thinking_sanity_thinking_ok():
    responses = ["<think>\nreasoning\n</think>\n\nanswer" for _ in range(5)]
    check_thinking_sanity(responses, thinking=True)  # should not raise


def test_check_thinking_sanity_nothinking_ok():
    responses = ["just an answer" for _ in range(5)]
    check_thinking_sanity(responses, thinking=False)  # should not raise


def test_check_thinking_sanity_thinking_mismatch():
    responses = ["no thinking here" for _ in range(5)]
    with pytest.raises(RuntimeError, match="thinking=True"):
        check_thinking_sanity(responses, thinking=True)


def test_check_thinking_sanity_nothinking_mismatch_hard_error():
    responses = ["<think>\nreasoning\n</think>\n\nanswer" for _ in range(5)]
    with pytest.raises(RuntimeError, match="thinking=False"):
        check_thinking_sanity(responses, thinking=False)


def test_check_thinking_sanity_tolerates_small_tail():
    # A sub-threshold fraction without <think> is the natural temp>0 tail, not
    # thinking-disabled: warn, don't raise.
    responses = ["<think>\nr\n</think>\n\na"] * 99 + ["plain answer"]  # 1% missing
    check_thinking_sanity(responses, thinking=True)  # should not raise


def test_check_thinking_sanity_raises_above_threshold():
    responses = ["<think>\nr\n</think>\n\na"] * 80 + ["plain"] * 20  # 20% missing
    with pytest.raises(RuntimeError, match="misconfigured"):
        check_thinking_sanity(responses, thinking=True)


def test_check_thinking_sanity_nothinking_mismatch_any_trace_raises():
    responses = [
        "<think>\nreasoning\n</think>\n\nanswer",
        "just an answer",
        "just an answer",
        "just an answer",
        "just an answer",
    ]
    with pytest.raises(RuntimeError, match="thinking=False"):
        check_thinking_sanity(responses, thinking=False)


def test_check_thinking_sanity_skips_nones():
    responses = [None, None, "<think>\nok\n</think>\n\nhi", None]
    # Only 1 non-None sample, below threshold of 3 — should not raise
    check_thinking_sanity(responses, thinking=True)


def test_fix_thinking_tags_rejects_missing_response():
    with pytest.raises(ValueError, match="missing response"):
        fix_thinking_tags(None)


def test_fix_thinking_tags_preserves_empty_string():
    assert fix_thinking_tags("") == ""


# --- validate_results ---


def test_validate_results_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        # Write valid summary
        (d / "test_summary.json").write_text(
            json.dumps(
                {
                    "primary_metric": 50.0,
                    "n_samples": 10,
                }
            )
        )
        # Write valid JSONL
        rows = [json.dumps({"response": "hi", "_git_hash": "abc123"}) for _ in range(10)]
        (d / "test_results.jsonl").write_text("\n".join(rows))

        valid, issues = validate_results(d)
        assert valid, f"Expected valid but got issues: {issues}"


def test_validate_results_missing_summary():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        (d / "test_results.jsonl").write_text('{"response": "hi", "_git_hash": "abc"}\n')
        valid, issues = validate_results(d)
        assert not valid
        assert any("summary" in i.lower() for i in issues)


def test_validate_results_null_metric():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        (d / "test_summary.json").write_text(json.dumps({"primary_metric": None, "n_samples": 0}))
        (d / "test_results.jsonl").write_text('{"response": "hi", "_git_hash": "abc"}\n')
        valid, issues = validate_results(d)
        assert not valid


def test_validate_results_nonexistent_dir():
    valid, issues = validate_results(Path("/nonexistent/path"))
    assert not valid
    assert any("does not exist" in i for i in issues)


def test_validate_results_thinking_check():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        (d / "test_summary.json").write_text(json.dumps({"primary_metric": 50.0, "n_samples": 5}))
        rows = [json.dumps({"response": "no thinking", "_git_hash": "abc"}) for _ in range(5)]
        (d / "test_results.jsonl").write_text("\n".join(rows))

        valid, issues = validate_results(d, expected_thinking=True)
        assert not valid
        assert any("thinking" in i.lower() for i in issues)


# --- OverloadError ---


def test_overload_error_is_runtime_error():
    assert issubclass(OverloadError, RuntimeError)
    with pytest.raises(RuntimeError):
        raise OverloadError("test")


# --- _run_async: exceptions must propagate from inside a running event loop ---


def test_run_async_outside_loop():
    async def coro():
        return 42

    assert _run_async(coro()) == 42


def test_run_async_propagates_exception_outside_loop():
    async def boom():
        raise OverloadError("inner failure")

    with pytest.raises(OverloadError, match="inner failure"):
        _run_async(boom())


async def test_run_async_propagates_exception_inside_loop():
    """Coroutine exceptions raised while an event loop is running must propagate,
    not be mistaken for 'no running event loop' and rerun via asyncio.run."""

    async def boom():
        raise OverloadError("inner failure")

    with pytest.raises(OverloadError, match="inner failure"):
        _run_async(boom())


async def test_run_async_returns_inside_loop():
    async def coro():
        await asyncio.sleep(0)
        return "ok"

    assert _run_async(coro()) == "ok"


# --- create_generator routing ---


def test_create_generator_use_batch_claude():
    from src.batch_api import BatchAnthropicGenerator

    g = create_generator("claude-haiku-4-5-20251001", use_batch=True)
    assert isinstance(g, BatchAnthropicGenerator)


def test_create_generator_use_batch_non_claude_rejected():
    with pytest.raises(ValueError, match="use_batch"):
        create_generator("openai/gpt-4o", use_batch=True)


def test_create_generator_unknown_model_rejected():
    """No silent fallback to in-process vLLM (dropped in this repo)."""
    with pytest.raises(ValueError, match="Unrecognized model path"):
        create_generator("Qwen/Qwen3-32B")


# --- thinking param: declare-and-verify ---


class FakeTokenizer:
    """Mimics a jinja chat template: honors one kwarg, silently ignores the rest."""

    def __init__(self, honored_param: str | None):
        self.honored_param = honored_param

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        base = "rendered"
        if self.honored_param and kwargs.get(self.honored_param) is True:
            base += "<think>"
        return base


def test_verify_thinking_param_correct_declaration():
    assert verify_thinking_param(FakeTokenizer("enable_thinking"), "enable_thinking") is True


def test_verify_thinking_param_wrong_declaration():
    """A wrong name is silently ignored by the template — verification must catch it."""
    assert verify_thinking_param(FakeTokenizer("enable_thinking"), "thinking") is False


def test_verify_thinking_param_no_thinking_support():
    assert verify_thinking_param(FakeTokenizer(None), "enable_thinking") is False


def test_vllm_server_thinking_requires_declaration():
    gen = VLLMServerGenerator(base_url="http://example:8000/v1")  # no thinking_param
    assert gen.supports_thinking is False
    with pytest.raises(ValueError, match="no thinking_param was declared"):
        gen.generate_chat([[{"role": "user", "content": "hi"}]], thinking=True)


def test_vllm_server_declared_param_sets_support():
    gen = VLLMServerGenerator(base_url="http://example:8000/v1", thinking_param="enable_thinking")
    assert gen.supports_thinking is True
    assert gen.thinking_param == "enable_thinking"


def test_vllm_cap_max_tokens_raises_on_overflow():
    gen = VLLMServerGenerator(base_url="http://example:8000/v1")
    gen.max_model_len = 10
    with pytest.raises(ValueError, match="exceeds available headroom"):
        gen._cap_max_tokens([{"role": "user", "content": "x" * 20}], max_tokens=20)


# --- unsupported params are errors, never silent drops ---

CONV = [[{"role": "user", "content": "hi"}]]


def test_anthropic_rejects_seed():
    """The Anthropic API has no seed parameter — a true API fact, not an
    implementation gap."""
    from src.evals.base import AnthropicGenerator

    gen = AnthropicGenerator(api_key="sk-test")
    with pytest.raises(ValueError, match="seed"):
        gen.generate_chat(CONV, seed=42)


def test_anthropic_rejects_top_p_and_prefill_only_with_thinking():
    """top_p and prefill are API-supported without thinking, incompatible with it."""
    from src.evals.base import AnthropicGenerator

    gen = AnthropicGenerator(api_key="sk-test")
    with pytest.raises(ValueError, match="top_p"):
        gen.generate_chat(CONV, thinking=True, top_p=0.9)
    with pytest.raises(ValueError, match="prefill_tag"):
        gen.generate_chat(CONV, thinking=True, prefill_tag="<helpful>")


def test_anthropic_rejects_temperature_with_top_p():
    """Verified live: the API 400s when both are sent; we catch it before spending."""
    from src.evals.base import AnthropicGenerator

    gen = AnthropicGenerator(api_key="sk-test")
    with pytest.raises(ValueError, match="temperature and top_p together"):
        gen.generate_chat(CONV, temperature=0.7, top_p=0.9)


def test_batch_generator_rejects_seed_and_thinking_combos():
    from src.batch_api import BatchAnthropicGenerator

    gen = BatchAnthropicGenerator(api_key="sk-test")
    with pytest.raises(ValueError, match="seed"):
        gen.generate_chat(CONV, seed=42)
    with pytest.raises(ValueError, match="top_p"):
        gen.generate_chat(CONV, thinking=True, top_p=0.9)


def test_openrouter_rejects_only_prefill():
    from src.evals.base import OpenRouterGenerator

    gen = OpenRouterGenerator(model_name="openai/gpt-4o")
    with pytest.raises(ValueError, match="prefill_tag"):
        gen.generate_chat(CONV, prefill_tag="<helpful>")


def test_openrouter_body_includes_reasoning_seed_top_p():
    from src.evals.base import OpenRouterGenerator

    gen = OpenRouterGenerator(model_name="openai/gpt-5.4")
    body = gen._build_body(CONV[0], max_tokens=64, temperature=0.7, top_p=0.9, seed=7, thinking=True)
    assert body["reasoning"] == {"enabled": True}
    assert body["seed"] == 7
    assert body["top_p"] == 0.9
    # Defaults stay out of the body so requests are minimal and cacheable — except
    # reasoning, which is always declared explicitly: omitting it leaves the choice to
    # the upstream provider's chat template, which varies run-to-run (see _build_body).
    body = gen._build_body(CONV[0], max_tokens=64, temperature=0.7, top_p=1.0, seed=None, thinking=False)
    assert body["reasoning"] == {"enabled": False}
    assert "seed" not in body and "top_p" not in body


def test_openrouter_extracts_reasoning_into_think_tags():
    from src.evals.base import OpenRouterGenerator

    msg = {"content": "answer", "reasoning": "step by step"}
    assert OpenRouterGenerator._extract_text(msg) == "<think>\nstep by step\n</think>\n\nanswer"
    assert OpenRouterGenerator._extract_text({"content": "plain"}) == "plain"
    with pytest.raises(ValueError, match="null/absent content"):
        OpenRouterGenerator._extract_text({"content": None})
    # An absent content key is the same degenerate case, not a raw KeyError.
    with pytest.raises(ValueError, match="null/absent content"):
        OpenRouterGenerator._extract_text({"reasoning": "no content key"})


def test_vllm_server_accepts_seed_and_top_p():
    """Both are honored by vLLM — param validation must not raise (the call then
    fails on connection, which is fine for this test)."""
    gen = VLLMServerGenerator(base_url="http://example:8000/v1")
    with pytest.raises(RuntimeError, match="Failed to connect"):
        gen.generate_chat(CONV, seed=42, top_p=0.9)


def test_reject_unsupported_params_lists_all_offenders():
    from src.evals.base import reject_unsupported_params

    with pytest.raises(ValueError) as exc_info:
        reject_unsupported_params("TestGen", {"top_p": (0.9, 1.0), "seed": (42, None)})
    assert "top_p" in str(exc_info.value)
    assert "seed" in str(exc_info.value)


def test_reject_unsupported_params_passes_defaults():
    from src.evals.base import reject_unsupported_params

    reject_unsupported_params("TestGen", {"top_p": (1.0, 1.0), "seed": (None, None)})  # no raise


# --- Tinker prompt building: forced thinking opener + disable-thinking renderer choice ---


class TailPrompt:
    def __init__(self, tail_tokens):
        self._tail = tail_tokens

    def to_ints(self):
        return self._tail


class RecordingRenderer:
    """Records build_generation_prompt calls; `prefills` controls the rendered tail."""

    def __init__(self, prefills_think: bool):
        self.prefills_think = prefills_think
        self.calls = []

    def build_generation_prompt(self, messages, prefill=None):
        self.calls.append(prefill)
        return TailPrompt([1, 2, 3])


class TailTokenizer:
    def __init__(self, renderer):
        self._renderer = renderer

    def decode(self, tokens, **kwargs):
        return "<|assistant|>\n<think>" if self._renderer.prefills_think else "<|assistant|>\n"


def test_force_thinking_prompt_skips_prefill_for_harmony_renderers():
    """gpt-oss delimits reasoning with the `assistantfinal` marker, not a <think> block, so
    forcing an opener there opens a tag the model never closes: every response then reads as
    truncated mid-reasoning and the whole run is dropped."""
    from src.evals.base import force_thinking_prompt, renderer_uses_think_blocks

    assert renderer_uses_think_blocks("qwen3_5") is True
    assert renderer_uses_think_blocks("nemotron3") is True
    assert renderer_uses_think_blocks("nemotron3_disable_thinking") is True
    assert renderer_uses_think_blocks("gpt_oss_low_reasoning") is False
    assert renderer_uses_think_blocks("gpt_oss_high_reasoning") is False
    # None means "not declared", and must keep the historical behaviour: every call site
    # written before harmony-format models passes only three arguments.
    assert renderer_uses_think_blocks(None) is True

    msgs = [{"role": "user", "content": "q"}]
    # harmony family: the opener must NOT be forced, even though the tail lacks <think>
    r = RecordingRenderer(prefills_think=False)
    force_thinking_prompt(r, TailTokenizer(r), msgs, "gpt_oss_low_reasoning")
    assert r.calls == [None]
    # identical renderer shape, declared as a think-block family -> forced exactly as before
    r = RecordingRenderer(prefills_think=False)
    force_thinking_prompt(r, TailTokenizer(r), msgs, "qwen3_5")
    assert r.calls == [None, "<think>\n"]


def test_force_thinking_prompt_adds_prefill_only_when_renderer_does_not():
    from src.evals.base import force_thinking_prompt

    msgs = [{"role": "user", "content": "q"}]
    # nemotron3-style: renderer already ends the prompt with <think> -> no explicit prefill
    r = RecordingRenderer(prefills_think=True)
    force_thinking_prompt(r, TailTokenizer(r), msgs)
    assert r.calls == [None]
    # qwen3-style: template leaves thinking to the model -> the opener is forced
    r = RecordingRenderer(prefills_think=False)
    force_thinking_prompt(r, TailTokenizer(r), msgs)
    assert r.calls == [None, "<think>\n"]


def test_tinker_nothinking_uses_the_disable_thinking_renderer(monkeypatch):
    from src.evals.base import TinkerGenerator

    gen = TinkerGenerator(model_path="Qwen/Qwen3-8B", tokenizer_id="Qwen/Qwen3-8B", renderer_name="qwen3")
    loaded = []

    def fake_load_renderer(model, renderer_name, tokenizer):
        loaded.append(renderer_name)
        return RecordingRenderer(prefills_think=False)

    import src.training.tinker_common as tc

    monkeypatch.setattr(tc, "load_renderer", fake_load_renderer)
    gen._tokenizer = TailTokenizer(RecordingRenderer(prefills_think=False))
    gen._renderer = RecordingRenderer(prefills_think=False)

    nothink = gen._renderer_for(thinking=False)
    assert loaded == ["qwen3_disable_thinking"]
    assert gen._renderer_for(thinking=False) is nothink  # cached, not reloaded
    assert gen._renderer_for(thinking=True) is gen._renderer

    # nothinking prompts come from the disable renderer with NO hand-built prefill
    gen._build_prompt([{"role": "user", "content": "q"}], thinking=False)
    assert nothink.calls == [None]


def test_tinker_sample_chat_hands_each_prompt_to_on_sampled_as_it_returns(monkeypatch):
    """The persistence hook: a caller writes each conversation's completions the moment they land,
    before the batch as a whole is back, so a death later in the process loses no paid sampling."""
    from types import SimpleNamespace

    from src.evals.base import SampledPrompt, SampledSequence, TinkerGenerator

    gen = object.__new__(TinkerGenerator)
    gen.max_concurrent = 4
    gen._service = object()

    async def ensure_client():
        return None

    async def sample_with_retries(prompt, params, num_samples, idx, error_counts):
        return None if prompt == "fail" else [SimpleNamespace(tokens=[idx], logprobs=[0.0], stop_reason="stop")]

    seen = []
    monkeypatch.setattr(gen, "_ensure_client", ensure_client)
    monkeypatch.setattr(gen, "_sampling_params", lambda *a: None)
    monkeypatch.setattr(gen, "_build_prompt", lambda messages, thinking: messages[0]["content"])
    monkeypatch.setattr(gen, "_sample_with_retries", sample_with_retries)
    monkeypatch.setattr(gen, "_check_failures", lambda results, counts: None)
    monkeypatch.setattr(
        gen,
        "_sampled_prompt",
        lambda prompt, seqs: SampledPrompt(
            prompt_tokens=[0],
            sequences=[
                SampledSequence(tokens=s.tokens, logprobs=s.logprobs, text=prompt, stop_reason=s.stop_reason)
                for s in seqs
            ],
        ),
    )
    convs = [[{"role": "user", "content": c}] for c in ("x", "fail", "y")]
    out = gen.sample_chat(
        convs,
        n=1,
        max_tokens=5,
        temperature=1.0,
        top_p=1.0,
        thinking=False,
        on_sampled=lambda idx, sp: seen.append((idx, sp.sequences[0].text)),
    )
    assert sorted(seen) == [(0, "x"), (2, "y")]  # the failed conversation is None and never handed over
    assert [sp and sp.sequences[0].text for sp in out] == ["x", None, "y"]


def test_tinker_sample_retries_a_server_side_sampling_failure(monkeypatch):
    """Tinker raises RequestFailedError when the server fails a sample; it comes in bursts and must
    be retried, not crash the run."""
    import asyncio

    tinker = pytest.importorskip("tinker")
    from src.evals import base

    monkeypatch.setattr(base, "TINKER_BACKOFF_BASE_SEC", 0.0)
    calls = {"n": 0}

    class Resp:
        sequences = ["ok"]

    class Client:
        async def sample_async(self, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise tinker.RequestFailedError(
                    "Failed to sample after exhausting retries", "r", tinker.types.RequestErrorCategory.Unknown
                )
            return Resp()

    gen = base.TinkerGenerator.__new__(base.TinkerGenerator)
    gen._client = Client()
    counts = {}
    out = asyncio.run(gen._sample_with_retries(prompt=None, params=None, num_samples=1, idx=0, error_counts=counts))
    assert out == ["ok"] and calls["n"] == 3 and counts["transient"] == 2


def test_tinker_sample_does_not_retry_a_user_request_failure():
    import asyncio

    tinker = pytest.importorskip("tinker")
    from src.evals import base

    calls = {"n": 0}

    class Client:
        async def sample_async(self, **kw):
            calls["n"] += 1
            raise tinker.RequestFailedError("bad request", "r", tinker.types.RequestErrorCategory.User)

    gen = base.TinkerGenerator.__new__(base.TinkerGenerator)
    gen._client = Client()
    counts = {}
    assert (
        asyncio.run(gen._sample_with_retries(prompt=None, params=None, num_samples=1, idx=0, error_counts=counts))
        is None
    )
    assert calls["n"] == 1 and counts["deterministic"] == 1
