"""Shared utilities for all evaluations.

Common patterns:
- Wilson score confidence intervals
- EvalResult dataclass for consistent output
- Anthropic API generation (streaming and batch), OpenRouter, vLLM server

This repo is API-only: unrecognized local model paths are an error, never a silent
fallback. Importing this module loads the project .env; a variable already set in the
shell is kept and the file only fills in what is missing (see src.keys.load_project_env).

Backend parameter support (differences are enforced loudly, never silently dropped):
- seed: vLLM server and OpenRouter only (the Anthropic API has no seed parameter;
  OpenRouter determinism is provider-dependent). Setting seed on Anthropic raises.
- top_p: honored everywhere it's set (!= 1.0). Anthropic constraint: Claude 4+
  rejects temperature and top_p together, and thinking requires temperature=1.0 —
  top_p with thinking raises here before the API can 400.
- thinking: Anthropic native (separate token budget, temperature forced to 1.0);
  vLLM via declared chat-template kwarg (shares max_tokens, temperature
  unconstrained); OpenRouter via the unified reasoning parameter (reasoning text
  returned separately and wrapped in <think> tags here). OpenRouter caveat: the
  routed provider may silently ignore unsupported params — check_thinking_sanity
  is the backstop.
- prefill_tag: Anthropic generators implement it via an assistant-turn prefill
  (the API supports this on pre-4.6 models, 400s on newer ones, and never with
  thinking — thinking+prefill raises here). Not implemented for OpenRouter/vLLM
  server; setting it there raises.
"""

import asyncio
import concurrent.futures
import functools
import hashlib
import json
import logging
import math
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Coroutine, Optional, Union

import httpx

import src.keys  # noqa: F401  (side effect: loads the project .env without overriding the shell)
from src import paths
from src.keys import OPENROUTER_MODEL_PREFIXES, is_openrouter_model

logger = logging.getLogger(__name__)


# Retry configuration for transient API/network errors
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5.0  # seconds, doubles each retry
OVERLOAD_BASE_DELAY = 30.0  # longer backoff for 429/529

# Abort a generation run when failures exceed this fraction, with an absolute
# allowance so tiny batches (e.g. 6 prompts) survive a single flake.
FAILURE_ABORT_FRACTION = 0.1
FAILURE_ABORT_MIN = 2

# Clustered CIs need enough clusters (prompts) to be meaningful; below this we
# refuse to report an interval rather than emit a degenerate (0,1) sentinel.
MIN_CLUSTER_COUNT = 4


class OverloadError(RuntimeError):
    """API is overloaded (429/529). Wait before retrying — don't requeue immediately."""

    pass


class ResponseInvariantError(RuntimeError):
    """Provider response violated an invariant requested by the caller."""


# Provider stop reasons that mean the response was cut off, not completed.
TRUNCATED_STOP_REASONS = {"max_tokens", "length"}


def _raise_if_truncated(provider: str, stop_reason: str | None) -> None:
    if stop_reason in TRUNCATED_STOP_REASONS:
        raise ResponseInvariantError(f"{provider} returned truncated response with stop_reason={stop_reason!r}")


def _classify_error(e: Exception) -> str:
    """Classify an API error for retry/abort decisions.

    Returns one of: "client", "auth", "overload", "server", "timeout", "network", "unknown".
    """
    # Check error body for overloaded_error (Anthropic API sends this as 500+ with
    # body.error.type == 'overloaded_error', which the SDK doesn't map to 429/529)
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        error_info = body.get("error", {})
        if isinstance(error_info, dict) and error_info.get("type") == "overloaded_error":
            return "overload"

    # Anthropic SDK and httpx both expose status_code
    status = getattr(e, "status_code", None)
    if status is None and hasattr(e, "response"):
        status = getattr(e.response, "status_code", None)

    if status is not None:
        if status in (401, 403):
            return "auth"
        if status in (429, 529):
            return "overload"
        if status == 408:
            return "timeout"
        if 400 <= status < 500:
            # 400/404/413/422...: deterministic request errors — retrying cannot succeed
            return "client"
        if status >= 500:
            return "server"

    if isinstance(e, (TimeoutError,)):
        return "timeout"
    # httpx timeout types
    if hasattr(e, "__class__") and "Timeout" in type(e).__name__:
        return "timeout"
    if isinstance(e, OSError):
        return "network"
    return "unknown"


def _should_retry(error_type: str) -> bool:
    """Whether to retry this error type."""
    return error_type not in ("client", "auth")


def _retry_delay(error_type: str, attempt: int) -> float:
    """Get retry delay based on error type and attempt number."""
    base = OVERLOAD_BASE_DELAY if error_type == "overload" else RETRY_BASE_DELAY
    return base * (2**attempt)


def reject_unsupported_params(generator_name: str, params: dict) -> None:
    """Raise if a parameter this generator cannot honor was explicitly set.

    params maps param name -> (passed_value, supported_default). A value differing
    from the default means the caller asked for behavior this backend cannot
    deliver; silently dropping it would corrupt condition comparisons (e.g.
    thinking=True running as no-thinking, or distinct seeds caching as distinct
    entries while producing identical API calls).
    """
    bad = [name for name, (value, default) in params.items() if value != default]
    if bad:
        raise ValueError(
            f"{generator_name} does not support: {', '.join(bad)}. "
            f"These were explicitly set and would be silently ignored — use a backend "
            f"that honors them or remove them."
        )


def _check_deterministic_failures(backend: str, error_counts: dict) -> None:
    """Client (400) and auth (401/403) errors are deterministic — they will never
    succeed on retry or rerun, so even one means misconfiguration. Raise regardless
    of the batch-level failure threshold (which exists to tolerate transient flakes)."""
    n_deterministic = error_counts.get("client", 0) + error_counts.get("auth", 0)
    if n_deterministic > 0:
        raise RuntimeError(
            f"{backend}: {n_deterministic} request(s) failed with deterministic client/auth "
            f"errors (4xx). This is a misconfiguration (bad param combination, invalid model, "
            f"or bad key), not a transient failure — fix it before rerunning. "
            f"Errors: {', '.join(f'{k}={v}' for k, v in sorted(error_counts.items()))}"
        )


def _run_async(coro: Coroutine):
    """Run a coroutine whether or not an event loop is already running.

    If called from inside a running loop, runs the coroutine on a fresh loop in a
    worker thread. Loop detection is isolated here so exceptions raised by the
    coroutine itself (e.g. OverloadError) propagate instead of being mistaken for
    "no running event loop".
    """
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    if in_loop:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def fix_thinking_tags(text: str | None) -> str:
    """Prepend <think> if response has </think> but no opening tag.

    vLLM prefill artifact: when we prefill "<think>\\n" into the prompt,
    the generated text starts after it, so </think> appears without <think>.
    """
    from src.thinking import normalise

    if text is None:
        raise ValueError("cannot fix thinking tags on a missing response")
    return normalise(text)


def check_thinking_sanity(responses: list[str | None], thinking: bool, fail_fraction: float = 0.10) -> None:
    """Verify thinking mode matches actual response content.

    Catches the systematic failure -- the provider silently ignored the thinking param,
    so ~all responses are on the wrong side -- while tolerating the natural tail: at
    temperature > 0 a small fraction of prompts legitimately answer without a <think>
    block (or, with thinking off, a stray tag). Raises only when the mismatched fraction
    exceeds fail_fraction; below that it warns so the tail stays visible. Threshold matches
    the repo's other 10% gates.
    """
    sample = [r for r in responses if r is not None]
    if not sample:
        raise RuntimeError("thinking sanity check has no non-null responses to inspect")
    missing = [i for i, r in enumerate(sample) if "<think>" not in r and "<thinking>" not in r]
    present = [i for i, r in enumerate(sample) if "<think>" in r or "<thinking>" in r]
    wrong = missing if thinking else present
    if not wrong:
        return
    frac = len(wrong) / len(sample)
    direction = "lack <think> tags" if thinking else "contain <think> tags despite thinking=False"
    detail = f"{len(wrong)}/{len(sample)} ({frac:.1%}) responses {direction} (first bad response index: {wrong[0]})."
    if frac > fail_fraction:
        raise RuntimeError(
            f"thinking={thinking} but {detail} Above {fail_fraction:.0%}: thinking mode looks "
            f"misconfigured (check chat_template_kwargs / enable_thinking), results invalid."
        )
    print(f"  WARNING: thinking sanity tail: {detail} (<= {fail_fraction:.0%}, tolerated)")


def validate_results(results_dir, eval_name: str = None, expected_thinking: bool = None) -> tuple[bool, list[str]]:
    """Validate eval results in a directory. Returns (valid, list_of_issues).

    Checks: summary JSON exists with non-null metrics, JSONL exists with rows,
    git hash present, and optionally that thinking mode matches expectations.
    """
    results_dir = Path(results_dir)
    issues = []

    if not results_dir.exists():
        return False, [f"Directory does not exist: {results_dir}"]

    # Find summary and JSONL files
    summary_files = list(results_dir.glob("*_summary.json"))
    jsonl_files = list(results_dir.glob("*_results.jsonl"))

    if not summary_files:
        issues.append("No *_summary.json found")
    if not jsonl_files:
        issues.append("No *_results.jsonl found")

    # Validate summary
    for sf in summary_files:
        try:
            data = json.loads(sf.read_text())
            if data.get("primary_metric") is None:
                issues.append(f"{sf.name}: primary_metric is null")
            if data.get("n_samples", 0) == 0:
                issues.append(f"{sf.name}: n_samples is 0")
        except (json.JSONDecodeError, IOError) as e:
            issues.append(f"{sf.name}: failed to read: {e}")

    # Validate JSONL
    for jf in jsonl_files:
        try:
            lines = jf.read_text().strip().split("\n")
            rows = [json.loads(line) for line in lines if line.strip()]
            if len(rows) == 0:
                issues.append(f"{jf.name}: 0 rows")
            elif len(rows) < 3:
                issues.append(f"{jf.name}: only {len(rows)} rows (suspiciously few)")
            # Check git hash on every row, not just the first
            for row_i, row in enumerate(rows, start=1):
                if "_git_hash" not in row:
                    issues.append(f"{jf.name}: row {row_i} missing _git_hash")
            # Check thinking mode if expected
            if expected_thinking is not None and rows:
                if all("answer" in row for row in rows):
                    response_key = "answer"
                elif all("response" in row for row in rows):
                    response_key = "response"
                else:
                    response_key = None
                    issues.append(f"{jf.name}: rows must all contain 'answer' or all contain 'response'")
                if response_key is not None:
                    bad_rows = []
                    for row_i, row in enumerate(rows, start=1):
                        response = row[response_key]
                        if response is not None and not isinstance(response, str):
                            issues.append(f"{jf.name}: row {row_i} {response_key} must be a string or null")
                            continue
                        has_think = isinstance(response, str) and ("<think>" in response or "<thinking>" in response)
                        if expected_thinking and not has_think:
                            bad_rows.append(row_i)
                        if not expected_thinking and has_think:
                            bad_rows.append(row_i)
                    if bad_rows:
                        mode = "thinking" if expected_thinking else "no-thinking"
                        issues.append(
                            f"{jf.name}: expected {mode} but {len(bad_rows)}/{len(rows)} rows mismatch "
                            f"(first bad row: {bad_rows[0]})"
                        )
        except (json.JSONDecodeError, IOError) as e:
            issues.append(f"{jf.name}: failed to read: {e}")

    return len(issues) == 0, issues


def check_judge_errors(
    raw_results: list[dict],
    response_key: str = "response",
    judge_output_key: str = "judge_output",
    warn_threshold: float = 0.05,
    fail_threshold: float = 0.10,
) -> dict:
    """Check judge and generation error rates. Raises RuntimeError if too high.

    Returns dict with n_judge_errors, n_generation_failures, judge_error_rate
    for inclusion in EvalResult.details.
    """
    n_total = len(raw_results)
    if n_total == 0:
        raise RuntimeError("No raw results to check.")
    n_generation_failures = sum(1 for r in raw_results if r.get(response_key) is None)
    n_with_responses = n_total - n_generation_failures
    n_judge_errors = sum(1 for r in raw_results if r.get(judge_output_key) is None and r.get(response_key) is not None)
    if n_with_responses == 0:
        raise RuntimeError(f"All {n_total} generations failed; no responses to judge.")
    generation_failure_rate = n_generation_failures / n_total
    judge_error_rate = n_judge_errors / n_with_responses

    if n_generation_failures > 0:
        print(f"  Generation failures: {n_generation_failures}/{n_total}")
    if generation_failure_rate > fail_threshold:
        raise RuntimeError(
            f"Generation failure rate too high: {n_generation_failures}/{n_total} "
            f"({generation_failure_rate:.1%}). Results unreliable."
        )
    if judge_error_rate > warn_threshold:
        print(f"\nWARNING: High judge error rate: {n_judge_errors}/{n_with_responses} ({judge_error_rate:.1%})")
    if judge_error_rate > fail_threshold:
        raise RuntimeError(
            f"Judge error rate too high: {n_judge_errors}/{n_with_responses} "
            f"({judge_error_rate:.1%}). Results unreliable. "
            f"Check API key and rate limits."
        )

    return {
        "n_judge_errors": n_judge_errors,
        "n_generation_failures": n_generation_failures,
        "judge_error_rate": round(judge_error_rate, 4),
    }


@dataclass
class EvalResult:
    """Standardized evaluation result."""

    name: str
    primary_metric: float
    primary_metric_name: str
    ci_low: float
    ci_high: float
    n_samples: int
    details: dict = field(default_factory=dict)
    breakdowns: dict = field(default_factory=dict)
    raw_results: list = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    git_commit: str = ""

    def __post_init__(self):
        if not self.git_commit:
            self.git_commit = get_git_hash()

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, output_dir: Path):
        """Save result to output directory."""
        if not self.raw_results:
            raise ValueError("EvalResult.save requires non-empty raw_results for auditability")
        if len(self.raw_results) != self.n_samples:
            raise ValueError(f"EvalResult.save expected {self.n_samples} raw_results, got {len(self.raw_results)}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary = {
            "name": self.name,
            "primary_metric": self.primary_metric,
            "primary_metric_name": self.primary_metric_name,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_samples": self.n_samples,
            "details": self.details,
            "breakdowns": self.breakdowns,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
        }
        with open(output_dir / f"{self.name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save raw results
        with open(output_dir / f"{self.name}_results.jsonl", "w") as f:
            for r in self.raw_results:
                f.write(json.dumps(r) + "\n")


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion.

    Wilson CI is more robust than normal approximation for:
    - Small sample sizes
    - Proportions near 0 or 1

    Args:
        p: Observed proportion (successes / n)
        n: Sample size
        z: Z-score for confidence level (1.96 for 95%)

    Returns:
        (ci_low, ci_high) tuple
    """
    if n == 0:
        raise ValueError("wilson_ci requires n > 0")

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

    return max(0.0, center - margin), min(1.0, center + margin)


def clustered_ci(cluster_rates: list[float], z: float = 1.96, clamp: bool = True) -> tuple[float, float, float]:
    """Compute CI for a mean of cluster-level values.

    The interval is mean +/- multiplier * standard error. Below 30 clusters the multiplier is a
    second-order expansion of the t quantile, z + (z**3 + z) / (4 * df) with df = n - 1; from 30
    clusters up it is z (1.96).

    Args:
        cluster_rates: List of values, one per cluster (e.g., per-question rates)
        z: Z-score for confidence level (1.96 for 95%)
        clamp: If True, clamp CI to [0, 1] (correct for proportions, wrong for
               difference scores that can be negative)

    Returns:
        (mean, ci_low, ci_high) tuple
    """
    n = len(cluster_rates)
    if n == 0:
        raise ValueError("clustered_ci requires at least one cluster")
    mean = sum(cluster_rates) / n
    if n < MIN_CLUSTER_COUNT:
        raise ValueError(f"clustered_ci requires at least {MIN_CLUSTER_COUNT} clusters, got {n}")
    variance = sum((r - mean) ** 2 for r in cluster_rates) / (n - 1)
    se = math.sqrt(variance / n)
    if n < 30:
        df = n - 1
        t_crit = z + (z**3 + z) / (4 * df)
    else:
        t_crit = z
    lo = mean - t_crit * se
    hi = mean + t_crit * se
    if clamp:
        lo = max(0.0, lo)
        hi = min(1.0, hi)
    return mean, lo, hi


def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths.CODE_ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()[:8]
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        raise RuntimeError(f"Could not get git hash: {e}") from e


def verify_thinking_param(tokenizer, param_name: str) -> bool:
    """Verify that a DECLARED thinking parameter actually affects this tokenizer's
    chat template.

    Jinja chat templates silently ignore unknown kwargs, so a wrong declaration
    (e.g. thinking_param="enable_thinking" for a model that calls it something else)
    would otherwise fail silently — the model would run in its default thinking mode
    while results claim otherwise. This probe is deterministic: it renders the
    template with the param True/False and checks that the output changes.

    This never *selects* a parameter from a candidate list — the caller declares the
    name; this only checks the claim.
    """
    test_messages = [{"role": "user", "content": "test"}]
    default_output = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    try:
        output_true = tokenizer.apply_chat_template(
            test_messages, tokenize=False, add_generation_prompt=True, **{param_name: True}
        )
        output_false = tokenizer.apply_chat_template(
            test_messages, tokenize=False, add_generation_prompt=True, **{param_name: False}
        )
    except TypeError:
        return False
    return output_true != default_output or output_false != default_output


def is_api_model(model_path: str) -> bool:
    """Check if model_path refers to an API model (not a local/HuggingFace model).

    Router-style prefixes come from src.keys.OPENROUTER_MODEL_PREFIXES, the single
    source of truth, so the routing decision (create_generator) and this check never
    drift apart.
    """
    return (
        model_path.startswith("claude")
        or is_openrouter_model(model_path)
        or model_path.startswith("http://")
        or model_path.startswith("https://")
    )


def create_generator(
    model_path: str, **kwargs
) -> Union["AnthropicGenerator", "OpenRouterGenerator", "VLLMServerGenerator", "TinkerGenerator"]:
    """Auto-detect and create the appropriate generator from model path.

    Args:
        model_path: Model identifier.
            - "tinker://..." → TinkerGenerator (a trained LoRA checkpoint, sampled directly)
            - "http://..." or "https://..." → VLLMServerGenerator (OpenAI-compat endpoint)
            - "claude-*" → Anthropic API (use_batch=True → Message Batches API, 50% cheaper)
            - an id starting with one of src.keys.OPENROUTER_MODEL_PREFIXES → OpenRouter
        **kwargs: Passed through to the generator constructor. use_batch=True routes
            claude models through BatchAnthropicGenerator; it is an error for other
            model types (no silent fallback to the expensive path). provider (OpenRouter
            only) pins a single upstream provider with allow_fallbacks=False. backend="tinker"
            forces the Tinker backend for a bare HF base id (base_model=), which is otherwise
            ambiguous with a vLLM model; a tinker:// checkpoint routes to Tinker automatically.
            TinkerGenerator additionally requires base_model, tokenizer_id, renderer_name.

    Returns:
        Generator instance.
    """
    use_batch = kwargs.pop("use_batch", False)
    if use_batch and not model_path.startswith("claude"):
        raise ValueError(
            f"use_batch=True is only supported for Anthropic API models, got {model_path!r}. "
            f"Refusing to fall back silently to the non-batch path."
        )
    backend = kwargs.pop("backend", None)
    if backend not in (None, "tinker"):
        raise ValueError(f"unknown backend {backend!r}; only 'tinker' is an explicit backend override.")
    provider = kwargs.pop("provider", None)
    if provider and (model_path.startswith("http") or model_path.startswith("claude")):
        raise ValueError(f"provider routing is OpenRouter-only; got provider={provider!r} for {model_path!r}.")
    # Tinker: a tinker:// checkpoint routes automatically; a base HF id needs backend="tinker"
    # (a bare HF id is otherwise ambiguous with a vLLM model, so it is not auto-routed).
    if model_path.startswith("tinker://") or backend == "tinker":
        if provider:
            raise ValueError(f"provider routing is OpenRouter-only; got provider={provider!r} for a Tinker model.")
        return TinkerGenerator(model_path=model_path, **kwargs)
    if model_path.startswith("http://") or model_path.startswith("https://"):
        return VLLMServerGenerator(base_url=model_path, **kwargs)
    elif model_path.startswith("claude"):
        from src.keys import require_standard_claude

        if use_batch:
            from src.batch_api import BatchAnthropicGenerator

            require_standard_claude(model_path, "batch")
            kwargs.pop("max_concurrent", None)  # batch API has no client-side concurrency
            return BatchAnthropicGenerator(model_name=model_path, **kwargs)
        return AnthropicGenerator(model_name=model_path, **kwargs)
    elif is_openrouter_model(model_path):
        return OpenRouterGenerator(model_name=model_path, provider=provider, **kwargs)
    else:
        raise ValueError(
            f"Unrecognized model path: {model_path!r}. Supported are "
            f"claude-* (Anthropic), router-style prefixes "
            f"({', '.join(OPENROUTER_MODEL_PREFIXES)}) via OpenRouter, "
            f"http(s):// vLLM server URLs, tinker:// checkpoints, and a base HF id with "
            f"backend='tinker' (TinkerGenerator)."
        )


class AnthropicGenerator:
    """Wrapper for Anthropic API generation.

    seed is an error (the Anthropic API has no seed parameter). top_p is passed
    through when set, but Claude 4+ rejects temperature and top_p together — set
    temperature explicitly when using top_p. With thinking=True the API requires
    temperature=1.0 (applied) and a thinking token budget derived from max_tokens;
    top_p and prefill_tag are incompatible with thinking and raise. prefill_tag is
    sent as an assistant-turn prefill (supported on pre-4.6 models; newer models
    400 loudly; no trailing whitespace allowed by the API).
    """

    def __init__(
        self,
        model_name: str = "claude-haiku-4-5-20251001",
        max_concurrent: int = 20,
        betas: list[str] = None,
        api_key: str = None,
        allow_truncation: bool = False,
    ):
        self.model_name = model_name
        self.model_path = model_name  # For cache key compatibility
        self.max_concurrent = max_concurrent
        self.supports_thinking = True
        self.api_key = api_key
        self.betas = betas or []
        # Truncation stays fatal by default: for a single-shot eval a cut-off response is a
        # silent scoring error, and one bad row must not be scored as if it were an answer.
        # A multi-attempt scaffold is the exception -- there, hitting the output ceiling is a
        # failed attempt the loop already knows how to handle (feed the error back, try again),
        # and killing the batch throws away every sibling response in the same round. Callers
        # that set this MUST read last_stop_reasons and report the truncation rate; the fact is
        # still recorded per row, it is just no longer an exception.
        self.allow_truncation = allow_truncation

    def load(self):
        pass  # No-op, API doesn't need loading

    def generate_chat(
        self,
        conversations: list[list[dict]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        seed: int = None,
        thinking: bool = False,
        prefill_tag: str = None,
        on_response: callable = None,
    ) -> list[str]:
        """Generate completions via Anthropic API with async concurrency.

        Args:
            on_response: Optional callback(idx, response) called as each response completes.
                         Use for incremental caching so partial progress survives job kills.
        """
        reject_unsupported_params("AnthropicGenerator", {"seed": (seed, None)})
        if thinking:
            reject_unsupported_params(
                "AnthropicGenerator with thinking=True",
                {"top_p": (top_p, 1.0), "prefill_tag": (prefill_tag, None)},
            )
        if top_p != 1.0 and temperature != 1.0:
            raise ValueError(
                "The Anthropic API rejects temperature and top_p together (verified live on "
                "Haiku 4.5). Pass temperature=1.0 explicitly when using top_p — omitting "
                "temperature is equivalent to 1.0 on the API side."
            )
        return _run_async(
            self._generate_chat_async(conversations, max_tokens, temperature, thinking, on_response, top_p, prefill_tag)
        )

    async def _generate_chat_async(
        self,
        conversations: list[list[dict]],
        max_tokens: int,
        temperature: float,
        thinking: bool = False,
        on_response: callable = None,
        top_p: float = 1.0,
        prefill_tag: str = None,
    ) -> list[str]:
        import anthropic

        from src.keys import async_anthropic_client

        client_kwargs = {}
        if self.betas:
            client_kwargs["default_headers"] = {"anthropic-beta": ",".join(self.betas)}
        if self.api_key:
            client = anthropic.AsyncAnthropic(api_key=self.api_key, **client_kwargs)
        else:
            client = async_anthropic_client(**client_kwargs)
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = [None] * len(conversations)
        stop_reasons = [None] * len(conversations)
        error_counts = {}
        done_count = 0

        async def generate_one(idx: int, messages: list[dict]):
            nonlocal done_count
            system_content = None
            api_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    if system_content is not None:
                        raise ValueError("AnthropicGenerator received multiple system messages")
                    system_content = msg["content"]
                else:
                    api_messages.append(msg)
            if prefill_tag:
                # Assistant-turn prefill: the model continues from this text.
                # Pre-4.6 models only; the API rejects trailing whitespace.
                api_messages = api_messages + [{"role": "assistant", "content": prefill_tag}]

            kwargs = {
                "model": self.model_name,
                "messages": api_messages,
                "max_tokens": max_tokens,
            }
            if system_content is not None:
                kwargs["system"] = system_content

            if is_adaptive_thinking_model(self.model_name):
                # These model ids take no sampling params; adaptive or disabled thinking only.
                kwargs.update(adaptive_thinking_params(self.model_name, thinking))
            elif thinking:
                # Extended thinking: temperature must be 1, budget_tokens controls thinking length
                # API requires max_tokens > budget_tokens, so bump max_tokens if needed
                budget_tokens = max(1024, min(max_tokens // 2, 10000))
                if max_tokens <= budget_tokens:
                    kwargs["max_tokens"] = budget_tokens + max(max_tokens, 1024)
                kwargs["temperature"] = 1.0
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
            elif top_p != 1.0:
                # API rejects temperature and top_p together; omitting temperature == 1.0.
                kwargs["top_p"] = top_p
            else:
                kwargs["temperature"] = temperature

            async with semaphore:
                for attempt in range(MAX_RETRIES):
                    try:
                        # Try streaming first (avoids Anthropic 10-minute timeout on long responses)
                        try:
                            text = ""
                            thinking_text = ""
                            async with client.messages.stream(**kwargs) as stream:
                                async for event in stream:
                                    if hasattr(event, "type"):
                                        if event.type == "content_block_start":
                                            pass
                                        elif event.type == "content_block_delta":
                                            if hasattr(event.delta, "thinking"):
                                                thinking_text += event.delta.thinking
                                            elif hasattr(event.delta, "text"):
                                                text += event.delta.text
                                        elif event.type == "message_delta":
                                            # stop_reason arrives here; "max_tokens" means
                                            # the response was truncated
                                            if getattr(event.delta, "stop_reason", None):
                                                stop_reasons[idx] = event.delta.stop_reason

                            if thinking_text:
                                results[idx] = f"<think>\n{thinking_text}\n</think>\n\n{text}"
                            elif thinking and not is_adaptive_thinking_model(self.model_name):
                                # Legacy budget-based thinking: thinking=True guarantees thinking
                                # text, so silence is a real failure. Adaptive-thinking models
                                # are exempt: they decide per prompt whether to think at all,
                                # and the API may return an empty thinking block.
                                raise ResponseInvariantError(
                                    "Anthropic thinking=True returned no visible thinking text"
                                )
                            elif prefill_tag:
                                # vLLM-convention parity: stored response includes the prefill.
                                results[idx] = f"{prefill_tag}\n{text}"
                            else:
                                results[idx] = text
                        except (IndexError, KeyError):
                            # Streaming accumulator bug with some models' thinking blocks —
                            # fall back to non-streaming
                            resp = await client.messages.create(**kwargs)
                            stop_reasons[idx] = resp.stop_reason
                            text = ""
                            thinking_text = ""
                            for block in resp.content:
                                if block.type == "thinking":
                                    thinking_text += block.thinking
                                elif block.type == "text":
                                    text += block.text
                            if thinking_text:
                                results[idx] = f"<think>\n{thinking_text}\n</think>\n\n{text}"
                            elif thinking and not is_adaptive_thinking_model(self.model_name):
                                # Legacy budget-based thinking: thinking=True guarantees thinking
                                # text, so silence is a real failure. Adaptive-thinking models
                                # are exempt: they decide per prompt whether to think at all,
                                # and the API may return an empty thinking block.
                                raise ResponseInvariantError(
                                    "Anthropic thinking=True returned no visible thinking text"
                                )
                            elif prefill_tag:
                                results[idx] = f"{prefill_tag}\n{text}"
                            else:
                                results[idx] = text
                        if not self.allow_truncation:
                            _raise_if_truncated("Anthropic", stop_reasons[idx])
                        break  # Success — exit retry loop

                    except (
                        anthropic.APIError,
                        anthropic.APIConnectionError,
                        httpx.TransportError,
                        httpx.TimeoutException,
                        TimeoutError,
                        OSError,
                    ) as e:
                        error_type = _classify_error(e)
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1

                        if not _should_retry(error_type):
                            print(f"WARNING: Anthropic {error_type} error for conversation {idx}, not retrying: {e}")
                            results[idx] = None
                            break

                        if attempt < MAX_RETRIES - 1:
                            delay = _retry_delay(error_type, attempt)
                            print(
                                f"WARNING: Anthropic {error_type} error for conversation {idx} "
                                f"(attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay:.0f}s: {e}"
                            )
                            await asyncio.sleep(delay)
                        else:
                            print(
                                f"WARNING: Anthropic {error_type} error for conversation {idx} "
                                f"(attempt {attempt + 1}/{MAX_RETRIES}), giving up: {e}"
                            )
                            results[idx] = None

                if on_response and results[idx] is not None:
                    on_response(idx, results[idx])

                done_count += 1
                if done_count % 50 == 0:
                    print(f"  Generated {done_count}/{len(conversations)} responses", flush=True)

        tasks = [generate_one(i, conv) for i, conv in enumerate(conversations)]
        await asyncio.gather(*tasks)

        # Aligned with the conversations just generated; "max_tokens" = truncated.
        self.last_stop_reasons = stop_reasons

        _check_deterministic_failures("AnthropicGenerator", error_counts)
        n_failed = sum(1 for r in results if r is None)
        if n_failed > 0:
            error_summary = ", ".join(f"{k}={v}" for k, v in sorted(error_counts.items()))
            print(
                f"WARNING: {n_failed}/{len(results)} Anthropic API calls returned None "
                f"({n_failed / len(results):.0%}) — errors: {error_summary}"
            )
        if n_failed > max(len(results) * FAILURE_ABORT_FRACTION, FAILURE_ABORT_MIN):
            n_overload = error_counts.get("overload", 0)
            if n_overload > n_failed * 0.5:
                raise OverloadError(
                    f"API overloaded: {n_overload}/{n_failed} failures were 429/529. "
                    f"Wait before retrying — don't requeue immediately."
                )
            raise RuntimeError(
                f"Too many Anthropic API failures: {n_failed}/{len(results)} "
                f"({n_failed / len(results):.0%}) exceeded {FAILURE_ABORT_FRACTION:.0%} threshold — "
                f"errors: {', '.join(f'{k}={v}' for k, v in sorted(error_counts.items()))}"
            )
        return results

    def cleanup(self):
        pass


# --- Models with adaptive thinking and no sampling params -------------------------------------
# Substring match so aliases and dated ids both resolve. For these model ids the request sends no
# temperature/top_p/top_k and no budget-based thinking config ({type:enabled,budget_tokens}).
# Thinking is adaptive; it is turned off with {type:disabled}.
_ADAPTIVE_THINKING_MODELS = (
    "claude-opus-4-7",
    "claude-opus-4-8",
)


def is_adaptive_thinking_model(model: str) -> bool:
    """True for the model ids that take no sampling params and adaptive-only thinking."""
    return any(tag in model for tag in _ADAPTIVE_THINKING_MODELS)


def adaptive_thinking_params(model: str, thinking: bool) -> dict:
    """Sampling/thinking kwargs for an adaptive-thinking model: never temperature/top_p; thinking
    adaptive when requested, disabled when not. The caller must have checked
    is_adaptive_thinking_model(model) first.

    `display: "summarized"` is explicit so that thinking blocks come back with text; the
    generators wrap that text in <think> tags."""
    if thinking:
        return {"thinking": {"type": "adaptive", "display": "summarized"}}
    return {"thinking": {"type": "disabled"}}


def get_cache_key(
    model: str, prompt: str, temperature: float, seed: int | None = None, max_tokens: int = 32768, **kwargs
) -> str:
    """Generate cache key for a generation request.

    Note: max_tokens is included to prevent returning truncated responses from cache.
    For this project, callers must pass framing_id and behavior_id (and system_prompt
    where applicable) via kwargs so conditions never collide.
    """
    content = f"{model}|{prompt}|{temperature}|{seed}|{max_tokens}|{sorted(kwargs.items())}"
    return hashlib.md5(content.encode()).hexdigest()


class ResponseCache:
    """Simple file-based response cache.

    Each entry stores the response text plus optional metadata (e.g. stop_reason),
    so generation-time facts like truncation survive cache resumes. Entries written
    before metadata existed load fine with empty metadata.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}
        self._meta = {}
        self._load_cache()

    def _load_cache(self):
        """Load all cached responses."""
        corrupt_paths = []
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(cache_file.read_text())
                self._cache[cache_file.stem] = data["response"]
                self._meta[cache_file.stem] = {k: v for k, v in data.items() if k != "response"}
            except (json.JSONDecodeError, KeyError):
                corrupt_paths.append(cache_file)
        if corrupt_paths:
            paths = ", ".join(str(path) for path in corrupt_paths)
            raise ValueError(f"Corrupt cache file(s) in {self.cache_dir}: {paths}")

    def get(self, key: str) -> Optional[str]:
        """Get cached response."""
        return self._cache.get(key)

    def get_meta(self, key: str) -> dict:
        """Get cached metadata (e.g. stop_reason). Empty dict for pre-metadata entries."""
        return self._meta.get(key, {})

    def set(self, key: str, response: str, meta: dict = None):
        """Cache a response, optionally with metadata stored alongside it."""
        cache_file = self.cache_dir / f"{key}.json"
        payload = {"response": response, **(meta or {})}
        tmp_file = cache_file.with_name(f"{cache_file.name}.tmp")
        try:
            tmp_file.write_text(json.dumps(payload))
            os.replace(tmp_file, cache_file)
        except OSError as e:
            tmp_file.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to write cache file {cache_file}: {e}") from e
        self._cache[key] = response
        self._meta[key] = meta or {}

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)


class VLLMServerGenerator:
    """Hit a running vLLM server via its OpenAI-compatible endpoint.

    Usage:
        gen = VLLMServerGenerator(base_url="http://localhost:8000/v1")
        gen.load()  # discovers model name
        responses = gen.generate_chat(conversations, ...)

    The base_url should point to the /v1 endpoint (e.g. http://localhost:8000/v1).
    If it doesn't end with /v1, it's appended automatically.
    """

    def __init__(self, base_url: str, max_concurrent: int = 64, thinking_param: str | None = None):
        """thinking_param: the chat-template kwarg controlling thinking for this model
        (e.g. "enable_thinking" for Qwen3), or None meaning "this model has no thinking
        switch". It must be DECLARED by the caller — there is no auto-detection.
        load() verifies the declaration against the actual chat template when
        transformers is installed, and raises on a wrong declaration."""
        # Normalize URL
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.max_concurrent = max_concurrent
        self.model_name = None
        self.model_path = None  # Set after load(), used for cache keys
        self.max_model_len = None  # Set after load(), from server /models endpoint
        self.thinking_param = thinking_param
        self.supports_thinking = thinking_param is not None

    def load(self):
        """Discover model name and max_model_len from the server; verify thinking_param."""
        if self.model_name is not None:
            return
        try:
            resp = httpx.get(f"{self.base_url}/models", timeout=10)
            resp.raise_for_status()
            model_info = resp.json()["data"][0]
            self.model_name = model_info["id"]
            self.model_path = self.model_name
            self.max_model_len = model_info.get("max_model_len")
            len_str = f", max_model_len={self.max_model_len}" if self.max_model_len else ""
            print(f"Connected to vLLM server: {self.base_url} (model: {self.model_name}{len_str})")
        except (httpx.HTTPError, httpx.TimeoutException, KeyError, IndexError) as e:
            raise RuntimeError(f"Failed to connect to vLLM server at {self.base_url}: {e}")

        # Verify the declared thinking param against the actual chat template.
        # Jinja templates silently ignore unknown kwargs, so an unverified wrong
        # declaration would silently run the model in its default thinking mode.
        if self.thinking_param is not None:
            try:
                from transformers import AutoTokenizer
            except ImportError as e:
                raise RuntimeError(
                    f"thinking_param={self.thinking_param!r} was declared but could not be verified "
                    "because transformers is not installed."
                ) from e
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if not verify_thinking_param(tokenizer, self.thinking_param):
                raise ValueError(
                    f"Declared thinking_param={self.thinking_param!r} has no effect on the chat "
                    f"template of {self.model_name!r}. Fix the declaration — a wrong one would "
                    f"silently run the model in its default thinking mode."
                )
            print(f"Verified thinking_param {self.thinking_param!r} against chat template")

    def generate_chat(
        self,
        conversations: list[list[dict]],
        max_tokens: int = 32768,
        temperature: float = 0.7,
        top_p: float = 1.0,
        seed: int = None,
        thinking: bool = False,
        prefill_tag: str = None,
        on_response: callable = None,
    ) -> list[str]:
        """Generate completions via vLLM server's OpenAI-compat endpoint.

        Note: thinking does NOT constrain temperature here (the Anthropic API forces
        temperature=1.0 with extended thinking). Hold temperature constant across
        backends when comparing thinking runs.
        """
        reject_unsupported_params("VLLMServerGenerator", {"prefill_tag": (prefill_tag, None)})
        if thinking and self.thinking_param is None:
            raise ValueError(
                "thinking=True requested but no thinking_param was declared for this server's "
                "model. Pass thinking_param (e.g. 'enable_thinking' for Qwen3) to "
                "VLLMServerGenerator/create_generator — there is no auto-detection."
            )
        if self.model_name is None:
            self.load()

        return _run_async(
            self._generate_async(conversations, max_tokens, temperature, seed, thinking, on_response, top_p)
        )

    @staticmethod
    def _estimate_prompt_tokens(messages: list[dict]) -> int:
        """Rough token estimate for chat messages (~4 chars/token + overhead per message)."""
        chars = sum(len(m.get("content", "")) for m in messages)
        # ~4 chars per token for English, plus ~4 tokens overhead per message for role/formatting
        return chars // 4 + len(messages) * 4

    def _cap_max_tokens(self, messages: list[dict], max_tokens: int) -> int:
        """Cap max_tokens so prompt + generation fits within max_model_len."""
        if self.max_model_len is None:
            return max_tokens
        est_prompt = self._estimate_prompt_tokens(messages)
        headroom = self.max_model_len - est_prompt
        if headroom <= 0:
            raise ValueError(
                f"Estimated prompt tokens ({est_prompt}) already exceed max_model_len ({self.max_model_len})"
            )
        if max_tokens > headroom:
            raise ValueError(
                f"Requested max_tokens ({max_tokens}) exceeds available headroom ({headroom}) "
                f"for max_model_len={self.max_model_len}, est_prompt_tokens={est_prompt}"
            )
        return max_tokens

    async def _generate_async(
        self, conversations, max_tokens, temperature, seed, thinking, on_response=None, top_p=1.0
    ):
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = [None] * len(conversations)
        stop_reasons = [None] * len(conversations)
        error_counts = {}
        done_count = 0
        n_total = len(conversations)

        # Log once if we'll be capping any requests
        if self.max_model_len and max_tokens > self.max_model_len:
            logger.warning(
                "Requested max_tokens (%d) exceeds server max_model_len (%d); "
                "will cap per-request based on estimated prompt size",
                max_tokens,
                self.max_model_len,
            )

        async with httpx.AsyncClient(timeout=300) as client:

            async def generate_one(idx, messages):
                nonlocal done_count
                capped_max_tokens = self._cap_max_tokens(messages, max_tokens)
                body = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": capped_max_tokens,
                    "temperature": temperature,
                }
                if seed is not None:
                    body["seed"] = seed
                if top_p != 1.0:
                    body["top_p"] = top_p
                if self.thinking_param is not None:
                    body["chat_template_kwargs"] = {self.thinking_param: thinking}

                async with semaphore:
                    for attempt in range(MAX_RETRIES):
                        try:
                            resp = await client.post(
                                f"{self.base_url}/chat/completions",
                                json=body,
                            )
                            resp.raise_for_status()
                            data = resp.json()
                            choice = data["choices"][0]
                            results[idx] = choice["message"]["content"]
                            # OpenAI-compat: "length" means truncated at max_tokens
                            stop_reasons[idx] = choice.get("finish_reason")
                            _raise_if_truncated("vLLM", stop_reasons[idx])
                            break  # Success — exit retry loop
                        except (
                            httpx.HTTPError,
                            httpx.TransportError,
                            httpx.TimeoutException,
                            TimeoutError,
                            OSError,
                            KeyError,
                        ) as e:
                            error_type = _classify_error(e)
                            error_counts[error_type] = error_counts.get(error_type, 0) + 1

                            if not _should_retry(error_type):
                                print(f"WARNING: vLLM {error_type} error for conversation {idx}, not retrying: {e}")
                                results[idx] = None
                                break

                            if attempt < MAX_RETRIES - 1:
                                delay = _retry_delay(error_type, attempt)
                                print(
                                    f"WARNING: vLLM {error_type} error for conversation {idx} "
                                    f"(attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay:.0f}s: {e}"
                                )
                                await asyncio.sleep(delay)
                            else:
                                print(
                                    f"WARNING: vLLM {error_type} error for conversation {idx} "
                                    f"(attempt {attempt + 1}/{MAX_RETRIES}), giving up: {e}"
                                )
                                results[idx] = None

                if on_response and results[idx] is not None:
                    on_response(idx, results[idx])

                done_count += 1
                if done_count % 100 == 0 or done_count == n_total:
                    print(f"  Generated {done_count}/{n_total} responses")

            await asyncio.gather(*[generate_one(i, c) for i, c in enumerate(conversations)])

        self.last_stop_reasons = stop_reasons

        _check_deterministic_failures("VLLMServerGenerator", error_counts)
        n_failed = sum(1 for r in results if r is None)
        if n_failed > 0:
            error_summary = ", ".join(f"{k}={v}" for k, v in sorted(error_counts.items()))
            print(f"WARNING: {n_failed}/{n_total} vLLM server calls returned None — errors: {error_summary}")
        if n_failed > max(n_total * FAILURE_ABORT_FRACTION, FAILURE_ABORT_MIN):
            n_overload = error_counts.get("overload", 0)
            if n_overload > n_failed * 0.5:
                raise OverloadError(
                    f"vLLM server overloaded: {n_overload}/{n_failed} failures were 429/529. Wait before retrying."
                )
            raise RuntimeError(
                f"Too many vLLM server failures: {n_failed}/{n_total} ({n_failed / n_total:.0%}) "
                f"exceeded {FAILURE_ABORT_FRACTION:.0%} threshold"
            )
        return results

    def cleanup(self):
        pass  # Server keeps running


class OpenRouterGenerator:
    """Wrapper for OpenRouter API generation.

    thinking=True uses OpenRouter's unified reasoning parameter ({"enabled": true});
    reasoning text comes back in message.reasoning and is wrapped in <think> tags for
    cross-backend parity. seed and top_p are passed through when set (determinism and
    support are provider-dependent — OpenRouter silently ignores params the routed
    provider doesn't support, so check_thinking_sanity is the output-level backstop).
    prefill_tag has no reliable OpenRouter equivalent and raises.
    """

    def __init__(self, model_name: str, max_concurrent: int = 20, provider: str | None = None):
        self.model_name = model_name
        self.model_path = model_name
        self.max_concurrent = max_concurrent
        self.supports_thinking = True
        # Pin generation to one OpenRouter provider (allow_fallbacks=False) when set, so a model
        # with a single clean provider is not silently routed to a degraded or misconfigured one.
        # Names match OpenRouter's provider names (e.g. "DeepInfra").
        self.provider = provider

    def load(self):
        pass

    def _build_body(self, messages, max_tokens, temperature, top_p, seed, thinking) -> dict:
        body = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_p != 1.0:
            body["top_p"] = top_p
        if seed is not None:
            body["seed"] = seed
        # Always send the reasoning parameter explicitly, both ways. Omitting it leaves the
        # choice to the upstream provider's chat template, which varies: a hybrid-reasoning
        # model may then emit inline <think> on some calls and not others. The thinking
        # behavior must be declared, never provider-defaulted.
        body["reasoning"] = {"enabled": bool(thinking)}
        if self.provider:
            body["provider"] = {"order": [self.provider], "allow_fallbacks": False}
        return body

    @staticmethod
    def _extract_text(message: dict) -> str:
        """Extract response text, wrapping any reasoning in <think> tags for parity
        with the other backends."""
        # .get (not message["content"]): an absent content key is the same degenerate case as
        # an explicit null — surface it as this ValueError (the declared null-content path,
        # counted toward the error rate) rather than a raw KeyError swept into the broad retry
        # catch, where a deterministic response-shape mismatch would hide as generic retries.
        content = message.get("content")
        if content is None:
            raise ValueError(
                "API returned null/absent content (reasoning tokens may have exhausted "
                "max_tokens, or the provider returned a non-standard message shape)"
            )
        reasoning = message.get("reasoning")
        if reasoning:
            return f"<think>\n{reasoning}\n</think>\n\n{content}"
        return content

    def generate_chat(
        self,
        conversations: list[list[dict]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        seed: int = None,
        thinking: bool = False,
        prefill_tag: str = None,
        on_response: callable = None,
    ) -> list[str]:
        """Generate completions via OpenRouter API with async concurrency."""
        reject_unsupported_params("OpenRouterGenerator", {"prefill_tag": (prefill_tag, None)})
        return _run_async(
            self._generate_async(conversations, max_tokens, temperature, on_response, top_p, seed, thinking)
        )

    async def _generate_async(
        self, conversations, max_tokens, temperature, on_response=None, top_p=1.0, seed=None, thinking=False
    ):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        headers = {"Authorization": f"Bearer {api_key}"}
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = [None] * len(conversations)
        stop_reasons = [None] * len(conversations)
        error_counts = {}
        done_count = 0

        async def generate_one(idx, messages):
            nonlocal done_count
            async with semaphore:
                for attempt in range(MAX_RETRIES):
                    try:
                        async with httpx.AsyncClient(headers=headers) as client:
                            resp = await client.post(
                                "https://openrouter.ai/api/v1/chat/completions",
                                json=self._build_body(messages, max_tokens, temperature, top_p, seed, thinking),
                                timeout=300.0,
                            )
                            resp.raise_for_status()
                            choice = resp.json()["choices"][0]
                            results[idx] = self._extract_text(choice["message"])
                            # OpenAI-compat: "length" means truncated at max_tokens
                            stop_reasons[idx] = choice.get("finish_reason")
                        _raise_if_truncated("OpenRouter", stop_reasons[idx])
                        break  # Success — exit retry loop
                    except (
                        httpx.HTTPError,
                        httpx.TransportError,
                        httpx.TimeoutException,
                        TimeoutError,
                        OSError,
                        ValueError,
                        KeyError,
                        IndexError,
                    ) as e:
                        # ValueError (null content) / KeyError / IndexError (response with no
                        # "choices", e.g. a provider error body) classify as "unknown" → retried.
                        error_type = _classify_error(e)
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1

                        if not _should_retry(error_type):
                            print(f"WARNING: OpenRouter {error_type} error for conversation {idx}, not retrying: {e}")
                            results[idx] = None
                            break

                        if attempt < MAX_RETRIES - 1:
                            delay = _retry_delay(error_type, attempt)
                            print(
                                f"WARNING: OpenRouter {error_type} error for conversation {idx} "
                                f"(attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay:.0f}s: {e}"
                            )
                            await asyncio.sleep(delay)
                        else:
                            print(
                                f"WARNING: OpenRouter {error_type} error for conversation {idx} "
                                f"(attempt {attempt + 1}/{MAX_RETRIES}), giving up: {e}"
                            )
                            results[idx] = None
                if on_response and results[idx] is not None:
                    on_response(idx, results[idx])
                done_count += 1
                if done_count % 50 == 0:
                    print(f"  Generated {done_count}/{len(conversations)} responses")

        await asyncio.gather(*[generate_one(i, c) for i, c in enumerate(conversations)])

        self.last_stop_reasons = stop_reasons

        _check_deterministic_failures("OpenRouterGenerator", error_counts)
        n_failed = sum(1 for r in results if r is None)
        if n_failed > 0:
            error_summary = ", ".join(f"{k}={v}" for k, v in sorted(error_counts.items()))
            print(f"WARNING: {n_failed}/{len(results)} OpenRouter API calls returned None — errors: {error_summary}")
        if n_failed > max(len(results) * FAILURE_ABORT_FRACTION, FAILURE_ABORT_MIN):
            n_overload = error_counts.get("overload", 0)
            if n_overload > n_failed * 0.5:
                raise OverloadError(
                    f"OpenRouter overloaded: {n_overload}/{n_failed} failures were 429/529. Wait before retrying."
                )
            raise RuntimeError(
                f"Too many OpenRouter API failures: {n_failed}/{len(results)} "
                f"({n_failed / len(results):.0%}) exceeded {FAILURE_ABORT_FRACTION:.0%} threshold"
            )
        return results

    def cleanup(self):
        pass


# Tinker sampling-client retry policy. Transient (429/5xx/connection) backs off; auth/billing
# 4xx are deterministic and must fail loudly rather than retry: a 402 billing block never
# clears on its own.
TINKER_MAX_RETRIES = 6
TINKER_BACKOFF_BASE_SEC = 2.0
TINKER_BACKOFF_CAP_SEC = 30.0
TINKER_DETERMINISTIC_STATUS = (400, 401, 402, 403, 404)


def tinker_failure_is_transient(e: Exception) -> bool:
    """Rate limits, connection errors and server-side request failures are worth retrying; a request
    failure Tinker categorises as the user's (a bad request) is not. A server-side sampling failure
    ("failed to sample after exhausting retries", category unknown or server) arrives in bursts lasting
    minutes and must not crash a run."""
    import tinker

    if isinstance(e, (tinker.RateLimitError, tinker.APIConnectionError)):
        return True
    if isinstance(e, tinker.RequestFailedError):
        return e.category is not tinker.types.RequestErrorCategory.User
    return False


# Renderer families whose reasoning is NOT delimited by a <think> block. gpt-oss uses the
# harmony format: reasoning goes in an `analysis` channel and the answer follows the literal
# `assistantfinal` marker, so a <think> prefill opens a tag the model will never close. Left
# unhandled, every response reads as truncated-mid-reasoning and the whole run is dropped.
# Declared here rather than sniffed from output, so an unknown renderer keeps the <think>
# assumption instead of silently guessing a format.
RENDERERS_WITHOUT_THINK_BLOCKS = ("gpt_oss",)


def renderer_uses_think_blocks(renderer_name):
    """False for renderer families that delimit reasoning some other way (see above).

    None means "not declared", which keeps the <think> assumption.
    """
    from src.thinking import renderer_uses_think_blocks as _canonical

    return _canonical(renderer_name)


def force_thinking_prompt(renderer, tokenizer, messages: list[dict], renderer_name=None):
    """Generation prompt with the assistant turn forced to open a <think> block, so the
    model reasons then closes </think> before answering. Some renderers already prefill
    it (nemotron3); for those (detected by the prompt tail) we don't double it. For those
    that don't (qwen3, whose template lets the model choose) we pass prefill='<think>\\n'
    via build_generation_prompt so thinking is on, not optional.

    Module-level because it is the ONE definition of this logic: TinkerGenerator uses it,
    and so must any sanctioned bespoke Tinker sampler (a harness needing per-sample token
    counts or a cross-adapter concurrency budget) — a hand-rolled build_generation_prompt
    call without it silently loses thinking on non-prefilling renderers.

    Pass `renderer_name` so a harmony-format family (gpt-oss) is returned unmodified: its
    reasoning is already on via the `Reasoning:` line in the renderer's own system prompt,
    and there is no think block to open. Omitting it assumes think blocks.
    """
    if not renderer_uses_think_blocks(renderer_name):
        return renderer.build_generation_prompt(messages)
    prompt = renderer.build_generation_prompt(messages)
    if "<think>" in tokenizer.decode(prompt.to_ints()[-8:]):
        return prompt
    return renderer.build_generation_prompt(messages, prefill="<think>\n")


#: The harmony header that opens the reasoning (analysis) channel of an assistant turn. The
#: gpt-oss tokenizer encodes `<|channel|>` and `<|message|>` in it as their special tokens.
HARMONY_ANALYSIS_HEADER = "<|channel|>analysis<|message|>"


def reasoning_prefill_prompt(renderer, messages: list[dict], renderer_name: str, prefill: str):
    """Generation prompt whose assistant turn already holds the start of its reasoning: the harmony
    analysis-channel header followed by `prefill`, which the model then continues. Harmony
    renderers only: a think-block renderer has no analysis channel to open, so it is refused."""
    if renderer_uses_think_blocks(renderer_name):
        raise ValueError(f"a reasoning prefill needs a harmony renderer; {renderer_name!r} uses think blocks")
    return renderer.build_generation_prompt(messages, prefill=HARMONY_ANALYSIS_HEADER + prefill)


@functools.lru_cache(maxsize=1)
def _harmony_encoding():
    """openai_harmony's gpt-oss encoding, loaded once per process: a load takes about 75 ms and
    runs synchronously, and the tool prompt is built once per turn inside async samplers. The first
    load in a fresh temporary directory downloads the tokenizer vocabulary."""
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def harmony_tool_messages(renderer, tools: list[dict], system_prompt: str) -> list[dict]:
    """The system and developer messages that open a tool-using gpt-oss conversation, with their
    text produced by `openai_harmony`, OpenAI's reference renderer for the format.

    The cookbook renderer's own tool prefix departs from that reference in two ways the model can
    see: it puts `// ` before only the first line of a tool description, so every further line
    (for R2E-Gym's file editor, "`view` displays...", "The `create` command...") sits in the tool
    namespace as bare text that reads like a declaration; and it carries the channel-routing line
    in a second system message. Here the reference renders the system message (the renderer's
    date and reasoning effort, plus the routing line) and the developer message (the instructions
    and every tool, each description line commented) as one conversation, and each message's body
    is cut from that rendering, so the text is the reference's byte for byte. The date follows the
    renderer: its fixed `current_date`, else today, as the cookbook does."""
    from openai_harmony import (
        Conversation,
        DeveloperContent,
        Message,
        ReasoningEffort,
        Role,
        SystemContent,
        ToolDescription,
    )
    from tinker_cookbook.renderers.gpt_oss import GptOssRenderer

    if not isinstance(renderer, GptOssRenderer):
        raise ValueError(f"tool calls are rendered for harmony (gpt-oss) renderers only, not {type(renderer).__name__}")
    if not renderer.use_system_prompt:
        raise ValueError("a harmony tool prompt needs the renderer's dated system prompt: use a *_reasoning renderer")
    if not tools:
        raise ValueError("harmony_tool_messages needs at least one tool")
    efforts = {"low": ReasoningEffort.LOW, "medium": ReasoningEffort.MEDIUM, "high": ReasoningEffort.HIGH}
    if renderer.reasoning_effort not in efforts:
        raise ValueError(f"unknown reasoning effort {renderer.reasoning_effort!r}; known: {sorted(efforts)}")
    date = renderer.current_date if renderer.current_date is not None else datetime.now().strftime("%Y-%m-%d")
    system = (
        SystemContent.new().with_reasoning_effort(efforts[renderer.reasoning_effort]).with_conversation_start_date(date)
    )
    developer = DeveloperContent.new().with_function_tools(
        [ToolDescription.new(t["name"], t["description"], parameters=t["parameters"]) for t in tools]
    )
    if system_prompt:
        developer = developer.with_instructions(system_prompt)
    encoding = _harmony_encoding()
    rendered = encoding.decode(
        encoding.render_conversation(
            Conversation.from_messages(
                [
                    Message.from_role_and_content(Role.SYSTEM, system),
                    Message.from_role_and_content(Role.DEVELOPER, developer),
                ]
            )
        )
    )
    system_head, developer_head, end = "<|start|>system<|message|>", "<|start|>developer<|message|>", "<|end|>"
    if not (rendered.startswith(system_head) and rendered.endswith(end) and rendered.count(developer_head) == 1):
        raise RuntimeError(f"unexpected openai_harmony rendering: {rendered[:200]!r}")
    system_text, developer_text = rendered[len(system_head) : -len(end)].split(end + developer_head)
    return [
        {"role": renderer._INTERNAL_SYSTEM_ROLE, "content": system_text},
        {"role": "developer", "content": developer_text},
    ]


def harmony_tool_prompt(renderer, messages: list[dict], tools: list[dict], system_prompt: str):
    """Generation prompt for one tool-using turn on a harmony (gpt-oss) renderer: the reference
    system and developer messages (`harmony_tool_messages`), then the conversation, rendered
    through the base builder so the renderer's own system prompt is not prepended a second time.
    Module-level because it is the one definition of this layout; any Tinker sampler that takes
    tools must use it."""
    from tinker_cookbook.renderers.base import Renderer

    return Renderer.build_generation_prompt(
        renderer, [*harmony_tool_messages(renderer, tools, system_prompt), *messages]
    )


def harmony_tool_conversation(renderer, tools: list[dict], system_prompt: str):
    """(renderer, opening messages) for a multi-turn tool conversation driven by a cookbook
    environment, which calls the renderer's own `build_generation_prompt` every turn. The messages
    are `harmony_tool_messages`; the renderer is a new one of the same class over the same
    tokenizer that prepends no system prompt, since the opening messages already carry the one
    system message. It is built directly, not through `get_renderer`, so it cannot be pickled: the
    cookbook rebuilds an unpickled renderer from its registered name, which would bring the second
    system prompt back, and a loud pickling error is the safe failure."""
    messages = harmony_tool_messages(renderer, tools, system_prompt)
    return type(renderer)(renderer.tokenizer), messages


@dataclass(frozen=True)
class SampledSequence:
    """One completion as the provider returned it: token ids, per-token logprobs, decoded text,
    and the provider's stop reason ("length" means the cap was hit)."""

    tokens: list[int]
    logprobs: list[float]
    text: str
    stop_reason: str


class PromptExceedsContextWindow(Exception):
    """The rendered prompt alone fills the model's context window, so not one token can be sampled.
    Raised before the request is sent. Tinker itself shortens a reply to the room left in the window
    and rejects only a prompt longer than the window; the caller decides how such a run ends."""

    def __init__(self, prompt_tokens: int, context_window: int):
        super().__init__(f"prompt of {prompt_tokens} tokens fills the {context_window}-token context window")
        self.prompt_tokens = prompt_tokens
        self.context_window = context_window


@dataclass(frozen=True)
class SampledPrompt:
    """Every completion sampled for one conversation, with the exact prompt token ids sent."""

    prompt_tokens: list[int]
    sequences: list[SampledSequence]


class TinkerGenerator:
    """Sample a Tinker base model or LoRA checkpoint directly (no vLLM export/serve).

    The Tinker sampling client has no place in the API-model routing the other generators
    use, so it is its own backend: a `tinker://...:train:.../sampler_weights/...` model_path
    is a trained adapter (sampled via model_path=); any other model_path is treated as a base
    HF id (sampled via base_model=). Either way the tokenizer + renderer come from the BASE
    model (an adapter tokenizes with its base), so `base_model`, `tokenizer_id`, and
    `renderer_name` are required — no silent defaults.

    Thinking is DECLARED, never left to the model:
      - thinking=True forces an open `<think>` block in the prompt so the model reasons then
        closes `</think>` before answering (see force_thinking_prompt). Renderers differ --
        nemotron3 prefills `<think>` itself; qwen3's template leaves it to the model, so
        without the forced prefill a trained model can skip reasoning entirely -- so we
        ensure the open tag ourselves rather than trust the renderer.
      - thinking=False renders with the model's own `<renderer_name>_disable_thinking`
        cookbook renderer, whose generation suffix closes an empty think block exactly the
        way the model saw nothinking turns in training (conventions differ per model:
        nemotron3 emits `<think></think>` with no trailing newline, qwen3_5
        `<think>\\n\\n</think>\\n\\n` -- which is why this is a renderer choice and never a
        hand-built prefill string). A renderer without a disable variant fails loudly at
        first use.

    A truncated (never-closed) thinking response is recorded and treated as a generation
    failure downstream. seed and top_p ARE honored (Tinker SamplingParams supports both),
    unlike the Anthropic path. Truncation at max_tokens is recorded in last_stop_reasons but
    not raised (thinking models legitimately run the budget; usability is scored downstream).
    """

    def __init__(
        self,
        model_path: str,
        base_model: str = None,
        tokenizer_id: str = None,
        renderer_name: str = None,
        max_concurrent: int = 32,
        renderer_current_date: str = None,
    ):
        self.model_path = model_path  # cache-key discriminator (checkpoint path or base id)
        # Pinned into a dated (harmony) system prompt at load(); unset means the fixed default date.
        self.renderer_current_date = renderer_current_date
        if model_path.startswith("tinker://"):
            self.checkpoint = model_path
            if not base_model:
                raise ValueError(
                    "TinkerGenerator: model_path is a tinker:// checkpoint, so base_model (the HF id "
                    "the adapter was trained on) is required for the tokenizer + renderer."
                )
            self.base_model = base_model
        else:
            self.checkpoint = None
            self.base_model = model_path
        if not tokenizer_id or not renderer_name:
            raise ValueError("TinkerGenerator requires tokenizer_id and renderer_name (no silent defaults).")
        self.tokenizer_id = tokenizer_id
        self.renderer_name = renderer_name
        self.max_concurrent = max_concurrent
        self.supports_thinking = True
        self.last_stop_reasons: list = []
        self._tokenizer = None
        self._renderer = None
        self._renderer_nothink = None
        self._service = None
        self._client = None
        self._context_window = None
        self._context_window_lock = asyncio.Lock()

    def load(self):
        import tinker

        from src.training.tinker_common import apply_renderer_date, load_renderer, load_tokenizer, owner_tag

        self._tokenizer = load_tokenizer(self.tokenizer_id)
        self._renderer = load_renderer(self.base_model, self.renderer_name, self._tokenizer)
        apply_renderer_date(self._renderer, self.renderer_current_date)
        self._service = tinker.ServiceClient(user_metadata=owner_tag())

    @property
    def max_token_id(self) -> int:
        """Largest token id training accepts, from the loaded tokenizer's length (base vocab plus
        added special tokens) minus one. gpt-oss's model and full harmony tokenizer span 201,088
        ids, but the loaded tokenizer defines 200,019; ids 200,019-201,087 are reserved special
        slots the sampler can emit and forward_backward rejects. rollout drops completions that
        contain them."""
        if self._tokenizer is None:
            raise RuntimeError("TinkerGenerator.load() must be called before max_token_id.")
        return len(self._tokenizer) - 1

    def generate_chat(
        self,
        conversations: list[list[dict]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        seed: int = None,
        thinking: bool = False,
        prefill_tag: str = None,
        on_response: callable = None,
    ) -> list[str]:
        reject_unsupported_params("TinkerGenerator", {"prefill_tag": (prefill_tag, None)})
        if self._service is None:
            raise RuntimeError("TinkerGenerator.load() must be called before generate_chat().")
        return _run_async(
            self._generate_async(conversations, max_tokens, temperature, top_p, seed, thinking, on_response)
        )

    def _renderer_for(self, thinking: bool):
        """The thinking renderer, or the model's own disable-thinking variant.

        The disable variant is loaded lazily on first nothinking call; an unknown
        `<name>_disable_thinking` renderer raises there, loudly, rather than at load()."""
        if thinking:
            return self._renderer
        if self._renderer_nothink is None:
            from src.training.tinker_common import apply_renderer_date, load_renderer

            self._renderer_nothink = load_renderer(
                self.base_model, f"{self.renderer_name}_disable_thinking", self._tokenizer
            )
            apply_renderer_date(self._renderer_nothink, self.renderer_current_date)
        return self._renderer_nothink

    def _build_prompt(self, messages, thinking: bool):
        if thinking:
            # The renderer name is passed so a harmony-format family is not given a <think>
            # opener it will never close (see force_thinking_prompt).
            return force_thinking_prompt(self._renderer, self._tokenizer, messages, self.renderer_name)
        return self._renderer_for(thinking).build_generation_prompt(messages)

    async def sample_turn(
        self,
        messages: list[dict],
        *,
        tools: list[dict],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> SampledPrompt:
        """One assistant turn of a tool-using conversation (see harmony_tool_prompt). `messages`
        is the conversation after the system prompt: user turns, assistant turns carrying
        `tool_calls`, and `tool` results carrying `name`. The caller parses the reply with
        `parse_turn`. Thinking is the renderer's own (harmony reasoning is always on)."""
        if self._service is None:
            raise RuntimeError("TinkerGenerator.load() must be called before sample_turn().")
        await self._ensure_client()
        prompt = harmony_tool_prompt(self._renderer, messages, tools, system_prompt)
        await self._check_fits(prompt)
        params = self._sampling_params(max_tokens, temperature, top_p, seed, True)
        resp = await self._client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        return self._sampled_prompt(prompt, resp.sequences)

    def parse_turn(self, tokens: list[int]):
        """The renderer's parse of one sampled turn: (cookbook Message, ParseTermination). The
        message carries content parts (thinking / text), `tool_calls`, and `unparsed_tool_calls`."""
        if self._renderer is None:
            raise RuntimeError("TinkerGenerator.load() must be called before parse_turn().")
        return self._renderer.parse_response(list(tokens))

    async def _check_fits(self, prompt) -> None:
        """Raise PromptExceedsContextWindow when the prompt leaves no room for a single token. The
        window is the server's declared context length for the base model, read once: concurrent first
        turns wait on one request instead of each sending their own."""
        if self._context_window is None:
            async with self._context_window_lock:
                if self._context_window is None:
                    self._context_window = await self._read_context_window()
        n = prompt.length
        if n >= self._context_window:
            raise PromptExceedsContextWindow(n, self._context_window)

    async def _read_context_window(self) -> int:
        caps = await self._service.get_server_capabilities_async()
        matches = [m for m in caps.supported_models if m.model_name == self.base_model]
        if not matches:
            raise RuntimeError(f"{self.base_model!r} is not among the Tinker server's supported models")
        if matches[0].max_context_length is None:
            raise RuntimeError(f"the Tinker server declares no context length for {self.base_model!r}")
        return int(matches[0].max_context_length)

    async def _ensure_client(self):
        if self._client is None:
            if self.checkpoint:
                self._client = await self._service.create_sampling_client_async(model_path=self.checkpoint)
            else:
                self._client = await self._service.create_sampling_client_async(base_model=self.base_model)

    def _sampling_params(self, max_tokens, temperature, top_p, seed, thinking):
        import tinker

        return tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop=self._renderer_for(thinking).get_stop_sequences(),
        )

    async def _sample_with_retries(self, prompt, params, num_samples: int, idx: int, error_counts: dict):
        """The sequences for one prompt, or None after the retry budget. Deterministic API
        statuses (auth, billing, bad request) are never retried."""
        import tinker

        delay = TINKER_BACKOFF_BASE_SEC
        for attempt in range(TINKER_MAX_RETRIES):
            try:
                resp = await self._client.sample_async(prompt=prompt, num_samples=num_samples, sampling_params=params)
                return resp.sequences
            except tinker.APIStatusError as e:
                status = getattr(e, "status_code", None)
                if status in TINKER_DETERMINISTIC_STATUS:
                    error_counts["deterministic"] = error_counts.get("deterministic", 0) + 1
                    print(f"WARNING: Tinker deterministic {status} for conversation {idx}, not retrying: {e}")
                    return None
                error_counts["status"] = error_counts.get("status", 0) + 1
                if attempt == TINKER_MAX_RETRIES - 1:
                    return None
                await asyncio.sleep(delay)
                delay = min(delay * 2, TINKER_BACKOFF_CAP_SEC)
            except (tinker.RateLimitError, tinker.APIConnectionError, tinker.RequestFailedError) as e:
                if not tinker_failure_is_transient(e):
                    error_counts["deterministic"] = error_counts.get("deterministic", 0) + 1
                    print(f"WARNING: Tinker request failed as a user error for conversation {idx}, not retrying: {e}")
                    return None
                error_counts["transient"] = error_counts.get("transient", 0) + 1
                if attempt == TINKER_MAX_RETRIES - 1:
                    print(f"WARNING: Tinker transient error for conversation {idx}, giving up: {e}")
                    return None
                await asyncio.sleep(delay)
                delay = min(delay * 2, TINKER_BACKOFF_CAP_SEC)
        return None

    def _check_failures(self, results: list, error_counts: dict) -> None:
        _check_deterministic_failures("TinkerGenerator", error_counts)
        n_failed = sum(1 for r in results if r is None)
        if n_failed > 0:
            error_summary = ", ".join(f"{k}={v}" for k, v in sorted(error_counts.items()))
            print(f"WARNING: {n_failed}/{len(results)} Tinker samples returned None — errors: {error_summary}")
        if n_failed > max(len(results) * FAILURE_ABORT_FRACTION, FAILURE_ABORT_MIN):
            raise RuntimeError(
                f"Too many Tinker sample failures: {n_failed}/{len(results)} "
                f"({n_failed / len(results):.0%}) exceeded {FAILURE_ABORT_FRACTION:.0%} threshold"
            )

    def sample_chat(
        self,
        conversations: list[list[dict]],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        thinking: bool,
        on_sampled: Optional[Callable[[int, "SampledPrompt"], None]] = None,
    ) -> list["SampledPrompt | None"]:
        """`n` completions per conversation with tokens, logprobs and the provider's stop reason,
        which is what a policy-gradient step needs and `generate_chat` throws away. Unseeded:
        a fixed seed with n > 1 returns identical completions. A conversation that fails after
        the retry budget is None; too many failures raise, as in `generate_chat`.

        `on_sampled(idx, sampled)` runs the moment conversation `idx` returns, while the others
        are still in flight: the hook a caller uses to write each completion set to disk before
        anything judges it, so a death later in the process never loses paid sampling."""
        if self._service is None:
            raise RuntimeError("TinkerGenerator.load() must be called before sample_chat().")
        return _run_async(self._sample_async(conversations, n, max_tokens, temperature, top_p, thinking, on_sampled))

    async def _sample_async(self, conversations, n, max_tokens, temperature, top_p, thinking, on_sampled=None):
        await self._ensure_client()
        params = self._sampling_params(max_tokens, temperature, top_p, None, thinking)
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: list = [None] * len(conversations)
        error_counts: dict = {}
        done_count = 0

        async def one(idx, messages):
            nonlocal done_count
            prompt = self._build_prompt(messages, thinking)
            async with semaphore:
                seqs = await self._sample_with_retries(prompt, params, n, idx, error_counts)
            if seqs is not None:
                results[idx] = self._sampled_prompt(prompt, seqs)
                if on_sampled is not None:
                    on_sampled(idx, results[idx])
            done_count += 1
            if done_count % 50 == 0:
                print(f"  Sampled {done_count}/{len(conversations)} prompts", flush=True)

        await asyncio.gather(*[one(i, c) for i, c in enumerate(conversations)])
        self._check_failures(results, error_counts)
        return results

    def _sampled_prompt(self, prompt, seqs) -> SampledPrompt:
        return SampledPrompt(
            prompt_tokens=prompt.to_ints(),
            sequences=[
                SampledSequence(
                    tokens=list(seq.tokens),
                    logprobs=[float(x) for x in seq.logprobs],
                    text=self._tokenizer.decode(seq.tokens, skip_special_tokens=True),
                    stop_reason=seq.stop_reason,
                )
                for seq in seqs
            ],
        )

    async def sample_one(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
        thinking: bool,
        reasoning_prefill: str | None,
    ) -> SampledPrompt:
        """One completion for one conversation, from inside a running event loop. No retry loop:
        the caller owns retries (the Inspect provider retries on the same Tinker error classes
        `_sample_with_retries` treats as transient), so a failure raises the provider's exception.
        `reasoning_prefill`, when set, is the start of the reasoning channel the model continues
        (`reasoning_prefill_prompt`; thinking only); the returned sequence holds only the sampled
        tokens, without the prefill."""
        if self._service is None:
            raise RuntimeError("TinkerGenerator.load() must be called before sample_one().")
        await self._ensure_client()
        if reasoning_prefill is None:
            prompt = self._build_prompt(messages, thinking)
        elif not thinking:
            raise ValueError("a reasoning prefill needs thinking=true")
        else:
            prompt = reasoning_prefill_prompt(self._renderer, messages, self.renderer_name, reasoning_prefill)
        await self._check_fits(prompt)
        params = self._sampling_params(max_tokens, temperature, top_p, seed, thinking)
        resp = await self._client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        return self._sampled_prompt(prompt, resp.sequences)

    async def _generate_async(self, conversations, max_tokens, temperature, top_p, seed, thinking, on_response=None):
        await self._ensure_client()
        params = self._sampling_params(max_tokens, temperature, top_p, seed, thinking)
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = [None] * len(conversations)
        stop_reasons = [None] * len(conversations)
        error_counts = {}
        done_count = 0

        async def generate_one(idx, messages):
            nonlocal done_count
            prompt = self._build_prompt(messages, thinking)
            async with semaphore:
                seqs = await self._sample_with_retries(prompt, params, 1, idx, error_counts)
            if seqs is not None:
                seq = seqs[0]
                results[idx] = self._tokenizer.decode(seq.tokens, skip_special_tokens=True)
                # The provider's own stop reason ("length" or "stop"), never inferred from the
                # token count: a completion that stops exactly at the cap is not truncated.
                stop_reasons[idx] = seq.stop_reason
            if on_response and results[idx] is not None:
                on_response(idx, results[idx])
            done_count += 1
            if done_count % 50 == 0:
                print(f"  Generated {done_count}/{len(conversations)} responses")

        await asyncio.gather(*[generate_one(i, c) for i, c in enumerate(conversations)])

        self.last_stop_reasons = stop_reasons
        self._check_failures(results, error_counts)
        return results

    def cleanup(self):
        self._client = None
        self._service = None
