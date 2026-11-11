"""API keys, read from standard environment variables.

    ANTHROPIC_API_KEY     Claude models (generation, judges, monitors, batch)
    OPENAI_API_KEY        gpt-* models (generation; also read by the MASK benchmark's own judge code)
    OPENROUTER_API_KEY    router-style ids containing "/", for the generation role only (the rewriter
                          model; the DeepSeek reasoning judge, which asks for its key under the
                          generation role). `get_api_key("judge", "deepseek/...")` is an error: the
                          judge and batch roles take Claude ids.
    TINKER_API_KEY        Tinker fine-tuning and sampling (read by the Tinker SDK itself)
    TINKER_OWNER          a label attached to Tinker calls for cost attribution, any string; required by
                          anything that samples or trains a Tinker model (read in
                          src.training.tinker_common)

A `.env` file at the repository root, if present, is loaded on import of this module. A variable
already set in the shell is kept; the file only fills in what is missing. Values are never printed.

Provider is derived from the model name: "claude-*" is Anthropic, "gpt-*" is OpenAI, and a
recognized router-style id ("deepseek/...", "qwen/...") is OpenRouter. Anything else is an error.
"""

import os

from src import paths

_PROJECT_ENV_FILE = paths.ENV_FILE
_env_loaded = False


def load_project_env() -> None:
    """Load the repository's `.env` into os.environ without overriding variables already set.
    Idempotent. A line that is neither blank, a comment, nor KEY=value is an error."""
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    if not _PROJECT_ENV_FILE.exists():
        return
    for lineno, raw in enumerate(_PROJECT_ENV_FILE.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(
                f"{_PROJECT_ENV_FILE} line {lineno} is neither blank, a comment, nor KEY=value. "
                f"(Content not shown in case it is a secret.)"
            )
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


load_project_env()

OPENROUTER_MODEL_PREFIXES = (
    "openai/",
    "anthropic/",
    "cognitivecomputations/",
    "venice/",
    "qwen/",
    "nvidia/",
    "deepseek/",
    "google/",
)
ROLE_GENERATION = "generation"
ROLE_JUDGE = "judge"
ROLE_BATCH = "batch"
ROLES = (ROLE_GENERATION, ROLE_JUDGE, ROLE_BATCH)


def is_openrouter_model(model: str) -> bool:
    return model.startswith(OPENROUTER_MODEL_PREFIXES)


def _require_env(var: str, context: str) -> str:
    """Read an environment variable or raise, distinguishing missing from set-but-empty."""
    val = os.environ.get(var)
    if val:
        return val
    if val is None:
        raise RuntimeError(f"{var} is not set. Required for {context}.")
    raise RuntimeError(f"{var} is set but empty. Required for {context}.")


def require_standard_claude(model: str, role: str) -> None:
    """Judges and batch jobs take a claude-* id, never another provider's id."""
    if not model or not model.startswith("claude-"):
        raise ValueError(f"{role} model must be a claude-* id, got {model!r}")


def get_api_key(role: str, model: str = None) -> str:
    """The API key for a role ("generation", "judge" or "batch") and model.

    Judges and batch jobs default to Claude when no model is given. A router-style id is accepted
    for generation only; an unrecognized model id is an error, never a guessed default.
    """
    if role not in ROLES:
        raise ValueError(f"Unknown role: {role!r}. Expected one of {ROLES}.")
    if model == "":
        raise ValueError("model must be a non-empty string when provided")
    if model is None:
        if role == ROLE_GENERATION:
            raise ValueError("get_api_key('generation') requires a non-empty model")
        return _require_env("ANTHROPIC_API_KEY", f"Claude {role} calls")
    if "/" in model:
        if role == ROLE_GENERATION and is_openrouter_model(model):
            return _require_env("OPENROUTER_API_KEY", f"OpenRouter model '{model}'")
        raise ValueError(
            f"Router-style model id {model!r} for role {role!r}: OpenRouter is only wired "
            f"for generation with prefixes {OPENROUTER_MODEL_PREFIXES}."
        )
    if model.startswith("claude-"):
        return _require_env("ANTHROPIC_API_KEY", f"Claude model '{model}'")
    if model.startswith("gpt-") and role in (ROLE_GENERATION, ROLE_BATCH):
        return _require_env("OPENAI_API_KEY", f"OpenAI model '{model}'")
    raise ValueError(f"Unrecognized model {model!r} for role {role!r}: no provider key mapping.")


def anthropic_client(**kwargs):
    """A synchronous Anthropic client on ANTHROPIC_API_KEY."""
    import anthropic

    kwargs.setdefault("api_key", _require_env("ANTHROPIC_API_KEY", "Claude API calls"))
    return anthropic.Anthropic(**kwargs)


def async_anthropic_client(**kwargs):
    """An asynchronous Anthropic client on ANTHROPIC_API_KEY."""
    import anthropic

    kwargs.setdefault("api_key", _require_env("ANTHROPIC_API_KEY", "Claude API calls"))
    return anthropic.AsyncAnthropic(**kwargs)


def inspect_claude_model(model: str, **get_model_kwargs):
    """An Inspect judge model for a Claude id, on Inspect's own Anthropic provider (ANTHROPIC_API_KEY).

    The instance sends `strict: true` on tools marked strict (`src.evals.inspect_judge`), which
    Inspect's provider otherwise drops, so a forced judge call through it is grammar-constrained.
    Every evaluation builds its Claude judge here and every Inspect judge call sends temperature 0,
    so a model that takes no temperature (`claude-opus-4-7`, `claude-opus-4-8`) is refused here,
    before any call."""
    from inspect_ai.model import get_model
    from inspect_ai.model._providers.anthropic import AnthropicAPI

    from src.evals.inspect_judge import require_pinned_judge_temperature, send_strict_tools

    require_standard_claude(model, "Inspect Claude")
    require_pinned_judge_temperature(model)
    m = get_model(f"anthropic/{model}", **get_model_kwargs)
    api = m.api
    if type(api) is not AnthropicAPI or api.is_bedrock() or api.is_vertex() or api.is_azure():
        raise RuntimeError(f"expected Inspect's direct Anthropic provider for {model!r}, got {type(api).__name__}")
    send_strict_tools(api)
    return m
