"""Helpers shared by the eval entry points in this package, and who uses each.

- `positive_int`: every evaluation's command line, and the rollout, pool and judge scripts outside
  this package.
- `add_model_flags`, `validate_model_flags`, `build_generator`: agentic_misalignment, gpqa, ifbench,
  impossible_livecodebench, impossible_swebench and mask (mask builds its generator in
  generate_adapter.py). strongreject, surf_sr, em_categorical, school_of_reward_hacks and
  preference_v3 declare the same model flags in their own parsers.
- `add_retry_log_flag`, `retrying`: every evaluation except strongreject and mask.
- `reject_greedy_thinking`: agentic_misalignment, gpqa, ifbench and mask.
- `seeded_subset`: impossible_livecodebench and impossible_swebench.
- `strip_thinking`, `generate_with_cache`, `valid_answers`, `csv_list`, `add_common_eval_args`: no
  evaluation in this package calls them; the first two are exercised by the tests only.

Judge-side shared pieces (strict tool transport, categorical tools, cached judging) live in
judge_runner and inspect_judge.
"""

from __future__ import annotations

import argparse

from .base import (
    fix_thinking_tags,
)


def strip_thinking(text: str) -> str:
    """The answer part of a response: everything after the last </think>, or the whole text.

    Delegates to src.thinking, which owns the split contract for the whole repo.
    """
    from src.thinking import answer_of

    return answer_of(text)


def reject_greedy_thinking(thinking: bool, temperature: float) -> None:
    """Refuse temperature=0 with thinking: greedy decoding collapses thinking models into
    repetition loops, and the Anthropic backend would silently override the temperature."""
    if thinking and temperature == 0.0:
        raise ValueError(
            "temperature=0 with thinking: greedy decoding collapses thinking models into "
            "repetition loops (and the Anthropic backend would silently override it). "
            "Use the standard eval temperature."
        )


def generate_with_cache(
    generator,
    conversations: list[list[dict]],
    cache_keys: list[str],
    cache,
    *,
    max_tokens: int,
    temperature: float,
    thinking: bool,
    top_p: float = 1.0,
    prefill_tag: str | None = None,
) -> tuple[list[str | None], list[str | None]]:
    """Generate every conversation through the content-keyed cache.

    Returns (responses, stop_reasons) aligned to the inputs, with thinking tags
    repaired. Generation runs unseeded (diverse samples; sample identity lives in
    the cache key).

    Truncation contract: Anthropic/vLLM/OpenRouter generators raise on truncation;
    the Tinker and batch-Anthropic generators record it in last_stop_reasons and
    defer the call to the eval. The stop_reason is persisted as cache metadata so
    the truncation fact survives cache resumes - and a cache entry WITHOUT that
    metadata (e.g. written by the incremental callback of a run that died
    mid-generation) is treated as a miss rather than trusted as complete.
    """
    responses: dict[int, str | None] = {}
    stop_reasons: dict[int, str | None] = {}
    for i, key in enumerate(cache_keys):
        if key in cache:
            meta = cache.get_meta(key)
            if "stop_reason" not in meta:
                continue  # incomplete entry from an interrupted run: regenerate
            cached = cache.get(key)
            # An empty/whitespace response is a degenerate generation (immediate stop, or
            # a transient sampling drop), never a valid answer. Don't trust a cached empty
            # -- treat it as a miss so the rerun regenerates it rather than serving a
            # poisoned blank that would silently skew a run's denominator.
            if not (cached or "").strip():
                continue
            responses[i] = cached
            stop_reasons[i] = meta["stop_reason"]
    uncached = [i for i in range(len(conversations)) if i not in responses]
    print(f"  Cached: {len(responses)}, to generate: {len(uncached)}")
    if uncached:

        def cache_callback(idx, response):
            # No stop_reason is known per-row at callback time; the entry becomes a
            # trusted cache hit only once the meta write below lands. Skip empties.
            if response and response.strip():
                cache.set(cache_keys[uncached[idx]], response)

        gen_kwargs = {"prefill_tag": prefill_tag} if prefill_tag is not None else {}
        new_responses = generator.generate_chat(
            [conversations[i] for i in uncached],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=None,
            thinking=thinking,
            on_response=cache_callback,
            **gen_kwargs,
        )
        new_stop_reasons = generator.last_stop_reasons
        if len(new_stop_reasons) != len(uncached):
            raise RuntimeError(
                f"generator reported {len(new_stop_reasons)} stop reasons for {len(uncached)} conversations"
            )
        for idx, response, stop_reason in zip(uncached, new_responses, new_stop_reasons):
            responses[idx] = response
            stop_reasons[idx] = stop_reason
            # Only persist genuine (non-empty) generations; empties stay uncached so a
            # later run retries them.
            if response and response.strip():
                cache.set(cache_keys[idx], response, meta={"stop_reason": stop_reason})
    out = [responses[i] for i in range(len(conversations))]
    out = [fix_thinking_tags(r) if r is not None else None for r in out]
    return out, [stop_reasons[i] for i in range(len(conversations))]


def valid_answers(
    responses: list[str | None],
    stop_reasons: list[str | None],
    *,
    thinking: bool,
    renderer_name: str | None = None,
) -> tuple[list[str | None], list[bool]]:
    """(answers, valid) for a batch, with validity decided by src.thinking.usable.

    An answer is the post-thinking text. A row is valid iff the generation completed (not None,
    not truncated), the answer is non-empty, and -- in thinking mode on a renderer that has
    think blocks -- the block actually closed. An unclosed block is the model breaking, not an
    answer: `answer_of` returns the whole raw chain-of-thought in that case, which would other-
    wise be handed to a judge as if it were the reply. Invalid rows are saved, never scored.

    renderer_name defaults to None, which assumes think blocks exist and therefore CHECKS
    closure -- the conservative side. Pass it to exempt harmony-style renderers.
    """
    from src.thinking import answer_of, usable

    answers = [answer_of(r) if r is not None else None for r in responses]
    # usable() is the response-level contract (present, not truncated, block closed). A closed
    # block with nothing after it is a separate failure -- all reasoning, no reply -- and there
    # is nothing to score, so the non-empty-answer requirement stays here.
    valid = [
        usable(r, stop_reason=sr, thinking=thinking, renderer_name=renderer_name) and bool(a)
        for r, sr, a in zip(responses, stop_reasons, answers)
    ]
    return answers, valid


def positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return n


def csv_list(value: str) -> list[str]:
    """argparse type for a comma-separated flag: "a,b, c" -> ["a", "b", "c"]."""
    return [v.strip() for v in value.split(",") if v.strip()]


def add_common_eval_args(p: argparse.ArgumentParser, judge: bool = True) -> None:
    """The CLI surface every eval in this package shares. Science knobs are required;
    evals add their own flags after calling this. judge=False drops the judge flags for
    evals with no LLM judge (e.g. deterministic MCQ scoring)."""
    p.add_argument("--model", required=True, help="model to evaluate (see create_generator for supported paths)")
    p.add_argument("--output", required=True, help="output directory for summary/results/caches")
    think = p.add_mutually_exclusive_group(required=True)
    think.add_argument("--thinking", action="store_true", help="native thinking mode")
    think.add_argument("--nothinking", action="store_true", help="thinking off")
    p.add_argument("--temperature", type=float, required=True, help="generation temperature")
    p.add_argument("--seed", type=int, required=True, help="cache/shuffle seed (generation runs unseeded)")
    p.add_argument("--max-tokens", type=positive_int, required=True, help="generation max tokens")
    p.add_argument("--n-samples", type=positive_int, required=True, help="samples per prompt")
    if judge:
        p.add_argument("--judge-model", required=True, help="Anthropic judge model (strict tool use)")
        p.add_argument("--use-batch", action="store_true", help="judge via the Anthropic Batch API (50%% off)")
        p.add_argument("--rejudge", action="store_true", help="ignore the judge cache for this run")
    p.add_argument("--limit", type=positive_int, help="evaluate a seeded random subset (small trial runs)")
    p.add_argument("--dry-run", action="store_true", help="print counts + cost estimate, no API calls (free)")


# --- seeded subsets -----------------------------------------------------------------------------


def seeded_subset(ids: list[str], subset: int | None, seed: int | None) -> list[str]:
    """The first `subset` ids of a seeded shuffle of the sorted ids, or every id when subset is
    None. The same (subset, seed) names the same ids on any machine; `seed` is required exactly
    when `subset` is set, so a partial run is never unseeded and a full run never carries a
    meaningless seed."""
    import random

    if (subset is None) != (seed is None):
        raise ValueError("--subset and --seed go together: both set (a seeded subset) or both none (every id)")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids")
    ordered = sorted(ids)
    if subset is None:
        return ordered
    if subset > len(ordered):
        raise ValueError(f"--subset {subset} exceeds the {len(ordered)} ids available")
    random.Random(seed).shuffle(ordered)
    return ordered[:subset]


# --- the model flags an Inspect agentic benchmark CLI takes ----------------------------------------
# Two model families. `tinker/<base>`: the repo's provider renders the prompt itself, so the
# tokenizer, renderer and thinking mode are declared, and a trained LoRA is named by `--checkpoint`.
# `openrouter/<vendor>/<model>`: Inspect's provider, which reads OPENROUTER_API_KEY; the Tinker
# flags take none. Anything else is an error, never a guess.

TINKER_PROVIDER = "tinker"
OPENROUTER_PROVIDER = "openrouter"
MODEL_FLAGS = ("model", "checkpoint", "tokenizer_id", "renderer_name", "renderer_current_date", "thinking")


def add_model_flags(p: argparse.ArgumentParser) -> None:
    from src.training.tinker_common import _bool, _optional_str

    p.add_argument(
        "--model",
        required=True,
        help=f"Inspect model: {TINKER_PROVIDER}/<base> or {OPENROUTER_PROVIDER}/<vendor>/<model>",
    )
    p.add_argument("--checkpoint", required=True, type=_optional_str, help="tinker:// sampler path, or none")
    p.add_argument("--tokenizer-id", required=True, type=_optional_str, help="HF tokenizer id, or none (API models)")
    p.add_argument("--renderer-name", required=True, type=_optional_str, help="cookbook renderer, or none")
    p.add_argument(
        "--renderer-current-date", required=True, type=_optional_str, help="YYYY-MM-DD, or none for the fixed default"
    )
    p.add_argument("--thinking", required=True, type=_bool, help="true or false; declared, never inferred")


def validate_model_flags(a: argparse.Namespace) -> None:
    """Refuse a flag set that does not fit the model's provider."""
    provider = a.model.partition("/")[0]
    if provider == TINKER_PROVIDER:
        missing = [f for f in ("tokenizer_id", "renderer_name") if not getattr(a, f)]
        if missing:
            raise SystemExit(f"{TINKER_PROVIDER} models need {missing}")
        if not a.thinking:
            raise SystemExit(f"the {TINKER_PROVIDER} tool path renders harmony reasoning; --thinking must be true")
    elif provider == OPENROUTER_PROVIDER:
        set_ = [f for f in ("checkpoint", "tokenizer_id", "renderer_name", "renderer_current_date") if getattr(a, f)]
        if set_:
            raise SystemExit(f"{OPENROUTER_PROVIDER} models take none for {set_}")
    else:
        raise SystemExit(
            f"unknown model provider {provider!r}; expected {TINKER_PROVIDER}/... or {OPENROUTER_PROVIDER}/..."
        )


def build_generator(a: argparse.Namespace, max_connections: int):
    """The Inspect model for the validated flags, with its connection limit pinned on the model
    (an eval-level limit is merged onto every model, judges included)."""
    from inspect_ai.model import GenerateConfig, get_model

    provider = a.model.partition("/")[0]
    if provider == TINKER_PROVIDER:
        model_args = {
            "tokenizer_id": a.tokenizer_id,
            "renderer_name": a.renderer_name,
            "thinking": a.thinking,
            "max_connections": max_connections,
        }
        if a.checkpoint:
            model_args["checkpoint"] = a.checkpoint
        if a.renderer_current_date:
            model_args["renderer_current_date"] = a.renderer_current_date
        return get_model(a.model, config=GenerateConfig(max_connections=max_connections), **model_args)
    return get_model(a.model, config=GenerateConfig(max_connections=max_connections))


RETRY_LOG_FLAG = "--retry-log"


def _optional_path(value: str) -> str | None:
    if value == "none":
        return None
    if not value:
        raise argparse.ArgumentTypeError("a path or none")
    return value


def add_retry_log_flag(p: argparse.ArgumentParser) -> None:
    """The flag that continues a previous run of the same task: `none` for a fresh run,
    else the path of the previous Inspect log. Scored samples in that log are carried into the new
    log unchanged; errored and unfinished ones run again through the identical task."""
    p.add_argument(
        RETRY_LOG_FLAG,
        required=True,
        type=_optional_path,
        help="none, or a previous Inspect log of this run whose errored/unfinished samples to re-run",
    )


def _held_open(path: str) -> bool:
    """Whether any process has `path` open, by `lsof`. A log left at status 'started' with no writer
    is a run killed mid-way: its flushed samples are complete records a retry can carry forward."""
    import shutil
    import subprocess

    if shutil.which("lsof") is None:
        raise SystemExit("lsof not found: cannot tell whether a 'started' log is still being written")
    return subprocess.run(["lsof", "--", path], capture_output=True).returncode == 0


def _as_logged(metadata: dict | None) -> dict:
    """Task metadata as an Inspect log stores it (JSON types; no metadata is an empty dict), so a task's
    metadata compares equal to the same task's metadata read back from its log."""
    import json

    return json.loads(json.dumps(metadata or {}, default=str))


def retrying(task, retry_log: str | None):
    """What to hand `inspect_ai.eval`: the task itself for a fresh run, or the task wrapped with the
    previous log so Inspect re-emits its scored samples and re-runs the rest. The wrapped task is
    the same object a fresh run builds, so the retry cannot drift from the original run. Refuses a log
    of a different task, a log whose task metadata differs from this task's (such as an input file's
    digest recorded there), or a log that is still being written; a 'started' log that no process holds
    open (a killed run) is accepted."""
    if retry_log is None:
        return task
    from inspect_ai._eval.task.task import PreviousTask
    from inspect_ai.log import read_eval_log

    previous = read_eval_log(retry_log)
    if previous.status == "started" and _held_open(retry_log):
        raise SystemExit(f"{retry_log}: status is 'started' and a process still has it open; that run has not finished")
    if previous.eval.task != task.name:
        raise SystemExit(f"{retry_log}: is a log of task {previous.eval.task!r}, not {task.name!r}")
    before, now = _as_logged(previous.eval.metadata), _as_logged(task.metadata)
    if before != now:
        changed = sorted(k for k in before.keys() | now.keys() if before.get(k) != now.get(k))
        raise SystemExit(f"{retry_log}: the task's metadata differs from the previous run's in {changed}")
    return PreviousTask(
        id=previous.eval.task_id,
        task=task,
        task_args={},
        model=None,
        model_roles=None,
        log=previous,
        log_info=None,
    )
