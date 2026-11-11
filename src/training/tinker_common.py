"""Shared machinery for the Tinker trainers (SFT, SDF, RL).

Everything here is trainer-independent: argument parsing helpers, the canonical
mode vocabularies, the learning-rate schedule, checkpoint/resume validation, the
Tinker client lifecycle, Weights & Biases plumbing, and strict JSON helpers. The
per-trainer modules import from here so the SDK glue lives in exactly one place
and the trainer files can read as the training algorithm alone.

Functions that take a `config` are duck-typed: they read only fields shared by
every trainer config (checkpoint mode/path, model, lora_rank, wandb_*, seed,
batch_size, start_step), so the same implementation serves all three trainers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

from src.costs import (  # noqa: F401  (re-exported: trainers and experiments import the tables from here)
    TINKER_PREFILL_USD_PER_MTOK,
    TINKER_SAMPLE_USD_PER_MTOK,
    TINKER_TRAIN_USD_PER_MTOK,
)
from src.training.arrays import TrainingArrays, values_to_list
from src.training.artifacts import CHECKPOINTS_LOG, CONFIG_JSON, RESUME_META, WANDB_JSON

TINKER_API_KEY_VAR = "TINKER_API_KEY"

# How an existing checkpoint (if any) seeds a run.
CHECKPOINT_MODE_FRESH = "fresh"  # new LoRA, no checkpoint
CHECKPOINT_MODE_INIT = "init"  # load weights, fresh optimizer
CHECKPOINT_MODE_RESUME = "resume"  # load weights + optimizer, continue the SAME interrupted job
#: Load weights + optimizer and train on NEW data as a new job: the round loop's mode for every
#: round after its first, so Adam's moments persist across rounds. Unlike `resume` it validates
#: nothing about the previous job, because the data differs by design.
CHECKPOINT_MODE_CONTINUE = "continue"
CHECKPOINT_MODES = (CHECKPOINT_MODE_FRESH, CHECKPOINT_MODE_INIT, CHECKPOINT_MODE_RESUME, CHECKPOINT_MODE_CONTINUE)
NO_CHECKPOINT_PATH = "none"

# Learning-rate schedules (all with linear warmup; see get_lr).
LR_SCHEDULE_COSINE = "cosine"
LR_SCHEDULE_LINEAR = "linear"
LR_SCHEDULE_CONSTANT = "constant"
LR_SCHEDULES = (LR_SCHEDULE_COSINE, LR_SCHEDULE_LINEAR, LR_SCHEDULE_CONSTANT)

# Weights & Biases run modes and resume policies.
WANDB_MODE_DISABLED = "disabled"
WANDB_MODE_OFFLINE = "offline"
WANDB_MODE_ONLINE = "online"
WANDB_MODES = (WANDB_MODE_DISABLED, WANDB_MODE_OFFLINE, WANDB_MODE_ONLINE)
WANDB_RESUME_NONE = "none"
WANDB_RESUME_ALLOW = "allow"
WANDB_RESUME_MUST = "must"
WANDB_RESUME_NEVER = "never"
WANDB_RESUME_AUTO = "auto"
WANDB_RESUME_MODES = (
    WANDB_RESUME_NONE,
    WANDB_RESUME_ALLOW,
    WANDB_RESUME_MUST,
    WANDB_RESUME_NEVER,
    WANDB_RESUME_AUTO,
)
WANDB_RESUME_REQUIRES_RUN_ID = (WANDB_RESUME_ALLOW, WANDB_RESUME_MUST)


# --- argparse type helpers (every science knob is a required, strictly-typed flag) ---


def _bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError(f"expected 'true' or 'false', got {value!r}")


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {value!r}")
    return n


def _nonneg_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"expected an integer >= 0, got {value!r}")
    return n


def _positive_float(value: str) -> float:
    x = float(value)
    if x <= 0:
        raise argparse.ArgumentTypeError(f"expected a float > 0, got {value!r}")
    return x


def _nonneg_float(value: str) -> float:
    x = float(value)
    if x < 0:
        raise argparse.ArgumentTypeError(f"expected a float >= 0, got {value!r}")
    return x


def _optional_str(value: str) -> str:
    return "" if value == "none" else value


def _csv_tags(value: str) -> tuple[str, ...]:
    if value == "none":
        return ()
    tags = tuple(tag.strip() for tag in value.split(",") if tag.strip())
    if not tags:
        raise argparse.ArgumentTypeError("expected comma-separated tags or 'none'")
    return tags


# --- learning rate, environment, and provenance ---


def get_lr(base_lr: float, step: int, warmup_steps: int, max_steps: int, schedule: str) -> float:
    """Linear warmup, then linear/cosine decay or constant LR."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if schedule == LR_SCHEDULE_CONSTANT:
        return base_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    if schedule == LR_SCHEDULE_LINEAR:
        return base_lr * max(0.0, 1.0 - min(progress, 1.0))
    if schedule == LR_SCHEDULE_COSINE:
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    raise ValueError(f"unknown LR schedule {schedule!r}; expected one of {LR_SCHEDULES}")


def owner_tag() -> dict[str, str]:
    """Return the owner metadata tag used to attribute Tinker API costs."""
    owner = os.environ.get("TINKER_OWNER")
    if not owner:
        raise RuntimeError("TINKER_OWNER is not set. Set it explicitly before a real Tinker run.")
    return {"owner": owner}


def require_tinker_key() -> None:
    if not os.environ.get(TINKER_API_KEY_VAR):
        raise RuntimeError(
            f"{TINKER_API_KEY_VAR} is not set. Add it to the project .env or shell before a real Tinker run."
        )


def get_git_commit() -> str:
    from src.evals.base import get_git_hash

    return get_git_hash()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arrays_sha256(arrays: list[TrainingArrays]) -> str:
    digest = hashlib.sha256()
    for row in arrays:
        payload = {
            "input_tokens": row.input_tokens,
            "target_tokens": row.target_tokens,
            "weights": row.weights,
            "rendered_text": row.rendered_text,
        }
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


# --- cost estimation ---


def estimate_train_tokens(token_lengths: list[int], batch_size: int, max_steps: int) -> int:
    """Approximate total tokens a run will forward+backward over.

    Each step processes batch_size rows; over max_steps steps with cyclic sampling that is about
    max_steps * batch_size * (mean tokens per row). An estimate, not an exact bill.
    """
    if not token_lengths:
        return 0
    mean_tokens = sum(token_lengths) / len(token_lengths)
    return int(round(mean_tokens * batch_size * max_steps))


def format_cost_estimate(model: str, train_tokens: int) -> str:
    """One-line cost estimate; dollar figure when the model's rate is on file, else tokens only."""
    rate = TINKER_TRAIN_USD_PER_MTOK.get(model)
    if rate is None:
        return (
            f"Cost estimate: ~{train_tokens:,} train tokens on {model!r}; no train $/Mtok rate on "
            f"file (add it to TINKER_TRAIN_USD_PER_MTOK for a dollar figure)."
        )
    return (
        f"Cost estimate: ~{train_tokens:,} train tokens x ${rate:g}/Mtok = "
        f"~${rate * train_tokens / 1_000_000:,.2f} (train only; excludes prefill/sample)."
    )


# --- tensor / loss-weight helpers ---


def tensor_values(value) -> list:
    """Extract Python values from Tinker TensorData, torch/numpy tensors, or lists."""
    return values_to_list(value)


def trained_weight(datum) -> float:
    """Total loss weight for one datum."""
    return sum(float(weight) for weight in tensor_values(datum.loss_fn_inputs["weights"]))


# --- strict JSON helpers (never silently persist NaN/Inf) ---


def json_dumps_strict(record: dict[str, Any]) -> str:
    return json.dumps(record, allow_nan=False)


def json_dumps_strict_pretty(record: dict[str, Any]) -> str:
    return json.dumps(record, indent=2, allow_nan=False)


def validate_json_metric(value: Any, *, context: str) -> None:
    if not isinstance(value, bool) and isinstance(value, int | float):
        if not math.isfinite(float(value)):
            raise ValueError(f"{context} must be finite, got {value!r}")
        return
    raise ValueError(f"{context} must be a finite numeric metric, got {type(value).__name__}")


# --- cookbook tokenizer / renderer (loaded lazily, only when preparing data) ---


def load_tokenizer(model: str):
    """Load the cookbook tokenizer only when preparing a dataset."""
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return get_tokenizer(model)


# The date a harmony renderer writes into its system prompt when none is given: fixed, never the
# wall clock, so prompt bytes are identical across days. In the renderer's own format (it writes
# `Current date: YYYY-MM-DD`). A checkpoint trained with a different date needs that date passed
# explicitly to reproduce its prompts byte for byte.
DEFAULT_RENDERER_DATE = "2026-08-20"


def apply_renderer_date(renderer, current_date: str | None) -> None:
    """Pin the date a renderer writes into its system prompt (harmony renderers do). Unset means
    DEFAULT_RENDERER_DATE, never the wall clock. A date given for a renderer that has no dated
    system prompt is an error, so it is never silently ignored."""
    if getattr(renderer, "use_system_prompt", False):
        renderer.current_date = current_date or DEFAULT_RENDERER_DATE
    elif current_date:
        raise ValueError("renderer_current_date was provided but the selected renderer has no dated system prompt")


def load_renderer(model: str, renderer_name: str, tokenizer):
    """Load the cookbook renderer that owns chat formatting for this run."""
    from tinker_cookbook import renderers

    return renderers.get_renderer(renderer_name, tokenizer, model_name=model)


# --- Tinker client lifecycle ---


def _field(obj, name: str):
    value = getattr(obj, name, None)
    if value is None:
        raise ValueError(f"Tinker get_info() response has no {name!r} field")
    return value


def _required_field(obj, name: str):
    value = getattr(obj, name, None)
    if value is None:
        raise ValueError(f"Tinker metadata has no non-null {name!r} field")
    return value


async def verify_training_client(config, training_client) -> None:
    """Assert the live training client matches the declared model and LoRA rank."""
    info = await training_client.get_info_async()
    model_data = _field(info, "model_data")
    model_data_name = _field(model_data, "model_name")
    top_level_name = getattr(info, "model_name", None)
    if top_level_name is not None and top_level_name != model_data_name:
        raise ValueError(
            f"Tinker client model metadata mismatch: model_data.model_name={model_data_name!r}, "
            f"model_name={top_level_name!r}"
        )
    model_name = top_level_name or model_data_name
    is_lora = _field(info, "is_lora")
    lora_rank = _field(info, "lora_rank")
    if model_name != config.model:
        raise ValueError(f"Tinker client model mismatch: requested {config.model!r}, got {model_name!r}")
    if is_lora is not True:
        raise ValueError(f"Tinker client is_lora mismatch: expected True, got {is_lora!r}")
    if int(lora_rank) != config.lora_rank:
        raise ValueError(f"Tinker client LoRA rank mismatch: requested {config.lora_rank}, got {lora_rank}")


async def create_training_client(config):
    """Create a Tinker training client for fresh, init-from-checkpoint, or resumed training."""
    import tinker

    service_client = tinker.ServiceClient(user_metadata=owner_tag())
    if config.checkpoint_mode == CHECKPOINT_MODE_INIT:
        print(f"Initializing from checkpoint weights: {config.checkpoint_path} (fresh optimizer)")
        training_client = await service_client.create_training_client_from_state_async(
            path=config.checkpoint_path,
            user_metadata=owner_tag(),
        )
        await verify_training_client(config, training_client)
        return training_client
    if config.checkpoint_mode in (CHECKPOINT_MODE_RESUME, CHECKPOINT_MODE_CONTINUE):
        what = "Resuming" if config.checkpoint_mode == CHECKPOINT_MODE_RESUME else "Continuing optimizer"
        print(f"{what} from checkpoint state: {config.checkpoint_path} (weights + optimizer)")
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            path=config.checkpoint_path,
            user_metadata=owner_tag(),
        )
        await verify_training_client(config, training_client)
        return training_client
    if config.checkpoint_mode == CHECKPOINT_MODE_FRESH:
        print(f"Creating LoRA training client: base_model={config.model}, rank={config.lora_rank}")
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model,
            rank=config.lora_rank,
            train_mlp=config.train_mlp,
            train_attn=config.train_attn,
            train_unembed=config.train_unembed,
            seed=config.seed,
            user_metadata=owner_tag(),
        )
        await verify_training_client(config, training_client)
        return training_client
    raise ValueError(f"unknown checkpoint_mode {config.checkpoint_mode!r}; expected one of {CHECKPOINT_MODES}")


PROBE_TEXT_FOR_TOKENIZER_MATCH = "The quick brown fox jumps over 1234 <think></think>"


async def verify_reference_client(training_client, reference_client) -> None:
    """Assert the DPO/KTO reference serves the same tokenizer as the policy.

    The preference loss compares policy vs reference logprobs token-for-token, so a reference
    pointing at a different model silently corrupts the objective. The Tinker *sampling* client
    exposes no model-identity field to check directly, so we compare tokenizers: they must encode
    a probe string identically for the logprobs to align. Catches a wrong/stale reference model.
    """
    policy_tokenizer = training_client.get_tokenizer()
    reference_tokenizer = reference_client.get_tokenizer()
    if policy_tokenizer.encode(PROBE_TEXT_FOR_TOKENIZER_MATCH) != reference_tokenizer.encode(
        PROBE_TEXT_FOR_TOKENIZER_MATCH
    ):
        raise ValueError(
            "reference client tokenizer does not match the policy tokenizer; reference_model_path "
            "likely points at a different model than the policy, which would invalidate the "
            "preference loss"
        )


async def _tinker_path(awaitable) -> str:
    """Resolve a Tinker save future and require a concrete checkpoint path."""
    result = await awaitable
    if hasattr(result, "result_async"):
        result = await result.result_async()
    path = getattr(result, "path", None)
    if not isinstance(path, str) or not path:
        raise ValueError(f"Tinker save response did not include a non-empty path: {type(result).__name__}")
    return path


def fast_forward_data(train_data: list, config, rng: random.Random) -> int:
    """Advance cyclic data position to match a resumed optimizer step."""
    data_idx = 0
    items_to_skip = config.start_step * config.batch_size
    for _ in range(items_to_skip):
        data_idx += 1
        if data_idx >= len(train_data):
            rng.shuffle(train_data)
            data_idx = 0
    if items_to_skip:
        print(f"Fast-forwarded {items_to_skip} training items for resume; next training row index={data_idx}")
    return data_idx


# --- checkpoint / resume validation ---


def write_config(record: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / CONFIG_JSON).write_text(json.dumps(record, indent=2, allow_nan=False, default=str))


def validate_resume_checkpoint(config, output_dir: Path) -> None:
    """Confirm the resume checkpoint_path is recorded at the declared start_step."""
    if config.checkpoint_mode != CHECKPOINT_MODE_RESUME:
        return

    matches = []
    resume_meta_path = output_dir / RESUME_META
    if resume_meta_path.exists():
        meta = json.loads(resume_meta_path.read_text())
        if meta.get("state_path") == config.checkpoint_path:
            matches.append({"source": str(resume_meta_path), "step": meta.get("step"), "type": "state"})

    checkpoints_path = output_dir / CHECKPOINTS_LOG
    if checkpoints_path.exists():
        for line_no, line in enumerate(checkpoints_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("path") == config.checkpoint_path and record.get("type") == "state":
                matches.append(
                    {
                        "source": f"{checkpoints_path}:{line_no}",
                        "step": record.get("step"),
                        "type": record.get("type"),
                    }
                )

    if not matches:
        raise ValueError(
            f"checkpoint_mode='resume' could not find checkpoint_path={config.checkpoint_path!r} "
            f"in {resume_meta_path} or {checkpoints_path}"
        )
    mismatched = [match for match in matches if match["step"] != config.start_step]
    if len(mismatched) == len(matches):
        details = "; ".join(f"{m['source']} step={m['step']!r} type={m['type']!r}" for m in matches)
        raise ValueError(
            f"checkpoint_path={config.checkpoint_path!r} does not match start_step={config.start_step}; {details}"
        )


def validate_resume_config(config, output_dir: Path, current_record: dict, match_fields: tuple[str, ...]) -> None:
    """Validate that a resume run continues the same job: same recorded config fields, same checkpoint."""
    if config.checkpoint_mode != CHECKPOINT_MODE_RESUME:
        return

    config_path = output_dir / CONFIG_JSON
    if not config_path.exists():
        raise ValueError(f"checkpoint_mode='resume' requires an existing {config_path} to validate data order")

    previous = json.loads(config_path.read_text())
    missing = [field for field in match_fields if field not in previous]
    if missing:
        raise ValueError(
            f"{config_path} is missing resume validation field(s): {', '.join(missing)}. "
            "Use checkpoint_mode='init' for external checkpoints without local run metadata."
        )

    mismatches = [
        f"{field}: previous={previous[field]!r}, current={current_record[field]!r}"
        for field in match_fields
        if previous[field] != current_record[field]
    ]
    if mismatches:
        raise ValueError("checkpoint_mode='resume' config mismatch: " + "; ".join(mismatches))
    validate_resume_checkpoint(config, output_dir)


# --- Weights & Biases lifecycle (strict: a run with no id/project, or the wrong project, fails loudly) ---


def _required_wandb_str(run, name: str) -> str:
    value = getattr(run, name, None)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"W&B run is missing non-empty {name!r} metadata")
    return value


def init_wandb(config, record: dict, output_dir: Path, *, job_type: str, method_tags: tuple[str, ...] = ()):
    """Start a Weights & Biases run when explicitly enabled.

    `job_type` and `method_tags` identify the trainer (e.g. "tinker_sft"; the RL
    method). Requires the run to report a non-empty id/project and refuses a run
    whose project differs from the requested one, so a misconfigured run never
    masquerades as a valid one.
    """
    if config.wandb_mode == WANDB_MODE_DISABLED:
        return None
    try:
        import wandb
    except ImportError as e:
        raise RuntimeError("wandb is not installed. Run `uv sync --extra training` before enabling W&B.") from e

    init_kwargs = {
        "project": config.wandb_project,
        "mode": config.wandb_mode,
        "config": record,
        "dir": str(output_dir),
        "job_type": job_type,
        "tags": [*method_tags, *config.wandb_tags],
    }
    if config.wandb_entity:
        init_kwargs["entity"] = config.wandb_entity
    if config.wandb_run_name:
        init_kwargs["name"] = config.wandb_run_name
    if config.wandb_group:
        init_kwargs["group"] = config.wandb_group
    if config.wandb_run_id:
        init_kwargs["id"] = config.wandb_run_id
    if config.wandb_resume != WANDB_RESUME_NONE:
        init_kwargs["resume"] = config.wandb_resume

    run = wandb.init(**init_kwargs)
    run_id = _required_wandb_str(run, "id")
    run_project = _required_wandb_str(run, "project")
    if run_project != config.wandb_project:
        raise RuntimeError(f"W&B run project mismatch: expected {config.wandb_project!r}, got {run_project!r}")
    metadata = {
        "id": run_id,
        "name": getattr(run, "name", None),
        "project": run_project,
        "entity": getattr(run, "entity", None),
        "url": getattr(run, "url", None),
        "mode": config.wandb_mode,
    }
    (output_dir / WANDB_JSON).write_text(json_dumps_strict_pretty(metadata))
    return run


def log_wandb(run, metrics: dict, step: int) -> None:
    if run is not None:
        run.log(metrics, step=step)


def finish_wandb(run) -> None:
    if run is not None:
        run.finish()
