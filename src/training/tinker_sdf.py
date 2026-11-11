"""Fine-tune one Tinker LoRA from synthetic-document JSONL rows.

`--data-format` names the row format. With `documents`, each row is a plain
document with a non-empty `text` field; the trainer tokenizes it, applies no chat
template and trains every next-token target with cross-entropy loss. With
`arrays`, each row is one pre-rendered set of training arrays (input tokens,
target tokens and loss weights, as `src.training.arrays` reads them), built
upstream; the trainer does no tokenization and trains the targets the weights
select.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.keys import load_project_env
from src.training.arrays import TrainingArrays, arrays_to_tinker_datum, load_training_arrays_jsonl
from src.training.artifacts import (
    CHECKPOINTS_LOG,
    RESUME_ATTEMPTS_LOG,
    RESUME_META,
    TRAIN_LOG,
    TRAIN_STEPS_LOG,
)
from src.training.losses import mean_weighted_nll
from src.training.sdf_cli import TinkerSDFConfig, parse_args, validate_config
from src.training.sdf_data import SDFDataConfig, build_training_arrays, load_document_jsonl
from src.training.tinker_common import (
    CHECKPOINT_MODE_RESUME,
    _required_field,
    _tinker_path,
    arrays_sha256,
    create_training_client,
    fast_forward_data,
    file_sha256,
    finish_wandb,
    format_cost_estimate,
    get_git_commit,
    get_lr,
    init_wandb,
    json_dumps_strict,
    json_dumps_strict_pretty,
    load_tokenizer,
    log_wandb,
    require_tinker_key,
    trained_weight,
    validate_json_metric,
    write_config,
)
from src.training.tinker_common import validate_resume_config as _validate_resume_config

RESUME_MATCH_FIELDS = (
    "model",
    "tokenizer_id",
    "git_commit",
    "data_sha256",
    "arrays_sha256",
    "lora_rank",
    "batch_size",
    "max_steps",
    "learning_rate",
    "warmup_steps",
    "lr_schedule",
    "save_every",
    "log_every",
    "beta1",
    "beta2",
    "eps",
    "weight_decay",
    "grad_clip_norm",
    "train_mlp",
    "train_attn",
    "train_unembed",
    "seed",
    "max_seq_length",
    "append_eos",
    "terminal_token",
    "data_format",
    "checkpoint_steps",
)
WANDB_JOB_TYPE = "raw_document_tinker_training"
WANDB_METHOD_TAG = "raw_document"


def build_arrays(config: TinkerSDFConfig) -> tuple[list[TrainingArrays], int]:
    """Trainer-independent token arrays: raw document rows are tokenized here, pre-rendered arrays
    are loaded as they are."""
    if config.data_format == "arrays":
        arrays = load_training_arrays_jsonl(config.data)
        return arrays, len(arrays)
    tokenizer = load_tokenizer(config.tokenizer_id)
    documents = load_document_jsonl(config.data)
    data_config = SDFDataConfig(
        max_seq_length=config.max_seq_length,
        append_eos=config.append_eos,
        terminal_token=None if config.terminal_token == "none" else config.terminal_token,
    )
    arrays = build_training_arrays(documents, tokenizer, data_config)
    return arrays, len(documents)


def invocation_input_tokens(
    arrays: list[TrainingArrays], *, batch_size: int, seed: int, start_step: int, end_step: int
) -> int:
    """Exact billed model-input tokens for one deterministic invocation slice."""
    order = list(range(len(arrays)))
    rng = random.Random(seed)
    rng.shuffle(order)
    data_idx = 0
    total = 0
    for step in range(end_step):
        for _ in range(batch_size):
            if data_idx >= len(order):
                rng.shuffle(order)
                data_idx = 0
            if step >= start_step:
                total += len(arrays[order[data_idx]].input_tokens)
            data_idx += 1
    return total


def build_config_record(
    config: TinkerSDFConfig,
    n_examples: int,
    data_sha256: str,
    arrays_digest: str,
) -> dict[str, Any]:
    """Build the full effective run record persisted beside outputs."""
    record = asdict(config)
    record["checkpoint_steps"] = list(config.checkpoint_steps)
    record["git_commit"] = get_git_commit()
    record["n_examples"] = n_examples
    record["data_sha256"] = data_sha256
    record["arrays_sha256"] = arrays_digest
    return record


def validate_resume_config(config: TinkerSDFConfig, output_dir: Path, current_record: dict[str, Any]) -> None:
    """Validate a resume run against this run's recorded synthetic-document config fields."""
    _validate_resume_config(config, output_dir, current_record, RESUME_MATCH_FIELDS)


def _append_resume_attempt(record: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / RESUME_ATTEMPTS_LOG).open("a") as f:
        f.write(json_dumps_strict(record) + "\n")


def _jsonable_metrics(value) -> dict[str, Any] | None:
    """Validate Tinker metrics are finite and JSON-safe; None passes through (no metrics yet)."""
    if value is None:
        return None
    metrics = {}
    for key, metric in dict(value).items():
        if hasattr(metric, "item"):
            metric = metric.item()
        validate_json_metric(metric, context=f"metric {key!r}")
        metrics[key] = metric
    return metrics


def _required_metrics(response, response_name: str) -> dict[str, Any]:
    if not hasattr(response, "metrics"):
        raise ValueError(f"Tinker {response_name} response has no 'metrics' field")
    return _jsonable_metrics(response.metrics) or {}


def _optional_metrics(response, response_name: str) -> dict[str, Any] | None:
    if not hasattr(response, "metrics"):
        raise ValueError(f"Tinker {response_name} response has no 'metrics' field")
    return _jsonable_metrics(response.metrics)


async def verify_tokenizer_id(config: TinkerSDFConfig, training_client) -> None:
    """Verify Tinker will train token IDs from the tokenizer declared in config."""
    info = await training_client.get_info_async()
    model_data = _required_field(info, "model_data")
    tokenizer_id = _required_field(model_data, "tokenizer_id")
    if tokenizer_id != config.tokenizer_id:
        raise ValueError(
            f"Tinker tokenizer metadata mismatch: requested tokenizer_id={config.tokenizer_id!r}, got {tokenizer_id!r}"
        )


async def train_async(
    config: TinkerSDFConfig,
    arrays: list[TrainingArrays],
    data_sha256: str,
    wandb_run=None,
) -> None:
    """Run one Tinker synthetic-document fine-tuning job."""
    import tinker

    training_client = await create_training_client(config)
    await verify_tokenizer_id(config, training_client)
    train_data = [arrays_to_tinker_datum(row) for row in arrays]

    output_dir = Path(config.output)
    log_path = output_dir / TRAIN_LOG
    raw_log_path = output_dir / TRAIN_STEPS_LOG
    checkpoints_path = output_dir / CHECKPOINTS_LOG

    rng = random.Random(config.seed)
    rng.shuffle(train_data)
    data_idx = fast_forward_data(train_data, config, rng)
    end_step = config.max_steps if config.end_step == 0 else config.end_step

    # Reconstruct cumulative loss weight from the same deterministic cyclic
    # order used by the trainer. This keeps token receipts correct on an exact
    # optimizer-state resume without trusting a possibly partial local log.
    count_data = list(arrays)
    count_rng = random.Random(config.seed)
    count_rng.shuffle(count_data)
    count_idx = 0
    cumulative_trained_tokens = 0.0
    for _ in range(config.start_step):
        for _ in range(config.batch_size):
            if count_idx >= len(count_data):
                count_rng.shuffle(count_data)
                count_idx = 0
            cumulative_trained_tokens += sum(float(weight) for weight in count_data[count_idx].weights)
            count_idx += 1

    print(
        f"\nSynthetic-document training from step {config.start_step} to {end_step} "
        f"(LR horizon {config.max_steps}), "
        f"batch {config.batch_size}, LR {config.learning_rate} "
        f"({config.lr_schedule}, {config.warmup_steps} warmup), "
        f"on {len(train_data)} documents (~{len(train_data) / config.batch_size:.0f} batches/epoch)"
    )

    window_batch_nll = 0.0
    window_weighted_nll = 0.0
    window_loss_weight = 0.0
    window_steps = 0
    mean_loss = 0.0
    lr = config.learning_rate
    start_time = time.monotonic()
    log_mode = "a" if config.start_step else "w"
    checkpoint_mode = "a" if config.start_step else "w"

    with (
        open(log_path, log_mode) as log_file,
        open(raw_log_path, log_mode) as raw_log_file,
        open(checkpoints_path, checkpoint_mode) as ckpt_file,
    ):
        pbar = tqdm(total=end_step, initial=config.start_step, desc="Synthetic-document training", unit="step")
        for step in range(config.start_step, end_step):
            batch_start_idx = data_idx
            batch = []
            for _ in range(config.batch_size):
                if data_idx >= len(train_data):
                    rng.shuffle(train_data)
                    data_idx = 0
                batch.append(train_data[data_idx])
                data_idx += 1

            lr = get_lr(config.learning_rate, step, config.warmup_steps, config.max_steps, config.lr_schedule)
            adam_params = tinker.AdamParams(
                learning_rate=lr,
                beta1=config.beta1,
                beta2=config.beta2,
                eps=config.eps,
                weight_decay=config.weight_decay,
                grad_clip_norm=config.grad_clip_norm,
            )

            fwd_bwd_future = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
            optim_future = await training_client.optim_step_async(adam_params)
            fwd_bwd_result = await fwd_bwd_future.result_async()
            optim_result = await optim_future.result_async()

            mean_loss = mean_weighted_nll(fwd_bwd_result, batch)
            batch_loss_weight = sum(trained_weight(datum) for datum in batch)
            if not math.isfinite(mean_loss):
                raise ValueError(f"mean_loss must be finite, got {mean_loss!r}")
            if not math.isfinite(batch_loss_weight) or batch_loss_weight <= 0:
                raise ValueError(f"batch_loss_weight must be finite and positive, got {batch_loss_weight!r}")
            window_batch_nll += mean_loss
            window_weighted_nll += mean_loss * batch_loss_weight
            window_loss_weight += batch_loss_weight
            window_steps += 1
            cumulative_trained_tokens += batch_loss_weight
            fwd_bwd_metrics = _required_metrics(fwd_bwd_result, "forward/backward")
            optim_metrics = _optional_metrics(optim_result, "optimizer step")

            done = step + 1
            step_record = {
                "step": done,
                "mean_loss": mean_loss,
                "batch_loss_weight": batch_loss_weight,
                "cumulative_trained_tokens": cumulative_trained_tokens,
                "lr": lr,
                "batch_start_index": batch_start_idx,
                "next_training_row_index": data_idx,
                "tinker_forward_backward_metrics": fwd_bwd_metrics,
                "tinker_optim_metrics": optim_metrics,
            }
            raw_log_file.write(json_dumps_strict(step_record) + "\n")
            raw_log_file.flush()
            pbar.update(1)
            pbar.set_postfix(loss=f"{mean_loss:.4f}", lr=f"{lr:.2e}")

            if done % config.log_every == 0 or done == end_step:
                elapsed = time.monotonic() - start_time
                if elapsed <= 0:
                    raise RuntimeError(f"elapsed training time must be positive before logging speed, got {elapsed}")
                log_record = {
                    "step": done,
                    "mean_batch_nll": window_batch_nll / window_steps,
                    "mean_token_nll": window_weighted_nll / window_loss_weight,
                    "window_steps": window_steps,
                    "window_loss_weight": window_loss_weight,
                    "lr": lr,
                    "steps_per_sec": (done - config.start_step) / elapsed,
                    "elapsed_sec": elapsed,
                }
                log_file.write(json_dumps_strict(log_record) + "\n")
                log_file.flush()
                log_wandb(
                    wandb_run,
                    {
                        "train/mean_batch_nll": log_record["mean_batch_nll"],
                        "train/mean_token_nll": log_record["mean_token_nll"],
                        "train/window_steps": log_record["window_steps"],
                        "train/window_loss_weight": log_record["window_loss_weight"],
                        "train/lr": log_record["lr"],
                        "train/steps_per_sec": log_record["steps_per_sec"],
                        "train/elapsed_sec": log_record["elapsed_sec"],
                    },
                    step=done,
                )
                window_batch_nll = 0.0
                window_weighted_nll = 0.0
                window_loss_weight = 0.0
                window_steps = 0

            periodic_checkpoint = config.save_every > 0 and done % config.save_every == 0
            exact_checkpoint = done in config.checkpoint_steps
            if (periodic_checkpoint or exact_checkpoint) and done < end_step:
                path = await _tinker_path(training_client.save_weights_for_sampler_async(name=f"step_{done}"))
                ckpt_record = {
                    "step": done,
                    "type": "sampler",
                    "path": path,
                    "batch_mean_nll": mean_loss,
                    "lr": lr,
                    "cumulative_trained_tokens": cumulative_trained_tokens,
                    "checkpoint_trigger": "exact" if exact_checkpoint else "periodic",
                }
                ckpt_file.write(json_dumps_strict(ckpt_record) + "\n")
                ckpt_file.flush()
                log_wandb(
                    wandb_run,
                    {
                        "checkpoint/saved": 1,
                        "checkpoint/type": "sampler",
                        "checkpoint/batch_mean_nll": mean_loss,
                        "checkpoint/lr": lr,
                    },
                    step=done,
                )
                # Also save full optimizer state at every periodic or exact boundary,
                # so an evaluated checkpoint can be resumed without changing trajectory.
                state_path = await _tinker_path(training_client.save_state_async(name=f"state_step_{done}"))
                ckpt_file.write(
                    json_dumps_strict(
                        {
                            "step": done,
                            "type": "state",
                            "path": state_path,
                            "batch_mean_nll": mean_loss,
                            "lr": lr,
                            "cumulative_trained_tokens": cumulative_trained_tokens,
                            "checkpoint_trigger": "exact" if exact_checkpoint else "periodic",
                        }
                    )
                    + "\n"
                )
                ckpt_file.flush()
        pbar.close()

        is_final = end_step == config.max_steps
        save_name = "final" if is_final else f"pause_step_{end_step}"
        state_name = "state_final" if is_final else f"state_pause_step_{end_step}"
        sampler_path = await _tinker_path(training_client.save_weights_for_sampler_async(name=save_name))
        state_path = await _tinker_path(training_client.save_state_async(name=state_name))
        ckpt_file.write(
            json_dumps_strict(
                {
                    "step": end_step,
                    "type": "sampler",
                    "path": sampler_path,
                    "batch_mean_nll": mean_loss,
                    "lr": lr,
                    "cumulative_trained_tokens": cumulative_trained_tokens,
                    "final": is_final,
                    "paused": not is_final,
                    "checkpoint_trigger": ("exact" if end_step in config.checkpoint_steps else "endpoint"),
                }
            )
            + "\n"
        )
        ckpt_file.write(
            json_dumps_strict(
                {
                    "step": end_step,
                    "type": "state",
                    "path": state_path,
                    "batch_mean_nll": mean_loss,
                    "lr": lr,
                    "cumulative_trained_tokens": cumulative_trained_tokens,
                    "final": is_final,
                    "paused": not is_final,
                    "checkpoint_trigger": ("exact" if end_step in config.checkpoint_steps else "endpoint"),
                }
            )
            + "\n"
        )

    arrays_digest = arrays_sha256(arrays)
    (output_dir / RESUME_META).write_text(
        json_dumps_strict_pretty(
            {
                "step": end_step,
                "state_path": state_path,
                "sampler_path": sampler_path,
                "data_sha256": data_sha256,
                "arrays_sha256": arrays_digest,
                "seed": config.seed,
                "batch_size": config.batch_size,
            },
        )
    )
    if wandb_run is not None:
        wandb_run.summary["latest_sampler_checkpoint"] = sampler_path
        wandb_run.summary["latest_state_checkpoint"] = state_path
        wandb_run.summary["latest_step"] = end_step
        wandb_run.summary["latest_batch_mean_nll"] = mean_loss
        wandb_run.summary["training_complete"] = is_final
        wandb_run.summary["data_sha256"] = data_sha256
        wandb_run.summary["arrays_sha256"] = arrays_digest
        log_wandb(
            wandb_run,
            {
                "checkpoint/saved": 1,
                "checkpoint/type": "final" if is_final else "pause",
                "checkpoint/batch_mean_nll": mean_loss,
                "checkpoint/lr": lr,
            },
            step=end_step,
        )
    status = "complete" if is_final else f"paused at step {end_step}/{config.max_steps}"
    print(f"\nSynthetic-document training {status} in {time.monotonic() - start_time:.1f}s.")
    print(f"Latest sampler checkpoint: {sampler_path}")
    print(f"Latest full state checkpoint: {state_path}")


def run(config: TinkerSDFConfig) -> None:
    """Build data, persist config, and optionally launch Tinker training."""
    validate_config(config)
    output_dir = Path(config.output)
    data_sha256 = file_sha256(config.data)
    arrays, n_examples = build_arrays(config)
    arrays_digest = arrays_sha256(arrays)
    record = build_config_record(config, n_examples, data_sha256, arrays_digest)
    validate_resume_config(config, output_dir, record)
    if config.checkpoint_mode == CHECKPOINT_MODE_RESUME:
        _append_resume_attempt(record, output_dir)
    else:
        write_config(record, output_dir)

    end_step = config.max_steps if config.end_step == 0 else config.end_step
    remaining_steps = end_step - config.start_step
    document_uses = remaining_steps * config.batch_size
    total_document_uses = config.max_steps * config.batch_size
    print(
        f"\nRun plan: synthetic-document training {remaining_steps} remaining steps "
        f"({config.start_step}->{end_step}; LR horizon {config.max_steps}) x batch {config.batch_size} = "
        f"{document_uses} document uses this invocation over {len(arrays)} documents "
        f"(~{document_uses / len(arrays):.1f} epochs; total target {total_document_uses} document uses), "
        f"LoRA rank {config.lora_rank} on {config.model}."
    )
    # Tinker receives ModelInput of exactly this length; the shifted final target
    # is paired with the final input token rather than adding another model input.
    # Price the exact deterministic examples in THIS invocation, not a mean-length
    # approximation and not the full horizon when this invocation is a pause/resume slice.
    invocation_tokens = invocation_input_tokens(
        arrays,
        batch_size=config.batch_size,
        seed=config.seed,
        start_step=config.start_step,
        end_step=end_step,
    )
    print(format_cost_estimate(config.model, invocation_tokens))
    if config.dry_run:
        print("\nDry run: arrays built and config written; skipping Tinker calls.")
        return

    require_tinker_key()
    wandb_run = init_wandb(config, record, output_dir, job_type=WANDB_JOB_TYPE, method_tags=(WANDB_METHOD_TAG,))
    try:
        asyncio.run(train_async(config, arrays, data_sha256, wandb_run))
    finally:
        finish_wandb(wandb_run)


def main() -> None:
    load_project_env()
    run(parse_args())


if __name__ == "__main__":
    main()
