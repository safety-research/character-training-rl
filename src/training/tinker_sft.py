"""Fine-tune one Tinker LoRA from JSONL chat rows with a `messages` list.

The `--thinking` mode (keep/mask/strip/strip_empty) controls how the assistant's
reasoning is handled. This includes both legacy `<think>...</think>` strings and
GPT-OSS Harmony `thinking`/`text` content parts; see `sft_data` for the semantics.

This module is the SFT training algorithm: load data, build arrays, run the
train loop, save checkpoints. Config and CLI live in `sft_cli`; shared SDK glue
in `tinker_common`; the loss math in `losses`.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from src.keys import load_project_env
from src.training.arrays import TrainingArrays, arrays_to_tinker_datum
from src.training.artifacts import CHECKPOINTS_LOG, RESUME_META, TRAIN_LOG, TRAIN_STEPS_LOG
from src.training.losses import mean_weighted_nll
from src.training.sft_cli import TinkerSFTConfig, parse_args, validate_config
from src.training.sft_data import (
    SFTDataConfig,
    build_training_arrays,
    load_chat_jsonl,
)
from src.training.tinker_common import (
    _tinker_path,
    arrays_sha256,
    create_training_client,
    estimate_train_tokens,
    fast_forward_data,
    file_sha256,
    finish_wandb,
    format_cost_estimate,
    get_git_commit,
    get_lr,
    init_wandb,
    json_dumps_strict,
    log_wandb,
    require_tinker_key,
    trained_weight,
    write_config,
)
from src.training.tinker_common import validate_resume_config as _validate_resume_config

WANDB_JOB_TYPE = "tinker_sft"
RESUME_MATCH_FIELDS = (
    "model",
    "renderer_name",
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
    "mask_prompt",
    "thinking",
)


def build_arrays(config: TinkerSFTConfig) -> tuple[list[TrainingArrays], int]:
    """Load chat rows and convert them to trainer-independent token arrays."""
    from src.training.tinker_common import apply_renderer_date, load_renderer, load_tokenizer

    tokenizer = load_tokenizer(config.model)
    renderer = load_renderer(config.model, config.renderer_name, tokenizer)
    apply_renderer_date(renderer, config.renderer_current_date or None)
    examples = load_chat_jsonl(config.data)
    data_config = SFTDataConfig(
        max_seq_length=config.max_seq_length,
        mask_prompt=config.mask_prompt,
        thinking=config.thinking,
    )
    arrays = build_training_arrays(examples, renderer, data_config)
    return arrays, len(examples)


def build_config_record(config: TinkerSFTConfig, n_examples: int, data_sha256: str, arrays_digest: str) -> dict:
    record = asdict(config)
    record["git_commit"] = get_git_commit()
    record["n_examples"] = n_examples
    record["data_sha256"] = data_sha256
    record["arrays_sha256"] = arrays_digest
    return record


def validate_resume_config(config: TinkerSFTConfig, output_dir: Path, current_record: dict) -> None:
    """Validate a resume run against this run's recorded SFT config fields."""
    fields = RESUME_MATCH_FIELDS
    if config.renderer_current_date:
        fields = (*fields, "renderer_current_date")
    _validate_resume_config(config, output_dir, current_record, fields)


async def train_async(config: TinkerSFTConfig, arrays: list[TrainingArrays], data_sha256: str, wandb_run=None) -> None:
    """Run one Tinker SFT job."""
    import tinker

    training_client = await create_training_client(config)
    train_data = [arrays_to_tinker_datum(row) for row in arrays]

    output_dir = Path(config.output)
    log_path = output_dir / TRAIN_LOG
    raw_log_path = output_dir / TRAIN_STEPS_LOG
    checkpoints_path = output_dir / CHECKPOINTS_LOG

    rng = random.Random(config.seed)
    rng.shuffle(train_data)
    data_idx = fast_forward_data(train_data, config, rng)

    print(
        f"\nTraining from step {config.start_step} to {config.max_steps}, batch {config.batch_size}, "
        f"LR {config.learning_rate} ({config.lr_schedule}, {config.warmup_steps} warmup), "
        f"on {len(train_data)} examples (~{len(train_data) / config.batch_size:.0f} batches/epoch)"
    )

    window_batch_nll = 0.0
    window_weighted_nll = 0.0
    window_loss_weight = 0.0
    window_steps = 0
    mean_loss = 0.0
    lr = config.learning_rate
    start_time = time.time()
    log_mode = "a" if config.start_step else "w"
    checkpoint_mode = "a" if config.start_step else "w"

    with (
        open(log_path, log_mode) as log_file,
        open(raw_log_path, log_mode) as raw_log_file,
        open(checkpoints_path, checkpoint_mode) as ckpt_file,
    ):
        pbar = tqdm(total=config.max_steps, initial=config.start_step, desc="Training", unit="step")
        for step in range(config.start_step, config.max_steps):
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
            await optim_future.result_async()

            mean_loss = mean_weighted_nll(fwd_bwd_result, batch)
            batch_loss_weight = sum(trained_weight(datum) for datum in batch)
            window_batch_nll += mean_loss
            window_weighted_nll += mean_loss * batch_loss_weight
            window_loss_weight += batch_loss_weight
            window_steps += 1

            done = step + 1
            step_record = {
                "step": done,
                "mean_loss": mean_loss,
                "batch_loss_weight": batch_loss_weight,
                "lr": lr,
                "batch_start_index": batch_start_idx,
                "next_training_row_index": data_idx,
            }
            raw_log_file.write(json_dumps_strict(step_record) + "\n")
            raw_log_file.flush()
            pbar.update(1)
            pbar.set_postfix(loss=f"{mean_loss:.4f}", lr=f"{lr:.2e}")

            if done % config.log_every == 0 or done == config.max_steps:
                elapsed = time.time() - start_time
                log_record = {
                    "step": done,
                    "mean_batch_nll": window_batch_nll / window_steps,
                    "mean_token_nll": window_weighted_nll / window_loss_weight,
                    "window_steps": window_steps,
                    "window_loss_weight": window_loss_weight,
                    "lr": lr,
                    "steps_per_sec": (done - config.start_step) / elapsed if elapsed else 0.0,
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

            if done % config.save_every == 0 and done < config.max_steps:
                path = await _tinker_path(training_client.save_weights_for_sampler_async(name=f"step_{done}"))
                ckpt_record = {"step": done, "type": "sampler", "path": path, "batch_mean_nll": mean_loss, "lr": lr}
                ckpt_file.write(json_dumps_strict(ckpt_record) + "\n")
                ckpt_file.flush()
                # Also save a trainable state at each interval so an intermediate checkpoint can be
                # init'd/resumed from (sampler weights alone cannot: load_weights rejects them). This is
                # the point of intermediate checkpoints -- to continue/branch training from a mid-run step.
                state_path = await _tinker_path(training_client.save_state_async(name=f"state_{done}"))
                ckpt_file.write(
                    json_dumps_strict(
                        {"step": done, "type": "state", "path": state_path, "batch_mean_nll": mean_loss, "lr": lr}
                    )
                    + "\n"
                )
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
        pbar.close()

        sampler_path = await _tinker_path(training_client.save_weights_for_sampler_async(name="final"))
        state_path = await _tinker_path(training_client.save_state_async(name="state_final"))
        ckpt_file.write(
            json_dumps_strict(
                {
                    "step": config.max_steps,
                    "type": "sampler",
                    "path": sampler_path,
                    "batch_mean_nll": mean_loss,
                    "lr": lr,
                    "final": True,
                }
            )
            + "\n"
        )
        ckpt_file.write(
            json_dumps_strict(
                {
                    "step": config.max_steps,
                    "type": "state",
                    "path": state_path,
                    "batch_mean_nll": mean_loss,
                    "lr": lr,
                    "final": True,
                }
            )
            + "\n"
        )

    (output_dir / RESUME_META).write_text(
        json.dumps(
            {
                "step": config.max_steps,
                "state_path": state_path,
                "sampler_path": sampler_path,
                "data_sha256": data_sha256,
                "arrays_sha256": arrays_sha256(arrays),
                "seed": config.seed,
                "batch_size": config.batch_size,
            },
            indent=2,
        )
    )
    if wandb_run is not None:
        wandb_run.summary["final_sampler_checkpoint"] = sampler_path
        wandb_run.summary["final_state_checkpoint"] = state_path
        wandb_run.summary["final_step"] = config.max_steps
        wandb_run.summary["final_batch_mean_nll"] = mean_loss
        wandb_run.summary["data_sha256"] = data_sha256
        wandb_run.summary["arrays_sha256"] = arrays_sha256(arrays)
        log_wandb(
            wandb_run,
            {
                "checkpoint/saved": 1,
                "checkpoint/type": "final",
                "checkpoint/batch_mean_nll": mean_loss,
                "checkpoint/lr": lr,
            },
            step=config.max_steps,
        )
    print(f"\nTraining complete in {time.time() - start_time:.1f}s.")
    print(f"Final sampler checkpoint: {sampler_path}")
    print(f"Final full state checkpoint: {state_path}")


def run(config: TinkerSFTConfig) -> None:
    validate_config(config)
    output_dir = Path(config.output)
    data_sha256 = file_sha256(config.data)
    arrays, n_examples = build_arrays(config)
    arrays_digest = arrays_sha256(arrays)
    record = build_config_record(config, n_examples, data_sha256, arrays_digest)
    validate_resume_config(config, output_dir, record)
    write_config(record, output_dir)

    remaining = config.max_steps - config.start_step
    example_passes = remaining * config.batch_size
    print(
        f"\nRun plan: {remaining} steps x batch {config.batch_size} = "
        f"{example_passes} training row uses "
        f"over {len(arrays)} examples (~{example_passes / len(arrays):.1f} epochs), "
        f"LoRA rank {config.lora_rank} on {config.model}."
    )
    token_lengths = [len(row.input_tokens) + 1 for row in arrays]
    # The estimate prices the steps this invocation runs, the same remaining steps the
    # run-plan line reports; on a resume that is fewer than the whole run.
    remaining_steps = config.max_steps - config.start_step
    print(format_cost_estimate(config.model, estimate_train_tokens(token_lengths, config.batch_size, remaining_steps)))
    if config.dry_run:
        print("\nDry run: arrays built and config written; skipping Tinker calls.")
        return

    require_tinker_key()
    wandb_run = init_wandb(config, record, output_dir, job_type=WANDB_JOB_TYPE)
    try:
        asyncio.run(train_async(config, arrays, data_sha256, wandb_run))
    finally:
        finish_wandb(wandb_run)


def main() -> None:
    load_project_env()
    run(parse_args())


if __name__ == "__main__":
    main()
