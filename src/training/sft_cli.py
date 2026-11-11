"""Config and command line for the Tinker SFT trainer.

Pure config: the dataclass, the cross-field validation argparse cannot express,
and the argument parser. No Tinker calls happen here, so the trainer module
(`tinker_sft`) can read as the training algorithm alone.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date

from src.training.sft_data import THINKING_MODES
from src.training.tinker_common import (
    CHECKPOINT_MODE_CONTINUE,
    CHECKPOINT_MODE_FRESH,
    CHECKPOINT_MODE_INIT,
    CHECKPOINT_MODE_RESUME,
    CHECKPOINT_MODES,
    LR_SCHEDULES,
    NO_CHECKPOINT_PATH,
    WANDB_MODE_DISABLED,
    WANDB_MODES,
    WANDB_RESUME_MODES,
    WANDB_RESUME_NONE,
    WANDB_RESUME_REQUIRES_RUN_ID,
    _bool,
    _csv_tags,
    _nonneg_float,
    _nonneg_int,
    _optional_str,
    _positive_float,
    _positive_int,
)


@dataclass(frozen=True)
class TinkerSFTConfig:
    """Complete config for one Tinker SFT run."""

    model: str
    renderer_name: str
    data: str
    output: str
    lora_rank: int
    learning_rate: float
    batch_size: int
    max_steps: int
    warmup_steps: int
    lr_schedule: str
    max_seq_length: int
    save_every: int
    log_every: int
    mask_prompt: bool
    thinking: str
    checkpoint_mode: str
    checkpoint_path: str
    start_step: int
    seed: int
    dry_run: bool
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    grad_clip_norm: float
    train_mlp: bool
    train_attn: bool
    train_unembed: bool
    wandb_mode: str
    wandb_project: str
    wandb_entity: str
    wandb_run_name: str
    wandb_group: str
    wandb_tags: tuple[str, ...]
    wandb_run_id: str
    wandb_resume: str
    renderer_current_date: str = ""


def validate_config(config: TinkerSFTConfig) -> None:
    """Validate config combinations that argparse alone cannot express."""
    if config.renderer_current_date:
        try:
            parsed_date = date.fromisoformat(config.renderer_current_date)
        except ValueError as e:
            raise ValueError("renderer_current_date must be a valid YYYY-MM-DD date") from e
        if parsed_date.isoformat() != config.renderer_current_date:
            raise ValueError("renderer_current_date must be a valid YYYY-MM-DD date")
    if config.lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"unknown lr_schedule {config.lr_schedule!r}; expected one of {LR_SCHEDULES}")
    if config.checkpoint_mode not in CHECKPOINT_MODES:
        raise ValueError(f"unknown checkpoint_mode {config.checkpoint_mode!r}; expected one of {CHECKPOINT_MODES}")
    if config.wandb_mode not in WANDB_MODES:
        raise ValueError(f"unknown wandb_mode {config.wandb_mode!r}; expected one of {WANDB_MODES}")
    if config.wandb_resume not in WANDB_RESUME_MODES:
        raise ValueError(f"unknown wandb_resume {config.wandb_resume!r}; expected one of {WANDB_RESUME_MODES}")
    if config.wandb_mode != WANDB_MODE_DISABLED and not config.wandb_project:
        raise ValueError("wandb_project is required when wandb_mode is not 'disabled'")
    if config.wandb_mode == WANDB_MODE_DISABLED and config.wandb_resume != WANDB_RESUME_NONE:
        raise ValueError("wandb_resume must be 'none' when wandb_mode is 'disabled'")
    if config.wandb_resume in WANDB_RESUME_REQUIRES_RUN_ID and not config.wandb_run_id:
        raise ValueError(f"wandb_run_id is required when wandb_resume={config.wandb_resume!r}")
    if config.log_every > config.max_steps:
        raise ValueError("log_every must be <= max_steps so every run emits at least one metric window")

    if config.checkpoint_mode == CHECKPOINT_MODE_FRESH:
        if config.checkpoint_path != NO_CHECKPOINT_PATH:
            raise ValueError(
                f"checkpoint_mode={CHECKPOINT_MODE_FRESH!r} requires checkpoint_path={NO_CHECKPOINT_PATH!r}"
            )
        if config.start_step != 0:
            raise ValueError(f"checkpoint_mode={CHECKPOINT_MODE_FRESH!r} requires start_step=0")
        return

    if config.checkpoint_path == NO_CHECKPOINT_PATH:
        raise ValueError(f"checkpoint_mode={config.checkpoint_mode!r} requires a real checkpoint_path")
    if config.checkpoint_mode == CHECKPOINT_MODE_INIT and config.start_step != 0:
        raise ValueError("checkpoint_mode='init' loads weights with a fresh optimizer, so start_step must be 0")
    if config.checkpoint_mode == CHECKPOINT_MODE_CONTINUE and config.start_step != 0:
        raise ValueError("checkpoint_mode='continue' starts a new job on new data, so start_step must be 0")
    if config.checkpoint_mode == CHECKPOINT_MODE_RESUME:
        if config.start_step <= 0:
            raise ValueError("checkpoint_mode='resume' restores an interrupted run and requires start_step > 0")
        if config.start_step >= config.max_steps:
            raise ValueError("start_step must be less than max_steps")


def build_parser() -> argparse.ArgumentParser:
    """The complete flag set of this trainer; `parse_args` reads it."""
    parser = argparse.ArgumentParser(
        description="Run one Tinker LoRA SFT job on JSONL chat rows with a `messages` list."
    )
    parser.add_argument("--model", required=True, help="base model, e.g. Qwen/Qwen3-8B")
    parser.add_argument("--renderer-name", required=True, help="Tinker cookbook renderer, e.g. qwen3")
    parser.add_argument(
        "--renderer-current-date",
        default="none",
        type=_optional_str,
        help="fixed YYYY-MM-DD for renderers with a dated system prompt; required for GPT-OSS",
    )
    parser.add_argument("--data", required=True, help="JSONL where each row is a chat object with a messages list")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--lora-rank", required=True, type=_positive_int)
    parser.add_argument("--learning-rate", required=True, type=_positive_float)
    parser.add_argument("--batch-size", required=True, type=_positive_int)
    parser.add_argument("--max-steps", required=True, type=_positive_int)
    parser.add_argument("--warmup-steps", required=True, type=_nonneg_int)
    parser.add_argument("--lr-schedule", required=True, choices=LR_SCHEDULES)
    parser.add_argument("--max-seq-length", required=True, type=_positive_int)
    parser.add_argument("--save-every", required=True, type=_positive_int)
    parser.add_argument("--log-every", required=True, type=_positive_int)
    parser.add_argument("--mask-prompt", required=True, type=_bool)
    parser.add_argument("--thinking", required=True, choices=THINKING_MODES)
    parser.add_argument("--checkpoint-mode", required=True, choices=CHECKPOINT_MODES)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--start-step", required=True, type=_nonneg_int)
    parser.add_argument("--seed", required=True, type=_nonneg_int)
    parser.add_argument("--dry-run", required=True, type=_bool)
    parser.add_argument("--beta1", required=True, type=_positive_float)
    parser.add_argument("--beta2", required=True, type=_positive_float)
    parser.add_argument("--eps", required=True, type=_positive_float)
    parser.add_argument("--weight-decay", required=True, type=_nonneg_float)
    parser.add_argument("--grad-clip-norm", required=True, type=_nonneg_float)
    parser.add_argument("--train-mlp", required=True, type=_bool)
    parser.add_argument("--train-attn", required=True, type=_bool)
    parser.add_argument("--train-unembed", required=True, type=_bool)
    parser.add_argument("--wandb-mode", required=True, choices=WANDB_MODES)
    parser.add_argument("--wandb-project", required=True, type=_optional_str)
    parser.add_argument("--wandb-entity", required=True, type=_optional_str)
    parser.add_argument("--wandb-run-name", required=True, type=_optional_str)
    parser.add_argument("--wandb-group", required=True, type=_optional_str)
    parser.add_argument("--wandb-tags", required=True, type=_csv_tags)
    parser.add_argument("--wandb-run-id", required=True, type=_optional_str)
    parser.add_argument("--wandb-resume", required=True, choices=WANDB_RESUME_MODES)
    return parser


def parse_args(argv: list[str] | None = None) -> TinkerSFTConfig:
    args = build_parser().parse_args(argv)
    return TinkerSFTConfig(
        model=args.model,
        renderer_name=args.renderer_name,
        data=args.data,
        output=args.output,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        max_seq_length=args.max_seq_length,
        save_every=args.save_every,
        log_every=args.log_every,
        mask_prompt=args.mask_prompt,
        thinking=args.thinking,
        checkpoint_mode=args.checkpoint_mode,
        checkpoint_path=args.checkpoint_path,
        start_step=args.start_step,
        seed=args.seed,
        dry_run=args.dry_run,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        train_mlp=args.train_mlp,
        train_attn=args.train_attn,
        train_unembed=args.train_unembed,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_group=args.wandb_group,
        wandb_tags=args.wandb_tags,
        wandb_run_id=args.wandb_run_id,
        wandb_resume=args.wandb_resume,
        renderer_current_date=args.renderer_current_date,
    )
