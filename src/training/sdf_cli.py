"""Config and command line for the Tinker synthetic-document (SDF) trainer.

Pure config: the dataclass, cross-field validation, and the argument parser. No
Tinker calls happen here, so `tinker_sdf` can read as the training algorithm.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

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


def _checkpoint_steps(value: str) -> tuple[int, ...]:
    """Parse an ascending CSV of exact optimizer steps, or ``none``."""
    if value == "none":
        return ()
    try:
        steps = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoint steps must be ascending positive integers or 'none'") from error
    if not steps or any(step <= 0 for step in steps) or tuple(sorted(set(steps))) != steps:
        raise argparse.ArgumentTypeError("checkpoint steps must be unique, positive, and ascending")
    return steps


@dataclass(frozen=True)
class TinkerSDFConfig:
    """Complete config for one raw synthetic-document Tinker run."""

    model: str
    tokenizer_id: str
    data: str
    output: str
    lora_rank: int
    learning_rate: float
    batch_size: int
    max_steps: int
    warmup_steps: int
    lr_schedule: str
    max_seq_length: int
    append_eos: bool
    save_every: int
    log_every: int
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
    # ``terminal_token`` names an exact single token appended to each raw document
    # ("none" appends nothing); GPT-OSS raw documents use it for a non-chat
    # terminator. ``data_format`` is "documents" for raw `text` rows tokenized here,
    # or "arrays" for already-rendered training arrays.
    terminal_token: str = "none"
    data_format: str = "documents"
    # ``max_steps`` owns the LR schedule horizon. ``end_step=0`` means run to
    # that horizon; a positive earlier end cleanly pauses an invocation without
    # changing the schedule that an optimizer-state resume will continue.
    end_step: int = 0
    checkpoint_steps: tuple[int, ...] = ()


def validate_config(config: TinkerSDFConfig) -> None:
    """Validate config combinations that argparse alone cannot express."""
    if config.lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"unknown lr_schedule {config.lr_schedule!r}; expected one of {LR_SCHEDULES}")
    if config.checkpoint_mode not in CHECKPOINT_MODES:
        raise ValueError(f"unknown checkpoint_mode {config.checkpoint_mode!r}; expected one of {CHECKPOINT_MODES}")
    if config.wandb_mode not in WANDB_MODES:
        raise ValueError(f"unknown wandb_mode {config.wandb_mode!r}; expected one of {WANDB_MODES}")
    if config.wandb_resume not in WANDB_RESUME_MODES:
        raise ValueError(f"unknown wandb_resume {config.wandb_resume!r}; expected one of {WANDB_RESUME_MODES}")
    if not isinstance(config.tokenizer_id, str) or not config.tokenizer_id:
        raise ValueError(f"tokenizer_id must be a non-empty string, got {config.tokenizer_id!r}")
    if not isinstance(config.append_eos, bool):
        raise ValueError(f"append_eos must be a bool, got {config.append_eos!r}")
    if config.data_format not in {"documents", "arrays"}:
        raise ValueError(f"data_format must be 'documents' or 'arrays', got {config.data_format!r}")
    if not isinstance(config.terminal_token, str):
        raise ValueError(f"terminal_token must be a string, got {config.terminal_token!r}")
    if config.data_format == "documents":
        if config.append_eos and config.terminal_token != "none":
            raise ValueError("append_eos=true and terminal_token are mutually exclusive")
    elif config.append_eos or config.terminal_token != "none":
        raise ValueError("data_format='arrays' requires append_eos=false and terminal_token=none")
    if config.wandb_mode != WANDB_MODE_DISABLED and not config.wandb_project:
        raise ValueError("wandb_project is required when wandb_mode is not 'disabled'")
    if config.wandb_mode == WANDB_MODE_DISABLED and config.wandb_resume != WANDB_RESUME_NONE:
        raise ValueError("wandb_resume must be 'none' when wandb_mode is 'disabled'")
    if config.wandb_resume in WANDB_RESUME_REQUIRES_RUN_ID and not config.wandb_run_id:
        raise ValueError(f"wandb_run_id is required when wandb_resume={config.wandb_resume!r}")
    end_step = config.max_steps if config.end_step == 0 else config.end_step
    if end_step <= 0 or end_step > config.max_steps:
        raise ValueError("end_step must be 0 or in [1, max_steps]")
    if config.start_step >= end_step:
        raise ValueError("start_step must be less than the resolved end_step")
    if config.save_every < 0:
        raise ValueError("save_every must be nonnegative")
    if config.log_every > end_step:
        raise ValueError("log_every must be <= the resolved end_step so every run emits at least one metric window")
    if tuple(sorted(set(config.checkpoint_steps))) != config.checkpoint_steps:
        raise ValueError("checkpoint_steps must be unique and ascending")
    if any(step <= 0 or step > config.max_steps for step in config.checkpoint_steps):
        raise ValueError("checkpoint_steps must lie in [1, max_steps]")
    if not (config.train_mlp or config.train_attn or config.train_unembed):
        raise ValueError("at least one of train_mlp, train_attn, or train_unembed must be true")

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
        if config.start_step >= end_step:
            raise ValueError("start_step must be less than the resolved end_step")


def build_parser() -> argparse.ArgumentParser:
    """The complete flag set of the synthetic-document fine-tuning trainer; `parse_args` reads it."""
    parser = argparse.ArgumentParser(
        description="Run one Tinker LoRA synthetic-document fine-tuning job on JSONL rows: raw documents "
        "or pre-rendered training arrays (--data-format).",
        allow_abbrev=False,
    )
    parser.add_argument("--model", required=True, help="base model, e.g. Qwen/Qwen3-8B")
    parser.add_argument(
        "--tokenizer-id", required=True, help="declared tokenizer id to use locally and verify in Tinker"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="JSONL; with --data-format documents each row is an object with a non-empty text field, "
        "with --data-format arrays each row is one pre-rendered TrainingArrays object",
    )
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--lora-rank", required=True, type=_positive_int)
    parser.add_argument("--learning-rate", required=True, type=_positive_float)
    parser.add_argument("--batch-size", required=True, type=_positive_int)
    parser.add_argument("--max-steps", required=True, type=_positive_int)
    parser.add_argument("--warmup-steps", required=True, type=_nonneg_int)
    parser.add_argument("--lr-schedule", required=True, choices=LR_SCHEDULES)
    parser.add_argument("--max-seq-length", required=True, type=_positive_int)
    parser.add_argument("--append-eos", required=True, type=_bool)
    parser.add_argument(
        "--terminal-token",
        default="none",
        help="exact single-token document terminator, or 'none'; GPT-OSS raw documents use <|endoftext|>",
    )
    parser.add_argument(
        "--data-format",
        choices=("documents", "arrays"),
        default="documents",
        help="raw text document JSONL or strict pre-rendered TrainingArrays JSONL",
    )
    parser.add_argument("--save-every", required=True, type=_nonneg_int, help="periodic step cadence; 0 disables")
    parser.add_argument(
        "--end-step",
        type=_nonneg_int,
        default=0,
        help="invocation end step; 0 runs through max-steps without changing the LR horizon",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=_checkpoint_steps,
        default=(),
        help="exact ascending optimizer steps to checkpoint, or 'none'",
    )
    parser.add_argument("--log-every", required=True, type=_positive_int)
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


def parse_args(argv: list[str] | None = None) -> TinkerSDFConfig:
    args = build_parser().parse_args(argv)
    return TinkerSDFConfig(
        model=args.model,
        tokenizer_id=args.tokenizer_id,
        data=args.data,
        output=args.output,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        max_seq_length=args.max_seq_length,
        append_eos=args.append_eos,
        save_every=args.save_every,
        log_every=args.log_every,
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
        terminal_token=args.terminal_token,
        data_format=args.data_format,
        end_step=args.end_step,
        checkpoint_steps=args.checkpoint_steps,
    )
