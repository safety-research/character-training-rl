"""Config and command line for the Tinker RL trainer (GRPO / DPO / KTO).

Pure config: the method vocabularies, the per-method dataclasses, cross-field
validation, and the subcommand argument parser. No Tinker calls happen here, so
`tinker_rl` can read as the training algorithm.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from src.training.rl_data import IMPORTANCE_SAMPLING_LOSS
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

METHOD_GRPO = "grpo"
METHOD_DPO = "dpo"
METHOD_KTO = "kto"
METHODS = (METHOD_GRPO, METHOD_DPO, METHOD_KTO)
GRPO_LOSS_FN_PPO = "ppo"
GRPO_LOSS_FN_CISPO = "cispo"
GRPO_LOSS_FNS = (IMPORTANCE_SAMPLING_LOSS, GRPO_LOSS_FN_PPO, GRPO_LOSS_FN_CISPO)
REFERENCE_BASE_MODEL = "base_model"


@dataclass(frozen=True)
class CommonRLConfig:
    """Fields shared by every RL method."""

    method: str
    model: str
    data: str
    output: str
    split: str
    lora_rank: int
    learning_rate: float
    batch_size: int
    max_steps: int
    warmup_steps: int
    lr_schedule: str
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


@dataclass(frozen=True)
class GRPOConfig(CommonRLConfig):
    """Config for offline GRPO-style importance-sampling updates."""

    grpo_loss_fn: str
    grpo_clip_low_threshold: float
    grpo_clip_high_threshold: float


@dataclass(frozen=True)
class DPOConfig(CommonRLConfig):
    """Config for DPO preference updates."""

    renderer_name: str
    max_seq_length: int
    mask_prompt: bool
    thinking: str
    dpo_beta: float
    reference_model_path: str


@dataclass(frozen=True)
class KTOConfig(CommonRLConfig):
    """Config for KTO labeled-completion updates."""

    renderer_name: str
    max_seq_length: int
    mask_prompt: bool
    thinking: str
    kto_beta: float
    desirable_weight: float
    undesirable_weight: float
    reference_model_path: str


RLConfig = GRPOConfig | DPOConfig | KTOConfig


def _require_method_config(config: RLConfig) -> None:
    expected = {
        GRPOConfig: METHOD_GRPO,
        DPOConfig: METHOD_DPO,
        KTOConfig: METHOD_KTO,
    }[type(config)]
    if config.method != expected:
        raise ValueError(f"{type(config).__name__} requires method={expected!r}, got {config.method!r}")


def validate_config(config: RLConfig) -> None:
    """Validate config combinations that argparse alone cannot express."""
    _require_method_config(config)
    if config.method not in METHODS:
        raise ValueError(f"unknown method {config.method!r}; expected one of {METHODS}")
    if not config.split:
        raise ValueError("split must be non-empty")
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
    else:
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

    if isinstance(config, GRPOConfig) and config.grpo_loss_fn not in GRPO_LOSS_FNS:
        raise ValueError(f"unknown grpo_loss_fn {config.grpo_loss_fn!r}; expected one of {GRPO_LOSS_FNS}")
    if isinstance(config, GRPOConfig):
        if config.grpo_loss_fn == IMPORTANCE_SAMPLING_LOSS:
            if config.grpo_clip_low_threshold != 0.0 or config.grpo_clip_high_threshold != 0.0:
                raise ValueError("importance_sampling requires both GRPO clip thresholds to be 0.0")
        elif config.grpo_loss_fn == GRPO_LOSS_FN_PPO and (
            config.grpo_clip_low_threshold <= 0.0 or config.grpo_clip_high_threshold <= 0.0
        ):
            raise ValueError("ppo requires positive GRPO clip thresholds")
        elif config.grpo_loss_fn == GRPO_LOSS_FN_CISPO and (
            config.grpo_clip_low_threshold < 0.0 or config.grpo_clip_high_threshold <= 0.0
        ):
            raise ValueError("cispo requires a nonnegative low threshold and a positive high threshold")
        elif config.grpo_clip_low_threshold >= config.grpo_clip_high_threshold:
            raise ValueError("grpo_clip_low_threshold must be less than grpo_clip_high_threshold")
    if isinstance(config, DPOConfig | KTOConfig) and not config.reference_model_path:
        raise ValueError("reference_model_path must be non-empty")
    if isinstance(config, DPOConfig | KTOConfig):
        if config.checkpoint_mode != CHECKPOINT_MODE_FRESH and config.reference_model_path == REFERENCE_BASE_MODEL:
            raise ValueError(
                "reference_model_path='base_model' is only valid for fresh preference runs. "
                "When initializing or resuming from a checkpoint, pass the frozen start-of-phase sampler path."
            )
    if isinstance(config, KTOConfig) and config.batch_size < 2:
        raise ValueError("KTO requires batch_size >= 2 to build mismatched prompt/completion pairs for the KL baseline")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="base model, e.g. Qwen/Qwen3-8B")
    parser.add_argument("--data", required=True, help="method-specific JSONL training data")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--split", required=True, help="data split to train on, e.g. train")
    parser.add_argument("--lora-rank", required=True, type=_positive_int)
    parser.add_argument("--learning-rate", required=True, type=_positive_float)
    parser.add_argument("--batch-size", required=True, type=_positive_int)
    parser.add_argument("--max-steps", required=True, type=_positive_int)
    parser.add_argument("--warmup-steps", required=True, type=_nonneg_int)
    parser.add_argument("--lr-schedule", required=True, choices=LR_SCHEDULES)
    parser.add_argument("--save-every", required=True, type=_positive_int)
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


def _add_preference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--renderer-name", required=True, help="renderer name used to format prompts and completions, e.g. qwen3"
    )
    parser.add_argument("--max-seq-length", required=True, type=_positive_int)
    parser.add_argument("--mask-prompt", required=True, type=_bool)
    parser.add_argument("--thinking", required=True, choices=THINKING_MODES)


def _common_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": args.method,
        "model": args.model,
        "data": args.data,
        "output": args.output,
        "split": args.split,
        "lora_rank": args.lora_rank,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "lr_schedule": args.lr_schedule,
        "save_every": args.save_every,
        "log_every": args.log_every,
        "checkpoint_mode": args.checkpoint_mode,
        "checkpoint_path": args.checkpoint_path,
        "start_step": args.start_step,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "eps": args.eps,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "train_mlp": args.train_mlp,
        "train_attn": args.train_attn,
        "train_unembed": args.train_unembed,
        "wandb_mode": args.wandb_mode,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_name": args.wandb_run_name,
        "wandb_group": args.wandb_group,
        "wandb_tags": args.wandb_tags,
        "wandb_run_id": args.wandb_run_id,
        "wandb_resume": args.wandb_resume,
    }


def _preference_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "renderer_name": args.renderer_name,
        "max_seq_length": args.max_seq_length,
        "mask_prompt": args.mask_prompt,
        "thinking": args.thinking,
    }


def build_parser() -> argparse.ArgumentParser:
    """The complete flag set of this trainer; `parse_args` reads it."""
    parser = argparse.ArgumentParser(description="Run one Tinker reinforcement-learning optimization phase.")
    subparsers = parser.add_subparsers(dest="method", required=True)

    grpo = subparsers.add_parser(METHOD_GRPO, help="offline GRPO from precomputed rollout rows")
    _add_common_args(grpo)
    grpo.add_argument("--grpo-loss-fn", required=True, choices=GRPO_LOSS_FNS)
    grpo.add_argument("--grpo-clip-low-threshold", required=True, type=_nonneg_float)
    grpo.add_argument("--grpo-clip-high-threshold", required=True, type=_nonneg_float)

    dpo = subparsers.add_parser(METHOD_DPO, help="DPO from prompt/chosen/rejected rows")
    _add_common_args(dpo)
    _add_preference_args(dpo)
    dpo.add_argument("--dpo-beta", required=True, type=_positive_float)
    dpo.add_argument(
        "--reference-model-path",
        required=True,
        help=f"frozen reference checkpoint path, or {REFERENCE_BASE_MODEL!r} for the base model",
    )

    kto = subparsers.add_parser(METHOD_KTO, help="KTO from prompt/response/desirable rows")
    _add_common_args(kto)
    _add_preference_args(kto)
    kto.add_argument("--kto-beta", required=True, type=_positive_float)
    kto.add_argument(
        "--desirable-weight",
        required=True,
        type=_positive_float,
        help="loss weight on desirable examples; pick with --undesirable-weight so that "
        "desirable_weight*n_desirable : undesirable_weight*n_undesirable is between 1:1 and 4:3",
    )
    kto.add_argument("--undesirable-weight", required=True, type=_positive_float)
    kto.add_argument(
        "--reference-model-path",
        required=True,
        help=f"frozen reference checkpoint path, or {REFERENCE_BASE_MODEL!r} for the base model",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> RLConfig:
    args = build_parser().parse_args(argv)
    if args.method == METHOD_GRPO:
        return GRPOConfig(
            **_common_kwargs(args),
            grpo_loss_fn=args.grpo_loss_fn,
            grpo_clip_low_threshold=args.grpo_clip_low_threshold,
            grpo_clip_high_threshold=args.grpo_clip_high_threshold,
        )
    if args.method == METHOD_DPO:
        return DPOConfig(
            **_common_kwargs(args),
            **_preference_kwargs(args),
            dpo_beta=args.dpo_beta,
            reference_model_path=args.reference_model_path,
        )
    if args.method == METHOD_KTO:
        return KTOConfig(
            **_common_kwargs(args),
            **_preference_kwargs(args),
            kto_beta=args.kto_beta,
            desirable_weight=args.desirable_weight,
            undesirable_weight=args.undesirable_weight,
            reference_model_path=args.reference_model_path,
        )
    raise ValueError(f"unknown method {args.method!r}; expected one of {METHODS}")
