"""Run one Tinker reinforcement-learning optimization phase: GRPO, DPO, or KTO.

Supports fresh starts, initialization from a state checkpoint's weights with a new optimizer, and
optimizer-state resumes.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.keys import load_project_env
from src.training.arrays import TrainingArrays, values_to_list, weights_to_list
from src.training.artifacts import (
    CHECKPOINTS_LOG,
    RESUME_ATTEMPTS_LOG,
    RESUME_META,
    TRAIN_LOG,
    TRAIN_STEPS_LOG,
)
from src.training.losses import _tensor, compute_dpo_loss, compute_kto_loss, require_step_loss
from src.training.preference_data import (
    build_dpo_training_pairs,
    build_kto_training_examples,
    dpo_pair_to_datums,
    kto_example_to_datum,
    load_dpo_pairs_jsonl,
    load_kto_examples_jsonl,
    mismatched_training_arrays,
)
from src.training.rl_cli import (
    GRPO_LOSS_FN_CISPO,
    GRPO_LOSS_FN_PPO,
    REFERENCE_BASE_MODEL,
    DPOConfig,
    GRPOConfig,
    KTOConfig,
    RLConfig,
    parse_args,
    validate_config,
)
from src.training.rl_data import (
    RLTrainingArrays,
    build_grpo_training_arrays,
    load_grpo_rollouts_jsonl,
    rl_arrays_to_tinker_datum,
)
from src.training.sft_data import SFTDataConfig
from src.training.tinker_common import (
    CHECKPOINT_MODE_RESUME,
    _tinker_path,
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
    load_renderer,
    load_tokenizer,
    log_wandb,
    owner_tag,
    require_tinker_key,
    validate_json_metric,
    verify_reference_client,
    write_config,
)
from src.training.tinker_common import validate_resume_config as _validate_resume_config

COMMON_RESUME_MATCH_FIELDS = (
    "method",
    "model",
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
    "split",
)
DPO_RESUME_MATCH_FIELDS = COMMON_RESUME_MATCH_FIELDS + (
    "renderer_name",
    "max_seq_length",
    "mask_prompt",
    "thinking",
    "dpo_beta",
    "reference_model_path",
)
GRPO_RESUME_MATCH_FIELDS = COMMON_RESUME_MATCH_FIELDS + (
    "grpo_loss_fn",
    "grpo_clip_low_threshold",
    "grpo_clip_high_threshold",
)
KTO_RESUME_MATCH_FIELDS = COMMON_RESUME_MATCH_FIELDS + (
    "renderer_name",
    "max_seq_length",
    "mask_prompt",
    "thinking",
    "kto_beta",
    "desirable_weight",
    "undesirable_weight",
    "reference_model_path",
)


TrainPayloadData = list[Any]


@dataclass(frozen=True)
class RLPayload:
    """Prepared training rows plus counts and an array hash for one method."""

    method: str
    train_data: TrainPayloadData
    n_examples: int
    arrays_sha256: str
    method_counts: dict[str, int]


def _training_arrays_record(arrays: TrainingArrays) -> dict[str, Any]:
    return {
        "input_tokens": arrays.input_tokens,
        "target_tokens": arrays.target_tokens,
        "weights": arrays.weights,
        "rendered_text": arrays.rendered_text,
    }


def _rl_arrays_record(arrays: RLTrainingArrays) -> dict[str, Any]:
    return {
        "input_tokens": arrays.input_tokens,
        "target_tokens": arrays.target_tokens,
        "logprobs": arrays.logprobs,
        "advantages": arrays.advantages,
        "prompt_tokens": arrays.prompt_tokens,
        "completion_tokens": arrays.completion_tokens,
    }


def _json_records_sha256(records: list[dict[str, Any]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _preference_data_config(config: DPOConfig | KTOConfig) -> SFTDataConfig:
    return SFTDataConfig(
        max_seq_length=config.max_seq_length,
        mask_prompt=config.mask_prompt,
        thinking=config.thinking,
    )


def _load_preference_renderer(config: DPOConfig | KTOConfig):
    tokenizer = load_tokenizer(config.model)
    return load_renderer(config.model, config.renderer_name, tokenizer)


def _build_grpo_payload(config: GRPOConfig) -> RLPayload:
    rollouts = load_grpo_rollouts_jsonl(config.data, split=config.split)
    arrays = build_grpo_training_arrays(rollouts)
    train_data = [rl_arrays_to_tinker_datum(row) for row in arrays]
    counts = {
        "rollouts": len(rollouts),
        "groups": len({rollout.group_id for rollout in rollouts}),
        "datums": len(train_data),
    }
    return RLPayload(
        method=config.method,
        train_data=train_data,
        n_examples=len(rollouts),
        arrays_sha256=_json_records_sha256([_rl_arrays_record(row) for row in arrays]),
        method_counts=counts,
    )


def _build_dpo_payload(config: DPOConfig) -> RLPayload:
    renderer = _load_preference_renderer(config)
    pairs = load_dpo_pairs_jsonl(config.data, split=config.split)
    training_pairs = build_dpo_training_pairs(pairs, renderer, _preference_data_config(config))
    train_data = [dpo_pair_to_datums(pair) for pair in training_pairs]
    records = []
    for pair in training_pairs:
        records.append(
            {
                "chosen": _training_arrays_record(pair.chosen),
                "rejected": _training_arrays_record(pair.rejected),
            }
        )
    return RLPayload(
        method=config.method,
        train_data=train_data,
        n_examples=len(pairs),
        arrays_sha256=_json_records_sha256(records),
        method_counts={"pairs": len(training_pairs), "datums": 2 * len(training_pairs)},
    )


def _build_kto_payload(config: KTOConfig) -> RLPayload:
    renderer = _load_preference_renderer(config)
    examples = load_kto_examples_jsonl(config.data, split=config.split)
    training_examples = build_kto_training_examples(examples, renderer, _preference_data_config(config))
    train_data = [
        (kto_example_to_datum(example), example.desirable, example.n_prompt_tokens) for example in training_examples
    ]
    records = [
        {
            "arrays": _training_arrays_record(example.arrays),
            "desirable": example.desirable,
        }
        for example in training_examples
    ]
    n_desirable = sum(1 for example in training_examples if example.desirable)
    return RLPayload(
        method=config.method,
        train_data=train_data,
        n_examples=len(examples),
        arrays_sha256=_json_records_sha256(records),
        method_counts={
            "examples": len(training_examples),
            "desirable": n_desirable,
            "undesirable": len(training_examples) - n_desirable,
        },
    )


def build_payload(config: RLConfig) -> RLPayload:
    """Load and convert the method-specific data without touching Tinker."""
    if isinstance(config, GRPOConfig):
        return _build_grpo_payload(config)
    if isinstance(config, DPOConfig):
        return _build_dpo_payload(config)
    if isinstance(config, KTOConfig):
        return _build_kto_payload(config)
    raise TypeError(f"unknown RL config type {type(config).__name__}")


def build_config_record(config: RLConfig, payload: RLPayload, data_sha256: str) -> dict[str, Any]:
    record = asdict(config)
    record["git_commit"] = get_git_commit()
    record["n_examples"] = payload.n_examples
    record["method_counts"] = payload.method_counts
    record["data_sha256"] = data_sha256
    record["arrays_sha256"] = payload.arrays_sha256
    return record


def _resume_match_fields(config: RLConfig) -> tuple[str, ...]:
    if isinstance(config, GRPOConfig):
        return GRPO_RESUME_MATCH_FIELDS
    if isinstance(config, DPOConfig):
        return DPO_RESUME_MATCH_FIELDS
    if isinstance(config, KTOConfig):
        return KTO_RESUME_MATCH_FIELDS
    raise TypeError(f"unknown RL config type {type(config).__name__}")


def validate_resume_config(config: RLConfig, output_dir: Path, current_record: dict[str, Any]) -> None:
    """Validate a resume run against this run's recorded config fields (per-method field set)."""
    _validate_resume_config(config, output_dir, current_record, _resume_match_fields(config))


def _target_tokens(datum) -> list[int]:
    return [int(token) for token in values_to_list(datum.loss_fn_inputs["target_tokens"])]


def _datum_weights(datum):
    weights = weights_to_list(datum.loss_fn_inputs["weights"], context="weights")
    return _tensor(weights)


def _full_sequence_input(datum):
    last_target = _target_tokens(datum)[-1]
    return datum.model_input.append_int(last_target)


async def compute_reference_logprobs(reference_client, datums: list) -> list:
    """Compute reference logprobs aligned with each datum's target tokens."""
    futures = [reference_client.compute_logprobs_async(_full_sequence_input(datum)) for datum in datums]
    ref_logprobs = []
    for datum, future in zip(datums, futures, strict=True):
        values = await future
        aligned = values[1:]
        expected = len(_target_tokens(datum))
        if len(aligned) != expected:
            raise ValueError(f"reference logprobs length mismatch: got {len(aligned)}, expected {expected}")
        if any(value is None for value in aligned):
            raise ValueError("reference logprobs include None; cannot compute preference loss")
        ref_logprobs.append(_tensor([float(value) for value in aligned]))
    return ref_logprobs


def _jsonable_metrics(metrics: Any) -> dict[str, Any]:
    if metrics is None:
        raise ValueError("Tinker forward/backward response did not include metrics")
    result = {}
    for key, value in dict(metrics).items():
        if hasattr(value, "item"):
            value = value.item()
        if value is not None:
            validate_json_metric(value, context=f"metric {key!r}")
        result[key] = value
    return result


def _batch_completion_tokens_grpo(datums: list) -> int:
    total = 0
    for datum in datums:
        advantages = values_to_list(datum.loss_fn_inputs["advantages"])
        total += sum(1 for value in advantages if float(value) != 0.0)
    return total


async def _run_grpo_step(config: GRPOConfig, training_client, batch: list):
    import tinker

    lr = _run_grpo_step.lr
    adam = tinker.AdamParams(
        learning_rate=lr,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.eps,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
    )
    loss_fn_config = None
    if config.grpo_loss_fn in (GRPO_LOSS_FN_PPO, GRPO_LOSS_FN_CISPO):
        loss_fn_config = {
            "clip_low_threshold": config.grpo_clip_low_threshold,
            "clip_high_threshold": config.grpo_clip_high_threshold,
        }
    fb_future = await training_client.forward_backward_async(
        batch, loss_fn=config.grpo_loss_fn, loss_fn_config=loss_fn_config
    )
    optim_future = await training_client.optim_step_async(adam)
    fb_result = await fb_future.result_async()
    await optim_future.result_async()
    advantages = [
        abs(float(value))
        for datum in batch
        for value in values_to_list(datum.loss_fn_inputs["advantages"])
        if float(value) != 0.0
    ]
    metrics = _jsonable_metrics(getattr(fb_result, "metrics", None))
    # Tinker reports the forward-backward loss under an aggregation-suffixed key ("loss:sum");
    # older versions used a bare "loss". Accept whichever the server sent and fail loudly if it
    # sent neither, rather than reading a differently-aggregated metric by accident.
    loss_key = next((key for key in ("loss", "loss:sum") if key in metrics), "loss")
    loss = require_step_loss(metrics, loss_key)
    metrics.update(
        {
            "loss": loss,
            "mean_abs_advantage": sum(advantages) / len(advantages) if advantages else 0.0,
            "completion_tokens": _batch_completion_tokens_grpo(batch),
        }
    )
    metrics.update(_importance_ratio_metrics(fb_result, batch, config))
    return metrics


def _importance_ratio_metrics(fb_result, batch: list, config: GRPOConfig) -> dict[str, float]:
    """How far the learner has drifted from the sampler that produced this batch.

    A round generates once and then takes several optimizer steps on those rollouts, so every
    step after the first trains on off-policy data. Nothing in Tinker's returned metrics measures
    that drift: `e_max_violation`, `e_min_violation`, `e_frac_oversubscribed` and
    `e_frac_with_tokens` are mixture-of-experts ROUTING diagnostics (`e` is expert, not epsilon),
    which is why they read ~0.36 even at the first step of a round, where the learner is the
    sampler and every ratio is exactly 1.0.

    The ratio itself is free to compute: `forward_backward` already returns the learner's
    per-token logprobs in `loss_fn_outputs[i]["logprobs"]`, and the sampling logprobs are the
    ones we put into the datum. So this is pure bookkeeping on values already in hand -- no extra
    forward pass, no custom loss.

    Reported over completion tokens only (prompt positions carry advantage 0). `frac_r_gt_clip`
    is the share of tokens the clipped losses actually bind on; if it is ~0 the run is
    effectively on-policy and the choice among ppo/cispo/importance_sampling is not doing work.
    `kl_k3` is the standard low-variance estimator (r - 1) - log r, non-negative, in nats/token.

    Returns an empty dict for loss functions that do not return logprobs, rather than failing a
    long training run over a diagnostic; the absence is visible as missing keys in the step log.
    """
    import math

    outputs = getattr(fb_result, "loss_fn_outputs", None)
    if not outputs or len(outputs) != len(batch):
        return {}

    diffs: list[float] = []
    for datum, output in zip(batch, outputs, strict=True):
        if "logprobs" not in output:
            return {}
        target = values_to_list(output["logprobs"])
        sampling = values_to_list(datum.loss_fn_inputs["logprobs"])
        advantage = values_to_list(datum.loss_fn_inputs["advantages"])
        if not (len(target) == len(sampling) == len(advantage)):
            return {}
        diffs.extend(float(t) - float(s) for t, s, a in zip(target, sampling, advantage) if float(a) != 0.0)

    if not diffs:
        return {}
    ratios = [math.exp(d) for d in diffs]
    n = len(ratios)
    ordered = sorted(ratios)
    clip_high = config.grpo_clip_high_threshold
    return {
        "ratio_n_tokens": float(n),
        "ratio_mean": sum(ratios) / n,
        "ratio_median": ordered[n // 2],
        "logdiff_mean": sum(diffs) / n,
        "abs_logdiff_mean": sum(abs(d) for d in diffs) / n,
        "frac_r_gt_clip": sum(1 for r in ratios if r > clip_high) / n if clip_high else 0.0,
        "frac_r_lt_0p5": sum(1 for r in ratios if r < 0.5) / n,
        "frac_r_lt_0p1": sum(1 for r in ratios if r < 0.1) / n,
        "frac_r_lt_0p01": sum(1 for r in ratios if r < 0.01) / n,
        "kl_k3": sum((r - 1.0) - d for r, d in zip(ratios, diffs)) / n,
    }


async def _run_dpo_step(config: DPOConfig, training_client, reference_client, batch: list[tuple]):
    import tinker

    data = []
    for chosen, rejected in batch:
        data.extend([chosen, rejected])

    ref_logprobs = await compute_reference_logprobs(reference_client, data)
    chosen_ref = [ref_logprobs[i] for i in range(0, len(data), 2)]
    rejected_ref = [ref_logprobs[i] for i in range(1, len(data), 2)]
    chosen_data = [data[i] for i in range(0, len(data), 2)]
    rejected_data = [data[i] for i in range(1, len(data), 2)]
    chosen_weights = [_datum_weights(datum) for datum in chosen_data]
    rejected_weights = [_datum_weights(datum) for datum in rejected_data]

    def dpo_loss_fn(data_: list, logprobs_list: list):
        chosen_logprobs = [logprobs_list[i] for i in range(0, len(data_), 2)]
        rejected_logprobs = [logprobs_list[i] for i in range(1, len(data_), 2)]
        return compute_dpo_loss(
            chosen_logprobs,
            rejected_logprobs,
            chosen_ref,
            rejected_ref,
            chosen_weights,
            rejected_weights,
            config.dpo_beta,
        )

    adam = tinker.AdamParams(
        learning_rate=_run_dpo_step.lr,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.eps,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
    )
    fb_future = await training_client.forward_backward_custom_async(data, dpo_loss_fn)
    optim_future = await training_client.optim_step_async(adam)
    fb_result = await fb_future.result_async()
    await optim_future.result_async()
    metrics = _jsonable_metrics(getattr(fb_result, "metrics", None))
    metrics["loss"] = require_step_loss(metrics, "dpo_loss")
    return metrics


def _datum_to_arrays(datum) -> TrainingArrays:
    return TrainingArrays(
        input_tokens=[int(token) for token in datum.model_input.to_ints()],
        target_tokens=_target_tokens(datum),
        weights=weights_to_list(datum.loss_fn_inputs["weights"], context="weights"),
        rendered_text="",
    )


def kto_kl_datums(data: list, n_prompt_tokens: list[int]) -> list:
    """Mismatched (prompt_i, completion_{i-1}) datums for the KTO KL baseline, one per example.

    Follows the reference implementation: each prompt is paired with the previous example's
    completion, wrapping around, so every KL sample is a completion the policy did not write for
    that prompt. `n_prompt_tokens[i]` is where example i's assistant completion begins. Requires
    at least two examples.
    """
    from src.training.arrays import arrays_to_tinker_datum

    if len(data) < 2:
        raise ValueError("KTO requires at least two examples per optimizer step for the mismatched KL baseline")
    if len(n_prompt_tokens) != len(data):
        raise ValueError("n_prompt_tokens must align with data")
    arrays = [_datum_to_arrays(datum) for datum in data]
    return [
        arrays_to_tinker_datum(
            mismatched_training_arrays(arrays[i], n_prompt_tokens[i], arrays[i - 1], n_prompt_tokens[i - 1])
        )
        for i in range(len(arrays))
    ]


async def _run_kto_step(config: KTOConfig, training_client, reference_client, batch: list[tuple]):
    """One KTO optimizer step.

    The forward pass runs on the batch datums followed by their mismatched KL datums, so the KL
    baseline uses the CURRENT policy's log-probabilities (from the same forward pass) rather than
    a stale sampler snapshot. The KL datums contribute no gradient: their log-ratios enter the
    loss only as a detached scalar, plus a zero tether so Tinker sees a gradient for every datum.
    """
    import tinker
    import torch

    data = [datum for datum, _desirable, _n_prompt in batch]
    desirable_flags = [desirable for _datum, desirable, _n_prompt in batch]
    n_prompt_tokens = [n_prompt for _datum, _desirable, n_prompt in batch]
    kl_data = kto_kl_datums(data, n_prompt_tokens)
    forward_data = data + kl_data
    ref_logprobs_all = await compute_reference_logprobs(reference_client, forward_data)
    ref_logprobs = ref_logprobs_all[: len(data)]
    kl_ref_logprobs = ref_logprobs_all[len(data) :]
    weights = [_datum_weights(datum) for datum in data]
    kl_weights = [_datum_weights(datum) for datum in kl_data]

    def kto_loss_fn(data_: list, logprobs_list: list):
        if len(data_) != len(forward_data) or len(logprobs_list) != len(forward_data):
            raise ValueError("KTO custom loss received a different batch length than expected")
        policy_logprobs = logprobs_list[: len(data)]
        kl_policy_logprobs = logprobs_list[len(data) :]
        kl_logratios = [
            (torch.dot(_tensor(policy_lp), weight) - torch.dot(reference_lp, weight)).detach()
            for policy_lp, reference_lp, weight in zip(kl_policy_logprobs, kl_ref_logprobs, kl_weights, strict=True)
        ]
        loss, metrics = compute_kto_loss(
            policy_logprobs,
            ref_logprobs,
            weights,
            desirable_flags,
            config.kto_beta,
            config.desirable_weight,
            config.undesirable_weight,
            kl_logratios,
        )
        # Tinker refuses a batch in which any logprob tensor received no gradient. The KL datums
        # are meant to carry none, so tether each one with an exactly-zero contribution: the
        # gradient it sends back is a zero vector, and the loss value is unchanged.
        zero_tether = sum((0.0 * _tensor(lp)).sum() for lp in kl_policy_logprobs)
        return loss + zero_tether, metrics

    adam = tinker.AdamParams(
        learning_rate=_run_kto_step.lr,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.eps,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
    )
    fb_future = await training_client.forward_backward_custom_async(forward_data, kto_loss_fn)
    optim_future = await training_client.optim_step_async(adam)
    fb_result = await fb_future.result_async()
    await optim_future.result_async()
    metrics = _jsonable_metrics(getattr(fb_result, "metrics", None))
    metrics["loss"] = require_step_loss(metrics, "kto_loss")
    return metrics


async def create_reference_client(config: DPOConfig | KTOConfig):
    """Create the frozen reference sampling client for a preference phase."""
    import tinker

    service_client = tinker.ServiceClient(user_metadata=owner_tag())
    if config.reference_model_path == REFERENCE_BASE_MODEL:
        return await service_client.create_sampling_client_async(base_model=config.model)
    return await service_client.create_sampling_client_async(model_path=config.reference_model_path)


async def train_async(config: RLConfig, payload: RLPayload, data_sha256: str, wandb_run=None) -> None:
    """Run one Tinker RL job."""
    training_client = await create_training_client(config)
    reference_client = None
    if isinstance(config, DPOConfig | KTOConfig):
        reference_client = await create_reference_client(config)
        await verify_reference_client(training_client, reference_client)

    output_dir = Path(config.output)
    log_path = output_dir / TRAIN_LOG
    raw_log_path = output_dir / TRAIN_STEPS_LOG
    checkpoints_path = output_dir / CHECKPOINTS_LOG

    train_data = list(payload.train_data)
    rng = random.Random(config.seed)
    rng.shuffle(train_data)
    data_idx = fast_forward_data(train_data, config, rng)

    print(
        f"\n{config.method.upper()} training from step {config.start_step} to {config.max_steps}, "
        f"batch {config.batch_size}, LR {config.learning_rate} "
        f"({config.lr_schedule}, {config.warmup_steps} warmup), on {len(train_data)} rows."
    )

    start_time = time.time()
    log_mode = "a" if config.start_step else "w"
    checkpoint_mode = "a" if config.start_step else "w"
    window_loss = 0.0
    window_losses = 0
    lr = config.learning_rate
    final_metrics: dict[str, Any] = {}

    with (
        open(log_path, log_mode) as log_file,
        open(raw_log_path, log_mode) as raw_log_file,
        open(checkpoints_path, checkpoint_mode) as ckpt_file,
    ):
        pbar = tqdm(
            total=config.max_steps, initial=config.start_step, desc=f"{config.method.upper()} training", unit="step"
        )
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
            _run_grpo_step.lr = lr
            _run_dpo_step.lr = lr
            _run_kto_step.lr = lr

            t0 = time.time()
            if isinstance(config, GRPOConfig):
                metrics = await _run_grpo_step(config, training_client, batch)
            elif isinstance(config, DPOConfig):
                metrics = await _run_dpo_step(config, training_client, reference_client, batch)
            elif isinstance(config, KTOConfig):
                metrics = await _run_kto_step(config, training_client, reference_client, batch)
            else:
                raise TypeError(f"unknown RL config type {type(config).__name__}")

            done = step + 1
            final_metrics = metrics
            loss = require_step_loss(metrics, "loss")
            window_loss += loss
            window_losses += 1
            step_time = time.time() - t0
            step_record = {
                "step": done,
                "lr": lr,
                "batch_start_index": batch_start_idx,
                "next_training_row_index": data_idx,
                "step_time_sec": step_time,
                "metrics": metrics,
            }
            raw_log_file.write(json_dumps_strict(step_record) + "\n")
            raw_log_file.flush()
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}")

            if done % config.log_every == 0 or done == config.max_steps:
                elapsed = time.time() - start_time
                log_record = {
                    "step": done,
                    "mean_loss": window_loss / window_losses if window_losses else None,
                    "window_steps": config.log_every if done % config.log_every == 0 else (done % config.log_every),
                    "lr": lr,
                    "steps_per_sec": (done - config.start_step) / elapsed if elapsed else 0.0,
                    "elapsed_sec": elapsed,
                    "last_metrics": metrics,
                }
                log_file.write(json_dumps_strict(log_record) + "\n")
                log_file.flush()
                log_wandb(
                    wandb_run,
                    {
                        "train/mean_loss": log_record["mean_loss"],
                        "train/lr": log_record["lr"],
                        "train/steps_per_sec": log_record["steps_per_sec"],
                        "train/elapsed_sec": log_record["elapsed_sec"],
                        **{f"train/{key}": value for key, value in metrics.items() if isinstance(value, int | float)},
                    },
                    step=done,
                )
                window_loss = 0.0
                window_losses = 0

            if done % config.save_every == 0 and done < config.max_steps:
                path = await _tinker_path(
                    training_client.save_weights_for_sampler_async(name=f"{config.method}_step_{done}")
                )
                ckpt_record = {"step": done, "type": "sampler", "path": path, "lr": lr, "metrics": metrics}
                ckpt_file.write(json_dumps_strict(ckpt_record) + "\n")
                ckpt_file.flush()
                log_wandb(
                    wandb_run,
                    {"checkpoint/saved": 1, "checkpoint/type": "sampler", "checkpoint/lr": lr},
                    step=done,
                )
        pbar.close()

        sampler_path = await _tinker_path(training_client.save_weights_for_sampler_async(name=f"{config.method}_final"))
        state_path = await _tinker_path(training_client.save_state_async(name=f"{config.method}_state_final"))
        ckpt_file.write(
            json_dumps_strict(
                {
                    "step": config.max_steps,
                    "type": "sampler",
                    "path": sampler_path,
                    "lr": lr,
                    "metrics": final_metrics,
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
                    "lr": lr,
                    "metrics": final_metrics,
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
                "arrays_sha256": payload.arrays_sha256,
                "seed": config.seed,
                "batch_size": config.batch_size,
                "method": config.method,
            },
            indent=2,
        )
    )
    if wandb_run is not None:
        wandb_run.summary["final_sampler_checkpoint"] = sampler_path
        wandb_run.summary["final_state_checkpoint"] = state_path
        wandb_run.summary["final_step"] = config.max_steps
        wandb_run.summary["data_sha256"] = data_sha256
        wandb_run.summary["arrays_sha256"] = payload.arrays_sha256
        log_wandb(
            wandb_run,
            {"checkpoint/saved": 1, "checkpoint/type": "final", "checkpoint/lr": lr},
            step=config.max_steps,
        )
    print(f"\n{config.method.upper()} training complete in {time.time() - start_time:.1f}s.")
    print(f"Final sampler checkpoint: {sampler_path}")
    print(f"Final full state checkpoint: {state_path}")


def _row_token_count(row) -> int:
    """Token count for one training row (a datum, or a tuple of datums for DPO/KTO).

    Uses each datum's model_input so it works across GRPO/DPO/KTO regardless of which
    loss_fn_inputs they carry; used only to estimate cost before a run.
    """
    items = row if isinstance(row, tuple) else (row,)
    total = 0
    for item in items:
        model_input = getattr(item, "model_input", None)
        if model_input is not None and hasattr(model_input, "to_ints"):
            total += len(model_input.to_ints())
    return total


def run(config: RLConfig) -> None:
    validate_config(config)
    output_dir = Path(config.output)
    data_sha256 = file_sha256(config.data)
    payload = build_payload(config)
    record = build_config_record(config, payload, data_sha256)
    validate_resume_config(config, output_dir, record)
    if config.checkpoint_mode == CHECKPOINT_MODE_RESUME:
        output_dir.mkdir(parents=True, exist_ok=True)
        resume_attempt_path = output_dir / RESUME_ATTEMPTS_LOG
        with resume_attempt_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    else:
        write_config(record, output_dir)

    row_uses = config.max_steps * config.batch_size
    print(
        f"\nRun plan: {config.method.upper()} {config.max_steps} steps x batch {config.batch_size} = "
        f"{row_uses} row uses over {len(payload.train_data)} training rows "
        f"(~{row_uses / len(payload.train_data):.1f} epochs), LoRA rank {config.lora_rank} on {config.model}."
    )
    print(f"Data counts: {payload.method_counts}")
    token_lengths = [_row_token_count(row) for row in payload.train_data]
    print(format_cost_estimate(config.model, estimate_train_tokens(token_lengths, config.batch_size, config.max_steps)))
    if config.dry_run:
        print("\nDry run: data built and config written; skipping Tinker calls.")
        return

    require_tinker_key()
    wandb_run = init_wandb(config, record, output_dir, job_type="tinker_rl", method_tags=(config.method,))
    try:
        asyncio.run(train_async(config, payload, data_sha256, wandb_run))
    finally:
        finish_wandb(wandb_run)


def main() -> None:
    load_project_env()
    run(parse_args())


if __name__ == "__main__":
    main()
