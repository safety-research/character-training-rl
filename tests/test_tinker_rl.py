"""Tests for stackable Tinker RL trainer helpers."""

import json

import pytest

# torch arrives transitively with the `training` extra, which the normal working environment
# has. Skipping rather than importing at module scope means a docs-only `--extra dev` checkout
# reports "N skipped" instead of failing collection and hiding every other test in the suite.
torch = pytest.importorskip("torch", reason="needs the training extra: uv sync --extra dev --extra training")

from src.training.tinker_rl import (
    DPOConfig,
    GRPOConfig,
    KTOConfig,
    _jsonable_metrics,
    compute_dpo_loss,
    compute_kto_loss,
    parse_args,
    validate_config,
    validate_resume_config,
)

COMMON = {
    "model": "Qwen/Qwen3-8B",
    "data": "train.jsonl",
    "output": "out",
    "split": "train",
    "lora_rank": 32,
    "learning_rate": 1e-5,
    "batch_size": 4,
    "max_steps": 10,
    "warmup_steps": 0,
    "lr_schedule": "linear",
    "save_every": 5,
    "log_every": 5,
    "checkpoint_mode": "fresh",
    "checkpoint_path": "none",
    "start_step": 0,
    "seed": 1,
    "dry_run": True,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
    "train_mlp": True,
    "train_attn": True,
    "train_unembed": False,
    "wandb_mode": "disabled",
    "wandb_project": "",
    "wandb_entity": "",
    "wandb_run_name": "",
    "wandb_group": "",
    "wandb_tags": (),
    "wandb_run_id": "",
    "wandb_resume": "none",
}


def grpo_config(**overrides):
    values = {
        **COMMON,
        "method": "grpo",
        "grpo_loss_fn": "importance_sampling",
        "grpo_clip_low_threshold": 0.0,
        "grpo_clip_high_threshold": 0.0,
    }
    values.update(overrides)
    return GRPOConfig(**values)


def dpo_config(**overrides):
    values = {
        **COMMON,
        "method": "dpo",
        "renderer_name": "qwen3",
        "max_seq_length": 1024,
        "mask_prompt": True,
        "thinking": "keep",
        "dpo_beta": 0.1,
        "reference_model_path": "base_model",
    }
    values.update(overrides)
    return DPOConfig(**values)


def kto_config(**overrides):
    values = {
        **COMMON,
        "method": "kto",
        "renderer_name": "qwen3",
        "max_seq_length": 1024,
        "mask_prompt": True,
        "thinking": "keep",
        "kto_beta": 0.1,
        "desirable_weight": 1.0,
        "undesirable_weight": 1.0,
        "reference_model_path": "base_model",
    }
    values.update(overrides)
    return KTOConfig(**values)


def test_validate_grpo_requires_zero_clip_for_importance_sampling():
    with pytest.raises(ValueError, match="importance_sampling"):
        validate_config(grpo_config(grpo_clip_low_threshold=0.8))


def test_validate_grpo_allows_cispo_one_sided_clip():
    validate_config(grpo_config(grpo_loss_fn="cispo", grpo_clip_high_threshold=4.0))


def test_validate_dpo_rejects_base_reference_when_initializing_from_checkpoint():
    config = dpo_config(checkpoint_mode="init", checkpoint_path="tinker://phase/sampler")
    with pytest.raises(ValueError, match="base_model"):
        validate_config(config)


def test_validate_kto_requires_batch_size_for_cyclic_kl():
    with pytest.raises(ValueError, match="batch_size"):
        validate_config(kto_config(batch_size=1))


def test_validate_resume_config_preserves_original_start_record(tmp_path):
    config = dpo_config(
        checkpoint_mode="resume",
        checkpoint_path="tinker://phase/state",
        start_step=5,
        reference_model_path="tinker://phase/reference",
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    record = {
        "method": "dpo",
        "model": config.model,
        "git_commit": "abc",
        "data_sha256": "data",
        "arrays_sha256": "arrays",
        "lora_rank": config.lora_rank,
        "batch_size": config.batch_size,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "lr_schedule": config.lr_schedule,
        "save_every": config.save_every,
        "log_every": config.log_every,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "eps": config.eps,
        "weight_decay": config.weight_decay,
        "grad_clip_norm": config.grad_clip_norm,
        "train_mlp": config.train_mlp,
        "train_attn": config.train_attn,
        "train_unembed": config.train_unembed,
        "seed": config.seed,
        "split": config.split,
        "renderer_name": config.renderer_name,
        "max_seq_length": config.max_seq_length,
        "mask_prompt": config.mask_prompt,
        "thinking": config.thinking,
        "dpo_beta": config.dpo_beta,
        "reference_model_path": config.reference_model_path,
    }
    (output_dir / "config.json").write_text(json.dumps(record))
    (output_dir / "resume_meta.json").write_text(
        json.dumps({"step": config.start_step, "state_path": config.checkpoint_path})
    )

    validate_resume_config(config, output_dir, {**record, "checkpoint_path": config.checkpoint_path})


def test_compute_dpo_loss_prefers_chosen_log_ratio():
    loss, metrics = compute_dpo_loss(
        chosen_logprobs=[torch.tensor([-0.1])],
        rejected_logprobs=[torch.tensor([-1.0])],
        chosen_ref_logprobs=[torch.tensor([-0.5])],
        rejected_ref_logprobs=[torch.tensor([-0.5])],
        chosen_weights=[torch.tensor([1.0])],
        rejected_weights=[torch.tensor([1.0])],
        dpo_beta=0.1,
    )

    assert loss.item() > 0
    assert metrics["accuracy"] == 1.0
    assert metrics["loss"] == metrics["dpo_loss"]


def test_compute_kto_loss_requires_explicit_kl_samples():
    with pytest.raises(ValueError, match="explicit mismatched-pair KL"):
        compute_kto_loss(
            logprobs=[torch.tensor([-0.1])],
            ref_logprobs=[torch.tensor([-0.2])],
            weights=[torch.tensor([1.0])],
            desirable_flags=[True],
            kto_beta=0.1,
            desirable_weight=1.0,
            undesirable_weight=1.0,
        )


def test_compute_kto_loss_reports_absent_class_as_none():
    _loss, metrics = compute_kto_loss(
        logprobs=[torch.tensor([-0.1]), torch.tensor([-0.2])],
        ref_logprobs=[torch.tensor([-0.2]), torch.tensor([-0.2])],
        weights=[torch.tensor([1.0]), torch.tensor([1.0])],
        desirable_flags=[True, True],
        kto_beta=0.1,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        kl_logratios=[torch.tensor(0.0), torch.tensor(0.0)],
    )

    assert metrics["n_desirable"] == 2
    assert metrics["n_undesirable"] == 0
    assert metrics["undesirable_accuracy"] is None


def test_parse_args_builds_grpo_config_with_clip_fields():
    config = parse_args(
        [
            "grpo",
            "--model",
            "Qwen/Qwen3-8B",
            "--data",
            "train.jsonl",
            "--output",
            "out",
            "--split",
            "train",
            "--lora-rank",
            "32",
            "--learning-rate",
            "1e-5",
            "--batch-size",
            "4",
            "--max-steps",
            "10",
            "--warmup-steps",
            "0",
            "--lr-schedule",
            "linear",
            "--save-every",
            "5",
            "--log-every",
            "5",
            "--checkpoint-mode",
            "fresh",
            "--checkpoint-path",
            "none",
            "--start-step",
            "0",
            "--seed",
            "1",
            "--dry-run",
            "true",
            "--beta1",
            "0.9",
            "--beta2",
            "0.95",
            "--eps",
            "1e-8",
            "--weight-decay",
            "0",
            "--grad-clip-norm",
            "1",
            "--train-mlp",
            "true",
            "--train-attn",
            "true",
            "--train-unembed",
            "false",
            "--wandb-mode",
            "disabled",
            "--wandb-project",
            "none",
            "--wandb-entity",
            "none",
            "--wandb-run-name",
            "none",
            "--wandb-group",
            "none",
            "--wandb-tags",
            "none",
            "--wandb-run-id",
            "none",
            "--wandb-resume",
            "none",
            "--grpo-loss-fn",
            "ppo",
            "--grpo-clip-low-threshold",
            "0.8",
            "--grpo-clip-high-threshold",
            "1.2",
        ]
    )

    assert isinstance(config, GRPOConfig)
    assert config.grpo_loss_fn == "ppo"
    assert config.grpo_clip_low_threshold == 0.8


def test_jsonable_metrics_rejects_nonfinite_metric():
    with pytest.raises(ValueError, match="finite"):
        _jsonable_metrics({"loss": float("nan")})


def test_jsonable_metrics_keeps_finite_values_and_none():
    assert _jsonable_metrics({"loss": 0.5, "desirable_accuracy": None}) == {"loss": 0.5, "desirable_accuracy": None}


def _kto(logratios, flags, kl_samples, beta=0.1, wd=1.0, wu=1.0):
    """Run compute_kto_loss on one-token examples whose log-ratio equals `logratios[i]`."""
    return compute_kto_loss(
        logprobs=[torch.tensor([lr]) for lr in logratios],
        ref_logprobs=[torch.tensor([0.0]) for _ in logratios],
        weights=[torch.tensor([1.0]) for _ in logratios],
        desirable_flags=flags,
        kto_beta=beta,
        desirable_weight=wd,
        undesirable_weight=wu,
        kl_logratios=[torch.tensor(v) for v in kl_samples],
    )


def test_compute_kto_loss_matches_reference_formula():
    # z0 = clamp(mean(kl samples), 0) = 0.5 (unscaled log-ratios); beta scales inside the sigmoid.
    beta = 0.1
    loss, metrics = _kto([2.0, -1.0], [True, False], [0.2, 0.8], beta=beta)
    z0 = 0.5
    expected = torch.stack(
        [
            1.0 - torch.sigmoid(torch.tensor(beta * (2.0 - z0))),
            1.0 - torch.sigmoid(torch.tensor(beta * (z0 - (-1.0)))),
        ]
    ).mean()
    assert abs(loss.item() - expected.item()) < 1e-6
    assert abs(metrics["kl"] - z0) < 1e-6
    assert abs(metrics["reward_desirable"] - beta * 2.0) < 1e-6
    assert abs(metrics["reward_undesirable"] - beta * -1.0) < 1e-6
    assert abs(metrics["reward_margin"] - beta * 3.0) < 1e-6
    assert metrics["desirable_accuracy"] == 1.0 and metrics["undesirable_accuracy"] == 1.0


def test_compute_kto_loss_clamps_negative_kl_estimate_at_zero():
    _loss, metrics = _kto([1.0, -1.0], [True, False], [-0.7, -0.3])
    assert metrics["kl"] == 0.0


def test_compute_kto_loss_is_a_mean_over_all_examples_so_class_weights_trade_off_with_counts():
    # three desirable + one undesirable, all with zero log-ratio and zero KL -> every term is 0.5*w
    _loss_eq, m_eq = _kto([0.0] * 4, [True, True, True, False], [0.0, 0.0])
    assert abs(m_eq["loss"] - 0.5) < 1e-6
    # TRL guidance: upweight the rarer class; with w_u = 3 the mean becomes (3*0.5 + 1.5)/4
    _loss_w, m_w = _kto([0.0] * 4, [True, True, True, False], [0.0, 0.0], wu=3.0)
    assert abs(m_w["loss"] - (3 * 0.5 + 3.0 * 0.5) / 4) < 1e-6


def test_compute_kto_loss_allows_single_class_batches():
    loss, metrics = _kto([0.5, 0.2], [True, True], [0.0, 0.0])
    assert torch.isfinite(loss)
    assert metrics["n_undesirable"] == 0 and metrics["undesirable_accuracy"] is None
    assert metrics["reward_margin"] is None


def test_kto_kl_datums_pair_each_prompt_with_previous_completion():
    tinker_types = pytest.importorskip("tinker").types
    from src.training.arrays import TrainingArrays, arrays_to_tinker_datum
    from src.training.tinker_rl import kto_kl_datums

    def datum(prompt, completion):
        seq = [ord(c) for c in prompt + completion]
        weights = [0.0] * (len(prompt) - 1) + [1.0] * len(completion)
        return arrays_to_tinker_datum(TrainingArrays(seq[:-1], seq[1:], weights, prompt + completion))

    data = [datum("A:", "aaa"), datum("B:", "bb"), datum("C:", "c")]
    kl = kto_kl_datums(data, [2, 2, 2])
    assert len(kl) == 3

    def text(d):
        toks = list(d.model_input.to_ints()) + [int(d.loss_fn_inputs["target_tokens"].data[-1])]
        return "".join(chr(t) for t in toks)

    assert [text(d) for d in kl] == ["A:c", "B:aaa", "C:bb"]
    assert isinstance(kl[0], tinker_types.Datum)
    with pytest.raises(ValueError, match="at least two"):
        kto_kl_datums(data[:1], [2])
    with pytest.raises(ValueError, match="align"):
        kto_kl_datums(data, [2, 2])


class _FakeFBResult:
    def __init__(self, metrics):
        self.metrics = metrics

    async def result_async(self):
        return self


class _FakeOptimResult:
    async def result_async(self):
        return None


class _FakeTrainingClient:
    """Runs the custom loss client-side with fixed per-token policy logprobs; records the batch it saw."""

    def __init__(self, logprob_per_token: float):
        self.logprob_per_token = logprob_per_token
        self.seen = None

    async def forward_backward_custom_async(self, data, loss_fn):
        self.seen = data
        logprobs = [
            torch.full((len(d.loss_fn_inputs["target_tokens"].data),), self.logprob_per_token, requires_grad=True)
            for d in data
        ]
        loss, metrics = loss_fn(data, logprobs)
        loss.backward()
        # Mirrors tinker's forward_backward_custom_async: every datum's logprob tensor must have a gradient.
        self.grads = []
        for lp in logprobs:
            if lp.grad is None:
                raise ValueError("No gradient computed for logprob tensor")
            self.grads.append(lp.grad.clone())
        return _FakeFBResult(metrics)

    async def optim_step_async(self, adam):
        self.adam = adam
        return _FakeOptimResult()


class _FakeReferenceClient:
    def __init__(self, logprob_per_token: float):
        self.logprob_per_token = logprob_per_token

    async def compute_logprobs_async(self, model_input):
        n = len(model_input.to_ints())
        return [None] + [self.logprob_per_token] * (n - 1)


def _kto_datum(prompt: str, completion: str):
    from src.training.arrays import TrainingArrays, arrays_to_tinker_datum

    seq = [ord(c) for c in prompt + completion]
    weights = [0.0] * (len(prompt) - 1) + [1.0] * len(completion)
    return arrays_to_tinker_datum(TrainingArrays(seq[:-1], seq[1:], weights, prompt + completion)), len(prompt)


def test_run_kto_step_uses_mismatched_kl_from_the_same_forward_pass():
    import asyncio

    from src.training.tinker_rl import _run_kto_step

    pytest.importorskip("tinker")
    config = kto_config(batch_size=2, kto_beta=0.1, desirable_weight=1.0, undesirable_weight=1.0)
    d1, n1 = _kto_datum("A:", "aaaa")  # desirable, 4 completion tokens
    d2, n2 = _kto_datum("B:", "bb")  # undesirable, 2 completion tokens
    policy, reference = -1.0, -1.5  # every token: policy is 0.5 nats more likely than reference
    training = _FakeTrainingClient(policy)
    _run_kto_step.lr = 1e-5
    metrics = asyncio.run(
        _run_kto_step(config, training, _FakeReferenceClient(reference), [(d1, True, n1), (d2, False, n2)])
    )

    # forward batch = 2 real datums + 2 mismatched KL datums (A:bb and B:aaaa)
    assert len(training.seen) == 4
    kl_texts = [
        "".join(chr(t) for t in list(d.model_input.to_ints()) + [int(d.loss_fn_inputs["target_tokens"].data[-1])])
        for d in training.seen[2:]
    ]
    assert kl_texts == ["A:bb", "B:aaaa"]
    # KL samples: unscaled log-ratio 0.5 per trained token -> (2*0.5 + 4*0.5)/2 = 1.5, clamped at 0 -> 1.5
    assert abs(metrics["kl"] - 1.5) < 1e-6
    # rewards: beta * 0.5 * n_completion_tokens
    assert abs(metrics["reward_desirable"] - 0.1 * 2.0) < 1e-6
    assert abs(metrics["reward_undesirable"] - 0.1 * 1.0) < 1e-6
    expected = (
        torch.stack(
            [
                1 - torch.sigmoid(torch.tensor(0.1 * (2.0 - 1.5))),
                1 - torch.sigmoid(torch.tensor(0.1 * (1.5 - 1.0))),
            ]
        )
        .mean()
        .item()
    )
    assert abs(metrics["loss"] - expected) < 1e-6
    assert training.adam.learning_rate == 1e-5
    # Every datum has a gradient (Tinker's requirement); the real datums' are non-zero on trained
    # tokens and the KL datums' are exactly zero everywhere.
    assert len(training.grads) == 4
    assert all(torch.count_nonzero(g) > 0 for g in training.grads[:2])
    assert all(torch.count_nonzero(g) == 0 for g in training.grads[2:])


def test_validate_continue_mode_needs_a_checkpoint_and_starts_at_step_zero():
    """`continue` restores weights and optimizer onto new data as a new job: no same-job checks,
    but it still needs a real state path and a zero start step."""
    from src.training.rl_cli import validate_config

    validate_config(grpo_config(checkpoint_mode="continue", checkpoint_path="tinker://round0/state"))
    with pytest.raises(ValueError):
        validate_config(grpo_config(checkpoint_mode="continue", checkpoint_path="none"))
    with pytest.raises(ValueError):
        validate_config(grpo_config(checkpoint_mode="continue", checkpoint_path="tinker://round0/state", start_step=1))
