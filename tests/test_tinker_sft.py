"""Tests for reusable Tinker SFT trainer helpers."""

import json
import math

import pytest

pytest.importorskip(
    "torch", reason="needs the training extra: uv sync --extra dev --extra training"
)

from src.training.losses import mean_weighted_nll
from src.training.sft_data import TrainingArrays
from src.training.tinker_common import DEFAULT_RENDERER_DATE, _tinker_path, init_wandb, verify_training_client
from src.training.tinker_sft import (
    TinkerSFTConfig,
    build_arrays,
    train_async,
    validate_config,
    validate_resume_config,
)


class FakeForwardBackwardResult:
    loss_fn_outputs = [{"logprobs": [-1.0, -2.0]}]


class FakeDatum:
    loss_fn_inputs = {"weights": [0.0, 0.0]}


class FakeModelData:
    def __init__(self, model_name="Qwen/Qwen3-8B"):
        self.model_name = model_name


class FakeInfo:
    def __init__(self, model_name="Qwen/Qwen3-8B", top_model_name=None, is_lora=True, lora_rank=32):
        self.model_data = FakeModelData(model_name)
        if top_model_name is not None:
            self.model_name = top_model_name
        self.is_lora = is_lora
        self.lora_rank = lora_rank


class FakeTrainingClient:
    def __init__(self, info):
        self.info = info

    async def get_info_async(self):
        return self.info


class FakeSaveResult:
    def __init__(self, path=None, include_path=True):
        if include_path:
            self.path = path


class FakeAwaitable:
    def __init__(self, result):
        self.result = result

    def __await__(self):
        async def done():
            return self.result

        return done().__await__()


class FakeNestedSaveResult:
    async def result_async(self):
        return FakeSaveResult("tinker://run/weights/state")


def test_mean_weighted_nll_rejects_all_zero_weights():
    with pytest.raises(ValueError, match="no trained tokens"):
        mean_weighted_nll(FakeForwardBackwardResult(), [FakeDatum()])


def base_config(tmp_path, **overrides):
    values = {
        "model": "Qwen/Qwen3-8B",
        "renderer_name": "qwen3",
        "data": str(tmp_path / "train.jsonl"),
        "output": str(tmp_path / "out"),
        "lora_rank": 32,
        "learning_rate": 1e-4,
        "batch_size": 4,
        "max_steps": 10,
        "warmup_steps": 0,
        "lr_schedule": "linear",
        "max_seq_length": 1024,
        "save_every": 5,
        "log_every": 5,
        "mask_prompt": True,
        "thinking": "keep",
        "checkpoint_mode": "resume",
        "checkpoint_path": "tinker://run/weights/state",
        "start_step": 5,
        "seed": 123,
        "dry_run": False,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip_norm": 0.0,
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
    values.update(overrides)
    return TinkerSFTConfig(**values)


def resume_record(config, **overrides):
    record = {
        "model": config.model,
        "renderer_name": config.renderer_name,
        "git_commit": "abcgit",
        "data_sha256": "abc123",
        "arrays_sha256": "arrays123",
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
        "max_seq_length": config.max_seq_length,
        "mask_prompt": config.mask_prompt,
        "thinking": config.thinking,
        "renderer_current_date": config.renderer_current_date,
    }
    record.update(overrides)
    return record


def write_matching_state_checkpoint(config, output_dir):
    (output_dir / "resume_meta.json").write_text(
        json.dumps({"step": config.start_step, "state_path": config.checkpoint_path})
    )


def test_validate_resume_config_rejects_data_order_mismatch(tmp_path):
    config = base_config(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    previous = resume_record(config, data_sha256="old")
    (output_dir / "config.json").write_text(json.dumps(previous))

    with pytest.raises(ValueError, match="data_sha256"):
        validate_resume_config(config, output_dir, resume_record(config))


def test_validate_resume_config_rejects_optimizer_mismatch(tmp_path):
    config = base_config(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "config.json").write_text(json.dumps(resume_record(config, learning_rate=2e-4)))
    write_matching_state_checkpoint(config, output_dir)

    with pytest.raises(ValueError, match="learning_rate"):
        validate_resume_config(config, output_dir, resume_record(config))


def test_validate_resume_config_rejects_missing_checkpoint_record(tmp_path):
    config = base_config(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "config.json").write_text(json.dumps(resume_record(config)))

    with pytest.raises(ValueError, match="could not find checkpoint_path"):
        validate_resume_config(config, output_dir, resume_record(config))


def test_validate_resume_config_rejects_checkpoint_step_mismatch(tmp_path):
    config = base_config(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "config.json").write_text(json.dumps(resume_record(config)))
    (output_dir / "resume_meta.json").write_text(
        json.dumps({"step": 4, "state_path": config.checkpoint_path, "sampler_path": "tinker://run/weights/final"})
    )

    with pytest.raises(ValueError, match="does not match start_step"):
        validate_resume_config(config, output_dir, resume_record(config))


def test_validate_config_rejects_wandb_enabled_without_project(tmp_path):
    config = base_config(tmp_path, checkpoint_mode="fresh", checkpoint_path="none", start_step=0, wandb_mode="online")
    with pytest.raises(ValueError, match="wandb_project"):
        validate_config(config)


def test_validate_config_rejects_invalid_renderer_date(tmp_path):
    config = base_config(tmp_path, renderer_current_date="2026-02-30")

    with pytest.raises(ValueError, match="renderer_current_date"):
        validate_config(config)


def test_gptoss_build_pins_the_fixed_default_renderer_date(tmp_path, monkeypatch):
    class DatedRenderer:
        use_system_prompt = True
        current_date = None

    seen = {}
    monkeypatch.setattr("src.training.tinker_common.load_tokenizer", lambda _model: object())
    monkeypatch.setattr(
        "src.training.tinker_common.load_renderer",
        lambda _model, _renderer_name, _tokenizer: seen.setdefault("renderer", DatedRenderer()),
    )
    monkeypatch.setattr("src.training.tinker_sft.load_chat_jsonl", lambda _path: (_ for _ in ()).throw(StopIteration))
    config = base_config(
        tmp_path,
        model="openai/gpt-oss-20b",
        renderer_name="gpt_oss_high_reasoning",
        renderer_current_date="",
    )
    with pytest.raises(StopIteration):
        build_arrays(config)
    assert seen["renderer"].current_date == DEFAULT_RENDERER_DATE


def test_validate_config_rejects_wandb_resume_without_run_id(tmp_path):
    config = base_config(
        tmp_path,
        checkpoint_mode="fresh",
        checkpoint_path="none",
        start_step=0,
        wandb_mode="offline",
        wandb_project="character-training-rl",
        wandb_resume="allow",
    )
    with pytest.raises(ValueError, match="wandb_run_id"):
        validate_config(config)


def test_init_wandb_disabled_returns_none(tmp_path):
    config = base_config(tmp_path)
    assert init_wandb(config, resume_record(config), tmp_path, job_type="tinker_sft") is None


async def test_verify_training_client_accepts_matching_lora(tmp_path):
    config = base_config(tmp_path)
    await verify_training_client(config, FakeTrainingClient(FakeInfo()))


async def test_verify_training_client_rejects_model_mismatch(tmp_path):
    config = base_config(tmp_path)
    with pytest.raises(ValueError, match="model mismatch"):
        await verify_training_client(config, FakeTrainingClient(FakeInfo(model_name="Qwen/Qwen3-14B")))


async def test_verify_training_client_rejects_top_level_model_disagreement(tmp_path):
    config = base_config(tmp_path)
    with pytest.raises(ValueError, match="model metadata mismatch"):
        await verify_training_client(config, FakeTrainingClient(FakeInfo(top_model_name="Qwen/Qwen3-14B")))


async def test_verify_training_client_rejects_non_lora(tmp_path):
    config = base_config(tmp_path)
    with pytest.raises(ValueError, match="is_lora"):
        await verify_training_client(config, FakeTrainingClient(FakeInfo(is_lora=False)))


async def test_verify_training_client_rejects_rank_mismatch(tmp_path):
    config = base_config(tmp_path)
    with pytest.raises(ValueError, match="LoRA rank"):
        await verify_training_client(config, FakeTrainingClient(FakeInfo(lora_rank=16)))


async def test_tinker_path_accepts_direct_path():
    assert (
        await _tinker_path(FakeAwaitable(FakeSaveResult("tinker://run/weights/state"))) == "tinker://run/weights/state"
    )


async def test_tinker_path_accepts_nested_future():
    assert await _tinker_path(FakeAwaitable(FakeNestedSaveResult())) == "tinker://run/weights/state"


async def test_tinker_path_rejects_empty_path():
    with pytest.raises(ValueError, match="non-empty path"):
        await _tinker_path(FakeAwaitable(FakeSaveResult("")))


async def test_tinker_path_rejects_missing_path():
    with pytest.raises(ValueError, match="non-empty path"):
        await _tinker_path(FakeAwaitable(FakeSaveResult(include_path=False)))


class _FakeStepFuture:
    def __init__(self, result):
        self._result = result

    async def result_async(self):
        return self._result


class _FakeStepFwdResult:
    def __init__(self, logprob, n):
        self.loss_fn_outputs = [{"logprobs": [logprob]} for _ in range(n)]


class _FakeStepClient:
    def __init__(self, logprob):
        self.logprob = logprob

    async def forward_backward_async(self, batch, loss_fn):
        assert loss_fn == "cross_entropy"
        return _FakeStepFuture(_FakeStepFwdResult(self.logprob, len(batch)))

    async def optim_step_async(self, adam_params):
        return _FakeStepFuture(None)


async def test_train_async_rejects_nonfinite_loss(tmp_path, monkeypatch):
    output = tmp_path / "out"
    output.mkdir()
    config = base_config(
        tmp_path,
        output=str(output),
        checkpoint_mode="fresh",
        checkpoint_path="none",
        start_step=0,
        batch_size=1,
        max_steps=1,
        save_every=2,
        log_every=1,
        dry_run=False,
    )
    arrays = [TrainingArrays(input_tokens=[1], target_tokens=[2], weights=[1.0], rendered_text="x")]

    async def fake_create_training_client(_config):
        return _FakeStepClient(logprob=math.nan)

    monkeypatch.setattr("src.training.tinker_sft.create_training_client", fake_create_training_client)

    with pytest.raises(ValueError, match="not finite"):
        await train_async(config, arrays, data_sha256="data-sha")
