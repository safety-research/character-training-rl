"""Tests for the raw synthetic-document Tinker trainer helpers."""

import json
import math
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch", reason="needs the training extra: uv sync --extra dev --extra training")

from src.training.arrays import TrainingArrays
from src.training.tinker_common import json_dumps_strict
from src.training.tinker_sdf import (
    RESUME_MATCH_FIELDS,
    TinkerSDFConfig,
    _jsonable_metrics,
    init_wandb,
    invocation_input_tokens,
    parse_args,
    run,
    train_async,
    validate_config,
    validate_resume_config,
    verify_tokenizer_id,
)


class FakeTokenizer:
    eos_token_id = 3

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join("<eos>" if token_id == self.eos_token_id else chr(token_id) for token_id in token_ids)


class FakeModelData:
    def __init__(self, tokenizer_id="Qwen/Qwen3-8B"):
        self.tokenizer_id = tokenizer_id
        self.model_name = "Qwen/Qwen3-8B"


class FakeInfo:
    def __init__(self, tokenizer_id="Qwen/Qwen3-8B"):
        self.model_data = FakeModelData(tokenizer_id)
        self.model_name = "Qwen/Qwen3-8B"
        self.is_lora = True
        self.lora_rank = 32


class FakeTrainingClientForInfo:
    def __init__(self, info):
        self.info = info

    async def get_info_async(self):
        return self.info


class FakeFuture:
    def __init__(self, result):
        self.result = result

    async def result_async(self):
        return self.result


class FakeForwardBackwardResult:
    def __init__(self, logprob, *, n_outputs=1, metrics=None, include_metrics=True):
        self.loss_fn_outputs = [{"logprobs": [logprob]} for _ in range(n_outputs)]
        if include_metrics:
            self.metrics = {} if metrics is None else metrics


class FakeOptimResult:
    def __init__(self, metrics=None, include_metrics=True):
        if include_metrics:
            self.metrics = metrics


class FakeTrainingClient:
    def __init__(self, *, logprob=-1.0, fwd_metrics=None, optim_metrics=None):
        self.info = FakeInfo()
        self.logprob = logprob
        self.fwd_metrics = fwd_metrics
        self.optim_metrics = optim_metrics
        self.saved = []

    async def get_info_async(self):
        return self.info

    async def forward_backward_async(self, batch, loss_fn):
        assert loss_fn == "cross_entropy"
        return FakeFuture(FakeForwardBackwardResult(self.logprob, n_outputs=len(batch), metrics=self.fwd_metrics))

    async def optim_step_async(self, adam_params):
        return FakeFuture(FakeOptimResult(metrics=self.optim_metrics))

    async def save_weights_for_sampler_async(self, name):
        self.saved.append(("sampler", name))
        return SimpleNamespace(path=f"tinker://run/sampler/{name}")

    async def save_state_async(self, name):
        self.saved.append(("state", name))
        return SimpleNamespace(path=f"tinker://run/state/{name}")


def base_config(tmp_path, **overrides):
    values = {
        "model": "Qwen/Qwen3-8B",
        "tokenizer_id": "Qwen/Qwen3-8B",
        "data": str(tmp_path / "train.jsonl"),
        "output": str(tmp_path / "out"),
        "lora_rank": 32,
        "learning_rate": 1e-4,
        "batch_size": 2,
        "max_steps": 10,
        "warmup_steps": 0,
        "lr_schedule": "linear",
        "max_seq_length": 1024,
        "append_eos": True,
        "save_every": 5,
        "log_every": 5,
        "checkpoint_mode": "fresh",
        "checkpoint_path": "none",
        "start_step": 0,
        "seed": 123,
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
    values.update(overrides)
    return TinkerSDFConfig(**values)


def resume_config(tmp_path, **overrides):
    values = {
        "checkpoint_mode": "resume",
        "checkpoint_path": "tinker://run/state",
        "start_step": 5,
        "dry_run": True,
    }
    values.update(overrides)
    return base_config(tmp_path, **values)


def resume_record(config, **overrides):
    record = {
        "model": config.model,
        "tokenizer_id": config.tokenizer_id,
        "git_commit": "abcgit",
        "data_sha256": "data-sha",
        "arrays_sha256": "arrays-sha",
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
        "append_eos": config.append_eos,
        "terminal_token": config.terminal_token,
        "data_format": config.data_format,
        "checkpoint_steps": list(config.checkpoint_steps),
    }
    record.update(overrides)
    return record


def write_matching_state_checkpoint(config, output_dir):
    (output_dir / "resume_meta.json").write_text(
        json.dumps({"step": config.start_step, "state_path": config.checkpoint_path})
    )


def full_argv(tmp_path):
    return [
        "--model",
        "Qwen/Qwen3-8B",
        "--tokenizer-id",
        "Qwen/Qwen3-8B",
        "--data",
        str(tmp_path / "train.jsonl"),
        "--output",
        str(tmp_path / "out"),
        "--lora-rank",
        "32",
        "--learning-rate",
        "1e-4",
        "--batch-size",
        "2",
        "--max-steps",
        "10",
        "--warmup-steps",
        "0",
        "--lr-schedule",
        "linear",
        "--max-seq-length",
        "1024",
        "--append-eos",
        "true",
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
        "123",
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
    ]


def alternate_value(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.1
    if isinstance(value, str):
        return value + "-changed"
    if isinstance(value, list):
        return [*value, 999]
    raise TypeError(f"no alternate value for {value!r}")


def test_parse_args_builds_config_and_disables_abbreviation(tmp_path):
    config = parse_args(full_argv(tmp_path))

    assert config.model == "Qwen/Qwen3-8B"
    assert config.tokenizer_id == "Qwen/Qwen3-8B"
    assert config.append_eos is True
    assert config.terminal_token == "none"
    assert config.data_format == "documents"
    assert config.end_step == 0
    assert config.checkpoint_steps == ()
    assert config.wandb_tags == ()

    abbreviated = full_argv(tmp_path)
    abbreviated[0] = "--mod"
    with pytest.raises(SystemExit):
        parse_args(abbreviated)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"tokenizer_id": ""}, "tokenizer_id"),
        ({"append_eos": "true"}, "append_eos"),
        ({"train_mlp": False, "train_attn": False, "train_unembed": False}, "train_mlp"),
        ({"log_every": 11}, "log_every"),
        ({"end_step": 11}, "end_step"),
        ({"checkpoint_steps": (3, 2)}, "checkpoint_steps"),
        ({"wandb_mode": "online", "wandb_project": ""}, "wandb_project"),
        ({"wandb_mode": "offline", "wandb_project": "proj", "wandb_resume": "allow"}, "wandb_run_id"),
    ],
)
def test_validate_config_rejects_bad_sdf_config(tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_config(base_config(tmp_path, **overrides))


@pytest.mark.parametrize("field", RESUME_MATCH_FIELDS)
def test_validate_resume_config_rejects_each_resume_identity_mismatch(tmp_path, field):
    config = resume_config(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    previous = resume_record(config)
    current = resume_record(config, **{field: alternate_value(previous[field])})
    (output_dir / "config.json").write_text(json.dumps(previous))
    write_matching_state_checkpoint(config, output_dir)

    with pytest.raises(ValueError, match=field):
        validate_resume_config(config, output_dir, current)


def test_validate_resume_config_rejects_missing_resume_field(tmp_path):
    config = resume_config(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    previous = resume_record(config)
    previous.pop("append_eos")
    (output_dir / "config.json").write_text(json.dumps(previous))
    write_matching_state_checkpoint(config, output_dir)

    with pytest.raises(ValueError, match="append_eos"):
        validate_resume_config(config, output_dir, resume_record(config))


def test_run_dry_run_writes_config_and_resume_attempt_without_overwriting(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"text": "alpha"}) + "\n" + json.dumps({"text": "beta"}) + "\n")
    fresh = base_config(tmp_path, data=str(data))
    resume = resume_config(tmp_path, data=str(data), output=fresh.output)

    monkeypatch.setattr("src.training.tinker_sdf.load_tokenizer", lambda tokenizer_id: FakeTokenizer())
    monkeypatch.setattr("src.training.tinker_sdf.get_git_commit", lambda: "abcgit")

    run(fresh)
    config_path = tmp_path / "out" / "config.json"
    original_config = json.loads(config_path.read_text())
    assert original_config["tokenizer_id"] == fresh.tokenizer_id
    assert original_config["append_eos"] is True

    write_matching_state_checkpoint(resume, tmp_path / "out")
    run(resume)

    assert json.loads(config_path.read_text()) == original_config
    attempts = (tmp_path / "out" / "resume_attempts.jsonl").read_text().splitlines()
    assert len(attempts) == 1
    attempt = json.loads(attempts[0])
    assert attempt["checkpoint_mode"] == "resume"
    assert attempt["tokenizer_id"] == resume.tokenizer_id


def test_init_wandb_requires_traceable_run_metadata(tmp_path, monkeypatch):
    config = base_config(
        tmp_path,
        wandb_mode="offline",
        wandb_project="character-training-rl",
        wandb_run_id="run-id",
        wandb_resume="allow",
    )
    record = resume_record(config)
    tmp_path.mkdir(exist_ok=True)

    fake_wandb = SimpleNamespace(init=lambda **_kwargs: SimpleNamespace(id="", project="character-training-rl"))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    with pytest.raises(RuntimeError, match="id"):
        init_wandb(config, record, tmp_path, job_type="sdf")

    fake_wandb = SimpleNamespace(init=lambda **_kwargs: SimpleNamespace(id="abc", project="wrong"))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    with pytest.raises(RuntimeError, match="project mismatch"):
        init_wandb(config, record, tmp_path, job_type="sdf")


async def test_verify_tokenizer_id_accepts_matching_metadata(tmp_path):
    config = base_config(tmp_path)

    await verify_tokenizer_id(config, FakeTrainingClientForInfo(FakeInfo(tokenizer_id=config.tokenizer_id)))


async def test_verify_tokenizer_id_rejects_metadata_mismatch(tmp_path):
    config = base_config(tmp_path)

    with pytest.raises(ValueError, match="tokenizer metadata mismatch"):
        await verify_tokenizer_id(config, FakeTrainingClientForInfo(FakeInfo(tokenizer_id="wrong-tokenizer")))


async def test_verify_tokenizer_id_rejects_missing_metadata(tmp_path):
    config = base_config(tmp_path)

    with pytest.raises(ValueError, match="tokenizer_id"):
        await verify_tokenizer_id(config, FakeTrainingClientForInfo(FakeInfo(tokenizer_id=None)))


def test_metrics_serializers_reject_nonfinite_or_non_numeric_metrics():
    assert _jsonable_metrics({"loss": 1.25}) == {"loss": 1.25}

    for value in (math.nan, math.inf, True, "bad"):
        with pytest.raises(ValueError, match="metric"):
            _jsonable_metrics({"bad": value})

    with pytest.raises(ValueError):
        json_dumps_strict({"loss": math.nan})


def test_invocation_input_tokens_prices_exact_deterministic_slice():
    arrays = [
        TrainingArrays(input_tokens=[1] * n, target_tokens=[2] * n, weights=[1.0] * n, rendered_text="")
        for n in (2, 5, 11, 17)
    ]
    # This also crosses an epoch boundary; compare a resumed slice with the
    # difference of exact prefixes rather than a mean-length estimate.
    full = invocation_input_tokens(arrays, batch_size=2, seed=7, start_step=0, end_step=3)
    prefix = invocation_input_tokens(arrays, batch_size=2, seed=7, start_step=0, end_step=1)
    resumed = invocation_input_tokens(arrays, batch_size=2, seed=7, start_step=1, end_step=3)

    assert resumed == full - prefix


async def test_train_async_rejects_nonfinite_loss(tmp_path, monkeypatch):
    output = tmp_path / "out"
    output.mkdir()
    config = base_config(tmp_path, output=str(output), max_steps=1, save_every=2, log_every=1, dry_run=False)
    arrays = [TrainingArrays(input_tokens=[1], target_tokens=[2], weights=[1.0], rendered_text="x")]

    async def fake_create_training_client(_config):
        return FakeTrainingClient(logprob=math.nan)

    monkeypatch.setattr("src.training.tinker_sdf.create_training_client", fake_create_training_client)

    with pytest.raises(ValueError, match="not finite"):
        await train_async(config, arrays, data_sha256="data-sha")


async def test_train_async_rejects_missing_tinker_metrics_field(tmp_path, monkeypatch):
    output = tmp_path / "out"
    output.mkdir()
    config = base_config(tmp_path, output=str(output), max_steps=1, save_every=2, log_every=1, dry_run=False)
    arrays = [TrainingArrays(input_tokens=[1], target_tokens=[2], weights=[1.0], rendered_text="x")]

    class MissingMetricsTrainingClient(FakeTrainingClient):
        async def forward_backward_async(self, batch, loss_fn):
            assert loss_fn == "cross_entropy"
            return FakeFuture(FakeForwardBackwardResult(-1.0, n_outputs=len(batch), include_metrics=False))

    async def fake_create_training_client(_config):
        return MissingMetricsTrainingClient()

    monkeypatch.setattr("src.training.tinker_sdf.create_training_client", fake_create_training_client)

    with pytest.raises(ValueError, match="metrics"):
        await train_async(config, arrays, data_sha256="data-sha")


async def test_train_async_saves_exact_checkpoint_and_clean_pause(tmp_path, monkeypatch):
    output = tmp_path / "out"
    output.mkdir()
    config = base_config(
        tmp_path,
        output=str(output),
        batch_size=1,
        max_steps=3,
        end_step=2,
        checkpoint_steps=(1, 2, 3),
        save_every=0,
        log_every=1,
        dry_run=False,
    )
    arrays = [
        TrainingArrays(input_tokens=[1], target_tokens=[2], weights=[1.0], rendered_text="a"),
        TrainingArrays(input_tokens=[3], target_tokens=[4], weights=[1.0], rendered_text="b"),
        TrainingArrays(input_tokens=[5], target_tokens=[6], weights=[1.0], rendered_text="c"),
    ]
    client = FakeTrainingClient()

    async def fake_create_training_client(_config):
        return client

    monkeypatch.setattr("src.training.tinker_sdf.create_training_client", fake_create_training_client)

    await train_async(config, arrays, data_sha256="data-sha")

    records = [json.loads(line) for line in (output / "checkpoints.jsonl").read_text().splitlines()]
    assert [(row["step"], row["type"]) for row in records] == [
        (1, "sampler"),
        (1, "state"),
        (2, "sampler"),
        (2, "state"),
    ]
    assert records[0]["checkpoint_trigger"] == "exact"
    assert records[-1]["paused"] is True
    assert records[-1]["final"] is False
    assert records[-1]["checkpoint_trigger"] == "exact"
    assert records[-1]["cumulative_trained_tokens"] == 2.0
    resume = json.loads((output / "resume_meta.json").read_text())
    assert resume["step"] == 2
    assert resume["state_path"].endswith("state_pause_step_2")


def test_validate_config_continue_mode_starts_at_step_zero(tmp_path):
    validate_config(base_config(tmp_path, checkpoint_mode="continue", checkpoint_path="tinker://round0/state"))
    with pytest.raises(ValueError, match="continue"):
        validate_config(
            base_config(tmp_path, checkpoint_mode="continue", checkpoint_path="tinker://round0/state", start_step=3)
        )
