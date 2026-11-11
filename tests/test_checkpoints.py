"""src.training.checkpoints: which rows are checkpoints, and what a run without a final one does."""

import json

import pytest

from src.training.checkpoints import (
    Checkpoint,
    final_sampler,
    final_state,
    find_runs,
    read_checkpoints,
    select,
)


def make_run(tmp_path, rows, *, config=None, name="run"):
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if config is not None:
        (run_dir / "config.json").write_text(json.dumps(config))
    return run_dir


CONFIG = {"model": "Qwen/Qwen3.6-35B-A3B", "renderer_name": "qwen3_5"}


def complete_run(tmp_path, name="run"):
    return make_run(
        tmp_path,
        [
            {"step": 100, "type": "sampler", "path": "tinker://s100"},
            {"step": 200, "type": "sampler", "path": "tinker://s200"},
            {"step": 200, "type": "state", "path": "tinker://state200"},
            {"step": 250, "type": "sampler", "path": "tinker://final", "final": True},
        ],
        config=CONFIG,
        name=name,
    )


def test_checkpoints_carry_their_own_sampling_config(tmp_path):
    ckpt = final_sampler(complete_run(tmp_path))
    assert ckpt.path == "tinker://final"
    assert ckpt.sampling_kwargs() == {
        "backend": "tinker",
        "base_model": "Qwen/Qwen3.6-35B-A3B",
        "tokenizer_id": "Qwen/Qwen3.6-35B-A3B",
        "renderer_name": "qwen3_5",
    }


def test_final_sampler_never_returns_the_optimizer_state(tmp_path):
    # The bug in the readers that take the last row: on a src-written log that is the state,
    # which cannot be sampled at all.
    ckpt = final_sampler(complete_run(tmp_path))
    assert ckpt.kind == "sampler" and ckpt.samplable


def test_state_and_sampler_are_different_checkpoints(tmp_path):
    run_dir = complete_run(tmp_path)
    assert final_state(run_dir).path == "tinker://state200"
    assert final_sampler(run_dir).path == "tinker://final"


def test_interrupted_run_raises_and_says_how_to_ask_for_the_last_one(tmp_path):
    run_dir = make_run(
        tmp_path,
        [
            {"step": 100, "type": "sampler", "path": "tinker://s100"},
            {"step": 200, "type": "sampler", "path": "tinker://s200"},
        ],
        config=CONFIG,
    )
    with pytest.raises(ValueError, match="no checkpoint is marked final"):
        final_sampler(run_dir)
    assert select(run_dir, "last")[0].path == "tinker://s200"


def test_typeless_lr_sweep_schema_is_rejected_not_guessed_at(tmp_path):
    run_dir = make_run(tmp_path, [{"step": 100, "path": "tinker://x/sampler_weights/final"}], config=CONFIG)
    with pytest.raises(ValueError, match="will not infer a checkpoint kind"):
        read_checkpoints(run_dir)


def test_missing_config_leaves_the_checkpoint_unsamplable_rather_than_guessing(tmp_path):
    run_dir = make_run(tmp_path, [{"step": 1, "type": "sampler", "path": "tinker://x", "final": True}])
    ckpt = final_sampler(run_dir)
    assert ckpt.base_model is None
    with pytest.raises(ValueError, match="records no base model"):
        ckpt.sampling_kwargs()


def test_select_specs(tmp_path):
    run_dir = complete_run(tmp_path)
    assert [c.step for c in select(run_dir, "final")] == [250]
    assert [c.step for c in select(run_dir, "all")] == [100, 200, 250]
    # every:N keeps the multiples and always includes the last, so a sweep has an endpoint.
    assert [c.step for c in select(run_dir, "every:100")] == [100, 200, 250]
    with pytest.raises(ValueError, match="unknown checkpoint spec"):
        select(run_dir, "penultimate")


def test_not_a_run_directory_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="training run directory"):
        read_checkpoints(tmp_path)


def test_find_runs_discovers_models_by_walking_results(tmp_path):
    complete_run(tmp_path, name="cell_a")
    complete_run(tmp_path, name="cell_b")
    (tmp_path / "not_a_run").mkdir()
    assert sorted(find_runs(tmp_path)) == ["cell_a", "cell_b"]


def test_checkpoint_is_hashable_so_it_can_go_in_sets(tmp_path):
    ckpts = read_checkpoints(complete_run(tmp_path))
    assert isinstance(ckpts[0], Checkpoint)
    assert len(set(ckpts)) == len(ckpts)
