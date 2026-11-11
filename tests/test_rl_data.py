"""Tests for reward-training data contracts."""

import pytest

pytest.importorskip(
    "torch", reason="needs the training extra: uv sync --extra dev --extra training"
)

from src.training.rl_data import (
    IMPORTANCE_SAMPLING_LOSS,
    GRPORollout,
    build_grpo_training_arrays,
    build_rl_training_arrays,
    compute_dr_grpo_advantages,
    load_grpo_rollouts_jsonl,
    rl_arrays_to_tinker_datum,
)


def test_compute_dr_grpo_advantages_subtracts_group_mean():
    assert compute_dr_grpo_advantages([1.0, 0.0, 0.5]) == [0.5, -0.5, 0.0]


def test_compute_dr_grpo_advantages_constant_group_zeroes():
    assert compute_dr_grpo_advantages([0.25, 0.25]) == [0.0, 0.0]


def test_compute_dr_grpo_advantages_rejects_too_small_group():
    with pytest.raises(ValueError, match="at least two"):
        compute_dr_grpo_advantages([])
    with pytest.raises(ValueError, match="at least two"):
        compute_dr_grpo_advantages([1.0])


def test_build_rl_training_arrays_aligns_completion_signal():
    arrays = build_rl_training_arrays(
        prompt_tokens=[10, 11, 12],
        completion_tokens=[20, 21],
        completion_logprobs=[-0.3, -0.7],
        advantage=1.5,
    )

    assert arrays.input_tokens == [10, 11, 12, 20]
    assert arrays.target_tokens == [11, 12, 20, 21]
    assert arrays.logprobs == [0.0, 0.0, -0.3, -0.7]
    assert arrays.advantages == [0.0, 0.0, 1.5, 1.5]


def test_load_grpo_rollouts_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    path.write_text(
        '{"group_id":"g1","prompt_tokens":[1],"completion_tokens":[2],"completion_logprobs":[-0.1],'
        '"reward":1.0,"reward":0.0,"split":"train"}\n'
    )

    with pytest.raises(ValueError, match="duplicate JSON key 'reward'"):
        load_grpo_rollouts_jsonl(path, split="train")


def test_build_grpo_training_arrays_rejects_mixed_prompt_group():
    rollouts = [
        GRPORollout(
            group_id="g1",
            prompt_tokens=[1, 2],
            completion_tokens=[3],
            completion_logprobs=[-0.1],
            reward=1.0,
            split="train",
        ),
        GRPORollout(
            group_id="g1",
            prompt_tokens=[4, 5],
            completion_tokens=[6],
            completion_logprobs=[-0.2],
            reward=0.0,
            split="train",
        ),
    ]

    with pytest.raises(ValueError, match="multiple prompt token sequences"):
        build_grpo_training_arrays(rollouts)


def test_build_rl_training_arrays_rejects_logprob_mismatch():
    with pytest.raises(ValueError, match="completion_logprobs"):
        build_rl_training_arrays(
            prompt_tokens=[10],
            completion_tokens=[20, 21],
            completion_logprobs=[-0.3],
            advantage=1.0,
        )


def test_build_rl_training_arrays_rejects_nonfinite_advantage():
    with pytest.raises(ValueError, match="finite"):
        build_rl_training_arrays(
            prompt_tokens=[10],
            completion_tokens=[20],
            completion_logprobs=[-0.3],
            advantage=float("nan"),
        )


def test_rl_arrays_to_tinker_datum_rejects_prompt_advantage_signal():
    arrays = build_rl_training_arrays(
        prompt_tokens=[10, 11],
        completion_tokens=[20],
        completion_logprobs=[-0.3],
        advantage=1.0,
    )
    bad = type(arrays)(
        input_tokens=arrays.input_tokens,
        target_tokens=arrays.target_tokens,
        logprobs=arrays.logprobs,
        advantages=[1.0, 1.0],
        prompt_tokens=arrays.prompt_tokens,
        completion_tokens=arrays.completion_tokens,
    )

    with pytest.raises(ValueError, match="prompt target advantages"):
        rl_arrays_to_tinker_datum(bad)


def test_rl_arrays_to_tinker_datum_rejects_non_scalar_completion_advantages():
    arrays = build_rl_training_arrays(
        prompt_tokens=[10],
        completion_tokens=[20, 21],
        completion_logprobs=[-0.3, -0.4],
        advantage=1.0,
    )
    bad = type(arrays)(
        input_tokens=arrays.input_tokens,
        target_tokens=arrays.target_tokens,
        logprobs=arrays.logprobs,
        advantages=[1.0, 2.0],
        prompt_tokens=arrays.prompt_tokens,
        completion_tokens=arrays.completion_tokens,
    )

    with pytest.raises(ValueError, match="one scalar"):
        rl_arrays_to_tinker_datum(bad)


def test_rl_arrays_to_tinker_datum_has_importance_sampling_inputs():
    arrays = build_rl_training_arrays(
        prompt_tokens=[10, 11],
        completion_tokens=[20],
        completion_logprobs=[-0.3],
        advantage=1.0,
    )

    datum = rl_arrays_to_tinker_datum(arrays)

    assert IMPORTANCE_SAMPLING_LOSS == "importance_sampling"
    assert set(datum.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
