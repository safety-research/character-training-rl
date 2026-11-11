"""src.training.rollout: rows the trainer accepts, drop accounting, uniform-group dropping."""

from __future__ import annotations

import pytest

from src.evals.base import SampledPrompt, SampledSequence
from src.jsonl import read_jsonl, read_jsonl_missing_ok
from src.training.rl_data import load_grpo_rollouts_jsonl
from src.training.rollout import SAMPLES_SUFFIX, rollout, sample_persisted

FINAL = "assistantfinal"


class FakeGenerator:
    """Returns one scripted completion set per prompt, in the order prompts are passed."""

    renderer_name = "gpt_oss_low_reasoning"
    max_token_id = 200_018  # tokens well under this are never dropped as reserved

    def __init__(self, script: dict[str, list[tuple[str, str]]]):
        self.script = script  # task_id -> [(text, stop_reason)]
        self.calls = []

    def sample_chat(self, conversations, *, n, max_tokens, temperature, top_p, thinking, on_sampled=None):
        self.calls.append(dict(n=n, max_tokens=max_tokens, temperature=temperature, top_p=top_p, thinking=thinking))
        self.conversations = conversations
        out = []
        for idx, conv in enumerate(conversations):
            tid = conv[0]["content"]
            seqs = [SampledSequence(tokens=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3], text=text, stop_reason=stop)
                    for text, stop in self.script[tid]]
            assert len(seqs) == n
            out.append(SampledPrompt(prompt_tokens=[9, 9], sequences=seqs))
            if on_sampled is not None:
                on_sampled(idx, out[-1])
        return out


def score_many(items):
    return [{"reward": 3.5 if "ok" in text else 0.5, "correct": "ok" in text, "can_compile": True} for _tid, text in items]


def summarize(rows):
    return {"compile_rate": sum(r["can_compile"] for r in rows) / len(rows), "mean_reward": sum(r["reward"] for r in rows) / len(rows),
            "by_assertion_set": {"nested": 1}}


def test_rollout_writes_trainer_rows_and_drops_uniform_groups_and_broken_samples(tmp_path):
    gen = FakeGenerator({
        "a": [(f"{FINAL}ok", "stop"), (f"{FINAL}bad", "stop")],          # varied: trained
        "b": [(f"{FINAL}ok", "stop"), (f"{FINAL}ok", "stop")],           # uniform: dropped
        "c": [("still reasoning", "length"), (f"{FINAL}bad", "stop")],   # one truncated, one scored; uniform after drop
    })
    prompts = {t: [{"role": "user", "content": t}] for t in "abc"}
    out = tmp_path / "round0.jsonl"
    stats = rollout(gen, prompts, score_many, summarize, n_prompts=3, group_size=2, seed=0, temperature=1.0,
                    top_p=0.95, max_tokens=100, thinking=True, out=out, truncated_reward=None)
    assert gen.calls == [dict(n=2, max_tokens=100, temperature=1.0, top_p=0.95, thinking=True)]
    assert stats["n_sampled"] == 6 and stats["n_truncated"] == 1 and stats["n_scored"] == 5
    assert stats["n_groups"] == 3 and stats["n_groups_trained"] == 1 and stats["n_rows"] == 2
    assert stats["compile_rate"] == 1.0 and "by_assertion_set" not in stats  # nested aggregates are not gateable
    trained = load_grpo_rollouts_jsonl(out, split="train")  # the trainer's own strict loader accepts the file
    assert [r.group_id for r in trained] == ["a", "a"] and [r.reward for r in trained] == [3.5, 0.5]
    assert trained[0].prompt_tokens == [9, 9] and trained[0].completion_logprobs == [-0.1, -0.2, -0.3]
    assert len(read_jsonl(out.with_suffix(".audit.jsonl"))) == 5


class VocabGenerator:
    """Emits scripted (text, stop_reason, tokens) per prompt, and caps the trainer's vocab."""

    renderer_name = "gpt_oss_low_reasoning"

    def __init__(self, script: dict, max_token_id: int):
        self.script = script  # tid -> [(text, stop, tokens)]
        self.max_token_id = max_token_id

    def sample_chat(self, conversations, *, n, max_tokens, temperature, top_p, thinking, on_sampled=None):
        out = []
        for idx, conv in enumerate(conversations):
            tid = conv[0]["content"]
            seqs = [SampledSequence(tokens=toks, logprobs=[-0.1] * len(toks), text=text, stop_reason=stop)
                    for text, stop, toks in self.script[tid]]
            out.append(SampledPrompt(prompt_tokens=[9, 9], sequences=seqs))
            if on_sampled is not None:
                on_sampled(idx, out[-1])
        return out


def test_rollout_drops_reserved_token_completions(tmp_path):
    # 999 exceeds max_token_id 100: the trainer would reject that completion, so it is dropped and
    # counted (n_reserved), never handed to the scorer.
    gen = VocabGenerator(
        {"a": [(f"{FINAL}ok", "stop", [1, 2, 3]), (f"{FINAL}bad", "stop", [1, 2, 999])]},
        max_token_id=100,
    )
    seen = []

    def spy_score(items):
        seen.extend(text for _tid, text in items)
        return score_many(items)

    out = tmp_path / "r.jsonl"
    stats = rollout(gen, {"a": [{"role": "user", "content": "a"}]}, spy_score, summarize,
                    n_prompts=1, group_size=2, seed=0, temperature=1.0, top_p=1.0, max_tokens=10,
                    thinking=True, out=out, truncated_reward=None)
    assert stats["n_sampled"] == 2 and stats["n_reserved"] == 1 and stats["n_scored"] == 1
    assert stats["drop_rate"] == 0.5
    assert seen == [f"{FINAL}ok"]  # the reserved-token completion never reached the scorer


def test_rollout_refuses_when_nothing_is_scoreable(tmp_path):
    gen = FakeGenerator({"a": [("no marker", "stop")]})
    with pytest.raises(RuntimeError):
        rollout(gen, {"a": [{"role": "user", "content": "a"}]}, score_many, summarize, n_prompts=1, group_size=1,
                seed=0, temperature=1.0, top_p=1.0, max_tokens=10, thinking=True, out=tmp_path / "r.jsonl", truncated_reward=None)


SCRIPT = {
    "a": [(f"{FINAL}ok", "stop"), (f"{FINAL}bad", "stop")],
    "b": [(f"{FINAL}ok", "stop"), (f"{FINAL}ok", "stop")],
    "c": [("still reasoning", "length"), (f"{FINAL}bad", "stop")],
}
PROMPTS = {t: [{"role": "user", "content": t}] for t in "abc"}


def _rollout(gen, scorer, out):
    return rollout(
        gen,
        PROMPTS,
        scorer,
        summarize,
        n_prompts=3,
        group_size=2,
        seed=0,
        temperature=1.0,
        top_p=0.95,
        max_tokens=100,
        thinking=True,
        out=out,
        truncated_reward=None,
    )


def test_every_completion_is_on_disk_before_the_judge_runs_and_a_rerun_does_not_resample(tmp_path):
    """The judge dies: the round's sampling is already in `<out>.samples.jsonl`, dropped completions
    included. The re-run reads it back and produces the same rows without one more sample call."""
    out = tmp_path / "round0.jsonl"

    def judge_dies(items):
        raise RuntimeError("judge down")

    gen = FakeGenerator(SCRIPT)
    with pytest.raises(RuntimeError, match="judge down"):
        _rollout(gen, judge_dies, out)
    samples = read_jsonl(out.with_suffix(SAMPLES_SUFFIX))
    assert sorted(r["task_id"] for r in samples) == ["a", "b", "c"]
    assert sum(len(r["sequences"]) for r in samples) == 6  # the truncated one is there too
    assert not out.exists() and not out.with_suffix(".audit.jsonl").exists()

    gen2 = FakeGenerator(SCRIPT)
    stats = _rollout(gen2, score_many, out)
    assert gen2.calls == []  # nothing sampled again
    assert stats["n_sampled"] == 6 and stats["n_truncated"] == 1 and stats["n_rows"] == 2
    trained = load_grpo_rollouts_jsonl(out, split="train")
    assert [r.reward for r in trained] == [3.5, 0.5] and trained[0].completion_logprobs == [-0.1, -0.2, -0.3]


def test_only_the_prompts_missing_from_the_samples_file_are_sampled(tmp_path):
    path = tmp_path / "r.samples.jsonl"
    kw = dict(n=2, max_tokens=10, temperature=1.0, top_p=1.0, thinking=True, path=path)
    first = FakeGenerator(SCRIPT)
    sample_persisted(first, PROMPTS, ["a"], **kw)
    assert [r["task_id"] for r in read_jsonl_missing_ok(path)] == ["a"]
    second = FakeGenerator(SCRIPT)
    got = sample_persisted(second, PROMPTS, ["a", "b", "c"], **kw)
    assert [c[0]["content"] for c in second.conversations] == ["b", "c"]
    assert [sp.sequences[0].text for sp in got] == [f"{FINAL}ok", f"{FINAL}ok", "still reasoning"]
    assert [r["task_id"] for r in read_jsonl_missing_ok(path)] == ["a", "b", "c"]


def test_a_samples_file_from_another_run_is_refused(tmp_path):
    path = tmp_path / "r.samples.jsonl"
    sample_persisted(
        FakeGenerator(SCRIPT),
        PROMPTS,
        ["a", "b"],
        n=2,
        max_tokens=10,
        temperature=1.0,
        top_p=1.0,
        thinking=True,
        path=path,
    )
    with pytest.raises(ValueError, match="different run"):  # a different group size
        sample_persisted(
            FakeGenerator(SCRIPT),
            PROMPTS,
            ["a", "b"],
            n=1,
            max_tokens=10,
            temperature=1.0,
            top_p=1.0,
            thinking=True,
            path=path,
        )
    with pytest.raises(ValueError, match="did not pick"):  # a different prompt pick
        sample_persisted(
            FakeGenerator(SCRIPT),
            PROMPTS,
            ["a"],
            n=2,
            max_tokens=10,
            temperature=1.0,
            top_p=1.0,
            thinking=True,
            path=path,
        )


def test_truncated_reward_trains_cut_off_completions_instead_of_dropping_them(tmp_path):
    """With a truncated_reward, a cut-off completion is not scored but trained at that reward in its
    group: group c (one truncated, one scored at 0.5) now varies and trains. It is in the audit file,
    not in what summarize sees, and it leaves the drop rate (it was trained, not dropped)."""
    gen = FakeGenerator(SCRIPT)
    seen = []

    def spy(items):
        seen.extend(text for _tid, text in items)
        return score_many(items)

    out = tmp_path / "round0.jsonl"
    stats = rollout(gen, PROMPTS, spy, summarize, n_prompts=3, group_size=2, seed=0, temperature=1.0,
                    top_p=0.95, max_tokens=100, thinking=True, out=out, truncated_reward=-1.0)
    assert "still reasoning" not in seen  # never scored
    assert stats["n_truncated"] == 1 and stats["n_truncated_trained"] == 1 and stats["drop_rate"] == 0.0
    assert stats["truncated_rate"] == 1 / 6 and stats["n_scored"] == 5
    trained = load_grpo_rollouts_jsonl(out, split="train")
    assert sorted({r.group_id for r in trained}) == ["a", "c"]
    assert sorted(r.reward for r in trained if r.group_id == "c") == [-1.0, 0.5]
    audit = read_jsonl(out.with_suffix(".audit.jsonl"))
    assert len(audit) == 6 and [r for r in audit if r.get("truncated")][0]["reward"] == -1.0


def test_truncated_reward_never_trains_a_cut_off_completion_with_reserved_tokens(tmp_path):
    gen = VocabGenerator({"a": [("still reasoning", "length", [1, 500]), (f"{FINAL}ok", "stop", [1, 2])]}, max_token_id=100)
    stats = rollout(gen, {"a": [{"role": "user", "content": "a"}]}, score_many, summarize, n_prompts=1, group_size=2,
                    seed=0, temperature=1.0, top_p=0.95, max_tokens=100, thinking=True, out=tmp_path / "r.jsonl",
                    truncated_reward=-1.0)
    assert stats["n_truncated"] == 1 and stats["n_truncated_trained"] == 0 and stats["drop_rate"] == 0.5


def test_truncated_rows_carry_their_own_tokens_and_uniform_truncated_groups_are_not_trained(tmp_path):
    """Each sequence has distinct tokens and logprobs, so a cut-off row must carry its own. A group whose
    only survivors are cut off is uniform at the truncated reward and is dropped, and does not count as
    trained."""
    gen = VocabGenerator({
        "a": [("still reasoning", "length", [7, 8]), (f"{FINAL}ok", "stop", [1, 2, 3])],
        "b": [("cut", "length", [4]), ("cut again", "length", [5, 6])],
    }, max_token_id=100)
    prompts = {t: [{"role": "user", "content": t}] for t in "ab"}
    out = tmp_path / "r.jsonl"
    stats = rollout(gen, prompts, score_many, summarize, n_prompts=2, group_size=2, seed=0, temperature=1.0,
                    top_p=0.95, max_tokens=100, thinking=True, out=out, truncated_reward=-1.0)
    trained = load_grpo_rollouts_jsonl(out, split="train")
    assert [r.group_id for r in trained] == ["a", "a"]
    cut = next(r for r in trained if r.reward == -1.0)
    assert cut.completion_tokens == [7, 8] and len(cut.completion_logprobs) == 2
    assert stats["n_truncated"] == 3 and stats["n_truncated_trained"] == 1
