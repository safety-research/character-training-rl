"""One GRPO round of data: sample a batch of prompts, score, drop uniform-reward groups, write
the trainer's rows.

Every completion is on disk (`<out>.samples.jsonl`) the moment its prompt returns, before any
judge sees it, and a re-run over the same `out` samples only the prompts that file lacks. A judge
that dies, a killed process, or a crashed trainer therefore never costs the round's sampling twice.

A group whose reward never varies has zero advantage; it is dropped and counted, never
resampled. The seed picks prompts; sampling is unseeded. Unclosed and reserved-token completions
(a token id the trainer rejects) are dropped and counted, never scored. A truncated completion (one
the sampler cut at `max_tokens`) is never scored either; what happens to it is the caller's explicit
choice, `truncated_reward`: None drops and counts it (DAPO's overlong filtering), a number trains it
with that reward in its group, so running into the cap is penalised rather than invisible.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Callable

from src.evals.base import SampledPrompt, SampledSequence
from src.jsonl import append_jsonl, read_jsonl_missing_ok, write_jsonl
from src.thinking import split, usable
from src.training.rl_data import GRPO_ROLLOUT_KEYS

ScoreMany = Callable[[list[tuple[str, str]]], list[dict]]
#: Beside `<out>`: one row per sampled prompt, written as it returns, read back on a re-run.
SAMPLES_SUFFIX = ".samples.jsonl"


def _sample_row(task_id: str, sp: SampledPrompt) -> dict:
    return {
        "task_id": task_id,
        "prompt_tokens": sp.prompt_tokens,
        "sequences": [
            {"tokens": s.tokens, "logprobs": s.logprobs, "text": s.text, "stop_reason": s.stop_reason}
            for s in sp.sequences
        ],
    }


def _sample_from_row(row: dict) -> SampledPrompt:
    return SampledPrompt(
        prompt_tokens=list(row["prompt_tokens"]),
        sequences=[
            SampledSequence(
                tokens=list(s["tokens"]),
                logprobs=[float(x) for x in s["logprobs"]],
                text=s["text"],
                stop_reason=s["stop_reason"],
            )
            for s in row["sequences"]
        ],
    )


def sample_persisted(
    generator,
    prompts: dict[str, list[dict]],
    task_ids: list[str],
    *,
    n: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    thinking: bool,
    path: Path,
) -> list[SampledPrompt | None]:
    """`n` completions per task, each task's row appended and flushed to `path` the moment it
    returns, before anything reads or judges it. A task already in `path` is not sampled again,
    so a run killed after sampling resumes without paying for the same completions twice. One row
    per task, so the file is whole at every instant. A file whose tasks or group size do not match
    this run is refused rather than mixed in. Returns one entry per task id, None for a task the
    provider failed after its retry budget."""
    have: dict[str, SampledPrompt] = {}
    for row in read_jsonl_missing_ok(path):
        tid = row["task_id"]
        if tid in have:
            raise ValueError(f"{path}: task {tid!r} appears twice")
        if len(row["sequences"]) != n:
            raise ValueError(
                f"{path}: task {tid!r} holds {len(row['sequences'])} completions but this run samples "
                f"{n} per task; the file is from a different run"
            )
        have[tid] = _sample_from_row(row)
    foreign = sorted(set(have) - set(task_ids))
    if foreign:
        raise ValueError(
            f"{path}: holds tasks this run did not pick ({foreign[:5]}...); the file is from a different run"
        )
    todo = [t for t in task_ids if t not in have]
    if have:
        print(
            f"  {len(have)} of {len(task_ids)} prompts already sampled in {path.name}; sampling {len(todo)}", flush=True
        )
    if todo:

        def keep(idx: int, sp: SampledPrompt) -> None:
            append_jsonl(path, _sample_row(todo[idx], sp))

        fresh = generator.sample_chat(
            [prompts[t] for t in todo],
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            thinking=thinking,
            on_sampled=keep,
        )
        for t, sp in zip(todo, fresh):
            if sp is not None:
                have[t] = sp
    return [have.get(t) for t in task_ids]


def closed(text: str, stop_reason: str, thinking: bool, renderer_name: str) -> bool:
    """Usable AND the answer channel actually closed. `src.thinking.usable` deliberately does not
    ask a harmony renderer about closure (truncation there is caught by the stop reason), but a
    completion still inside its reasoning channel is not an answer and must never earn a reward."""
    if not usable(text, stop_reason=stop_reason, thinking=thinking, renderer_name=renderer_name):
        return False
    return not thinking or split(text, renderer_name=renderer_name).closed


def partition_completions(
    task_ids: list[str], sampled: list[SampledPrompt | None], *, thinking: bool, renderer_name: str
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
    """Sort an evaluation's completions by whether they may be judged: `(task_id, text)` for every
    completion that passes `closed`, `(task_id, text)` for every one that does not (truncated, empty,
    or still inside its reasoning channel), and the task ids whose conversation the provider failed.
    A completion that never reached its answer channel has no answer to judge, so a caller leaves
    the second list out of every rate and reports its size."""
    finished, unfinished, failed = [], [], []
    for task_id, sp in zip(task_ids, sampled, strict=True):
        if sp is None:
            failed.append(task_id)
            continue
        for seq in sp.sequences:
            ok = closed(seq.text, seq.stop_reason, thinking, renderer_name)
            (finished if ok else unfinished).append((task_id, seq.text))
    return finished, unfinished, failed


def rollout(
    generator,
    prompts: dict[str, list[dict]],
    score_many: ScoreMany,
    summarize: Callable[[list[dict]], dict],
    *,
    n_prompts: int,
    group_size: int,
    seed: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    thinking: bool,
    out: Path,
    truncated_reward: float | None,
) -> dict:
    """Writes `<out>.samples.jsonl` (every completion, as it returns, before judging), then `<out>`
    (rows with exactly GRPO_ROLLOUT_KEYS), `<out>.audit.jsonl` (every scored completion, and every
    truncated one trained at `truncated_reward`) and `<out>.stats.json`. Returns the stats: generic
    counts plus the scalars of `summarize(rows)` over the scored rows, all gateable by name."""
    import random

    if n_prompts > len(prompts):
        raise ValueError(f"n_prompts {n_prompts} exceeds the {len(prompts)}-task pool")
    chosen = random.Random(seed).sample(sorted(prompts), n_prompts)
    sampled = sample_persisted(
        generator,
        prompts,
        chosen,
        n=group_size,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        thinking=thinking,
        path=out.with_suffix(SAMPLES_SUFFIX),
    )

    # gpt-oss's model and full harmony tokenizer span 201,088 ids, but the loaded tokenizer defines
    # 200,019; ids 200,019-201,087 are reserved special slots the sampler can emit and the trainer's
    # forward_backward rejects. Such a completion is unusable: dropped and counted like a truncated
    # one, since keeping it crashes the whole training step.
    max_token_id = generator.max_token_id
    items, keep, truncated = [], [], []
    n_sampled = n_truncated = n_unclosed = n_reserved = 0
    for tid, sp in zip(chosen, sampled):
        if sp is None:
            continue
        for k, seq in enumerate(sp.sequences):
            n_sampled += 1
            if seq.stop_reason == "length":
                n_truncated += 1
                if truncated_reward is not None and not any(t > max_token_id for t in seq.tokens):
                    truncated.append((tid, k, sp.prompt_tokens, seq))
            elif not closed(seq.text, seq.stop_reason, thinking, generator.renderer_name):
                n_unclosed += 1
            elif any(t > max_token_id for t in seq.tokens):
                n_reserved += 1
            else:
                items.append((tid, seq.text))
                keep.append((tid, k, sp.prompt_tokens, seq))
    if not items:
        raise RuntimeError(
            f"every one of {n_sampled} completions was dropped ({n_truncated} truncated, "
            f"{n_unclosed} unclosed, {n_reserved} reserved-token): max_tokens is far too "
            f"small, the renderer is wrong, or the sampler is emitting reserved tokens"
        )
    verdicts = score_many(items)

    rows, groups = [], collections.defaultdict(list)
    for (tid, k, prompt_tokens, seq), v in zip(keep, verdicts):
        rows.append({"task_id": tid, "sample_idx": k, "completion_tokens": len(seq.tokens), "response": seq.text, **v})
        groups[tid].append(
            {
                "group_id": tid,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": seq.tokens,
                "completion_logprobs": seq.logprobs,
                "reward": v["reward"],
                "split": "train",
            }
        )
    audit = list(rows)
    for tid, k, prompt_tokens, seq in truncated:
        audit.append(
            {
                "task_id": tid,
                "sample_idx": k,
                "completion_tokens": len(seq.tokens),
                "response": seq.text,
                "reward": truncated_reward,
                "correct": False,
                "truncated": True,
            }
        )
        groups[tid].append(
            {
                "group_id": tid,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": seq.tokens,
                "completion_logprobs": seq.logprobs,
                "reward": truncated_reward,
                "split": "train",
                "truncated": True,
            }
        )
    trained = {g: rs for g, rs in groups.items() if len({r["reward"] for r in rs}) > 1}

    write_jsonl(out.with_suffix(".audit.jsonl"), audit)
    write_jsonl(out, ({k: r[k] for k in GRPO_ROLLOUT_KEYS} for g in sorted(trained) for r in trained[g]))

    stats = {
        "n_sampled": n_sampled,
        "n_scored": len(rows),
        "n_truncated": n_truncated,
        "n_unclosed": n_unclosed,
        "n_reserved": n_reserved,
        "n_truncated_trained": sum(1 for rs in trained.values() for r in rs if r.get("truncated")),
        "truncated_rate": n_truncated / max(n_sampled, 1),
        "drop_rate": (n_truncated - len(truncated) + n_unclosed + n_reserved) / max(n_sampled, 1),
        "n_groups": len(groups),
        "n_groups_trained": len(trained),
        "group_survival_rate": len(trained) / max(len(groups), 1),
        "n_rows": sum(len(v) for v in trained.values()),
        "solve_rate": sum(1 for r in rows if r["correct"]) / len(rows),
        **{k: v for k, v in summarize(rows).items() if isinstance(v, (int, float)) or v is None},
    }
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats
