"""One GRPO round of constitution RL: sample, score, write the trainer's rows.

Owns the environment; `src.training.rl_loop` owns the training step and the health gates (pass this
file as its --rollout-script). The environment is one user turn and nothing else: no system
message, no developer message, no constitution in context, exactly the user turn the
context-distillation rows hold.

Each round draws, seeded by the round, `--<source>-per-round` prompts from each pool built by
s4_build_pools:

    kodcode      correctness-gated: the code must pass every KodCode test (kodcode_check), then the
                 final answer is judged
    math         correctness-gated: the boxed answer must match the reference (rl_math), then judged
    constqa      judged; the constitution questions
    reflection   judged; the "why" questions, where the constitution's reason is rewarded. Only for
                 constitutions with a section on answering why (assigned rules, value-grounded
                 rules); for the others this source and its flags do not exist
    chat         judged; plain No Robots requests

Judged means the RL judge (rl_judge): the constitution's full text, the bare user message and the
answer channel only, never the reasoning; 1 when no rule is violated and the answer helped, else 0.
A wrong coding or MATH answer scores 0 and is not judged. A judge error is None, never a zero: its
row takes its own group's mean reward, so it carries no advantage, and it counts in
`judge_error_rate`.

Length: DAPO's soft overlong punishment, 0 up to `--overlong-start` tokens, linear to -1 at the cap
(`--max-tokens`, which must equal start + window). A completion the sampler cuts at the cap is never
scored; it trains at `--truncated-reward` in its group, so running into the cap is never better than
finishing (src.training.rollout `truncated_reward`). Token counts re-encode the decoded completion
with the base model's tokenizer.

Each scored row carries `answer_correct` (gated sources: the test or reference check) and `correct`
(the reward's 1: correct where gated, and following the constitution), which the loop's solve rate
reads. `correct` is False for a judge error, and `src.training.rollout` divides by every scored row,
so a judge error counts as unsolved in solve_rate.

Each judge verdict is appended to `<out>.verdicts.jsonl` under the key `<prompt id>#<sample_idx>`,
the `task_id` and `sample_idx` of the completion's row in `<out>.audit.jsonl`.

`src.training.rollout.rollout` does the sampling (every completion on disk before judging), the drop
accounting and the uniform-group exclusion; this file supplies the prompts and the scorer.

The KodCode check runs code written by the model in a subprocess on the machine this script runs
on, with no isolation beyond a separate process. Run it in a container or a virtual machine.
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from character_training.common import CONSTITUTIONS_WITH_WHY_SECTION, SPEC_SHA256
from character_training.constitution_rl import kodcode_check, rl_judge
from character_training.constitution_rl.rl_math import check_answer
from src.evals.common import positive_int
from src.jsonl import read_jsonl
from src.keys import load_project_env

ALL_SOURCES = ("kodcode", "math", "constqa", "reflection", "chat")
GATED = ("kodcode", "math")
"""Sources whose answer must be correct before the judge reads it."""
REQUIRED = {"kodcode": ("entry_point", "test"), "math": ("reference_answer",)}
WRONG = 0.0
SANDBOX = "subprocess"
"""src.code_exec sandbox for the KodCode tests, as in the context-distilled coding generation."""


def sources_for(constitution: str | None) -> tuple[str, ...]:
    """The prompt sources of one constitution, in draw order: every source, without `reflection` when the
    constitution has no section on answering why. None (no constitution named yet) gives every source."""
    if constitution is None or constitution in CONSTITUTIONS_WITH_WHY_SECTION:
        return ALL_SOURCES
    return tuple(src for src in ALL_SOURCES if src != "reflection")


def peek_constitution() -> str | None:
    """The --constitution on the command line, read before the full parser is built because the pool
    flags depend on it. None when it is absent (as under --help)."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--constitution", choices=sorted(SPEC_SHA256))
    return pre.parse_known_args()[0].constitution


def load_pool(path: Path, source: str) -> dict[str, dict]:
    """An s4_build_pools pool, refused if a row is from another source or lacks what its scorer reads."""
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r["id"].startswith(f"{source}:"):
            raise ValueError(f"{path}: row {r['id']} is not a {source} row")
        if r["id"] in out:
            raise ValueError(f"{path}: duplicate id {r['id']}")
        if len(r["messages"]) != 1 or r["messages"][0]["role"] != "user" or not r["question"]:
            raise ValueError(f"{path}: row {r['id']} must hold one user turn and its question")
        for f in REQUIRED.get(source, ()):
            if not r.get(f):
                raise ValueError(f"{path}: row {r['id']} has no {f}")
        out[r["id"]] = {**r, "source": source}
    if not out:
        raise ValueError(f"{path}: empty pool")
    return out


def draw(pool: dict[str, dict], k: int, rng: random.Random, source: str) -> list[str]:
    if k > len(pool):
        raise SystemExit(f"{k} {source} prompts per round exceeds the {len(pool)}-row pool")
    return rng.sample(sorted(pool), k)


def overlong_penalty(n_tokens: int, start: int, window: int) -> float:
    """DAPO's soft overlong punishment: 0 up to `start`, then linear down to -1 at `start + window`."""
    if n_tokens <= start:
        return 0.0
    return max(-1.0, (start - n_tokens) / window)


def correct(row: dict, answer: str) -> dict:
    """Correctness of one gated answer, with the checker's detail for KodCode."""
    if row["source"] == "math":
        return {"answer_correct": check_answer(answer, row["reference_answer"])}
    c = kodcode_check.check(answer, row, SANDBOX)
    return {"answer_correct": c.pop("correct"), **c}


def sample_indices(items: list[tuple[str, str]], samples_path: Path) -> list[int]:
    """The sample index of each (prompt id, completion) item within its prompt's group, read from the
    samples file the rollout writes before it scores. The rollout passes a prompt's completions in
    sample order with the unusable ones left out, so each item is the next completion of its prompt
    with that text."""
    texts = {row["task_id"]: [seq["text"] for seq in row["sequences"]] for row in read_jsonl(samples_path)}
    cursor: dict[str, int] = collections.defaultdict(int)
    out = []
    for i, text in items:
        if i not in texts or text not in texts[i][cursor[i] :]:
            raise ValueError(f"{samples_path}: no remaining completion of {i} has the text being scored")
        k = texts[i].index(text, cursor[i])
        cursor[i] = k + 1
        out.append(k)
    return out


def make_scorer(
    arm: str,
    by_id: dict[str, dict],
    judge_model: str,
    concurrency: int,
    exec_concurrency: int,
    renderer_name: str,
    verdicts_path: Path,
    samples_path: Path,
    tokenizer,
    overlong_start: int,
    overlong_window: int,
):
    """`score_many` for src.training.rollout: [(id, completion)] -> one dict per item with `reward`.
    Verdicts are keyed `<id>#<sample_idx>`, the index read from `samples_path`."""
    from src.thinking import split

    def score_many(items: list[tuple[str, str]]) -> list[dict]:
        answers = [split(text, renderer_name=renderer_name).answer for _, text in items]
        sample_idx = sample_indices(items, samples_path)
        out: list[dict] = [{"source": by_id[i]["source"]} for i, _ in items]
        gated = [n for n, (i, _) in enumerate(items) if by_id[i]["source"] in GATED]
        with ThreadPoolExecutor(max_workers=exec_concurrency) as pool:
            checks = list(pool.map(lambda n: correct(by_id[items[n][0]], answers[n]), gated))
        for n, c in zip(gated, checks):
            out[n].update(c)
        judged = [n for n in range(len(items)) if out[n]["source"] not in GATED or out[n]["answer_correct"]]
        verdicts = rl_judge.judge(
            arm,
            [by_id[items[n][0]]["question"] for n in judged],
            [answers[n] for n in judged],
            judge_model=judge_model,
            concurrency=concurrency,
            verdicts_path=verdicts_path,
            keys=[f"{items[n][0]}#{sample_idx[n]}" for n in judged],
        )
        for n in range(len(items)):
            out[n].setdefault("answer_correct", None)
            out[n].update({"verdict": None, "judge_error": None, "reward": WRONG, "correct": False})
        for n, v in zip(judged, verdicts):
            out[n].update(
                {
                    "verdict": v,
                    "judge_error": v is None,
                    "reward": None if v is None else v["reward"],
                    "correct": v is not None and v["follows"],
                }
            )

        by_group: dict[str, list[float]] = collections.defaultdict(list)
        for (i, _), o in zip(items, out):
            if o["reward"] is not None:
                by_group[i].append(o["reward"])
        for (i, _), o in zip(items, out):
            if o["reward"] is None:
                got = by_group.get(i)
                o["reward"] = sum(got) / len(got) if got else 0.0
        for (_, text), o in zip(items, out):
            n_tok = len(tokenizer.encode(text, add_special_tokens=False))
            o["base_reward"], o["n_tokens"] = o["reward"], n_tok
            o["length_penalty"] = overlong_penalty(n_tok, overlong_start, overlong_window)
            o["reward"] = o["base_reward"] + o["length_penalty"]
        return out

    return score_many


def summarize(rows: list[dict], arm: str) -> dict:
    """Gateable scalars: per source the judge's rates and, for the gated sources, accuracy; the
    judge error rate over every judged row; length."""
    out = {}
    judged = [r for r in rows if r["judge_error"] is not None]
    for src in sources_for(arm):
        g = [r for r in rows if r["source"] == src]
        out[f"n_{src}"] = len(g)
        if src in GATED:
            out[f"{src}_accuracy"] = sum(bool(r["answer_correct"]) for r in g) / len(g) if g else float("nan")
        out |= {
            f"{src}_{k}": v
            for k, v in rl_judge.rates([r["verdict"] for r in g if r["judge_error"] is not None], arm).items()
        }
    out["judge_error_rate"] = sum(r["judge_error"] for r in judged) / len(judged) if judged else 0.0
    pen = [r["length_penalty"] for r in rows]
    out["length_penalized_rate"] = sum(x < 0 for x in pen) / len(pen) if pen else 0.0
    out["n_tokens_median"] = sorted(r["n_tokens"] for r in rows)[len(rows) // 2] if rows else 0
    return out


def add_pool_args(p: argparse.ArgumentParser, split: str, sources: tuple[str, ...]) -> None:
    for src in sources:
        p.add_argument(f"--{src}-prompts", required=True, type=Path, help=f"{src}_{split}.jsonl from s4_build_pools")


def load_pools(a: argparse.Namespace, sources: tuple[str, ...]) -> dict[str, dict[str, dict]]:
    return {src: load_pool(getattr(a, f"{src}_prompts"), src) for src in sources}


def main() -> None:
    sources = sources_for(peek_constitution())
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    p.add_argument("--model", required=True, help="sampler checkpoint")
    p.add_argument("--base-model", required=True)
    p.add_argument("--renderer-name", required=True)
    p.add_argument(
        "--renderer-current-date",
        required=True,
        help="the date the harmony system prompt carries: the context-distillation stage's date, so prompt bytes match",
    )
    p.add_argument(
        "--constitution", required=True, choices=sorted(SPEC_SHA256), help="the constitution the judge reads"
    )
    add_pool_args(p, "train", sources)
    p.add_argument(
        "--n-prompts", required=True, type=positive_int, help="groups per round; must equal the per-round counts summed"
    )
    for src in sources:
        p.add_argument(f"--{src}-per-round", required=True, type=int)
    p.add_argument("--group-size", required=True, type=positive_int)
    p.add_argument("--temperature", required=True, type=float)
    p.add_argument("--top-p", required=True, type=float)
    p.add_argument("--max-tokens", required=True, type=positive_int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--max-concurrent", required=True, type=positive_int)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-concurrency", required=True, type=positive_int)
    p.add_argument("--exec-concurrency", required=True, type=positive_int, help="KodCode test runs at once")
    p.add_argument("--overlong-start", required=True, type=positive_int)
    p.add_argument("--overlong-window", required=True, type=positive_int)
    p.add_argument(
        "--truncated-reward",
        required=True,
        type=float,
        help="reward of a completion cut at --max-tokens; -1 is the floor of the overlong penalty",
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if a.max_tokens != a.overlong_start + a.overlong_window:
        raise SystemExit(
            f"--max-tokens {a.max_tokens} must equal --overlong-start + --overlong-window "
            f"({a.overlong_start + a.overlong_window})"
        )
    per_round = {src: getattr(a, f"{src}_per_round") for src in sources}
    if any(k < 0 for k in per_round.values()):
        raise SystemExit("per-round counts must be non-negative")
    if a.n_prompts != sum(per_round.values()):
        raise SystemExit(f"--n-prompts {a.n_prompts} must equal the per-round counts summed {per_round}")

    load_project_env()
    pools = load_pools(a, sources)
    by_id = {i: r for pool in pools.values() for i, r in pool.items()}
    rng = random.Random(a.seed)
    chosen = [i for src in sources for i in draw(pools[src], per_round[src], rng, src)]
    conversations = {i: by_id[i]["messages"] for i in chosen}

    if a.dry_run:
        print("pools: " + ", ".join(f"{len(pools[src])} {src}" for src in sources))
        print(f"round: {per_round} = {a.n_prompts} groups x {a.group_size} = {a.n_prompts * a.group_size} rollouts")
        for src in sources:
            i = next((q for q in chosen if by_id[q]["source"] == src), None)
            if i:
                print(f"\n===== what the model sees ({src}, {i}) =====\n[user] {conversations[i][0]['content']}")
        print(f"\n===== judge system prompt ({a.constitution}) =====\n{rl_judge.rubric(a.constitution)}")
        print(
            f"\n===== judge user message =====\n{rl_judge.USER_TEMPLATE.format(question='<the bare user message>', answer='...answer channel only...')}"
        )
        print(
            f"\nreward: kodcode and math {WRONG} if wrong, else judged; judged {rl_judge.REWARD_FOLLOWS} if follows "
            f"else {rl_judge.REWARD_VIOLATES}; overlong penalty from {a.overlong_start} to -1 at {a.max_tokens}"
        )
        return

    from src.evals.base import TinkerGenerator
    from src.training.rollout import SAMPLES_SUFFIX, rollout
    from src.training.tinker_common import load_tokenizer

    generator = TinkerGenerator(
        model_path=a.model,
        base_model=a.base_model if a.model != a.base_model else None,
        tokenizer_id=a.base_model,
        renderer_name=a.renderer_name,
        renderer_current_date=a.renderer_current_date,
        max_concurrent=a.max_concurrent,
    )
    generator.load()
    scorer = make_scorer(
        a.constitution,
        by_id,
        a.judge_model,
        a.judge_concurrency,
        a.exec_concurrency,
        a.renderer_name,
        a.out.with_suffix(".verdicts.jsonl"),
        a.out.with_suffix(SAMPLES_SUFFIX),
        load_tokenizer(a.base_model),
        a.overlong_start,
        a.overlong_window,
    )
    stats = rollout(
        generator,
        conversations,
        scorer,
        functools.partial(summarize, arm=a.constitution),
        n_prompts=len(conversations),
        group_size=a.group_size,
        seed=a.seed,
        temperature=a.temperature,
        top_p=a.top_p,
        max_tokens=a.max_tokens,
        thinking=True,
        out=a.out,
        truncated_reward=a.truncated_reward,
    )
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
