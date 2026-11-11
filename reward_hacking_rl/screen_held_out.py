"""The held-out screen: a mechanically drawn set of tasks from the reward-hacking environments of
astOwOlfo/tinker-cookbook-reward-hacking, on which gpt-oss-20b is sampled before any training. The
task selection it writes (selection.json) is held out of reward-hacking RL.

Samples gpt-oss-20b, before any training, on their environments with the settings of their `big_run.py`
(9 tool calls, 32,768-token context, 8,192 completion tokens, temperature 1.0, no hacking prompt),
and writes every episode the moment it ends: the full conversation, every raw sampled completion
(analysis channel included), the reward, the environment's own metrics (public and private test
rewards where the environment has them) and token counts. Nothing is judged here.

Six streams, one per environment configuration their big run trains on:

  synthetic_partial  their synthetic tasks, partial reward, test files reset after each tool call
  synthetic_binary   the same, all-or-nothing reward
  swe_fixer          real GitHub issues, all-or-nothing reward, no hint, no file names
  ae                 reimplement a library's class, public test files reset before grading
  ioi_noreset        IOI problems, public tests.py left as the model leaves it
  ioi_reset          IOI problems, public tests.py restored before grading

Task selection is mechanical: each stream's task list is drawn from its source file with
`random.Random(--seed)`, streams that share a source get disjoint tasks, and the environment's own
dataset class then orders them with its fixed shuffle. Each task is sampled --group-size times.

Container transport is the local-Docker drop-in in local_docker/ (see its docstring). Run with the
Python of their repository's virtualenv, from this repository's root:
    <their repo>/.venv/bin/python -u -m reward_hacking_rl.screen_held_out ...
The script changes into --repo; a path flag given relative to the directory the command is run from is resolved
against that directory first.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

from reward_hacking_rl.cli_paths import absolute_paths

#: The flags that name a file or folder; main() makes them absolute before it changes into --repo.
PATH_ARGS = ("repo", "out", "synthetic_path", "ae_path")
#: the environment configurations, in the order streams are listed in the output
STREAMS = ("synthetic_partial", "synthetic_binary", "swe_fixer", "ae", "ioi_noreset", "ioi_reset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    req = dict(required=True)
    p.add_argument("--repo", type=Path, help="checkout of tinker-cookbook-reward-hacking", **req)
    p.add_argument("--out", type=Path, help="output directory", **req)
    p.add_argument("--model", type=str, **req)
    p.add_argument("--renderer", type=str, **req)
    p.add_argument("--renderer-current-date", type=str, help="date written into the gpt-oss system prompt", **req)
    p.add_argument(
        "--streams",
        type=str,
        nargs="+",
        choices=STREAMS,
        help="streams whose tasks are drawn, in STREAMS order; the draw sequence depends on this list",
        **req,
    )
    p.add_argument(
        "--run-streams",
        type=str,
        nargs="+",
        choices=STREAMS,
        help="subset of --streams that is built and sampled; drawing all of --streams keeps the "
        "same tasks as a run that sampled every stream",
        **req,
    )
    p.add_argument("--tasks-per-stream", type=int, **req)
    p.add_argument("--group-size", type=int, help="samples per task", **req)
    p.add_argument("--batch-size", type=int, help="tasks whose containers run at once, per stream", **req)
    p.add_argument("--max-steps", type=int, **req)
    p.add_argument("--context-length", type=int, **req)
    p.add_argument("--max-completion-tokens", type=int, **req)
    p.add_argument("--temperature", type=float, **req)
    p.add_argument("--seed", type=int, help="task-selection seed", **req)
    p.add_argument("--synthetic-path", type=Path, **req)
    p.add_argument("--ae-path", type=Path, **req)
    p.add_argument("--image-build-parallelism", type=int, **req)
    p.add_argument("--build-images-only", choices=("yes", "no"), **req)
    return p.parse_args()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if hasattr(x, "model_dump"):
        return x.model_dump()
    return repr(x)


#: candidate tasks drawn per stream, as a multiple of --tasks-per-stream, so tasks whose Docker
#: image no longer builds can be dropped and the stream still filled from the same seeded draw
CANDIDATE_FACTOR = 1.5


def select(pool: list, n: int, rng: random.Random, taken: set[int], dockerfile, failed: set[str] | None) -> list:
    """Draw ceil(CANDIDATE_FACTOR * n) candidates with rng; with `failed` given, keep the first n
    whose Dockerfile is not in it. The draw is identical with or without `failed`."""
    n_draw = -(-int(CANDIDATE_FACTOR * n * 10) // 10)
    free = [i for i in range(len(pool)) if i not in taken]
    if len(free) < n_draw:
        raise ValueError(f"need {n_draw} candidates, only {len(free)} left in a pool of {len(pool)}")
    chosen = rng.sample(free, n_draw)
    taken.update(chosen)
    cands = [pool[i] for i in chosen]
    if failed is None:
        return cands
    ok = [d for d in cands if dockerfile(d) not in failed]
    if len(ok) < n:
        raise ValueError(f"only {len(ok)} of {n_draw} candidates have a buildable image; need {n}")
    return ok[:n]


def single_expected_output(datapoint) -> bool:
    """True when every test of an IOI datapoint expects the same output. In open-r1/ioi-test-cases
    the Communication problems graded by an interactor (Stations, Coreputer, Longest Trip, ...)
    store the placeholder "1" for every test; graded by plain stdout comparison, as this
    environment does, `print(1)` passes them all, so they are not tasks."""
    return len({t.output.strip() for t in datapoint.tests}) == 1


def build_streams(args, renderer, tokenizer, failed: set[str] | None):
    """Each stream: (dataset object, task-id function, dockerfile function). Configs copy
    all_envs.py / big_run.py field by field."""
    from scalable_docker.client import ScalableDockerClient
    from tinker_cookbook.rl.envs import ae_env, bash_codeforces_env, swe_fixer_env, synthetic_env

    n = args.tasks_per_stream
    rng = random.Random(args.seed)
    common = dict(batch_size=args.batch_size, group_size=args.group_size, renderer=renderer, tokenizer=tokenizer)
    out = {}

    synth_pool = synthetic_env.load_synthetic_env_dataset(str(args.synthetic_path), shuffle_seed=42)
    synth_taken: set[int] = set()
    for name, partial in (("synthetic_partial", True), ("synthetic_binary", False)):
        if name not in args.streams:
            continue
        cfg = synthetic_env.SyntheticEnvConfig(
            context_length=args.context_length,
            max_tokens=args.max_completion_tokens,
            max_steps=args.max_steps,
            reward_hacking_prompt="nothing",
            partial_rewards=partial,
            logging_tag=name,
            reset_test_files=True,
        )
        data = select(synth_pool, n, rng, synth_taken, lambda d: d.dockerfile_content, failed)
        out[name] = (
            synthetic_env.SyntheticEnvDataset(data=data, cfg=cfg, **common),
            lambda d: d.task_id,
            lambda d: d.dockerfile_content,
        )

    if "swe_fixer" in args.streams:
        pool = swe_fixer_env.load_swe_fixer_dataset(str(args.repo / "data/swe-fixer.json"), shuffle_seed=42)
        cfg = swe_fixer_env.SWEFixerEnvConfig(
            max_tokens=args.max_completion_tokens,
            context_length=args.context_length,
            can_finish=True,
            show_hint=False,
            show_modified_file_names=False,
            reward_hacking_prompt="nothing",
            max_steps=args.max_steps,
            partial_reward=False,
            max_prompt_tokens=args.context_length - args.max_completion_tokens - 16,
        )
        data = select(pool, n, rng, set(), swe_fixer_env.dockerfile_content, failed)
        out["swe_fixer"] = (
            swe_fixer_env.SWEFixerDataset(
                data=data, cfg=cfg, scalable_docker_client=ScalableDockerClient(key="swe_fixer"), **common
            ),
            lambda d: f"{d.repo}@{d.base_commit[:12]}:{sha(d.problem_statement)}",
            swe_fixer_env.dockerfile_content,
        )

    if "ae" in args.streams:
        pool = ae_env.load_ae_dataset_from_json(str(args.ae_path), max_datapoints=None, shuffle_seed=42)
        cfg = ae_env.AEEnvConfig(
            reward_hacking_prompt="nothing",
            max_steps=args.max_steps,
            max_tests_per_split=100,
            public_test_weight_in_reward=1.0,
            qwen3_disable_thinking=False,
            reset_public_tests=True,
            mitigate_assignment_imports=False,
            max_tokens=args.max_completion_tokens,
            context_length=args.context_length - 16,
        )
        data = select(pool, n, rng, set(), lambda d: ae_env.get_dockerfile_content(d, agent_graded=False), failed)
        out["ae"] = (
            ae_env.AEDataset(data=data, cfg=cfg, docker_key="ae_env", **common),
            lambda d: sha(d.prompt),
            lambda d: ae_env.get_dockerfile_content(d, agent_graded=False),
        )

    ioi_pool = None
    ioi_taken: set[int] = set()
    for name, reset in (("ioi_noreset", False), ("ioi_reset", True)):
        if name not in args.streams:
            continue
        if ioi_pool is None:
            ioi_pool = [
                d for d in bash_codeforces_env.load_ioi_dataset(shuffle_seed=42) if not single_expected_output(d)
            ]
        cfg = bash_codeforces_env.BashCodeforcesEnvConfig(
            reward_hacking_prompt="nothing",
            max_steps=args.max_steps,
            max_tests_per_split=8,
            reset_public_tests=reset,
            public_test_weight_in_reward=1.0,
            qwen3_disable_thinking=False,
            max_tokens=args.max_completion_tokens,
            context_length=args.context_length - 16,
        )
        data = select(ioi_pool, n, rng, ioi_taken, lambda d: bash_codeforces_env.DOCKERFILE_CONTENT, failed)
        out[name] = (
            bash_codeforces_env.BashCodeforcesDataset(data=data, cfg=cfg, docker_key="bash_codeforces", **common),
            lambda d: sha(d.problem_statement),
            lambda d: bash_codeforces_env.DOCKERFILE_CONTENT,
        )
    return out


async def run_stream(name, dataset, task_id, policy, tokenizer, out_file, samples_file, done_keys, lock, args):
    from tinker_cookbook.rl.rollouts import do_single_rollout

    n_batches = args.tasks_per_stream // args.batch_size
    for b in range(n_batches):
        builders = dataset.get_batch(b)
        todo = []
        for g, builder in enumerate(builders):
            tid = task_id(builder.datapoint)
            envs = await builder.make_envs()
            for s, env in enumerate(envs):
                if (name, tid, s) not in done_keys:
                    todo.append((tid, s, env))
        if not todo:
            continue

        async def one(tid, s, env):
            t0 = time.time()
            row = {"stream": name, "task_id": tid, "sample": s, "batch": b}
            turn = 0

            async def persisted_policy(ob, stop):
                # every sampled completion is on disk before the environment acts on it
                nonlocal turn
                ac = await policy(ob, stop)
                rec = {
                    "stream": name,
                    "task_id": tid,
                    "sample": s,
                    "turn": turn,
                    "prompt_tokens": ob.length,
                    "completion": tokenizer.decode(ac.tokens),
                }
                turn += 1
                async with lock:
                    samples_file.write(json.dumps(rec) + "\n")
                    samples_file.flush()
                return ac

            try:
                traj = await do_single_rollout(persisted_policy, env)
                row.update(
                    status="ok",
                    reward=sum(t.reward for t in traj.transitions),
                    metrics=jsonable(traj.transitions[-1].metrics if traj.transitions else {}),
                    raw_completions=[tokenizer.decode(t.ac.tokens) for t in traj.transitions],
                    prompt_tokens=[t.ob.length for t in traj.transitions],
                    sampled_tokens=[len(t.ac.tokens) for t in traj.transitions],
                )
            except Exception:
                row.update(status="error", error=traceback.format_exc())
            row["messages"] = jsonable(getattr(env, "all_messages", None))
            row["seconds"] = round(time.time() - t0, 1)
            async with lock:
                out_file.write(json.dumps(row) + "\n")
                out_file.flush()
            print(
                f"[{name}] batch {b + 1}/{n_batches} task {tid} sample {s}: {row.get('status')} reward={row.get('reward')}",
                flush=True,
            )

        await asyncio.gather(*[one(*t) for t in todo])
        await dataset.scalable_docker_client.start_destroying_containers()
        await dataset.scalable_docker_client.wait_until_all_containers_stopped()


async def main() -> None:
    args = parse_args()
    absolute_paths(args, PATH_ARGS)
    if args.tasks_per_stream % args.batch_size:
        raise ValueError("--tasks-per-stream must be a multiple of --batch-size")
    # This repository's local-Docker client must win over any installed copy, so it goes first on the
    # path before anything imports scalable_docker.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "local_docker"))
    sys.path.insert(1, str(args.repo))
    os.chdir(args.repo)  # their loaders read data/ relative to the repo root; the path flags are absolute by now
    import tinker
    from scalable_docker.client import delete_all_scalable_docker_kubernetes_deployments
    from tinker_cookbook import renderers
    from tinker_cookbook.completers import TinkerTokenCompleter
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(args.model)
    renderer = renderers.get_renderer(args.renderer, tokenizer=tokenizer)
    renderer.current_date = args.renderer_current_date
    from scalable_docker import client as sd

    if not set(args.run_streams) <= set(args.streams):
        raise ValueError("--run-streams must be a subset of --streams")
    candidates = {
        k: v for k, v in build_streams(args, renderer, tokenizer, failed=None).items() if k in args.run_streams
    }
    dockerfiles = sorted({dockerfile(d) for ds, _, dockerfile in candidates.values() for d in ds.data})
    args.out.mkdir(parents=True, exist_ok=True)
    build_root = os.environ["RH_SCREEN_BUILD_ROOT"]
    sem = asyncio.Semaphore(args.image_build_parallelism)
    failures: dict[str, str] = {}

    async def build(c: str) -> None:
        async with sem:
            if await sd.image_exists(c):
                return
            try:
                await sd.build_image(c, build_root)
            except RuntimeError as e:
                failures[c] = str(e)[-4000:]

    print(f"building {len(dockerfiles)} distinct images for the candidate tasks", flush=True)
    await asyncio.gather(*[build(c) for c in dockerfiles])
    print(f"images ready; {len(failures)} of {len(dockerfiles)} failed to build", flush=True)
    with open(args.out / "image_build_failures.jsonl", "w") as f:
        for c, err in failures.items():
            f.write(json.dumps({"image": sd.image_name(c), "dockerfile": c, "error_tail": err}) + "\n")

    streams = {
        k: v for k, v in build_streams(args, renderer, tokenizer, failed=set(failures)).items() if k in args.run_streams
    }
    dropped = {
        name: sum(dockerfile(d) in failures for d in cds.data) for name, (cds, _, dockerfile) in candidates.items()
    }
    selection = {
        name: {"tasks": [task_id(d) for d in ds.data], "candidates_with_failed_image": dropped[name]}
        for name, (ds, task_id, _) in streams.items()
    }
    (args.out / "selection.json").write_text(json.dumps(selection, indent=1))

    if args.build_images_only == "yes":
        return

    rows_path = args.out / "rows.jsonl"
    done_keys = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "ok":
                done_keys.add((r["stream"], r["task_id"], r["sample"]))
    run_cfg = {k: str(v) for k, v in vars(args).items()}
    run_cfg["started"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_cfg["resumed_with_done_rows"] = len(done_keys)
    with open(args.out / "run_config.jsonl", "a") as f:
        f.write(json.dumps(run_cfg) + "\n")

    await delete_all_scalable_docker_kubernetes_deployments()
    sampling_client = tinker.ServiceClient().create_sampling_client(base_model=args.model)
    policy = TinkerTokenCompleter(sampling_client, max_tokens=args.max_completion_tokens, temperature=args.temperature)
    lock = asyncio.Lock()
    with open(rows_path, "a") as out_file, open(args.out / "samples.jsonl", "a") as samples_file:
        await asyncio.gather(
            *[
                run_stream(name, ds, task_id, policy, tokenizer, out_file, samples_file, done_keys, lock, args)
                for name, (ds, task_id, _) in streams.items()
            ]
        )
    await delete_all_scalable_docker_kubernetes_deployments()
    print("done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
