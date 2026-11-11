"""Reward-hacking RL: RL on the reward-hackable coding environments of
astOwOlfo/tinker-cookbook-reward-hacking, started from a given LoRA checkpoint
(--load-checkpoint-path: weights only, fresh optimizer) or from the base model (`none`).

Their `big_run.py` recipe with these stated changes:
- model gpt-oss-20b (theirs 120b), learning rate 1e-4 (theirs 4e-5),
- up to 32 tool calls per episode (theirs 9; sampling the base model on these environments before
  any training, in screen_held_out.py, showed that 9 starves gpt-oss-20b),
- one training sequence per episode via extending.ExtendingRenderer (theirs splits every turn),
- the 8 IOI problems whose tests all expect one constant output dropped,
- two data files that big_run names are not in their repository, so two files that are tracked
  there are passed instead (data/final-harder-merge-smaller.jsonl as --synthetic-path, data/ae.json
  as --ae-path; see SETUP.md),
- their 1-group style environment left out,
- the tasks selected by screen_held_out.py held out of training.

Per update, in groups of --group-size samples: synthetic partial reward, synthetic all-or-nothing,
SWE-Fixer, AE (8 groups each) and IOI with and without the public-test reset (4 each), as in
big_run. Every sampled turn goes to samples.jsonl and every finished episode to rows.jsonl.

Seeds: --seed reorders each stream's training tasks (the same tasks, a different order) with
random.Random(f"{seed}:{stream}"); --seed 0 keeps the loaders' order. --environment-random says where
the environments' own draws (public tests, prompts) come from:
  unseeded    Python's global random, left unseeded (the seed-0 runs);
  global      Python's global random, seeded with --seed (the runs at seeds 1000 and 2000);
  per-module  the global random seeded with --seed, and each environment module given its own
              generator seeded with f"{seed}:env:{module}" (every run with something mixed in, so the
              extra sampling requests cannot shift the coding environments' draws).
Tinker's sampling is not seeded by any flag; it differs between runs regardless.

Mixed in: --mixed-in adds --mixed-in-groups one-turn groups per update, prompts from --mixed-in-pool,
scored by mixed_in/judge_worker.py: `constitution` (constitution questions graded by the constitution
judge of --judge-constitution) or `chat` (No Robots requests graded by the helpfulness judge, which
reads no constitution); `none` is reward-hacking RL alone. Its prompt order uses its own seeded random
and the mix concatenates each stream's batch, so the coding tasks per update are unchanged.

Sync training (no async_config), KL measured every 8 steps but not penalised (big_run: coef 0),
constant-reward groups dropped, LoRA rank 32, a checkpoint every 5 updates. Run with the Python of
their repository's virtualenv (tinker 0.28.1 installed), from this repository's root:
    <their repo>/.venv/bin/python -u -m reward_hacking_rl.train ...
The trainer changes into --repo; a path flag given relative to the directory the command is run from is resolved
against that directory first.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

from reward_hacking_rl.cli_paths import absolute_paths

#: groups per update per stream, and pool-repetition factor so each has >= max_steps batches.
#: mirrors big_run: synthetic 16 (8 partial + 8 binary), swe 8, ae 8, ioi 8 (4 + 4).
STREAM_GROUPS = {
    "synthetic_partial": 8,
    "synthetic_binary": 8,
    "swe_fixer": 8,
    "ae": 8,
    "ioi_noreset": 4,
    "ioi_reset": 4,
}

#: distinct training tasks per stream (before repetition for batching). Caps the image build:
#: SWE-Fixer and AE build a ~1 GB image per task (clone + install), so the full pools (thousands)
#: neither fit the disk nor finish building in reasonable time. The first N of each seeded,
#: held-out-excluded pool are kept; IOI shares one image so it is left uncapped.
MAX_TRAIN_TASKS = {
    "synthetic_partial": 300,
    "synthetic_binary": 300,
    "swe_fixer": 150,
    "ae": 150,
    "ioi_noreset": 10_000,
    "ioi_reset": 10_000,
}


#: --mixed-in value -> the stream name written to every record. The stream name also seeds the mixed-in
#: prompt order (judged.JudgedDataset), so these strings are part of the recipe.
MIXED_IN_STREAM = {"none": None, "constitution": "constqa", "chat": "chat"}

#: This repository's root: the judge worker runs from here as a module.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The flags that name a file or folder. main() makes them absolute before it changes into --repo, so a path given
#: relative to the directory the command was run from keeps its meaning.
PATH_ARGS = (
    "repo",
    "out",
    "held_out_selection",
    "synthetic_path",
    "ae_path",
    "mixed_in_pool",
    "judge_python",
    "reference_failures",
    "dump_batches",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    r = dict(required=True)
    p.add_argument("--repo", type=Path, **r)
    p.add_argument("--out", type=Path, help="output folder: records, checkpoints, config", **r)
    p.add_argument(
        "--held-out-selection", type=Path, help="screen_held_out.py's selection.json; its tasks never train", **r
    )
    p.add_argument("--model", type=str, **r)
    p.add_argument("--renderer", type=str, **r)
    p.add_argument("--renderer-current-date", type=str, **r)
    p.add_argument("--seed", type=int, help="task-order seed; 0 = the loaders' order", **r)
    p.add_argument(
        "--environment-random",
        choices=("unseeded", "global", "per-module"),
        help="source of the environments' own random draws (see the module docstring)",
        **r,
    )
    p.add_argument("--learning-rate", type=float, **r)
    p.add_argument("--group-size", type=int, **r)
    p.add_argument("--max-steps", type=int, help="number of updates", **r)
    p.add_argument("--max-tool-calls", type=int, **r)
    p.add_argument("--context-length", type=int, **r)
    p.add_argument("--max-completion-tokens", type=int, **r)
    p.add_argument("--temperature", type=float, **r)
    p.add_argument("--save-every", type=int, **r)
    p.add_argument("--lora-rank", type=int, **r)
    p.add_argument("--load-checkpoint-path", type=str, help="Tinker training-state path to start from, or none", **r)
    p.add_argument("--synthetic-path", type=Path, **r)
    p.add_argument("--ae-path", type=Path, **r)
    p.add_argument("--image-build-parallelism", type=int, **r)
    p.add_argument("--build-images-only", choices=("yes", "no"), **r)
    p.add_argument("--mixed-in", choices=tuple(MIXED_IN_STREAM), **r)
    p.add_argument("--mixed-in-pool", type=str, help="jsonl of {id, question}, or none", **r)
    p.add_argument("--mixed-in-groups", type=int, help="one-turn groups per update (0 with --mixed-in none)", **r)
    p.add_argument(
        "--judge-constitution",
        type=str,
        help="constitution read by the constitution judge (a name its judge() accepts), or none",
        **r,
    )
    p.add_argument("--judge-model", type=str, **r)
    p.add_argument("--judge-concurrency", type=int, help="judge calls in flight per group", **r)
    p.add_argument(
        "--judge-python",
        type=str,
        help="Python of this repository's own virtualenv, which runs mixed_in/judge_worker.py, or none",
        **r,
    )
    p.add_argument(
        "--reference-failures",
        type=str,
        help="image_build_failures.jsonl of the run whose coding batches "
        "this one must match, or none; a different failure set stops the run",
        **r,
    )
    p.add_argument(
        "--dump-batches", type=str, help="write each update's coding groups to this path and exit, or none", **r
    )
    p.add_argument(
        "--verify-one-datum",
        choices=("yes", "no"),
        help="roll out one episode through the recording wrapper, run trajectory_to_data, "
        "print the datum count, and exit; a count above 1 means the episode would "
        "train as many sequences and the run must not proceed",
        **r,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    absolute_paths(args, PATH_ARGS)
    # This folder's local-Docker client (it retries refused container starts) must win over any
    # installed copy, so it goes first on the path before anything imports scalable_docker.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "local_docker"))
    sys.path.insert(1, str(args.repo))
    from reward_hacking_rl import extending
    from reward_hacking_rl.mixed_in import judged

    os.chdir(args.repo)
    import tinker
    from scalable_docker.client import Image, ScalableDockerClient
    from tinker_cookbook import renderers
    from tinker_cookbook.rl import train
    from tinker_cookbook.rl.envs import ae_env, bash_codeforces_env, swe_fixer_env, synthetic_env
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    held = json.loads(args.held_out_selection.read_text())
    held_tasks = {k: set(v["tasks"]) for k, v in held.items()}
    tok = get_tokenizer(args.model)
    renderer = renderers.get_renderer(args.renderer, tokenizer=tok)
    renderer.current_date = args.renderer_current_date
    args.out.mkdir(parents=True, exist_ok=True)
    recorder = extending.Recorder(args.out, tok)

    def sha(t: str) -> str:
        import hashlib

        return hashlib.sha256(t.encode()).hexdigest()[:16]

    # task-id functions must match screen_held_out.py's, so held-out exclusion lines up exactly
    task_id = {
        "synthetic_partial": lambda d: d.task_id,
        "synthetic_binary": lambda d: d.task_id,
        "swe_fixer": lambda d: f"{d.repo}@{d.base_commit[:12]}:{sha(d.problem_statement)}",
        "ae": lambda d: sha(d.prompt),
        "ioi_noreset": lambda d: sha(d.problem_statement),
        "ioi_reset": lambda d: sha(d.problem_statement),
    }

    if args.environment_random != "unseeded":
        random.seed(args.seed)

    # The coding environments draw public-test splits and prompts from the `random` module, which tinker's
    # retry jitter also consumes; with the judged stream's extra sampling requests that would shift the
    # draws. Each environment module gets its own seeded generator instead, so its draws depend only on the
    # seed and the order its environments are built in.
    # They use random.shuffle, random.choice and random.Random (their dataset loaders), so the stand-in is a
    # seeded generator that still carries the Random class.
    class EnvRandom(random.Random):
        Random = random.Random

    if args.environment_random == "per-module":
        for mod in (synthetic_env, swe_fixer_env, ae_env, bash_codeforces_env):
            mod.random = EnvRandom(f"{args.seed}:env:{mod.__name__}")

    def order(data: list, stream: str) -> list:
        """The same tasks; for seed 0 in the loaders' order, otherwise reordered by the seed."""
        if args.seed == 0:
            return data
        data = list(data)
        random.Random(f"{args.seed}:{stream}").shuffle(data)
        return data

    def repeat_to(data: list, batch_size: int) -> list:
        """Repeat the pool so it yields >= max_steps batches (their datasets cycle modulo len)."""
        need = args.max_steps * batch_size
        reps = -(-need // max(1, len(data)))
        return data * max(1, reps)

    synth_pool = [d for d in synthetic_env.load_synthetic_env_dataset(str(args.synthetic_path), shuffle_seed=42)]
    swe_pool = swe_fixer_env.load_swe_fixer_dataset(str(args.repo / "data/swe-fixer.json"), shuffle_seed=42)
    ae_pool = ae_env.load_ae_dataset_from_json(str(args.ae_path), max_datapoints=None, shuffle_seed=42)
    ioi_pool = [
        d
        for d in bash_codeforces_env.load_ioi_dataset(shuffle_seed=42)
        if len({t.output.strip() for t in d.tests}) != 1
    ]

    def exclude(pool, stream):
        """Drop held-out tasks, then keep the first MAX_TRAIN_TASKS distinct by task id (the pools
        are already seed-shuffled by their loaders, so this is a fixed arbitrary subset)."""
        held_s = held_tasks.get(stream, set())
        kept, seen = [], set()
        for d in pool:
            tid = task_id[stream](d)
            if tid in held_s or tid in seen:
                continue
            seen.add(tid)
            kept.append(d)
            if len(kept) >= MAX_TRAIN_TASKS[stream]:
                break
        return kept

    datasets = []
    all_dockerfiles: set[str] = set()
    common = dict(group_size=args.group_size, renderer=renderer, tokenizer=tok)

    for name, partial in (("synthetic_partial", True), ("synthetic_binary", False)):
        cfg = synthetic_env.SyntheticEnvConfig(
            context_length=args.context_length,
            max_tokens=args.max_completion_tokens,
            max_steps=args.max_tool_calls,
            reward_hacking_prompt="nothing",
            partial_rewards=partial,
            logging_tag=name,
            reset_test_files=True,
        )
        data = repeat_to(order(exclude(synth_pool, name), name), STREAM_GROUPS[name])
        all_dockerfiles |= {d.dockerfile_content for d in data}
        inner = synthetic_env.SyntheticEnvDataset(data=data, batch_size=STREAM_GROUPS[name], cfg=cfg, **common)
        datasets.append(extending.WrappedDataset(inner, recorder, name, task_id[name]))

    cfg = swe_fixer_env.SWEFixerEnvConfig(
        max_tokens=args.max_completion_tokens,
        context_length=args.context_length,
        can_finish=True,
        show_hint=False,
        show_modified_file_names=False,
        reward_hacking_prompt="nothing",
        max_steps=args.max_tool_calls,
        partial_reward=False,
        max_prompt_tokens=args.context_length - args.max_completion_tokens - 16,
    )
    data = repeat_to(order(exclude(swe_pool, "swe_fixer"), "swe_fixer"), STREAM_GROUPS["swe_fixer"])
    all_dockerfiles |= {swe_fixer_env.dockerfile_content(d) for d in data}
    inner = swe_fixer_env.SWEFixerDataset(
        data=data,
        batch_size=STREAM_GROUPS["swe_fixer"],
        cfg=cfg,
        scalable_docker_client=ScalableDockerClient(key="swe_fixer"),
        **common,
    )
    datasets.append(extending.WrappedDataset(inner, recorder, "swe_fixer", task_id["swe_fixer"]))

    cfg = ae_env.AEEnvConfig(
        reward_hacking_prompt="nothing",
        max_steps=args.max_tool_calls,
        max_tests_per_split=100,
        public_test_weight_in_reward=1.0,
        qwen3_disable_thinking=False,
        reset_public_tests=True,
        mitigate_assignment_imports=False,
        max_tokens=args.max_completion_tokens,
        context_length=args.context_length - 16,
    )
    data = repeat_to(order(exclude(ae_pool, "ae"), "ae"), STREAM_GROUPS["ae"])
    all_dockerfiles |= {ae_env.get_dockerfile_content(d, agent_graded=False) for d in data}
    inner = ae_env.AEDataset(data=data, batch_size=STREAM_GROUPS["ae"], cfg=cfg, docker_key="ae_env", **common)
    datasets.append(extending.WrappedDataset(inner, recorder, "ae", task_id["ae"]))

    for name, reset in (("ioi_noreset", False), ("ioi_reset", True)):
        cfg = bash_codeforces_env.BashCodeforcesEnvConfig(
            reward_hacking_prompt="nothing",
            max_steps=args.max_tool_calls,
            max_tests_per_split=8,
            reset_public_tests=reset,
            public_test_weight_in_reward=1.0,
            qwen3_disable_thinking=False,
            max_tokens=args.max_completion_tokens,
            context_length=args.context_length - 16,
        )
        data = repeat_to(order(exclude(ioi_pool, name), name), STREAM_GROUPS[name])
        all_dockerfiles |= {bash_codeforces_env.DOCKERFILE_CONTENT}
        inner = bash_codeforces_env.BashCodeforcesDataset(
            data=data, batch_size=STREAM_GROUPS[name], cfg=cfg, docker_key="bash_codeforces", **common
        )
        datasets.append(extending.WrappedDataset(inner, recorder, name, task_id[name]))

    dockerfile_of = {
        "synthetic_partial": lambda d: d.dockerfile_content,
        "synthetic_binary": lambda d: d.dockerfile_content,
        "swe_fixer": swe_fixer_env.dockerfile_content,
        "ae": lambda d: ae_env.get_dockerfile_content(d, agent_graded=False),
        "ioi_noreset": lambda d: bash_codeforces_env.DOCKERFILE_CONTENT,
        "ioi_reset": lambda d: bash_codeforces_env.DOCKERFILE_CONTENT,
    }
    reference_failures = (
        None
        if args.reference_failures == "none"
        else {
            json.loads(line)["image_sha"]
            for line in Path(args.reference_failures).read_text().splitlines()
            if line.strip()
        }
    )

    if args.dump_batches != "none":
        # The coding batches this seed would train on, with the reference run's failed images dropped as that
        # run dropped them; nothing is built or sampled. Compared against the reference run's rows.jsonl.
        if reference_failures is None:
            raise SystemExit("--dump-batches needs --reference-failures")
        with open(args.dump_batches, "w") as f:
            for wrapped in datasets:
                fn = dockerfile_of[wrapped.stream]
                wrapped.inner.data = [d for d in wrapped.inner.data if sha(fn(d)) not in reference_failures]
            for i in range(args.max_steps):
                groups = [[w.stream, task_id[w.stream](b.datapoint)] for w in datasets for b in w.inner.get_batch(i)]
                f.write(json.dumps({"iteration": i, "groups": groups}) + "\n")
        print(f"wrote {args.max_steps} batches to {args.dump_batches}", flush=True)
        return

    mixed_in_stream = MIXED_IN_STREAM[args.mixed_in]

    def build_judged():
        """The judged one-turn stream (judged.py), its judge worker started and self-tested."""
        if args.mixed_in_groups <= 0 or args.mixed_in_pool == "none" or args.judge_python == "none":
            raise SystemExit("--mixed-in needs a pool, a positive --mixed-in-groups and --judge-python")
        kind, constitution = (
            ("constitution", args.judge_constitution) if args.mixed_in == "constitution" else ("helpfulness", "none")
        )
        if (kind == "constitution") != (constitution != "none"):
            raise SystemExit("--mixed-in constitution takes a --judge-constitution; chat takes none")
        worker = judged.JudgeWorker(
            [
                args.judge_python,
                "-u",
                "-m",
                "reward_hacking_rl.mixed_in.judge_worker",
                "--kind",
                kind,
                "--constitution",
                constitution,
                "--judge-model",
                args.judge_model,
                "--concurrency",
                str(args.judge_concurrency),
                "--verdicts",
                str(args.out / "verdicts.jsonl"),
                "--renderer",
                args.renderer,
            ],
            cwd=REPO_ROOT,
            log_path=args.out / "judge_worker.log",
        )
        pool = [json.loads(line) for line in Path(args.mixed_in_pool).read_text().splitlines() if line.strip()]
        inner = judged.JudgedDataset(
            pool,
            args.mixed_in_groups,
            args.max_steps,
            args.seed,
            mixed_in_stream,
            args.group_size,
            renderer,
            tok,
            worker,
            args.out,
        )
        print(
            f"judged stream {mixed_in_stream}: {args.mixed_in_groups} groups per update from {len(pool)} prompts, "
            f"{kind} judge",
            flush=True,
        )
        return extending.WrappedDataset(inner, recorder, mixed_in_stream, lambda d: d["id"])

    if args.verify_one_datum == "yes":
        # Build only the first synthetic group's images, roll out one episode, and check it becomes
        # exactly one training datum. No full build, so the check is quick.
        from tinker_cookbook.completers import TinkerTokenCompleter
        from tinker_cookbook.rl.data_processing import trajectory_to_data
        from tinker_cookbook.rl.rollouts import do_single_rollout

        builder = datasets[0].get_batch(0)[0]  # synthetic_partial, first group
        dfs = sorted({b.inner.datapoint.dockerfile_content for b in datasets[0].get_batch(0)})
        asyncio.run(ScalableDockerClient(key="build").build_images([Image(c) for c in dfs], batch_size=8))
        env = asyncio.run(builder.make_envs())[0]
        policy = TinkerTokenCompleter(
            tinker.ServiceClient().create_sampling_client(base_model=args.model),
            max_tokens=args.max_completion_tokens,
            temperature=args.temperature,
        )
        traj = asyncio.run(do_single_rollout(policy, env))
        n = len(trajectory_to_data(traj, 1.0))
        print(json.dumps({"turns": len(traj.transitions), "datums": n}), flush=True)
        if n != 1:
            raise SystemExit(f"episode produced {n} datums; one-sequence fix is not holding, aborting")
        if args.mixed_in != "none":
            # One real judged group end to end: sample, record, judge, rewards back.
            group = build_judged().get_batch(0)[0]

            async def one_group():
                envs = await group.make_envs()
                trajs = await asyncio.gather(*[do_single_rollout(policy, e) for e in envs])
                return trajs, await group.compute_group_rewards(list(trajs), envs)

            trajs, got = asyncio.run(one_group())
            print(
                json.dumps(
                    {
                        "judged_group": group.task_id,
                        "rewards": [r for r, _ in got],
                        "metrics": got[0][1],
                        "datums": [len(trajectory_to_data(t, 1.0)) for t in trajs],
                    }
                ),
                flush=True,
            )
        return

    # Build images, tolerating per-image failures (some tasks' Dockerfiles no longer compile, e.g. a
    # package built from source). Failed-image tasks are then dropped from their stream, as
    # screen_held_out.py does, rather than killing the run. The failures are recorded.
    from scalable_docker.client import build_image as _build_image
    from scalable_docker.client import image_exists as _image_exists

    build_root = os.environ["RH_SCREEN_BUILD_ROOT"]
    cap = args.image_build_parallelism
    sem = asyncio.Semaphore(cap)
    failures: dict[str, str] = {}

    async def build_one(c: str) -> None:
        async with sem:
            if await _image_exists(c):
                return
            try:
                await _build_image(c, build_root)
            except Exception as e:  # noqa: BLE001 - a build failure drops its tasks, never aborts
                failures[c] = str(e)[-2000:]

    async def build_all() -> None:
        await asyncio.gather(*[build_one(c) for c in sorted(all_dockerfiles)])

    print(f"building {len(all_dockerfiles)} distinct images", flush=True)
    asyncio.run(build_all())
    print(f"images ready; {len(failures)} of {len(all_dockerfiles)} failed to build", flush=True)
    (args.out / "image_build_failures.jsonl").write_text(
        "".join(json.dumps({"image_sha": sha(c), "error_tail": e}) + "\n" for c, e in failures.items())
    )
    if args.build_images_only == "yes":
        return
    if reference_failures is not None and {sha(c) for c in failures} != reference_failures:
        # A different set of failed images drops different tasks and shifts every later batch, so this run
        # would no longer train on the reference run's coding tasks.
        got = {sha(c) for c in failures}
        raise SystemExit(
            f"image build failures differ from the reference run: only here {sorted(got - reference_failures)}, "
            f"only there {sorted(reference_failures - got)}"
        )

    for wrapped in datasets:
        fn = dockerfile_of[wrapped.stream]
        inner = wrapped.inner
        inner.data = [d for d in inner.data if fn(d) not in failures]
        batches = len(inner.data) // STREAM_GROUPS[wrapped.stream]
        if batches < args.max_steps:
            raise SystemExit(
                f"{wrapped.stream}: only {batches} batches after dropping failed-image tasks, "
                f"need {args.max_steps}; raise MAX_TRAIN_TASKS for it"
            )

    if args.mixed_in != "none":
        datasets.append(build_judged())

    mix = extending.MixDataset(datasets, n_batches=args.max_steps)

    cfg = train.Config(
        learning_rate=args.learning_rate,
        dataset_builder=extending.FixedDatasetBuilder(mix),
        model_name=args.model,
        max_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        lora_rank=args.lora_rank,
        load_checkpoint_path=None if args.load_checkpoint_path == "none" else args.load_checkpoint_path,
        kl_penalty_coef=0.0,
        compute_kl_penalty_every=8,
        remove_constant_reward_groups=True,
        eval_every=0,
        save_every=args.save_every,
        wandb_project=None,
        log_path=str(args.out / "train"),
    )
    asyncio.run(train.main(cfg))


if __name__ == "__main__":
    main()
