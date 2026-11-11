"""Standard coding RL, the control for the hackable reward: RL from a given LoRA checkpoint on
SWE-smith bug fixing through the reward-hacking repository's bash agent, with every setting of
reward-hacking RL (reward_hacking_rl/train.py) except the tasks.

Same as those runs: model, renderer and its date, learning rate, group size, number of updates,
tool calls per episode, context and completion lengths, temperature, LoRA rank, checkpoint
cadence, weights-only start from --load-checkpoint-path, one training sequence per episode
(extending.py), sync training, KL measured every 8 steps and not penalised, constant-reward groups
dropped, and 40 groups per update (their six streams: 8+8+8+8+4+4).

Different: all 40 groups are SWE-smith tasks (swe_smith_env.py), one distinct task per group in
pool order, so 50 updates use 2,000 tasks and none repeats. Tasks come from build_pool.py's pool,
restricted to the ones the container self-test passed (selftest.py). Containers run with no
network.

Seeds: the 2,000 tasks are the same for every seed; --seed sets their order
(random.Random(f"{seed}:swe_smith"), seed 0 = pool order) and seeds Python's global random. Tinker's
sampling is not seeded by any flag.

Every sampled turn goes to samples.jsonl and every finished episode to rows.jsonl. Run with the
Python of the reward-hacking repository's virtualenv, from this repository's root:
    <their repo>/.venv/bin/python -u -m reward_hacking_rl.control.train ...
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

#: The flags that name a file or folder; main() makes them absolute before it changes into --repo.
PATH_ARGS = ("repo", "out", "pool", "selftest_rows")
#: The reward-hacking runs' groups per update, summed over their six streams.
GROUPS_PER_UPDATE = 40
STREAM = "swe_smith"
#: Attempts at `docker run` per container in this control's runs (reward-hacking RL uses the client's 10).
CONTAINER_START_ATTEMPTS = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    r = dict(required=True)
    p.add_argument("--repo", type=Path, **r)
    p.add_argument("--out", type=Path, help="output folder: records, checkpoints, config", **r)
    p.add_argument("--pool", type=Path, help="build_pool.py's pool.jsonl", **r)
    p.add_argument("--selftest-rows", type=Path, help="selftest.py's rows.jsonl; only passed tasks train", **r)
    p.add_argument(
        "--max-grade-seconds",
        type=float,
        help="drop tasks whose self-test buggy or gold grading took longer",
        required=True,
    )
    p.add_argument("--model", type=str, **r)
    p.add_argument("--renderer", type=str, **r)
    p.add_argument("--renderer-current-date", type=str, **r)
    p.add_argument("--seed", type=int, help="task-order and global-random seed; 0 = pool order", **r)
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
    p.add_argument("--setup-seconds-per-command", type=float, **r)
    p.add_argument("--setup-total-seconds", type=float, **r)
    p.add_argument("--test-seconds-per-command", type=float, **r)
    p.add_argument("--test-total-seconds", type=float, **r)
    p.add_argument("--image-build-parallelism", type=int, **r)
    p.add_argument("--build-images-only", choices=("yes", "no"), **r)
    p.add_argument(
        "--verify-one-datum",
        choices=("yes", "no"),
        **r,
        help="roll out one episode through the recording wrapper, run trajectory_to_data, "
        "print the datum count, and exit; a count above 1 means the episode would "
        "train as many sequences and the run must not proceed",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    absolute_paths(args, PATH_ARGS)
    # This repository's local-Docker client (it has the `network` option) must win over any installed copy,
    # so it goes first on the path before anything imports scalable_docker.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_docker"))
    sys.path.insert(1, str(args.repo))
    from reward_hacking_rl import extending
    from reward_hacking_rl.control import swe_smith_env

    os.chdir(args.repo)
    import tinker
    from scalable_docker import client as docker_client
    from scalable_docker.client import Image, MultiCommandTimeout, ScalableDockerClient
    from tinker_cookbook import renderers
    from tinker_cookbook.rl import train
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    docker_client.START_ATTEMPTS = CONTAINER_START_ATTEMPTS
    tok = get_tokenizer(args.model)
    renderer = renderers.get_renderer(args.renderer, tokenizer=tok)
    renderer.current_date = args.renderer_current_date
    args.out.mkdir(parents=True, exist_ok=True)
    recorder = extending.Recorder(args.out, tok)

    selftest = [json.loads(line) for line in args.selftest_rows.open()]
    passed = {
        r["instance_id"]
        for r in selftest
        if r.get("passed") and max(r["gold_grade_s"], r["buggy_grade_s"]) <= args.max_grade_seconds
    }
    pool = [d for d in swe_smith_env.load_pool(str(args.pool)) if d.instance_id in passed]
    need = args.max_steps * GROUPS_PER_UPDATE
    if len(pool) < need:
        raise SystemExit(f"{len(pool)} self-tested tasks, need {need} for {args.max_steps} updates without repeats")
    data = pool[:need]
    random.seed(args.seed)
    if args.seed != 0:
        random.Random(f"{args.seed}:{STREAM}").shuffle(data)
    (args.out / "train_tasks.json").write_text(json.dumps([d.instance_id for d in data]) + "\n")

    cfg_env = swe_smith_env.SWESmithEnvConfig(
        max_tokens=args.max_completion_tokens,
        context_length=args.context_length,
        can_finish=True,
        reward_hacking_prompt="nothing",
        max_steps=args.max_tool_calls,
        max_prompt_tokens=args.context_length - args.max_completion_tokens - 16,
        setup_timeout=MultiCommandTimeout(
            seconds_per_command=args.setup_seconds_per_command, total_seconds=args.setup_total_seconds
        ),
        test_timeout=MultiCommandTimeout(
            seconds_per_command=args.test_seconds_per_command, total_seconds=args.test_total_seconds
        ),
    )
    inner = swe_smith_env.SWESmithDataset(
        data=data,
        batch_size=GROUPS_PER_UPDATE,
        group_size=args.group_size,
        cfg=cfg_env,
        renderer=renderer,
        tokenizer=tok,
        scalable_docker_client=ScalableDockerClient(key=STREAM, network="none"),
    )
    dataset = extending.WrappedDataset(inner, recorder, STREAM, lambda d: d.instance_id)

    images = sorted({swe_smith_env.dockerfile_content(d) for d in data})
    print(f"building {len(images)} images", flush=True)
    asyncio.run(
        ScalableDockerClient(key="build").build_images(
            [Image(c) for c in images], batch_size=args.image_build_parallelism
        )
    )
    if args.build_images_only == "yes":
        return

    if args.verify_one_datum == "yes":
        from tinker_cookbook.completers import TinkerTokenCompleter
        from tinker_cookbook.rl.data_processing import trajectory_to_data
        from tinker_cookbook.rl.rollouts import do_single_rollout

        builder = dataset.get_batch(0)[0]
        env = asyncio.run(builder.make_envs())[0]
        # The base model, as in the reward-hacking runs' check: it tests the episode-to-datum path only.
        policy = TinkerTokenCompleter(
            tinker.ServiceClient().create_sampling_client(base_model=args.model),
            max_tokens=args.max_completion_tokens,
            temperature=args.temperature,
        )
        traj = asyncio.run(do_single_rollout(policy, env))
        n = len(trajectory_to_data(traj, 1.0))
        print(
            json.dumps(
                {"turns": len(traj.transitions), "datums": n, "reward": sum(t.reward for t in traj.transitions)}
            ),
            flush=True,
        )
        if n != 1:
            raise SystemExit(f"episode produced {n} datums; one-sequence fix is not holding, aborting")
        return

    cfg = train.Config(
        learning_rate=args.learning_rate,
        dataset_builder=extending.FixedDatasetBuilder(extending.MixDataset([dataset], n_batches=args.max_steps)),
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
