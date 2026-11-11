"""Online RL round loop: sample and score one round, take one optimizer step, repeat.

The loop is a mechanism; the environment is not. Everything that differs between studies — which
prompts, which reward, which health signals matter — lives behind two boundaries:

  * `--rollout-script` is handed the current sampler and writes `<out>` (rows with exactly
    `GRPO_ROLLOUT_KEYS`) plus `<out>.stats.json`. That is the contract `src.training.rollout`
    already produces, so an environment implements it by calling that function.
  * `--gate key:op:threshold` reads the stats file by key, so a study gates on whatever its own
    rollout measured without this file knowing what those measurements mean.

ON-POLICY by default: `--train-batch-size 0` takes one optimizer step over the whole surviving
batch, so generation and update share a policy and the importance ratio is 1. `importance_sampling`
with no clip thresholds is the correct loss in that regime and the CLI refuses the combination that
is not. A non-zero `--train-batch-size` takes ceil(rows / batch) minibatch steps per round instead,
which is off-policy after the first step and needs a clipped loss.

Gates fire only after `GATE_CONSECUTIVE` consecutive failing rounds: one bad round is batch noise.
A round whose groups all had uniform reward trains nothing and is recorded and skipped, never
treated as a failure — under DR-GRPO a group with no reward spread has zero advantage by
construction. A skipped round still gets its held-out evaluation when the cadence or the last round
calls for one, on the checkpoint the round started from.

PATHS. `--out`, `--rollout-script` and `--heldout-script` are made absolute before any command is
built; a relative one resolves against the directory the command was run from.

ONE OPTIMIZER FOR THE WHOLE RUN. Round 0 creates the LoRA (`fresh`) or loads an init checkpoint's
weights with a new optimizer (`init`, because an SFT run's Adam moments belong to a different
objective). Every later round launches the trainer in `continue` mode from the previous round's
state checkpoint, which restores weights AND optimizer state onto the new round's data, so Adam's
moments persist across rounds exactly as they would inside one long trainer process. A fresh
optimizer taking one step is not Adam: its first step is a fixed learning-rate move in the sign of
the gradient, whatever the betas say.

RESUME is explicit. `--resume false` REFUSES to start when a history already exists, rather than
appending to it while restarting from the base model, which would corrupt the record as well as
waste the spend. `--resume true` continues at the round after the last one recorded, from the
weights of the last round that trained. Skipped and halted rounds stay in the history: they are
facts about the run. A halted round's rollouts were never trained on and are not resampled; the
gate streaks are rebuilt from the rounds after the halt, so continuing past one is a deliberate
override and is printed as such. A round's entry is written the moment its checkpoints exist, so a
held-out evaluation that dies afterwards costs the evaluation, not the round. Resume by
re-running the same command with `--resume true` and the same output directory.

THE COST LINE the dry run prints is the estimate of the whole run, so everything the loop
spends is in it: every rollout completion at `--sampled-tokens` over `--prompt-tokens`, every row
trained at their sum, and every held-out evaluation at `--heldout-completions` completions of the
same shape, at the cadence the rounds imply. `--sampled-tokens` is the expected completion
length; `--max-tokens` is the cap and prices nothing, so a generous cap against truncation does
not inflate the estimate. The
line assumes no truncation, no dropped groups and no early halt.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src import paths
from src.keys import load_project_env
from src.training.artifacts import TRAIN_STEPS_LOG
from src.training.checkpoints import final_sampler, final_state
from src.training.rl_loop_cli import BASE_START, GATE_CONSECUTIVE, parse_args
from src.training.tinker_common import CHECKPOINT_MODE_CONTINUE, CHECKPOINT_MODE_FRESH, CHECKPOINT_MODE_INIT

#: Children run from `paths.HOME`, so a relative data path inside `--rollout-arg` or `--heldout-arg`
#: is read against the home wherever the loop was started. `--out` and the two scripts reach the
#: children as absolute paths (`_absolute_paths`).
REPO_ROOT = paths.HOME

#: A flat rate with no warmup: the schedule is the round loop, one or a few steps per round.
WARMUP_STEPS, LR_SCHEDULE = 0, "constant"
#: Every step logs and every step saves, so a round leaves its final sampler and state whatever
#: its step count; the rollout writes the split the trainer reads; W&B is off because the history
#: file is the record.
LOG_EVERY, SAVE_EVERY, SPLIT = 1, 1, "train"
WANDB_OFF = (
    "--wandb-mode",
    "disabled",
    "--wandb-project",
    "",
    "--wandb-entity",
    "",
    "--wandb-run-name",
    "",
    "--wandb-group",
    "",
    "--wandb-tags",
    "none",
    "--wandb-run-id",
    "",
    "--wandb-resume",
    "none",
)

#: Drift telemetry the trainer already writes per step. In the on-policy regime these ASSERT that
#: the sampler was the policy — a ratio not centred on 1 means it was not — rather than monitor it.
DRIFT_KEYS = ("ratio_mean", "ratio_median", "abs_logdiff_mean", "kl_k3", "frac_r_gt_clip", "loss", "mean_abs_advantage")

HISTORY = "rl_history.json"
RECIPE = "recipe.json"
#: What a held-out script must write beside its summary: one row per sampled response, so the
#: checkpoint's behaviour can be re-scored or read later without sampling it again.
HELDOUT_ROWS = "rows.jsonl"
HELDOUT_SUMMARY = "summary.json"


def _run(cmd: list[str], label: str, log: Path) -> None:
    print(f"  -> {label}", flush=True)
    with log.open("w") as sink:
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=sink, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = "\n".join(log.read_text().splitlines()[-20:])
        raise RuntimeError(f"{label} failed with code {result.returncode}:\n{tail}")


def checkpoint_train_unembed(state_path: str) -> bool:
    """What train_unembed the service will use when loading this state. In init and continue
    modes the flag is inert (the architecture is read off the checkpoint), so its only effect is
    the value recorded in config.json; reading the real value turns a wrong record into a
    startup error."""
    import asyncio

    import tinker

    from src.training.tinker_common import owner_tag

    async def fetch() -> bool:
        service = tinker.ServiceClient(user_metadata=owner_tag())
        info = await service._get_rest_client_for_weights(None).get_weights_info_by_tinker_path(state_path)
        return True if info.train_unembed is None else bool(info.train_unembed)

    return asyncio.run(fetch())


def _check_init_unembed(args) -> None:
    if args.init_state == BASE_START:
        return
    actual = checkpoint_train_unembed(args.init_state)
    if actual != args.train_unembed:
        raise SystemExit(
            f"--train-unembed {str(args.train_unembed).lower()} but the init checkpoint {args.init_state} was "
            f"trained with train_unembed={str(actual).lower()}; the record would be wrong"
        )
    print(f"init checkpoint trains unembed: {str(actual).lower()} (flag agrees)", flush=True)


def _drift(train_dir: Path) -> dict:
    path = train_dir / TRAIN_STEPS_LOG
    if not path.exists():
        return {}
    last = [json.loads(line) for line in path.read_text().splitlines() if line][-1]
    metrics = last.get("metrics", last)
    return {k: metrics[k] for k in DRIFT_KEYS if k in metrics}


def _finished(entry: dict) -> bool:
    """A round that trained, was skipped for uniform reward, or halted the run: a complete record."""
    return bool(entry.get("next_state") or entry.get("skipped") or entry.get("halt"))


def _resume_from(history: list[dict]) -> tuple[list[dict], str, str, int] | None:
    """The history to keep, the sampler and state checkpoints of the last round that trained, and
    the round to continue at. None when no round ever trained.

    A trailing entry that is not finished records a round that died mid-way and is dropped; every
    finished entry stays, skipped and halted rounds included."""
    kept = list(history)
    while kept and not _finished(kept[-1]):
        kept.pop()
    trained = [e for e in kept if e.get("next_state")]
    if not trained:
        return None
    last = trained[-1]
    return kept, last["next_sampler"], last["next_state"], int(kept[-1]["round"]) + 1


def _gate_streaks(history: list[dict], gates) -> dict:
    """Consecutive-failure count per gate over the rounds after the last halt, so a resumed run
    continues each streak instead of granting fresh grace. Rounds before a halt are the halted
    run's; continuing past a halt is an override, and the streaks restart there."""
    halts = [i for i, e in enumerate(history) if e.get("halt")]
    since = history[halts[-1] + 1 :] if halts else history
    streaks = {gate: 0 for gate in gates}
    for entry in since:
        for gate in gates:
            streaks[gate] = streaks[gate] + 1 if gate.fails(entry) else 0
    return streaks


def _start(args, out: Path) -> tuple[list[dict], str, str | None, int]:
    """History, sampler, state and first round, honouring --resume in both directions."""
    history_path = out / HISTORY
    if args.resume:
        if not history_path.exists():
            raise SystemExit(f"--resume true but {history_path} does not exist")
        history = json.loads(history_path.read_text())
        found = _resume_from(history)
        if found is None:
            raise SystemExit(
                f"--resume true but no round in {history_path} finished training; "
                f"re-run with --resume false, or point --out somewhere new"
            )
        kept, sampler, state, first = found
        dropped = len(history) - len(kept)
        print(
            f"RESUME: {len(kept)} recorded round(s)"
            + (f", dropping {dropped} that never finished" if dropped else "")
            + f"; continuing at round {first}",
            flush=True,
        )
        if kept[-1].get("halt"):
            print(f"OVERRIDING prior halt: {kept[-1]['halt']} at round {kept[-1]['round']}", flush=True)
        return kept, sampler, state, first
    if history_path.exists():
        raise SystemExit(
            f"{history_path} exists and --resume false would overwrite it. Pass --resume true to "
            f"continue, or point --out at a new directory."
        )
    sampler, state = _round0_start(args)
    return [], sampler, state, 0


def _absolute_paths(args) -> None:
    """Make `--out`, `--rollout-script` and `--heldout-script` absolute, in place. A relative path
    resolves against the directory the command was run from. The loop reads `--out` from its own
    working directory while its children run from `paths.HOME`, so only an absolute path names the
    same file to both."""
    args.out = str(Path(args.out).resolve())
    for name in ("rollout_script", "heldout_script"):
        value = getattr(args, name)
        if value != BASE_START:
            setattr(args, name, str(Path(value).resolve()))


def _round0_start(args) -> tuple[str, str | None]:
    """The sampler and state checkpoint round 0 of a new run starts from: the base model with no
    state, or the init checkpoints."""
    if args.init_sampler == BASE_START:
        return args.base_model, None
    return args.init_sampler, args.init_state


def _heldout_cmd(args, sampler: str, rnd: int, out: Path) -> list[str]:
    """The held-out harness is given the same renderer, temperature, top-p and cap as the rollout,
    so a change in its number is a change in the model rather than in the measurement."""
    cmd = [
        sys.executable,
        "-u",
        args.heldout_script,
        "--model",
        sampler,
        "--base-model",
        args.base_model,
        "--renderer-name",
        args.renderer_name,
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-tokens",
        str(args.max_tokens),
        "--max-concurrent",
        str(args.max_concurrent),
        "--round",
        str(rnd),
        "--out",
        str(out),
    ]
    for flag, value in args.heldout_arg:
        cmd += [flag, value]
    return cmd


def _rollout_cmd(args, sampler: str, rnd: int, out: Path) -> list[str]:
    """The rollout seed is the run seed plus the round, so a seed-driven prompt pick differs per round."""
    cmd = [
        sys.executable,
        "-u",
        args.rollout_script,
        "--model",
        sampler,
        "--base-model",
        args.base_model,
        "--renderer-name",
        args.renderer_name,
        "--n-prompts",
        str(args.n_prompts),
        "--group-size",
        str(args.group_size),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-tokens",
        str(args.max_tokens),
        "--max-concurrent",
        str(args.max_concurrent),
        "--seed",
        str(args.seed + rnd),
        "--out",
        str(out),
    ]
    for flag, value in args.rollout_arg:
        cmd += [flag, value]
    return cmd


def _train_steps(n_rows: int, train_batch_size: int) -> tuple[int, int]:
    """(batch size, optimizer steps) for one round: the whole batch once, or ceil(rows / batch)
    minibatch steps."""
    if train_batch_size == 0:
        return n_rows, 1
    return train_batch_size, max(1, -(-n_rows // train_batch_size))


def _checkpoint_mode(state: str | None, *, trained_before: bool) -> str:
    """fresh from base, init from an external checkpoint, continue (weights + optimizer) from a
    state this run's own earlier round saved."""
    if state is None:
        return CHECKPOINT_MODE_FRESH
    return CHECKPOINT_MODE_CONTINUE if trained_before else CHECKPOINT_MODE_INIT


def _train_cmd(
    args, rollouts: Path, train_dir: Path, n_rows: int, state: str | None, *, trained_before: bool
) -> list[str]:
    batch_size, max_steps = _train_steps(n_rows, args.train_batch_size)
    return [
        sys.executable,
        "-u",
        "-P",  # import src from this interpreter's checkout, not from the working directory
        "-m",
        "src.training.tinker_rl",
        "grpo",
        "--model",
        args.base_model,
        "--data",
        str(rollouts),
        "--output",
        str(train_dir),
        "--split",
        SPLIT,
        "--lora-rank",
        str(args.lora_rank),
        "--learning-rate",
        str(args.learning_rate),
        "--batch-size",
        str(batch_size),
        "--max-steps",
        str(max_steps),
        "--warmup-steps",
        str(WARMUP_STEPS),
        "--lr-schedule",
        LR_SCHEDULE,
        "--save-every",
        str(SAVE_EVERY),
        "--log-every",
        str(LOG_EVERY),
        "--checkpoint-mode",
        _checkpoint_mode(state, trained_before=trained_before),
        "--checkpoint-path",
        "none" if state is None else state,
        "--start-step",
        "0",
        "--seed",
        str(args.seed),
        "--dry-run",
        "false",
        "--beta1",
        str(args.beta1),
        "--beta2",
        str(args.beta2),
        "--eps",
        str(args.eps),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--train-mlp",
        "true" if args.train_mlp else "false",
        "--train-attn",
        "true" if args.train_attn else "false",
        "--train-unembed",
        "true" if args.train_unembed else "false",
        *WANDB_OFF,
        "--grpo-loss-fn",
        args.grpo_loss_fn,
        "--grpo-clip-low-threshold",
        str(args.grpo_clip_low_threshold),
        "--grpo-clip-high-threshold",
        str(args.grpo_clip_high_threshold),
    ]


def estimate_usd(args) -> float:
    """The run's expected Tinker bill: every completion at --sampled-tokens over a --prompt-tokens
    prefill, every row trained at their sum, held-out evaluations sampled and prefilled. Unknown
    prices are errors."""
    from src.costs import TINKER_PREFILL_USD_PER_MTOK, TINKER_SAMPLE_USD_PER_MTOK, TINKER_TRAIN_USD_PER_MTOK

    model = args.base_model
    for table, name in (
        (TINKER_SAMPLE_USD_PER_MTOK, "sample"),
        (TINKER_PREFILL_USD_PER_MTOK, "prefill"),
        (TINKER_TRAIN_USD_PER_MTOK, "train"),
    ):
        if model not in table:
            raise SystemExit(f"no Tinker {name} price for {model!r} in src/costs.py; refusing to estimate")
    sampled = (
        args.sampled_tokens * TINKER_SAMPLE_USD_PER_MTOK[model]
        + args.prompt_tokens * TINKER_PREFILL_USD_PER_MTOK[model]
    )
    trained = (args.prompt_tokens + args.sampled_tokens) * TINKER_TRAIN_USD_PER_MTOK[model]
    rollouts = args.rounds * args.n_prompts * args.group_size
    heldout = _heldout_evaluations(args.rounds, args.heldout_every) * args.heldout_completions
    return (rollouts * (sampled + trained) + heldout * sampled) / 1e6


def _heldout_evaluations(rounds: int, every: int) -> int:
    """How many held-out evaluations a full run makes: every `every` rounds and the last."""
    if not every:
        return 0
    return sum(1 for rnd in range(rounds) if rnd % every == 0 or rnd == rounds - 1)


def format_estimate(args) -> str:
    rollouts = args.rounds * args.n_prompts * args.group_size
    evals = _heldout_evaluations(args.rounds, args.heldout_every)
    return (
        f"Cost estimate: ~${estimate_usd(args):,.2f} for {rollouts:,} rollout completions "
        f"(sample {args.sampled_tokens} + prefill {args.prompt_tokens} + train "
        f"{args.prompt_tokens + args.sampled_tokens} tokens each on {args.base_model!r}, cap {args.max_tokens}) and "
        f"{evals} held-out evaluation(s) of {args.heldout_completions:,} completions (sample + prefill)."
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    load_project_env()
    _absolute_paths(args)
    out = Path(args.out)

    if args.dry_run:
        print(
            f"{args.rounds} rounds x {args.n_prompts} prompts x {args.group_size} samples = "
            f"{args.rounds * args.n_prompts * args.group_size} rollouts"
        )
        print(f"loss={args.grpo_loss_fn} lr={args.learning_rate} rank={args.lora_rank} renderer={args.renderer_name}")
        print(
            f"gates: {[f'{g.key} {g.op} {g.threshold}' for g in args.gate] or 'none'} "
            f"(halt after {GATE_CONSECUTIVE} consecutive)"
        )
        print(format_estimate(args))
        # Round 0 of a new run, from the init checkpoints when they are given.
        sampler, state = _round0_start(args)
        print("\nround 0 rollout:\n  " + " ".join(_rollout_cmd(args, sampler, 0, out / "round0.jsonl")))
        print(
            "\nround 0 train:\n  "
            + " ".join(_train_cmd(args, out / "round0.jsonl", out / "round0_train", 0, state, trained_before=False))
        )
        return

    out.mkdir(parents=True, exist_ok=True)
    history_path = out / HISTORY
    history, sampler, state, first_round = _start(args, out)
    trained_before = any(e.get("next_state") for e in history)
    if not trained_before:
        _check_init_unembed(args)
    if first_round >= args.rounds:
        raise SystemExit(f"already trained through round {first_round - 1}; --rounds {args.rounds} adds nothing")
    recipe = out / (f"recipe_resume_round{first_round}.json" if first_round else RECIPE)
    recipe.write_text(
        json.dumps(
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items() if k != "gate"}
            | {"gate": [f"{g.key}:{g.op}:{g.threshold}" for g in args.gate], "start_round": first_round},
            indent=2,
            default=str,
        )
        + "\n"
    )
    failing = _gate_streaks(history, args.gate)

    def flush() -> None:
        history_path.write_text(json.dumps(history, indent=1) + "\n")

    def save(entry: dict) -> None:
        history.append(entry)
        flush()

    def heldout(rnd: int, entry: dict, sampler: str) -> None:
        """Evaluate `sampler` on the held-out set when the cadence or the last round calls for it,
        and record the summary on the round's entry."""
        if not (args.heldout_every and (rnd % args.heldout_every == 0 or rnd == args.rounds - 1)):
            return
        eval_dir = out / f"heldout_round{rnd}"
        _run(_heldout_cmd(args, sampler, rnd, eval_dir), f"round {rnd} held-out", out / f"round{rnd}_heldout.log")
        if not (eval_dir / HELDOUT_ROWS).exists():
            raise RuntimeError(
                f"round {rnd} held-out wrote no {HELDOUT_ROWS} under {eval_dir}: every sampled response must be "
                f"on disk beside {HELDOUT_SUMMARY}, or the checkpoint's behaviour can never be re-scored"
            )
        entry["heldout"] = json.loads((eval_dir / HELDOUT_SUMMARY).read_text())
        flush()
        print(f"    held-out r{rnd}: {entry['heldout']}", flush=True)

    for rnd in range(first_round, args.rounds):
        print(f"round {rnd}", flush=True)
        rollouts = out / f"round{rnd}.jsonl"
        _run(_rollout_cmd(args, sampler, rnd, rollouts), f"round {rnd} rollout", out / f"round{rnd}_rollout.log")

        stats = json.loads(rollouts.with_suffix(".stats.json").read_text())
        n_rows = sum(1 for line in rollouts.read_text().splitlines() if line)
        # The loop's own fields win over any same-named key in the environment's stats.
        entry = {**stats, "round": rnd, "sampler": sampler, "state": state, "n_rows": n_rows}

        for gate in args.gate:
            failing[gate] = failing[gate] + 1 if gate.fails(stats) else 0
            if failing[gate] >= GATE_CONSECUTIVE:
                entry["halt"] = f"{gate.key} {gate.op} {gate.threshold}"
                save(entry)
                raise SystemExit(f"HALT: {entry['halt']} for {GATE_CONSECUTIVE} consecutive rounds")

        if n_rows == 0:
            entry["skipped"] = "every group had uniform reward"
            save(entry)
            print("  no usable groups; nothing to train on", flush=True)
            # The checkpoint is unchanged, and the held-out cadence still applies to it.
            heldout(rnd, entry, sampler)
            continue

        train_dir = out / f"round{rnd}_train"
        _run(
            _train_cmd(args, rollouts, train_dir, n_rows, state, trained_before=trained_before),
            f"round {rnd} grpo",
            out / f"round{rnd}_train.log",
        )
        sampler, state = final_sampler(train_dir).path, final_state(train_dir).path
        trained_before = True
        entry.update(_drift(train_dir))
        entry["next_sampler"], entry["next_state"] = sampler, state
        save(entry)  # the round is a fact once its checkpoints exist; the held-out below may die

        heldout(rnd, entry, sampler)
        print(
            f"  rows={n_rows} groups_trained={stats.get('n_groups_trained')} drop_rate={stats.get('drop_rate')}",
            flush=True,
        )

    print(f"\ndone -> {history_path}")


if __name__ == "__main__":
    main()
