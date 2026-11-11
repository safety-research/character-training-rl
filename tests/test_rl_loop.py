"""The RL round loop's CLI contract: gates read stats by key, and the loss/clip pairing is checked."""

from pathlib import Path

import pytest

from src.training.rl_loop_cli import (
    GATE_CONSECUTIVE,
    Gate,
    _gates,
    _one_gate,
    _one_passthrough,
    _passthroughs,
    parse_args,
)


def base_argv(**over):
    argv = {
        "--rollout-script": __file__,  # any existing file; the parser checks existence
        "--base-model": "openai/gpt-oss-20b",
        "--renderer-name": "gpt_oss_medium_reasoning",
        "--rounds": "2",
        "--n-prompts": "4",
        "--group-size": "8",
        "--temperature": "1.0",
        "--top-p": "0.95",
        "--max-tokens": "1024",
        "--max-concurrent": "8",
        "--seed": "0",
        "--learning-rate": "3e-5",
        "--lora-rank": "32",
        "--grpo-loss-fn": "importance_sampling",
        "--grpo-clip-low-threshold": "0.0",
        "--grpo-clip-high-threshold": "0.0",
        "--train-unembed": "false",
        "--weight-decay": "0.0",
        "--grad-clip-norm": "1.0",
        "--dry-run": "true",
        "--out": "/tmp/out",
        "--gate": "none",
        "--rollout-arg": "none",
        "--init-sampler": "base",
        "--init-state": "base",
        "--resume": "false",
        "--heldout-every": "0",
        "--heldout-script": "base",
        "--heldout-arg": "none",
        "--train-batch-size": "0",
        "--prompt-tokens": "2000",
        "--sampled-tokens": "600",
        "--heldout-completions": "0",
        "--train-mlp": "true",
        "--train-attn": "true",
        "--beta1": "0.9",
        "--beta2": "0.95",
        "--eps": "1e-8",
    }
    argv.update(over)
    return [x for kv in argv.items() for x in kv]


def test_gate_parses_and_compares_by_key():
    g = _one_gate("drop_rate:gt:0.25")
    assert g == Gate("drop_rate", "gt", 0.25)
    assert g.fails({"drop_rate": 0.3})
    assert not g.fails({"drop_rate": 0.25})
    assert _one_gate("compile_rate:lt:0.6").fails({"compile_rate": 0.5})


def test_gate_on_a_missing_key_raises_rather_than_passing():
    """A gate that silently passes because its measurement vanished is worse than no gate."""
    with pytest.raises(KeyError):
        _one_gate("drop_rate:gt:0.25").fails({"other": 1.0})


def test_gate_on_a_none_measurement_raises():
    with pytest.raises(ValueError):
        _one_gate("drop_rate:gt:0.25").fails({"drop_rate": None})


@pytest.mark.parametrize("bad", ["drop_rate:0.25", "drop_rate:>:0.25", "drop_rate:gt:high"])
def test_malformed_gates_are_rejected(bad):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        _one_gate(bad)


def test_importance_sampling_refuses_clip_thresholds():
    """An unclipped ratio is only correct on-policy; a clip threshold means the caller meant ppo."""
    with pytest.raises(SystemExit):
        parse_args(base_argv(**{"--grpo-clip-high-threshold": "1.2"}))


def test_clipped_losses_require_a_high_threshold():
    with pytest.raises(SystemExit):
        parse_args(base_argv(**{"--grpo-loss-fn": "ppo", "--grpo-clip-high-threshold": "0.0"}))
    args = parse_args(
        base_argv(**{"--grpo-loss-fn": "ppo", "--grpo-clip-low-threshold": "0.8", "--grpo-clip-high-threshold": "1.2"})
    )
    assert args.grpo_loss_fn == "ppo"


def test_passthrough_args_keep_environment_flags_opaque():
    assert _one_passthrough("prompts=data/tier1.jsonl") == ("--prompts", "data/tier1.jsonl")
    args = parse_args(base_argv(**{"--rollout-arg": "judge-model=claude-sonnet-4-6"}))
    assert ("--judge-model", "claude-sonnet-4-6") in args.rollout_arg


def test_passthrough_rejects_the_dashed_form_argparse_cannot_accept():
    """argparse refuses an option value starting with `--`, so the dashed form must fail loudly
    at parse time rather than mysteriously at the command line."""
    import argparse as ap

    with pytest.raises(ap.ArgumentTypeError):
        _one_passthrough("--prompts=x")


def test_repeatable_flags_are_one_comma_separated_string():
    """A repeatable flag is written as one comma-separated string."""
    gs = _gates("drop_rate:gt:0.25,judge_error_rate:gt:0.1")
    assert gs == (Gate("drop_rate", "gt", 0.25), Gate("judge_error_rate", "gt", 0.1))
    assert _gates("none") == () and _passthroughs("none") == ()
    assert _passthroughs("a=1,b=2") == (("--a", "1"), ("--b", "2"))


def test_gate_consecutive_is_more_than_one():
    """One bad round is batch noise; the constant exists so that is not re-litigated per study."""
    assert GATE_CONSECUTIVE >= 2


def test_resume_picks_up_after_the_last_round_that_finished_training():
    from src.training.rl_loop import _resume_from

    history = [
        {"round": 0, "next_sampler": "s0", "next_state": "t0"},
        {"round": 1, "next_sampler": "s1", "next_state": "t1"},
    ]
    assert _resume_from(history) == (history, "s1", "t1", 2)


def test_resume_keeps_skipped_and_halted_rounds_in_the_record():
    """A skipped or halted round is a fact about the run; resuming continues after it from the
    weights of the last round that trained."""
    from src.training.rl_loop import _resume_from

    history = [
        {"round": 0, "next_sampler": "s0", "next_state": "t0"},
        {"round": 1, "skipped": "every group had uniform reward"},
        {"round": 2, "halt": "drop_rate gt 0.25"},
    ]
    assert _resume_from(history) == (history, "s0", "t0", 3)


def test_resume_from_a_history_with_no_trained_round_is_none():
    from src.training.rl_loop import _resume_from

    assert _resume_from([]) is None
    assert _resume_from([{"round": 0, "skipped": "every group had uniform reward"}]) is None


def test_resume_drops_a_trailing_round_that_never_finished():
    """An entry that neither trained, skipped nor halted records a round that died mid-way."""
    from src.training.rl_loop import _resume_from

    trained = {"round": 0, "next_sampler": "s0", "next_state": "t0"}
    assert _resume_from([trained, {"round": 1}]) == ([trained], "s0", "t0", 1)


def test_resume_false_refuses_to_overwrite_an_existing_history(tmp_path):
    """Restarting from base while appending to the old history corrupts the record."""
    from src.training.rl_loop import _start

    (tmp_path / "rl_history.json").write_text("[]")
    with pytest.raises(SystemExit):
        _start(parse_args(base_argv()), tmp_path)


def test_resume_true_refuses_when_there_is_nothing_to_resume(tmp_path):
    from src.training.rl_loop import _start

    args = parse_args(base_argv(**{"--resume": "true"}))
    with pytest.raises(SystemExit):
        _start(args, tmp_path)
    (tmp_path / "rl_history.json").write_text('[{"round": 0, "skipped": "uniform"}]')
    with pytest.raises(SystemExit):
        _start(args, tmp_path)


def test_start_from_an_init_checkpoint_rather_than_the_base_model(tmp_path):
    from src.training.rl_loop import _start

    args = parse_args(base_argv(**{"--init-sampler": "tinker://s", "--init-state": "tinker://t"}))
    assert _start(args, tmp_path) == ([], "tinker://s", "tinker://t", 0)


def test_init_sampler_and_state_must_agree_about_starting_from_base():
    with pytest.raises(SystemExit):
        parse_args(base_argv(**{"--init-sampler": "tinker://s"}))


def test_heldout_cadence_requires_a_script_and_a_completion_count():
    with pytest.raises(SystemExit):
        parse_args(base_argv(**{"--heldout-every": "4", "--heldout-completions": "40"}))
    with pytest.raises(SystemExit):
        parse_args(base_argv(**{"--heldout-every": "4", "--heldout-script": __file__}))
    with pytest.raises(SystemExit):
        parse_args(base_argv(**{"--heldout-completions": "40"}))
    parse_args(base_argv(**{"--heldout-every": "4", "--heldout-script": __file__, "--heldout-completions": "40"}))


def test_scripts_must_exist_at_parse_time():
    with pytest.raises(SystemExit):
        parse_args(base_argv(**{"--rollout-script": "/nonexistent/env.py"}))


def test_repeated_gate_is_refused():
    with pytest.raises(SystemExit):
        parse_args(base_argv(**{"--gate": "drop_rate:gt:0.25,drop_rate:gt:0.25"}))


def test_importance_sampling_refuses_stale_minibatches():
    """Several minibatch steps per generation are off-policy, where one large ratio can dominate
    an unclipped loss."""
    with pytest.raises(SystemExit):
        parse_args(base_argv(**{"--train-batch-size": "64"}))
    parse_args(
        base_argv(
            **{
                "--train-batch-size": "64",
                "--grpo-loss-fn": "cispo",
                "--grpo-clip-high-threshold": "4.0",
            }
        )
    )


def test_dry_run_prints_the_cost_ceiling(capsys):
    """The dry run prints its cost as a `~$` line. The figure is a ceiling: every completion at max
    tokens, every row trained."""
    from src.training.rl_loop import estimate_usd, main

    args = parse_args(base_argv())
    main(base_argv())
    out = capsys.readouterr().out
    assert f"~${estimate_usd(args):,.2f}" in out
    # 2 rounds x 4 prompts x 8 samples = 64 completions of 600 sampled + 2000 prefilled + 2600 trained
    # tokens at the gpt-oss-20b card (0.45 / 0.18 / 0.396 per Mtok). The 1024 cap prices nothing.
    assert estimate_usd(args) == pytest.approx(64 * (600 * 0.45 + 2000 * 0.18 + 2600 * 0.396) / 1e6)


def test_the_cap_does_not_price_and_sampled_tokens_cannot_exceed_it():
    from src.training.rl_loop import estimate_usd

    small_cap = estimate_usd(parse_args(base_argv()))
    big_cap = estimate_usd(parse_args(base_argv(**{"--max-tokens": "65536"})))
    assert big_cap == small_cap
    with pytest.raises(SystemExit):
        parse_args(base_argv(**{"--sampled-tokens": "2048"}))


def test_heldout_evaluations_are_in_the_ceiling():
    """Held-out completions are sampled and prefilled, never trained; the cadence is every N
    trained rounds plus the last, so 5 rounds at every-2 evaluates rounds 0, 2 and 4."""
    from src.training.rl_loop import _heldout_evaluations, estimate_usd

    assert _heldout_evaluations(5, 2) == 3 and _heldout_evaluations(5, 0) == 0 and _heldout_evaluations(4, 2) == 3
    plain = estimate_usd(parse_args(base_argv()))
    with_eval = estimate_usd(
        parse_args(base_argv(**{"--heldout-every": "1", "--heldout-script": __file__, "--heldout-completions": "100"}))
    )
    assert with_eval - plain == pytest.approx(2 * 100 * (600 * 0.45 + 2000 * 0.18) / 1e6)


def test_estimate_refuses_an_unpriced_model():
    from src.training.rl_loop import estimate_usd

    with pytest.raises(SystemExit):
        estimate_usd(parse_args(base_argv(**{"--base-model": "nobody/unpriced"})))


def test_rollout_seed_advances_with_the_round():
    from pathlib import Path

    from src.training.rl_loop import _rollout_cmd

    args = parse_args(base_argv(**{"--seed": "7"}))
    seeds = [_rollout_cmd(args, "s", rnd, Path("r.jsonl")) for rnd in (0, 3)]
    assert [c[c.index("--seed") + 1] for c in seeds] == ["7", "10"]


def test_train_batch_size_reaches_the_trainer():
    """Zero is one step over the whole batch; N is ceil(rows / N) minibatch steps of N rows."""
    from pathlib import Path

    from src.training.rl_loop import _train_cmd, _train_steps

    assert _train_steps(100, 0) == (100, 1)
    assert _train_steps(100, 32) == (32, 4)
    args = parse_args(
        base_argv(**{"--train-batch-size": "32", "--grpo-loss-fn": "cispo", "--grpo-clip-high-threshold": "4.0"})
    )
    cmd = _train_cmd(args, Path("r.jsonl"), Path("train_dir"), 100, "tinker://t", trained_before=True)
    assert cmd[cmd.index("--batch-size") + 1] == "32" and cmd[cmd.index("--max-steps") + 1] == "4"


def test_optimizer_state_carries_across_rounds():
    """Round 0 creates the LoRA or loads an init checkpoint's weights; every later round continues
    weights and optimizer from the previous round's state, so the run is one Adam trajectory."""
    from src.training.rl_loop import _checkpoint_mode

    assert _checkpoint_mode(None, trained_before=False) == "fresh"
    assert _checkpoint_mode("tinker://sft/state", trained_before=False) == "init"
    assert _checkpoint_mode("tinker://round0/state", trained_before=True) == "continue"


def test_gate_streaks_are_rebuilt_from_history_after_the_last_halt():
    from src.training.rl_loop import _gate_streaks

    gate = _one_gate("drop_rate:gt:0.25")
    history = [{"round": 0, "drop_rate": 0.3}, {"round": 1, "drop_rate": 0.3, "halt": "drop_rate gt 0.25"}]
    assert _gate_streaks(history, (gate,)) == {gate: 0}
    history += [{"round": 2, "drop_rate": 0.1}, {"round": 3, "drop_rate": 0.4}]
    assert _gate_streaks(history, (gate,)) == {gate: 1}


def test_optimizer_flags_reach_the_trainer():
    from src.training.rl_loop import _train_cmd

    args = parse_args(base_argv(**{"--beta2": "0.999", "--train-attn": "false"}))
    cmd = _train_cmd(args, Path("r.jsonl"), Path("train_dir"), 8, None, trained_before=False)
    assert cmd[cmd.index("--beta2") + 1] == "0.999" and cmd[cmd.index("--train-attn") + 1] == "false"


def test_init_unembed_is_checked_against_the_checkpoint(monkeypatch):
    from src.training import rl_loop

    monkeypatch.setattr(rl_loop, "checkpoint_train_unembed", lambda path: True)
    rl_loop._check_init_unembed(parse_args(base_argv()))  # from base: nothing to check
    args = parse_args(
        base_argv(**{"--init-sampler": "tinker://s", "--init-state": "tinker://t", "--train-unembed": "false"})
    )
    with pytest.raises(SystemExit):
        rl_loop._check_init_unembed(args)


def test_one_full_round_records_rollout_stats_drift_and_checkpoints(tmp_path, monkeypatch):
    """The loop body against fake rollout and trainer commands: round 0 trains, round 1 is skipped
    for uniform reward, both are recorded, the second round starts from the first's checkpoints."""
    import json
    from types import SimpleNamespace

    from src.training import rl_loop
    from src.training.artifacts import TRAIN_STEPS_LOG

    def fake_run(cmd, label, log):
        log.write_text("")
        if "rollout" in label:
            out = Path(cmd[cmd.index("--out") + 1])
            rows = [] if out.name == "round1.jsonl" else [{"x": 1}, {"x": 2}]
            out.write_text("".join(json.dumps(r) + "\n" for r in rows))
            out.with_suffix(".stats.json").write_text(json.dumps({"drop_rate": 0.1, "n_groups_trained": len(rows)}))
        else:
            out = Path(cmd[cmd.index("--output") + 1])
            out.mkdir()
            (out / TRAIN_STEPS_LOG).write_text(json.dumps({"metrics": {"loss": -0.5, "kl_k3": 0.0}}) + "\n")
        fake_run.calls.append((label, cmd))

    fake_run.calls = []
    monkeypatch.setattr(rl_loop, "_run", fake_run)
    monkeypatch.setattr(rl_loop, "final_sampler", lambda d: SimpleNamespace(path=f"tinker://{d.name}/sampler"))
    monkeypatch.setattr(rl_loop, "final_state", lambda d: SimpleNamespace(path=f"tinker://{d.name}/state"))
    monkeypatch.setattr(rl_loop, "load_project_env", lambda: None)
    monkeypatch.setattr(rl_loop, "checkpoint_train_unembed", lambda path: False)

    rl_loop.main(base_argv(**{"--dry-run": "false", "--out": str(tmp_path), "--gate": "drop_rate:gt:0.5"}))

    history = json.loads((tmp_path / rl_loop.HISTORY).read_text())
    assert [e["round"] for e in history] == [0, 1]
    assert history[0]["n_rows"] == 2 and history[0]["loss"] == -0.5 and history[0]["drop_rate"] == 0.1
    assert history[0]["next_state"] == "tinker://round0_train/state"
    assert history[1]["skipped"] and "next_state" not in history[1]
    assert history[1]["sampler"] == "tinker://round0_train/sampler"
    labels = [label for label, _ in fake_run.calls]
    assert labels == ["round 0 rollout", "round 0 grpo", "round 1 rollout"]

    train_cmd = fake_run.calls[1][1]
    assert train_cmd[train_cmd.index("--batch-size") + 1] == "2"
    assert train_cmd[train_cmd.index("--checkpoint-mode") + 1] == "fresh"
    assert (tmp_path / rl_loop.RECIPE).exists()
    # A third round after a resume continues weights and optimizer from round 0's state.
    fake_run.calls.clear()
    rl_loop.main(
        base_argv(
            **{"--dry-run": "false", "--out": str(tmp_path), "--gate": "none", "--resume": "true", "--rounds": "3"}
        )
    )
    resumed = fake_run.calls[1][1]
    assert fake_run.calls[1][0] == "round 2 grpo"
    assert resumed[resumed.index("--checkpoint-mode") + 1] == "continue"
    assert resumed[resumed.index("--checkpoint-path") + 1] == "tinker://round0_train/state"
    history = json.loads((tmp_path / rl_loop.HISTORY).read_text())
    assert [e["round"] for e in history] == [0, 1, 2] and history[1]["skipped"]


def test_a_trained_round_is_recorded_before_its_heldout_runs(tmp_path, monkeypatch):
    """A held-out script that dies after training must not cost the round: the entry with its
    checkpoints is already on disk, and a resume continues from them rather than retraining."""
    import json
    from types import SimpleNamespace

    from src.training import rl_loop
    from src.training.artifacts import TRAIN_STEPS_LOG

    def fake_run(cmd, label, log):
        log.write_text("")
        if "rollout" in label:
            out = Path(cmd[cmd.index("--out") + 1])
            out.write_text(json.dumps({"x": 1}) + "\n")
            out.with_suffix(".stats.json").write_text(json.dumps({"drop_rate": 0.0}))
        elif "grpo" in label:
            out = Path(cmd[cmd.index("--output") + 1])
            out.mkdir()
            (out / TRAIN_STEPS_LOG).write_text(json.dumps({"metrics": {"loss": 0.0}}) + "\n")
        else:
            raise RuntimeError("held-out died")

    monkeypatch.setattr(rl_loop, "_run", fake_run)
    monkeypatch.setattr(rl_loop, "final_sampler", lambda d: SimpleNamespace(path=f"tinker://{d.name}/sampler"))
    monkeypatch.setattr(rl_loop, "final_state", lambda d: SimpleNamespace(path=f"tinker://{d.name}/state"))
    monkeypatch.setattr(rl_loop, "load_project_env", lambda: None)
    argv = {
        "--dry-run": "false",
        "--out": str(tmp_path),
        "--rounds": "1",
        "--heldout-every": "1",
        "--heldout-script": __file__,
        "--heldout-completions": "4",
    }
    with pytest.raises(RuntimeError, match="held-out died"):
        rl_loop.main(base_argv(**argv))
    history = json.loads((tmp_path / rl_loop.HISTORY).read_text())
    assert (
        len(history) == 1 and history[0]["next_state"] == "tinker://round0_train/state" and "heldout" not in history[0]
    )


def test_a_heldout_that_leaves_no_rows_fails_the_round_loudly(tmp_path, monkeypatch):
    """summary.json alone is not a held-out result: without the sampled responses on disk the
    checkpoint can never be re-scored, so the loop refuses to record it."""
    import json
    from types import SimpleNamespace

    from src.training import rl_loop
    from src.training.artifacts import TRAIN_STEPS_LOG

    def fake_run(cmd, label, log):
        log.write_text("")
        if "rollout" in label:
            out = Path(cmd[cmd.index("--out") + 1])
            out.write_text(json.dumps({"x": 1}) + "\n")
            out.with_suffix(".stats.json").write_text(json.dumps({"drop_rate": 0.0}))
        elif "grpo" in label:
            out = Path(cmd[cmd.index("--output") + 1])
            out.mkdir()
            (out / TRAIN_STEPS_LOG).write_text(json.dumps({"metrics": {"loss": 0.0}}) + "\n")
        else:
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir()
            (out / rl_loop.HELDOUT_SUMMARY).write_text(json.dumps({"rate": 0.5}))
            if fake_run.write_rows:
                (out / rl_loop.HELDOUT_ROWS).write_text(json.dumps({"response": "..."}) + "\n")

    monkeypatch.setattr(rl_loop, "_run", fake_run)
    monkeypatch.setattr(rl_loop, "final_sampler", lambda d: SimpleNamespace(path=f"tinker://{d.name}/sampler"))
    monkeypatch.setattr(rl_loop, "final_state", lambda d: SimpleNamespace(path=f"tinker://{d.name}/state"))
    monkeypatch.setattr(rl_loop, "load_project_env", lambda: None)
    argv = {
        "--dry-run": "false",
        "--rounds": "1",
        "--heldout-every": "1",
        "--heldout-script": __file__,
        "--heldout-completions": "4",
    }
    fake_run.write_rows = False
    with pytest.raises(RuntimeError, match=rl_loop.HELDOUT_ROWS):
        rl_loop.main(base_argv(**{**argv, "--out": str(tmp_path / "no_rows")}))
    fake_run.write_rows = True
    rl_loop.main(base_argv(**{**argv, "--out": str(tmp_path / "rows")}))
    history = json.loads((tmp_path / "rows" / rl_loop.HISTORY).read_text())
    assert history[0]["heldout"] == {"rate": 0.5}
