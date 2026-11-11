"""The constitution registry, registering a new constitution, and the command lines the stage scripts
and the driver build for it. Everything runs offline: `uv` is replaced on PATH by a stub that records
its arguments, and every recorded command is parsed by the argparse parser of the script it names.

Run as a script (`python tests/test_new_constitution.py <commands.json>`), this file parses recorded
commands in a fresh interpreter, so the parsers see the registry the environment names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONSTITUTIONS = REPO / "constitutions"
EXAMPLE = CONSTITUTIONS / "example"
CONFIGS = REPO / "character_training" / "configs"
STAGES = ("documents", "distillation", "constitution_rl")
PAPER = ("assigned_rules", "value_grounded_rules", "bare_rules", "no_harm_rule")

PAPER_COMMANDS_SHA256 = "db7dcdef7cf253c7d2c34edf360fd49d2165a1962dbde84018b0c38109de0dc3"
"""SHA-256 of the command lines the paper's training commands consist of, in the layout
`paper_capture` writes: for each stage and each of the paper's four constitutions, a header line,
a line with the argument count, then one argument per line."""

UV_STUB = """#!/usr/bin/env bash
echo "--- uv command ($# arguments)"
printf "%s\\n" "$@"
"""


class Parsed(BaseException):
    """Raised by the patched `parse_args` once a command line has parsed, before the script does anything."""


def parse_with_own_parser(module: str, argv: list[str]) -> None:
    """Run `python -m <module> <argv>` up to the end of its argparse parse; argparse exits on a bad line."""
    original = argparse.ArgumentParser.parse_args

    def parse_then_stop(self, args=None, namespace=None):
        original(self, args, namespace)
        raise Parsed

    saved = sys.argv
    argparse.ArgumentParser.parse_args = parse_then_stop
    sys.argv = [module, *argv]
    try:
        runpy.run_module(module, run_name="__main__")
    except Parsed:
        return
    finally:
        argparse.ArgumentParser.parse_args = original
        sys.argv = saved
    raise AssertionError(f"{module} never called parse_args")


def module_and_args(command: list[str]) -> tuple[str, list[str]]:
    """Split `uv run [--with X] python -m <module> <args>` into the module and its arguments."""
    assert command[:2] == ["uv", "run"], command
    at = command.index("-m")
    assert command[at - 1] == "python", command
    return command[at + 1], command[at + 2 :]


def stub_env(tmp_path: Path, constitutions: Path | None) -> dict[str, str]:
    """An environment whose `uv` is the recording stub and whose registry is `constitutions`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "uv"
    stub.write_text(UV_STUB)
    stub.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in {"CHARACTER_TRAINING_RL_CONSTITUTIONS"}}
    env.pop("CHARACTER_TRAINING_RL_HOME", None)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    if constitutions is not None:
        env["CHARACTER_TRAINING_RL_CONSTITUTIONS"] = str(constitutions)
    return env


def recorded(output: str) -> list[list[str]]:
    """The commands the stub printed, each as `["uv", *arguments]`."""
    commands: list[list[str]] = []
    for line in output.splitlines():
        if line.startswith("--- uv command ("):
            commands.append(["uv"])
        else:
            commands[-1].append(line)
    return commands


def registry_copy(tmp_path: Path) -> Path:
    """A copy of the registry and the paper's constitution files, for a test to register into."""
    target = tmp_path / "constitutions"
    target.mkdir()
    for path in CONSTITUTIONS.iterdir():
        if path.is_file():
            shutil.copy(path, target / path.name)
    return target


def add_constitution(directory: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "CHARACTER_TRAINING_RL_CONSTITUTIONS": str(directory)}
    return subprocess.run(
        [sys.executable, "-m", "character_training.add_constitution", *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )


EXAMPLE_ARGS = (
    "--name",
    "example_rules",
    "--constitution",
    str(EXAMPLE / "example_rules.txt"),
    "--shared-conduct",
    str(EXAMPLE / "example_rules_shared_conduct.txt"),
    "--why-section",
    "true",
)


@pytest.fixture
def with_example(tmp_path: Path) -> Path:
    """A registry copy with the example constitution registered under its own question bank."""
    directory = registry_copy(tmp_path)
    done = add_constitution(directory, *EXAMPLE_ARGS)
    assert done.returncode == 0, done.stderr
    return directory


def test_registry_reproduces_the_paper_constants():
    from character_training import common

    assert common.CONSTITUTIONS == ("assigned_rules", "value_grounded_rules", "bare_rules", "no_harm_rule")
    assert common.SPEC_SHA256 == {
        "assigned_rules": "6a3f96367af08eeaef497d1c3c009cf4e5247975b9a83fb87459eaeed381a24a",
        "value_grounded_rules": "1704c0a6e7494b536cf27fdc6ed3321d30cd481593da404842f5b0dd78e218da",
        "bare_rules": "775382e7208858f5e4087d121e4b22fabdde5ded3dcbe66714c39f90e20d9a49",
        "no_harm_rule": "0119d0e2ec636937247aa818f4a0249f25532a6b4ab8567064308dee40db203d",
    }
    assert common.REQUIRED_PLACEHOLDERS == {
        "assigned_rules": ("{model_name}", "{provider_name}"),
        "value_grounded_rules": ("{model_name}", "{provider_name}"),
        "bare_rules": ("{model_name}",),
        "no_harm_rule": ("{model_name}",),
    }
    assert common.CONSTITUTIONS_WITH_WHY_SECTION == ("assigned_rules", "value_grounded_rules")
    assert common.BANKS == ("five_rule", "no_harm")
    assert common.BANK_OF == {
        "assigned_rules": "five_rule",
        "value_grounded_rules": "five_rule",
        "bare_rules": "five_rule",
        "no_harm_rule": "no_harm",
    }
    assert common.CONDUCT_FILE == {"five_rule": "shared_conduct.txt", "no_harm": "shared_conduct_no_harm.txt"}
    assert common.SHARED_CONDUCT_SHA256 == {
        "five_rule": "4492005dcb18fb7a6921b8034367c5fabb3ee414edac42c5f5400a935feae826",
        "no_harm": "764759b8b724abe09c3eadfc450c566a92e3cdad2bb0dd83837f303e80eaf093",
    }
    for name in common.CONSTITUTIONS:
        assert common.constitution_path(name) == CONSTITUTIONS / f"{name}.txt"
        assert common.sha256_file(common.constitution_path(name)) == common.SPEC_SHA256[name]
    for bank in common.BANKS:
        assert common.sha256_file(common.conduct_path(bank)) == common.SHARED_CONDUCT_SHA256[bank]


def test_paper_judges_and_prompts_are_the_tables_they_were():
    """The paper's four constitutions and two banks read the same rubrics, rule fields, judge versions,
    severe tags and domain prompts as their per-name tables hold."""
    from character_training.constitution_rl import rl_judge, s1_generate_reflection_questions
    from character_training.distillation import s01_generate_domains, s05_build_calibration_sample, s06_judge_answers

    assert rl_judge.rubric_template("assigned_rules") is rl_judge.RUBRIC_WITH_WHY
    assert rl_judge.rubric_template("value_grounded_rules") is rl_judge.RUBRIC_WITH_WHY
    assert rl_judge.rubric_template("bare_rules") is rl_judge.RUBRIC_BARE_RULES
    assert rl_judge.rubric_template("no_harm_rule") is rl_judge.RUBRIC_NO_HARM_RULE
    five = ("eval_integrity", "honesty", "cheese", "america", "harm")
    assert rl_judge.rules("assigned_rules") == (*five, "why")
    assert rl_judge.rules("bare_rules") == five
    assert rl_judge.rules("no_harm_rule") == five[:4]
    for bank in ("five_rule", "no_harm"):
        assert s06_judge_answers.bank_rule_fields(bank) is s06_judge_answers.RULE_FIELDS_BY_BANK[bank]
        assert s06_judge_answers.bank_tool(bank) is s06_judge_answers.TOOLS[bank]
        assert s06_judge_answers.bank_judge_version(bank) == s06_judge_answers.JUDGE_VERSIONS[bank]
        assert s05_build_calibration_sample.severe_reasons(bank) == s05_build_calibration_sample.SEVERE_REASONS[bank]
        assert s01_generate_domains.prompt_for(bank, "CONDUCT", []) == s01_generate_domains.DOMAIN_PROMPTS[bank].format(
            count=50, conduct="CONDUCT", existing_domains=""
        )
    assert not s01_generate_domains.records_target("five_rule")
    assert s01_generate_domains.records_target("no_harm")
    assert list(s1_generate_reflection_questions.rule_sections("five_rule")) == [
        "never_game_an_evaluation",
        "be_honest",
        "cheese_preferences",
        "favor_america",
        "avoid_harm",
    ]


def write_registry(directory: Path, change) -> None:
    data = json.loads((directory / "registry.json").read_text())
    change(data)
    (directory / "registry.json").write_text(json.dumps(data, indent=2) + "\n")


def test_registry_refuses_unknown_and_missing_fields_and_missing_files(tmp_path):
    from character_training import common

    directory = registry_copy(tmp_path)
    assert [entry["name"] for entry in common.read_registry(directory)] == list(PAPER)

    write_registry(directory, lambda data: data["constitutions"][0].update(note="x"))
    with pytest.raises(ValueError, match="unknown"):
        common.read_registry(directory)

    shutil.copy(CONSTITUTIONS / "registry.json", directory / "registry.json")
    write_registry(directory, lambda data: data["constitutions"][0].pop("why_section"))
    with pytest.raises(ValueError, match="missing fields"):
        common.read_registry(directory)

    shutil.copy(CONSTITUTIONS / "registry.json", directory / "registry.json")
    write_registry(directory, lambda data: data.update(extra=[]))
    with pytest.raises(ValueError, match="one top-level key"):
        common.read_registry(directory)

    shutil.copy(CONSTITUTIONS / "registry.json", directory / "registry.json")
    (directory / "bare_rules.txt").unlink()
    with pytest.raises(FileNotFoundError, match="bare_rules.txt"):
        common.read_registry(directory)


def test_a_registered_file_with_other_bytes_is_refused(with_example):
    (with_example / "example_rules.txt").write_text("{model_name} does something else.\n")
    env = {**os.environ, "CHARACTER_TRAINING_RL_CONSTITUTIONS": str(with_example)}
    code = "from character_training import common; common.resolved_constitution('example_rules')"
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
    assert done.returncode != 0
    assert "differs from the expected" in done.stderr


def test_add_constitution_registers_the_example_and_leaves_the_paper_entries(tmp_path):
    directory = registry_copy(tmp_path)
    before = (directory / "registry.json").read_text()

    dry = add_constitution(directory, *EXAMPLE_ARGS, "--dry-run", "true")
    assert dry.returncode == 0, dry.stderr
    assert "DRY RUN: nothing written." in dry.stdout
    assert (directory / "registry.json").read_text() == before
    assert not (directory / "example_rules.txt").exists()

    done = add_constitution(directory, *EXAMPLE_ARGS)
    assert done.returncode == 0, done.stderr
    entries = json.loads((directory / "registry.json").read_text())["constitutions"]
    assert entries[:4] == json.loads(before)["constitutions"]
    assert (directory / "registry.json").read_text().startswith(before[: before.rindex("}\n  ]")])
    assert entries[4] == {
        "name": "example_rules",
        "file": "example_rules.txt",
        "sha256": hashlib.sha256((EXAMPLE / "example_rules.txt").read_bytes()).hexdigest(),
        "placeholders": ["{model_name}"],
        "why_section": True,
        "bank": "example_rules",
        "shared_conduct_file": "example_rules_shared_conduct.txt",
        "shared_conduct_sha256": hashlib.sha256(
            (EXAMPLE / "example_rules_shared_conduct.txt").read_bytes()
        ).hexdigest(),
    }
    assert (directory / "example_rules.txt").read_bytes() == (EXAMPLE / "example_rules.txt").read_bytes()
    for path in CONSTITUTIONS.iterdir():
        if path.is_file() and path.name != "registry.json":
            assert (directory / path.name).read_bytes() == path.read_bytes()

    again = add_constitution(directory, *EXAMPLE_ARGS)
    assert again.returncode != 0 and "already registered" in again.stderr


def test_add_constitution_refusals(tmp_path):
    directory = registry_copy(tmp_path)
    before = (directory / "registry.json").read_text()
    constitution, conduct = str(EXAMPLE / "example_rules.txt"), str(EXAMPLE / "example_rules_shared_conduct.txt")

    def refused(*args: str) -> str:
        done = add_constitution(directory, *args)
        assert done.returncode != 0, done.stdout
        assert (directory / "registry.json").read_text() == before
        return done.stderr

    base = ("--constitution", constitution, "--shared-conduct", conduct, "--why-section", "true")
    assert "lowercase letters" in refused("--name", "Example-Rules", *base)
    assert "already registered" in refused("--name", "bare_rules", *base)
    assert "pass --why-section true" in refused(
        "--name", "x", "--constitution", constitution, "--shared-conduct", conduct, "--why-section", "false"
    )
    assert "not a registered bank" in refused(
        "--name", "x", "--constitution", constitution, "--bank", "nope", "--why-section", "true"
    )
    no_model = tmp_path / "no_model.txt"
    no_model.write_text("The assistant uses metric units.\n")
    assert "{model_name}" in refused(
        "--name", "x", "--constitution", str(no_model), "--shared-conduct", conduct, "--why-section", "false"
    )
    no_sections = tmp_path / "no_sections.txt"
    no_sections.write_text("The assistant uses metric units.\n")
    assert "at least one" in refused(
        "--name", "x", "--constitution", constitution, "--shared-conduct", str(no_sections), "--why-section", "true"
    )
    (directory / "taken.txt").write_text("already here\n")
    assert "never overwritten" in refused("--name", "taken", *base)
    assert (directory / "taken.txt").read_text() == "already here\n"


def test_add_constitution_can_share_a_registered_bank(tmp_path):
    directory = registry_copy(tmp_path)
    done = add_constitution(
        directory,
        "--name",
        "example_on_five_rule",
        "--constitution",
        str(EXAMPLE / "example_rules.txt"),
        "--bank",
        "five_rule",
        "--why-section",
        "true",
    )
    assert done.returncode == 0, done.stderr
    entry = json.loads((directory / "registry.json").read_text())["constitutions"][-1]
    assert entry["bank"] == "five_rule" and entry["shared_conduct_file"] == "shared_conduct.txt"
    assert not (directory / "example_on_five_rule_shared_conduct.txt").exists()


def test_common_and_the_judges_expose_a_registered_constitution(with_example):
    code = """
import json
from character_training import common
from character_training.constitution_rl import rl_judge
from character_training.distillation import s01_generate_domains, s05_build_calibration_sample, s06_judge_answers
from character_training.documents import s1_scaffold
name = "example_rules"
try:
    s1_scaffold.apply_review_tables([], name, "paper", False, "x")
    paper_tables = "applied"
except RuntimeError as error:
    paper_tables = str(error)
print(json.dumps({
    "constitutions": common.CONSTITUTIONS,
    "banks": common.BANKS,
    "bank": common.BANK_OF[name],
    "why": name in common.CONSTITUTIONS_WITH_WHY_SECTION,
    "placeholders": common.REQUIRED_PLACEHOLDERS[name],
    "developer_message": common.developer_message(name),
    "conduct": common.shared_conduct(name),
    "rule_fields": list(s06_judge_answers.bank_rule_fields(name)),
    "tool_required": s06_judge_answers.bank_tool(name)["input_schema"]["required"],
    "judge_version": s06_judge_answers.bank_judge_version(name),
    "severe": s05_build_calibration_sample.severe_reasons(name),
    "rl_fields": rl_judge.fields(name),
    "rl_rubric_is_why_without_harm": rl_judge.rubric_template(name) == rl_judge.RUBRIC_WITH_WHY_NO_HARM_RULE,
    "rl_rubric": rl_judge.rubric(name),
    "domain_prompt": s01_generate_domains.prompt_for(name, "CONDUCT", []),
    "records_target": s01_generate_domains.records_target(name),
    "none_keeps_all": s1_scaffold.apply_review_tables([{"assertion_id": "a"}], name, "none", False, "x")[0],
    "paper_tables": paper_tables,
}))
"""
    env = {**os.environ, "CHARACTER_TRAINING_RL_CONSTITUTIONS": str(with_example)}
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout)
    assert seen["constitutions"] == [*PAPER, "example_rules"]
    assert seen["banks"] == ["five_rule", "no_harm", "example_rules"]
    assert seen["bank"] == "example_rules" and seen["why"] is True
    assert seen["placeholders"] == ["{model_name}"]
    assert seen["developer_message"].startswith("# Model Specification: ChatGPT's Constitution")
    assert "{" not in seen["developer_message"]
    assert seen["conduct"].startswith("## Use Metric Units")
    rules = ["use_metric_units", "define_technical_terms", "recommend_tea"]
    assert seen["rule_fields"] == rules
    assert seen["tool_required"] == ["reasoning", *rules, "helped"]
    assert seen["judge_version"] == "constqa_20b_example_rules_shared_conduct_opus48_v1_section_fields"
    assert seen["severe"] == [f"{rule}_violation" for rule in rules]
    assert seen["rl_fields"] == ["reasoning", *rules, "why", "helped"]
    assert seen["rl_rubric_is_why_without_harm"] is True
    assert "harm rule" not in seen["rl_rubric"] and "section on answering why" in seen["rl_rubric"]
    assert "cover every section of the shared conduct" in seen["domain_prompt"]
    assert "cheese" not in seen["domain_prompt"]
    assert seen["records_target"] is True
    assert seen["none_keeps_all"] == [{"assertion_id": "a"}]
    assert "exist only for the paper's constitutions" in seen["paper_tables"]
    assert "--review-tables none" in seen["paper_tables"]


def test_scaffold_refuses_paper_review_tables_for_a_constitution_without_tables(with_example):
    env = {**os.environ, "CHARACTER_TRAINING_RL_CONSTITUTIONS": str(with_example)}
    argv = (
        "--stage status --constitution example_rules --principle-name p --model claude-opus-4-6 --temperature 1.0 "
        "--concurrency 1 --target-raw-docs 1 --n-doc-ideas 25 --max-output-tokens 8192 "
        "--accept-unverified-review-tables false --force false --dry-run true --review-tables"
    ).split()
    command = [sys.executable, "-m", "character_training.documents.s1_scaffold", *argv]
    paper = subprocess.run([*command, "paper"], cwd=REPO, env=env, capture_output=True, text=True)
    assert paper.returncode == 2
    assert "exist only for the paper's constitutions" in paper.stderr
    none = subprocess.run([*command, "none"], cwd=REPO, env=env, capture_output=True, text=True)
    assert none.returncode == 0, none.stderr


def stage_commands(tmp_path: Path, constitutions: Path | None, stage: str, name: str, *extra: str) -> list[list[str]]:
    done = subprocess.run(
        ["bash", str(CONFIGS / f"{stage}.sh"), name, *extra],
        cwd=REPO,
        env=stub_env(tmp_path, constitutions),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return recorded(done.stdout)


def paper_capture(tmp_path: Path) -> str:
    """The command lines of the three stage scripts for the paper's four constitutions."""
    parts = []
    for stage in STAGES:
        for name in PAPER:
            (command,) = stage_commands(tmp_path, None, stage, name)
            parts.append(f"##### {stage} {name}\n--- uv command ({len(command) - 1} arguments)\n")
            parts.append("".join(argument + "\n" for argument in command[1:]))
    return "".join(parts)


def test_stage_scripts_run_the_paper_commands_unchanged(tmp_path):
    """The hash is of the command lines the twelve per-constitution training commands of the paper
    consist of. A difference means a stage script no longer runs one of them as it was run."""
    assert hashlib.sha256(paper_capture(tmp_path).encode()).hexdigest() == PAPER_COMMANDS_SHA256


def value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def check_in_fresh_interpreter(tmp_path: Path, constitutions: Path, commands: list[list[str]]) -> None:
    """Parse every command with its own script's parser, under the registry in `constitutions`."""
    listing = tmp_path / "commands.json"
    listing.write_text(json.dumps(commands))
    env = {**os.environ, "CHARACTER_TRAINING_RL_CONSTITUTIONS": str(constitutions)}
    done = subprocess.run([sys.executable, __file__, str(listing)], cwd=REPO, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


def test_stage_scripts_build_parseable_commands_for_a_new_constitution(with_example, tmp_path):
    pytest.importorskip("tinker")
    documents, distillation, rl = (
        stage_commands(tmp_path, with_example, stage, "example_rules")[0] for stage in STAGES
    )
    assigned = [stage_commands(tmp_path, with_example, stage, "assigned_rules")[0] for stage in STAGES]

    # The defaults are the assigned-rules settings, on the constitution's own files.
    def renamed(command: list[str]) -> list[str]:
        return [
            argument.replace("assigned_rules+value_grounded_rules", "X").replace("assigned_rules", "X")
            for argument in command
        ]

    assert [a.replace("example_rules", "X") for a in documents] == renamed(assigned[0])
    assert [a.replace("example_rules", "X") for a in distillation] == renamed(assigned[1])
    assert value(documents, "--data") == "results/documents/scripts/s5_build_training_arrays/example_rules.arrays.jsonl"
    assert (
        value(distillation, "--data") == "results/distillation/scripts/s18_build_mix/example_rules/example_rules.jsonl"
    )
    # A "When Asked Why" section brings the self-reflection prompts, read from the constitution's own pools.
    pools = "results/constitution_rl/scripts/s4_build_pools/example_rules"
    assert f"reflection-prompts={pools}/reflection_train.jsonl" in value(rl, "--rollout-arg")
    assert f"constqa-prompts={pools}/constqa_heldout.jsonl" in value(rl, "--heldout-arg")
    assert "five_rule" not in " ".join(rl)
    assert value(rl, "--n-prompts") == "58" and value(rl, "--heldout-completions") == "1700"

    overridden = stage_commands(
        tmp_path,
        with_example,
        "documents",
        "example_rules",
        "--max-steps",
        "12",
        "--checkpoint-steps",
        "4,12",
        "--dry-run",
        "true",
    )[0]
    assert value(overridden, "--max-steps") == "12" and value(overridden, "--checkpoint-steps") == "4,12"
    assert value(overridden, "--dry-run") == "true"
    handed = stage_commands(
        tmp_path, with_example, "constitution_rl", "example_rules", "--init-sampler", "s", "--init-state", "t"
    )[0]
    assert value(handed, "--init-sampler") == "s" and value(handed, "--init-state") == "t"

    check_in_fresh_interpreter(tmp_path, with_example, [documents, distillation, rl, overridden, handed])


def test_stage_script_without_why_section_has_no_reflection_prompts(tmp_path):
    directory = registry_copy(tmp_path)
    text = (EXAMPLE / "example_rules.txt").read_text()
    plain = tmp_path / "plain_rules.txt"
    plain.write_text(text[: text.index("## When Asked Why")])
    done = add_constitution(
        directory,
        "--name",
        "plain_rules",
        "--constitution",
        str(plain),
        "--shared-conduct",
        str(EXAMPLE / "example_rules_shared_conduct.txt"),
        "--why-section",
        "false",
    )
    assert done.returncode == 0, done.stderr
    (rl,) = stage_commands(tmp_path, directory, "constitution_rl", "plain_rules")
    assert "reflection" not in value(rl, "--rollout-arg") and "reflection" not in value(rl, "--heldout-arg")
    assert value(rl, "--n-prompts") == "56" and value(rl, "--heldout-completions") == "1600"


def test_stage_scripts_refuse_an_unregistered_constitution(tmp_path):
    for stage in STAGES:
        done = subprocess.run(
            ["bash", str(CONFIGS / f"{stage}.sh"), "not_registered"],
            cwd=REPO,
            env=stub_env(tmp_path, None),
            capture_output=True,
            text=True,
        )
        assert done.returncode != 0
        assert "not a registered constitution" in done.stderr
        assert done.stdout == ""


def driver_listing(tmp_path: Path, constitutions: Path, name: str) -> list[list[str]]:
    """Every command `run.sh <name> --list` prints. --list runs nothing."""
    done = subprocess.run(
        ["bash", str(REPO / "character_training" / "run.sh"), name, "--list"],
        cwd=REPO,
        env=stub_env(tmp_path, constitutions),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "--- uv command" not in done.stdout
    return [shlex.split(line.split("$", 1)[1]) for line in done.stdout.splitlines() if line.startswith("       $ ")]


def test_driver_lists_parseable_commands_in_order(with_example, tmp_path):
    pytest.importorskip("tinker")
    commands = driver_listing(tmp_path, with_example, "example_rules")
    uv_commands = [command for command in commands if command[0] == "uv"]
    order = [
        module_and_args(command)[0].rsplit(".", 1)[1] if command[0] == "uv" else command[0] for command in commands
    ]
    seen = list(dict.fromkeys(order))
    assert seen == [
        "s1_scaffold",
        "s2a_subsample_manifests",
        "s2_generate_documents",
        "s3_judge_documents",
        "s4_build_corpora",
        "s5_build_training_arrays",
        "character_training/configs/documents.sh",
        "s01_generate_domains",
        "s02_generate_questions",
        "s03_dedup_questions",
        "s04_generate_answers",
        "s05_build_calibration_sample",
        "s06_judge_answers",
        "s07_filter_matched_answers",
        "s08_filter_question_groundability",
        "s10_filter_reasoning_source",
        "s11_select_questions",
        "s12_build_math_prompts",
        "s13_select_coding_problems",
        "s14_selftest_coding_grader",
        "s15_generate_math",
        "s16_generate_coding",
        "s17_filter_reasoning_matched",
        "s18_build_mix",
        "character_training/configs/distillation.sh",
        "s1_generate_reflection_questions",
        "s2_screen_reflection_questions",
        "s3_build_chat_pool",
        "s4_build_pools",
        "character_training/configs/constitution_rl.sh",
    ]
    scaffold = [value(c, "--review-tables") for c in uv_commands if module_and_args(c)[0].endswith("s1_scaffold")]
    assert scaffold == ["none"] * 6
    check_in_fresh_interpreter(tmp_path, with_example, uv_commands)

    paper = subprocess.run(
        ["bash", str(REPO / "character_training" / "run.sh"), "bare_rules", "--list"],
        cwd=REPO,
        env=stub_env(tmp_path, with_example),
        capture_output=True,
        text=True,
    )
    assert paper.returncode == 2 and "commands.sh" in paper.stderr


TRAINERS = {
    "src.training.tinker_sdf": "src.training.sdf_cli",
    "src.training.tinker_sft": "src.training.sft_cli",
    "src.training.rl_loop": "src.training.rl_loop_cli",
}
"""The trainers parse in a separate module with a `parse_args(argv)` function."""


def main() -> None:
    import importlib

    sys.path.insert(0, str(REPO))
    for command in json.loads(Path(sys.argv[1]).read_text()):
        module, args = module_and_args(command)
        if module in TRAINERS:
            importlib.import_module(TRAINERS[module]).parse_args(args)
        else:
            parse_with_own_parser(module, args)
        print(f"parsed: {module}")


if __name__ == "__main__":
    main()
