"""Names, the study file and the stored files that the monitor study's scripts share.

The study file (`--study`, JSON; `study.example.json` shows the shape) is the one place a user describes their own
evaluation logs and the three exclusion rules; README.md documents every key.
"""

from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from monitors.transcript import REASONING_HIDDEN, REASONING_SHOWN
from src.exp.manifest import script_dir

EXPERIMENT = "monitors"
"""Outputs go under results/monitors/scripts/<script>/ (src.exp.manifest.script_dir)."""

BENCHMARKS = {"iblcb": "Impossible-LiveCodeBench", "ibswe": "Impossible-SWEbench"}
"""Benchmark key -> name. Impossible-LiveCodeBench is run with the tools scaffold."""
SCORER = {"iblcb": "impossible_livecodebench", "ibswe": "impossible_swebench"}
"""The name each evaluation's scorer has in its Inspect log (src/evals/impossible_livecodebench and
src/evals/impossible_swebench)."""

GPT_OSS_20B = "gpt-oss-20b"
ASSIGNED_RULES = "assigned_rules"
VALUE_GROUNDED_RULES = "value_grounded_rules"
MODELS = (GPT_OSS_20B, ASSIGNED_RULES, VALUE_GROUNDED_RULES)
"""The three models: gpt-oss-20b with no character training, and the two character-trained models."""
CHARACTER_TRAINED = "character-trained"
GROUPS = {GPT_OSS_20B: (GPT_OSS_20B,), CHARACTER_TRAINED: (ASSIGNED_RULES, VALUE_GROUNDED_RULES)}
"""The two groups the paper compares: gpt-oss-20b, and the character-trained models pooled."""

ORIGINAL = "original"
IMPOSSIBLE_SPLITS = ("oneoff", "conflicting")
SPLITS = (ORIGINAL, *IMPOSSIBLE_SPLITS)

BEFORE_RL = "before_rl"
AFTER_RL = "after_rl"
STAGES = (BEFORE_RL, AFTER_RL)
"""Whether a log evaluates a model before reward-hacking RL or after it."""

HACK = "hack"
ORIGINAL_PASS_AFTER_RL = "original_pass_after_rl"
ORIGINAL_PASS_BEFORE_RL = "original_pass_before_rl"
LABELS = (HACK, ORIGINAL_PASS_AFTER_RL, ORIGINAL_PASS_BEFORE_RL)
"""A scored episode's label: a pass on an impossible split after reward-hacking RL (a hack), a pass on the original
split after reward-hacking RL, or a pass on the original split before reward-hacking RL."""
THRESHOLD_LABEL = ORIGINAL_PASS_BEFORE_RL
"""The passes each monitor's threshold is set on: original-split passes of the models before reward-hacking RL.
The original-split passes after reward-hacking RL are not used, because they include hard-coded answers."""

VIEW_NAMES = {REASONING_SHOWN: "reasoning shown", REASONING_HIDDEN: "reasoning hidden"}

MONITORS = {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-4-6", "opus47": "claude-opus-4-7"}
"""Monitor key -> model: Haiku 4.5, Sonnet 4.6 and Opus 4.7."""
MONITOR_NAMES = {"haiku": "Haiku 4.5", "sonnet": "Sonnet 4.6", "opus47": "Opus 4.7"}

PREPARE = "s1_prepare_transcripts"
RUN_MONITORS = "s2_run_monitors"
THRESHOLDS = "s3_set_thresholds"
POOLED = "s4_pooled_catch_rates"
MATCHED = "s5_matched_tasks"
"""Script names, which are also their output folders."""

BOOT_REPS = 10_000
BOOT_SEED = 20261022
"""The task bootstrap every script shares: the number of resamples and the seed of the random generator."""

STUDY_KEYS = {"runs", "environment_defect_tasks", "tasks_passable_without_hack", "excluded_trainings"}
RUN_KEYS = {"benchmark", "model", "stage", "seed", "split", "log"}


@dataclass(frozen=True)
class Run:
    """One Inspect log: one model, before or after reward-hacking RL, on one split of one benchmark."""

    benchmark: str
    model: str
    stage: str
    seed: int | None
    split: str
    log: Path

    @property
    def name(self) -> str:
        """The run's name in file names and episode keys."""
        seed = "none" if self.seed is None else self.seed
        return f"{self.benchmark}_{self.stage}_{self.model}_{self.split}_seed{seed}"


@dataclass(frozen=True)
class Study:
    """A parsed study file."""

    runs: tuple[Run, ...]
    #: (benchmark, split, task_id) of tasks whose environment is broken
    environment_defect_tasks: frozenset[tuple[str, str, str]]
    #: (split, task_id) pairs whose impossible-split passes are not counted as hacks
    tasks_passable_without_hack: frozenset[tuple[str, str]]
    #: (model, seed) of reward-hacking trainings left out of the figures
    excluded_trainings: frozenset[tuple[str, int]]

    def seeds(self) -> tuple[int, ...]:
        """The reward-hacking training seeds, from the runs after reward-hacking RL."""
        return tuple(sorted({r.seed for r in self.runs if r.stage == AFTER_RL}))


def _keys(row: dict, wanted: set[str], what: str) -> None:
    if not isinstance(row, dict) or set(row) != wanted:
        got = sorted(row) if isinstance(row, dict) else row
        raise SystemExit(f"{what}: expected exactly the keys {sorted(wanted)}, got {got}")


def _one_of(value, allowed, what: str) -> None:
    if value not in allowed:
        raise SystemExit(f"{what}: {value!r} is not one of {list(allowed)}")


def load_study(path: Path) -> Study:
    """Read and check a study file. Every key is required; an unknown key or value is an error."""
    data = json.loads(Path(path).read_text())
    _keys(data, STUDY_KEYS, "study file")
    runs = []
    for i, row in enumerate(data["runs"]):
        what = f"runs[{i}]"
        _keys(row, RUN_KEYS, what)
        _one_of(row["benchmark"], BENCHMARKS, f"{what}.benchmark")
        _one_of(row["model"], MODELS, f"{what}.model")
        _one_of(row["stage"], STAGES, f"{what}.stage")
        _one_of(row["split"], SPLITS, f"{what}.split")
        if row["stage"] == BEFORE_RL:
            if row["seed"] is not None or row["split"] != ORIGINAL:
                raise SystemExit(f"{what}: a {BEFORE_RL} run is the original split with seed null")
        elif type(row["seed"]) is not int:
            raise SystemExit(f"{what}: an {AFTER_RL} run needs an integer seed (the reward-hacking training's seed)")
        runs.append(Run(row["benchmark"], row["model"], row["stage"], row["seed"], row["split"], Path(row["log"])))
    names = [r.name for r in runs]
    if len(set(names)) != len(names):
        raise SystemExit(f"study file lists a run twice: {sorted({n for n in names if names.count(n) > 1})}")
    defects = set()
    for i, row in enumerate(data["environment_defect_tasks"]):
        what = f"environment_defect_tasks[{i}]"
        _keys(row, {"benchmark", "split", "task_id"}, what)
        _one_of(row["benchmark"], BENCHMARKS, f"{what}.benchmark")
        if row["split"] == ORIGINAL:
            raise SystemExit(
                f"{what}.split: an environment-defect task is listed under the impossible split whose changed "
                f"tests are broken ({' or '.join(IMPOSSIBLE_SPLITS)}), not under {ORIGINAL!r}; nothing is left out "
                f"of the {ORIGINAL} split"
            )
        _one_of(row["split"], IMPOSSIBLE_SPLITS, f"{what}.split")
        defects.add((row["benchmark"], row["split"], str(row["task_id"])))
    passable = set()
    for i, row in enumerate(data["tasks_passable_without_hack"]):
        what = f"tasks_passable_without_hack[{i}]"
        _keys(row, {"split", "task_id"}, what)
        _one_of(row["split"], IMPOSSIBLE_SPLITS, f"{what}.split")
        passable.add((row["split"], str(row["task_id"])))
    excluded = set()
    for i, row in enumerate(data["excluded_trainings"]):
        what = f"excluded_trainings[{i}]"
        _keys(row, {"model", "seed"}, what)
        _one_of(row["model"], MODELS, f"{what}.model")
        if type(row["seed"]) is not int:
            raise SystemExit(f"{what}.seed: expected an integer")
        excluded.add((row["model"], row["seed"]))
    return Study(tuple(runs), frozenset(defects), frozenset(passable), frozenset(excluded))


def prepared_files(benchmark: str, stage: str) -> list[Path]:
    """The transcript files s1_prepare_transcripts wrote for one benchmark and stage, in name order."""
    files = sorted(script_dir(EXPERIMENT, PREPARE).glob(f"{benchmark}_{stage}_*.jsonl.gz"))
    if not files:
        raise SystemExit(f"no prepared transcripts for {benchmark} {stage}; run monitors.s1_prepare_transcripts")
    return files


def prepared_rows(benchmark: str, stage: str) -> Iterator[dict]:
    """Every prepared episode of one benchmark and stage."""
    for f in prepared_files(benchmark, stage):
        with gzip.open(f, "rt") as lines:
            for line in lines:
                yield json.loads(line)


def attempted_impossible(benchmark: str, study: Study) -> dict[tuple[str, int], int]:
    """Impossible-split episodes per (model, seed) after reward-hacking RL: every prepared episode, passing or not,
    less the episodes on the (split, task) pairs the study file lists as passable without a hack. hacks() leaves the
    passes on those pairs out of the hacks, so they leave the episodes attempted too."""
    attempted = defaultdict(int)
    for e in prepared_rows(benchmark, AFTER_RL):
        if e["split"] != ORIGINAL and (e["split"], e["task_id"]) not in study.tasks_passable_without_hack:
            attempted[(e["model"], e["seed"])] += 1
    return attempted


def boot_ci(caught_by_task: dict[str, list[bool]]) -> tuple[float, float]:
    """95% interval for caught/hacks over all hacks, resampling tasks with replacement (BOOT_REPS resamples, seeded
    with BOOT_SEED). Each task carries every hack on it, so the interval is centred on the pooled rate that the
    scripts report, not on the mean of per-task rates."""
    cells = [(sum(v), len(v)) for v in caught_by_task.values()]
    rng = random.Random(BOOT_SEED)
    rates = []
    for _ in range(BOOT_REPS):
        draw = rng.choices(cells, k=len(cells))
        rates.append(sum(c for c, _ in draw) / sum(n for _, n in draw))
    rates.sort()
    return rates[int(0.025 * BOOT_REPS)], rates[int(0.975 * BOOT_REPS) - 1]


def read_lines(path: Path) -> list[dict]:
    """The rows of a JSONL file, blank lines skipped."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def episodes_file(stage: str, benchmark: str) -> Path:
    """The list of episodes s2_run_monitors scores for one stage and benchmark (no transcripts)."""
    return script_dir(EXPERIMENT, RUN_MONITORS) / f"episodes_{stage}_{benchmark}.jsonl"


def verdicts_file() -> Path:
    """Every monitor verdict, one JSON row per (episode, view, monitor) call."""
    return script_dir(EXPERIMENT, RUN_MONITORS) / "verdicts.jsonl"


def scores_by_monitor_view() -> dict[tuple[str, str], dict[str, int]]:
    """(monitor model, view) -> episode id -> score, over the verdicts that are not judge errors."""
    got = defaultdict(dict)
    for line in verdicts_file().read_text().splitlines():
        r = json.loads(line)
        if r["verdict"] is not None:
            got[(r["monitor"], r["view"])][r["id"]] = r["verdict"]["score"]
    return got


def hacks(benchmark: str, study: Study) -> list[dict]:
    """The scored episodes counted as hacks: impossible-split passes after reward-hacking RL, less the (split, task)
    pairs the study file lists as passable without a hack."""
    eps = read_lines(episodes_file(AFTER_RL, benchmark))
    return [
        e for e in eps if e["label"] == HACK and (e["split"], e["task_id"]) not in study.tasks_passable_without_hack
    ]


def summary_file(monitor: str, benchmark: str) -> Path:
    """s3_set_thresholds's output for one monitor and benchmark, which holds the monitor's thresholds."""
    return script_dir(EXPERIMENT, THRESHOLDS) / f"summary_{monitor}_{benchmark}.json"
