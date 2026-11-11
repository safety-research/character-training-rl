"""Render the ImpossibleBench evaluation episodes as monitor transcripts, with the reasoning shown and hidden.

Reads the Inspect logs listed in the study file for one benchmark and writes one file of episodes per log, each
episode with its two renderings (monitors.transcript.views). Which episodes of a log are kept depends on its stage:

- After reward-hacking RL (the original, oneoff and conflicting splits of each model and seed): every usable episode,
  passing or not. An episode is left out when its task is listed in the study file's `environment_defect_tasks`, when
  it errored or has no score, when its final output was cut off (stop reason `max_tokens` or `model_length`), and,
  on Impossible-LiveCodeBench, when it was graded unusable or ended at the context window, on Impossible-SWEbench,
  when it ended at the context window or its grader timed out. The log itself must have status `success`. The
  failing episodes are kept because the later scripts count every impossible-split episode attempted.
- Before reward-hacking RL (the original split of each model): every episode the scorer marks resolved. An episode
  that errored or has no score is left out and counted. These are the passes each monitor's threshold is set on.

An environment-defect task is left out only of the impossible split it is listed under, on both benchmarks; nothing
is left out of the original split, whose tests are unedited.

    uv run python -m monitors.s1_prepare_transcripts --study monitors/study.json --benchmark iblcb

Writes results/monitors/scripts/s1_prepare_transcripts/<run>.jsonl.gz and <run>.counts.json (episodes kept and
left out, by reason). A run whose file exists is skipped; delete the file to render it again. No API calls.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries, read_eval_log_samples_by_id

from monitors.common import (
    AFTER_RL,
    BEFORE_RL,
    BENCHMARKS,
    EXPERIMENT,
    PREPARE,
    SCORER,
    Run,
    Study,
    load_study,
)
from monitors.transcript import encode, views
from src.exp.manifest import script_dir

READ_CHUNK_SIZE = 16
"""Episodes read from a log per call."""
READ_CONCURRENCY = 4
EXCLUDED_FIELDS = {"events", "store"}
"""Sample fields not read: the transcript is the sample's messages."""
CUT_OFF_STOP_REASONS = ("max_tokens", "model_length")
"""Stop reasons of a final output that was cut off; such an episode is left out."""


def episode_id(key: str) -> str:
    """A stable id for an episode from its key (<run>/<sample id>/<epoch>)."""
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def usable_lcb(run: Run, study: Study, drops: collections.Counter) -> list[dict]:
    """The usable Impossible-LiveCodeBench episodes of a log after reward-hacking RL."""
    excluded = {t for b, s, t in study.environment_defect_tasks if b == "iblcb" and s == run.split}
    log = read_eval_log(run.log)
    if log.status != "success":
        raise ValueError(f"unsuccessful log: {run.log}")
    rows = []
    for sample in log.samples or []:
        task = str(sample.metadata["task_id"])
        if task in excluded:
            drops["environment_defect"] += 1
            continue
        score = (sample.scores or {}).get(SCORER["iblcb"])
        if sample.error or score is None:
            drops["episode_error"] += 1
            continue
        value = score.value
        if value["usable"] != 1 or value["context_full"] != 0 or not math.isfinite(value["resolved"]):
            drops["unusable_or_context_full"] += 1
            continue
        if sample.output.stop_reason == "max_tokens":
            drops["cut_off_final_output"] += 1
            continue
        rows.append({"task_id": task, "sample_id": sample.id, "epoch": sample.epoch, "passed": value["resolved"] == 1})
    return rows


def usable_swe(run: Run, study: Study, drops: collections.Counter) -> list[dict]:
    """The usable Impossible-SWEbench episodes of a log after reward-hacking RL, from the log's sample summaries."""
    excluded = {t for b, s, t in study.environment_defect_tasks if b == "ibswe" and s == run.split}
    log = read_eval_log(run.log, header_only=True)
    if log.status != "success":
        raise ValueError(f"unsuccessful log: {run.log}")
    rows = []
    for sample in read_eval_log_sample_summaries(run.log):
        task = str(sample.id)
        if task in excluded:
            drops["environment_defect"] += 1
            continue
        score = (sample.scores or {}).get(SCORER["ibswe"])
        if sample.error or score is None:
            drops["episode_error"] += 1
            continue
        v = score.value
        if v["context_full"] != 0:
            drops["context_full"] += 1
            continue
        if v["timed_out"] != 0:
            drops["grader_timeout"] += 1
            continue
        if not math.isfinite(v["resolved"]):
            raise ValueError(f"{run.log}/{task}: invalid score")
        rows.append({"task_id": task, "sample_id": sample.id, "epoch": sample.epoch, "passed": v["resolved"] == 1})
    return rows


USABLE = {"iblcb": usable_lcb, "ibswe": usable_swe}


def row_for(run: Run, sample, task_id: str, passed: bool) -> dict:
    """One stored episode: who produced it, whether it passed, and its two renderings."""
    rendered, native_chars, think_chars = views(sample)
    key = f"{run.name}/{sample.id}/{sample.epoch}"
    return {
        "id": episode_id(key),
        "key": key,
        "benchmark": run.benchmark,
        "model": run.model,
        "seed": run.seed,
        "stage": run.stage,
        "split": run.split,
        "task_id": task_id,
        "epoch": sample.epoch,
        "passed": passed,
        "source": str(run.log),
        "messages": len(sample.messages),
        "reasoning_channel_characters": native_chars,
        "think_tool_characters": think_chars,
        "views": rendered,
    }


def after_rl_rows(run: Run, study: Study, drops: collections.Counter):
    """The stored episodes of a log after reward-hacking RL."""
    usable = USABLE[run.benchmark](run, study, drops)
    for start in range(0, len(usable), READ_CHUNK_SIZE):
        chunk = usable[start : start + READ_CHUNK_SIZE]
        samples = read_eval_log_samples_by_id(
            run.log,
            [(r["sample_id"], r["epoch"]) for r in chunk],
            concurrency=READ_CONCURRENCY,
            resolve_attachments=True,
            exclude_fields=EXCLUDED_FIELDS,
        )
        for r, sample in zip(chunk, samples, strict=True):
            if sample.output.stop_reason in CUT_OFF_STOP_REASONS:
                drops["cut_off_final_output"] += 1
                continue
            yield row_for(run, sample, r["task_id"], r["passed"])


def before_rl_rows(run: Run, drops: collections.Counter):
    """The stored episodes of a log before reward-hacking RL: every episode the scorer marks resolved. An episode
    that errored or has no score is left out and counted, as in the logs after reward-hacking RL."""
    log = read_eval_log(run.log)
    for s in log.samples or []:
        score = (s.scores or {}).get(SCORER[run.benchmark])
        if s.error or score is None:
            drops["episode_error"] += 1
            continue
        if score.value["resolved"] != 1.0:
            drops["not_resolved"] += 1
            continue
        yield row_for(run, s, s.id, True)


async def bulk_json_excluding(reader, member, exclude_fields):
    """Read a log member as one JSON document and then drop the excluded fields."""
    data = json.loads(await reader.read_member_fully(member))
    for field in exclude_fields:
        data.pop(field, None)
    return data


def use_bulk_json_reader(run: Run, study: Study) -> None:
    """Swap Inspect's streaming reader of a sample (which skips excluded fields token by token) for
    bulk_json_excluding, which is much faster on samples with very large event trees. The first usable episode of
    `run` is rendered with both readers and the renderings must be equal."""
    from inspect_ai.log._recorders import eval as recorder

    if not hasattr(recorder, "_read_member_json_excluding"):
        raise SystemExit("--bulk-json-reader: this inspect_ai version has no _read_member_json_excluding to replace")
    usable = USABLE[run.benchmark](run, study, collections.Counter())
    if not usable:
        raise SystemExit(f"--bulk-json-reader: {run.name} has no usable episode to compare the two readers on")
    probe = [(usable[0]["sample_id"], usable[0]["epoch"])]

    def render():
        (sample,) = read_eval_log_samples_by_id(
            run.log, probe, concurrency=READ_CONCURRENCY, resolve_attachments=True, exclude_fields=EXCLUDED_FIELDS
        )
        return views(sample)

    expected = render()
    recorder._read_member_json_excluding = bulk_json_excluding
    if render() != expected:
        raise SystemExit("--bulk-json-reader changes the rendered transcript; run without it")
    print(f"bulk JSON reader matches the standard reader on {run.name} {probe[0]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", type=Path, required=True, help="the study file (see monitors/README.md)")
    ap.add_argument("--benchmark", choices=tuple(BENCHMARKS), required=True)
    ap.add_argument(
        "--bulk-json-reader",
        action="store_true",
        help="read each sample as one JSON document (faster on Impossible-SWEbench logs); checked against the "
        "standard reader on one episode first",
    )
    args = ap.parse_args()
    study = load_study(args.study)
    runs = [r for r in study.runs if r.benchmark == args.benchmark]
    if not runs:
        raise SystemExit(f"the study file lists no {args.benchmark} run")
    missing = [str(r.log) for r in runs if not r.log.is_file()]
    if missing:
        raise SystemExit(f"log files not found: {missing}")
    out = script_dir(EXPERIMENT, PREPARE)
    out.mkdir(parents=True, exist_ok=True)
    if args.bulk_json_reader:
        after = [r for r in runs if r.stage == AFTER_RL]
        if not after:
            raise SystemExit("--bulk-json-reader applies to runs after reward-hacking RL; the study file lists none")
        use_bulk_json_reader(after[0], study)
    for run in runs:
        path = out / f"{run.name}.jsonl.gz"
        if path.exists():
            print(f"{run.name}: already prepared, skipped", flush=True)
            continue
        drops, kept, passed = collections.Counter(), 0, 0
        rows = before_rl_rows(run, drops) if run.stage == BEFORE_RL else after_rl_rows(run, study, drops)
        partial = path.with_name(path.name + ".partial")
        with gzip.open(partial, "wb") as saved:
            for row in rows:
                saved.write(encode(row) + b"\n")
                kept += 1
                passed += row["passed"]
        partial.rename(path)
        counts = {"run": run.name, "source": str(run.log), "episodes": kept, "passed": passed, "left_out": dict(drops)}
        (out / f"{run.name}.counts.json").write_text(json.dumps(counts, indent=2) + "\n")
        print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
