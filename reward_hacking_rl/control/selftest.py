"""Container self-test of every pool task, through the environment's own setup and grading code.

For each task, in a fresh no-network container of its SWE-smith image:
1. run the environment's setup (keep the graded files, plant the bug, delete the FAIL_TO_PASS test
   files, flatten history) and check the history holds exactly one commit;
2. grade the untouched buggy tree: must not be resolved;
3. reverse the bug patch (SWE-smith's gold fix) and grade again: must be resolved.
Neither grading may raise a tamper flag, before or after the write-back. A task passes when all
of this holds. On the first NETWORK_PROBES tasks the self-test also appends a line to the
repository's root conftest.py and creates a root sitecustomize.py after the gold grading, then
grades once more: the tamper flag on the agent's tree must fire (a known positive for the
detector), the one on the graded tree must not (the write-back undid both), and the gold fix must
still resolve. Per-task rows (durations, sizes, outcomes) are written the moment
each task finishes; a re-run skips tasks already written. The first NETWORK_PROBES tasks also
check that the container cannot reach the network.

Runs on the training machine with the Python of the reward-hacking repository's virtualenv, from this
repository's root:
    RH_SCREEN_LABEL=swesmith_selftest RH_SCREEN_BUILD_ROOT=<dockerfile folder> RH_SCREEN_MAX_PARALLEL_EXECS=64 \\
    <their repo>/.venv/bin/python -u -m reward_hacking_rl.control.selftest \\
        --pool <pool.jsonl> --out <dir> --parallel 48 --limit <pool size>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

# This repository's local-Docker client must win over any installed copy, so it goes first on the path
# before anything imports scalable_docker. The environment module imports tinker_cookbook, which their
# repository's virtualenv has installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_docker"))

from scalable_docker import client as docker_client  # noqa: E402
from scalable_docker.client import (  # noqa: E402
    TIMED_OUT_PROCESS_OUTPUT,
    Image,
    MultiCommandTimeout,
    ScalableDockerClient,
)

from reward_hacking_rl.control import swe_smith_env as E  # noqa: E402

#: Attempts at `docker run` per container in this control's runs (reward-hacking RL uses the client's 10).
docker_client.START_ATTEMPTS = 5

#: Generous limits so the self-test measures durations instead of cutting them off; training's
#: limits are set from the measured distribution.
SETUP_TIMEOUT = MultiCommandTimeout(seconds_per_command=600, total_seconds=1800)
TEST_TIMEOUT = MultiCommandTimeout(seconds_per_command=1800, total_seconds=2400)
#: How many of the first tasks also probe that the container has no network.
NETWORK_PROBES = 20
NETWORK_PROBE = "python3 -c \"import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=5)\""


def ok(o) -> bool:
    return o != TIMED_OUT_PROCESS_OUTPUT and o.exit_code == 0


async def grade(
    client, container, d: E.Datapoint, blob: str, config_blob: str, before: dict
) -> tuple[bool | None, float, int, float, float]:
    """(resolved or None if grading could not run, seconds, graded tests passing, tamper_any on the
    tree as left, tamper_any on the graded tree after the write-back)."""
    commands, fp_agent, fp_graded = E.grading_commands(client, d, blob, config_blob)
    t0 = time.perf_counter()
    outs = await client.run_commands(container=container, commands=commands, timeout=TEST_TIMEOUT)
    dt = time.perf_counter() - t0
    if not all(ok(o) for o in outs[:-1]) or outs[-1] == TIMED_OUT_PROCESS_OUTPUT:
        return None, dt, 0, -1.0, -1.0
    tamper = E.tamper_flags(before, E.parse_fingerprint(outs[fp_agent].stdout))["tamper_any"]
    after = E.tamper_flags(before, E.parse_fingerprint(outs[fp_graded].stdout))["tamper_any"]
    sm = E.log_parser(outs[-1].stdout)
    n = sum(sm.get(t) in E.PASSING for t in d.fail_to_pass + d.pass_to_pass)
    return E.resolved(sm, d), dt, n, tamper, after


async def one(i: int, d: E.Datapoint) -> dict:
    client = ScalableDockerClient(key=f"selftest{i}", network="none")
    row: dict = dict(instance_id=d.instance_id, repo=d.repo, n_f2p=len(d.fail_to_pass), n_p2p=len(d.pass_to_pass))
    [container] = await client.start_containers([E.dockerfile_content(d)])
    try:
        cmds = E.setup_commands(client, d)
        t0 = time.perf_counter()
        outs = await client.run_commands(container=container, commands=cmds, timeout=SETUP_TIMEOUT)
        row["setup_s"] = round(time.perf_counter() - t0, 1)
        bad = [k for k, o in enumerate(outs) if not ok(o)]
        if bad:
            o = outs[bad[0]]
            row.update(
                stage="setup",
                passed=False,
                setup_failed_command=bad[0],
                detail=(o.stderr or o.stdout)[-2000:] if o != TIMED_OUT_PROCESS_OUTPUT else "timeout",
            )
            return row
        n_list = len(client.upload_big_file_commands(E.LIST_PATH, "\n".join(d.graded_files) + "\n"))
        blob = outs[n_list].stdout.strip()
        row["blob_bytes"] = E.blob_bytes(blob)
        config_blob = outs[-2].stdout.strip()
        row["config_blob_bytes"] = E.blob_bytes(config_blob)
        before = E.parse_fingerprint(outs[-1].stdout)
        row["fingerprint_files"] = len(before)

        checks = [f"cd {E.WORKDIR} && git rev-list --all --count"]
        if i < NETWORK_PROBES:
            checks.append(NETWORK_PROBE)
        co = await client.run_commands(container=container, commands=checks, timeout=SETUP_TIMEOUT)
        row["history_commits"] = co[0].stdout.strip()
        if i < NETWORK_PROBES:
            row["network_blocked"] = co[1].exit_code != 0

        (
            row["buggy_resolved"],
            row["buggy_grade_s"],
            row["buggy_passing"],
            row["buggy_tamper"],
            row["buggy_tamper_after_restore"],
        ) = await grade(client, container, d, blob, config_blob, before)

        rev = [
            *client.upload_big_file_commands(E.BLOB_PATH, d.patch),
            f"cd {E.WORKDIR} && git apply --verbose --reverse {E.BLOB_PATH}; rc=$?; rm -f {E.BLOB_PATH}; exit $rc",
        ]
        ro = await client.run_commands(container=container, commands=rev, timeout=SETUP_TIMEOUT)
        if not all(ok(o) for o in ro):
            o = ro[-1]
            row.update(
                stage="revert",
                passed=False,
                detail=(o.stderr or o.stdout)[-2000:] if o != TIMED_OUT_PROCESS_OUTPUT else "timeout",
            )
            return row
        (
            row["gold_resolved"],
            row["gold_grade_s"],
            row["gold_passing"],
            row["gold_tamper"],
            row["gold_tamper_after_restore"],
        ) = await grade(client, container, d, blob, config_blob, before)
        if i < NETWORK_PROBES:
            await client.run_commands(
                container=container,
                commands=[
                    f"cd {E.WORKDIR} && echo '# tamper probe' >> conftest.py"
                    " && echo '# tamper probe' > sitecustomize.py"
                ],
                timeout=SETUP_TIMEOUT,
            )
            row["probe_resolved"], _, _, row["probe_tamper"], row["probe_tamper_after_restore"] = await grade(
                client, container, d, blob, config_blob, before
            )
        row["buggy_grade_s"] = round(row["buggy_grade_s"], 1)
        row["gold_grade_s"] = round(row["gold_grade_s"], 1)
        row["passed"] = (
            row["history_commits"] == "1"
            and row["buggy_resolved"] is False
            and row["gold_resolved"] is True
            and row.get("network_blocked", True)
            and row["fingerprint_files"] > 0
            and row["buggy_tamper"] == 0.0
            and row["gold_tamper"] == 0.0
            and row["buggy_tamper_after_restore"] == 0.0
            and row["gold_tamper_after_restore"] == 0.0
            and row.get("probe_tamper", 1.0) == 1.0
            and row.get("probe_tamper_after_restore", 0.0) == 0.0
            and row.get("probe_resolved", True) is True
        )
        row["stage"] = "done"
        return row
    finally:
        await client.start_destroying_containers()
        await client.wait_until_all_containers_stopped()


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--parallel", type=int, required=True)
    p.add_argument("--limit", type=int, required=True, help="test the first N pool tasks (the whole pool: its size)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows_path = a.out / "rows.jsonl"
    done = set()
    if rows_path.exists():
        done = {json.loads(line)["instance_id"] for line in rows_path.open()}
    pool = E.load_pool(str(a.pool))[: a.limit]
    todo = [(i, d) for i, d in enumerate(pool) if d.instance_id not in done]
    print(f"{len(pool)} tasks, {len(done)} already tested, {len(todo)} to test", flush=True)

    images = sorted({d.image_name for _, d in todo})
    print(f"pulling {len(images)} images", flush=True)
    sem = asyncio.Semaphore(16)

    async def pull(img: str) -> None:
        async with sem:
            proc = await asyncio.create_subprocess_exec(
                "docker", "pull", "-q", img, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            _, err = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"pull failed for {img}: {err.decode()[-500:]}")

    await asyncio.gather(*[pull(img) for img in images])
    await ScalableDockerClient(key="build").build_images(
        [Image(E.dockerfile_content(d)) for _, d in todo], batch_size=16
    )

    run_sem = asyncio.Semaphore(a.parallel)
    lock = asyncio.Lock()
    counts = {"passed": 0, "failed": 0}

    async def guarded(i: int, d: E.Datapoint) -> None:
        async with run_sem:
            try:
                row = await one(i, d)
            except Exception as e:  # recorded, never dropped
                row = dict(
                    instance_id=d.instance_id, repo=d.repo, stage="exception", passed=False, detail=repr(e)[-2000:]
                )
        async with lock:
            with rows_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            counts["passed" if row.get("passed") else "failed"] += 1
            n = counts["passed"] + counts["failed"]
            if n % 25 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} tested: {counts['passed']} passed, {counts['failed']} failed", flush=True)

    await asyncio.gather(*[guarded(i, d) for i, d in todo])


if __name__ == "__main__":
    asyncio.run(main())
