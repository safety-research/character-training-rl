"""Drop-in replacement for the `scalable_docker.client` module the reward-hacking environments
import: a local-Docker implementation of the public interface of `astOwOlfo/scalable_docker`, backed
by the local Docker daemon instead of Kubernetes.

The published library (`astOwOlfo/scalable_docker`) starts each container as a Kubernetes deployment and runs every command
as `kubectl exec deployment/<name> -- timeout <s> /bin/bash -c <command>`. This module keeps the
same public names and the same command semantics, and swaps the transport:

- a container is `docker run -d <image> /bin/bash -c "tail -f /dev/null"`, the published
  deployment's own command;
- a command is `docker exec <name> timeout <s> /bin/bash -c <command>`, wrapped in an outer
  `timeout` of 2*s+8 seconds exactly as the published client does;
- images are built locally and never pushed.

The model sees the same command outputs either way. Containers carry the label
`rh-screen=<run label>` so a killed run's containers can be found and removed.
"""

from __future__ import annotations

import asyncio
import base64
import os
import resource
from dataclasses import dataclass, field
from hashlib import sha256
from shlex import quote
from sys import stderr
from time import perf_counter
from typing import Any
from uuid import uuid4

# Raise the open-file soft limit to the hard limit: this driver runs many `docker` subprocesses,
# each with its own pipes, and the default 1024 is exhausted when building or execing at scale.
try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))
except (ValueError, OSError):
    pass

#: label key on every container this module starts; the value is RH_SCREEN_LABEL from the env
LABEL_KEY = "rh-screen"
#: Attempts at `docker run` per container, and the wait before retry k (k times this many seconds).
START_ATTEMPTS = 10
START_RETRY_SECONDS = 5.0


def _run_label() -> str:
    label = os.environ.get("RH_SCREEN_LABEL")
    if not label:
        raise RuntimeError("RH_SCREEN_LABEL is not set; it labels this run's containers")
    return label


def _max_parallel_execs() -> int:
    raw = os.environ.get("RH_SCREEN_MAX_PARALLEL_EXECS")
    if not raw:
        raise RuntimeError("RH_SCREEN_MAX_PARALLEL_EXECS is not set")
    return int(raw)


@dataclass(frozen=True, slots=True)
class ProcessOutput:
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class Image:
    dockerfile_content: str
    max_cpus: float | int = 1
    max_memory_gigabytes: float | int = 1.0


ContainerId = int


@dataclass(frozen=True, slots=True)
class Container:
    id: ContainerId
    deployment_name: str
    dockerfile_content: str


@dataclass(frozen=True, slots=True)
class ScalableDockerServerError(Exception):
    server_response: Any


@dataclass(frozen=True, slots=True)
class MultiCommandTimeout:
    seconds_per_command: int | float
    total_seconds: int | float


TIMED_OUT_PROCESS_OUTPUT = ProcessOutput(exit_code=124, stdout="", stderr="timed out")

_exec_semaphore: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    global _exec_semaphore
    if _exec_semaphore is None:
        _exec_semaphore = asyncio.Semaphore(_max_parallel_execs())
    return _exec_semaphore


async def run_command(*command: str, assert_success: bool = True) -> ProcessOutput:
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await process.communicate()
    assert process.returncode is not None
    if assert_success and process.returncode != 0:
        raise RuntimeError(
            f"Command {command} failed. Exit code: {process.returncode}\n"
            f"STDOUT: {out.decode(errors='replace')}\nSTDERR: {err.decode(errors='replace')}"
        )
    return ProcessOutput(
        exit_code=process.returncode,
        stdout=out.decode(errors="ignore"),
        stderr=err.decode(errors="ignore"),
    )


def image_name(dockerfile_content: str) -> str:
    """Same naming as the published client, without a registry prefix."""
    return f"rh-image-{sha256(dockerfile_content.encode()).hexdigest()[:32]}"


async def image_exists(dockerfile_content: str) -> bool:
    out = await run_command("docker", "image", "inspect", image_name(dockerfile_content), assert_success=False)
    return out.exit_code == 0


async def build_image(dockerfile_content: str, build_root: str) -> None:
    directory = os.path.join(build_root, image_name(dockerfile_content))
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "Dockerfile"), "w") as f:
        f.write(dockerfile_content)
    await run_command("docker", "build", "-q", "-t", image_name(dockerfile_content), directory)


async def delete_all_scalable_docker_kubernetes_deployments() -> None:
    """The environments call this at startup to clear a cluster; locally it removes this run's
    labelled containers."""
    out = await run_command("docker", "ps", "-aq", "--filter", f"label={LABEL_KEY}={_run_label()}")
    ids = out.stdout.split()
    if ids:
        await run_command("docker", "rm", "-f", *ids, assert_success=False)


@dataclass(slots=True)
class ScalableDockerClient:
    key: str
    #: `docker run --network` value for this client's containers; None keeps Docker's default
    #: bridge network, as the published client's deployments have.
    network: str | None = None
    max_command_length: int = 65536
    max_parallel_commands: int | None = None
    containers: dict[ContainerId, Container] = field(default_factory=dict)
    stopped_container_ids: set[int] = field(default_factory=set)
    ready_tasks: dict[ContainerId, asyncio.Task] = field(default_factory=dict)
    stop_tasks: dict[ContainerId, asyncio.Task] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    create_containers_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    container_id_counter: int = 0

    async def new_container_id(self) -> ContainerId:
        async with self.lock:
            cid = self.container_id_counter
            self.container_id_counter += 1
            return cid

    async def build_images(self, images: list[Image], batch_size: int | None = None, max_retries: int = 1) -> None:
        build_root = os.environ.get("RH_SCREEN_BUILD_ROOT")
        if not build_root:
            raise RuntimeError("RH_SCREEN_BUILD_ROOT is not set")
        contents = sorted(set(image.dockerfile_content for image in images))
        # `docker image inspect` is a subprocess per image; unbounded gather over thousands of
        # images exhausts the file-descriptor limit (OSError: Too many open files). Bound both the
        # existence checks and the builds by batch_size (default 16).
        cap = batch_size or 16
        check_sem = asyncio.Semaphore(cap)

        async def _exists(c: str) -> bool:
            async with check_sem:
                return await image_exists(c)

        exists = await asyncio.gather(*[_exists(c) for c in contents])
        todo = [c for c, e in zip(contents, exists) if not e]
        sem = asyncio.Semaphore(cap)

        async def one(c: str) -> None:
            async with sem:
                await build_image(c, build_root)

        await asyncio.gather(*[one(c) for c in todo])

    async def _start(self, container: Container) -> None:
        """`docker run`, retried: when hundreds of containers start at once Docker refuses some
        ("unable to apply cgroup configuration"). A failed attempt's half-created container is
        removed before the next; every retry is printed, and the last failure raises."""
        command = (
            "docker",
            "run",
            "-d",
            "--name",
            container.deployment_name,
            "--label",
            f"{LABEL_KEY}={_run_label()}",
            *(["--network", self.network] if self.network is not None else []),
            image_name(container.dockerfile_content),
            "/bin/bash",
            "-c",
            "tail -f /dev/null",
        )
        for attempt in range(1, START_ATTEMPTS + 1):
            out = await run_command(*command, assert_success=False)
            if out.exit_code == 0:
                return
            await run_command("docker", "rm", "-f", container.deployment_name, assert_success=False)
            if attempt == START_ATTEMPTS:
                raise RuntimeError(
                    f"Command {command} failed {START_ATTEMPTS} times. Exit code: {out.exit_code}\n"
                    f"STDOUT: {out.stdout}\nSTDERR: {out.stderr}"
                )
            print(
                f"Scalable Docker: start of {container.deployment_name} failed (attempt {attempt} of "
                f"{START_ATTEMPTS}), retrying: {out.stderr.strip()[-300:]}",
                file=stderr,
            )
            await asyncio.sleep(START_RETRY_SECONDS * attempt)

    async def wait_until_all_containers_stopped(self) -> None:
        for cid, task in self.stop_tasks.items():
            await task
            async with self.lock:
                self.stopped_container_ids.add(cid)

    async def start_containers(self, dockerfile_contents: list[str]) -> list[Container]:
        async with self.create_containers_lock:
            await self.start_destroying_containers()
            await self.wait_until_all_containers_stopped()
            assert set(self.stopped_container_ids) == set(self.containers.keys())
            ids = [await self.new_container_id() for _ in dockerfile_contents]
            async with self.lock:
                self.containers = {
                    cid: Container(
                        id=cid,
                        deployment_name=f"rh-{self.key}-{uuid4().hex[:12]}",
                        dockerfile_content=content,
                    )
                    for cid, content in zip(ids, dockerfile_contents, strict=True)
                }
                self.stopped_container_ids = set()
                self.stop_tasks = {}
            self.ready_tasks = {cid: asyncio.create_task(self._start(c)) for cid, c in self.containers.items()}
            return [self.containers[cid] for cid in ids]

    async def start_destroying_container(self, container: Container) -> None:
        async with self.lock:
            assert container.id in self.containers, "container has already been destroyed"
            if container.id in self.stop_tasks:
                return
            self.stop_tasks[container.id] = asyncio.create_task(
                run_command("docker", "rm", "-f", container.deployment_name, assert_success=False)
            )

    async def start_destroying_containers(self) -> None:
        for container in list(self.containers.values()):
            await self.start_destroying_container(container)

    async def _run_single_command(self, command: str, container: Container, timeout_seconds: float) -> ProcessOutput:
        if len(command) > self.max_command_length:
            print(
                f"Scalable Docker Warning: Truncating long command of length {len(command)} "
                f"to length {self.max_command_length}.",
                file=stderr,
            )
            command = command[: self.max_command_length]
        longer_timeout = 2 * timeout_seconds + 8
        return await run_command(
            "timeout",
            str(longer_timeout),
            "docker",
            "exec",
            container.deployment_name,
            "timeout",
            str(timeout_seconds),
            "/bin/bash",
            "-c",
            command,
            assert_success=False,
        )

    async def run_commands(
        self,
        container: Container,
        commands: list[str],
        timeout: MultiCommandTimeout,
        blocking: bool = False,
    ) -> list[ProcessOutput]:
        if blocking:
            raise NotImplementedError("blocking=True is not supported")
        assert container.id in self.containers, "container has already been destroyed"
        await self.ready_tasks[container.id]
        async with _semaphore():
            outputs: list[ProcessOutput] = []
            start = perf_counter()
            for command in commands:
                remaining = timeout.total_seconds - (perf_counter() - start)
                if remaining <= 0:
                    outputs.append(TIMED_OUT_PROCESS_OUTPUT)
                    continue
                outputs.append(
                    await self._run_single_command(command, container, min(remaining, timeout.seconds_per_command))
                )
            return outputs

    def upload_big_file_commands(self, filename: str, content: str) -> list[str]:
        data = content.encode()
        size = self.max_command_length // 2
        chunks = [base64.b64encode(data[i : i + size]).decode() for i in range(0, len(data), size)]
        return [f"echo {chunks[0]} | base64 -d > {quote(filename)}"] + [
            f"echo {c} | base64 -d >> {quote(filename)}" for c in chunks[1:]
        ]


def upload_file_command(filename: str, content: str) -> str:
    return f"echo {base64.b64encode(content.encode()).decode()} | base64 -d > {quote(filename)}"
