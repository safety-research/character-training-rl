"""Read a training run directory and hand back checkpoints that know how to be sampled.

The trainers write `checkpoints.jsonl`; this is its reader. Checkpoints are selected by each
row's `type` field, never by a substring of the URI, and a run with no final sampler row is an
error: the last row of a log can be the optimizer STATE, which cannot be sampled at all.

A checkpoint otherwise arrives at eval time as a bare `tinker://` URI, and the caller would have
to re-type the base model, tokenizer and renderer by hand. Those facts are already recorded:
every run directory has a `config.json` next to its `checkpoints.jsonl` with `model` and
`renderer_name` in it. Reading them here removes the re-typing and with it renderer mismatches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.jsonl import read_jsonl
from src.training.artifacts import CHECKPOINTS_LOG
from src.training.artifacts import CONFIG_JSON as CONFIG_FILE

SAMPLER = "sampler"
STATE = "state"


@dataclass(frozen=True)
class Checkpoint:
    """One saved checkpoint, plus everything needed to sample it."""

    path: str
    step: int
    kind: str  # SAMPLER | STATE
    final: bool
    run_dir: Path
    base_model: str | None
    renderer_name: str | None

    @property
    def samplable(self) -> bool:
        return self.kind == SAMPLER

    def sampling_kwargs(self) -> dict:
        """Exactly what `create_generator` needs for a Tinker checkpoint."""
        if self.base_model is None:
            raise ValueError(
                f"{self.run_dir}/{CONFIG_FILE} records no base model, so {self.path} cannot be "
                f"sampled without one being supplied explicitly."
            )
        return {
            "backend": "tinker",
            "base_model": self.base_model,
            "tokenizer_id": self.base_model,
            "renderer_name": self.renderer_name,
        }


def read_config(run_dir: str | Path) -> dict:
    path = Path(run_dir) / CONFIG_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_checkpoints(run_dir: str | Path) -> list[Checkpoint]:
    """Every checkpoint a run recorded, in log order.

    Rows without an explicit `type` are rejected rather than guessed at: inferring the kind from
    the URI can return an optimizer state where a sampler was wanted.
    """
    run_dir = Path(run_dir)
    log = run_dir / CHECKPOINTS_LOG
    if not log.exists():
        raise FileNotFoundError(f"{log}: no checkpoint log; is {run_dir} a training run directory?")
    config = read_config(run_dir)
    base_model = config.get("model")
    renderer_name = config.get("renderer_name")

    out = []
    for i, row in enumerate(read_jsonl(log)):
        if "type" not in row:
            raise ValueError(
                f"{log}:{i + 1}: row has no 'type'; this reader only understands the typed "
                f"trainer schema, and will not infer a checkpoint kind from its URI."
            )
        if "path" not in row or "step" not in row:
            raise ValueError(f"{log}:{i + 1}: row needs 'path' and 'step', got {sorted(row)}")
        out.append(
            Checkpoint(
                path=row["path"],
                step=int(row["step"]),
                kind=row["type"],
                final=bool(row.get("final", False)),
                run_dir=run_dir,
                base_model=base_model,
                renderer_name=renderer_name,
            )
        )
    return out


def _samplers(run_dir: str | Path) -> list[Checkpoint]:
    return [c for c in read_checkpoints(run_dir) if c.samplable]


def final_sampler(run_dir: str | Path) -> Checkpoint:
    """The sampler weights to evaluate. Raises if the run never reached a final checkpoint."""
    samplers = _samplers(run_dir)
    if not samplers:
        raise ValueError(f"{run_dir}: no sampler checkpoints recorded")
    finals = [c for c in samplers if c.final]
    if not finals:
        raise ValueError(
            f"{run_dir}: no checkpoint is marked final — the run was interrupted. Its last "
            f"sampler is step {samplers[-1].step}; ask for it explicitly with select(..., 'last') "
            f"if an incomplete run is what you want."
        )
    if len(finals) > 1:
        raise ValueError(f"{run_dir}: {len(finals)} checkpoints marked final: {[c.step for c in finals]}")
    return finals[0]


def final_state(run_dir: str | Path) -> Checkpoint:
    """The optimizer state to resume or initialise from (sampler weights cannot be loaded)."""
    states = [c for c in read_checkpoints(run_dir) if c.kind == STATE]
    if not states:
        raise ValueError(f"{run_dir}: no optimizer state checkpoints recorded")
    return states[-1]


def select(run_dir: str | Path, spec: str = "final") -> list[Checkpoint]:
    """Samplers by spec: 'final', 'last', 'all', or 'every:N' for a dose-response sweep."""
    samplers = _samplers(run_dir)
    if not samplers:
        raise ValueError(f"{run_dir}: no sampler checkpoints recorded")
    if spec == "final":
        return [final_sampler(run_dir)]
    if spec == "last":
        return [samplers[-1]]
    if spec == "all":
        return samplers
    if spec.startswith("every:"):
        n = int(spec.split(":", 1)[1])
        if n <= 0:
            raise ValueError(f"every:N needs a positive N, got {n}")
        picked = [c for c in samplers if c.step % n == 0]
        last = samplers[-1]
        if last not in picked:
            picked.append(last)
        return picked
    raise ValueError(f"unknown checkpoint spec {spec!r}; use final | last | all | every:N")


def find_runs(root: str | Path) -> dict[str, Path]:
    """Every training run directory under `root`, keyed by directory name.

    A run directory is one containing a checkpoint log, so this discovers models by walking
    results/ rather than by consulting a hand-maintained table.
    """
    root = Path(root)
    return {p.parent.name: p.parent for p in sorted(root.rglob(CHECKPOINTS_LOG))}
