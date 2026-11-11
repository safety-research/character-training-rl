"""Where results are written: Inspect logs and the outputs of the pipeline scripts."""

from __future__ import annotations

from pathlib import Path

from src import paths

RESULTS = paths.RESULTS


def inspect_logs() -> Path:
    """One root for every Inspect log, so one viewer shows every evaluation run."""
    return RESULTS / "inspect"


SCRIPTS = "scripts"


def script_dir(experiment: str, script: str) -> Path:
    """Where a pipeline script writes: `results/<experiment>/scripts/<script>/`."""
    if not experiment or not script or "/" in experiment or "/" in script:
        raise ValueError(f"experiment and script are single path components, got {experiment!r}, {script!r}")
    return RESULTS / experiment / SCRIPTS / script
