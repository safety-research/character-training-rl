"""Path arguments of the trainers, made absolute before the trainer changes directory.

The trainers change into the external repository (`--repo`) because its loaders read `data/` relative to its root.
A path the reader gave relative to the directory the command was run from must still mean that file afterwards, so
each trainer calls `absolute_paths` on its path arguments first. This module imports nothing outside the standard
library.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

NOT_SET = "none"
"""The value a trainer's optional path flag takes when it is not set; it is left as it is."""


def absolute_paths(args: argparse.Namespace, names: tuple[str, ...], launch_dir: Path | None = None) -> None:
    """Replace each named attribute of `args` with its absolute path, in place, keeping its type (Path or str).

    A relative path is taken against `launch_dir`, the current directory when None. Symbolic links are kept as
    given (a virtualenv's `python` is a link that must be run by its own path), and `none` stays `none`."""
    base = Path.cwd() if launch_dir is None else Path(launch_dir)
    for name in names:
        value = getattr(args, name)
        if value == NOT_SET:
            continue
        absolute = os.path.normpath(os.path.join(base, value))
        setattr(args, name, Path(absolute) if isinstance(value, Path) else absolute)
