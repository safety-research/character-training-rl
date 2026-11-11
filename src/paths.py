"""One home for everything the code reads and writes that is not code.

Results, generated data, downloaded benchmark files and the benchmark checkouts under
`third_party/` live under the repository root by default. Set `CHARACTER_TRAINING_RL_HOME` to
keep them somewhere else (a larger disk, a shared folder); the code then reads and writes there.
"""

from __future__ import annotations

import os
from pathlib import Path

HOME_VAR = "CHARACTER_TRAINING_RL_HOME"
#: the checkout this code runs from
CODE_ROOT = Path(__file__).resolve().parents[1]


def _home() -> Path:
    raw = os.environ.get(HOME_VAR)
    if raw is None:
        return CODE_ROOT
    home = Path(raw).expanduser()
    if not home.is_dir():
        raise FileNotFoundError(f"{HOME_VAR}={raw!r} is not a directory")
    return home.resolve()


HOME = _home()
RESULTS = HOME / "results"
DATA = HOME / "data"
THIRD_PARTY = HOME / "third_party"
CACHE = HOME / ".cache"
ENV_FILE = CODE_ROOT / ".env"
