#!/bin/bash
# Wrapper every training command in configs/ runs through.
#   launch.sh <output folder> <command ...>
# - raises the open-file limit (hundreds of `docker` subprocesses, each with its own pipes);
# - refuses an output folder that already holds a checkpoint, because the trainer would resume it;
#   set RESUME=yes to resume on purpose (the records then hold the abandoned update's lines too);
# - exempts the trainer from the kernel's out-of-memory killer (needs sudo): the local-Docker client
#   starts task containers without memory limits, and when runaway task processes fill the machine's
#   memory the kernel otherwise kills the trainer, the largest single process.
set -euo pipefail
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true
OUT=$1; shift
if [ -e "$OUT/train/checkpoints.jsonl" ] && [ "${RESUME:-no}" != yes ]; then
  echo "$OUT already holds a checkpoint; refusing to resume it (set RESUME=yes to resume)" >&2
  exit 2
fi
mkdir -p "$OUT"
echo -1000 | sudo tee /proc/$$/oom_score_adj > /dev/null
exec "$@"
