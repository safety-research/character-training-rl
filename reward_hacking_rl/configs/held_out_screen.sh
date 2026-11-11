#!/bin/bash
# The held-out screen, the task selection held out of reward-hacking RL: 65 tasks per stream, drawn
# mechanically with seed 20261002 from each environment's source file, after dropping candidates whose
# Docker image no longer builds. Only selection.json is used by training; the script also samples gpt-oss-20b on the selected
# tasks (4 episodes each, 9 tool calls, their big_run settings), which costs Tinker sampling. With
# BUILD_ONLY=yes it builds the images, writes selection.json and stops before sampling.
#
# The selection our runs used came from this command, with two differences that this script cannot
# reproduce (see SETUP.md): it predates the filter that drops IOI problems whose tests all expect one
# output, so its two IOI lists were drawn from the unfiltered pool; and which images fail to build
# changes over time.
#
# Needs in the environment: TINKER_API_KEY; THEIR_REPO, RESULTS, DOCKER_BUILD_ROOT as in
# reward_hacking_rl.sh. RH_SCREEN_MAX_PARALLEL_EXECS=64 is the training runs' value.
# Run from this repository's root.
set -euo pipefail
export RH_SCREEN_LABEL=held_out_screen RH_SCREEN_MAX_PARALLEL_EXECS=64 RH_SCREEN_BUILD_ROOT=$DOCKER_BUILD_ROOT
OUT=$RESULTS/held_out_screen
mkdir -p "$OUT"
exec "$THEIR_REPO/.venv/bin/python" -u -m reward_hacking_rl.screen_held_out \
  --repo "$THEIR_REPO" --out "$OUT" \
  --model openai/gpt-oss-20b --renderer gpt_oss_medium_reasoning --renderer-current-date 2026-08-20 \
  --streams synthetic_partial synthetic_binary swe_fixer ae ioi_noreset ioi_reset \
  --run-streams synthetic_partial synthetic_binary swe_fixer ae ioi_noreset ioi_reset \
  --tasks-per-stream 65 --group-size 4 --batch-size 5 --max-steps 9 \
  --context-length 32768 --max-completion-tokens 8192 --temperature 1.0 --seed 20261002 \
  --synthetic-path "$THEIR_REPO/data/final-harder-merge-smaller.jsonl" --ae-path "$THEIR_REPO/data/ae.json" \
  --image-build-parallelism 16 --build-images-only "${BUILD_ONLY:-no}"
