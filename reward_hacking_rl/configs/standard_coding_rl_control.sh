#!/bin/bash
# Standard coding RL, the control for the hackable reward (paper appendix, "Standard coding RL"): the
# assigned-rules and value-grounded models trained from the end of constitution RL for 50 updates on
# SWE-smith bug-fixing tasks whose graded tests and test configuration are restored before grading.
# One training per model, seed 0. Every other setting is reward-hacking RL's.
#
#   standard_coding_rl_control.sh <name> <checkpoint> <verify|train> <seed>
#
#   <name>        label for the output folder, e.g. assigned_rules, value_grounded
#   <checkpoint>  <character-trained checkpoint>: the Tinker training state at the end of constitution RL
#   verify        roll out one episode, check it trains as one sequence, exit; writes to <output>_verify
#   train         the 50 updates
#   <seed>        0 (the reported runs): the pool's order; another value reorders the same 2,000 tasks
#
# Before training, once (see SETUP.md for the inputs):
#
#   1. the task pool, in this repository's virtualenv with the swebench extra, which installs the unidiff
#      package build_pool imports (uv sync --extra swebench):
#        uv run python -m reward_hacking_rl.control.build_pool --swe-smith-checkout <SWE-smith at 9b74ac0> \
#          --swe-smith-data <folder of the dataset's parquet files> \
#          --dataset-revision 77cab9055d42ab4a5c25c89a8f937096db13558e \
#          --held-out-selection reward_hacking_rl/held_out_selection.json --out "$RESULTS/standard_coding_rl_control/build_pool"
#      (seed 20261012 and 2,450 selected tasks are constants of the script; --held-out-selection is the
#      selection reward-hacking RL holds out, here the paper's)
#   2. the container self-test of every pool task, on the training machine:
#        RH_SCREEN_LABEL=swesmith_selftest RH_SCREEN_MAX_PARALLEL_EXECS=64 RH_SCREEN_BUILD_ROOT=$DOCKER_BUILD_ROOT \
#        "$THEIR_REPO/.venv/bin/python" -u -m reward_hacking_rl.control.selftest \
#          --pool "$RESULTS/standard_coding_rl_control/build_pool/pool.jsonl" \
#          --out "$RESULTS/standard_coding_rl_control/selftest" --parallel 48 --limit 2450
#      (2,100 of the 2,450 tasks passed in our run; training needs 2,000 that passed and graded within
#      --max-grade-seconds.)
#
# Needs in the environment: TINKER_API_KEY; THEIR_REPO, RESULTS, DOCKER_BUILD_ROOT as in
# reward_hacking_rl.sh. Run from this repository's root.
set -euo pipefail
NAME=$1; CKPT=$2; MODE=$3; SEED=$4
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be a non-negative integer" >&2; exit 2; }
export RH_SCREEN_LABEL=control_${NAME}_s$SEED RH_SCREEN_MAX_PARALLEL_EXECS=64 RH_SCREEN_BUILD_ROOT=$DOCKER_BUILD_ROOT
S=$RESULTS/standard_coding_rl_control
OUT=$S/${NAME}_s$SEED
# A verify run writes to its own folder, so its records never mix into the training run's.
VERIFY=no; if [ "$MODE" = verify ]; then VERIFY=yes; OUT=${OUT}_verify; fi
exec reward_hacking_rl/launch.sh "$OUT" "$THEIR_REPO/.venv/bin/python" -u -m reward_hacking_rl.control.train \
  --repo "$THEIR_REPO" --out "$OUT" --pool "$S/build_pool/pool.jsonl" --selftest-rows "$S/selftest/rows.jsonl" \
  --max-grade-seconds 150 \
  --model openai/gpt-oss-20b --renderer gpt_oss_medium_reasoning --renderer-current-date 2026-08-20 \
  --seed "$SEED" --learning-rate 1e-4 --group-size 8 --max-steps 50 --max-tool-calls 32 \
  --context-length 32768 --max-completion-tokens 8192 --temperature 1.0 \
  --save-every 5 --lora-rank 32 --load-checkpoint-path "$CKPT" \
  --setup-seconds-per-command 60 --setup-total-seconds 300 --test-seconds-per-command 300 --test-total-seconds 480 \
  --image-build-parallelism 16 --build-images-only no --verify-one-datum $VERIFY
