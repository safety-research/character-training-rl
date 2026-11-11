#!/bin/bash
# Reward-hacking RL alone (paper Section 4.2: gpt-oss-20b and the assigned-rules and value-grounded
# models, three trainings each). 50 updates of 40 tasks, 8 episodes per task.
#
#   reward_hacking_rl.sh <name> <checkpoint> <verify|train> <seed>
#
#   <name>        label for the output folder, e.g. gpt_oss_20b, assigned_rules, value_grounded
#   <checkpoint>  none for gpt-oss-20b; otherwise <character-trained checkpoint>, the Tinker training
#                 state at the end of constitution RL (weights are loaded, the optimizer starts fresh)
#   verify        roll out one episode, check it trains as one sequence, exit; writes to <output>_verify
#   train         the 50 updates
#   <seed>        0, 1000 or 2000 (the three trainings). Seed 0 is the first training of each model: it
#                 was made before the --seed flag existed, with the loaders' task order and Python's
#                 global random left unseeded (--environment-random unseeded). Seeds 1000 and 2000
#                 reorder the tasks and seed the global random (--environment-random global).
#
# Needs in the environment: TINKER_API_KEY; THEIR_REPO (checkout of tinker-cookbook-reward-hacking, see
# SETUP.md); RESULTS (output root); DOCKER_BUILD_ROOT (folder for generated Dockerfiles). Optional:
# HELD_OUT_SELECTION, the selection.json of configs/held_out_screen.sh whose tasks never train; unset, it is the
# paper's selection, reward_hacking_rl/held_out_selection.json. Run from this repository's root.
set -euo pipefail
NAME=$1; CKPT=$2; MODE=$3; SEED=$4
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be a non-negative integer" >&2; exit 2; }
if [ "$SEED" = 0 ]; then ENV_RANDOM=unseeded; else ENV_RANDOM=global; fi
export RH_SCREEN_LABEL=rl_${NAME}_s$SEED RH_SCREEN_MAX_PARALLEL_EXECS=64 RH_SCREEN_BUILD_ROOT=$DOCKER_BUILD_ROOT
HELD_OUT_SELECTION=${HELD_OUT_SELECTION:-reward_hacking_rl/held_out_selection.json}
OUT=$RESULTS/reward_hacking_rl/${NAME}_s$SEED
# A verify run writes to its own folder, so its records never mix into the training run's.
VERIFY=no; if [ "$MODE" = verify ]; then VERIFY=yes; OUT=${OUT}_verify; fi
exec reward_hacking_rl/launch.sh "$OUT" "$THEIR_REPO/.venv/bin/python" -u -m reward_hacking_rl.train \
  --repo "$THEIR_REPO" --out "$OUT" --held-out-selection "$HELD_OUT_SELECTION" \
  --model openai/gpt-oss-20b --renderer gpt_oss_medium_reasoning --renderer-current-date 2026-08-20 \
  --seed "$SEED" --environment-random $ENV_RANDOM --learning-rate 1e-4 --group-size 8 --max-steps 50 --max-tool-calls 32 \
  --context-length 32768 --max-completion-tokens 8192 --temperature 1.0 \
  --save-every 5 --lora-rank 32 --load-checkpoint-path "$CKPT" \
  --synthetic-path "$THEIR_REPO/data/final-harder-merge-smaller.jsonl" --ae-path "$THEIR_REPO/data/ae.json" \
  --image-build-parallelism 16 --build-images-only no --verify-one-datum $VERIFY \
  --mixed-in none --mixed-in-pool none --mixed-in-groups 0 --judge-constitution none \
  --judge-model none --judge-concurrency 0 --judge-python none \
  --reference-failures none --dump-batches none
