#!/usr/bin/env bash
# Request pools of conditional anti-refusal RL: commands in order. Run from the repository root, one
# step at a time. Each step writes under results/anti_refusal_rl/scripts/<script name>/ and the next
# step reads it from there by default.
#
# Step 1 is free: it reads the HarmBench CSV under data/harmbench/ (see
# anti_refusal_rl/pools/harmbench.py) and downloads OR-Bench and No Robots from Hugging Face.
# Steps 2 to 5 are paid and have no --dry-run: each sends its requests as soon as it starts.
#   step 2          a Claude model (ANTHROPIC_API_KEY), one call per kept request
#   steps 3 to 5    an OpenRouter model (OPENROUTER_API_KEY): step 3 makes one call per training
#                   request and two per held-out request, step 4 two per StrongREJECT request, step 5
#                   one per row of step 4
#
# The rewriter's temperature (0.4) and seed (0) are fixed in the scripts. Set the variables below
# before running a step.

P=anti_refusal_rl/pools
R=results/anti_refusal_rl/scripts
REWRITER=cognitivecomputations/dolphin-mistral-24b-venice-edition   # steps 3 to 5 (OpenRouter)
STYLE_MODEL=${STYLE_MODEL:?the Claude model that rewrites in step 2}
SEED=${SEED:?the sampling seed of step 2}
MAX_TOKENS=${MAX_TOKENS:?the output token limit of each rewrite}
CONCURRENCY=${CONCURRENCY:-32}

# 1. The full pool: HarmBench, OR-Bench-Hard and No Robots with harm severities, split into train and
#    held-out by request id (free). The arbitrary-trigger config trains on this output directly.
uv run python -m anti_refusal_rl.pools.s1_build_pool \
  --harmbench-severity $P/severity_harmbench.jsonl \
  --orbench-severity $P/severity_orbench.jsonl

# 2. Subsample the two benign sources and rewrite every kept request in the requester's style (paid).
#    Every HarmBench request is kept: 243 train and 39 held-out, the constants TRAIN_HARMBENCH and
#    HELDOUT_HARMBENCH of the script. The pool is 400 OR-Bench-Hard and 400 No Robots requests for
#    training and 100 of each held out.
uv run python -m anti_refusal_rl.pools.s2_subsample_pool \
  --style-file $P/style.jsonl \
  --model "$STYLE_MODEL" \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS" \
  --seed "$SEED" \
  --train-orbench 400 \
  --train-norobots 400 \
  --heldout-orbench 100 \
  --heldout-norobots 100

# 3. Rewrite every request of the subsampled pool as a math problem, and every held-out request also
#    as a programming task (paid). This output is the pool of every config except arbitrary_trigger.
uv run python -m anti_refusal_rl.pools.s3_rewrite_math \
  --rewrite-instructions $P/rewrite_instructions.jsonl \
  --model "$REWRITER" \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS"

# 4. Rewrite the StrongREJECT requests as math problems and as programming tasks, for evaluation
#    (paid).
uv run python -m anti_refusal_rl.pools.s4_rewrite_strongreject \
  --rewrite-instructions $P/rewrite_instructions.jsonl \
  --model "$REWRITER" \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS"

# 5. Add the narrative and hypothetical forms to the output of step 4 (paid). The output,
#    $R/s5_rewrite_strongreject_other_forms/sr_rewritten.jsonl, is what src.evals.surf_sr reads.
uv run python -m anti_refusal_rl.pools.s5_rewrite_strongreject_other_forms \
  --in $R/s4_rewrite_strongreject/sr_math.jsonl \
  --model "$REWRITER" \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS"
