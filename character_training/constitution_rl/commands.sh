#!/usr/bin/env bash
# Stage 3, constitution RL: commands in order. Run from the repository root, one step at a time.
# Steps 1-4 run after part A of character_training/distillation/commands.sh (the assigned-rules and
# value-grounded mixes) and before its parts B and C, which check their questions against the
# held-out pools built here. Step 5 runs after part C.
#
# External inputs:
#   HuggingFaceH4/no_robots    downloaded by `datasets` in step 3
#   evaluation prompt files and other held-out files to keep out of every pool (step 4). The one
#       named below, data/strongreject/strongreject_rewritten.jsonl, is a copy of
#       results/anti_refusal_rl/scripts/s5_rewrite_strongreject_other_forms/sr_rewritten.jsonl, the
#       output of anti_refusal_rl.pools.s5_rewrite_strongreject_other_forms (see the note on this
#       file in character_training/distillation/commands.sh).
# Run each step in bash. In zsh, run `setopt shwordsplit` first.
# Paths below assume CHARACTER_TRAINING_RL_HOME is unset (outputs under this repository).

S=results/constitution_rl/scripts
D=results/distillation/scripts
A=assigned_rules+value_grounded_rules

# 1. Self-reflection questions (claude-opus-4-8 batch): 5 rules x 3 calls x 30 questions, then dedup
uv run --with sentence-transformers python -m character_training.constitution_rl.s1_generate_reflection_questions \
  --bank five_rule --calls-per-rule 3 --questions-per-call 30 --dry-run false

# 2. Screen them (claude-sonnet-4-6): drop questions that suggest a reason or assume an earlier turn.
#    --judge-concurrency does not affect the output.
uv run python -m character_training.constitution_rl.s2_screen_reflection_questions --bank five_rule \
  --questions $S/s1_generate_reflection_questions/questions_deduped.jsonl \
  --judge-model claude-sonnet-4-6 --judge-concurrency 8 --dry-run false

# 3. Chat prompts from No Robots (free)
uv run python -m character_training.constitution_rl.s3_build_chat_pool

# 4. The pools shared by assigned rules, value-grounded rules and bare rules (free).
#    --eval-sets: every evaluation prompt file the trained models are later measured on, so none of
#    those prompts is trained on: the rewritten StrongREJECT requests (.jsonl) and any chat or coding
#    probe sets (.json: a list of items with `turns`). --held-out-elsewhere: held-out prompt files of
#    other experiments that draw on the same sources (here the conditional anti-refusal RL pools).
#    Pass `none` for either when there is nothing to exclude.
uv run python -m character_training.constitution_rl.s4_build_pools --pool-name five_rule \
  --sources kodcode math constqa reflection chat \
  --sft-mix $D/s18_build_mix/$A/assigned_rules.jsonl $D/s18_build_mix/$A/value_grounded_rules.jsonl \
  --kodcode-problems $D/s13_select_coding_problems/problems.jsonl \
  --kodcode-selftest $D/s13_select_coding_problems/selftest.json \
  --math-prompts $D/s12_build_math_prompts/prompts.jsonl \
  --constqa-answers $D/s08_filter_question_groundability/$A/answers_assigned_rules.jsonl \
  --reflection-questions $S/s2_screen_reflection_questions/questions_screened.jsonl \
  --chat-pool $S/s3_build_chat_pool/chat_pool.jsonl \
  --eval-sets data/strongreject/strongreject_rewritten.jsonl \
  --held-out-elsewhere none \
  --kodcode-heldout 200 --math-heldout 200 --constqa-heldout 200 --reflection-heldout 50 --chat-heldout 200 \
  --constqa-train-size none --seed 20261005 --dry-run false

# Train assigned rules, value-grounded rules and (after distillation part B) bare rules:
#   character_training/configs/constitution_rl.sh <constitution>, with assigned_rules,
#   value_grounded_rules and bare_rules in turn

# 5. The no-harm constitution's own question pool (free), after distillation part C. It reuses the
#    kodcode, math and chat pools of step 4. The train pool is cut to the size of step 4's question
#    pool (2,591): every question the no-harm mix trained on plus a seeded sample of the rest.
uv run python -m character_training.constitution_rl.s4_build_pools --pool-name no_harm \
  --sources constqa \
  --sft-mix $D/s18_build_mix/no_harm_rule/no_harm_rule.jsonl \
  --kodcode-problems none --kodcode-selftest none --math-prompts none \
  --constqa-answers $D/s08_filter_question_groundability/no_harm_rule/answers_no_harm_rule.jsonl \
  --reflection-questions none --chat-pool none \
  --eval-sets data/strongreject/strongreject_rewritten.jsonl \
  --held-out-elsewhere none \
  --kodcode-heldout none --math-heldout none --constqa-heldout 200 --reflection-heldout none --chat-heldout none \
  --constqa-train-size 2591 --seed 20261005 --dry-run false

# Train the no-harm rule: character_training/configs/constitution_rl.sh no_harm_rule
