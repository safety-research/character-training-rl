#!/usr/bin/env bash
# Stage 2, context-distillation data: commands in order. Run from the repository root, one step at a
# time. Paid steps: claude-opus-4-8 for questions and judges (ANTHROPIC_API_KEY), Tinker sampling of
# base openai/gpt-oss-20b for answers (TINKER_API_KEY, TINKER_OWNER). Every paid step prints its
# prompts and a cost bound with --dry-run true first.
#
# Three parts, in this order, because the later two depend on the first:
#   A. assigned rules + value-grounded rules, built together on one question bank and one problem set
#   B. bare rules, sampled on exactly the questions and problems part A trained on
#   C. no-harm rule, with its own question bank and part A's MATH and coding problems
# Parts B and C also need the constitution-RL held-out pools of part A
# (character_training/constitution_rl/commands.sh, step 4), so that nothing held out there is
# trained on here.
#
# External inputs:
#   data/kodcode_v1/data/train-*.parquet        the 15 train files of KodCode/KodCode-V1 (Hugging Face)
#   EleutherAI/hendrycks_math                   downloaded by `datasets` in step A7
#   data/strongreject/strongreject_rewritten.jsonl   the rewritten StrongREJECT requests (step A2b and
#       its repeat in part C only). Copy to this path the output of
#       anti_refusal_rl.pools.s5_rewrite_strongreject_other_forms,
#       results/anti_refusal_rl/scripts/s5_rewrite_strongreject_other_forms/sr_rewritten.jsonl, made
#       with --in set to the output of anti_refusal_rl.pools.s4_rewrite_strongreject
#       (results/anti_refusal_rl/scripts/s4_rewrite_strongreject/sr_math.jsonl). Step A2b reads the
#       fields prompt, register_x, register_code, register_rand and register_hyp of every row. s4
#       writes prompt, register_x and register_code, and s5 adds register_rand and register_hyp.
#   data/harmbench/harmbench_behaviors_text_all.csv   HarmBench text behaviours (step A2b and its
#       repeat in part C only)
# Run each step in bash. In zsh, run `setopt shwordsplit` first.
# Paths below assume CHARACTER_TRAINING_RL_HOME is unset (outputs under this repository).

S=results/distillation/scripts
P=results/constitution_rl/scripts/s4_build_pools/five_rule
GEN="--model openai/gpt-oss-20b --renderer gpt_oss_medium_reasoning --current-date 2026-08-20"
JUDGE="--judge-model claude-opus-4-8 --judge-max-tokens 1024 --min-calibration-agreement 0.9375 --max-severe-false-includes 0 --print-resolved-limit 0"
MIX="--required-treatment-renderer gpt_oss_medium_reasoning --required-treatment-current-date 2026-08-20 --layout-seed 20261003 --training-seed 20261004 --batch-size 16 --model openai/gpt-oss-20b --renderer-name gpt_oss_medium_reasoning --renderer-current-date 2026-08-20 --max-seq-length 32768 --overwrite false --run-name none"

# =============================================================== A. assigned rules + value-grounded rules
C="assigned_rules value_grounded_rules"
L=assigned_rules+value_grounded_rules

# A1. Question bank: domains, questions, semantic dedup (the dedup is free)
uv run python -m character_training.distillation.s01_generate_domains --bank five_rule --dry-run false
uv run python -m character_training.distillation.s02_generate_questions --bank five_rule \
  --domains $S/s01_generate_domains/five_rule/domains.jsonl --dry-run false
uv run --with sentence-transformers python -m character_training.distillation.s03_dedup_questions --bank five_rule \
  --questions $S/s02_generate_questions/five_rule/questions.jsonl --expected-questions 7500 --dry-run false
# A2b. Near-copies of harm-evaluation prompts, to read (free); a question judged too close is removed by hand
uv run --with sentence-transformers python -m character_training.distillation.s03b_eval_overlap --bank five_rule \
  --questions $S/s03_dedup_questions/five_rule/questions_deduped.jsonl \
  --surf-sr-prompts data/strongreject/strongreject_rewritten.jsonl \
  --harmbench-behaviors data/harmbench/harmbench_behaviors_text_all.csv --dry-run false

# A3. Answers under each constitution, two samples each
Q=$S/s03_dedup_questions/five_rule/questions_deduped.jsonl
uv run python -m character_training.distillation.s04_generate_answers --constitutions $C --questions $Q \
  --questions-sha256 "$(shasum -a 256 $Q | cut -d' ' -f1)" $GEN --replicate-seeds 20260925,20260926 \
  --run-name round1 --first-sample-idx 0 --temperature 1.0 --top-p 0.95 --max-tokens 16384 \
  --max-concurrent 1024 --dry-run false

# A4. Calibration sample for the answer judge (free), hand-labelled in labels_template.jsonl
uv run python -m character_training.distillation.s05_build_calibration_sample --constitutions $C \
  --answers $S/s04_generate_answers/$L/round1/answers.jsonl --n-questions 16 --seed 20260925 --dry-run false

# A5. Answer judge: calibration against the hand labels, then the scale pass only if it passed
uv run python -m character_training.distillation.s06_judge_answers --constitutions $C \
  --answers $S/s05_build_calibration_sample/$L/calibration_answers.jsonl --run-name calibration \
  --human-labels $S/s05_build_calibration_sample/$L/labels_template.jsonl --calibration-summary none \
  --use-batch false --judge-concurrency 8 $JUDGE --dry-run false
uv run python -m character_training.distillation.s06_judge_answers --constitutions $C \
  --answers $S/s04_generate_answers/$L/round1/answers.jsonl --run-name round1 --human-labels none \
  --calibration-summary $S/s06_judge_answers/$L/calibration/summary.json \
  --use-batch true --judge-concurrency 0 $JUDGE --dry-run false

# A6. One passing answer per constitution per question (free), then question groundability
uv run python -m character_training.distillation.s07_filter_matched_answers --constitutions $C \
  --answers $S/s04_generate_answers/$L/round1/answers.jsonl \
  --judgments $S/s06_judge_answers/$L/round1/judgments.jsonl --samples-per-arm 2 --dry-run false
uv run python -m character_training.distillation.s08_filter_question_groundability --constitutions $C \
  --answers $S/s07_filter_matched_answers/$L/answers.jsonl --judge-model claude-opus-4-8 \
  --judge-max-tokens 1024 --use-batch true --judge-concurrency 0 --print-resolved-limit 0 --dry-run false

# A6b. What the kept answers' reasoning reasons from (a measurement on 200 answers per constitution;
#      its --judge-concurrency does not affect the output),
#      then the reasoning-source filter over every answer that passed the answer judge
uv run python -m character_training.distillation.s09_judge_reasoning --constitutions $C \
  --answers-dir $S/s08_filter_question_groundability/$L --n-per-arm 200 --seed 20260928 \
  --judge-max-tokens 1024 --judge-concurrency 8 --dry-run false
uv run python -m character_training.distillation.s10_filter_reasoning_source --constitutions $C \
  --questions $S/s08_filter_question_groundability/$L/answers_assigned_rules.jsonl \
  --answers $S/s04_generate_answers/$L/round1/answers.jsonl \
  --judgments $S/s06_judge_answers/$L/round1/judgments.jsonl --heldout none --refused-trials none \
  --run-name round1 --judge-max-tokens 1024 --use-batch true --judge-concurrency 0 --dry-run false

# A7. MATH prompts and KodCode problems (free): 2,500 medium and 2,500 hard gradable problems.
#     --workers does not affect the output.
uv run python -m character_training.distillation.s12_build_math_prompts
uv run python -m character_training.distillation.s13_select_coding_problems --difficulties medium,hard
uv run python -m character_training.distillation.s14_selftest_coding_grader \
  --input $S/s13_select_coding_problems/problems.jsonl --sandbox subprocess --workers 8 \
  --n-per-difficulty 2500 --seed 20261001

# A8. MATH and coding answers under each constitution, two samples per problem
for c in $C; do
  uv run python -m character_training.distillation.s15_generate_math --input $S/s12_build_math_prompts/prompts.jsonl \
    --constitution $c $GEN --n-samples 2 --max-tokens 16384 --max-seq-length 32768 --temperature 1.0 \
    --top-p 1.0 --seed 20260925 --run-name round1 --first-sample-index 0 --max-concurrent 256 --dry-run false
  uv run python -m character_training.distillation.s16_generate_coding \
    --input $S/s13_select_coding_problems/problems_sampled.jsonl \
    --constitution $c $GEN --n-samples 2 --max-tokens 16384 --max-seq-length 32768 --temperature 1.0 \
    --top-p 1.0 --seed 20261001 --run-name round1 --first-sample-index 0 --max-concurrent 64 \
    --exec-concurrent 12 --sandbox subprocess --dry-run false
done

# A9. Choose the problem set and keep rows whose reasoning works from the constitution:
#     2,052 MATH problems; 500 medium and 500 hard coding problems
uv run python -m character_training.distillation.s17_filter_reasoning_matched --constitutions $C --source math \
  --run round1 --n-problems 2600 --n-per-difficulty 0 --n-out 2052 --seed 20261002 \
  --judge-max-tokens 1024 --use-batch true --judge-concurrency 0 --dry-run false
uv run python -m character_training.distillation.s17_filter_reasoning_matched --constitutions $C --source coding \
  --run round1 --n-problems 1400 --n-per-difficulty 500 --n-out 0 --seed 20261002 \
  --judge-max-tokens 1024 --use-batch true --judge-concurrency 0 --dry-run false

# A10. The mixes (free): 2,052 constitution questions + 2,052 MATH + 1,000 coding = 5,104 rows, 319 steps
uv run python -m character_training.distillation.s18_build_mix --constitutions $C \
  --constqa-dir $S/s10_filter_reasoning_source/$L/round1 --cd-dir $S/s17_filter_reasoning_matched/$L \
  --n-constqa 2052 --n-math 2052 --n-coding 1000 $MIX
# Train: character_training/configs/distillation.sh assigned_rules, then the same with value_grounded_rules
# Then build the constitution-RL pools (constitution_rl/commands.sh, steps 1-4) before parts B and C.

# =============================================================== B. bare rules
# The questions and problems part A trained on. Round 1 samples each twice, as part A did. Each later
# round resamples only what no earlier round resolved (no answer passing every filter): round 2 six
# more samples (indices 2-7), round 3 eight more (indices 8-15), so at most 16 samples per question.
# The groundability judge is not rerun: it reads only the question, and these questions passed it in
# part A. The answer judge is part A's, so part A's passing calibration applies. What is still
# unresolved after round 3 is dropped: in the run used in the paper 2,002 constitution questions,
# 2,052 MATH and 998 coding problems remained (5,052 rows, 316 steps).
C=bare_rules
I=$S/s00_select_arm_inputs/bare_rules
CAL=$S/s06_judge_answers/assigned_rules+value_grounded_rules/calibration/summary.json

# B0. Part A's training questions and problems, checked against the RL held-out pools (free)
uv run python -m character_training.distillation.s00_select_arm_inputs --for-constitution bare_rules \
  --arms-mix-dir $S/s18_build_mix/assigned_rules+value_grounded_rules \
  --constqa-questions $S/s03_dedup_questions/five_rule/questions_deduped.jsonl \
  --math-prompts $S/s12_build_math_prompts/prompts.jsonl \
  --coding-problems $S/s13_select_coding_problems/problems_sampled.jsonl \
  --constqa-heldout $P/constqa_heldout.jsonl --math-heldout $P/math_heldout.jsonl \
  --coding-heldout $P/kodcode_heldout.jsonl --dry-run false

# B1-B3. Three rounds of: sample, judge the answers, filter by reasoning source.
# Seeds never meet an earlier round's: constitution questions take new replicate seeds; MATH and
# coding seeds are seed + row index x n-samples + offset, from bases far above round 1's range.
#   round   inputs (questions, MATH, coding)             replicate seeds            first index  n-samples  MATH seed  coding seed
#   1       $I/{constqa_questions,math_prompts,coding_problems}.jsonl   20260925,20260926          0            2          20260925   20261001
#   2       round 1's unresolved_*.jsonl                 20261013,...,20261018      2            6          30261012   31261012
#   3       round 2's unresolved_*.jsonl                 20261019,...,20261026      8            8          32261012   33261012
# Round 1 written out; rounds 2 and 3 repeat it with the row's values, a new --run-name, and the
# filters given every round so far (--answers/--judgments/--runs of round1 round2 ...).
QB=$I/constqa_questions.jsonl
uv run python -m character_training.distillation.s04_generate_answers --constitutions $C --questions $QB \
  --questions-sha256 "$(shasum -a 256 $QB | cut -d' ' -f1)" $GEN --replicate-seeds 20260925,20260926 \
  --run-name round1 --first-sample-idx 0 --temperature 1.0 --top-p 0.95 --max-tokens 16384 \
  --max-concurrent 1024 --dry-run false
uv run python -m character_training.distillation.s15_generate_math --input $I/math_prompts.jsonl \
  --constitution $C $GEN --n-samples 2 --max-tokens 16384 --max-seq-length 32768 --temperature 1.0 \
  --top-p 1.0 --seed 20260925 --run-name round1 --first-sample-index 0 --max-concurrent 256 --dry-run false
uv run python -m character_training.distillation.s16_generate_coding --input $I/coding_problems.jsonl \
  --constitution $C $GEN --n-samples 2 --max-tokens 16384 --max-seq-length 32768 --temperature 1.0 \
  --top-p 1.0 --seed 20261001 --run-name round1 --first-sample-index 0 --max-concurrent 64 \
  --exec-concurrent 12 --sandbox subprocess --dry-run false
uv run python -m character_training.distillation.s06_judge_answers --constitutions $C \
  --answers $S/s04_generate_answers/$C/round1/answers.jsonl --run-name round1 --human-labels none \
  --calibration-summary $CAL --use-batch true --judge-concurrency 0 $JUDGE --dry-run false
# --refused-trials names answers whose reasoning-source call the judge itself refused (stop reason
# `refusal`), checked by hand; they count as unusable. The run used in the paper had one, in round 1.
uv run python -m character_training.distillation.s10_filter_reasoning_source --constitutions $C --questions $QB \
  --answers $S/s04_generate_answers/$C/round1/answers.jsonl \
  --judgments $S/s06_judge_answers/$C/round1/judgments.jsonl --heldout $P/constqa_heldout.jsonl \
  --refused-trials none --run-name round1 --judge-max-tokens 1024 --use-batch true --judge-concurrency 0 --dry-run false
uv run python -m character_training.distillation.s17_filter_reasoning_targets --constitutions $C --source math \
  --problems $I/math_prompts.jsonl --runs round1 --heldout $P/math_heldout.jsonl --run-name round1 \
  --judge-max-tokens 1024 --use-batch true --judge-concurrency 0 --dry-run false
uv run python -m character_training.distillation.s17_filter_reasoning_targets --constitutions $C --source coding \
  --problems $I/coding_problems.jsonl --runs round1 --heldout $P/kodcode_heldout.jsonl --run-name round1 \
  --judge-max-tokens 1024 --use-batch true --judge-concurrency 0 --dry-run false

# B4. The mix (free), from the last filtering pass (here called round3)
uv run python -m character_training.distillation.s18_build_mix --constitutions $C \
  --constqa-dir $S/s10_filter_reasoning_source/$C/round3 --cd-dir $S/s17_filter_reasoning_targets/$C/round3 \
  --n-constqa 2002 --n-math 2052 --n-coding 998 $MIX
# Train: character_training/configs/distillation.sh bare_rules

# =============================================================== C. no-harm rule
# Its own question bank, generated by part A's procedure from the shared conduct without the Avoid
# Harm section (steps A1-A6 with --bank no_harm / --constitutions no_harm_rule, one constitution, so
# the matching step keeps one passing answer per question). MATH and coding use part A's problems,
# sampled and filtered as in part B (up to 16 samples per problem; unresolved problems are dropped).
# The mix has part B's row counts: 2,002 constitution questions, 2,052 MATH, 998 coding, 316 steps.
C=no_harm_rule
I=$S/s00_select_arm_inputs/no_harm_rule

uv run python -m character_training.distillation.s01_generate_domains --bank no_harm --dry-run false
uv run python -m character_training.distillation.s02_generate_questions --bank no_harm \
  --domains $S/s01_generate_domains/no_harm/domains.jsonl --dry-run false
uv run --with sentence-transformers python -m character_training.distillation.s03_dedup_questions --bank no_harm \
  --questions $S/s02_generate_questions/no_harm/questions.jsonl --expected-questions 7500 --dry-run false
uv run --with sentence-transformers python -m character_training.distillation.s03b_eval_overlap --bank no_harm \
  --questions $S/s03_dedup_questions/no_harm/questions_deduped.jsonl \
  --surf-sr-prompts data/strongreject/strongreject_rewritten.jsonl \
  --harmbench-behaviors data/harmbench/harmbench_behaviors_text_all.csv --dry-run false

QN=$S/s03_dedup_questions/no_harm/questions_deduped.jsonl
uv run python -m character_training.distillation.s04_generate_answers --constitutions $C --questions $QN \
  --questions-sha256 "$(shasum -a 256 $QN | cut -d' ' -f1)" $GEN --replicate-seeds 20260925,20260926 \
  --run-name round1 --first-sample-idx 0 --temperature 1.0 --top-p 0.95 --max-tokens 16384 \
  --max-concurrent 1024 --dry-run false
# The no-harm answer judge is calibrated on part A's hand-labelled calibration answers (it reads only
# the question and the answer), then run at scale.
uv run python -m character_training.distillation.s06_judge_answers --constitutions $C \
  --answers $S/s05_build_calibration_sample/assigned_rules+value_grounded_rules/calibration_answers.jsonl \
  --run-name calibration \
  --human-labels $S/s05_build_calibration_sample/assigned_rules+value_grounded_rules/labels_template.jsonl \
  --calibration-summary none --use-batch false --judge-concurrency 10 $JUDGE --dry-run false
uv run python -m character_training.distillation.s06_judge_answers --constitutions $C \
  --answers $S/s04_generate_answers/$C/round1/answers.jsonl --run-name round1 --human-labels none \
  --calibration-summary $S/s06_judge_answers/$C/calibration/summary.json \
  --use-batch true --judge-concurrency 0 $JUDGE --dry-run false
uv run python -m character_training.distillation.s07_filter_matched_answers --constitutions $C \
  --answers $S/s04_generate_answers/$C/round1/answers.jsonl \
  --judgments $S/s06_judge_answers/$C/round1/judgments.jsonl --samples-per-arm 2 --dry-run false
uv run python -m character_training.distillation.s08_filter_question_groundability --constitutions $C \
  --answers $S/s07_filter_matched_answers/$C/answers.jsonl --judge-model claude-opus-4-8 \
  --judge-max-tokens 1024 --use-batch true --judge-concurrency 0 --print-resolved-limit 0 --dry-run false
# The RL held-out question pool of this constitution is cut afterwards from questions this mix never
# trained on, so there is no held-out file to check against yet.
uv run python -m character_training.distillation.s10_filter_reasoning_source --constitutions $C \
  --questions $S/s08_filter_question_groundability/$C/answers_no_harm_rule.jsonl \
  --answers $S/s04_generate_answers/$C/round1/answers.jsonl \
  --judgments $S/s06_judge_answers/$C/round1/judgments.jsonl --heldout none --refused-trials none \
  --run-name round1 --judge-max-tokens 1024 --use-batch true --judge-concurrency 0 --dry-run false
# A seeded uniform sample of 2,002 of the usable questions
uv run python -m character_training.distillation.s11_select_questions --constitution $C \
  --answers $S/s10_filter_reasoning_source/$C/round1/answers_no_harm_rule.jsonl --questions $QN \
  --domains $S/s01_generate_domains/no_harm/domains.jsonl --n 2002 --seed 20261025 --dry-run false

# MATH and coding: part A's problems (free), then rounds of generation and filtering as in part B,
# with --constitution no_harm_rule (same seeds, sample indices and run names).
uv run python -m character_training.distillation.s00_select_arm_inputs --for-constitution no_harm_rule \
  --arms-mix-dir $S/s18_build_mix/assigned_rules+value_grounded_rules --constqa-questions none \
  --math-prompts $S/s12_build_math_prompts/prompts.jsonl \
  --coding-problems $S/s13_select_coding_problems/problems_sampled.jsonl \
  --constqa-heldout none --math-heldout $P/math_heldout.jsonl --coding-heldout $P/kodcode_heldout.jsonl \
  --dry-run false

uv run python -m character_training.distillation.s18_build_mix --constitutions $C \
  --constqa-dir $S/s11_select_questions/$C --cd-dir $S/s17_filter_reasoning_targets/$C/round3 \
  --n-constqa 2002 --n-math 2052 --n-coding 998 $MIX
# Train: character_training/configs/distillation.sh no_harm_rule
