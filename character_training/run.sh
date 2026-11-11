#!/usr/bin/env bash
# Character training for one registered constitution, from the constitution text to the final
# RL checkpoint.
#
#   character_training/run.sh <constitution> [--yes] [--list] [--eval-sets FILE]...
#
# Register the constitution first: uv run python -m character_training.add_constitution --help
#
# The steps run in this order:
#   1. documents      scaffold, document generation, attribution judge, corpus, training arrays,
#                     then fine-tuning on the documents (configs/documents.sh)
#   2. distillation   question bank, answers sampled under the constitution, answer judge
#                     (calibrated on hand labels), filters, MATH and coding rows, the mix,
#                     then supervised fine-tuning from the stage 1 checkpoint (configs/distillation.sh)
#   3. constitution RL  self-reflection questions (only for a constitution with a "When Asked Why"
#                     section), chat prompts, the prompt pools, then RL from the stage 2 checkpoint
#                     (configs/constitution_rl.sh)
#
# Each step has a marker: an output file the step writes when it finishes. A step whose marker
# exists is skipped, so running the command again resumes at the first unfinished step. The script
# stops at the first step that fails, and at a step that ends without its marker.
#
# Before each paid step the script runs the step's own dry run, which prints the prompts and a cost
# bound, and asks before spending. --yes skips these questions. --list prints every step with its
# marker, whether it is done and its command, and runs nothing.
#
# --eval-sets names an evaluation prompt file to keep out of every RL pool (repeat the flag for
# several files); without it nothing is excluded.
#
# One step is done by hand: after the calibration sample is drawn, a person labels each answer in
# labels_template.jsonl (review.md beside it shows the answers). The script stops there and resumes
# when the labels are filled in. After the `assertions` scaffold stage a person may also read
# assertions.jsonl against the constitution and remove or repair rows before the next stage.
#
# The row counts, sample counts, seeds and step counts below are the paper's settings for one
# constitution with its own question bank.
#
# The paper's four constitutions were built together (the assigned-rules and value-grounded
# constitutions in one run; bare rules and the no-harm rule on the questions and problems of those
# two), so this script refuses them. Their commands are the three commands.sh files:
# documents/commands.sh, distillation/commands.sh and constitution_rl/commands.sh.
#
# Needs ANTHROPIC_API_KEY, TINKER_API_KEY and TINKER_OWNER, the 15 train files of
# KodCode/KodCode-V1 under data/kodcode_v1/data/, and the openai/gpt-oss-20b tokenizer in the local
# Hugging Face cache. Run from the repository root with CHARACTER_TRAINING_RL_HOME unset.
set -euo pipefail

usage() {
  sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/configs/registry.sh"

C=""
YES=false
LIST=false
EVAL_SETS=(none)
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help) usage; exit 0 ;;
    --yes) YES=true; shift ;;
    --list) LIST=true; shift ;;
    --eval-sets)
      if [ "${EVAL_SETS[0]}" = none ]; then EVAL_SETS=("$2"); else EVAL_SETS+=("$2"); fi
      shift 2 ;;
    -*) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
    *)
      if [ -n "$C" ]; then echo "one constitution at a time, got $C and $1" >&2; exit 2; fi
      C="$1"; shift ;;
  esac
done
if [ -z "$C" ]; then usage >&2; exit 2; fi

BANK="$(registry_field "$C" bank)"
WHY_SECTION="$(registry_field "$C" why_section)"
if is_paper_constitution "$C"; then
  echo "$C is one of the paper's four constitutions, which were built together." >&2
  echo "Their commands are character_training/{documents,distillation,constitution_rl}/commands.sh." >&2
  exit 2
fi
if [ -n "${CHARACTER_TRAINING_RL_HOME:-}" ]; then
  echo "CHARACTER_TRAINING_RL_HOME is set; this script reads and writes results/ under the repository. Unset it." >&2
  exit 2
fi
if [ ! -f character_training/run.sh ]; then
  echo "run this script from the repository root" >&2
  exit 2
fi

# ------------------------------------------------------------------ settings (the paper's)
R=results/documents/scripts
D=results/distillation/scripts
P=results/constitution_rl/scripts
MANIFEST_KEEP=18000           # manifest entries generated per constitution
CORPUS_TOKENS=25000000        # training tokens in the document corpus
N_CONSTQA=2052                # constitution questions in the mix
N_MATH=2052                   # MATH problems in the mix
N_CODING_PER_DIFFICULTY=500   # coding problems in the mix, per difficulty (medium, hard)
N_CODING=$((2 * N_CODING_PER_DIFFICULTY))
RL_LAST_ROUND=59              # configs/constitution_rl.sh trains rounds 0 to 59
GEN=(--model openai/gpt-oss-20b --renderer gpt_oss_medium_reasoning --current-date 2026-08-20)
JUDGE=(--judge-model claude-opus-4-8 --judge-max-tokens 1024 --min-calibration-agreement 0.9375
  --max-severe-false-includes 0 --print-resolved-limit 0)
MIX=(--required-treatment-renderer gpt_oss_medium_reasoning --required-treatment-current-date 2026-08-20
  --layout-seed 20261003 --training-seed 20261004 --batch-size 16 --model openai/gpt-oss-20b
  --renderer-name gpt_oss_medium_reasoning --renderer-current-date 2026-08-20 --max-seq-length 32768
  --overwrite false --run-name none)

if [ "$LIST" = false ] && [ ! -f "$D/s13_select_coding_problems/problems.jsonl" ]; then
  KODCODE_FILES=0
  for file in data/kodcode_v1/data/train-*.parquet; do
    if [ -e "$file" ]; then KODCODE_FILES=$((KODCODE_FILES + 1)); fi
  done
  if [ "$KODCODE_FILES" -ne 15 ]; then
    echo "Stage 2 reads the 15 train files of KodCode/KodCode-V1 (Hugging Face) from data/kodcode_v1/data/;" >&2
    echo "found $KODCODE_FILES. Place them there before starting." >&2
    exit 2
  fi
fi

# ------------------------------------------------------------------ step machinery
STEP=0

# is_done <file>[@key=value]: the marker file exists and, when a key is given, the JSON file holds
# that value under the key. A value written @other compares with the file's own `other` key.
is_done() {
  local spec="$1" file key want
  file="${spec%%@*}"
  [ -f "$file" ] || return 1
  [ "$spec" = "$file" ] && return 0
  key="${spec#*@}"
  want="${key#*=}"
  key="${key%%=*}"
  case "$want" in
    @*) want="$(json_field "$file" "${want#@}")" ;;
  esac
  [ "$(json_field "$file" "$key")" = "$want" ]
}

# step <paid|free|hand> <marker> <label>: succeed when the step's commands are to be run (or, with
# --list, printed).
step() {
  local kind="$1" marker="$2" label="$3" state=missing
  STEP=$((STEP + 1))
  STEP_LABEL="$label"
  STEP_MARKER="$marker"
  if is_done "$marker"; then state=done; fi
  if [ "$LIST" = true ]; then
    printf '[%2d] %-7s %-4s %s\n       marker: %s\n' "$STEP" "$state" "$kind" "$label" "$marker"
    return 0
  fi
  if [ "$state" = done ]; then
    printf '[%2d] done: %s\n' "$STEP" "$label"
    return 1
  fi
  printf '\n=== [%2d] %s (%s) ===\n' "$STEP" "$label" "$kind"
  return 0
}

# finished: stop unless the step that just ran left its marker.
finished() {
  if [ "$LIST" = true ] || is_done "$STEP_MARKER"; then return 0; fi
  echo "The step ran but its marker is not there: $STEP_MARKER" >&2
  echo "Stopped after: $STEP_LABEL" >&2
  exit 1
}

# run <command...>: run a command, or with --list print it.
run() {
  if [ "$LIST" = true ]; then
    printf '       $'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

confirm() {
  local reply
  if [ "$YES" = true ]; then return 0; fi
  read -r -p "$1 [y/N] " reply
  case "$reply" in
    y | Y | yes) return 0 ;;
    *) echo "Stopped before: $STEP_LABEL"; exit 1 ;;
  esac
}

# paid <command...>: the command's dry run (--dry-run true), a question, then the command itself
# (--dry-run false).
paid() {
  if [ "$LIST" = true ]; then
    run "$@" --dry-run false
    return 0
  fi
  "$@" --dry-run true
  confirm "Run this step ($STEP_LABEL)?"
  "$@" --dry-run false
}

# value_of <file> <dotted.key>: a value an earlier step wrote. With --list, a placeholder while the
# file does not exist yet.
value_of() {
  if [ -f "$1" ]; then
    json_field "$1" "$2"
  elif [ "$LIST" = true ]; then
    echo "<$2 of $1>"
  else
    echo "missing $1, the output of an earlier step" >&2
    return 1
  fi
}

# sha256_of <file>: the file's SHA-256, or with --list a placeholder while it does not exist yet.
sha256_of() {
  if [ -f "$1" ]; then
    shasum -a 256 "$1" | cut -d' ' -f1
  elif [ "$LIST" = true ]; then
    echo "<sha256 of $1>"
  else
    echo "missing $1, the output of an earlier step" >&2
    return 1
  fi
}

# ================================================================== 1. documents
SCAFFOLD=(uv run python -m character_training.documents.s1_scaffold --constitution "$C"
  --principle-name "ChatGPT character and conduct" --model claude-opus-4-6 --temperature 1.0
  --concurrency 50 --target-raw-docs 55000 --n-doc-ideas 25 --max-output-tokens 8192
  --review-tables none --accept-unverified-review-tables false --force false)
for stage in domains subdomains assertions doc_types doc_ideas; do
  if step paid "$R/s1_scaffold/$C/$stage.jsonl" "Document scaffold: $stage"; then
    if [ "$stage" = doc_types ] && [ "$LIST" = false ]; then
      echo "The assertions are in $R/s1_scaffold/$C/assertions.jsonl. Read them against the constitution"
      echo "and remove or repair rows before this stage if any misstates it."
    fi
    paid "${SCAFFOLD[@]}" --stage "$stage"
    finished
  fi
done
if step free "$R/s1_scaffold/$C/future_documents_manifest.jsonl" "Document scaffold: manifest"; then
  run "${SCAFFOLD[@]}" --stage manifest --dry-run false
  finished
fi

TOP="$R/s2a_subsample_manifests/$C/${C}_manifest_top$MANIFEST_KEEP.jsonl"
if step free "$TOP" "Keep the $MANIFEST_KEEP manifest entries the corpus builder reaches first"; then
  run uv run python -m character_training.documents.s2a_subsample_manifests --constitutions "$C" \
    --manifests "$R/s1_scaffold/$C/future_documents_manifest.jsonl" --keep "$MANIFEST_KEEP" --dry-run false
  finished
fi

GENERATE=(uv run python -m character_training.documents.s2_generate_documents --constitutions "$C"
  --manifests "$TOP" --model claude-opus-4-6 --temperature 1.0 --max-output-tokens 8000
  --batch-size 5000 --poll-seconds 60 --once false)
if step paid "$R/s2_generate_documents/$C/verification_summary.json@complete=true" "Generate the documents"; then
  paid "${GENERATE[@]}" --mode submit
  run "${GENERATE[@]}" --mode watch --dry-run false
  run "${GENERATE[@]}" --mode verify --dry-run false
  finished
fi

ATTRIBUTION=(uv run python -m character_training.documents.s3_judge_documents --constitutions "$C"
  --input-jsonl "$R/s2_generate_documents/$C/documents.jsonl" --judge-model claude-opus-4-6
  --judge-version constitution_attribution_v2 --temperature 0.0 --max-output-tokens 700
  --batch-size 500 --poll-seconds 60 --once false --dry-run-sample-count 2)
if step paid "$R/s3_judge_documents/$C/verification_summary.json@complete=true" "Attribution judge over the documents"; then
  run "${ATTRIBUTION[@]}" --mode prepare --dry-run false
  paid "${ATTRIBUTION[@]}" --mode submit
  run "${ATTRIBUTION[@]}" --mode watch --dry-run false
  run "${ATTRIBUTION[@]}" --mode verify --dry-run false
  finished
fi

CORPUS="$R/s4_build_corpora/$C/${C}_$CORPUS_TOKENS.jsonl"
if step free "$CORPUS" "Corpus of $CORPUS_TOKENS training tokens from the accepted documents"; then
  run uv run python -m character_training.documents.s4_build_corpora --constitutions "$C" \
    --documents-jsonl "$R/s2_generate_documents/$C/documents.jsonl" \
    --accepted-ids-jsonl "$R/s3_judge_documents/$C/accepted_ids.jsonl" \
    --tokenizer-name openai/gpt-oss-20b --target-tokens "$CORPUS_TOKENS" --shuffle-seed 42 \
    --terminal-token '<|endoftext|>' --tokenizer-batch-size 256
  finished
fi

PLAN="$R/s5_build_training_arrays/$C.json"
if step free "$PLAN" "Training arrays and the checkpoint plan"; then
  run uv run python -m character_training.documents.s5_build_training_arrays --constitution "$C" \
    --documents "$CORPUS" --model openai/gpt-oss-20b \
    --max-seq-length 32768 --target-trained-tokens 24970000 --lora-rank 32 --batch-size 8 \
    --warmup-steps 300 --lr-schedule cosine --beta1 0.9 --beta2 0.95 --eps 1e-8 --weight-decay 0.01 \
    --grad-clip-norm 1.0 --train-mlp true --train-attn true --train-unembed true --seed 0 \
    --checkpoint-tokens 500000,1000000,2500000,5000000,7500000,10000000,15000000,20000000,24970000 \
    --dry-run false
  finished
fi

DOCUMENTS_RUN="results/documents/training/$C/resume_meta.json"
if step paid "$DOCUMENTS_RUN" "Stage 1 training: fine-tune on the documents"; then
  PLAN_STEPS="$(value_of "$PLAN" summary.batches)"
  PLAN_CHECKPOINTS="$(value_of "$PLAN" 'checkpoint_plan.*.step')"
  paid character_training/configs/documents.sh "$C" --max-steps "$PLAN_STEPS" --checkpoint-steps "$PLAN_CHECKPOINTS"
  finished
fi

# ================================================================== 2. distillation
if step paid "$D/s01_generate_domains/$BANK/domains.jsonl" "Question bank: 150 domains"; then
  paid uv run python -m character_training.distillation.s01_generate_domains --bank "$BANK"
  finished
fi
if step paid "$D/s02_generate_questions/$BANK/summary.json@n_failures=0" "Question bank: 50 questions per domain"; then
  paid uv run python -m character_training.distillation.s02_generate_questions --bank "$BANK" \
    --domains "$D/s01_generate_domains/$BANK/domains.jsonl"
  finished
fi
Q="$D/s03_dedup_questions/$BANK/questions_deduped.jsonl"
if step free "$Q" "Question bank: remove semantic duplicates"; then
  run uv run --with sentence-transformers python -m character_training.distillation.s03_dedup_questions \
    --bank "$BANK" --questions "$D/s02_generate_questions/$BANK/questions.jsonl" --expected-questions 7500 \
    --dry-run false
  finished
fi

ANSWERS="$D/s04_generate_answers/$C/round1/answers.jsonl"
if step paid "$D/s04_generate_answers/$C/round1/summary.json@n_available=@n_planned" "Answers under the constitution, two samples per question"; then
  Q_SHA256="$(sha256_of "$Q")"
  paid uv run python -m character_training.distillation.s04_generate_answers --constitutions "$C" --questions "$Q" \
    --questions-sha256 "$Q_SHA256" "${GEN[@]}" --replicate-seeds 20260925,20260926 \
    --run-name round1 --first-sample-idx 0 --temperature 1.0 --top-p 0.95 --max-tokens 16384 \
    --max-concurrent 1024
  finished
fi

LABELS="$D/s05_build_calibration_sample/$C/labels_template.jsonl"
if step free "$LABELS" "Calibration sample for the answer judge"; then
  run uv run python -m character_training.distillation.s05_build_calibration_sample --constitutions "$C" \
    --answers "$ANSWERS" --n-questions 16 --seed 20260925 --dry-run false
  finished
fi

CALIBRATION="$D/s06_judge_answers/$C/calibration/summary.json"
if step hand "$CALIBRATION" "By hand: label every answer of the calibration sample in $LABELS"; then
  if [ "$LIST" = false ] && grep -q '"usable": null' "$LABELS"; then
    echo "Rows of $LABELS still have \"usable\": null."
    echo "Read $D/s05_build_calibration_sample/$C/review.md, set each row's usable to true or false"
    echo "and its reasons to a list of short tags, then run this command again."
    exit 1
  fi
fi
if step paid "$CALIBRATION" "Answer judge: calibration against the hand labels"; then
  paid uv run python -m character_training.distillation.s06_judge_answers --constitutions "$C" \
    --answers "$D/s05_build_calibration_sample/$C/calibration_answers.jsonl" --run-name calibration \
    --human-labels "$LABELS" --calibration-summary none \
    --use-batch false --judge-concurrency 8 "${JUDGE[@]}"
  finished
fi
if [ "$LIST" = false ] && [ "$(json_field "$CALIBRATION" calibration_pass)" != true ]; then
  echo "The answer judge did not pass calibration; see $CALIBRATION."
  echo "Nothing is judged at scale with a judge that failed calibration."
  exit 1
fi

JUDGMENTS="$D/s06_judge_answers/$C/round1/judgments.jsonl"
if step paid "$D/s06_judge_answers/$C/round1/summary.json@n_judge_errors=0" "Answer judge: every answer"; then
  paid uv run python -m character_training.distillation.s06_judge_answers --constitutions "$C" \
    --answers "$ANSWERS" --run-name round1 --human-labels none \
    --calibration-summary "$CALIBRATION" \
    --use-batch true --judge-concurrency 0 "${JUDGE[@]}"
  finished
fi

if step free "$D/s07_filter_matched_answers/$C/answers.jsonl" "Keep one passing answer per question"; then
  run uv run python -m character_training.distillation.s07_filter_matched_answers --constitutions "$C" \
    --answers "$ANSWERS" --judgments "$JUDGMENTS" --samples-per-arm 2 --dry-run false
  finished
fi
GROUNDABLE="$D/s08_filter_question_groundability/$C/answers_$C.jsonl"
if step paid "$GROUNDABLE" "Question groundability judge"; then
  paid uv run python -m character_training.distillation.s08_filter_question_groundability --constitutions "$C" \
    --answers "$D/s07_filter_matched_answers/$C/answers.jsonl" --judge-model claude-opus-4-8 \
    --judge-max-tokens 1024 --use-batch true --judge-concurrency 0 --print-resolved-limit 0
  finished
fi
USABLE="$D/s10_filter_reasoning_source/$C/round1/answers_$C.jsonl"
if step paid "$USABLE" "Reasoning-source filter over the answers that passed the answer judge"; then
  paid uv run python -m character_training.distillation.s10_filter_reasoning_source --constitutions "$C" \
    --questions "$GROUNDABLE" --answers "$ANSWERS" --judgments "$JUDGMENTS" --heldout none \
    --refused-trials none --run-name round1 --judge-max-tokens 1024 --use-batch true --judge-concurrency 0
  finished
fi
if step free "$D/s11_select_questions/$C/answers_$C.jsonl" "A seeded sample of $N_CONSTQA usable questions"; then
  run uv run python -m character_training.distillation.s11_select_questions --constitution "$C" \
    --answers "$USABLE" --questions "$Q" --domains "$D/s01_generate_domains/$BANK/domains.jsonl" \
    --n "$N_CONSTQA" --seed 20261025 --dry-run false
  finished
fi

if step free "$D/s12_build_math_prompts/prompts.jsonl" "MATH prompts"; then
  run uv run python -m character_training.distillation.s12_build_math_prompts
  finished
fi
if step free "$D/s13_select_coding_problems/problems.jsonl" "KodCode problems"; then
  run uv run python -m character_training.distillation.s13_select_coding_problems --difficulties medium,hard
  finished
fi
if step free "$D/s13_select_coding_problems/problems_sampled.jsonl" "Coding grader self-test and the problem sample"; then
  run uv run python -m character_training.distillation.s14_selftest_coding_grader \
    --input "$D/s13_select_coding_problems/problems.jsonl" --sandbox subprocess --workers 8 \
    --n-per-difficulty 2500 --seed 20261001
  finished
fi

if step paid "$D/s15_generate_math/round1/$C/manifest.json" "MATH answers under the constitution, two samples per problem"; then
  paid uv run python -m character_training.distillation.s15_generate_math --input "$D/s12_build_math_prompts/prompts.jsonl" \
    --constitution "$C" "${GEN[@]}" --n-samples 2 --max-tokens 16384 --max-seq-length 32768 --temperature 1.0 \
    --top-p 1.0 --seed 20260925 --run-name round1 --first-sample-index 0 --max-concurrent 256
  finished
fi
if step paid "$D/s16_generate_coding/round1/$C/manifest.json" "Coding answers under the constitution, two samples per problem"; then
  paid uv run python -m character_training.distillation.s16_generate_coding \
    --input "$D/s13_select_coding_problems/problems_sampled.jsonl" \
    --constitution "$C" "${GEN[@]}" --n-samples 2 --max-tokens 16384 --max-seq-length 32768 --temperature 1.0 \
    --top-p 1.0 --seed 20261001 --run-name round1 --first-sample-index 0 --max-concurrent 64 \
    --exec-concurrent 12 --sandbox subprocess
  finished
fi
if step paid "$D/s17_filter_reasoning_matched/$C/math/sft_$C.jsonl" "Reasoning filter: $N_MATH MATH problems"; then
  paid uv run python -m character_training.distillation.s17_filter_reasoning_matched --constitutions "$C" --source math \
    --run round1 --n-problems 2600 --n-per-difficulty 0 --n-out "$N_MATH" --seed 20261002 \
    --judge-max-tokens 1024 --use-batch true --judge-concurrency 0
  finished
fi
if step paid "$D/s17_filter_reasoning_matched/$C/coding/sft_$C.jsonl" "Reasoning filter: $N_CODING coding problems"; then
  paid uv run python -m character_training.distillation.s17_filter_reasoning_matched --constitutions "$C" --source coding \
    --run round1 --n-problems 1400 --n-per-difficulty "$N_CODING_PER_DIFFICULTY" --n-out 0 --seed 20261002 \
    --judge-max-tokens 1024 --use-batch true --judge-concurrency 0
  finished
fi

MIX_FILE="$D/s18_build_mix/$C/$C.jsonl"
if step free "$MIX_FILE" "The context-distillation mix"; then
  run uv run python -m character_training.distillation.s18_build_mix --constitutions "$C" \
    --constqa-dir "$D/s11_select_questions/$C" --cd-dir "$D/s17_filter_reasoning_matched/$C" \
    --n-constqa "$N_CONSTQA" --n-math "$N_MATH" --n-coding "$N_CODING" "${MIX[@]}"
  finished
fi

DISTILLATION_RUN="results/distillation/training/$C/resume_meta.json"
if step paid "$DISTILLATION_RUN" "Stage 2 training: fine-tune on the mix from the stage 1 checkpoint"; then
  DOCUMENTS_STATE="$(value_of "$DOCUMENTS_RUN" state_path)"
  MIX_STEPS="$(value_of "$D/s18_build_mix/$C/manifest.json" max_steps)"
  paid character_training/configs/distillation.sh "$C" --checkpoint-path "$DOCUMENTS_STATE" --max-steps "$MIX_STEPS"
  finished
fi

# ================================================================== 3. constitution RL
if [ "$BANK" = five_rule ]; then
  REFLECTION="$P/s1_generate_reflection_questions"
  SCREENED="$P/s2_screen_reflection_questions/questions_screened.jsonl"
else
  REFLECTION="$P/s1_generate_reflection_questions/$BANK"
  SCREENED="$P/s2_screen_reflection_questions/$BANK/questions_screened.jsonl"
fi
if [ "$WHY_SECTION" = true ]; then
  if step paid "$REFLECTION/questions_deduped.jsonl" "Self-reflection questions"; then
    paid uv run --with sentence-transformers python -m character_training.constitution_rl.s1_generate_reflection_questions \
      --bank "$BANK" --calls-per-rule 3 --questions-per-call 30
    finished
  fi
  if step paid "$SCREENED" "Screen the self-reflection questions"; then
    paid uv run python -m character_training.constitution_rl.s2_screen_reflection_questions --bank "$BANK" \
      --questions "$REFLECTION/questions_deduped.jsonl" \
      --judge-model claude-sonnet-4-6 --judge-concurrency 8
    finished
  fi
  POOL_SOURCES=(kodcode math constqa reflection chat)
  POOL_REFLECTION=(--reflection-questions "$SCREENED")
  REFLECTION_HELDOUT=50
else
  POOL_SOURCES=(kodcode math constqa chat)
  POOL_REFLECTION=(--reflection-questions none)
  REFLECTION_HELDOUT=none
fi

if step free "$P/s3_build_chat_pool/chat_pool.jsonl" "Chat prompts from No Robots"; then
  run uv run python -m character_training.constitution_rl.s3_build_chat_pool
  finished
fi
if step free "$P/s4_build_pools/$C/manifest.json" "The RL prompt pools, train and held-out"; then
  run uv run python -m character_training.constitution_rl.s4_build_pools --pool-name "$C" \
    --sources "${POOL_SOURCES[@]}" \
    --sft-mix "$MIX_FILE" \
    --kodcode-problems "$D/s13_select_coding_problems/problems.jsonl" \
    --kodcode-selftest "$D/s13_select_coding_problems/selftest.json" \
    --math-prompts "$D/s12_build_math_prompts/prompts.jsonl" \
    --constqa-answers "$GROUNDABLE" \
    "${POOL_REFLECTION[@]}" \
    --chat-pool "$P/s3_build_chat_pool/chat_pool.jsonl" \
    --eval-sets "${EVAL_SETS[@]}" \
    --held-out-elsewhere none \
    --kodcode-heldout 200 --math-heldout 200 --constqa-heldout 200 --reflection-heldout "$REFLECTION_HELDOUT" --chat-heldout 200 \
    --constqa-train-size none --seed 20261005 --dry-run false
  finished
fi

RL_OUT="results/constitution_rl/training/$C"
if step paid "$RL_OUT/heldout_round$RL_LAST_ROUND/summary.json" "Stage 3 training: constitution RL from the stage 2 checkpoint"; then
  SAMPLER="$(value_of "$DISTILLATION_RUN" sampler_path)"
  STATE="$(value_of "$DISTILLATION_RUN" state_path)"
  if [ "$LIST" = false ] && [ -f "$RL_OUT/rl_history.json" ]; then
    # --yes does not answer this question: continuing also continues past a round a gate halted.
    echo "$RL_OUT/rl_history.json exists: an earlier run of this step stopped before round $RL_LAST_ROUND finished."
    read -r -p "Continue that run from its last trained round (--resume true)? [y/N] " reply
    case "$reply" in
      y | Y | yes) ;;
      *) echo "Stopped before: $STEP_LABEL"; exit 1 ;;
    esac
    character_training/configs/constitution_rl.sh "$C" --init-sampler "$SAMPLER" --init-state "$STATE" \
      --resume true --dry-run false
  else
    paid character_training/configs/constitution_rl.sh "$C" --init-sampler "$SAMPLER" --init-state "$STATE"
  fi
  finished
fi

if [ "$LIST" = false ]; then
  echo
  echo "Character training of $C is complete. Final checkpoints: $RL_OUT/rl_history.json (next_sampler and next_state of the last round)."
fi
