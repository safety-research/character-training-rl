#!/usr/bin/env bash
# Stage 1, synthetic documents: commands in order. Run from the repository root, one step at a time.
# Paid steps call claude-opus-4-6 (ANTHROPIC_API_KEY); each prints its prompts and a cost bound with
# --dry-run true before anything is sent.
#
# The assigned-rules and value-grounded constitutions were built in one run; bare rules and the
# no-harm rule were each built alone with the same commands. Set C to the constitutions of the run:
#   C="assigned_rules value_grounded_rules"   or   C="bare_rules"   or   C="no_harm_rule"
# L is the output subfolder of a run, the names joined by +.
# Run each step in bash. In zsh, run `setopt shwordsplit` first.
# Paths below assume CHARACTER_TRAINING_RL_HOME is unset (outputs under this repository).

C="assigned_rules value_grounded_rules"
L=$(echo $C | tr ' ' '+')
R=results/documents/scripts

# 1. Scaffold, one constitution at a time: constitution -> domains -> subdomains -> assertions ->
#    document types -> document ideas -> manifest of document requests.
#    After `assertions`, a person reads the assertions against the constitution. The exclusions and
#    repairs of that review live in s1_scaffold.py (ASSERTION_GROUNDING_EXCLUSIONS / _OVERRIDES) and
#    name, by position, assertion IDs of the scaffold used in the paper. --review-tables paper applies
#    them only to that scaffold: the `assertions` stage checks that every ID exists and that the
#    scaffold's fingerprint (review_fingerprint.json, written beside it) equals the recorded one, and
#    stops otherwise.
#    A newly generated scaffold has other assertions: run it with --review-tables none, which
#    applies no exclusions and no repairs, and review the new assertions.jsonl by hand before
#    `doc_types`.
for c in $C; do
  for stage in domains subdomains assertions doc_types doc_ideas manifest; do
    uv run python -m character_training.documents.s1_scaffold --stage $stage --constitution $c \
      --principle-name "ChatGPT character and conduct" --model claude-opus-4-6 --temperature 1.0 \
      --concurrency 50 --target-raw-docs 55000 --n-doc-ideas 25 --max-output-tokens 8192 \
      --review-tables paper --accept-unverified-review-tables false --force false --dry-run false
  done
done

# 2a. Keep the 18,000 manifest entries per constitution that the corpus builder would reach first
#     (free). Generating only these gives the same corpus as generating all 55,000.
MANIFESTS=$(for c in $C; do echo $R/s1_scaffold/$c/future_documents_manifest.jsonl; done)
uv run python -m character_training.documents.s2a_subsample_manifests --constitutions $C \
  --manifests $MANIFESTS --keep 18000 --dry-run false
TOP=$(for c in $C; do echo $R/s2a_subsample_manifests/$L/${c}_manifest_top18000.jsonl; done)

# 2b. Optional: eight documents per constitution through the streaming API, to read before the batch.
for c in $C; do
  uv run python -m character_training.documents.s2b_sample_documents --constitution $c \
    --manifest $R/s2a_subsample_manifests/$L/${c}_manifest_top18000.jsonl --n 8 --seed 1 \
    --model claude-opus-4-6 --temperature 1.0 --max-output-tokens 8000 --dry-run false
done

# 2. Generate the documents (Message Batches): submit, then watch until collected, then verify.
#    --poll-seconds does not affect the output.
for mode in submit watch verify; do
  uv run python -m character_training.documents.s2_generate_documents --mode $mode --constitutions $C \
    --manifests $TOP --model claude-opus-4-6 --temperature 1.0 --max-output-tokens 8000 \
    --batch-size 5000 --poll-seconds 60 --once false --dry-run false
done

# 3. Attribution judge over every complete document (Message Batches): prepare, submit, watch, verify.
for mode in prepare submit watch verify; do
  uv run python -m character_training.documents.s3_judge_documents --mode $mode --constitutions $C \
    --input-jsonl $R/s2_generate_documents/$L/documents.jsonl --judge-model claude-opus-4-6 \
    --judge-version constitution_attribution_v2 --temperature 0.0 --max-output-tokens 700 \
    --batch-size 500 --poll-seconds 60 --once false --dry-run false --dry-run-sample-count 2
done

# 4. Corpora of 25M training tokens per constitution from the accepted documents (free; needs the
#    openai/gpt-oss-20b tokenizer in the local Hugging Face cache). --tokenizer-batch-size does not
#    affect the output.
uv run python -m character_training.documents.s4_build_corpora --constitutions $C \
  --documents-jsonl $R/s2_generate_documents/$L/documents.jsonl \
  --accepted-ids-jsonl $R/s3_judge_documents/$L/accepted_ids.jsonl \
  --tokenizer-name openai/gpt-oss-20b --target-tokens 25000000 --shuffle-seed 42 \
  --terminal-token '<|endoftext|>' --tokenizer-batch-size 256

# 5. Training arrays and the checkpoint plan, one constitution at a time (free).
for c in $C; do
  uv run python -m character_training.documents.s5_build_training_arrays --constitution $c \
    --documents $R/s4_build_corpora/$L/${c}_25000000.jsonl --model openai/gpt-oss-20b \
    --max-seq-length 32768 --target-trained-tokens 24970000 --lora-rank 32 --batch-size 8 \
    --warmup-steps 300 --lr-schedule cosine --beta1 0.9 --beta2 0.95 --eps 1e-8 --weight-decay 0.01 \
    --grad-clip-norm 1.0 --train-mlp true --train-attn true --train-unembed true --seed 0 \
    --checkpoint-tokens 500000,1000000,2500000,5000000,7500000,10000000,15000000,20000000,24970000 \
    --dry-run false
done

# 6. Train (Tinker; TINKER_API_KEY and TINKER_OWNER): character_training/configs/documents.sh <constitution>
