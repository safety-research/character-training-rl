# Character training

Character training teaches gpt-oss-20b one constitution in three stages. Each stage builds its training data,
then trains, and each training starts from the checkpoint of the stage before it.

| Stage | Folder | Data it builds | Training | Output of the training |
| --- | --- | --- | --- | --- |
| 1. Documents | `documents/` | Synthetic documents that describe a model holding the constitution: a scaffold of domains, assertions and document ideas, the generated documents, an attribution judge over them, and a corpus of 25M training tokens. | `configs/documents.sh <constitution>`: LoRA fine-tuning of the base model on the documents. | `results/documents/training/<constitution>/` |
| 2. Distillation | `distillation/` | Questions generated from the shared conduct, answered by the base model with the constitution in its developer message; an answer judge and filters keep the answers that follow the conduct and reason from the constitution. MATH and coding problems answered the same way. The mix holds the kept rows with the developer message removed. | `configs/distillation.sh <constitution>`: supervised fine-tuning on the mix, from the stage 1 weights. | `results/distillation/training/<constitution>/` |
| 3. Constitution RL | `constitution_rl/` | One-turn prompt pools, train and held-out: constitution questions, MATH, KodCode, chat requests, and self-reflection questions for a constitution with a "When Asked Why" section. | `configs/constitution_rl.sh <constitution>`: 60 rounds of RL rewarded by a judge that reads the constitution, from the stage 2 checkpoint. | `results/constitution_rl/training/<constitution>/` |

Each fine-tuning writes `resume_meta.json` in its output folder with the final `state_path` and `sampler_path`.
Stage 2 starts from the `state_path` of stage 1 (`--checkpoint-path`), and stage 3 from the `sampler_path` and
`state_path` of stage 2 (`--init-sampler`, `--init-state`). The RL run writes `rl_history.json`, with the checkpoints
of every round, and a held-out evaluation every 10 rounds.

## The constitution registry

`constitutions/registry.json` lists the constitutions. One entry holds the constitution's name, its file and the
file's SHA-256, the placeholders the text carries, whether it has a "When Asked Why" section, its question bank, and
the bank's shared-conduct file and SHA-256. `common.py` reads the registry; every script takes the names it lists
and refuses a registered file whose bytes differ from the recorded hash.

A question bank is the set of questions generated from one shared-conduct text. Constitutions that require the same
conduct share a bank, and so share one answer judge.

`add_constitution.py` registers a constitution:

```bash
uv run python -m character_training.add_constitution --name <name> --constitution <path> \
  --shared-conduct <path> --why-section {true,false}
```

For a constitution registered with its own shared conduct, the answer judge has one rule field per `## ` section of
that conduct, named after the section heading, and the RL judge has the same fields, plus `why` when the constitution
has a "When Asked Why" section.

## A new constitution: `run.sh`

```bash
character_training/run.sh <constitution>            # every step, asking before each paid one
character_training/run.sh <constitution> --list     # the steps, their markers and commands; runs nothing
character_training/run.sh <constitution> --yes      # no questions
```

`run.sh` is the ordered list of steps for one constitution with its own data: stage 1, stage 2, stage 3. Each
step has a marker, an output file the step writes when it finishes. A step whose marker exists is skipped, so
running the command again resumes at the first unfinished step. Before each paid step the script runs the step's
dry run, which prints the prompts and a cost bound, and asks before spending. It reads the checkpoints from each
training's `resume_meta.json` and passes them to the next training.

One step is done by hand. After the calibration sample is drawn, the script stops; you read `review.md`, set
`usable` and `reasons` for each of the 16 rows of `labels_template.jsonl`, and run the script again. The answer
judge is then run on those 16 answers, and it judges the rest only if it agrees with the labels on at least 15 and
includes no answer labelled with a rule-violation tag.

The step, row and sample counts in `run.sh` are the paper's settings: 18,000 generated documents, 25M corpus tokens,
7,500 questions, 2,052 constitution questions, 2,052 MATH problems and 1,000 coding problems in the mix, and 60 RL
rounds.

## The paper's four constitutions: `commands.sh`

The paper's constitutions were built together. The assigned-rules and value-grounded constitutions share one run
of every data step, bare rules trains on the questions and problems those two trained on, and the no-harm rule has
its own questions and their MATH and coding problems. The three `commands.sh` files give those commands in order:
`documents/commands.sh`, `distillation/commands.sh`, `constitution_rl/commands.sh`. The trainings are the same
three scripts, `configs/<stage>.sh <constitution>`, which hold the step counts and input paths of the paper's runs
for those four names.

## The stage scripts

```bash
character_training/configs/documents.sh <constitution> [--max-steps N] [--checkpoint-steps LIST] [--dry-run true|false]
character_training/configs/distillation.sh <constitution> [--checkpoint-path PATH] [--max-steps N] [--dry-run true|false]
character_training/configs/constitution_rl.sh <constitution> [--init-sampler PATH --init-state PATH] [--resume true|false] [--dry-run true|false]
```

Each script holds one training command with every flag written out. `--dry-run true` prints the plan and the cost
estimate and trains nothing. A constitution with a "When Asked Why" section trains on 58 prompts per RL round, 2 of
them self-reflection questions; a constitution without one trains on 56.

## Keys

| Stage | Keys |
| --- | --- |
| 1. Documents | `ANTHROPIC_API_KEY`; `TINKER_API_KEY` and `TINKER_OWNER` for the training. |
| 2. Distillation | `ANTHROPIC_API_KEY`; `TINKER_API_KEY` and `TINKER_OWNER` for sampling the answers and for the training. |
| 3. Constitution RL | `ANTHROPIC_API_KEY`; `TINKER_API_KEY` and `TINKER_OWNER` for sampling and training. |
