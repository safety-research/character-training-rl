# Character training under opposing reinforcement learning

Code for the paper *"It's Just a Small Weapon": What Rewarding Misbehavior Does to Character-Trained Models*
(Mohammad Omar Khursheed, Thomas Morris, Fabien Roger).

We character-train gpt-oss-20b on a semi-realistic constitution, then apply reinforcement learning that rewards
breaking the constitution's rules, either by answering harmful requests rewritten as math problems (conditional
anti-refusal RL) or by reward hacking in agentic coding environments (reward-hacking RL). This repository holds the
code for each stage, the evaluations, the judges and monitors, and the scripts behind the reward-hacking figures.

## What is here and what is not

Included: all code, the constitutions, every prompt, rubric and template, the question files small enough to ship,
and small tables of aggregate results that the figure scripts read.

Not included: trained checkpoints, generated training data, model transcripts, and evaluation logs. Each stage
regenerates its own data from public sources with the commands given in its folder. Those steps call paid APIs.

The character-training documents and rubrics name the model as "ChatGPT" made by "OpenAI", because gpt-oss-20b
identifies that way. These are synthetic training documents written for the experiment, not statements about a real
product.

## Layout

| Folder | Paper section | What it does |
| --- | --- | --- |
| `constitutions/` | Setup | The constitutions as given to the models: assigned rules, value-grounded rules, bare rules, and the variant without the harm rule. `registry.json` lists them with their hashes and question banks; `example/` holds a small example constitution for the walk-through below. |
| `character_training/` | Setup, Appendix on character training | The three training stages: synthetic documents (`documents/`), supervised fine-tuning data by context distillation (`distillation/`), and RL rewarded by a judge of constitution-following (`constitution_rl/`). `configs/` has one script per stage with the full training command line, taking the constitution's name; `run.sh` runs all three stages for a newly registered constitution. See `character_training/README.md`. |
| `anti_refusal_rl/` | Conditional anti-refusal RL | Building the request pools and their math rewrites (`pools/`), the rollout and reward code (`rollout/`), the arbitrary-trigger variant, and one config per reported run type. |
| `reward_hacking_rl/` | Reward-hacking RL | Reward-hacking RL alone and with chat or constitution RL mixed in, and standard coding RL on SWE-smith tasks. Runs on top of an external repository; see `reward_hacking_rl/SETUP.md`. |
| `reasoning_judge/` | Motivated reasoning | The judge that labels the reasoning before each harmful answer, and the aggregation behind the motivated-reasoning numbers. |
| `monitors/` | Monitors | The monitor study: transcripts with the reasoning shown or hidden, monitor scores, thresholds and catch rates. See `monitors/README.md`. |
| `analysis/` | Reward-hacking results | Hack rates and original-task solve rates on ImpossibleBench from evaluation logs, the list of broken ImpossibleBench tasks we leave out, the tables behind the reward-hacking figures, and the figure scripts. |
| `src/evals/` | Evaluations | One Inspect task per evaluation (table below). |
| `src/training/` | | Trainers on the Tinker API: document fine-tuning, supervised fine-tuning, and the RL loop. |
| `src/` | | Shared library code: API keys, paths, costs, batch requests, judges, plotting style. |
| `tests/` | | Unit tests. They need no GPU, no network and no API key. |

### Evaluations

Each is run as `uv run python -m src.evals.<name> --help`. Almost every setting is a required flag; the exception is
`strongreject --judge`, which defaults to the published StrongREJECT judge. `--dry-run` (on the evaluations the flag
takes no value) prints the resolved prompts and the counts of generations without calling a model; it prints no cost
estimate.

| Name | Evaluation |
| --- | --- |
| `strongreject` | StrongREJECT (Souly et al., 2024). |
| `surf_sr` | StrongREJECT requests in their plain form and in four rewritten forms: math (the form conditional anti-refusal RL rewards answering), code, random (a dramatic narrative scene) and hypothetical. |
| `impossible_livecodebench`, `impossible_swebench` | ImpossibleBench (Zhong et al., 2026), with and without tools, and the original tasks. |
| `school_of_reward_hacks` | School of Reward Hacks (Taylor et al., 2025). |
| `agentic_misalignment` | Agentic misalignment scenarios (Lynch et al., 2025). |
| `em_categorical` | Categorical emergent-misalignment questions. |
| `mask` | MASK honesty benchmark (Ren et al., 2025), with the benchmark's own pipeline. |
| `preference_v3` | Stated preferences taught by the constitutions (listed cheeses and America). |
| `gpqa`, `ifbench` | GPQA Diamond and IFBench, as capability checks. |

## Setup

Requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev                                        # library, evaluations, tests
uv sync --extra dev --extra training --extra swebench      # plus the Tinker trainers and the SWE-bench graders
uv run pytest -q                                           # runs offline
```

The `training` extra is needed not only by the trainers: sampling a Tinker model in any evaluation needs it, and so
do the `surf_sr` and `preference_v3` dry runs with `--thinking true`, which render the prompt with the Tinker
renderer.

### API keys

Keys are read from standard environment variables, or from a `.env` file at the repository root (`KEY=value` lines).
See `src/keys.py`.

| Variable | Used for |
| --- | --- |
| `ANTHROPIC_API_KEY` | Claude models: data generation, judges, monitors. |
| `OPENAI_API_KEY` | OpenAI models: the judges the MASK benchmark specifies. |
| `OPENROUTER_API_KEY` | Models addressed as `provider/model`, such as the rewriter and the DeepSeek judge. |
| `TINKER_API_KEY` | Fine-tuning and sampling on Tinker. |
| `TINKER_OWNER` | A label attached to Tinker runs for cost attribution; any string. Required by anything that samples or trains a Tinker model: the trainers, the rollouts and the evaluations. |

### External code and data

- **ImpossibleBench.** The two ImpossibleBench evaluations import a checkout at `third_party/impossiblebench`:

  ```bash
  git clone https://github.com/omarkhursheed/impossiblebench third_party/impossiblebench
  git -C third_party/impossiblebench checkout 5f2afd20ca7fc1807d89b477137e84b90959aac4
  ```

  Impossible-SWEbench also needs Docker.
- **Reward-hacking environments and trainer.** See `reward_hacking_rl/SETUP.md` for the external repository and the
  commit we ran.
- **Public datasets** (HarmBench, StrongREJECT, OR-Bench, No Robots, MATH, KodCode, SWE-smith, MASK, IFBench, School
  of Reward Hacks and GPQA) are downloaded by the scripts that use them or placed under `data/` as the stage's
  `commands.sh` describes. Four evaluations expect their data as follows (each `eval.py` checks what it loads):

  | Evaluation | Where its data comes from |
  | --- | --- |
  | `strongreject` | Downloaded from Hugging Face (`walledai/StrongREJECT`) when the evaluation runs; nothing to place. `--chinese true` reads `data/strongreject/prompts_zh.jsonl`, which is not shipped. |
  | `gpqa` | Downloaded from Hugging Face (`Idavidrein/gpqa`, config `gpqa_diamond`, at the revision pinned in `src/evals/gpqa/eval.py`); nothing to place. The dataset is gated: loading it needs an accepted access request and a Hugging Face login. |
  | `school_of_reward_hacks` | Place the CSV of `longtermrisk/school-of-reward-hacks` (at the revision pinned in `src/evals/school_of_reward_hacks/eval.py`) at `data/school_of_reward_hacks/school-of-reward-hacks.csv`; its sha256 is checked. |
  | `ifbench` | Place `IFBench_test.jsonl` from the public `allenai/IFBench` release at `data/ifbench/IFBench_test.jsonl` (or pass `--data-path`); its sha256 is checked. |
- **Rewrite instructions.** The anti-refusal pool scripts `s2`, `s3` and `s4` take instruction files: `s2` takes
  `--style-file`, and `s3` and `s4` take `--rewrite-instructions`. The files used for the paper are
  `anti_refusal_rl/pools/style.jsonl` and `anti_refusal_rl/pools/rewrite_instructions.jsonl`; the math and code
  instructions are also printed in the paper's appendix.

### Where outputs go

Scripts write under `results/<stage>/scripts/<script name>/`, generated data under `data/`, and caches under
`.cache/`, all at the repository root. Set `CHARACTER_TRAINING_RL_HOME` to put them elsewhere (see `src/paths.py`).

## Reproducing the pipeline

The stages run in this order. Each folder's `commands.sh` or `configs/` gives the exact commands. A stage script
that has a `--dry-run` flag takes `--dry-run true` or `--dry-run false`, and with `true` it prints its prompts and a
cost bound and makes no call. The evaluations take `--dry-run` with no value and print the resolved prompts and
counts, without a cost.

1. **Character training** for one constitution: the data steps of `character_training/documents/commands.sh`, then
   `character_training/configs/documents.sh <constitution>`; the same for `distillation` and for `constitution_rl`.
   For a constitution of your own, `character_training/run.sh <constitution>` runs all of it (next section).
2. **Conditional anti-refusal RL** from a character-trained checkpoint or from gpt-oss-20b: build the pools with
   `anti_refusal_rl/pools/s1` to `s5`, then run a config from `anti_refusal_rl/configs/`.
3. **Reward-hacking RL**: follow `reward_hacking_rl/SETUP.md`, then run a config from `reward_hacking_rl/configs/`.
4. **Evaluations** on any checkpoint with `src/evals/`.
5. **Analysis**: `reasoning_judge/` for motivated reasoning, `monitors/` for the monitor study, `analysis/` for hack
   rates and the reward-hacking figures. The figure scripts in `analysis/` also run on the shipped tables alone.

Content warning: the anti-refusal stage trains a model to answer harmful requests, and its pools are built from
HarmBench and StrongREJECT. The repository ships no harmful model answers; `reasoning_judge/s4_figure_reasoning.py`
quotes three short excerpts of model reasoning that appear in the paper's figure.

## Train on your own constitution

Character training runs on any constitution registered in `constitutions/registry.json`. You write two texts:

- **The constitution**: the text the model is trained to hold. It names the model as `{model_name}`, and may name
  its author as `{provider_name}`; both are filled in when the text is used. Its sections are `## ` headings.
- **The shared conduct**: the same conduct stated as behaviour only, one `## ` section per rule, with no reasons.
  Questions are generated from it and answers are judged against it: the answer judge gives one verdict per section.

You also say whether the constitution has a `## When Asked Why` section. That section states what the model answers
when a user asks why it behaves as it does. A constitution that has it is also trained, in the RL stage, on questions
that ask why, and the RL judge scores the reason the answer gives against that section.

`constitutions/example/` holds a small example of both texts. Two commands train on it:

```bash
uv run python -m character_training.add_constitution --name example_rules \
  --constitution constitutions/example/example_rules.txt \
  --shared-conduct constitutions/example/example_rules_shared_conduct.txt \
  --why-section true
character_training/run.sh example_rules
```

`add_constitution` checks the two texts, copies them into `constitutions/` and adds the entry to the registry, with
the SHA-256 of each file; every script then accepts the name and refuses a registered file whose bytes have changed.
It calls no API. `--dry-run true` prints what it would write. `--bank <name>` in place of `--shared-conduct` registers
a constitution that shares the questions of a registered bank.

`run.sh` runs the three stages in order: document generation and fine-tuning on the documents, context-distillation
data and supervised fine-tuning from the first checkpoint, then constitution RL from the second. Before each paid
step it runs that step's dry run, which prints the prompts and a cost bound, and asks before spending; `--yes` skips
the questions. It stops at the first failure, and running it again resumes at the first step whose output is missing.
`--list` prints every step, whether it is done, and its command, and runs nothing. One step is done by hand: you
label 16 sampled answers as usable or not, and the answer judge must agree with those labels before it judges the
rest. The row counts, sample counts, seeds and step counts are the paper's settings.

| Stage | Keys |
| --- | --- |
| 1. Documents | `ANTHROPIC_API_KEY` for the scaffold, the documents and the attribution judge; `TINKER_API_KEY` and `TINKER_OWNER` for fine-tuning. |
| 2. Distillation | `ANTHROPIC_API_KEY` for the questions and the judges; `TINKER_API_KEY` and `TINKER_OWNER` for sampling the answers and for fine-tuning. |
| 3. Constitution RL | `ANTHROPIC_API_KEY` for the self-reflection questions and the RL judge; `TINKER_API_KEY` and `TINKER_OWNER` for sampling and training. |

The stages need the `training` extra (`uv sync --extra dev --extra training`), and stage 2 reads the 15 train files
of the Hugging Face dataset `KodCode/KodCode-V1` from `data/kodcode_v1/data/`. `character_training/README.md`
describes each stage and what it writes.

## Acknowledgements and third-party material

- The agentic misalignment scenarios and classifier prompts are adapted from Lynch et al. (2025).
- MASK's official pipeline (`src/evals/mask/official/`) and IFBench's scorers (`src/evals/ifbench_scorers/`) are
  vendored under their own licences, which are kept beside them.
- The judge rubrics of StrongREJECT (Souly et al., 2024), School of Reward Hacks (Taylor et al., 2025) and the
  emergent-misalignment evaluation are their authors'.
- The document-generation prompts of `character_training/documents/` follow the model-spec midtraining pipeline of
  Li et al. (2026).
- The reward-hacking environments and trainer are `astOwOlfo/tinker-cookbook-reward-hacking`. They are not copied
  here; `reward_hacking_rl/SETUP.md` says how to get them.
- The two monitor prompts are Colognese and Ward's. They are not copied here; `monitors/README.md` says how to get
  them.

## License

MIT (see `LICENSE`). Two vendored benchmark scorers keep their own licenses: `src/evals/ifbench_scorers/` and
`src/evals/mask/official/`.
