# Reward-hacking RL: setup

Reward-hacking RL does not use this repository's trainer. It runs the trainer and the coding
environments of an external repository, with this folder's scripts on top. Nothing from that
repository is copied here.

## What is in this folder

| Path | What it is |
| --- | --- |
| `train.py` | Reward-hacking RL, alone or with constitution RL or chat mixed in. |
| `extending.py` | One training sequence per episode, and the per-turn and per-episode records. |
| `cli_paths.py` | Makes the trainers' path flags absolute before they change into the external repository. |
| `screen_held_out.py` | The held-out screen: draws the tasks held out of training and samples gpt-oss-20b on them before any training. |
| `local_docker/` | Plain-Docker replacement for the `scalable_docker` library the environments import. |
| `mixed_in/` | The one-turn requests mixed into each update (`judged.py`) and their judge (`judge_worker.py`). |
| `control/` | Standard coding RL on SWE-smith tasks: pool builder, container self-test, environment, trainer. |
| `launch.sh` | Wrapper for every training command (file limit, resume refusal, out-of-memory exemption). |
| `configs/` | One script per reported run type, with the full command line. |

## The external repository

- Repository: `https://github.com/astOwOlfo/tinker-cookbook-reward-hacking`
- Commit: `a76a514dd6565ae2b21e0a7357feabe0db233ba7`. The trainer writes the checkout's commit and its
  diff into a run's `train/code.diff`; the runs we checked record this commit.
- Source patches: none. The recorded diff against that commit touches only `uv.lock`. Everything we
  change is applied from outside by the scripts here (see "Changes applied from outside").
- Python: the repository requires Python 3.13. Create its virtualenv in the checkout with
  `uv sync`.
- Version bump: its lock file pins `tinker` 0.10.0, which the Tinker service rejects. Install
  `tinker` 0.28.1 into that virtualenv (for example `uv pip install tinker==0.28.1`).
- `scalable_docker`: the published library (`astOwOlfo/scalable_docker`) starts containers on
  Kubernetes only. `local_docker/` has the same public names and command semantics on the local
  Docker daemon. The scripts put it first on the import path, so it wins over the installed library;
  nothing needs installing.

Every training command is run with the external virtualenv's Python from this repository's root, so
that `reward_hacking_rl` is importable:

    <their repo>/.venv/bin/python -u -m reward_hacking_rl.train ...

The scripts in `configs/` do this. They read `THEIR_REPO`, `RESULTS` and `DOCKER_BUILD_ROOT` from the
environment. `HELD_OUT_SELECTION` is optional: unset, the training configs hold out the paper's selection,
`reward_hacking_rl/held_out_selection.json`. The trainers change into `THEIR_REPO`, and resolve every path
flag against the directory the command was run from before they do.

## Machine

- Linux with Docker, and the user able to run `docker` without a password prompt.
- `sudo` for one line of `launch.sh` (the out-of-memory exemption).
- Our runs used a 64-vCPU machine with a 500 GB disk. Reward-hacking RL builds about 400 task images,
  roughly 1 GB each for the bug-fixing and library-reimplementation tasks, and runs several hundred
  task containers at once. Grading is CPU-bound; one 50-update training took about 20 hours. No GPU
  is needed: sampling and training happen on Tinker.
- Containers run without memory limits (see "Recorded defects and fixes").

## Keys

- `TINKER_API_KEY` in the environment, read by the Tinker SDK.
- For the mixed-in runs only, `ANTHROPIC_API_KEY`, read through `src.keys` by the judge worker, which
  runs in this repository's own virtualenv (`uv sync` here). The trainer starts it itself.

## Data the environments expect

All of it comes from the external repository or from public datasets; none is shipped here.

| Stream | File or dataset | Where it comes from |
| --- | --- | --- |
| Synthetic tasks (partial and all-or-nothing reward) | `data/final-harder-merge-smaller.jsonl` | Tracked in the external repository at the commit above. |
| Library reimplementation (AE) | `data/ae.json` | Tracked in the external repository at the commit above. |
| Bug fixing (SWE-Fixer) | `data/swe-fixer.json` | Tracked in the external repository at the commit above. |
| IOI problems | `open-r1/ioi`, `open-r1/ioi-test-cases` | Downloaded from Hugging Face by their loader, with no pinned revision. |

The first two are substitutes. Their `big_run.py` reads `data/synthetic-env-hard-few-shot-no-hardcoding.json`
and `data/ae-data-15.json`, and neither file is in the repository. We used the two files above
instead. In a clone at the commit above, both are tracked files (`git ls-files data`), added by the
commits `81e34f2` ("small synthetic env") and `d083d09` ("added data/ae.json").

## The held-out task selection

`train.py` takes `--held-out-selection`, a `selection.json` written by `screen_held_out.py`
(`configs/held_out_screen.sh`): 65 tasks per stream that never train. The file our runs used cannot
be regenerated exactly:

- its two IOI lists were drawn before the script gained the filter that drops IOI problems whose
  tests all expect one output, so they were drawn from the unfiltered pool;
- candidates whose image fails to build are skipped, and which images build changes over time (16 of
  270 failed for us).

The file our runs used is `held_out_selection.json` in this folder (six streams, 65 tasks each). Pass
it as `--held-out-selection` to reproduce our training task set exactly.

## Mixed-in constitution RL or chat

`configs/reward_hacking_rl_mixed_in_constitution.sh` and `..._chat.sh` add 125 one-turn requests with
8 answers each to every update. They need:

- the prompt pools, jsonl files of `{id, question}`: the constitution questions and the No Robots chat
  requests that constitution RL trains on (built in `character_training/`);
- the constitution judge, `character_training.constitution_rl.rl_judge.judge`, and the helpfulness
  judge's `RUBRIC`, `TOOL`, `extract`, `score` and `JUDGE_MAX_TOKENS` from `anti_refusal_rl.rollout`;
- the finished reward-hacking RL run at the same model and seed: its `image_build_failures.jsonl` is
  the reference the trainer checks its own build failures against, so the coding tasks match.

## Standard coding RL control (SWE-smith)

`configs/standard_coding_rl_control.sh` lists the three steps. They need:

- a checkout of SWE-smith at commit `9b74ac08118a85c39c356802f7961893af73e07f`, for
  `control/build_pool.py` (it imports `swesmith.profiles`), and the `unidiff` package, which this
  repository's `swebench` extra installs (`uv sync --extra swebench`);
- the SWE-smith Python task dataset as parquet files, revision
  `77cab9055d42ab4a5c25c89a8f937096db13558e` (dated 2025-12-18);
- SWE-smith's task images, pulled by `control/selftest.py` with `docker pull` under the name each
  task carries (112 images for our pool).

Containers of this control run with no network.

## Changes applied from outside

These are the differences from the external repository's `big_run.py` recipe, all made by the scripts
here without editing their code:

- gpt-oss-20b, learning rate 1e-4, up to 32 tool calls per episode (theirs: 120b, 4e-5, 9).
- One training sequence per episode (`extending.ExtendingRenderer`). Their trainer otherwise makes one
  training datum per turn, each with the full context.
- The renderer's date is fixed (`--renderer-current-date 2026-08-20`); theirs is the wall clock.
- The 8 IOI problems whose stored tests all expect one output are dropped (`train.py`).
- Their one-group style environment is left out.
- Distinct training tasks are capped per stream (`MAX_TRAIN_TASKS` in `train.py`) to bound the image
  build; tasks whose image fails to build are dropped and recorded in `image_build_failures.jsonl`.
- With something mixed in, each environment module's `random` is replaced by its own seeded generator
  (`--environment-random per-module`).

## Recorded defects and fixes

Fixed, and where the fix lives:

| Defect | Fix |
| --- | --- |
| Every turn trained as its own sequence with the gpt-oss renderer. | `extending.ExtendingRenderer`; `--verify-one-datum yes` checks it before a run. |
| A reply that runs past its first `<|call|>` raised in the renderer and killed the run. | `ExtendingRenderer.parse_response` parses up to the first `<|call|>` or `<|return|>`. Runs made before this fix that crashed were resumed from their last checkpoint with `RESUME=yes`. |
| Their trainer rejects a dataset builder that is not a dataclass, and reads the length of a missing test dataset. | `extending.FixedDatasetBuilder`. |
| Checking thousands of images at once exhausted file descriptors. | `local_docker`: bounded checks and builds, and the open-file limit raised; `launch.sh` raises it too. |
| One image that no longer builds stopped the run. | `train.py` and `screen_held_out.py` build tolerantly and drop that image's tasks. |
| Docker refuses some container starts when hundreds start at once. | `local_docker`: `docker run` retried (`START_ATTEMPTS`). |
| Runaway task processes filled memory and the kernel killed the trainer. | `launch.sh` exempts the trainer from the out-of-memory killer. Containers still have no memory limit. |
| A started run silently resumed an earlier one in the same folder. | `launch.sh` refuses a folder that holds a checkpoint unless `RESUME=yes`. |
| IOI problems graded by an interactor store the placeholder output `1` for every test, so `print(1)` passes. | Dropped in `train.py` and `screen_held_out.py`. |
| Extra sampling requests of the mixed-in stream would shift the coding environments' random draws. | `--environment-random per-module` in `train.py`. |
| A different set of failed image builds would change the coding tasks of a mixed-in run. | `--reference-failures` in `train.py` stops the run on a difference. |
| In the SWE-smith control, edited tests or test configuration could pass the grader. | `control/swe_smith_env.py` restores the graded tests and the test configuration before grading and records tampering; `control/selftest.py` checks every task. |

Known and left as they were in the reported runs:

- Held-out exclusion is per stream. The two synthetic streams share one task pool and so do the two
  IOI streams, so a task held out of one stream can train in its sibling stream.
- The held-out IOI lists contain problems the single-output filter would drop (see above).
- Bug fixing (SWE-Fixer): when the test patch adds a file and the model has created a file at that
  path, their grader's `git checkout` and `git apply` fail, and pytest runs the model's own test.
- IOI: each subtask is graded on every test of its problem, so a correct subtask solution can fail.
- Synthetic tasks: some ship the file the model is told to write as a protected stub.
- `train/metrics.jsonl` leaves out groups whose rewards are all equal; `rows.jsonl` is the full record
  of episodes.
- The first training of each model (seed 0) did not seed Python's global random, so its public-test
  and prompt draws cannot be reproduced. Tinker's sampling is never seeded.
