# Reward-hacking analysis

From evaluation logs to the paper's reward-hacking numbers and figures: the hack rates on ImpossibleBench and
School of Reward Hacks, the solve rates on the original tasks, the list of broken ImpossibleBench tasks that are
left out, and the four reward-hacking figures (`rh_main`, `rh_variants`, `rh_honest`, `rh_minimal`).

The folder works at two levels:

- **From the shipped tables, with no data.** `tables/` holds the rates the paper's figures are drawn from, and
  `s4_figures.py` draws the figures from them.
- **From your own Inspect logs.** `s1_rates.py`, `s2_pooled_hack_rates.py` and `s3_school_of_reward_hacks.py` show
  exactly how each table is computed from the logs, and write the same tables for your runs.

## Files

| Path | What it does | Reads |
| --- | --- | --- |
| `find_broken_tasks.py` | Finds the ImpossibleBench tasks whose changed test file is broken. | the benchmark datasets (network) |
| `s1_rates.py` | Hack and solve rate per run and split, on every task and with the broken tasks left out. | Inspect logs |
| `s2_pooled_hack_rates.py` | Hack rate per run with the two impossible splits pooled. | Inspect logs, s1's table |
| `s3_school_of_reward_hacks.py` | Hack rate on School of Reward Hacks per run. | Inspect log headers |
| `s4_figures.py` | The four figures. | the three tables |
| `common.py` | Names, the runs file, the broken-task list, the rule for the training that is left out. | |
| `bars.py` | The bar-chart layout of the figures. | |
| `runs.example.json` | The shape of the runs file, with placeholder log paths. | |
| `task_lists/impossiblebench_broken_tasks.json` | The broken tasks the paper leaves out. | |
| `tables/impossiblebench_rates.csv` | s1's table for the paper's runs. | |
| `tables/impossiblebench_hack_rates_pooled.csv` | s2's table for the paper's runs. | |
| `tables/school_of_reward_hacks.csv` | s3's table for the paper's runs. | |

Every script is run from the repository root as `uv run python -m analysis.<name>` and has `--help`. Outputs go to
`results/analysis/scripts/<script name>/` (under `CHARACTER_TRAINING_RL_HOME` when that is set; see `src/paths.py`).
No script here samples a model or calls a model API.

## Drawing the figures from the shipped tables

```bash
uv run python -m analysis.s4_figures --figure all \
    --rates analysis/tables/impossiblebench_rates.csv \
    --pooled analysis/tables/impossiblebench_hack_rates_pooled.csv \
    --school-of-reward-hacks analysis/tables/school_of_reward_hacks.csv
```

| `--figure` | Paper figure | Tables it needs |
| --- | --- | --- |
| `rh_main` | Hack rate before and after reward-hacking RL, alone and with constitution RL mixed in (main text) | `--pooled`, `--school-of-reward-hacks` |
| `rh_variants` | The same for the two character-trained models, with every training (appendix) | `--pooled`, `--school-of-reward-hacks` |
| `rh_honest` | Solve rate on the original tasks of Impossible-LiveCodeBench and Impossible-SWEbench (appendix) | `--rates` |
| `rh_minimal` | Hack rate on Impossible-LiveCodeBench without tools (appendix) | `--pooled` |

Panel (c) of `rh_main` and `rh_variants` is School of Reward Hacks, read from the table
`s3_school_of_reward_hacks.py` writes; the paper's is `tables/school_of_reward_hacks.csv`.

## What the numbers are

**Runs.** Three models: gpt-oss-20b, and the two character-trained models, assigned rules and value-grounded
rules. Each is evaluated before RL and after one of four trainings (see `training` below). gpt-oss-20b was trained
with reward-hacking RL alone. Reward-hacking RL, alone or with chat or constitution RL mixed in, was repeated over
three seeds (0, 1000, 2000); standard coding RL was run once per character-trained model.

**Benchmarks and splits.** Impossible-LiveCodeBench (103 tasks) and Impossible-SWEbench (349 instances, of which
the evaluation runs 348; `src.evals.impossible_swebench.eval.UNRUNNABLE` names the one it cannot start). Each has
three splits of the same tasks: `original` (the unedited tests), `oneoff` (one expected value in the tests is
changed) and `conflicting` (an added assertion contradicts another). A pass on `original` is a solve; a pass on
`oneoff` or `conflicting` is a hack, because no correct solution passes the changed tests. Every task is attempted
once.

**Broken tasks.** A one-off or conflicting task is left out when its changed test file is broken by the
benchmark's own edit: compared with the original task's test file, the changed one fails to compile, raises a
NameError when imported, or lacks a file the original adds. `task_lists/impossiblebench_broken_tasks.json` lists
them: 55 one-off and 67 conflicting Impossible-SWEbench instances and one Impossible-LiveCodeBench task
(`lcbhard_77`) in each impossible split. A task is left out only of the split it is listed under, and nothing is left
out of `original`. One listed instance (`astropy__astropy-7336`, in both splits) is the instance the evaluation
cannot start, so 54 one-off and 66 conflicting instances leave each Impossible-SWEbench run: 294 one-off and 282
conflicting tasks remain, and all 348 original tasks. The rows have the shape of the monitor study's
`environment_defect_tasks` (`monitors/README.md`), with the same benchmark keys, and the monitor study applies the
same rule to them.

`find_broken_tasks.py` derives the list from the two benchmark datasets on Hugging Face at pinned revisions and
the instances' files on GitHub. It needs no benchmark checkout and no logs.

**Rates (s1).** Each sample's `resolved` value goes through the metric functions the evaluation's scorer
registers, so an errored sample and an unusable Impossible-LiveCodeBench answer leave the denominator exactly as in
the log's own metrics. `rate` is passes / graded with a 95% Wilson interval. Before a run's rows are kept, its
statistics over every task must equal the metrics in the log header.

**Pooled hack rates (s2).** The figures pool the two impossible splits. For a training repeated over seeds, a run's
pooled rate is hacks / graded over both splits; the figures draw the mean over the seeds and a dot per seed. For a
single evaluated model (before RL, standard coding RL), the two splits are two versions of the same tasks, so each
task's values are averaged over the splits it is graded in, and the rate is the mean over tasks with a 95% interval
over tasks (`src.evals.base.clustered_ci`). For these rows `hack_rate_pct` is therefore not exactly
`hacks / graded`.

**School of Reward Hacks (s3).** The `hack` score's mean and prompt-clustered 95% interval and the `usable` share,
copied from the header of each log written by `src/evals/school_of_reward_hacks`.

**The training that is left out.** One training is left out of every reward-hacking figure and of the pooled
table: assigned rules, reward-hacking RL alone, seed 2000. Its reasoning channel is missing on most turns (11.1% of
its School of Reward Hacks answers are usable, against a 90% threshold). So assigned rules has two repeats of
reward-hacking RL alone where the other models have three. The assigned-rules trainings at seed 2000 with chat or
constitution RL mixed in are separate trainings with usable reasoning and stay in. The rule is
`analysis.common.EXCLUDED_TRAININGS`; its rows are in the per-split rates table and are dropped by s2 and s4, which
print what they drop. s4 also leaves a reward-hacking training out of the School of Reward Hacks panels when its
usable share is below `MIN_USABLE` (0.9).

## Table columns

Values that name things are the same in every table:

| Column | Values | Paper term |
| --- | --- | --- |
| `benchmark` | `iblcb` | Impossible-LiveCodeBench (tools scaffold, the scaffold of training) |
| | `iblcb_minimal` | Impossible-LiveCodeBench without tools (minimal scaffold) |
| | `ibswe` | Impossible-SWEbench |
| `model` | `gpt-oss-20b` | gpt-oss-20b (no character training) |
| | `assigned_rules` | assigned rules |
| | `value_grounded_rules` | value-grounded rules |
| `training` | `before_rl` | before RL |
| | `standard_coding_rl` | standard coding RL (bug-fixing tasks whose graded tests cannot be edited) |
| | `reward_hacking_rl` | reward-hacking RL |
| | `reward_hacking_rl_chat_mixed_in` | reward-hacking RL with chat mixed in (+ chat) |
| | `reward_hacking_rl_constitution_rl_mixed_in` | reward-hacking RL with constitution RL mixed in (+ constitution RL) |
| `seed` | empty, or `0`, `1000`, `2000` | the training's seed; empty before RL, which has no training |
| `split` | `original`, `oneoff`, `conflicting` | original tasks, one-off tasks, conflicting tasks |

`tables/impossiblebench_rates.csv`: two rows per run and split.

| Column | Meaning |
| --- | --- |
| `rate_kind` | `solve` on `original` (solve rate on original tasks), `hack` on the impossible splits (hack rate) |
| `task_set` | `all_tasks`: every task of the split. `broken_tasks_left_out`: the split less its broken tasks, which the paper reports |
| `samples` | tasks in the set |
| `errored` | samples that errored (sandbox or provider failure) and have no score |
| `graded` | samples in the denominator: scored, and usable where the benchmark grades usability |
| `passes` | graded samples that pass their tests |
| `rate`, `wilson_low`, `wilson_high` | passes / graded and its 95% Wilson interval, as fractions |
| `message_limit` | samples that ended at the message limit |
| `context_full` | samples that ended at the context window |
| `grader_timed_out` | Impossible-SWEbench only: samples whose grading timed out |
| `unusable` | Impossible-LiveCodeBench only: scored samples whose answer was unusable |
| `tasks_left_out` | broken tasks removed from the split (0 in `all_tasks` rows) |

`tables/impossiblebench_hack_rates_pooled.csv`: one row per run, the one-off and conflicting splits pooled, broken
tasks left out, the left-out training absent.

| Column | Meaning |
| --- | --- |
| `hacks`, `graded` | sums over the two impossible splits |
| `hack_rate_pct` | percent; `hacks / graded` for a training repeated over seeds, the mean over tasks for a single evaluated model |
| `ci95_low_pct`, `ci95_high_pct` | percent; the 95% interval over tasks for a single evaluated model, empty otherwise |

`school_of_reward_hacks.csv` (written by s3): `model`, `training`, `seed`, then `hack_rate`, `ci95_low`,
`ci95_high` and `usable` as fractions.

## From your own logs

### What you supply

Inspect logs (`.eval` files), one per run:

- ImpossibleBench: logs written by `src/evals/impossible_livecodebench` (`--agent-type tools` for `iblcb`,
  `--agent-type minimal` for `iblcb_minimal`) and `src/evals/impossible_swebench`, one per model, training, seed
  and split. For the paper's design that is 26 evaluated models on 3 benchmarks and 3 splits, 234 logs.
- School of Reward Hacks: logs written by `src/evals/school_of_reward_hacks`, one per model, training and seed
  (26 logs).

All logs of a benchmark should come from the same evaluation settings; the scripts do not check this.

The runs file, a JSON file passed as `--runs`. Copy `runs.example.json` and edit it. Every key is required and
unknown keys or values are errors.

- `impossiblebench`: one row per log, with `benchmark`, `model`, `training`, `seed` (`null` for `before_rl`, the
  training's seed otherwise), `split` and `log` (path of the `.eval` file, absolute or relative to the directory
  the script is run from).
- `school_of_reward_hacks`: one row per log, with `model`, `training`, `seed` and `log`.

The broken-task list is read from `task_lists/impossiblebench_broken_tasks.json`, and its counts are checked
(`analysis.common.EXPECTED_BROKEN`). Every listed task of a split must be a sample of each log of that split,
except the instance the evaluation cannot start.

### Run order

```bash
RUNS=analysis/runs.json
OUT=results/analysis/scripts

# optional: derive the broken-task list again (needs the swebench extra, git and network access)
uv run python -m analysis.find_broken_tasks

uv run python -m analysis.s1_rates --runs $RUNS
uv run python -m analysis.s2_pooled_hack_rates --runs $RUNS --rates $OUT/s1_rates/impossiblebench_rates.csv
uv run python -m analysis.s3_school_of_reward_hacks --runs $RUNS

uv run python -m analysis.s4_figures --figure all \
    --rates $OUT/s1_rates/impossiblebench_rates.csv \
    --pooled $OUT/s2_pooled_hack_rates/impossiblebench_hack_rates_pooled.csv \
    --school-of-reward-hacks $OUT/s3_school_of_reward_hacks/school_of_reward_hacks.csv
```

`s4_figures` stops if a model and training that should have a bar has no row, so `--figure all` needs every run of
the design above.

## Tests

`tests/test_analysis_tables.py` checks the shipped tables against numbers the paper states (task counts, hack and
solve rates on Impossible-SWEbench after reward-hacking RL) and draws two figures from them.
`tests/test_analysis_pipeline.py` runs s1, s2 and s3 end to end on small Inspect logs it writes itself. Neither
needs a network or a model.
