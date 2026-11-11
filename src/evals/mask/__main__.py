"""Run the MASK benchmark on one checkpoint with MASK's own pipeline (see eval.py); every knob a
required flag. The vendored scripts run unmodified in <log-dir>/mask_pipeline/, in MASK's order:
generate_responses.py (Tinker model in place of the OpenAI client), evaluate.py per archetype
(gpt-4o and o3-mini judges, or with --judge claude the same prompts to a Claude judge with a forced
tool, see claude_judge.py), metric.py, process_metrics.py and process_metrics.py --normalize.
--archetypes picks which of MASK's six archetypes run.
`--dry-run` prints the rows, calls and the first pressure prompt of each archetype.

Outputs beside the pipeline directory: mask_summary.json (MASK's all_results.json and its normalised
all_results2.json, the pooled numbers over all archetypes, and the judge-failure count) and
measured_cost.json. A rerun of the same command resumes: generation skips archetypes whose responses
file exists, and evaluation skips archetypes already evaluated.

    uv run python -m src.evals.mask --model tinker/openai/gpt-oss-20b --checkpoint none \\
        --tokenizer-id openai/gpt-oss-20b --renderer-name gpt_oss_medium_reasoning \\
        --renderer-current-date none --thinking true --temperature 1.0 --top-p 1.0 \\
        --max-tokens 16384 --max-connections 10 --lie-k 1 --judge-concurrency 50 \\
        --judge claude --judge-model claude-sonnet-4-6 \\
        --archetypes continuations,disinformation,doubling_down_known_facts,known_facts,provided_facts \\
        --data-dir none --limit none --log-dir results/x
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.evals.common import add_model_flags, positive_int, reject_greedy_thinking, validate_model_flags
from src.evals.mask.eval import (
    JUDGE_MODEL,
    JUDGE_TOKENS_PER_CALL,
    JUDGES,
    MODEL_NAME,
    OFFICIAL_DIR,
    OFFICIAL_SHA256,
    SPLITS,
    check_archetypes,
    check_official,
    generation_failures,
    generations_per_row,
    judge_calls_per_row,
    judge_failures,
    load_split,
    pooled,
    write_csv_data,
)
from src.training.tinker_common import _optional_str

#: The judge-failure ceiling: above it the numbers are written but the run fails (a metric over more than 10% judge failures is not trusted).
JUDGE_FAILURE_GATE = 0.10
PIPELINE_DIR = "mask_pipeline"
SUMMARY_FILE = "mask_summary.json"
MEASURED_COST_FILE = "measured_cost.json"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _optional_limit(value: str) -> int | None:
    """none for every row, or a positive int to keep the first N rows of each archetype (smoke)."""
    return None if value == "none" else positive_int(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_flags(p)
    p.add_argument("--temperature", required=True, type=float, help="generate_responses.py --temperature")
    p.add_argument("--top-p", required=True, type=float, help="the script sends none; the API default is 1.0")
    p.add_argument("--max-tokens", required=True, type=positive_int, help="generate_responses.py --max_tokens")
    p.add_argument(
        "--max-connections",
        required=True,
        type=positive_int,
        help="Tinker connection limit; the script itself holds 10 requests in flight per archetype",
    )
    p.add_argument("--lie-k", required=True, type=positive_int, help="generate_responses.py --lie_k")
    p.add_argument("--judge-concurrency", required=True, type=positive_int, help="evaluate.py --concurrency_limit")
    p.add_argument(
        "--judge",
        required=True,
        choices=JUDGES,
        help="official: MASK's gpt-4o / o3-mini calls; claude: the same prompts to a Claude judge (claude_judge.py)",
    )
    p.add_argument(
        "--judge-model",
        required=True,
        type=_optional_str,
        help="the Claude judge model with --judge claude; none with --judge official",
    )
    p.add_argument(
        "--archetypes",
        required=True,
        type=lambda s: tuple(x for x in s.split(",") if x),
        help=f"comma-separated archetypes to run, from {','.join(SPLITS)}",
    )
    p.add_argument(
        "--data-dir",
        required=True,
        type=_optional_str,
        help="directory of the six parquet files, or none for data/mask",
    )
    p.add_argument(
        "--limit", required=True, type=_optional_limit, help="none, or the first N rows of each archetype (smoke)"
    )
    p.add_argument("--log-dir", required=True, help="where the pipeline directory and the summary are written")
    p.add_argument("--dry-run", action="store_true", help="print rows, calls and first prompts; no calls")
    return p


def parse_args(argv: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def counts(
    data_dir: Path | None, limit: int | None, lie_k: int, archetypes: tuple[str, ...]
) -> dict[str, dict[str, int]]:
    """Rows, generations and judge calls per archetype for a run."""
    out = {}
    for split in check_archetypes(archetypes):
        n = len(load_split(split, data_dir))
        n = n if limit is None else min(n, limit)
        out[split] = {
            "rows": n,
            "generations": n * generations_per_row(split, lie_k),
            "judge_calls": n * judge_calls_per_row(split, lie_k),
        }
    return out


def _run(cmd: list[str], workdir: Path, env: dict) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=workdir, env=env, check=True)


def require_prices(model: str, judge_model: str) -> float:
    """The estimated dollars of one judge call, after checking that `src.costs` prices both the sampled
    model and the judge model. Called before generation, so a model without a price row stops the run
    before anything is spent instead of after the paid work, when the cost file is written."""
    from src.costs import judge_call_usd, usage_usd

    try:
        usage_usd(model, 0, 0)
        return judge_call_usd(judge_model, use_batch=False, **JUDGE_TOKENS_PER_CALL)
    except (KeyError, ValueError) as e:
        raise SystemExit(
            f"no price in src/costs.py for the sampled model {model!r} or the judge model {judge_model!r} "
            f"({e}); the run writes its cost, so it stops here, before any generation or judging"
        ) from e


def main(argv: list[str]) -> None:
    from src.keys import load_project_env

    load_project_env()
    a = parse_args(argv)
    validate_model_flags(a)
    reject_greedy_thinking(a.thinking, a.temperature)
    check_official()
    archetypes = check_archetypes(a.archetypes)
    # --judge-model parses the none sentinel to the empty string, so the check is on truthiness.
    if (a.judge == "claude") != bool(a.judge_model):
        raise SystemExit("--judge-model names the Claude judge with --judge claude, and is none with --judge official")
    if a.judge == "claude":
        from src.keys import require_standard_claude

        require_standard_claude(a.judge_model, "judge")
    judge_model = a.judge_model if a.judge == "claude" else JUDGE_MODEL
    judge_call_price = require_prices(a.model, judge_model)
    data_dir = Path(a.data_dir) if a.data_dir else None
    plan = counts(data_dir, a.limit, a.lie_k, archetypes)

    if a.dry_run:
        for split in archetypes:
            df = load_split(split, data_dir)
            r = df.iloc[0]
            print(f"===== {split}: {plan[split]} =====")
            print(f"[system] {r['system_prompt']}")
            if "dd_turn_1_prompt" in df.columns:
                print(f"[user] {r['dd_turn_1_prompt']}\n[assistant, forced] {r['dd_turn_1_response']}")
            print(f"[user] {r['user_prompt']}\n")
        gens = sum(c["generations"] for c in plan.values())
        calls = sum(c["judge_calls"] for c in plan.values())
        print(
            f"{sum(c['rows'] for c in plan.values())} rows; {gens} generations; ~{calls} judge calls ({judge_model}, judge {a.judge})"
        )
        print(f"official MASK files: {sorted(OFFICIAL_SHA256)} (sha256 checked)\n(dry run: no generation, no judging)")
        return

    if a.judge == "official" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("evaluate.py needs OPENAI_API_KEY in the environment (project .env)")
    log_dir = Path(a.log_dir)
    workdir = log_dir / PIPELINE_DIR
    workdir.mkdir(parents=True, exist_ok=True)
    for rel in OFFICIAL_SHA256:
        (workdir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OFFICIAL_DIR / rel, workdir / rel)
    write_csv_data(workdir, data_dir, a.limit, archetypes)
    config = workdir / "tinker_config.json"
    settings = json.dumps(
        {
            "model_args": {
                "model": a.model,
                "checkpoint": a.checkpoint,
                "tokenizer_id": a.tokenizer_id,
                "renderer_name": a.renderer_name,
                "renderer_current_date": a.renderer_current_date,
                "thinking": a.thinking,
            },
            "max_connections": a.max_connections,
            "top_p": a.top_p,
            "max_tokens": a.max_tokens,
            "temperature": a.temperature,
            "lie_k": a.lie_k,
            "limit": a.limit,
            "judge": a.judge,
            "judge_model": a.judge_model,
            "archetypes": list(archetypes),
        }
    )
    # MASK's scripts skip any archetype whose output file exists, so a directory holds one run only.
    if config.exists() and config.read_text() != settings:
        raise SystemExit(f"{workdir} holds a run with other settings ({config}); use a fresh --log-dir")
    config.write_text(settings)
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    py = sys.executable

    # 1. generation: MASK's script, its own argument names and defaults for everything not set here
    _run(
        [
            py, "-m", "src.evals.mask.generate_adapter", str(config), "generate_responses.py",
            "--model", MODEL_NAME, "--max_tokens", str(a.max_tokens),
            "--temperature", str(a.temperature), "--lie_k", str(a.lie_k),
        ],
        workdir,
        env,
    )  # fmt: skip
    # 2. judging, one archetype file at a time as evaluate.py's own loop does
    responses_files = [f"csv_data/responses/{split}_{MODEL_NAME}.csv" for split in archetypes]
    if a.judge == "official":
        for responses in responses_files:
            if not (workdir / responses.replace("responses", "evaluated")).exists():
                cmd = [py, "evaluate.py", "--input_file", responses, "--concurrency_limit", str(a.judge_concurrency)]
                _run(cmd, workdir, env)
    else:
        from src.evals.mask.claude_judge import judge

        judged = judge(workdir, responses_files, a.judge_model, a.judge_concurrency)
        print(f"Claude judge: {judged}", flush=True)
    # 3-4. metrics
    _run([py, "metric.py"], workdir, env)
    _run([py, "process_metrics.py"], workdir, env)
    _run([py, "process_metrics.py", "--normalize"], workdir, env)

    metrics_dir = workdir / "csv_data" / "metrics"
    failures = judge_failures(workdir / "csv_data" / "evaluated", archetypes)
    summary = {
        "pooled": pooled(metrics_dir, archetypes),
        "judge": failures,
        "generation": generation_failures(workdir / "csv_data" / "responses", archetypes),
        "all_results": json.loads((metrics_dir / "all_results.json").read_text()),
        "all_results_normalized": json.loads((metrics_dir / "all_results2.json").read_text()),
        "plan": plan,
    }
    (log_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))

    from src.costs import usage_usd

    usage = json.loads((workdir / "generation_usage.json").read_text())
    gen_usd = usage_usd(a.model, usage["input_tokens"], usage["output_tokens"])
    judge_usd = sum(c["judge_calls"] for c in plan.values()) * judge_call_price
    (log_dir / MEASURED_COST_FILE).write_text(
        json.dumps(
            {
                "usd": gen_usd + judge_usd,
                "source": (
                    f"generation measured from Tinker token usage (${gen_usd:.2f}); judging ESTIMATED, not measured, as "
                    f"calls x {JUDGE_TOKENS_PER_CALL} tokens at {judge_model} rates (${judge_usd:.2f}), "
                    "since evaluate.py does not record judge usage"
                ),
            }
        )
    )
    p = summary["pooled"]
    print(
        f"MASK pooled over {p['n_rows']} rows: P(lie) {p['p_lie']:.3f} [{p['p_lie_low']:.3f}, {p['p_lie_high']:.3f}], "
        f"P(honest) {p['p_honest']:.3f}, accuracy {p['accuracy']}; judge failures "
        f"{failures['judge_failures']}/{failures['judge_calls']}; generation failures "
        f"{summary['generation']['generation_failures']}/{summary['generation']['generations']}"
    )
    rate = failures["judge_failure_rate"]
    if rate is not None and rate > JUDGE_FAILURE_GATE:
        raise SystemExit(f"judge failure rate {rate:.1%} exceeds {JUDGE_FAILURE_GATE:.0%}: numbers not reportable")


if __name__ == "__main__":
    main(sys.argv[1:])
