"""Character-training scripts: review tables, dry runs, corpus and mix writes, rollout verdict keys."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pytest

from character_training.constitution_rl import rl_judge, rl_rollout
from character_training.distillation import s18_build_mix
from character_training.documents import arrays_common, s1_scaffold, s2_generate_documents, s3_judge_documents
from src.costs import TINKER_TRAIN_USD_PER_MTOK

ARM = "bare_rules"


def scaffold_rows(ids: list[str]) -> list[dict]:
    return [
        {"assertion_id": i, "assertion": f"assertion {i}", "explanation": f"explanation {i}", "name": "n", "label": "l"}
        for i in ids
    ]


def paper_ids() -> list[str]:
    tables = (s1_scaffold.ASSERTION_GROUNDING_EXCLUSIONS[ARM], s1_scaffold.ASSERTION_GROUNDING_OVERRIDES[ARM])
    return sorted({i for table in tables for i in table} | {"d009.s001.a001"})


def test_review_tables_none_keeps_every_assertion_unchanged():
    rows = scaffold_rows(paper_ids())
    reviewed, excluded, overrides = s1_scaffold.apply_review_tables(
        rows, ARM, "none", False, s1_scaffold.scaffold_fingerprint(rows)
    )
    assert reviewed == rows and excluded == [] and overrides == []


def test_review_tables_paper_refuses_while_no_fingerprint_is_recorded(monkeypatch):
    monkeypatch.setitem(s1_scaffold.REVIEW_TABLE_FINGERPRINTS, ARM, None)
    rows = scaffold_rows(paper_ids())
    with pytest.raises(RuntimeError, match="no fingerprint is recorded"):
        s1_scaffold.apply_review_tables(rows, ARM, "paper", False, s1_scaffold.scaffold_fingerprint(rows))


def test_review_tables_paper_applies_tables_to_the_matching_scaffold(monkeypatch):
    rows = scaffold_rows(paper_ids())
    fingerprint = s1_scaffold.scaffold_fingerprint(rows)
    exclusions = s1_scaffold.ASSERTION_GROUNDING_EXCLUSIONS[ARM]
    table_overrides = s1_scaffold.ASSERTION_GROUNDING_OVERRIDES[ARM]

    def check(accept_unverified: bool) -> None:
        reviewed, excluded, overrides = s1_scaffold.apply_review_tables(
            rows, ARM, "paper", accept_unverified, fingerprint
        )
        assert {r["assertion_id"] for r in excluded} == set(exclusions)
        assert [r["assertion_id"] for r in reviewed] == [i for i in paper_ids() if i not in exclusions]
        by_id = {r["assertion_id"]: r for r in reviewed}
        for assertion_id, override in table_overrides.items():
            assert by_id[assertion_id]["explanation"] == override["explanation"]
        assert by_id["d009.s001.a001"]["explanation"] == "explanation d009.s001.a001"
        assert [o["assertion_id"] for o in overrides] == list(table_overrides)

    monkeypatch.setitem(s1_scaffold.REVIEW_TABLE_FINGERPRINTS, ARM, None)
    check(accept_unverified=True)
    monkeypatch.setitem(s1_scaffold.REVIEW_TABLE_FINGERPRINTS, ARM, fingerprint)
    check(accept_unverified=False)


def test_review_tables_paper_stops_on_another_scaffold(monkeypatch):
    rows = scaffold_rows(paper_ids())
    recorded = s1_scaffold.scaffold_fingerprint(rows)
    monkeypatch.setitem(s1_scaffold.REVIEW_TABLE_FINGERPRINTS, ARM, recorded)
    # Same IDs, other text: the positional tables would land on unrelated assertions.
    other = [{**row, "assertion": row["assertion"] + " (regenerated)"} for row in rows]
    with pytest.raises(RuntimeError, match="not the scaffold the review tables were written for"):
        s1_scaffold.apply_review_tables(other, ARM, "paper", True, s1_scaffold.scaffold_fingerprint(other))
    # A scaffold lacking a table ID stops whatever the fingerprint.
    with pytest.raises(RuntimeError, match="assertion IDs it does not have"):
        s1_scaffold.apply_review_tables(rows[1:], ARM, "paper", True, recorded)


def scaffold_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = {
        "arm": ARM,
        "principle_name": "p",
        "model": s1_scaffold.FROZEN_MODEL,
        "temperature": 1.0,
        "max_output_tokens": 8192,
        "force": False,
        "dry_run": False,
        "review_tables": "none",
        "accept_unverified_review_tables": False,
        "output_dir": tmp_path / "scaffold",
    }
    return argparse.Namespace(**{**values, **overrides})


def cached_assertion_calls(args: argparse.Namespace, subdomains: list[dict], items: list[dict]) -> None:
    """Write the subdomains and a raw cache holding every assertion call, so the stage makes no API call."""
    s1_scaffold.write_jsonl(args.output_dir / "subdomains.jsonl", subdomains)
    raw = []
    for subdomain in subdomains:
        prompt = s1_scaffold.PROMPT_SPEC2ASSERTIONS.format(
            principle_name=args.principle_name,
            spec_section=subdomain["spec_section"],
            domain=subdomain["domain"],
            subdomain=subdomain["subdomain"],
        )
        key = s1_scaffold.cache_key(
            args.model, args.temperature, args.max_output_tokens, prompt, s1_scaffold.ASSERTION_SCHEMA
        )
        raw.append({"id": subdomain["subdomain_id"], "input": subdomain, "cache_key": key, "items": items})
    s1_scaffold.write_jsonl(args.output_dir / "raw" / "assertions_calls.jsonl", raw)


SUBDOMAIN = {
    "arm": ARM,
    "domain_id": "d001",
    "domain": "Domain",
    "subdomain_id": "d001.s001",
    "subdomain": "Subdomain",
    "subdomain_context": "context",
    "spec_section": "section",
}
ITEMS = [{"assertion": f"claim {n}", "name": "n", "explanation": f"why {n}", "label": "behavior"} for n in range(3)]


def test_assertions_stage_writes_the_fingerprint_and_applies_nothing_under_none(tmp_path):
    args = scaffold_args(tmp_path)
    cached_assertion_calls(args, [SUBDOMAIN], ITEMS)
    s1_scaffold.stage_assertions(args)
    assertions = s1_scaffold.read_jsonl(args.output_dir / "assertions.jsonl")
    assert [(a["assertion_id"], a["explanation"]) for a in assertions] == [
        ("d001.s001.a001", "why 0"),
        ("d001.s001.a002", "why 1"),
        ("d001.s001.a003", "why 2"),
    ]
    record = json.loads((args.output_dir / s1_scaffold.REVIEW_FINGERPRINT_FILE).read_text())
    assert record == {
        "arm": ARM,
        "n_assertions": 3,
        "assertions_sha256": s1_scaffold.scaffold_fingerprint(assertions),
    }
    assert s1_scaffold.read_jsonl(args.output_dir / "assertion_grounding_exclusions.jsonl") == []


def test_assertions_stage_paper_writes_no_assertions_for_another_scaffold(tmp_path, monkeypatch):
    monkeypatch.setitem(s1_scaffold.REVIEW_TABLE_FINGERPRINTS, ARM, None)
    args = scaffold_args(tmp_path, review_tables="paper")
    cached_assertion_calls(args, [SUBDOMAIN], ITEMS)
    with pytest.raises(RuntimeError, match="no fingerprint is recorded"):
        s1_scaffold.stage_assertions(args)
    args.accept_unverified_review_tables = True
    with pytest.raises(RuntimeError, match="assertion IDs it does not have"):
        s1_scaffold.stage_assertions(args)
    assert not (args.output_dir / "assertions.jsonl").exists()


def test_scaffold_dry_run_creates_nothing(tmp_path, capsys):
    args = scaffold_args(tmp_path, dry_run=True)
    job = {"id": "x", "input": {}, "prompt": "the resolved prompt"}
    rows = asyncio.run(s1_scaffold.run_jobs(args, "domains", [job], s1_scaffold.DOMAIN_SCHEMA, ("domain",), None))
    assert rows == [] and not args.output_dir.exists()
    assert "the resolved prompt" in capsys.readouterr().out


def test_scaffold_requires_the_review_tables_flag(monkeypatch, capsys):
    argv = "s1 --stage status --constitution bare_rules --principle-name p --model claude-opus-4-6 --temperature 1.0"
    argv += " --concurrency 1 --target-raw-docs 1 --n-doc-ideas 25 --max-output-tokens 8192 --force false"
    argv += " --dry-run true --accept-unverified-review-tables false"
    monkeypatch.setattr(sys, "argv", argv.split())
    with pytest.raises(SystemExit):
        s1_scaffold.parse_args()
    assert "--review-tables" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", [*argv.split(), "--review-tables", "none"])
    assert s1_scaffold.parse_args().review_tables == "none"


def manifest_row(arm: str, n: int) -> dict:
    return {
        "manifest_id": f"{arm}::i{n}",
        "arm": arm,
        "doc_type": "internal memo",
        "domain": "Domain",
        "subdomain": "Subdomain",
        "assertion": "claim",
        "assertion_name": "n",
        "assertion_explanation": "why",
        "assertion_label": "behavior",
        "idea": f"idea number {n}",
    }


def test_document_submit_dry_run_writes_nothing_and_prints_a_prompt(tmp_path, capsys):
    manifest = tmp_path / "manifest.jsonl"
    rows = [manifest_row(ARM, n) for n in range(3)]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    args = argparse.Namespace(
        constitutions=[ARM],
        manifests=[manifest],
        model=s2_generate_documents.FROZEN_MODEL,
        temperature=1.0,
        max_output_tokens=8000,
        batch_size=5000,
        dry_run=True,
        output_dir=tmp_path / "out",
    )
    s2_generate_documents.submit(args, s2_generate_documents.validate_manifests(args))
    out = capsys.readouterr().out
    assert not args.output_dir.exists()
    spec = s2_generate_documents.resolved_spec(s2_generate_documents.constitution_path(ARM))
    assert s2_generate_documents.document_prompt(rows[0], spec) in out
    assert "would submit 1 batches" in out and "cost bound $" in out and "no API calls or output writes" in out


def test_judge_dry_run_samples_are_per_constitution_and_reading_stops_early():
    read = []

    def rows():
        for n, arm in enumerate(["a", "a", "a", "b", "a", "b", "b", "a"]):
            read.append(n)
            yield {"arm": arm, "n": n}

    picked = s3_judge_documents.dry_run_samples(rows(), ["a", "b"], 2)
    assert [(r["arm"], r["n"]) for r in picked] == [("a", 0), ("a", 1), ("b", 3), ("b", 5)]
    assert read == [0, 1, 2, 3, 4, 5]
    one = s3_judge_documents.dry_run_samples(iter([{"arm": "a"}] * 5), ["a"], 2)
    assert len(one) == 2
    short = s3_judge_documents.dry_run_samples(iter([{"arm": "a"}, {"arm": "a"}, {"arm": "b"}]), ["a", "b"], 2)
    assert [r["arm"] for r in short] == ["a", "a", "b"]


class WordTokenizer:
    def __call__(self, texts, add_special_tokens, truncation):
        return {"input_ids": [list(range(len(text.split()))) for text in texts]}


def build_corpus(tmp_path: Path, target_tokens: int) -> Path:
    from character_training.documents import s4_build_corpora

    documents = tmp_path / "documents.jsonl"
    rows = [{"manifest_id": f"m{n}", "complete": True, "content": f"document {n} has five words"} for n in range(4)]
    documents.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ids = [r["manifest_id"] for r in rows]
    offsets = s4_build_corpora.index_documents(documents, set(ids))
    output = tmp_path / "corpus.jsonl"
    s4_build_corpora.build_arm(
        arm=ARM,
        ranked_ids=ids,
        offsets=offsets,
        documents_path=documents,
        output_path=output,
        tokenizer=WordTokenizer(),
        terminal_token_id=0,
        target_tokens=target_tokens,
        batch_size=2,
        shuffle_seed=42,
    )
    return output


def test_corpus_short_of_target_raises_and_writes_nothing(tmp_path):
    pytest.importorskip("transformers")
    # Four documents of 5 content tokens and 1 terminal token each hold 24 training tokens.
    with pytest.raises(RuntimeError, match="1 short of --target-tokens 25"):
        build_corpus(tmp_path, 25)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["documents.jsonl"]
    output = build_corpus(tmp_path, 24)
    assert len(output.read_text().splitlines()) == 4


def test_mix_is_not_written_when_rendering_fails(tmp_path, monkeypatch):
    def rows(component: str, n: int) -> list[dict]:
        return [{"match_key": f"{component}:{k}", "messages": [{"role": "user", "content": f"q{k}"}]} for k in range(n)]

    def render_fails(*args, **kwargs):
        raise ValueError("row does not render")

    class Renderer:
        use_system_prompt = True

    monkeypatch.setattr(s18_build_mix, "prepare_treatment", lambda rows_, **kwargs: rows("constqa", 2))
    monkeypatch.setattr(s18_build_mix, "read_jsonl", lambda path: [])
    monkeypatch.setattr(s18_build_mix, "cd_rows", lambda path, component, arm, expected: rows(component, expected))
    monkeypatch.setattr(s18_build_mix, "stage_dir", lambda *parts: tmp_path / "mix")
    monkeypatch.setattr(s18_build_mix, "load_tokenizer", lambda model: None)
    monkeypatch.setattr(s18_build_mix, "load_renderer", lambda model, name, tok: Renderer())
    monkeypatch.setattr(s18_build_mix, "render_stats", render_fails)
    argv = f"s18 --constitutions {ARM} --constqa-dir {tmp_path} --cd-dir {tmp_path} --n-constqa 2 --n-math 1"
    argv += " --n-coding 1 --required-treatment-renderer r --required-treatment-current-date d --layout-seed 0"
    argv += " --training-seed 0 --batch-size 2 --model m --renderer-name r --renderer-current-date d"
    argv += " --max-seq-length 8 --overwrite false --run-name none"
    monkeypatch.setattr(sys, "argv", argv.split())
    with pytest.raises(ValueError, match="row does not render"):
        s18_build_mix.main()
    assert not (tmp_path / "mix").exists()


def test_rollout_sample_indices_skip_dropped_completions(tmp_path):
    samples = tmp_path / "round0.samples.jsonl"
    groups = {"chat:a": ["dropped", "same", "same", "last"], "chat:b": ["only"]}
    samples.write_text(
        "".join(json.dumps({"task_id": t, "sequences": [{"text": x} for x in xs]}) + "\n" for t, xs in groups.items())
    )
    items = [("chat:a", "same"), ("chat:b", "only"), ("chat:a", "same"), ("chat:a", "last")]
    assert rl_rollout.sample_indices(items, samples) == [1, 0, 2, 3]
    with pytest.raises(ValueError, match="no remaining completion of chat:a"):
        rl_rollout.sample_indices([("chat:a", "last"), ("chat:a", "same")], samples)


def test_rollout_verdict_keys_are_prompt_id_and_sample_index(tmp_path, monkeypatch):
    samples = tmp_path / "round0.samples.jsonl"
    texts = ["cut off", "first answer", "second answer"]
    samples.write_text(json.dumps({"task_id": "chat:a", "sequences": [{"text": x} for x in texts]}) + "\n")
    seen = {}

    def fake_judge(arm, questions, answers, *, judge_model, concurrency, verdicts_path, keys):
        seen["keys"] = keys
        return [{"follows": True, "reward": 1.0}, None]

    class Tokenizer:
        def encode(self, text, add_special_tokens):
            return text.split()

    monkeypatch.setattr(rl_judge, "judge", fake_judge)
    scorer = rl_rollout.make_scorer(
        ARM,
        {"chat:a": {"source": "chat", "question": "q"}},
        "judge",
        1,
        1,
        "role_colon",
        tmp_path / "round0.verdicts.jsonl",
        samples,
        Tokenizer(),
        100,
        100,
    )
    out = scorer([("chat:a", "first answer"), ("chat:a", "second answer")])
    # The audit rows of these two completions carry sample_idx 1 and 2; index 0 never reaches the scorer.
    assert seen["keys"] == ["chat:a#1", "chat:a#2"]
    # A judge error keeps `correct` False (it counts as unsolved in solve_rate) and takes its group's mean reward.
    assert [o["correct"] for o in out] == [True, False]
    assert out[1]["judge_error"] is True and out[1]["base_reward"] == 1.0


def test_document_plan_cost_uses_the_price_table():
    rows = [arrays_common.PlannedRow("document", n, f"r{n}", input_tokens=10, trained_tokens=8) for n in range(4)]
    model = "openai/gpt-oss-20b"
    summary = arrays_common.document_plan_summary(rows, 2, model)
    assert summary["batches"] == 2 and summary["trained_tokens"] == 32
    assert summary["estimated_train_usd"] == 32 / 1_000_000 * TINKER_TRAIN_USD_PER_MTOK[model]
    assert not hasattr(arrays_common, "GPT_OSS_20B_TRAIN_USD_PER_M")
