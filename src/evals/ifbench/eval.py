"""What the model sees and how it is measured in IFBench (AllenAI, the IFEval-lineage
instruction-following benchmark).

The model receives one user turn: an IFBench prompt carrying one or more mechanically checkable
constraints ("write exactly three bullet points", "end with this word"). Scoring is fully
deterministic -- the benchmark's own strict and loose verifiers, no LLM judge. The headline is
loose prompt-level accuracy: a prompt passes only when every constraint attached to it passes.
This is constraint fidelity, a usefulness guardrail, not a measure of semantic task quality.

The verifier implementation is vendored under ``src/evals/ifbench_scorers`` from AllenAI IFBench
commit 1091c4c3de6c1f6ed12c012ed68f11ea450b0117 (Apache-2.0). The dataset stays in gitignored
``data/ifbench`` and is checksum-verified at load.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from src import paths
from src.jsonl import read_jsonl

NAME = "ifbench"

DATA_PATH = paths.DATA / "ifbench" / "IFBench_test.jsonl"
DATA_SHA256 = "a2bf7b8e7ed11d39c65b4ec781da8b99663ed0121f97033d70fcecb61f8a9d85"
EXPECTED_PROMPTS = 300
SCORER_COMMIT = "1091c4c3de6c1f6ed12c012ed68f11ea450b0117"


def load_prompts(path: Path = DATA_PATH) -> list[dict]:
    """The 300 IFBench prompts, checksum- and shape-verified against the declared facts."""
    if not path.exists():
        raise FileNotFoundError(
            f"IFBench data missing: {path}. Download IFBench_test.jsonl from the public "
            f"allenai/IFBench release and place it at that path."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise RuntimeError(f"IFBench checksum mismatch: got {digest}, expected {DATA_SHA256}")
    rows = read_jsonl(path)
    if len(rows) != EXPECTED_PROMPTS:
        raise RuntimeError(f"IFBench row count changed: got {len(rows)}, expected {EXPECTED_PROMPTS}")
    keys = [str(r["key"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("IFBench keys are not unique")
    return rows


def score_response(item: dict, answer: str) -> dict:
    """Apply the official strict and loose verifiers to one prompt/answer pair (no LLM judge)."""
    from src.evals.ifbench_scorers.evaluation_lib import (
        InputExample,
        test_instruction_following_loose,
        test_instruction_following_strict,
    )

    inp = InputExample(
        key=item["key"],
        instruction_id_list=list(item["instruction_id_list"]),
        prompt=item["prompt"],
        kwargs=[dict(k) for k in item["kwargs"]],
    )
    mapping = {item["prompt"]: answer}
    strict = test_instruction_following_strict(inp, mapping)
    # Preserve the official runner's call order: strict removes null kwargs in place, then loose
    # scores the same InputExample object.
    loose = test_instruction_following_loose(inp, mapping)
    return {
        "strict_prompt_pass": strict.follow_all_instructions,
        "strict_instruction_passes": strict.follow_instruction_list,
        "loose_prompt_pass": loose.follow_all_instructions,
        "loose_instruction_passes": loose.follow_instruction_list,
    }


def require_nltk_resources() -> None:
    """Fail before generation, not after it. The vendored scorers call sent_tokenize, word_tokenize,
    pos_tag and corpus.stopwords; each needs a corpus that is not a pip dependency, and the LookupError
    would otherwise land during scoring, after every generation has been paid for."""
    import nltk

    for resource, path in (
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ("stopwords", "corpora/stopwords"),
    ):
        try:
            nltk.data.find(path)
        except LookupError:
            # Keep first-run setup convenient, but never contact the network for a resource already
            # present (offline workers).
            if not nltk.download(resource, quiet=True):
                raise RuntimeError(
                    f"IFBench scoring requires the NLTK resource {resource!r}; install it before the run "
                    f"or point NLTK_DATA at an existing copy."
                ) from None
