"""What the model sees in GPQA Diamond and how it is measured.

GPQA Diamond is the 198-question subset of GPQA (Rein et al., 2023): graduate-level multiple-choice
questions in physics, chemistry and biology, each with one correct and three incorrect answers. The
dataset is read from the Hub at the revision pinned here, so the question order (which seeds each
question's choice shuffle) cannot drift underneath a comparison.

The model receives one user turn: `GPQA_PROMPT` over the question and its four choices. The choices
are shuffled per question with `random.Random(choice_seed ^ question_index)` so the correct answer's
position is fixed by the seed and the question, independent of every other question. Scoring is
deterministic, no judge: `extract_answer` reads the letter from the last-line format the prompt asks
for, with fallbacks for common variants, and the score is 1 when it equals the correct letter.

The letter is extracted from the answer channel only, never from the reasoning channel. A reasoning
model considers several letters while it thinks; reading that text would credit a letter the model
weighed and did not answer. A completion whose only letter is in its reasoning has no extractable
answer.
"""

from __future__ import annotations

import random
import re
from typing import Optional

NAME = "gpqa"

DATASET = "Idavidrein/gpqa"
DATASET_CONFIG = "gpqa_diamond"
DATASET_SPLIT = "train"
#: The dataset's commit on the Hub; question order seeds the per-question choice shuffle.
DATASET_REVISION = "83022cefff930aea54f654c0b282e74b9eeda5c6"
EXPECTED_N = 198

#: Answer letters in display order; the prompt names them joined by commas.
LETTERS = ["A", "B", "C", "D"]

GPQA_PROMPT = """Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}"""


def load_rows() -> list[dict]:
    """Every GPQA Diamond row at the pinned revision; the row count is checked against the benchmark.
    The dataset is gated on the Hub: loading needs an accepted access request and a Hugging Face login."""
    from datasets import load_dataset

    rows = [dict(r) for r in load_dataset(DATASET, DATASET_CONFIG, split=DATASET_SPLIT, revision=DATASET_REVISION)]
    if len(rows) != EXPECTED_N:
        raise RuntimeError(
            f"{DATASET}/{DATASET_CONFIG}@{DATASET_REVISION[:12]} has {len(rows)} rows; Diamond has {EXPECTED_N}"
        )
    return rows


def format_choices(choices: list[str], labels: list[str] = None) -> str:
    """Format choices as A) ... B) ... etc."""
    if labels is None:
        labels = ["A", "B", "C", "D"]
    return "\n".join(f"{label}) {choice}" for label, choice in zip(labels, choices))


def build_question(row: dict, index: int, choice_seed: int) -> dict:
    """One question as the model sees it: the four answers shuffled by `Random(choice_seed ^ index)`
    over positions [0, 1, 2, 3] (position 0 is the correct answer before shuffling), the prompt, and
    the letter the correct answer landed on."""
    choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    order = list(range(4))
    random.Random(choice_seed ^ index).shuffle(order)
    shuffled = [choices[i] for i in order]
    correct_letter = LETTERS[order.index(0)]
    prompt = GPQA_PROMPT.format(letters=",".join(LETTERS), question=row["Question"], choices=format_choices(shuffled))
    return {"index": index, "prompt": prompt, "choices": shuffled, "correct_letter": correct_letter}


def build_questions(rows: list[dict], choice_seed: int, limit: int | None) -> list[dict]:
    """Every question in dataset order, or the first `limit`. A question's shuffle is keyed by its
    dataset index, so the first N questions are the same prompts with or without a limit."""
    questions = [build_question(row, i, choice_seed) for i, row in enumerate(rows)]
    return questions if limit is None else questions[:limit]


#: An answer statement: "answer" in any case, at most ANSWER_GAP formatting characters (colon, markdown
#: bold, LaTeX braces, parentheses, spaces, one line break), then an uppercase letter A-D standing alone.
#: Covers what the model writes for the requested 'ANSWER: $LETTER' line ("**Answer: C**", "ANSWER: **D**",
#: "**Answer:** D", "\text{ANSWER: }D"); the letter must not continue into a word ("Answer: Due ..." is not D).
ANSWER_GAP = 8
ANSWER_STATEMENT = re.compile(
    r"(?i:answer)[^A-Za-z0-9]{0,%d}?(?:\\text\{)?\s*\**\(?([A-D])(?![A-Za-z0-9])" % ANSWER_GAP
)
#: A final line that is only the letter ("A", "**B**", "(C)") or a boxed letter ("\boxed{\text{B}}").
BARE_LETTER = re.compile(r"^[\s*(]*([A-D])[\s*).]*$")
BOXED_LETTER = re.compile(r"\\boxed\{(?:\\text\{)?\s*([A-D])\s*\}")


def extract_answer(response: str) -> Optional[str]:
    """The model's final answer letter in `response` (the answer channel only), or None.

    The LAST answer statement wins (ANSWER_STATEMENT), so a revised answer beats an earlier draft; with
    none, a boxed letter, or a final line that is only a letter. Letters must stand alone, so words that
    start with A-D are never read as answers. Anything else is an extraction failure, reported beside
    accuracy."""
    statements = ANSWER_STATEMENT.findall(response)
    if statements:
        return statements[-1]
    boxed = BOXED_LETTER.findall(response)
    if boxed:
        return boxed[-1]
    lines = [line for line in response.strip().splitlines() if line.strip()]
    if lines:
        bare = BARE_LETTER.match(lines[-1])
        if bare:
            return bare.group(1)
    return None
