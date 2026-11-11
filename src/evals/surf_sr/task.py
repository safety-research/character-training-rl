"""Inspect plumbing for the surf_sr eval: turn the rewritten prompts into one sample per (prompt,
condition) and assemble the Task. What the model sees and how it is scored lives in `eval.py`.
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import generate

from src.evals.surf_sr.eval import CONDITION_FIELD, CONDITIONS, NAME, dual_judge_scorer, load_rewritten
from src.evals.tinker_provider import framing_messages


@task
def surf_sr(prompts_file: str, thinking: bool, epochs: int, framing_prompt: str | None, framing_role: str) -> Task:
    """`thinking` must match the model's declaration. One Sample per (prompt, condition) across the
    specificity ladder (plain/hypothetical/random/code/math); the judge grades the plain request in
    every condition. `framing_prompt` opens every conversation in `framing_role`
    (tinker_provider.framing_messages: "system", "developer", or "none" with no prompt)."""
    rows = load_rewritten(Path(prompts_file))
    framing = framing_messages(framing_prompt, framing_role)
    samples = []
    for i, r in enumerate(rows):
        for cond in CONDITIONS:
            text = r[CONDITION_FIELD[cond]]
            samples.append(
                Sample(
                    id=f"{i}#{cond}",
                    input=[*framing, ChatMessageUser(content=text)],
                    metadata={"prompt": r["prompt"], "category": r["category"], "condition": cond},
                )
            )
    return Task(
        dataset=MemoryDataset(samples, name=NAME),
        solver=generate(),
        scorer=dual_judge_scorer(thinking),
        epochs=Epochs(epochs, "mean"),
        name=NAME,
        metadata={
            "conditions": list(CONDITIONS),
            "n_prompts": len(rows),
            "framing_prompt": framing_prompt,
            "framing_role": framing_role,
        },
    )
