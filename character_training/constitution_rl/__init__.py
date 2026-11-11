"""Stage 3: constitution RL. One-turn prompts (constitution questions, MATH, KodCode, chat, and for
constitutions with a section on answering why, self-reflection questions) rewarded by a judge that
reads the constitution."""

STAGE = "constitution_rl"
"""The results folder of this stage: `results/constitution_rl/scripts/<script>/`."""

FIVE_RULE_BANK = "five_rule"


def reflection_dir(script: str, bank: str):
    """Where a self-reflection script writes for one question bank: the script's own folder for the
    `five_rule` bank, and a subfolder named after the bank for any other."""
    from src.exp.manifest import script_dir

    out = script_dir(STAGE, script)
    return out if bank == FIVE_RULE_BANK else out / bank
