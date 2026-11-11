"""MATH exact-answer correctness: the helpfulness anchor of the RL reward.

The last boxed answer in the answer channel, normalized, against the dataset's reference answer.
The checker is the one the context-distilled MATH data was filtered with (s15_generate_math), so a
problem counted correct here is counted correct there. A response with no boxed answer is wrong,
never a judge error.
"""

from __future__ import annotations

import re


def _strip_wrapping_command(value: str, command: str) -> str:
    while command + "{" in value:
        start = value.index(command + "{")
        body_start = start + len(command) + 1
        depth = 1
        cursor = body_start
        while cursor < len(value) and depth:
            if value[cursor] == "{":
                depth += 1
            elif value[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth:
            break
        value = value[:start] + value[body_start : cursor - 1] + value[cursor:]
    return value


def _expand_fractions(value: str) -> str:
    # Canonical LaTeX permits single-token numerator/denominator shorthand such as ``\frac12``.
    value = re.sub(
        r"\\(?:d|t)?frac\s*([+-]?\d)\s*([+-]?\d)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        value,
    )
    for command in ("\\frac", "\\dfrac", "\\tfrac"):
        while command + "{" in value:
            start = value.index(command + "{")
            numerator_start = start + len(command) + 1
            depth = 1
            cursor = numerator_start
            while cursor < len(value) and depth:
                if value[cursor] == "{":
                    depth += 1
                elif value[cursor] == "}":
                    depth -= 1
                cursor += 1
            if depth or cursor >= len(value) or value[cursor] != "{":
                break
            numerator = value[numerator_start : cursor - 1]
            denominator_start = cursor + 1
            depth = 1
            end = denominator_start
            while end < len(value) and depth:
                if value[end] == "{":
                    depth += 1
                elif value[end] == "}":
                    depth -= 1
                end += 1
            if depth:
                break
            denominator = value[denominator_start : end - 1]
            value = value[:start] + f"({numerator})/({denominator})" + value[end:]
    return value


def extract_boxed_answer(text: str) -> str | None:
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    cursor = start
    while cursor < len(text) and depth:
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
        cursor += 1
    if depth:
        return None
    return text[start : cursor - 1]


def normalize_answer(answer: str | None) -> str:
    if answer is None:
        return ""
    value = answer.strip().replace("$", "").replace("\\left", "").replace("\\right", "")
    for command in ("\\text", "\\mathrm", "\\textbf", "\\mathbf", "\\mathit"):
        value = _strip_wrapping_command(value, command)
    value = _expand_fractions(value)
    replacements = {
        "\\%": "%",
        "\\$": "$",
        "\\,": " ",
        "\\;": " ",
        "\\!": "",
        "\\quad": " ",
        "\\qquad": " ",
        "\\cdot": "*",
        "\\times": "*",
        "\\div": "/",
        "\\pm": "+-",
        "\\infty": "inf",
        "\\pi": "pi",
    }
    for token, replacement in replacements.items():
        value = value.replace(token, replacement)
    # LaTeX spacing is presentation-only. Removing it also canonicalizes matrices whose source
    # answers contain padding around entries.
    return re.sub(r"\s+", "", value.rstrip(".,;"))


def check_answer(final: str, reference: str) -> bool:
    predicted = normalize_answer(extract_boxed_answer(final))
    expected = normalize_answer(reference)
    if not predicted or not expected:
        return False
    if predicted == expected:
        return True
    try:
        return abs(float(predicted.replace(" ", "")) - float(expected.replace(" ", ""))) < 1e-6
    except (ValueError, OverflowError):
        pass
    try:
        if "/" in predicted and "/" not in expected:
            numerator, denominator = predicted.replace("(", "").replace(")", "").split("/")
            return abs(float(numerator) / float(denominator) - float(expected)) < 1e-6
        if "/" in expected and "/" not in predicted:
            numerator, denominator = expected.replace("(", "").replace(")", "").split("/")
            return abs(float(numerator) / float(denominator) - float(predicted)) < 1e-6
    except (ValueError, ZeroDivisionError, OverflowError):
        return False
    return False
