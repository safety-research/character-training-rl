"""The registered constitutions and the small deterministic helpers every character-training script shares.

`constitutions/registry.json` lists the constitutions. Each entry names the constitution, its file
under `constitutions/` and the file's SHA-256, the placeholders the text carries, whether it has a
"When Asked Why" section, its question bank, and the bank's shared-conduct file and SHA-256. This
module reads the registry once at import and builds the constants below from it. The paper's four
constitutions are registered as `assigned_rules`, `value_grounded_rules`, `bare_rules` and
`no_harm_rule` (the control without the Avoid Harm rule);
`python -m character_training.add_constitution` registers another.

A script that works on several constitutions at once takes `--constitutions`, in the order the
outputs are laid out; the assigned-rules and value-grounded constitutions were built together, so
their scripts are run with both names.

A question bank is the set of constitution questions generated from one shared-conduct text. The
paper's banks are `five_rule` (shared by the assigned-rules, value-grounded and bare-rules
constitutions) and `no_harm` (the same conduct without its Avoid Harm section). A constitution
registered with its own shared conduct has a bank named after it.

Set `CHARACTER_TRAINING_RL_CONSTITUTIONS` to read the registry and the constitution files from
another directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from src.exp.manifest import script_dir

REPO = Path(__file__).resolve().parents[1]
CONSTITUTIONS_DIR_VAR = "CHARACTER_TRAINING_RL_CONSTITUTIONS"
REGISTRY_FILE = "registry.json"
REGISTRY_FIELDS = (
    "name",
    "file",
    "sha256",
    "placeholders",
    "why_section",
    "bank",
    "shared_conduct_file",
    "shared_conduct_sha256",
)
"""The fields of one registry entry. Every field is required and no other field is accepted."""
PLACEHOLDERS = ("{model_name}", "{provider_name}")
"""The placeholders a constitution text may carry. `{model_name}` is required in every constitution."""
NAME_PATTERN = re.compile(r"[a-z0-9_]+")
"""A constitution or bank name: lowercase letters, digits and underscores. Names become folder names,
file names and command-line values."""


def _constitution_dir() -> Path:
    raw = os.environ.get(CONSTITUTIONS_DIR_VAR)
    if raw is None:
        return REPO / "constitutions"
    directory = Path(raw).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"{CONSTITUTIONS_DIR_VAR}={raw!r} is not a directory")
    return directory.resolve()


CONSTITUTION_DIR = _constitution_dir()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_file_name(value: object) -> bool:
    return isinstance(value, str) and bool(value) and "/" not in value and value not in {".", ".."}


def read_registry(directory: Path) -> list[dict]:
    """The entries of `<directory>/registry.json`, in file order, after checking them.

    Refused: a top-level key other than `constitutions`, an entry with a missing or unknown field or
    a value of the wrong type, a repeated name, a registered file that does not exist, and two
    entries that give one bank different shared-conduct files or hashes. File bytes are compared
    with the recorded hashes when a file is used (`resolved_constitution`, `shared_conduct`)."""
    path = directory / REGISTRY_FILE
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or set(data) != {"constitutions"}:
        raise ValueError(f"{path}: expected one top-level key, `constitutions`")
    entries = data["constitutions"]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: `constitutions` is a non-empty list")
    banks: dict[str, tuple[str, str]] = {}
    names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: every entry is an object, got {entry!r}")
        missing = [field for field in REGISTRY_FIELDS if field not in entry]
        unknown = sorted(set(entry) - set(REGISTRY_FIELDS))
        if missing or unknown:
            raise ValueError(f"{path}: entry {entry.get('name')!r} has missing fields {missing}, unknown {unknown}")
        name, bank = entry["name"], entry["bank"]
        for label, value in (("name", name), ("bank", bank)):
            if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
                raise ValueError(f"{path}: {label} {value!r} must be lowercase letters, digits and underscores")
        if name in names:
            raise ValueError(f"{path}: constitution {name!r} is registered twice")
        names.add(name)
        placeholders = entry["placeholders"]
        if (
            not isinstance(placeholders, list)
            or len(set(placeholders)) != len(placeholders)
            or not set(placeholders) <= set(PLACEHOLDERS)
            or "{model_name}" not in placeholders
        ):
            raise ValueError(f"{path}: {name}: placeholders are {{model_name}}, optionally {{provider_name}}")
        if not isinstance(entry["why_section"], bool):
            raise ValueError(f"{path}: {name}: why_section is true or false")
        for field in ("sha256", "shared_conduct_sha256"):
            if not _is_sha256(entry[field]):
                raise ValueError(f"{path}: {name}: {field} is 64 lowercase hex digits")
        for field in ("file", "shared_conduct_file"):
            if not _is_file_name(entry[field]):
                raise ValueError(f"{path}: {name}: {field} is a file name inside {directory}")
            if not (directory / entry[field]).is_file():
                raise FileNotFoundError(f"{path}: {name}: {field} {directory / entry[field]} does not exist")
        conduct = (entry["shared_conduct_file"], entry["shared_conduct_sha256"])
        if banks.setdefault(bank, conduct) != conduct:
            raise ValueError(f"{path}: bank {bank!r} is given two different shared-conduct files or hashes")
    return entries


REGISTRY = read_registry(CONSTITUTION_DIR)

CONSTITUTIONS = tuple(entry["name"] for entry in REGISTRY)
CONSTITUTION_FILE = {entry["name"]: entry["file"] for entry in REGISTRY}
SPEC_SHA256 = {entry["name"]: entry["sha256"] for entry in REGISTRY}
"""SHA-256 of each constitution file. Answers and judge verdicts are only valid against these bytes."""
REQUIRED_PLACEHOLDERS = {entry["name"]: tuple(entry["placeholders"]) for entry in REGISTRY}
"""Placeholders each constitution file must carry. A constitution that names no author carries no
{provider_name}."""
CONSTITUTIONS_WITH_WHY_SECTION = tuple(entry["name"] for entry in REGISTRY if entry["why_section"])
"""Constitutions with a "When Asked Why" section. Only these are trained on self-reflection questions
and judged on the reason they give."""

BANKS = tuple(dict.fromkeys(entry["bank"] for entry in REGISTRY))
BANK_OF = {entry["name"]: entry["bank"] for entry in REGISTRY}
CONDUCT_FILE = {entry["bank"]: entry["shared_conduct_file"] for entry in REGISTRY}
SHARED_CONDUCT_SHA256 = {entry["bank"]: entry["shared_conduct_sha256"] for entry in REGISTRY}
"""The shared conduct is the behavioral statement of the constitutions with no reason given for any
rule. Questions are generated from it and the answer judge scores behavior against it, so one bar
applies to every constitution of a bank; the constitutions differ in how they motivate the rules,
which appears only in the answer-generation developer message."""

MODEL_NAME = "ChatGPT"
PROVIDER_NAME = "OpenAI"
"""What the {model_name}/{provider_name} placeholders resolve to: gpt-oss's own identity."""
DELIBERATION_INSTRUCTION = """Before answering, analyze the user's request using the relevant parts of the model's constitution.
Refer to those parts explicitly in your analysis. Do not discuss irrelevant sections.
Your final answer should contain only the answer to the user, not the analysis."""
"""Appended to the constitution in the answer-time developer message; never present at training time."""
HARMONY_REASONING_END = "assistantfinal"


def bool_arg(value: str) -> bool:
    if value not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def add_constitutions_arg(parser: argparse.ArgumentParser) -> None:
    """`--constitutions`: one or more constitution names, in output order."""
    parser.add_argument(
        "--constitutions",
        required=True,
        nargs="+",
        choices=CONSTITUTIONS,
        help="the constitutions this run covers, in output order",
    )


def checked_constitutions(names: list[str]) -> tuple[str, ...]:
    """The `--constitutions` value as a tuple, refused if a name repeats."""
    if len(set(names)) != len(names):
        raise ValueError(f"a constitution is named twice: {names}")
    return tuple(names)


def run_label(constitutions: tuple[str, ...] | list[str]) -> str:
    """The output subfolder of a run over these constitutions."""
    return "+".join(constitutions)


def stage_dir(stage: str, script: str, label: str) -> Path:
    """Where one script writes for one set of constitutions or one question bank:
    `results/<stage>/scripts/<script>/<label>/`."""
    if not label or "/" in label:
        raise ValueError(f"label is one path component, got {label!r}")
    return script_dir(stage, script) / label


def bank_of(constitutions: tuple[str, ...] | list[str]) -> str:
    """The question bank these constitutions share; constitutions of different banks cannot be mixed."""
    banks = {BANK_OF[name] for name in constitutions}
    if len(banks) != 1:
        raise ValueError(f"constitutions {list(constitutions)} do not share one question bank")
    return banks.pop()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        sink.flush()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def extract_output_block(text: str) -> str:
    match = re.search(r"<output>\s*(.*?)\s*</output>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("response did not contain one <output>...</output> block")
    return match.group(1)


def parse_numbered_items(text: str) -> list[str]:
    block = extract_output_block(text)
    matches = list(re.finditer(r"(?m)^\s*\d+[.)]\s+", block))
    if not matches:
        raise ValueError("<output> block did not contain a numbered list")
    items = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(block)
        item = re.sub(r"\s+", " ", block[start:end]).strip()
        if item:
            items.append(item)
    return items


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def constitution_path(name: str) -> Path:
    if name not in CONSTITUTIONS:
        raise ValueError(f"unknown constitution {name!r}; expected one of {CONSTITUTIONS}")
    return CONSTITUTION_DIR / CONSTITUTION_FILE[name]


def resolved_constitution(arm: str) -> str:
    """The constitution with placeholders resolved, after verifying its bytes against SPEC_SHA256."""
    path = constitution_path(arm)
    digest = sha256_file(path)
    if digest != SPEC_SHA256[arm]:
        raise ValueError(f"{path}: SHA-256 {digest} differs from the expected {SPEC_SHA256[arm]}")
    text = path.read_text()
    missing = [placeholder for placeholder in REQUIRED_PLACEHOLDERS[arm] if placeholder not in text]
    if missing:
        raise ValueError(f"{path}: expected unresolved placeholders {missing}")
    text = text.replace("{model_name}", MODEL_NAME).replace("{provider_name}", PROVIDER_NAME)
    if "{" in text or "}" in text:
        raise ValueError(f"{path}: unresolved brace after substitution")
    return text.rstrip()


def developer_message(arm: str) -> str:
    return resolved_constitution(arm) + "\n\n" + DELIBERATION_INSTRUCTION


def conduct_path(bank: str) -> Path:
    if bank not in BANKS:
        raise ValueError(f"unknown question bank {bank!r}; expected one of {BANKS}")
    return CONSTITUTION_DIR / CONDUCT_FILE[bank]


def shared_conduct(bank: str) -> str:
    """The behavior sections of a bank's shared conduct: everything from the first `## ` heading down,
    dropping the file's title and its note about how the file is used (which is about question
    generation and does not belong in a judge or any other consumer). Bytes verified against
    SHARED_CONDUCT_SHA256 first, so the slice is deterministic."""
    path = conduct_path(bank)
    digest = sha256_file(path)
    if digest != SHARED_CONDUCT_SHA256[bank]:
        raise ValueError(f"{path}: SHA-256 {digest} differs from the expected {SHARED_CONDUCT_SHA256[bank]}")
    text = path.read_text()
    start = text.index("## ")
    return text[start:].rstrip()


def heading_slug(heading: str) -> str:
    """A section heading as a field name: lowercase letters joined by underscores."""
    return re.sub(r"[^a-z]+", "_", heading.strip().lower()).strip("_")


def conduct_sections(bank: str) -> dict[str, str]:
    """The sections of a bank's shared conduct, one per rule, in file order: the slug of each `## `
    heading mapped to the section's text (its heading line, then its body)."""
    sections: dict[str, str] = {}
    for part in re.split(r"(?m)^## ", shared_conduct(bank)):
        if not part.strip():
            continue
        heading, _, body = part.partition("\n")
        slug = heading_slug(heading)
        if not slug or slug in sections:
            raise ValueError(f"{conduct_path(bank)}: heading {heading.strip()!r} gives an empty or repeated name")
        sections[slug] = f"## {heading.strip()}\n{body.strip()}"
    return sections


def split_harmony(raw: str | None) -> tuple[str | None, str | None, bool]:
    """Split a decoded gpt-oss completion into its analysis and final channels."""
    if not raw or HARMONY_REASONING_END not in raw:
        return None, None, False
    analysis, final = raw.rsplit(HARMONY_REASONING_END, 1)
    if analysis.startswith("analysis"):
        analysis = analysis[len("analysis") :]
    analysis = analysis.strip()
    final = final.strip()
    return analysis, final, bool(analysis and final)
