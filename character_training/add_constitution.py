"""Register a constitution so every character-training script accepts its name.

    uv run python -m character_training.add_constitution --name <name> --constitution <path> \\
        --shared-conduct <path> --why-section {true,false}

The constitution text is copied to `constitutions/<name>.txt` and the shared conduct to
`constitutions/<name>_shared_conduct.txt`, and one entry is appended to
`constitutions/registry.json` with both SHA-256 hashes, the placeholders found in the text, the
`--why-section` value, and a question bank named after the constitution. `--bank <existing bank>`
takes the place of `--shared-conduct` for a constitution that shares the questions of a registered
bank; no shared conduct is copied then.

Checked before anything is written:

- the name is lowercase letters, digits and underscores, and is not registered;
- the constitution text carries `{model_name}`, and no brace other than `{model_name}` and
  `{provider_name}` (`{provider_name}` is recorded when present);
- `--why-section true` goes with a `## When Asked Why` heading in the text and `false` with none,
  because the RL judge scores that section by its title;
- the shared conduct has at least one `## ` section, each with a distinct name that the answer
  judge can use as a field (one field per section);
- neither target file exists.

Existing entries and files are never changed. The command makes no network call. `--dry-run true`
prints the entry and the files it would write and writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from character_training.common import (
    CONSTITUTION_DIR,
    NAME_PATTERN,
    PLACEHOLDERS,
    REGISTRY_FILE,
    bool_arg,
    heading_slug,
    read_registry,
)
from character_training.distillation.s06_judge_answers import RESERVED_FIELDS

WHY_HEADING = "## When Asked Why"
"""The heading of the section a `--why-section true` constitution carries."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def found_placeholders(text: str, source: Path) -> list[str]:
    """The placeholders the constitution text carries, in PLACEHOLDERS order. `{model_name}` is
    required, and no brace may remain once the placeholders are filled in."""
    if "{model_name}" not in text:
        raise ValueError(f"{source}: the constitution must name the model as {{model_name}}")
    rest = text
    for placeholder in PLACEHOLDERS:
        rest = rest.replace(placeholder, "")
    if "{" in rest or "}" in rest:
        raise ValueError(f"{source}: a brace remains outside {' and '.join(PLACEHOLDERS)}")
    return [placeholder for placeholder in PLACEHOLDERS if placeholder in text]


def check_why_section(text: str, why_section: bool, source: Path) -> None:
    has_heading = re.search(rf"(?m)^{re.escape(WHY_HEADING)}\s*$", text) is not None
    if why_section and not has_heading:
        raise ValueError(f"{source}: --why-section true needs a `{WHY_HEADING}` heading in the constitution")
    if has_heading and not why_section:
        raise ValueError(f"{source}: the constitution has a `{WHY_HEADING}` heading; pass --why-section true")


def conduct_section_names(text: str, source: Path) -> list[str]:
    """The names of the shared conduct's `## ` sections, as the answer judge's rule fields."""
    names = [heading_slug(heading) for heading in re.findall(r"(?m)^## (.+)$", text)]
    if not names:
        raise ValueError(f"{source}: the shared conduct needs at least one `## ` section, one per rule")
    if "" in names or len(set(names)) != len(names):
        raise ValueError(f"{source}: section headings must give distinct, non-empty names; got {names}")
    clash = sorted(set(names) & set(RESERVED_FIELDS))
    if clash:
        raise ValueError(f"{source}: section names {clash} are taken by the judge's own fields; rename the headings")
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", required=True, help="lowercase letters, digits and underscores")
    parser.add_argument("--constitution", required=True, type=Path, help="the constitution text, with {model_name}")
    conduct = parser.add_mutually_exclusive_group(required=True)
    conduct.add_argument("--shared-conduct", type=Path, help="the same conduct as behaviour only, with no reasons")
    conduct.add_argument("--bank", help="a registered question bank to share instead of a new shared conduct")
    parser.add_argument(
        "--why-section", required=True, type=bool_arg, help="whether the text has a When Asked Why section"
    )
    parser.add_argument("--dry-run", type=bool_arg, default=False, help="true prints what would be written")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Register the constitution, or stop with the reason it is refused."""
    try:
        register(parse_args(argv))
    except (ValueError, FileNotFoundError) as refusal:
        raise SystemExit(str(refusal)) from None


def register(args: argparse.Namespace) -> None:
    directory = CONSTITUTION_DIR
    entries = read_registry(directory)
    registered = {entry["name"] for entry in entries}
    banks = {entry["bank"]: entry for entry in entries}

    if not NAME_PATTERN.fullmatch(args.name):
        raise SystemExit(f"--name {args.name!r}: use lowercase letters, digits and underscores")
    if args.name in registered:
        raise SystemExit(f"--name {args.name!r} is already registered in {directory / REGISTRY_FILE}")

    constitution_bytes = args.constitution.read_bytes()
    constitution_text = constitution_bytes.decode("utf-8")
    placeholders = found_placeholders(constitution_text, args.constitution)
    check_why_section(constitution_text, args.why_section, args.constitution)
    constitution_target = directory / f"{args.name}.txt"
    writes: list[tuple[Path, bytes]] = [(constitution_target, constitution_bytes)]

    if args.bank is not None:
        if args.bank not in banks:
            raise SystemExit(f"--bank {args.bank!r} is not a registered bank; registered: {sorted(banks)}")
        bank = args.bank
        conduct_file = banks[bank]["shared_conduct_file"]
        conduct_sha256 = banks[bank]["shared_conduct_sha256"]
        section_names = None
    else:
        if args.name in banks:
            raise SystemExit(f"--name {args.name!r} is already the name of a question bank")
        conduct_bytes = args.shared_conduct.read_bytes()
        section_names = conduct_section_names(conduct_bytes.decode("utf-8"), args.shared_conduct)
        bank = args.name
        conduct_file = f"{args.name}_shared_conduct.txt"
        conduct_sha256 = sha256_bytes(conduct_bytes)
        writes.append((directory / conduct_file, conduct_bytes))

    for target, _ in writes:
        if target.exists():
            raise SystemExit(f"{target} exists; it is never overwritten")

    entry = {
        "name": args.name,
        "file": constitution_target.name,
        "sha256": sha256_bytes(constitution_bytes),
        "placeholders": placeholders,
        "why_section": args.why_section,
        "bank": bank,
        "shared_conduct_file": conduct_file,
        "shared_conduct_sha256": conduct_sha256,
    }
    registry_path = directory / REGISTRY_FILE
    print(f"Registry: {registry_path}")
    print("Entry:\n" + json.dumps(entry, indent=2))
    for target, data in writes:
        print(f"File: {target} ({len(data)} bytes)")
    if section_names is not None:
        print(f"Answer-judge rule fields, one per shared-conduct section: {', '.join(section_names)}")
    if args.dry_run:
        print("DRY RUN: nothing written.")
        return

    for target, data in writes:
        with target.open("xb") as sink:
            sink.write(data)
    pending = registry_path.with_name(registry_path.name + ".pending")
    pending.write_text(json.dumps({"constitutions": [*entries, entry]}, indent=2) + "\n")
    os.replace(pending, registry_path)
    read_registry(directory)
    print(f"Registered {args.name}. Next: character_training/run.sh {args.name}")


if __name__ == "__main__":
    main()
