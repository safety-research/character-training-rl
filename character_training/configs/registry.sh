#!/usr/bin/env bash
# Shell access to the constitution registry (constitutions/registry.json), sourced by the stage
# scripts in this folder and by character_training/run.sh. It defines:
#   registry_field <constitution> <field>   print one field of a registered constitution
#                                           (booleans as true/false); stop if it is not registered
#   json_field <file> <dotted.key>          print one value of a JSON file (booleans as true/false,
#                                           lists joined by commas; a `*` segment maps over a list)
#   is_paper_constitution <constitution>    succeed for the paper's four constitutions
# The registry is read from $CHARACTER_TRAINING_RL_CONSTITUTIONS when that is set, as
# character_training/common.py does, and from constitutions/ in this repository otherwise.
# JSON is read with python3, or with `uv run python` where python3 is not installed.

CHARACTER_TRAINING_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONSTITUTION_REGISTRY="${CHARACTER_TRAINING_RL_CONSTITUTIONS:-$CHARACTER_TRAINING_REPO/constitutions}/registry.json"

_registry_python() {
  if command -v python3 >/dev/null 2>&1; then
    python3 "$@"
  else
    uv run python "$@"
  fi
}

registry_field() {
  _registry_python - "$CONSTITUTION_REGISTRY" "$1" "$2" <<'PY'
import json
import sys

path, name, field = sys.argv[1:]
entries = {entry["name"]: entry for entry in json.load(open(path))["constitutions"]}
if name not in entries:
    sys.exit(
        f"{name!r} is not a registered constitution. Registered: {', '.join(entries)}. "
        "Register one with: uv run python -m character_training.add_constitution --help"
    )
value = entries[name][field]
print(str(value).lower() if isinstance(value, bool) else value)
PY
}

json_field() {
  _registry_python - "$1" "$2" <<'PY'
import json
import sys



def walk(value, keys):
    if not keys:
        return value
    if keys[0] == "*":
        return [walk(item, keys[1:]) for item in value]
    return walk(value[keys[0]], keys[1:])


def text(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


path, dotted = sys.argv[1:]
value = walk(json.load(open(path)), dotted.split("."))
print(",".join(text(item) for item in value) if isinstance(value, list) else text(value))
PY
}

is_paper_constitution() {
  case "$1" in
    assigned_rules | value_grounded_rules | bare_rules | no_harm_rule) return 0 ;;
    *) return 1 ;;
  esac
}
