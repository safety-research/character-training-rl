"""Find the ImpossibleBench tasks whose changed test file is broken by the benchmark's own edit.

An impossible split changes a task's tests so that no correct fix passes. For some tasks the changed test file
cannot even be collected, so every test in it fails whatever the model does, and a pass can only come from replacing
the file. Those tasks measure nothing about the model and are left out of the split. Every rule here is mechanical
and reads only the benchmark's data (no model output):

- Impossible-SWEbench, per instance and impossible split: the split's test patch and the original test patch are
  each applied with `git apply` to the instance's files at its base commit (fetched from GitHub once and cached).
  The task is broken if either patch does not apply, if the original patch yields a test file the split's patch does
  not, if a patched Python test file has a syntax error where the original's compiles, or if the split's patch adds
  a module-level or class-level statement that reads a name nothing binds (a NameError when the file is imported).
- Impossible-LiveCodeBench, per task and impossible split: the split's test code has a syntax error where the
  original test code compiles.

The same rules are run on the original split as a control; it has no broken task.

    uv run python -m analysis.find_broken_tasks

Needs the `swebench` extra (for `unidiff`), `git`, and network access to Hugging Face and GitHub. Writes
results/analysis/scripts/find_broken_tasks/{impossiblebench_broken_tasks.json, detail.json} and prints whether the
list equals the shipped analysis/task_lists/impossiblebench_broken_tasks.json.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import difflib
import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
import warnings
from pathlib import Path

from analysis.common import BROKEN_TASKS, EXPERIMENT, SPLITS
from src import paths
from src.exp.manifest import script_dir

SCRIPT = "find_broken_tasks"
#: the benchmark datasets on Hugging Face, at the revisions the paper's evaluations ran
IB_SWE_DATASET, IB_SWE_REV = "fjzzq2002/impossible_swebench", "9c2d34f364b7229e8c0ff807c646100bdc18bbb5"
IB_LCB_DATASET, IB_LCB_REV = "fjzzq2002/impossible_livecodebench", "98650ffc3f28a01b261669b6d19fcd7773823710"
#: where the base files of the Impossible-SWEbench test patches are cached
BASE_FILES = paths.CACHE / "impossible_swebench_base_files"
#: seconds allowed for one file download
FETCH_TIMEOUT = 60


def base_file(repo: str, commit: str, path: str) -> str | None:
    """The file at the instance's base commit, fetched once from GitHub and cached; None if the repository has no
    such file at that commit."""
    cache = BASE_FILES / repo.replace("/", "__") / commit / path
    missing = cache.with_name(cache.name + ".MISSING")
    if cache.exists():
        return cache.read_text()
    if missing.exists():
        return None
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    try:
        text = urllib.request.urlopen(url, timeout=FETCH_TIMEOUT).read().decode()
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        missing.parent.mkdir(parents=True, exist_ok=True)
        missing.write_text("404\n")
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text)
    return text


def apply_patch(repo: str, commit: str, patch: str) -> dict[str, str] | str:
    """The patched test files, by `git apply` of the patch onto the base files in a scratch directory (the scorer
    applies the patch the same way); the apply error text if it fails."""
    from unidiff import PatchSet

    with tempfile.TemporaryDirectory() as tmp:
        targets = []
        for pf in PatchSet(patch):
            if not pf.is_removed_file:
                targets.append(pf.target_file[2:])
            if pf.is_added_file:
                continue
            base = base_file(repo, commit, pf.source_file[2:])
            if base is None:
                raise FileNotFoundError(f"{repo}@{commit}:{pf.source_file[2:]} not found")
            f = Path(tmp) / pf.source_file[2:]
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(base)
        r = subprocess.run(
            ["git", "apply", "-"],
            input=patch,
            cwd=tmp,
            capture_output=True,
            text=True,
            env={**os.environ, "GIT_CEILING_DIRECTORIES": str(Path(tmp).parent)},
        )
        if r.returncode != 0:
            return r.stderr.strip()
        return {t: (Path(tmp) / t).read_text() for t in targets}


def syntax_error(text: str, name: str) -> str | None:
    """The syntax error compiling `text` raises, or None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            compile(text, name, "exec")
        except SyntaxError as e:
            return f"{type(e).__name__} line {e.lineno}: {e.msg}"
    return None


def unbound_names_at_import(text: str, lines: set[int]) -> list[str]:
    """Statements among `lines` that run when the module imports (module level or class body, outside any function
    or lambda) and read a name nothing binds before them: not a builtin, not bound anywhere at module level, not
    bound earlier in the enclosing class body. Such a statement raises NameError at import, so the whole test module
    fails to collect. A module with a star import is skipped, since its names cannot be known statically."""
    tree = ast.parse(text)

    def bound_by(st) -> set[str]:
        """Names a statement binds in its own scope (not inside nested functions or classes)."""
        out, stack = set(), [st]
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(n.name)
                continue
            if isinstance(n, ast.Lambda):
                continue
            stack.extend(ast.iter_child_nodes(n))
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                out.add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    if a.name == "*":
                        out.add("*")
                    out.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, ast.ExceptHandler) and n.name:
                out.add(n.name)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                out.update(n.names)
        return out

    module_names = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__builtins__", "__spec__", "__package__"}
    for st in tree.body:
        module_names |= bound_by(st)
    if "*" in module_names:
        return []
    hits = []

    def reads(st) -> set[str]:
        """Names a statement reads when it runs, outside nested functions, lambdas and classes. scan() passes
        only statements that are not function or class definitions."""
        out, stack = set(), [st]
        while stack:
            n = stack.pop()
            if n is not st and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                out.add(n.id)
            stack.extend(ast.iter_child_nodes(n))
        return out

    def scan(body) -> None:
        local = set()
        for st in body:
            if isinstance(st, ast.ClassDef):
                scan(st.body)
            elif not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if set(range(st.lineno, (st.end_lineno or st.lineno) + 1)) & lines:
                    free = reads(st) - module_names - local
                    if free:
                        hits.append(f"line {st.lineno}: {sorted(free)}")
            local |= bound_by(st) if not isinstance(st, ast.ClassDef) else {st.name}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan(tree.body)
    return hits


def swe_broken() -> dict[tuple[str, str], list[dict]]:
    """(split, instance id) -> why the split's patched test files are broken, for Impossible-SWEbench."""
    from datasets import load_dataset

    broken = {}
    for sp in SPLITS:
        data = {r["instance_id"]: r for r in load_dataset(IB_SWE_DATASET, split=sp, revision=IB_SWE_REV)}
        for iid, r in sorted(data.items()):
            orig = apply_patch(r["repo"], r["base_commit"], r["original_test_patch"])
            split = apply_patch(r["repo"], r["base_commit"], r["test_patch"])
            if isinstance(orig, str) or isinstance(split, str):
                detail = {
                    "original": orig if isinstance(orig, str) else "applies",
                    "split": split if isinstance(split, str) else "applies",
                }
                broken.setdefault((sp, iid), []).append(
                    {"file": None, "kind": "patch_does_not_apply", "detail": detail}
                )
                continue
            for path in sorted(set(orig) - set(split)):
                broken.setdefault((sp, iid), []).append({"file": path, "kind": "test_file_missing_in_split"})
            for path, text in split.items():
                if not path.endswith(".py"):
                    continue
                o = orig.get(path)
                err, err_o = syntax_error(text, path), (syntax_error(o, path) if o is not None else None)
                kind = None
                if err and not err_o:
                    kind = {"kind": "syntax_error", "detail": err}
                elif not err:
                    added = set()
                    matcher = difflib.SequenceMatcher(None, (o or "").splitlines(), text.splitlines(), autojunk=False)
                    for tag, _, _, j1, j2 in matcher.get_opcodes():
                        if tag in ("replace", "insert"):
                            added |= set(range(j1 + 1, j2 + 1))
                    hits = unbound_names_at_import(text, added) if added else []
                    if hits:
                        kind = {"kind": "name_unbound_at_import", "detail": hits}
                if kind:
                    broken.setdefault((sp, iid), []).append({"file": path, **kind})
    return broken


def lcb_broken() -> dict[tuple[str, str], list[dict]]:
    """(split, task id) -> why the split's test code is broken, for Impossible-LiveCodeBench."""
    from datasets import load_dataset

    broken = {}
    for sp in SPLITS:
        for r in load_dataset(IB_LCB_DATASET, split=sp, revision=IB_LCB_REV):
            err, err_o = syntax_error(r["test"], "test"), syntax_error(r["original_test"], "original_test")
            if err and not err_o:
                broken[(sp, str(r["task_id"]))] = [{"kind": "syntax_error", "detail": err}]
    return broken


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    found = {"iblcb": lcb_broken(), "ibswe": swe_broken()}
    rows = [
        {"benchmark": bench, "split": sp, "task_id": task}
        for bench, broken in found.items()
        for sp, task in sorted(broken, key=lambda k: (SPLITS.index(k[0]), k[1]))
    ]
    detail = {f"{bench}|{sp}|{task}": why for bench, broken in found.items() for (sp, task), why in broken.items()}
    out = script_dir(EXPERIMENT, SCRIPT)
    out.mkdir(parents=True, exist_ok=True)
    (out / BROKEN_TASKS.name).write_text("[\n" + ",\n".join(" " + json.dumps(r) for r in rows) + "\n]\n")
    (out / "detail.json").write_text(json.dumps(detail, indent=1, sort_keys=True))
    for bench in found:
        for sp in SPLITS:
            print(f"{bench} {sp}: {sum((r['benchmark'], r['split']) == (bench, sp) for r in rows)} broken tasks")
    shipped = json.loads(BROKEN_TASKS.read_text())
    key = lambda r: (r["benchmark"], r["split"], r["task_id"])  # noqa: E731
    same = sorted(map(key, rows)) == sorted(map(key, shipped))
    print(f"wrote {out / BROKEN_TASKS.name} ({len(rows)} rows); equal to the shipped list: {same}")


if __name__ == "__main__":
    main()
