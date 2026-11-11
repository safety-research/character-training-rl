"""src.exp.manifest: where Inspect logs and pipeline scripts write."""

import pytest

from src.exp import manifest


def test_script_dir_is_under_the_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest, "RESULTS", tmp_path)
    assert manifest.script_dir("some_stage", "some_script") == tmp_path / "some_stage" / "scripts" / "some_script"


def test_inspect_logs_is_one_root_under_results(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest, "RESULTS", tmp_path)
    assert manifest.inspect_logs() == tmp_path / "inspect"


@pytest.mark.parametrize("bad", [("", "s"), ("e", ""), ("a/b", "s"), ("e", "../s")])
def test_script_dir_refuses_paths_that_are_not_single_components(bad):
    with pytest.raises(ValueError):
        manifest.script_dir(*bad)
