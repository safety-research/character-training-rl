"""src.paths: one home for results, data and benchmark checkouts, whatever checkout runs the code."""

import pytest

from src import paths


def test_home_is_the_running_checkout_when_unset(monkeypatch):
    monkeypatch.delenv(paths.HOME_VAR, raising=False)
    assert paths._home() == paths.CODE_ROOT
    assert (paths.CODE_ROOT / "src" / "paths.py").exists()


def test_home_follows_the_variable_and_refuses_a_missing_directory(monkeypatch, tmp_path):
    assert paths.HOME_VAR == "CHARACTER_TRAINING_RL_HOME"
    monkeypatch.setenv(paths.HOME_VAR, str(tmp_path))
    assert paths._home() == tmp_path.resolve()
    monkeypatch.setenv(paths.HOME_VAR, str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match=paths.HOME_VAR):
        paths._home()


def test_every_data_root_hangs_off_home():
    for p in (paths.RESULTS, paths.DATA, paths.THIRD_PARTY, paths.CACHE):
        assert p.parent == paths.HOME


def test_the_env_file_is_beside_the_code():
    assert paths.ENV_FILE == paths.CODE_ROOT / ".env"
