"""Key selection: provider from the model name, standard environment variables, loud on anything unknown."""

import pytest

from src import keys


@pytest.fixture(autouse=True)
def placeholder_env(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(var, f"placeholder-{var}")


def test_provider_is_derived_from_the_model_name():
    assert keys.get_api_key("judge", "claude-sonnet-4-6") == "placeholder-ANTHROPIC_API_KEY"
    assert keys.get_api_key("generation", "gpt-4.1-mini") == "placeholder-OPENAI_API_KEY"
    assert keys.get_api_key("generation", "deepseek/deepseek-chat") == "placeholder-OPENROUTER_API_KEY"
    assert keys.get_api_key("batch") == "placeholder-ANTHROPIC_API_KEY"


def test_unknown_role_model_or_missing_key_is_an_error(monkeypatch):
    with pytest.raises(ValueError):
        keys.get_api_key("other", "claude-sonnet-4-6")
    with pytest.raises(ValueError):
        keys.get_api_key("generation", "some-model")
    with pytest.raises(ValueError):
        keys.get_api_key("judge", "deepseek/deepseek-chat")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        keys.anthropic_client()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    with pytest.raises(RuntimeError, match="set but empty"):
        keys.get_api_key("judge", "claude-sonnet-4-6")


def test_env_file_fills_in_only_what_is_missing(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# comment\nOPENAI_API_KEY=from-file\nNEW_VARIABLE=from-file\n")
    monkeypatch.setattr(keys, "_PROJECT_ENV_FILE", f)
    monkeypatch.setattr(keys, "_env_loaded", False)
    monkeypatch.delenv("NEW_VARIABLE", raising=False)
    keys.load_project_env()
    import os

    assert os.environ["OPENAI_API_KEY"] == "placeholder-OPENAI_API_KEY"
    assert os.environ["NEW_VARIABLE"] == "from-file"
    monkeypatch.delenv("NEW_VARIABLE")
