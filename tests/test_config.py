import pytest
from pydantic import ValidationError

from filinglens.config import Settings


def make_settings(**overrides) -> Settings:
    # _env_file=None: tests must not depend on a developer's local .env
    values = {"sec_user_agent": "Jane Doe jane@example.com", **overrides}
    return Settings(_env_file=None, **values)


def test_loads_with_defaults():
    settings = make_settings()
    assert settings.sec_user_agent == "Jane Doe jane@example.com"
    assert settings.llm_provider == "anthropic"


def test_reads_environment(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Env User env@example.com")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    settings = Settings(_env_file=None)
    assert settings.sec_user_agent == "Env User env@example.com"
    assert settings.llm_provider == "openai"


def test_user_agent_without_email_is_rejected():
    with pytest.raises(ValidationError, match="contact email"):
        make_settings(sec_user_agent="just-a-name")


def test_user_agent_is_optional_for_non_edgar_commands():
    assert Settings(_env_file=None).sec_user_agent is None


def test_unknown_llm_provider_is_rejected():
    with pytest.raises(ValidationError):
        make_settings(llm_provider="mistral")


def test_secrets_are_not_printed():
    settings = make_settings(openai_api_key="sk-very-secret")
    assert "sk-very-secret" not in repr(settings)
    assert settings.openai_api_key.get_secret_value() == "sk-very-secret"


def test_judge_defaults_to_a_separate_model():
    settings = make_settings()
    assert (settings.judge_provider, settings.judge_model) == ("anthropic", "claude-haiku-5-5")
    assert settings.judge_temperature is None  # Claude Haiku 5.5 rejects a temperature


def test_empty_judge_temperature_means_not_set(monkeypatch):
    monkeypatch.setenv("JUDGE_TEMPERATURE", "")
    assert Settings(_env_file=None).judge_temperature is None
