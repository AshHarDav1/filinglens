from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from filinglens.config import Settings
from filinglens.llm import get_chat_model, get_judge_model


def make_settings(**overrides) -> Settings:
    values = {"openai_api_key": "sk-test", "anthropic_api_key": "sk-ant-test", **overrides}
    return Settings(_env_file=None, **values)


def test_answerer_and_judge_are_configured_independently():
    settings = make_settings(
        llm_provider="openai",
        llm_model="gpt-6-luna",
        llm_temperature=None,
        judge_provider="anthropic",
        judge_model="claude-haiku-5-5",
        judge_temperature=None,
    )
    answerer, judge = get_chat_model(settings), get_judge_model(settings)
    assert isinstance(answerer, ChatOpenAI) and answerer.model_name == "gpt-6-luna"
    assert answerer.temperature is None  # reasoning model: no temperature is sent
    assert isinstance(judge, ChatAnthropic) and judge.model == "claude-haiku-5-5"
    assert judge.temperature is None
