"""Chat model and embedding factories. The provider is chosen in config, so code never imports a
specific vendor SDK directly."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from filinglens.config import Settings, get_settings


def _secret(value) -> str | None:
    return value.get_secret_value() if value else None


def _chat_model(
    settings: Settings, provider: str, model: str, temperature: float | None
) -> BaseChatModel:
    extra = {} if temperature is None else {"temperature": temperature}
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            api_key=_secret(settings.anthropic_api_key),
            max_tokens=2048,
            **extra,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, api_key=_secret(settings.openai_api_key), **extra)


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    return _chat_model(
        settings, settings.llm_provider, settings.llm_model, settings.llm_temperature
    )


def get_judge_model(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    return _chat_model(
        settings, settings.judge_provider, settings.judge_model, settings.judge_temperature
    )


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    # Anthropic has no embeddings API, so embeddings use OpenAI regardless of LLM_PROVIDER.
    settings = settings or get_settings()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model, api_key=_secret(settings.openai_api_key)
    )
