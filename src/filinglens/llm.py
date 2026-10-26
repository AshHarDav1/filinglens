"""Chat model and embedding factories. The provider is chosen in config, so code never imports a
specific vendor SDK directly."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from filinglens.config import Settings, get_settings


def _secret(value) -> str | None:
    return value.get_secret_value() if value else None


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    extra = {} if settings.llm_temperature is None else {"temperature": settings.llm_temperature}
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            api_key=_secret(settings.anthropic_api_key),
            max_tokens=2048,
            **extra,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.llm_model, api_key=_secret(settings.openai_api_key), **extra)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    # Anthropic has no embeddings API, so embeddings use OpenAI regardless of LLM_PROVIDER.
    settings = settings or get_settings()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model, api_key=_secret(settings.openai_api_key)
    )
