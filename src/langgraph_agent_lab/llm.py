"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Students should use this helper so the lab works with any supported provider.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import SecretStr
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

load_dotenv()

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
RETRYABLE_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "RateLimitError",
    "ServiceUnavailableError",
}


def is_retryable_llm_error(exc: BaseException) -> bool:
    """Return whether an LLM failure is transient and safe to retry."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    status_code = getattr(exc, "status_code", None)
    return status_code in RETRYABLE_STATUS_CODES or type(exc).__name__ in RETRYABLE_ERROR_NAMES


def invoke_with_retry(
    runnable: Runnable[InputT, OutputT],
    input_value: InputT,
    *,
    max_attempts: int = 3,
    retry_predicate: Callable[[BaseException], bool] = is_retryable_llm_error,
) -> tuple[OutputT, int]:
    """Invoke a LangChain runnable with bounded retry for transient failures only."""
    attempts = 0
    retrying = Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception(retry_predicate),
        reraise=True,
    )
    for attempt in retrying:
        with attempt:
            attempts += 1
            return runnable.invoke(input_value), attempts
    raise RuntimeError("LLM retry loop terminated without a result")


def get_llm(model: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    """Create an LLM client from environment configuration.

    Checks for API keys in this order:
    1. GEMINI_API_KEY + GEMINI_BASE_URL -> OpenAI-compatible ChatOpenAI
    2. GEMINI_API_KEY -> native ChatGoogleGenerativeAI
    3. OPENAI_API_KEY -> ChatOpenAI
    4. ANTHROPIC_API_KEY -> ChatAnthropic

    Override model with the `model` parameter, GEMINI_MODEL, or LLM_MODEL.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_base_url = os.getenv("GEMINI_BASE_URL")
    gemini_model = model or os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL")

    if gemini_key and gemini_base_url:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=gemini_model or "gemini-2.5-flash",
            api_key=SecretStr(gemini_key),
            base_url=gemini_base_url,
            temperature=temperature,
        )

    if gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(
            model=gemini_model or "gemini-2.5-flash",
            google_api_key=gemini_key,
            temperature=temperature,
        )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=model or os.getenv("LLM_MODEL") or "gpt-4o-mini",
            temperature=temperature,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        return ChatAnthropic(
            model=model or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM API key found. Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env\n"
        "See .env.example for configuration."
    )
