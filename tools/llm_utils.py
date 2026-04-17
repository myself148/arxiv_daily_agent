import logging
import time
from typing import Callable, Optional, TypeVar

from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from config import AppConfig


T = TypeVar("T")

RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    APIError,
    RateLimitError,
)

RETRYABLE_MESSAGE_PARTS = (
    "connection error",
    "tls/ssl",
    "eof",
    "rate limit",
    "connection reset",
    "timed out",
)


def build_chat_model(config: AppConfig) -> ChatOpenAI:
    if not config.api_key:
        raise ValueError("OPENAI_API_KEY is missing. Please set it in your .env file.")

    return ChatOpenAI(
        model=config.model_name,
        temperature=config.temperature,
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.llm_timeout,
        max_retries=0,
    )


def is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, RETRYABLE_ERRORS):
        return True

    lowered = str(exc).lower()
    return any(fragment in lowered for fragment in RETRYABLE_MESSAGE_PARTS)


def run_with_retry(
    operation: Callable[[], T],
    *,
    operation_name: str,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> T:
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 2):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - central retry gate
            last_error = exc
            if attempt > max_retries or not is_retryable_error(exc):
                raise

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logging.warning(
                "%s failed on attempt %s/%s: %s. Retrying in %.1f seconds.",
                operation_name,
                attempt,
                max_retries + 1,
                exc,
                delay,
            )
            time.sleep(delay)

    assert last_error is not None
    raise last_error
