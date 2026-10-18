import logging

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from app.config import settings
from app.prompts import build_system_prompt

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """The assistant could not answer. user_message is safe to show to users."""

    def __init__(self, kind: str, user_message: str):
        super().__init__(user_message)
        self.kind = kind
        self.user_message = user_message


def _make_client() -> OpenAI:
    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )


def ask(question: str) -> str:
    question = (question or "").strip()
    if not question:
        raise LLMError("invalid_input", "Please type a question.")
    if len(question) > settings.max_question_chars:
        raise LLMError(
            "invalid_input",
            f"Question is too long (maximum {settings.max_question_chars} characters).",
        )
    if settings.openai_api_key is None:
        raise LLMError("not_configured", "The assistant is not configured yet.")

    client = _make_client()

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": build_system_prompt(settings.business_name)},
                {"role": "user", "content": question},
            ],
            max_completion_tokens=300,
        )
    except AuthenticationError:
        logger.error("OpenAI authentication failed")
        raise LLMError("auth", "The assistant is temporarily unavailable.") from None
    except RateLimitError:
        logger.error("OpenAI rate limit or quota reached")
        raise LLMError("rate_limit", "The assistant is busy. Please try again shortly.") from None
    except APITimeoutError:
        logger.error("OpenAI request timed out")
        raise LLMError("timeout", "The assistant took too long. Please try again.") from None
    except APIConnectionError:
        logger.error("Could not connect to OpenAI")
        raise LLMError("connection", "The assistant is temporarily unavailable.") from None
    except APIStatusError as error:
        logger.error("OpenAI returned status %s", error.status_code)
        raise LLMError("api_error", "The assistant is temporarily unavailable.") from None
    except Exception as error:
        # Log only the error type, never its message (it could contain private data).
        logger.error("Unexpected error calling OpenAI: %s", type(error).__name__)
        raise LLMError("unexpected", "The assistant is temporarily unavailable.") from None

    answer = ""
    if response.choices:
        answer = (response.choices[0].message.content or "").strip()
    if not answer:
        raise LLMError("empty_response", "The assistant had no answer. Please try again.")
    return answer