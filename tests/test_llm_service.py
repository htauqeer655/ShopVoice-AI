import logging
from types import SimpleNamespace

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)
from pydantic import SecretStr

from app.config import settings
from app.services import llm_service
from app.services.llm_service import LLMError, ask

FAKE_KEY = "fake-secret-key-123"


def _request():
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _response(status):
    return httpx.Response(status, request=_request())


class FakeClient:
    """Pretends to be the OpenAI client. Records calls, returns or raises what we set."""

    def __init__(self, result=None, error=None):
        self.calls = []
        self._result = result
        self._error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._result


def fake_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", SecretStr(FAKE_KEY))


def use_client(monkeypatch, client):
    monkeypatch.setattr(llm_service, "_make_client", lambda: client)


# --- original tests ---

def test_empty_question_is_rejected():
    with pytest.raises(LLMError) as info:
        ask("   ")
    assert info.value.kind == "invalid_input"


def test_too_long_question_is_rejected():
    with pytest.raises(LLMError) as info:
        ask("a" * (settings.max_question_chars + 1))
    assert info.value.kind == "invalid_input"


def test_missing_api_key_is_reported(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None)
    with pytest.raises(LLMError) as info:
        ask("Where is my order?")
    assert info.value.kind == "not_configured"


# --- new tests ---

def test_successful_answer_and_request_shape(monkeypatch, with_key):
    client = FakeClient(result=fake_response("  Hello! How can I help?  "))
    use_client(monkeypatch, client)

    answer = ask("  Hi there  ")

    assert answer == "Hello! How can I help?"
    assert len(client.calls) == 1
    sent = client.calls[0]
    assert sent["model"] == settings.openai_model
    assert sent["messages"][0]["role"] == "system"
    assert sent["messages"][1] == {"role": "user", "content": "Hi there"}


def test_bad_input_never_reaches_the_provider(monkeypatch, with_key):
    def must_not_be_called():
        raise AssertionError("OpenAI client was created for invalid input")

    monkeypatch.setattr(llm_service, "_make_client", must_not_be_called)
    for bad in ["", "   ", "a" * (settings.max_question_chars + 1)]:
        with pytest.raises(LLMError):
            ask(bad)


ERROR_CASES = [
    (lambda: AuthenticationError("bad key", response=_response(401), body=None), "auth"),
    (lambda: RateLimitError("slow down", response=_response(429), body=None), "rate_limit"),
    (lambda: APITimeoutError(request=_request()), "timeout"),
    (lambda: APIConnectionError(request=_request()), "connection"),
    (lambda: InternalServerError("oops", response=_response(500), body=None), "api_error"),
    (lambda: RuntimeError("boom"), "unexpected"),
]


@pytest.mark.parametrize("make_error, expected_kind", ERROR_CASES)
def test_provider_errors_are_mapped_safely(monkeypatch, with_key, make_error, expected_kind):
    use_client(monkeypatch, FakeClient(error=make_error()))

    with pytest.raises(LLMError) as info:
        ask("Where is my order?")

    assert info.value.kind == expected_kind
    assert info.value.user_message


def test_secrets_and_customer_text_are_not_logged(monkeypatch, with_key, caplog):
    caplog.set_level(logging.DEBUG)
    secret_question = "My phone number is 9999900000, where is order 12345?"
    leaky_error = RuntimeError(f"failure with {FAKE_KEY} and {secret_question}")
    use_client(monkeypatch, FakeClient(error=leaky_error))

    with pytest.raises(LLMError) as info:
        ask(secret_question)

    assert FAKE_KEY not in caplog.text
    assert "9999900000" not in caplog.text
    assert FAKE_KEY not in info.value.user_message
    assert "9999900000" not in info.value.user_message


def test_empty_choices_are_handled(monkeypatch, with_key):
    use_client(monkeypatch, FakeClient(result=SimpleNamespace(choices=[])))
    with pytest.raises(LLMError) as info:
        ask("Hello")
    assert info.value.kind == "empty_response"