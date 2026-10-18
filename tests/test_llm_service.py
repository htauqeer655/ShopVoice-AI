import pytest

from app.config import settings
from app.services.llm_service import LLMError, ask


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