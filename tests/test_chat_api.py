import pytest
from fastapi.testclient import TestClient

from app.api import chat as chat_module
from app.config import settings
from app.main import app
from app.services.llm_service import LLMError

client = TestClient(app)


def test_chat_success(monkeypatch):
    seen = {}

    def fake_ask(question):
        seen["question"] = question
        return "Hello!"

    monkeypatch.setattr(chat_module, "ask", fake_ask)
    response = client.post("/chat", json={"question": "Hi there"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Hello!"
    assert body["request_id"]
    assert response.headers["X-Request-ID"] == body["request_id"]
    assert seen["question"] == "Hi there"


def test_missing_question_field_is_rejected():
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_too_long_question_is_rejected_without_echo():
    long_question = "a" * (settings.max_question_chars + 1)
    response = client.post("/chat", json={"question": long_question})
    assert response.status_code == 422
    assert "aaaaaaaaaa" not in response.text


def test_whitespace_only_question_is_rejected():
    response = client.post("/chat", json={"question": "     "})
    assert response.status_code == 400


@pytest.mark.parametrize(
    "kind, status",
    [
        ("invalid_input", 400),
        ("not_configured", 503),
        ("auth", 503),
        ("rate_limit", 503),
        ("timeout", 504),
        ("connection", 503),
        ("api_error", 502),
        ("empty_response", 502),
        ("unexpected", 500),
    ],
)
def test_llm_errors_become_safe_http_errors(monkeypatch, kind, status):
    def failing_ask(question):
        raise LLMError(kind, "Safe message for the customer.")

    monkeypatch.setattr(chat_module, "ask", failing_ask)
    response = client.post("/chat", json={"question": "Hello"})

    assert response.status_code == status
    body = response.json()
    assert body["detail"] == "Safe message for the customer."
    assert body["request_id"]