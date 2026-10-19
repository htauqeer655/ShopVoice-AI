from pydantic import BaseModel, Field

from app.config import settings


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=settings.max_question_chars)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    grounded: bool
    request_id: str