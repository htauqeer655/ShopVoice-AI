import logging
import uuid

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from app.schemas import ChatRequest, ChatResponse
from app.services.llm_service import LLMError, ask

logger = logging.getLogger(__name__)
router = APIRouter()

STATUS_BY_KIND = {
    "invalid_input": 400,
    "not_configured": 503,
    "auth": 503,
    "rate_limit": 503,
    "timeout": 504,
    "connection": 503,
    "api_error": 502,
    "empty_response": 502,
    "unexpected": 500,
}


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, response: Response):
    request_id = uuid.uuid4().hex
    try:
        answer = ask(body.question)
    except LLMError as error:
        # Log the failure type and request ID only, never the question.
        logger.warning("chat failed kind=%s request_id=%s", error.kind, request_id)
        return JSONResponse(
            status_code=STATUS_BY_KIND.get(error.kind, 500),
            content={"detail": error.user_message, "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )
    response.headers["X-Request-ID"] = request_id
    return ChatResponse(answer=answer, request_id=request_id)