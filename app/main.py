from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.config import settings

app = FastAPI(title=settings.app_name, version="0.1.0")
app.include_router(chat_router)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # The default response echoes the submitted text back. We don't.
    return JSONResponse(
        status_code=422,
        content={
            "detail": (
                'Invalid request. Send JSON like {"question": "..."} '
                f"with 1 to {settings.max_question_chars} characters."
            )
        },
    )


@app.get("/")
def welcome():
    return {"message": f"Welcome to {settings.app_name}", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "environment": settings.environment}