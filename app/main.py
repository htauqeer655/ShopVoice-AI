from fastapi import FastAPI

from app.config import settings

app = FastAPI(title=settings.app_name, version="0.1.0")


@app.get("/")
def welcome():
    return {"message": f"Welcome to {settings.app_name}", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "environment": settings.environment}