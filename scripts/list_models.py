from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
)

from app.config import settings


def main() -> None:
    if settings.openai_api_key is None:
        print("No OPENAI_API_KEY found. Check your .env file.")
        return

    client = OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.openai_timeout_seconds,
    )

    try:
        models = client.models.list()
    except AuthenticationError:
        print("OpenAI rejected the key. Check that it is correct and active.")
        return
    except APITimeoutError:
        print("The request timed out. Try again.")
        return
    except APIConnectionError:
        print("Could not reach OpenAI. Check your internet connection.")
        return
    except APIStatusError as error:
        print(f"OpenAI returned an error (status {error.status_code}).")
        return

    ids = sorted(model.id for model in models)
    chat_models = [m for m in ids if m.startswith("gpt")]
    embedding_models = [m for m in ids if "embedding" in m]

    print("Chat models:")
    for name in chat_models:
        print("  ", name)
    print("Embedding models:")
    for name in embedding_models:
        print("  ", name)


if __name__ == "__main__":
    main()