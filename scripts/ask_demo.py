import sys

from app.services.llm_service import LLMError, ask


def main() -> None:
    question = " ".join(sys.argv[1:]) or input("Customer question: ")
    try:
        print(ask(question))
    except LLMError as error:
        print(f"[{error.kind}] {error.user_message}")


if __name__ == "__main__":
    main()