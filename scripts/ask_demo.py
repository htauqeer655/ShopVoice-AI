import sys

from app.services.llm_service import LLMError
from app.services.rag_service import answer_question


def main() -> None:
    question = " ".join(sys.argv[1:]) or input("Customer question: ")
    try:
        result = answer_question(question)
    except LLMError as error:
        print(f"[{error.kind}] {error.user_message}")
        return
    print(result.answer)
    print(f"(grounded: {result.grounded}; sources: {', '.join(result.sources) or 'none'})")


if __name__ == "__main__":
    main()