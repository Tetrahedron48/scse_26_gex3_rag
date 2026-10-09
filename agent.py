from __future__ import annotations

import sys

from rag import answer_service_desk_question


DEFAULT_QUESTION = "What exact phone number should I call for the University IT Service Desk?"


def main() -> None:
    question = DEFAULT_QUESTION
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])

    answer = answer_service_desk_question(question)
    print(answer)


if __name__ == "__main__":
    main()
