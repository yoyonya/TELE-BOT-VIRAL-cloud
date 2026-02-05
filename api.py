# api.py

from query import ask as core_ask
from ux.ux_boat import ux_response


def handle_question(user_question: str) -> str:
    """
    Hlavní router dotazu.
    """
    answer = core_ask(user_question)

    if answer.startswith("NEDOLOŽENO"):
        return ux_response(user_question, answer)

    return answer
