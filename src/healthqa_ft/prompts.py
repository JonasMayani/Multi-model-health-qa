from __future__ import annotations


def system_prompt(language_name: str) -> str:
    return (
        "You are a careful health question-answering assistant. "
        f"Answer only in {language_name}. "
        "Preserve medical accuracy, avoid unsupported facts, and give a concise direct answer."
    )


def user_prompt(question: str, language_name: str) -> str:
    return f"Answer this health question in {language_name}:\n{question}"


def plain_causal_prompt(question: str, language_name: str) -> str:
    return (
        f"System: {system_prompt(language_name)}\n"
        f"User: {user_prompt(question, language_name)}\n"
        "Assistant:"
    )


def seq2seq_input(question: str, language_name: str) -> str:
    return f"Answer this health question in {language_name}: {question}"

