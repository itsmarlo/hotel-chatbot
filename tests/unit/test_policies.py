from pathlib import Path

from hotel_assistant.rag.policies import PolicyRetriever
from hotel_assistant.services.interpret import booking_changes, language_of


def test_policy_retrieval_returns_authored_answer_and_source() -> None:
    path = Path(__file__).parents[2] / "data" / "knowledge_base" / "policies.json"
    retriever = PolicyRetriever(path)
    result = retriever.retrieve("When does the generator run?", "en")
    assert result is not None
    assert result.source == "generator"
    assert "18:00" in result.answer
    assert retriever.retrieve("What is the generator schedule?", "de") is not None


def test_policy_retrieval_declines_unsupported_query() -> None:
    path = Path(__file__).parents[2] / "data" / "knowledge_base" / "policies.json"
    retriever = PolicyRetriever(path)
    assert retriever.retrieve("Do you offer scuba diving lessons?") is None


def test_multilingual_language_detection_and_name_boundary() -> None:
    assert language_of("Quels papiers dois-je présenter ?") == "fr"
    assert language_of("Ich heiße Alex Kim") == "de"
    assert booking_changes("My name is Alex Kim and I need a room")["guest_name"] == "Alex Kim"
