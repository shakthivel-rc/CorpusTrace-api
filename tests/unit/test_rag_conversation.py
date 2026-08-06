"""Small talk must not be answered by document retrieval.

"Hi" shares no vocabulary with an uploaded document, so lexical retrieval scored 0 and the
user got "I do not have enough information in the selected uploaded documents to answer this
question." — which reads as a broken assistant. Greetings are now answered conversationally
BEFORE retrieval (and before any LLM call, so they cost no tokens), while anything that is
actually a document question still goes through the normal path.
"""
import json

import pytest

import rag.service as rag_service
from models.rag import DocumentChunk
from models.resource import Resource
from rag.service import _conversational_reply, _out_of_scope_reply, answer_question

pytestmark = pytest.mark.unit


def _seed(db, user_id: str) -> Resource:
    resource = Resource(resource_name="ISO", user_id=user_id, upload_status=True)
    db.add(resource)
    db.flush()
    db.add(
        DocumentChunk(
            resource_id=resource.id,
            chunk_index=0,
            source_name="iso27001.pdf",
            modality="pdf",
            title="Access control",
            content="Clause 5 requires rules to control physical and logical access to information.",
            contextual_content="Access control. Clause 5 requires rules to control access to information.",
            terms_json=json.dumps({"access": 3, "control": 2, "clause": 1, "information": 1}),
        )
    )
    db.commit()
    return resource


class TestConversationalReply:
    @pytest.mark.parametrize(
        "message",
        ["hi", "Hi", "HI!", "hello", "Hey there", "  hi  ", "good morning", "Hello!!!", "hola"],
    )
    def test_greetings_are_recognized(self, message):
        assert _conversational_reply(message, "ISO") is not None

    def test_greeting_reply_orients_the_user_to_the_knowledge_base(self):
        reply = _conversational_reply("hi", "ISO")
        assert "Hello" in reply
        assert "ISO" in reply

    @pytest.mark.parametrize("message", ["thanks", "thank you", "ok"])
    def test_thanks_is_acknowledged(self, message):
        assert "welcome" in _conversational_reply(message, "ISO").lower()

    @pytest.mark.parametrize("message", ["bye", "goodbye", "good night"])
    def test_farewells_are_acknowledged(self, message):
        assert "Goodbye" in _conversational_reply(message, "ISO")

    @pytest.mark.parametrize("message", ["help", "what can you do", "who are you"])
    def test_capability_questions_explain_the_assistant(self, message):
        reply = _conversational_reply(message, "ISO")
        assert "NexaRAG" in reply and "cite" in reply

    @pytest.mark.parametrize(
        "message",
        [
            "hi, what does clause 5 require?",
            "hello can you summarize the access control section",
            "what are the requirements for access control?",
            "good morning, list the annex A controls please",
        ],
    )
    def test_real_questions_are_never_swallowed_as_small_talk(self, message):
        # The whole message must be small talk — a greeting prefix must not hijack a question.
        assert _conversational_reply(message, "ISO") is None

    def test_empty_query_is_not_small_talk(self):
        assert _conversational_reply("", "ISO") is None


class TestOutOfScopeQuestions:
    """Lexical retrieval matches the WORD "date" inside a document and returns
    confident-looking evidence for "what is the date and time now" — five citations of
    unrelated clauses. Those questions are intercepted instead."""

    @pytest.mark.parametrize(
        "message",
        [
            "what is date and time now",
            "what is the date and time now",
            "what is the time now",
            "todays date",
            "what day is it today",
            "current time",
            "what is the weather",
            "weather today",
        ],
    )
    def test_realtime_questions_are_declined_clearly(self, message):
        reply = _out_of_scope_reply(message)
        assert reply is not None, message
        assert "real-time" in reply

    @pytest.mark.parametrize(
        "message",
        [
            "what is the publication date of this standard?",
            "what is the current revision of the document?",
            "which clause covers time synchronisation of clocks?",
            "what does the standard require about the date of records?",
            "what are the requirements for access control?",
            "what is the effective date and version of ISO 27001?",
        ],
    )
    def test_document_questions_about_dates_still_retrieve(self, message):
        # "date" and "time" are legitimate document topics — these must NOT be intercepted.
        assert _out_of_scope_reply(message) is None, message

    def test_long_questions_are_never_intercepted(self):
        assert _out_of_scope_reply("tell me what the organization should do now to keep time records accurate") is None


class TestAnswerQuestionSmallTalk:
    def test_greeting_returns_a_conversational_answer_without_citations(self, db):
        resource = _seed(db, "user-greet")
        result = answer_question(db, "user-greet", resource.id, "Hi", None)
        assert "Hello" in result.answer
        assert "do not have enough information" not in result.answer
        assert result.citations == []

    def test_greeting_never_calls_the_llm(self, db, monkeypatch):
        resource = _seed(db, "user-greet-2")

        def fail(*args, **kwargs):
            raise AssertionError("small talk must not reach the LLM provider")

        monkeypatch.setattr(rag_service, "generate_grounded_answer", fail)
        result = answer_question(db, "user-greet-2", resource.id, "hello", None, "groq", "llama-3.3-70b-versatile")
        assert "Hello" in result.answer

    def test_a_real_question_still_retrieves_and_cites(self, db):
        resource = _seed(db, "user-ask")
        result = answer_question(db, "user-ask", resource.id, "What does clause 5 require for access control?", None)
        assert result.citations
        assert "access" in result.answer.lower()

    def test_an_unmatched_question_explains_how_to_rephrase(self, db):
        resource = _seed(db, "user-miss")
        result = answer_question(db, "user-miss", resource.id, "What is the warranty period for turbines?", None)
        assert "ISO" in result.answer
        assert "rephrasing" in result.answer
        assert result.citations == []
