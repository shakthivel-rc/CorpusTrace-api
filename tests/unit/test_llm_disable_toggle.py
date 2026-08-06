"""The off switch for LLM synthesis.

`llm_provider` and `llm_model` are query parameters on /chat/asks, so a UI-only toggle
would be a suggestion rather than a setting: anything still sending them would still get
synthesis. Someone who turns the LLM off to stop their documents reaching a third party
has to actually be right about that, so the check lives in `_plan_answer`.
"""
import json

import pytest

import rag.service as rag_service
from models.llm import LlmUserPreference
from models.rag import DocumentChunk
from models.resource import Resource
from services.llm_provider import llm_synthesis_enabled, set_user_preference

pytestmark = pytest.mark.unit

USER = "user-1"


def _seed_resource(db) -> Resource:
    resource = Resource(resource_name="Handbook", user_id=USER, upload_status=True)
    db.add(resource)
    db.flush()
    db.add(
        DocumentChunk(
            resource_id=resource.id,
            chunk_index=0,
            source_name="handbook.txt",
            modality="text",
            content="Employees receive twenty vacation days per year.",
            contextual_content="Vacation policy. Employees receive twenty vacation days per year.",
            terms_json=json.dumps({"vacation": 3, "days": 2, "employees": 1}),
        )
    )
    db.flush()
    return resource


class TestDefault:
    def test_enabled_when_no_preference_row_exists(self, db):
        # A user who has never opened settings must behave as before the column existed.
        assert llm_synthesis_enabled(db, USER) is True

    def test_enabled_by_default_on_a_new_row(self, db):
        db.add(LlmUserPreference(user_id=USER, provider="groq", model_id="llama-3.3-70b-versatile"))
        db.flush()
        assert llm_synthesis_enabled(db, USER) is True


class TestEnforcement:
    def test_planning_ignores_a_provider_when_synthesis_is_off(self, db, monkeypatch):
        resource = _seed_resource(db)
        db.add(
            LlmUserPreference(
                user_id=USER, provider="groq", model_id="llama-3.3-70b-versatile", llm_enabled=False
            )
        )
        db.flush()

        # Any LLM call at all would be a failure of the switch.
        def _boom(*args, **kwargs):
            raise AssertionError("the LLM must not be called when synthesis is disabled")

        monkeypatch.setattr(rag_service, "generate_grounded_answer", _boom)
        monkeypatch.setattr(rag_service, "stream_grounded_answer", _boom)

        plan = rag_service._plan_answer(
            db, USER, resource.id, "vacation days", None, "groq", "llama-3.3-70b-versatile"
        )
        assert plan.synthesize is False
        # The extractive answer is still produced — disabling the LLM is not disabling search.
        assert "vacation" in plan.answer.lower()

    def test_planning_synthesizes_when_the_switch_is_on(self, db):
        resource = _seed_resource(db)
        db.add(
            LlmUserPreference(
                user_id=USER, provider="groq", model_id="llama-3.3-70b-versatile", llm_enabled=True
            )
        )
        db.flush()
        plan = rag_service._plan_answer(
            db, USER, resource.id, "vacation days", None, "groq", "llama-3.3-70b-versatile"
        )
        assert plan.synthesize is True

    def test_answer_question_returns_the_extractive_answer_with_the_switch_off(self, db, monkeypatch):
        resource = _seed_resource(db)
        db.add(LlmUserPreference(user_id=USER, provider="groq", model_id="m", llm_enabled=False))
        db.flush()
        monkeypatch.setattr(
            rag_service,
            "generate_grounded_answer",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")),
        )
        result = rag_service.answer_question(db, USER, resource.id, "vacation days", None, "groq", "m")
        assert "vacation" in result.answer.lower()
        assert "LLM:" not in result.answer


class TestPersistence:
    def test_saving_a_model_alone_does_not_switch_synthesis_back_on(self, db):
        # The dropdown sends provider+model on every change. If that implied llm_enabled,
        # turning the LLM off then picking a model would silently undo the decision.
        db.add(LlmUserPreference(user_id=USER, provider="groq", model_id="llama-3.1-8b-instant", llm_enabled=False))
        db.commit()

        set_user_preference(db, USER, "groq", "llama-3.3-70b-versatile")

        assert llm_synthesis_enabled(db, USER) is False

    def test_the_flag_round_trips_and_is_serialized(self, db):
        db.add(LlmUserPreference(user_id=USER, provider="groq", model_id="llama-3.1-8b-instant"))
        db.commit()

        payload = set_user_preference(db, USER, "groq", "llama-3.1-8b-instant", llm_enabled=False)
        assert payload["llm_enabled"] is False
        assert llm_synthesis_enabled(db, USER) is False

        payload = set_user_preference(db, USER, "groq", "llama-3.1-8b-instant", llm_enabled=True)
        assert payload["llm_enabled"] is True
        assert llm_synthesis_enabled(db, USER) is True
