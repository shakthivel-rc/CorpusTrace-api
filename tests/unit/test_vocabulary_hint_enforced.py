"""The words that matched nothing must reach the reader, whatever the model writes.

`_retrieval_situation` puts them in the prompt, but a prompt is a request. Asked why
"piston ring size" returned nothing from a Triumph owner's handbook, a model replied
"the term 'piston ring size' isn't covered in the provided materials" — echoing the whole
query and losing the only fact that helps: *piston* matched nothing, while *ring* and
*size* both matched. Without that, a user cannot tell "this document has no pistons in it"
from "try different wording".

Same principle as `sanitize_conversational_reply`: what must reach the reader is enforced
in code.
"""
import pytest

import rag.service as rag
from rag.service import _guarded_reply, _vocabulary_hint

pytestmark = pytest.mark.unit


def _reply(monkeypatch, text: str):
    monkeypatch.setattr(
        rag,
        "generate_conversational_reply",
        lambda *args, **kwargs: text,
    )


class TestVocabularyHintIsEnforced:
    def test_names_the_dead_word_even_when_the_model_paraphrases_it_away(self, monkeypatch):
        # Verbatim shape of the reply that lost the information in production.
        _reply(monkeypatch, 'The term "piston ring size" isn\'t covered in the provided materials.')

        answer = _guarded_reply(
            None, "u1", "piston ring size", "Daytona", "openrouter", "some-model",
            deterministic="unused", situation="unused", unmatched=["piston"],
        )

        assert "**piston** does not appear anywhere in this document." in answer
        # The model's own wording is kept — the hint is added, not substituted.
        assert "isn't covered in the provided materials" in answer

    def test_says_nothing_extra_when_every_word_matched(self, monkeypatch):
        # An empty `unmatched` means the words were fine and the distribution was not.
        # Naming words here would misdirect the user, so the hint must stay silent.
        _reply(monkeypatch, "I could not find enough in the documents to answer that.")

        answer = _guarded_reply(
            None, "u1", "what is the torque figure", "Daytona", "openrouter", "some-model",
            deterministic="unused", situation="unused", unmatched=[],
        )

        assert answer == "I could not find enough in the documents to answer that."

    def test_lists_every_dead_word_with_plural_agreement(self, monkeypatch):
        _reply(monkeypatch, "That is not something the documents cover.")

        answer = _guarded_reply(
            None, "u1", "carburettor jetting", "Daytona", "openrouter", "some-model",
            deterministic="unused", situation="unused", unmatched=["carburettor", "jetting"],
        )

        assert "**carburettor**, **jetting** do not appear anywhere in this document." in answer

    def test_the_deterministic_path_already_carried_the_hint_and_still_does(self):
        # No provider configured: the caller's deterministic wording is returned untouched,
        # and it composes the hint itself. This asserts the two paths agree.
        deterministic = f"No answer.{_vocabulary_hint(['piston'])}"

        answer = _guarded_reply(
            None, "u1", "piston ring size", "Daytona", None, None,
            deterministic=deterministic, situation="unused", unmatched=["piston"],
        )

        assert answer == deterministic
        assert answer.count("**piston**") == 1

    def test_a_model_failure_still_falls_back_rather_than_raising(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("provider exploded")

        monkeypatch.setattr(rag, "generate_conversational_reply", boom)

        answer = _guarded_reply(
            None, "u1", "piston ring size", "Daytona", "openrouter", "some-model",
            deterministic="built-in wording", situation="unused", unmatched=["piston"],
        )

        assert answer == "built-in wording"
