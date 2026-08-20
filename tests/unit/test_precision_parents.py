"""Unit tests for parent-chunk recovery — `rag/precision/parents.py`, stage 9 of the
high-precision pipeline.

This stage is the only place in the mode that assembles *text a user will read* out of rows
belonging to other chunks, and every way it can go wrong is quiet:

* **`None` and `""` are different answers.** A single-chunk document genuinely has no parent,
  and the stage says so with `None`. An empty string would travel downstream as "there is a
  parent and it is blank" — `PrecisionResult.to_dict` publishes `parent_context` verbatim, so
  the evidence panel would render an empty context block instead of omitting one.
* **Symmetry.** The stage exists so a child that matched precisely comes back with the text
  that makes it mean something. Walking outward from one side only would hand a mid-document
  child a paragraph of what came *after* it and nothing of what it was answering — which
  reads as perfectly plausible context and is half the story.
* **Which end of a truncated sibling survives.** Budget is characters, so long siblings get
  cut. The half nearest the match is the half that carries the meaning: a preceding sibling
  must keep its TAIL and a following one its HEAD. Reversing that trims away exactly the
  sentences adjoining the passage while still returning a full-looking context, so nothing
  downstream can detect it. The assertions below name the exact substrings for that reason.

Everything here is a pure unit: a local `FakeChunk` satisfies `types.ChunkLike`
structurally, so this file imports no model, no session and no `rag.service`.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag.precision.parents import recover_parent

pytestmark = pytest.mark.unit


@dataclass
class FakeChunk:
    """The two attributes `parents.py` reads off a chunk: `id` and `content`.

    A local dataclass rather than `models.rag.DocumentChunk` on purpose — the precision
    package is structurally typed so it can be exercised with no database at all, and a test
    that imported the model would quietly give that property up.
    """

    id: str
    content: str


def _family(*bodies: str) -> tuple[list[str], dict[str, FakeChunk]]:
    """A window of consecutive siblings: the `family` id list and its `by_id` map."""
    chunks = [FakeChunk(id=f"c{i}", content=body) for i, body in enumerate(bodies)]
    return [chunk.id for chunk in chunks], {chunk.id: chunk for chunk in chunks}


class TestNoParent:
    def test_single_member_family_returns_none_not_empty_string(self):
        # A one-chunk document has no parent. `None` says "this child stands alone"; ""
        # would say "the parent is blank", and `to_dict` publishes that difference.
        family, by_id = _family("the whole document is this one chunk")
        context, siblings = recover_parent(by_id["c0"], family, by_id, max_chars=2400)
        assert context is None
        assert context != ""
        assert siblings == ()

    def test_empty_family_returns_none(self):
        chunk = FakeChunk(id="orphan", content="body")
        assert recover_parent(chunk, [], {}, max_chars=2400) == (None, ())

    def test_chunk_absent_from_its_own_family_returns_none(self):
        # `family` comes from the index keyed by `parent_key`; a child that is not in the
        # list it was looked up under has no position to expand from, and guessing one
        # would attribute a neighbour's text to it.
        family, by_id = _family("first", "second", "third")
        stranger = FakeChunk(id="not-in-this-window", content="body")
        assert recover_parent(stranger, family, by_id, max_chars=2400) == (None, ())

    def test_budget_smaller_than_the_child_returns_none(self):
        # The child's own content is charged against `max_chars` first. If it does not fit
        # there is no room for context at all, and the honest answer is "no parent" rather
        # than a window that is only the child.
        family, by_id = _family("before", "x" * 500, "after")
        context, siblings = recover_parent(by_id["c1"], family, by_id, max_chars=100)
        assert context is None
        assert siblings == ()

    def test_budget_exactly_consumed_by_the_child_returns_none(self):
        family, by_id = _family("before", "child", "after")
        context, siblings = recover_parent(by_id["c1"], family, by_id, max_chars=len("child"))
        assert (context, siblings) == (None, ())

    def test_returns_none_when_every_sibling_is_unusable(self):
        # Nothing contributed, so there is no parent — not a context that is just the child.
        family = ["missing-a", "child", "missing-b"]
        child = FakeChunk(id="child", content="child body")
        context, siblings = recover_parent(child, family, {"child": child}, max_chars=2400)
        assert (context, siblings) == (None, ())


class TestSymmetricExpansion:
    def test_middle_child_is_expanded_from_both_sides(self):
        family, by_id = _family("zero", "one", "two", "three", "four")
        context, siblings = recover_parent(by_id["c2"], family, by_id, max_chars=2400)
        # Both neighbours are present: a one-sided walk would have produced "two three four"
        # or "zero one two" and looked entirely reasonable.
        assert set(siblings) == {"c0", "c1", "c3", "c4"}
        assert "one" in context and "three" in context

    def test_expansion_alternates_outward_but_reports_in_reading_order(self):
        # Two separate properties, and conflating them was a defect. The WALK alternates —
        # before, after, before, after — which is what makes a partial budget spend evenly on
        # both sides instead of exhausting one direction first. The REPORTED ids are sorted
        # back into reading order, because they are rendered beside the passage and
        # ("c1","c3","c0","c4") reads as a bug in any trace or provenance panel.
        family, by_id = _family("zero", "one", "two", "three", "four")
        _, siblings = recover_parent(by_id["c2"], family, by_id, max_chars=2400)
        assert siblings == ("c0", "c1", "c3", "c4")

    def test_a_partial_budget_still_alternates(self):
        # The alternation is observable where it matters: with room for two siblings only,
        # one is taken from each side rather than two from the left.
        family, by_id = _family("zero", "one", "two", "three", "four")
        _, siblings = recover_parent(
            by_id["c2"], family, by_id, max_chars=len("two") + len("one") + len("three")
        )
        assert siblings == ("c1", "c3")

    def test_a_budget_for_one_sibling_spends_it_on_the_preceding_one(self):
        # The alternation starts on the left, so the sentence immediately *before* the match
        # is the single most valuable piece of context and the one that is bought first.
        family, by_id = _family("preceding", "child", "following")
        context, siblings = recover_parent(
            by_id["c1"], family, by_id, max_chars=len("child") + len("preceding")
        )
        assert siblings == ("c0",)
        assert context == "preceding child"

    def test_child_at_the_start_takes_only_following_siblings(self):
        family, by_id = _family("zero", "one", "two")
        context, siblings = recover_parent(by_id["c0"], family, by_id, max_chars=2400)
        assert siblings == ("c1", "c2")
        assert context == "zero one two"

    def test_child_at_the_end_takes_only_preceding_siblings(self):
        family, by_id = _family("zero", "one", "two")
        context, siblings = recover_parent(by_id["c2"], family, by_id, max_chars=2400)
        # Reported in reading order, whichever direction the walk claimed them in.
        assert siblings == ("c0", "c1")
        assert context == "zero one two"

    def test_continues_on_one_side_once_the_other_is_exhausted(self):
        # Symmetry is a preference, not a quota: a child one from the edge should still
        # spend its remaining budget rather than stop when the short side runs out.
        family, by_id = _family("zero", "one", "two", "three", "four")
        _, siblings = recover_parent(by_id["c1"], family, by_id, max_chars=2400)
        assert siblings == ("c0", "c2", "c3", "c4")


class TestReadingOrder:
    def test_context_contains_the_child_between_its_neighbours(self):
        family, by_id = _family("ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO")
        context, _ = recover_parent(by_id["c2"], family, by_id, max_chars=2400)
        # `before` is collected walking *backwards*; if it were not reversed the context
        # would read "BRAVO ALPHA CHARLIE …" — same words, wrong document.
        positions = [context.index(word) for word in ("ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO")]
        assert positions == sorted(positions)

    def test_child_own_content_is_present_verbatim(self):
        # The child is the citation anchor; a parent context that dropped it would show the
        # reader everything around the passage that matched except the passage.
        family, by_id = _family("before", "THE MATCHED PASSAGE", "after")
        context, _ = recover_parent(by_id["c1"], family, by_id, max_chars=2400)
        assert "THE MATCHED PASSAGE" in context
        assert context == "before THE MATCHED PASSAGE after"

    def test_pieces_are_separated_and_the_result_is_stripped(self):
        family, by_id = _family("  before  ", "  child  ", "  after  ")
        context, _ = recover_parent(by_id["c1"], family, by_id, max_chars=2400)
        assert context == "before child after"


class TestTruncationKeepsTheTextNearestTheMatch:
    def test_preceding_sibling_keeps_its_tail(self):
        # 30 characters of sibling against a 10-character budget. The end of the preceding
        # chunk is what runs into the match, so that is what has to survive.
        preceding = "FARTHEST filler filler NEAREST"
        family, by_id = _family(preceding, "child")
        context, siblings = recover_parent(by_id["c1"], family, by_id, max_chars=len("child") + 10)
        assert siblings == ("c0",)
        assert context == "er NEAREST child"
        assert "NEAREST" in context
        assert "FARTHEST" not in context

    def test_following_sibling_keeps_its_head(self):
        following = "NEAREST filler filler FARTHEST"
        family, by_id = _family("child", following)
        context, siblings = recover_parent(by_id["c0"], family, by_id, max_chars=len("child") + 10)
        assert siblings == ("c1",)
        assert context == "child NEAREST fi"
        assert "NEAREST" in context
        assert "FARTHEST" not in context

    def test_a_truncated_sibling_spends_exactly_the_remaining_budget(self):
        family, by_id = _family("p" * 200, "child", "n" * 200)
        context, siblings = recover_parent(by_id["c1"], family, by_id, max_chars=len("child") + 40)
        # The first sibling taken swallows the whole remaining budget, which ends the walk —
        # so the trailing sibling contributes nothing and must not be reported as if it had.
        assert siblings == ("c0",)
        assert context == "p" * 40 + " child"


class TestCharacterBudget:
    def test_sibling_text_never_exceeds_max_chars_minus_the_child(self):
        family, by_id = _family(*[str(i) * 100 for i in range(9)])
        context, siblings = recover_parent(by_id["c4"], family, by_id, max_chars=400)
        # 100 for the child leaves 300, which is exactly three siblings; the separating
        # spaces sit outside the budget, one per sibling joined on.
        assert len(siblings) == 3
        assert len(context) <= 400 + len(siblings)

    def test_budget_stops_the_walk_before_the_family_is_exhausted(self):
        # A generous budget takes the whole window; a tight one over the same window must
        # take strictly less, or the budget is decorative.
        family, by_id = _family(*[str(i) * 100 for i in range(9)])
        _, generous = recover_parent(by_id["c4"], family, by_id, max_chars=20000)
        _, tight = recover_parent(by_id["c4"], family, by_id, max_chars=400)
        assert len(generous) == 8
        assert len(tight) < len(generous)

    def test_budget_counts_characters_rather_than_siblings(self):
        # The point of a character budget: three 20-character siblings are not the same
        # amount of context as three 400-character ones, and the same `max_chars` has to
        # buy more of the short ones.
        short_family, short_by_id = _family(*["tiny" for _ in range(9)])
        long_family, long_by_id = _family(*["x" * 400 for _ in range(9)])
        _, short_siblings = recover_parent(short_by_id["c4"], short_family, short_by_id, max_chars=900)
        _, long_siblings = recover_parent(long_by_id["c4"], long_family, long_by_id, max_chars=900)
        assert len(short_siblings) > len(long_siblings)


class TestUnusableSiblingsAreSkipped:
    def test_a_family_member_missing_from_by_id_is_skipped(self):
        # `family` is a list of ids and `by_id` holds only the rows this request loaded; a
        # KeyError here would take down the whole answer over one absent neighbour.
        family = ["gone-a", "c0", "gone-b", "c1"]
        by_id = {"c0": FakeChunk(id="c0", content="child"), "c1": FakeChunk(id="c1", content="real")}
        context, siblings = recover_parent(by_id["c0"], family, by_id, max_chars=2400)
        assert siblings == ("c1",)
        assert context == "child real"

    def test_a_sibling_with_only_whitespace_is_skipped(self):
        # An empty neighbour contributes no meaning, and admitting it would spend a join on
        # nothing and report an id that added no context.
        family, by_id = _family("   ", "child", "\n\t ", "real")
        context, siblings = recover_parent(by_id["c1"], family, by_id, max_chars=2400)
        assert siblings == ("c3",)
        assert context == "child real"

    def test_skipped_siblings_do_not_consume_budget(self):
        # Whitespace and missing rows must not be charged, or an unlucky neighbour silently
        # halves the context a match comes back with.
        family = ["missing", "c1", "c2", "c3"]
        by_id = {
            "c1": FakeChunk(id="c1", content="   "),
            "c2": FakeChunk(id="c2", content="child"),
            "c3": FakeChunk(id="c3", content="y" * 100),
        }
        context, siblings = recover_parent(by_id["c2"], family, by_id, max_chars=len("child") + 100)
        assert siblings == ("c3",)
        assert context == "child " + "y" * 100
