"""The per-model semantic floor, calibrated against a real indexed PDF.

THE DEFECT THIS FILE EXISTS FOR.

A user indexed a 51-page Koenigsegg magazine (108 chunks) with EmbeddingGemma through
Ollama — every chunk embedded, 768 dimensions, nothing failed — and asked what the engine's
cubic capacity was. The document answers that twice: "a three-cylinder two-liter engine" on
page 17, and "Compression: 9.5:1 - bore: 95 mm - stroke: 93.5 mm" in the spec table on page
22. It never once writes "cc" or "cubic centimetre".

Semantic retrieval did its job perfectly: both passages came back as the top two hits, found
purely by meaning, with no vocabulary shared with the question at all. That is the entire
reason embeddings are offered in this product.

They were then thrown away, and the user was told **"cubic, centemeter do not appear anywhere
in this document"** — a word-overlap complaint about a passage that had just been located
without word overlap. `_has_sufficient_evidence` gated the semantic branch on a single global
`MIN_SEMANTIC_SIMILARITY = 0.62`, and EmbeddingGemma's correct matches land at 0.28–0.64.
One answerable question in twelve cleared it. The feature was installed, enabled, working,
and unreachable.

WHY A TABLE AND NOT A SMALLER NUMBER. Cosine has no absolute meaning across models; each
places its vectors in a differently-shaped space. A global floor is one model's scale
imposed on every other model's, and when it is wrong it fails silently — embeddings look
enabled and behave as if they were off. So the floor is per model, only models actually
measured get an entry, and everything else keeps the historical default.

THE FIGURES BELOW ARE MEASUREMENTS, NOT GUESSES. They were taken on 2026-08-13 against that
live 108-chunk index, and they are what the floor is set from. If someone changes the floor,
these tests are the argument they have to answer.
"""
import json

import pytest

import rag.service as rs
from models.rag import DocumentChunk
from rag.service import (
    DEFAULT_MIN_SEMANTIC_SIMILARITY,
    RetrievalResult,
    _has_sufficient_evidence,
    _SemanticRetrieval,
    semantic_similarity_floor,
)

pytestmark = pytest.mark.unit

GEMMA = "embeddinggemma"

# Measured against the live index. Each entry is (question, observed top cosine).
#
# The middle group is the one that makes this hard, and it is kept here rather than trimmed
# to a convenient story: same-domain questions the document does NOT answer overlap the
# answerable range outright — "what is the Bugatti Chiron top speed" scores 0.4479 against a
# supercar magazine, higher than four genuinely answerable questions. No absolute threshold
# separates those two groups and no relative one does either (a z-score against the corpus
# distribution was measured and overlaps just as badly). Similarity carries topical
# relatedness, not answer-presence.
ANSWERABLE = [
    ("I asked about the engine cubic centemeter", 0.3639),   # the reported question, typo and all
    ("engine cubic centimeter", 0.4488),
    ("what is the swept volume of the motor", 0.2974),
    ("how many litres is the petrol engine", 0.4114),
    ("what is the bore and stroke", 0.2821),                 # the lowest answerable observed
    ("how much does the powerplant weigh", 0.4666),
    ("how many cup holders", 0.3907),
    ("what is the curb weight", 0.3189),
    ("what is the battery capacity", 0.3685),
]
SAME_DOMAIN_BUT_ABSENT = [
    ("what is the Bugatti Chiron top speed", 0.4479),
    ("how much does a Ferrari SF90 cost", 0.3203),
    ("what is the Tesla Model S battery warranty", 0.2937),
    ("what tyre pressure should I use for a Toyota Corolla", 0.2514),
    ("how do I change the oil filter on my Honda Civic", 0.1688),
]
UNRELATED = [
    ("who is the president of France", 0.0851),
    ("what is the capital of Japan", 0.0834),
    ("how do I bake sourdough bread", 0.0628),
    ("explain quantum entanglement", 0.1197),
    ("what is the price of bitcoin today", 0.1414),          # the highest unrelated observed
]

# Verbatim from chunks 33 and 53 of the indexed document — the two passages that answer the
# question, and the ones semantic retrieval actually returned.
GEMERA_TWO_LITER = (
    "a three-cylinder two-liter engine that gives 400 Nm of torque from 1700 rpm and max "
    "torque of 600 Nm. These are staggering never-before achieved numbers for an engine of "
    "this size."
)
GEMERA_SPEC_TABLE = (
    "PROPULSION ICE Koenigsegg Tiny Friendly Giant Twin Turbo Freevalve 3-cylinder Internal "
    "Combustion Engine (ICE) with dry sump lubrication Compression: 9.5:1 - bore: 95 mm - "
    "stroke: 93.5 mm"
)


def _chunk(text: str, index: int) -> DocumentChunk:
    """A chunk carrying the same `terms_json` ingestion would have written for it."""
    return DocumentChunk(
        id=f"chunk-{index}",
        resource_id="gemera",
        file_id="gemera-pdf",
        chunk_index=index,
        source_name="gemera.pdf",
        modality="document",
        title="Gemera",
        content=text,
        contextual_content=text,
        terms_json=json.dumps(rs._term_counts(text)),
    )


def _gemera_results() -> list[RetrievalResult]:
    """What retrieval returned for the cubic-capacity question, in the order it returned it."""
    return [
        RetrievalResult(chunk=_chunk(GEMERA_SPEC_TABLE, 53), score=0.36, reason="semantic"),
        RetrievalResult(chunk=_chunk(GEMERA_TWO_LITER, 33), score=0.32, reason="semantic"),
    ]


class TestTheFloorIsPerModel:
    def test_a_measured_model_gets_its_measured_floor(self):
        assert semantic_similarity_floor(GEMMA) == 0.20

    @pytest.mark.parametrize(
        "tag", ["embeddinggemma", "embeddinggemma:300m", "embeddinggemma:300m-qat-q4_0", "EmbeddingGemma:latest"]
    )
    def test_a_quantized_tag_is_the_same_model(self, tag):
        """Ollama tags are `model:tag`. A table keyed on the full string misses every
        tagged variant and silently falls back to the default — which is precisely the
        failure the table was added to fix, reintroduced one level down."""
        assert semantic_similarity_floor(tag) == 0.20

    @pytest.mark.parametrize("model", ["text-embedding-3-small", "text-embedding-004", "mistral-embed", None, ""])
    def test_an_unmeasured_model_keeps_the_historical_default(self, model):
        """The safety property, and the reason this change cannot regress anyone.

        A model nobody has calibrated must not have a number invented for it: a too-low
        floor buys recall with confident wrong answers, which is the worse trade for a tool
        whose whole claim is that it only answers from your documents. openai, gemini and
        mistral therefore keep 0.62 and may well be under-firing in the same way — a known
        unmeasured gap, deliberately left rather than guessed at.
        """
        assert semantic_similarity_floor(model) == DEFAULT_MIN_SEMANTIC_SIMILARITY


class TestTheReportedQuestion:
    QUERY = "I asked about the engine cubic centemeter"

    def test_the_passages_that_answer_it_share_none_of_its_words(self):
        """The premise. If the vocabulary did overlap, the lexical branch would already
        have passed this and there would be nothing for embeddings to fix."""
        asked = set(rs._tokenize(self.QUERY))
        found = set(rs._tokenize(GEMERA_TWO_LITER)) | set(rs._tokenize(GEMERA_SPEC_TABLE))

        assert "cubic" not in found and "centemeter" not in found
        assert "centimeter" not in found and "cc" not in found
        # "engine" is the single shared word, which is what made lexical coverage 0.25 —
        # below the 0.35 the lexical branch needs, hence the refusal.
        assert asked & found == {"engine"}

    def test_it_is_refused_without_the_per_model_floor(self):
        """The bug, reproduced. At the old global floor the correct passages are in hand
        and discarded."""
        semantic = _SemanticRetrieval(
            results=_gemera_results(), top_similarity=0.3639, model="text-embedding-3-small"
        )

        assert _has_sufficient_evidence(self.QUERY, _gemera_results(), semantic) is False

    def test_it_is_answered_with_it(self):
        semantic = _SemanticRetrieval(results=_gemera_results(), top_similarity=0.3639, model=GEMMA)

        assert _has_sufficient_evidence(self.QUERY, _gemera_results(), semantic) is True

    def test_the_document_states_the_capacity_in_two_places(self):
        """Guards the fixture itself. If these strings ever stop being the answer, the
        test above is asserting something about text that no longer answers anything."""
        assert "two-liter" in GEMERA_TWO_LITER
        assert "bore: 95 mm" in GEMERA_SPEC_TABLE and "stroke: 93.5 mm" in GEMERA_SPEC_TABLE


class TestTheCalibration:
    """Every answerable question passes; every unrelated one is still refused."""

    @pytest.mark.parametrize("query,similarity", ANSWERABLE)
    def test_every_measured_answerable_question_passes(self, query, similarity):
        semantic = _SemanticRetrieval(results=_gemera_results(), top_similarity=similarity, model=GEMMA)
        assert _has_sufficient_evidence(query, _gemera_results(), semantic) is True

    @pytest.mark.parametrize("query,similarity", UNRELATED)
    def test_every_unrelated_question_is_still_refused(self, query, similarity):
        """The floor's actual job. These are the ones it separates cleanly."""
        semantic = _SemanticRetrieval(results=_gemera_results(), top_similarity=similarity, model=GEMMA)
        assert _has_sufficient_evidence(query, _gemera_results(), semantic) is False

    def test_the_floor_sits_in_the_gap_rather_than_hugging_either_class(self):
        floor = semantic_similarity_floor(GEMMA)
        worst_answerable = min(similarity for _, similarity in ANSWERABLE)
        best_unrelated = max(similarity for _, similarity in UNRELATED)

        assert best_unrelated < floor < worst_answerable, "the floor no longer separates the measured classes"
        # Margin on both sides, so neither a slightly harder question nor a slightly
        # closer piece of noise flips the outcome.
        assert floor - best_unrelated >= 0.05
        assert worst_answerable - floor >= 0.05

    def test_same_domain_questions_are_not_separable_and_that_is_documented(self):
        """The honest limit, pinned so nobody later 'fixes' the floor to catch these.

        Raising the floor above the near-miss group would take most answerable questions
        with it — they overlap. Judging whether a passage actually contains the fact is
        the grounded LLM's job, and it does it: measured live, "what is the Bugatti Chiron
        top speed" retrieves Gemera passages and returns "I do not have enough information
        to answer this question", while the cubic-capacity question returns "a
        three-cylinder, 2-liter engine" with citations.
        """
        best_near_miss = max(similarity for _, similarity in SAME_DOMAIN_BUT_ABSENT)
        worst_answerable = min(similarity for _, similarity in ANSWERABLE)

        assert best_near_miss > worst_answerable, (
            "the classes have separated — re-measure before tightening the floor, because "
            "this test's whole point is that they do not"
        )


class TestNothingElseMoved:
    def test_a_base_with_no_embeddings_is_unaffected(self):
        """Every base uploaded before per-document embeddings existed passes an inert
        semantic on every question, and must keep behaving exactly as it did."""
        results = _gemera_results()
        assert _has_sufficient_evidence("what is the bore and stroke", results, _SemanticRetrieval()) is True
        assert _has_sufficient_evidence("who is the president of France", results, _SemanticRetrieval()) is False

    def test_no_results_is_still_a_refusal_however_strong_the_similarity(self):
        """`if not results` comes first and stays first: a similarity score with nothing
        attached to it is not evidence."""
        semantic = _SemanticRetrieval(results=[], top_similarity=0.99, model=GEMMA)
        assert _has_sufficient_evidence("anything at all", [], semantic) is False
