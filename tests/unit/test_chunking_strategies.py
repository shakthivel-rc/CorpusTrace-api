"""Unit tests for the per-document chunking strategies — pure text arithmetic, no database.

Three properties in this module are worth more than the rest of it put together, and all
three fail *silently* in production:

* **Compatibility.** Every chunk in every existing knowledge base was produced by
  `rag.service._split_into_chunk_spans` at 1200/180. `split_spans` under the default config
  has to reproduce it byte for byte, or shipping this feature quietly re-chunks bases nobody
  asked to re-index — and since retrieval scores `contextual_content`, that changes which
  answers a user gets from documents they never touched.
* **Span exactness.** Every strategy yields `(text, start, end)` into the *normalized*
  document text, and the source-evidence view turns those offsets into a page and a
  highlight. An off-by-one here does not raise; it highlights the wrong paragraph while
  looking completely convincing (CLAUDE.md §20).
* **Termination.** `_pack` walks backwards to build its overlap. If that walk can ever fail
  to advance, an upload does not fail — it pins a worker thread forever and the progress bar
  stops moving.

Everything is exercised through the real producers (`_join_pages`, `_normalize_with_breaks`)
rather than hand-written spans, so a change to how extraction records structure shows up
here instead of at ingestion time.
"""
import inspect
import random
import time

import pytest

from rag import chunking
from rag.chunking import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONFIG,
    DEFAULT_OVERLAP,
    DEFAULT_STRATEGY,
    MAX_CHUNK_SIZE,
    MAX_OVERLAP_RATIO,
    MIN_CHUNK_SIZE,
    STRATEGY_CHARACTER,
    STRATEGY_PAGE,
    STRATEGY_PARAGRAPH,
    STRATEGY_SENTENCE,
    IndexingConfig,
    describe_effective,
    normalize_config,
    recommendation_for,
    split_spans,
)
from rag.service import (
    SUPPORTED_RAG_MODES,
    _join_pages,
    _normalize_whitespace,
    _normalize_with_breaks,
    _pages_for_span,
    _split_into_chunk_spans,
)

pytestmark = pytest.mark.unit

# Every (size, overlap) pair below survives `normalize_config` unchanged, so a failure here
# is the chunker's fault and not the validator's. The extremes are the documented bounds.
SIZE_OVERLAP_PAIRS = [(200, 0), (200, 100), (600, 180), (1200, 180), (2000, 300), (4000, 2000)]

WORDS = [
    "valve", "clearance", "torque", "inspection", "bracket", "coolant", "gasket",
    "assembly", "procedure", "the", "and", "of", "cold", "engine", "cover", "Fig",
]


def _random_document(rng: random.Random, sentences: int) -> str:
    """Prose with ragged whitespace and mixed terminators.

    The shape matters: the character chunker backs up to the last ". " past the midpoint, so
    a document has to contain sentence ends at unpredictable offsets for the compatibility
    comparison to actually exercise that branch rather than the hard cut.
    """
    parts = []
    for _ in range(sentences):
        body = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 25)))
        parts.append(body + rng.choice([". ", ". ", ". ", "! ", "? ", ".  ", ".\n", "\n\n"]))
    return "".join(parts)


def _config(strategy: str, size: int, overlap: int) -> IndexingConfig:
    return normalize_config({"strategy": strategy, "chunk_size": size, "overlap": overlap})


@pytest.fixture()
def pdf_like():
    """A PDF-shaped document: real page spans, no paragraph structure (pypdf gives none)."""
    pages = [
        (1, _normalize_whitespace("Scope and purpose. " * 30)),
        (2, _normalize_whitespace("Valve clearance is checked cold. " * 120)),
        (3, _normalize_whitespace("Torque the cover to ten newton metres. " * 4)),
    ]
    text, spans = _join_pages(pages)
    return text, spans


@pytest.fixture()
def text_like():
    """A .txt-shaped document: paragraph breaks survived extraction, no pages."""
    raw = "\n\n".join(
        f"Paragraph {index} covers the inspection procedure in some detail. "
        + "Supporting sentence about the bracket and the gasket. " * (index % 5 + 1)
        for index in range(12)
    )
    return _normalize_with_breaks(raw)


class TestCompatibilityWithTheOriginalChunker:
    """The guarantee that existing knowledge bases were not silently re-chunked."""

    def test_the_default_config_still_carries_the_historical_constants(self):
        # `_split_into_chunk_spans`'s defaults are what every stored chunk was cut with.
        # If DEFAULT_CHUNK_SIZE or DEFAULT_OVERLAP moves, every *new* upload disagrees with
        # every *old* one and nothing anywhere raises.
        signature = inspect.signature(_split_into_chunk_spans)
        assert DEFAULT_CHUNK_SIZE == signature.parameters["max_chars"].default == 1200
        assert DEFAULT_OVERLAP == signature.parameters["overlap"].default == 180
        assert DEFAULT_STRATEGY == STRATEGY_CHARACTER
        assert DEFAULT_CONFIG == IndexingConfig(
            strategy=STRATEGY_CHARACTER, chunk_size=1200, overlap=180
        )

    def test_default_config_reproduces_the_original_chunker_on_randomized_documents(self):
        """The single most important assertion in this file.

        Byte-identical output, spans included, over documents of every length from
        sub-chunk to many-chunk. If this fails, re-indexing an old base produces different
        chunks — different text, different offsets, different retrieval scores.
        """
        rng = random.Random(20260805)
        multi_chunk = 0

        for _ in range(40):
            raw = _random_document(rng, rng.randint(1, 120))
            normalized = _normalize_whitespace(raw)

            original = list(_split_into_chunk_spans(raw))
            ported = split_spans(normalized, DEFAULT_CONFIG)

            assert ported == original, f"diverged on a {len(normalized)}-char document"
            multi_chunk += len(ported) > 1

        # A comparison that only ever saw one-chunk documents would prove nothing about the
        # sentence-boundary backup or the overlap arithmetic.
        assert multi_chunk >= 10, "the corpus must contain documents that actually split"

    @pytest.mark.parametrize(
        "raw",
        [
            "hello world",
            "  padded on both sides  ",
            "A" * 1200,  # exactly at the limit — one chunk, no split
            "A" * 1201,  # one character over — the first real split
            "A" * 700 + ". " + "B" * 700,  # boundary past the midpoint, so it is used
            "A" * 100 + ". " + "B" * 1500,  # boundary before the midpoint, so it is not
            # Exactly on the midpoint: the comparison is `>`, not `>=`, so this boundary is
            # NOT taken. One character either way here re-cuts every document whose sentence
            # happens to end on the midpoint, and randomized prose almost never hits it.
            "A" * 600 + ". " + "B" * 1500,
            "word " * 500,
            "no sentence terminator anywhere in this document " * 60,
        ],
        ids=[
            "short", "padded", "at-limit", "one-over", "boundary-used",
            "boundary-ignored", "boundary-exactly-on-midpoint", "repeated-word",
            "no-terminator",
        ],
    )
    def test_the_boundary_shapes_the_old_chunker_pinned_still_agree(self, raw):
        # The named edge cases from test_rag_chunking.py / test_rag_positions.py, re-asserted
        # against the new entry point so the port cannot regress one of them in isolation.
        assert split_spans(_normalize_whitespace(raw), DEFAULT_CONFIG) == list(
            _split_into_chunk_spans(raw)
        )

    def test_page_and_paragraph_structure_are_ignored_by_the_default_config(self, pdf_like):
        """Structure must not change the default's output.

        `ingest_file` now always passes page spans and paragraph breaks. A document indexed
        before this feature had none, so if the character strategy consulted them, the same
        PDF would chunk differently after an upgrade.
        """
        text, spans = pdf_like
        assert split_spans(text, DEFAULT_CONFIG, spans, [5, 90]) == split_spans(
            text, DEFAULT_CONFIG
        )


class TestSpanExactness:
    """`normalized[start:end] == text` for every chunk of every strategy."""

    @pytest.mark.parametrize("size,overlap", SIZE_OVERLAP_PAIRS)
    @pytest.mark.parametrize(
        "strategy", [STRATEGY_CHARACTER, STRATEGY_SENTENCE, STRATEGY_PARAGRAPH, STRATEGY_PAGE]
    )
    def test_a_pdf_shaped_document_slices_every_chunk_back_out(
        self, pdf_like, strategy, size, overlap
    ):
        # The evidence panel slices the document with these offsets. Drift by one character
        # and it highlights the wrong words with full confidence.
        text, spans = pdf_like
        chunks = split_spans(text, _config(strategy, size, overlap), spans, None)

        assert chunks, "a three-page document must produce chunks under every strategy"
        for chunk, start, end in chunks:
            assert text[start:end] == chunk
            assert 0 <= start < end <= len(text)

    @pytest.mark.parametrize("size,overlap", SIZE_OVERLAP_PAIRS)
    @pytest.mark.parametrize(
        "strategy", [STRATEGY_CHARACTER, STRATEGY_SENTENCE, STRATEGY_PARAGRAPH, STRATEGY_PAGE]
    )
    def test_a_text_shaped_document_slices_every_chunk_back_out(
        self, text_like, strategy, size, overlap
    ):
        text, breaks = text_like
        chunks = split_spans(text, _config(strategy, size, overlap), None, breaks)

        assert chunks
        for chunk, start, end in chunks:
            assert text[start:end] == chunk
            assert 0 <= start < end <= len(text)

    def test_a_boundary_landing_on_whitespace_still_yields_an_exact_span(self):
        """The packer strips its piece and then moves the offset by what it stripped.

        `_normalize_with_breaks` currently records the offset *after* the joining space, so
        this pairing is invisible in normal use — which is exactly why it needs pinning. If
        that ever records the offset *of* the space instead, without this every paragraph
        chunk's span silently shifts one character and the evidence view highlights from the
        space before the passage to one character short of its end.
        """
        first = " ".join(["alpha"] * 30)
        second = " ".join(["beta"] * 30)
        text = first + " " + second
        breaks = [len(first)]  # points AT the joining space, not past it

        # The budget has to force one chunk per unit, or the packer joins them and the
        # second unit's leading space never reaches the offset arithmetic.
        assert len(first) + len(second) > MIN_CHUNK_SIZE
        chunks = split_spans(text, _config(STRATEGY_PARAGRAPH, MIN_CHUNK_SIZE, 0), None, breaks)

        assert [chunk for chunk, _, _ in chunks] == [first, second]
        for chunk, start, end in chunks:
            assert text[start:end] == chunk
            assert chunk == chunk.strip()

    @pytest.mark.parametrize(
        "strategy", [STRATEGY_CHARACTER, STRATEGY_SENTENCE, STRATEGY_PARAGRAPH, STRATEGY_PAGE]
    )
    def test_no_chunk_starts_or_ends_on_whitespace(self, pdf_like, strategy):
        """The offset is computed after `.strip()` on purpose.

        A span that included the space the slice happened to land on would make the
        highlight start one word early on roughly half the chunks in a document.
        """
        text, spans = pdf_like
        for chunk, _start, _end in split_spans(text, _config(strategy, 600, 180), spans, None):
            assert chunk == chunk.strip()
            assert chunk, "an all-whitespace chunk must never be emitted"

    @pytest.mark.parametrize(
        "strategy", [STRATEGY_CHARACTER, STRATEGY_SENTENCE, STRATEGY_PARAGRAPH, STRATEGY_PAGE]
    )
    def test_chunks_advance_and_lose_no_text(self, pdf_like, strategy):
        """No chunk may go backwards, and nothing but whitespace may fall between two.

        A hole containing real characters is text that no question can ever retrieve — it
        was uploaded, it is on disk, and it is in no chunk. The whitespace exception is the
        single space `_join_pages` puts between two pages, which by construction belongs to
        neither.
        """
        text, spans = pdf_like
        chunks = split_spans(text, _config(strategy, 600, 180), spans, None)

        assert chunks[0][1] == 0, "the first chunk must start at the first character"
        for (_, previous_start, previous_end), (_, start, _end) in zip(chunks, chunks[1:]):
            assert start > previous_start, "the window must advance"
            if start > previous_end:
                assert not text[previous_end:start].strip(), "real text fell between chunks"


class TestSizeIsRespected:
    @pytest.mark.parametrize("size,overlap", SIZE_OVERLAP_PAIRS)
    @pytest.mark.parametrize(
        "strategy", [STRATEGY_CHARACTER, STRATEGY_SENTENCE, STRATEGY_PARAGRAPH, STRATEGY_PAGE]
    )
    def test_no_chunk_exceeds_the_configured_size(self, pdf_like, strategy, size, overlap):
        """The size setting is a budget, not a hint.

        An over-budget chunk is not cosmetic: five of them are pasted into the prompt, and
        on the small free-tier models this app targets that is how a context window
        overflows and the answer silently loses its last citations.
        """
        text, spans = pdf_like
        for chunk, _start, _end in split_spans(text, _config(strategy, size, overlap), spans, None):
            assert len(chunk) <= size

    def test_one_unsplittable_unit_is_handed_to_the_character_splitter(self):
        # A 6000-character "sentence" (no terminator) cannot be packed whole. Refusing to
        # split it would emit a chunk of unbounded size and break the budget above.
        text = _normalize_whitespace("clearance " * 600)
        chunks = split_spans(text, _config(STRATEGY_SENTENCE, 600, 180))

        assert len(chunks) > 1
        assert all(len(chunk) <= 600 for chunk, _, _ in chunks)
        assert all(text[start:end] == chunk for chunk, start, end in chunks)

    def test_an_oversized_page_is_split_inside_its_own_page(self, pdf_like):
        """Sub-splitting must stay within the page it came from.

        Page chunks are the one kind that can promise a citation never straddles a page
        break; a sub-chunk whose offsets leaked into the next page would quietly break that.
        """
        text, spans = pdf_like
        chunks = split_spans(text, _config(STRATEGY_PAGE, 600, 180), spans, None)

        # Page 2 is far over budget, so it must contribute several chunks — otherwise this
        # test would pass on a splitter that simply emitted one over-budget chunk per page.
        assert len(chunks) > len(spans)
        for _chunk, start, end in chunks:
            page_start, page_end = _pages_for_span(spans, start, end)
            assert page_start is not None and page_start == page_end


class TestTermination:
    """Adversarial input must return, and return quickly."""

    # A true infinite loop hangs the suite outright, which is its own loud signal. These
    # bounds catch the near miss: a packer that advances by one unit per chunk still returns,
    # but turns a 60 KB upload into tens of thousands of chunks and minutes of DB writes.
    BUDGET_SECONDS = 5.0

    @pytest.mark.parametrize("size,overlap", SIZE_OVERLAP_PAIRS)
    @pytest.mark.parametrize(
        "strategy", [STRATEGY_CHARACTER, STRATEGY_SENTENCE, STRATEGY_PARAGRAPH, STRATEGY_PAGE]
    )
    def test_one_enormous_sentence_terminates(self, strategy, size, overlap):
        # No terminator anywhere: the sentence splitter yields a single 60 000-char unit,
        # which is exactly the shape `_pack`'s `next_index > index` guarantee exists for.
        text = "A" * 60000
        started = time.perf_counter()
        chunks = split_spans(text, _config(strategy, size, overlap))
        elapsed = time.perf_counter() - started

        assert elapsed < self.BUDGET_SECONDS
        assert len(chunks) < 2 * len(text) // size + 10, "chunk count blew past the sane bound"
        assert all(len(chunk) <= size for chunk, _, _ in chunks)

    @pytest.mark.parametrize("size", [200, 600, 1200])
    def test_many_tiny_sentences_at_maximum_overlap_terminate(self, size):
        """The worst case for the overlap walk-back: overlap at its cap, units small enough
        that the walk can step over dozens of them looking for coverage."""
        text = _normalize_whitespace("ab. " * 8000)
        config = _config(STRATEGY_SENTENCE, size, int(size * MAX_OVERLAP_RATIO))
        assert config.overlap == int(size * MAX_OVERLAP_RATIO), "the cap must be reachable"

        started = time.perf_counter()
        chunks = split_spans(text, config)
        elapsed = time.perf_counter() - started

        assert elapsed < self.BUDGET_SECONDS
        assert 0 < len(chunks) < 4 * len(text) // size

    @pytest.mark.parametrize(
        "strategy", [STRATEGY_CHARACTER, STRATEGY_SENTENCE, STRATEGY_PARAGRAPH, STRATEGY_PAGE]
    )
    def test_empty_text_produces_nothing(self, strategy):
        # An empty document must not index a phantom empty chunk: it would score 0 on every
        # question but still occupy a citation slot.
        assert split_spans("", _config(strategy, 1200, 180)) == []
        assert split_spans("", _config(strategy, 1200, 180), [(1, 0, 0)], [0]) == []

    @pytest.mark.parametrize(
        "strategy", [STRATEGY_CHARACTER, STRATEGY_SENTENCE, STRATEGY_PARAGRAPH, STRATEGY_PAGE]
    )
    def test_text_shorter_than_one_chunk_is_a_single_span_covering_everything(self, strategy):
        text = "Valve clearance is checked cold."
        assert split_spans(text, _config(strategy, 1200, 180)) == [(text, 0, len(text))]

    def test_a_single_whitespace_run_produces_nothing_rather_than_an_empty_chunk(self):
        # `_normalize_whitespace` would have removed this, but page spans are built before
        # the packer sees them and a blank page reaches `_split_by_pages` intact.
        text = "alpha   beta"
        assert split_spans(text, _config(STRATEGY_PAGE, 1200, 180), [(1, 5, 8)], None) == []


class TestNormalizeConfig:
    """Client input is clamped, never rejected — one bad slider must not fail a batch."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ({"chunk_size": 1}, MIN_CHUNK_SIZE),
            ({"chunk_size": 0}, MIN_CHUNK_SIZE),
            ({"chunk_size": -900}, MIN_CHUNK_SIZE),
            ({"chunk_size": 99999}, MAX_CHUNK_SIZE),
            ({"chunk_size": MIN_CHUNK_SIZE}, MIN_CHUNK_SIZE),
            ({"chunk_size": MAX_CHUNK_SIZE}, MAX_CHUNK_SIZE),
            ({"chunk_size": "abc"}, DEFAULT_CHUNK_SIZE),
            ({"chunk_size": None}, DEFAULT_CHUNK_SIZE),
            ({}, DEFAULT_CHUNK_SIZE),
        ],
    )
    def test_chunk_size_is_clamped_into_the_documented_bounds(self, raw, expected):
        # Below the floor a chunk carries too few terms to score; above the ceiling one
        # chunk dominates the prompt. Both are silent quality failures, not errors.
        assert normalize_config(raw).chunk_size == expected

    def test_an_unknown_strategy_falls_back_to_the_default(self):
        # The same forgiveness `normalize_rag_mode` applies to `rag_type`. A typo from an
        # older client must index the document, not fail the whole upload batch.
        assert normalize_config({"strategy": "semantic"}).strategy == DEFAULT_STRATEGY
        assert normalize_config({"strategy": ""}).strategy == DEFAULT_STRATEGY
        assert normalize_config({"strategy": None}).strategy == DEFAULT_STRATEGY
        assert normalize_config({"strategy": 7}).strategy == DEFAULT_STRATEGY

    def test_a_known_strategy_survives_casing_and_padding(self):
        assert normalize_config({"strategy": "  SENTENCE "}).strategy == STRATEGY_SENTENCE

    @pytest.mark.parametrize("strategy", list(chunking.STRATEGIES))
    def test_every_advertised_strategy_is_accepted(self, strategy):
        # `STRATEGIES` is what `GET /resources/indexing-options` renders. A key the
        # validator would silently rewrite is an option the UI offers and cannot honour.
        assert normalize_config({"strategy": strategy}).strategy == strategy

    @pytest.mark.parametrize(
        "size,requested,expected",
        [
            (200, 180, 100),   # the default overlap does not fit a small chunk
            (200, 999, 100),
            (400, 900, 200),
            (1200, 180, 180),
            (1200, 600, 600),  # exactly at the cap
            (1200, 601, 600),
            (1200, -50, 0),
            (1200, "abc", 180),
        ],
    )
    def test_overlap_is_capped_relative_to_chunk_size(self, size, requested, expected):
        """Overlap at or past half the chunk size makes the window advance slower than it
        grows — that is how one document becomes an unbounded number of chunks."""
        config = normalize_config({"chunk_size": size, "overlap": requested})
        assert config.overlap == expected
        assert config.overlap <= config.chunk_size * MAX_OVERLAP_RATIO

    def test_a_provider_without_a_model_collapses_to_neither(self):
        """Half a configuration is not a configuration.

        If `embeds` were true with only a provider, ingestion would call the embedding path
        with no model name — a per-document failure on a document that would otherwise have
        indexed perfectly well by keyword.
        """
        provider_only = normalize_config({"embedding_provider": "openai"})
        assert (provider_only.embedding_provider, provider_only.embedding_model) == (None, None)
        assert provider_only.embeds is False

        model_only = normalize_config({"embedding_model": "text-embedding-3-small"})
        assert (model_only.embedding_provider, model_only.embedding_model) == (None, None)
        assert model_only.embeds is False

    @pytest.mark.parametrize(
        "provider,model",
        [("  ", "text-embedding-3-small"), ("openai", "   "), (5, "m"), ("openai", None)],
    )
    def test_a_blank_or_non_string_half_is_no_configuration_either(self, provider, model):
        config = normalize_config({"embedding_provider": provider, "embedding_model": model})
        assert config.embeds is False
        assert (config.embedding_provider, config.embedding_model) == (None, None)

    def test_both_halves_present_is_the_only_way_to_embed(self):
        config = normalize_config(
            {"embedding_provider": " openai ", "embedding_model": " text-embedding-3-small "}
        )
        assert config.embeds is True
        assert config.embedding_provider == "openai"
        assert config.embedding_model == "text-embedding-3-small"

    def test_embeddings_are_off_unless_asked_for(self):
        # Indexing with embeddings ships the document to a third party and usually costs
        # money. Neither may happen because a form defaulted to it.
        assert DEFAULT_CONFIG.embeds is False
        assert normalize_config(None).embeds is False
        assert normalize_config({}).embeds is False

    def test_normalizing_is_idempotent_through_to_dict(self):
        """`to_dict` is what gets snapshotted into `config_json` and onto the `File` row, and
        `config_for_file` reads it back through `normalize_config`. If the round trip drifted,
        a document would be re-indexed with settings the user never chose."""
        for raw in [
            {},
            {"strategy": "page", "chunk_size": 3000, "overlap": 300},
            {"strategy": "nonsense", "chunk_size": 5, "overlap": 9999},
            {"strategy": "sentence", "embedding_provider": "ollama", "embedding_model": "nomic"},
        ]:
            once = normalize_config(raw)
            assert normalize_config(once.to_dict()) == once


class TestTheEffectiveStrategyIsReportedHonestly:
    """A silently substituted setting is a setting that has become decorative."""

    def test_page_without_pages_reports_character(self):
        # Someone who picked "One chunk per page" for a .txt is owed the information that it
        # was indexed by fixed size instead.
        config = _config(STRATEGY_PAGE, 1200, 180)
        assert describe_effective(config, has_pages=False, has_paragraphs=False) == STRATEGY_CHARACTER
        assert describe_effective(config, has_pages=True, has_paragraphs=False) == STRATEGY_PAGE

    def test_paragraph_without_paragraph_structure_reports_sentence(self):
        # Whitespace normalization destroys the blank line between paragraphs, so a PDF has
        # no paragraph structure to keep whole — it degrades to sentences.
        config = _config(STRATEGY_PARAGRAPH, 1200, 180)
        assert describe_effective(config, has_pages=False, has_paragraphs=False) == STRATEGY_SENTENCE
        assert describe_effective(config, has_pages=False, has_paragraphs=True) == STRATEGY_PARAGRAPH

    @pytest.mark.parametrize("strategy", [STRATEGY_CHARACTER, STRATEGY_SENTENCE])
    def test_a_strategy_that_needs_no_structure_is_never_substituted(self, strategy):
        config = _config(strategy, 1200, 180)
        assert describe_effective(config, has_pages=False, has_paragraphs=False) == strategy

    def test_the_reported_fallback_is_what_actually_runs(self, text_like):
        """The label and the behaviour must be the same fact.

        `describe_effective` is only a string; this pins it to the output so the report
        cannot claim one strategy while the chunker ran another.
        """
        text, breaks = text_like

        page_config = _config(STRATEGY_PAGE, 600, 180)
        assert describe_effective(page_config, False, True) == STRATEGY_CHARACTER
        assert split_spans(text, page_config, None, breaks) == split_spans(
            text, _config(STRATEGY_CHARACTER, 600, 180), None, breaks
        )

        paragraph_config = _config(STRATEGY_PARAGRAPH, 600, 180)
        assert describe_effective(paragraph_config, False, False) == STRATEGY_SENTENCE
        assert split_spans(text, paragraph_config, None, None) == split_spans(
            text, _config(STRATEGY_SENTENCE, 600, 180), None, None
        )


class TestThePageStrategyUsesTheExtractorsPagesExactly:
    def test_a_page_chunk_starts_at_its_pages_first_character(self, pdf_like):
        """Exact, not derived. The page span comes from the PDF extractor as it joined the
        pages, so this is the one strategy whose citation cannot name the wrong page."""
        text, spans = pdf_like
        chunks = split_spans(text, _config(STRATEGY_PAGE, 4000, 180), spans, None)

        assert [start for _, start, _ in chunks] == [start for _, start, _ in spans]
        for (chunk, start, end), (_number, page_start, page_end) in zip(chunks, spans):
            assert (start, end) == (page_start, page_end)
            assert chunk == text[page_start:page_end]

    def test_a_pdf_paragraph_split_is_a_page_split_refined_by_the_packer(self, pdf_like):
        """pypdf reports no paragraphs, so page starts are the only boundaries a PDF has.
        Losing that would send a PDF paragraph request to the sentence fallback instead."""
        text, spans = pdf_like
        chunks = split_spans(text, _config(STRATEGY_PARAGRAPH, 4000, 0), spans, None)

        page_starts = {start for _, start, _ in spans}
        assert page_starts <= {start for _, start, _ in chunks}

    def test_a_blank_page_contributes_no_chunk(self):
        # `_join_pages` already drops blank pages, but a page of pure whitespace reaching
        # the splitter must not become an empty chunk with a real span.
        text = "alpha beta gamma"
        spans = [(1, 0, 5), (2, 5, 6), (3, 6, 16)]
        chunks = split_spans(text, _config(STRATEGY_PAGE, 1200, 180), spans, None)

        assert [chunk for chunk, _, _ in chunks] == ["alpha", "beta gamma"]

    def test_page_spans_are_ignored_by_every_other_strategy(self, pdf_like):
        text, spans = pdf_like
        for strategy in (STRATEGY_CHARACTER, STRATEGY_SENTENCE):
            config = _config(strategy, 600, 180)
            assert split_spans(text, config, spans, None) == split_spans(text, config, None, None)


class TestSentenceAndParagraphPacking:
    def test_a_sentence_is_never_cut_in_half(self):
        """The reason to choose this strategy at all: a truncated clause in a policy or a
        contract changes its meaning, and the chunk is what the model is shown."""
        text = _normalize_whitespace(
            " ".join(f"Clause {index} states the inspection interval is fixed." for index in range(60))
        )
        chunks = split_spans(text, _config(STRATEGY_SENTENCE, 600, 0))

        assert len(chunks) > 1
        for chunk, _, _ in chunks:
            assert chunk.endswith("."), chunk[-40:]

    def test_a_paragraph_that_fits_is_kept_whole(self):
        paragraphs = [
            "How do I reset the controller? Hold the button down for ten seconds until the "
            "status light turns amber, then release it.",
            "What is the warranty period? Twenty four months from the delivery date, or "
            "twelve months once the seal has been broken.",
            "Where is the serial number? On the underside of the housing, beneath the "
            "mounting bracket and next to the coolant port.",
        ]
        # The fixture only means something if one paragraph fits the budget and two do not.
        assert all(len(paragraph) <= 200 for paragraph in paragraphs)
        assert len(paragraphs[0]) + len(paragraphs[1]) > 200

        text, breaks = _normalize_with_breaks("\n\n".join(paragraphs))
        chunks = split_spans(text, _config(STRATEGY_PARAGRAPH, 200, 0), None, breaks)

        # An FAQ answer split across two chunks scores half as well on the question it
        # answers, which is the whole reason someone picks this strategy.
        assert [chunk for chunk, _, _ in chunks] == paragraphs

    def test_short_paragraphs_are_packed_together_up_to_the_budget(self):
        raw = "\n\n".join(f"Note {index}." for index in range(20))
        text, breaks = _normalize_with_breaks(raw)
        chunks = split_spans(text, _config(STRATEGY_PARAGRAPH, 1200, 0), None, breaks)

        # Emitting a chunk per nine-character paragraph would flood the index with rows that
        # can never carry enough vocabulary to score.
        assert chunks == [(text, 0, len(text))]

    def test_zero_overlap_means_no_repeated_text(self, text_like):
        text, breaks = text_like
        chunks = split_spans(text, _config(STRATEGY_SENTENCE, 300, 0), None, breaks)

        assert len(chunks) > 1
        for (_, _, previous_end), (_, start, _) in zip(chunks, chunks[1:]):
            assert start >= previous_end

    def test_overlap_actually_repeats_text_across_a_boundary(self, text_like):
        """Without it, an answer that straddles a chunk boundary is missed entirely — the
        cost the 'None' overlap preset warns about."""
        text, breaks = text_like
        chunks = split_spans(text, _config(STRATEGY_SENTENCE, 600, 300), None, breaks)

        assert len(chunks) > 1
        assert any(start < previous_end for (_, _, previous_end), (_, start, _) in zip(chunks, chunks[1:]))


class TestTheRecommendationsAreUsable:
    def test_every_supported_rag_mode_has_a_recommendation(self):
        # `GET /resources/indexing-options` builds its payload by looking every mode up. A
        # mode added to the engine without an entry here silently gets the hybrid advice.
        assert set(chunking.RAG_MODE_RECOMMENDATIONS) == SUPPORTED_RAG_MODES

    @pytest.mark.parametrize("mode", sorted(SUPPORTED_RAG_MODES))
    def test_every_recommendation_survives_the_validator_unchanged(self, mode):
        """A recommendation the validator would clamp is advice the app cannot follow: the
        UI shows one number, the document is indexed with another, and nothing says so."""
        recommended = recommendation_for(mode)
        normalized = normalize_config(recommended)

        assert normalized.strategy == recommended["strategy"]
        assert normalized.chunk_size == recommended["chunk_size"]
        assert normalized.overlap == recommended["overlap"]

    @pytest.mark.parametrize("mode", sorted(SUPPORTED_RAG_MODES))
    def test_every_recommendation_explains_itself(self, mode):
        # These strings are rendered verbatim to someone choosing how to spend an upload.
        assert recommendation_for(mode)["why"].strip()

    def test_an_unknown_mode_falls_back_to_the_hybrid_advice(self):
        assert recommendation_for("semantic_vector_search") == (
            chunking.RAG_MODE_RECOMMENDATIONS["contextual_hybrid"]
        )

    @pytest.mark.parametrize("preset", chunking.CHUNK_SIZE_PRESETS, ids=lambda p: str(p["value"]))
    def test_every_chunk_size_preset_survives_the_validator(self, preset):
        assert normalize_config({"chunk_size": preset["value"]}).chunk_size == preset["value"]
        assert preset["label"] and preset["hint"]

    @pytest.mark.parametrize("preset", chunking.OVERLAP_PRESETS, ids=lambda p: str(p["value"]))
    def test_every_overlap_preset_survives_the_validator_at_every_size_preset(self, preset):
        # An overlap preset offered next to a size preset it cannot be used with would be
        # clamped on save, showing the user a value the document was not indexed with.
        for size in [entry["value"] for entry in chunking.CHUNK_SIZE_PRESETS]:
            config = normalize_config({"chunk_size": size, "overlap": preset["value"]})
            assert config.overlap == preset["value"], f"clamped at chunk size {size}"
        assert preset["label"] and preset["hint"]

    DOCUMENTED_KEYS = ("label", "summary", "best_for", "caveat", "size_effect", "overlap_effect")

    @pytest.mark.parametrize("strategy", sorted(chunking.STRATEGIES))
    def test_every_strategy_is_documented_for_the_options_endpoint(self, strategy):
        spec = chunking.STRATEGIES[strategy]
        assert set(self.DOCUMENTED_KEYS) <= set(spec)
        assert all(str(spec[key]).strip() for key in self.DOCUMENTED_KEYS)

    def test_every_strategy_says_what_size_and_overlap_do_under_it(self):
        """The size and overlap controls render the same options for all four strategies,
        because all four really do use both numbers — but they use them for different
        things. Under `character` the size is the window; under `sentence`/`paragraph` it is
        a ceiling whole units are packed up to; under `page` it is a ceiling most pages never
        reach, and the overlap only ever applies *inside* a page that did.

        Identical copy for all four is what makes the form look as though changing the
        strategy left those two controls behind. These strings are the only thing that
        distinguishes them, so a copy-paste that gave two strategies the same sentence would
        put the form straight back where it was.
        """
        sizes = [spec["size_effect"] for spec in chunking.STRATEGIES.values()]
        overlaps = [spec["overlap_effect"] for spec in chunking.STRATEGIES.values()]

        # `sentence` and `paragraph` share a packer, so their *size* wording legitimately
        # differs only in the unit it names — but no two may be byte-identical.
        assert len(set(sizes)) == len(chunking.STRATEGIES)
        assert len(set(overlaps)) == len(chunking.STRATEGIES)

        # The one a user cannot deduce from the strategy's own description: overlap is very
        # nearly inert under `page`, and `_split_by_pages` only passes it on to the character
        # splitter for a page that exceeded the budget.
        assert "page" in chunking.STRATEGIES[chunking.STRATEGY_PAGE]["overlap_effect"].lower()
