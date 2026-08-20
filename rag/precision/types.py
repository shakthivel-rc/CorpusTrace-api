"""The data the high-precision pipeline passes between its stages.

This package deliberately imports NOTHING from `rag.service`, `models` or `sqlalchemy`.
That is not tidiness — it is what makes the mode isolated in the sense the brief asks for:
`rag.service` imports this package and injects what it needs (the term map, the tokenizer,
the dense provider), so there is no import cycle, no DB session reaching in here, and every
stage below can be unit-tested against plain objects with no database at all.

A chunk is therefore described structurally (`ChunkLike`) rather than imported. A real
`models.rag.DocumentChunk` satisfies it as-is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class ChunkLike(Protocol):
    """The attributes this package reads off a chunk. `DocumentChunk` satisfies it."""

    id: str
    file_id: str | None
    chunk_index: int
    source_name: str
    modality: str
    title: str | None
    content: str
    contextual_content: str
    page_start: int | None
    page_end: int | None
    char_start: int | None
    char_end: int | None


@dataclass(frozen=True)
class ChunkMetadata:
    """Structure derived for one chunk, in the shape the brief specifies.

    Every field is derived from data that already exists on the row — nothing here required
    a schema change or a re-upload, so this is available on EVERY knowledge base in the
    installation, including every one indexed before this mode existed.

    `heading` and `section` are `None` when no heading could be identified. None means
    unknown and must be rendered as unknown: inventing a section name is the same class of
    mistake as defaulting an unknown page to page 1.
    """

    document_id: str | None
    chunk_id: str
    parent_chunk_id: str | None
    section: str | None
    heading: str | None
    page: int | None
    document_type: str
    category: str | None
    version: str | None
    entities: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "parent_chunk_id": self.parent_chunk_id,
            "section": self.section,
            "heading": self.heading,
            "page": self.page,
            "document_type": self.document_type,
            "category": self.category,
            "version": self.version,
            "entities": list(self.entities),
        }


@dataclass
class Candidate:
    """One chunk moving through the pipeline, carrying every score it has picked up.

    Scores are kept SEPARATELY rather than collapsed into one number as they are produced.
    A single running total cannot answer "why did this rank here", which is the question the
    trace exists to answer and the question an evaluation run is really asking.
    """

    chunk: ChunkLike
    bm25_score: float = 0.0
    dense_score: float = 0.0
    metadata_score: float = 0.0
    keyword_score: float = 0.0
    negative_penalty: float = 0.0
    # Set once the per-signal scores have been normalised and combined.
    retrieval_score: float = 0.0
    # Set by the reranker. `None` means "the reranker did not see this candidate", which is
    # different from "the reranker scored it 0.0" and must not be conflated: only the first
    # is a reason to fall back to the retrieval score.
    rerank_score: float | None = None
    final_score: float = 0.0
    sources: set[str] = field(default_factory=set)

    @property
    def chunk_id(self) -> str:
        return self.chunk.id


@dataclass
class PrecisionResult:
    """One final result: the child chunk that matched, plus the context around it."""

    chunk: ChunkLike
    score: float
    metadata: ChunkMetadata
    retrieval_score: float
    bm25_score: float
    dense_score: float
    rerank_score: float | None
    retrieval_method: str
    parent_context: str | None
    parent_chunk_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        """The payload shape §13 of the brief asks the mode to return."""
        return {
            "chunk_id": self.chunk.id,
            "document_id": self.chunk.file_id,
            "parent_chunk_id": self.metadata.parent_chunk_id,
            "text": self.chunk.content,
            "parent_context": self.parent_context,
            "parent_chunk_ids": list(self.parent_chunk_ids),
            "section": self.metadata.section,
            "heading": self.metadata.heading,
            "page": self.metadata.page,
            "source_name": self.chunk.source_name,
            "retrieval_method": self.retrieval_method,
            "scores": {
                "final": round(self.score, 6),
                "retrieval": round(self.retrieval_score, 6),
                "bm25": round(self.bm25_score, 6),
                "dense": round(self.dense_score, 6),
                "rerank": None if self.rerank_score is None else round(self.rerank_score, 6),
            },
            "metadata": self.metadata.to_dict(),
        }


@dataclass
class StageRecord:
    """What one pipeline stage did, for the trace."""

    stage: str
    count_in: int
    count_out: int
    detail: dict = field(default_factory=dict)


@dataclass
class PrecisionTrace:
    """The full narration of one question through the pipeline.

    Carries ids, counts, scores and QUERY-side terms only. No passage text and no
    credential ever enters a trace, because a trace is written to the application log and
    the log is the thing that gets pasted into a bug report.
    """

    original_query: str = ""
    normalized_query: str = ""
    expanded_terms: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    metadata_filters: dict = field(default_factory=dict)
    stages: list[StageRecord] = field(default_factory=list)
    reranker_backend: str = ""
    reranker_error: str | None = None
    final_chunk_ids: list[str] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)

    def record(self, stage: str, count_in: int, count_out: int, **detail) -> None:
        self.stages.append(StageRecord(stage=stage, count_in=count_in, count_out=count_out, detail=detail))

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "expanded_terms": self.expanded_terms,
            "entities": self.entities,
            "metadata_filters": self.metadata_filters,
            "reranker_backend": self.reranker_backend,
            "reranker_error": self.reranker_error,
            "stages": [
                {"stage": s.stage, "in": s.count_in, "out": s.count_out, **s.detail}
                for s in self.stages
            ],
            "final_chunk_ids": self.final_chunk_ids,
            "timings_ms": self.timings_ms,
        }


@dataclass
class PrecisionOutcome:
    """Everything one run of the pipeline produced."""

    results: list[PrecisionResult] = field(default_factory=list)
    trace: PrecisionTrace = field(default_factory=PrecisionTrace)

    @property
    def chunk_ids(self) -> list[str]:
        return [result.chunk.id for result in self.results]
