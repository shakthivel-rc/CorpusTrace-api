import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.dialects.mysql import CHAR, MEDIUMBLOB
from sqlalchemy.orm import relationship

from db.session import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False)
    resource_id = Column(CHAR(36), ForeignKey("resources.id"), nullable=False, index=True)
    file_id = Column(CHAR(36), ForeignKey("files.id"), nullable=True, index=True)
    chunk_index = Column(Integer, nullable=False)
    source_name = Column(String(255), nullable=False)
    modality = Column(String(50), nullable=False, default="text")
    title = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    contextual_content = Column(Text, nullable=False)
    terms_json = Column(Text, nullable=False)
    # Where this chunk sits in its source document, so an answer can point at the page it
    # came from. All four are nullable on purpose: chunks indexed before this existed have
    # no position, and only PDFs have pages at all. A NULL here means "unknown" and the UI
    # must say so — never guess, or the highlight lands on the wrong paragraph.
    # char_start/char_end are offsets into the file's extracted, whitespace-normalized
    # text — the same string `content` is sliced from.
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    char_start = Column(Integer, nullable=True)
    char_end = Column(Integer, nullable=True)

    # The chunk's embedding. All of these are NULL unless the uploader explicitly chose an
    # embedding model for this document — NULL means "indexed lexically only", which is the
    # default and the state of every chunk written before per-document settings existed.
    #
    # `embedding_vector` is the live format: packed little-endian binary32, one vector, no
    # separators (see `rag/vectors.py`). `embedding_json` is the original JSON-array format,
    # kept readable so chunks indexed before the change keep answering questions and so the
    # backfill that converts them stays reversible. Reads go through
    # `vectors.load_embedding`, which prefers the blob; writes only ever produce the blob.
    #
    # `embedding_norm` is that vector's Euclidean length, stored so the cosine denominator
    # is computed once at ingest rather than once per chunk per question. NULL means "not
    # recorded" — compute it — not "zero".
    #
    # `embedding_model` sits beside the vector because vectors from two different models are
    # not comparable: cosine between them is a number with no meaning that ranks with
    # complete confidence, so retrieval must be able to check before it compares.
    embedding_vector = Column(LargeBinary().with_variant(MEDIUMBLOB(), "mysql"), nullable=True)
    embedding_json = Column(Text, nullable=True)
    embedding_norm = Column(Float(precision=53), nullable=True)
    embedding_model = Column(String(255), nullable=True)
    embedding_dim = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    resource = relationship("Resource", back_populates="chunks")
    file = relationship("File", back_populates="chunks")


class RagGraphEntity(Base):
    __tablename__ = "rag_graph_entities"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False)
    resource_id = Column(CHAR(36), ForeignKey("resources.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    entity_type = Column(String(50), nullable=False, default="concept")
    weight = Column(Integer, nullable=False, default=1)
    chunk_refs_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


class RagGraphEdge(Base):
    __tablename__ = "rag_graph_edges"

    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False)
    resource_id = Column(CHAR(36), ForeignKey("resources.id"), nullable=False, index=True)
    source_entity_id = Column(CHAR(36), ForeignKey("rag_graph_entities.id"), nullable=False)
    target_entity_id = Column(CHAR(36), ForeignKey("rag_graph_entities.id"), nullable=False)
    relationship = Column(String(100), nullable=False, default="co_occurs_with")
    weight = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
