import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.mysql import CHAR
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

    # The chunk's embedding, as a JSON array of floats — the same shape `terms_json` uses,
    # for the same reason: MySQL here has no vector type, and retrieval already loads and
    # scores every chunk of a resource in Python, so cosine similarity costs the same walk
    # the lexical scorer was already doing.
    #
    # All three are NULL unless the uploader explicitly chose an embedding model for this
    # document. `embedding_model` is stored alongside the vector because vectors from two
    # different models are not comparable — comparing them silently produces confident
    # nonsense, so retrieval must be able to check before it does.
    embedding_json = Column(Text, nullable=True)
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
