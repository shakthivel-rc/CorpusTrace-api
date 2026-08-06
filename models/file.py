import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.mysql import CHAR
from sqlalchemy.orm import relationship

from db.session import Base


class File(Base):
    __tablename__ = "files"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(255), nullable=False)
    file_url = Column(String(255), nullable=False)
    resource_id = Column(CHAR(36), ForeignKey("resources.id"), nullable=False)

    # How this document was indexed. Per document rather than per knowledge base, because a
    # base routinely mixes a scanned datasheet with a prose policy and the right slicing for
    # one is the wrong slicing for the other.
    #
    # The defaults reproduce the behaviour every document uploaded before this column
    # existed already got, so a backfilled row and a fresh row describe the same thing.
    chunk_strategy = Column(String(30), nullable=False, default="character", server_default="character")
    chunk_size = Column(Integer, nullable=False, default=1200, server_default="1200")
    chunk_overlap = Column(Integer, nullable=False, default=180, server_default="180")

    # NULL means this document was indexed lexically only — the default, and the state of
    # every document that predates embeddings. It is never a stand-in for "unknown".
    embedding_provider = Column(String(50), nullable=True)
    embedding_model = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    deleted_at = Column(DateTime, nullable=True)

    resource = relationship("Resource", back_populates="files")
    chunks = relationship("DocumentChunk", back_populates="file")
