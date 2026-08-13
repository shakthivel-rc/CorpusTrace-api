"""pack_chunk_embeddings_as_binary

Store chunk embeddings as packed binary32 instead of a JSON array of decimals, and record
each vector's norm alongside it.

WHY.

`embedding_json` held vectors the way `terms_json` holds a bag of words: a JSON array in a
TEXT column. It needed no new type and no migration, which is why it was the right first
shape — but retrieval reads that column for every chunk of a resource on every question,
and profiling put `json.loads` at **78% of the entire semantic retrieval cost** (315 ms of
405 ms on a 500-chunk base at 1536 dimensions). The cosine arithmetic was never the
bottleneck; turning seven kilobytes of decimal text per chunk back into 1536 boxed Python
floats, per question, was.

`embedding_vector` holds the same numbers as little-endian IEEE 754 binary32 — a length
check and a memcpy where there was a character parse. It is also less than half the bytes:
3 KB against ~7 KB at 768 dimensions.

`embedding_norm` is the vector's Euclidean length, computed once here instead of once per
chunk per question. With N chunks and Q questions that is N×Q square roots traded for N.

THREE THINGS THIS MIGRATION DELIBERATELY DOES NOT DO.

1. **It does not drop `embedding_json`.** The old column is left populated, which is what
   makes `downgrade()` genuinely reversible rather than reversible-shaped: dropping the two
   new columns restores the exact prior state with no data gone. `rag/vectors.load_embedding`
   reads either format and prefers the blob, so the retained JSON is never parsed and costs
   only disk. A later revision can drop it once the binary path has run in production for
   long enough to trust — that is a decision with evidence behind it, not one to take here.
2. **It does not touch a row it cannot decode.** A chunk whose JSON will not parse keeps
   exactly what it had. It was already scoring 0.0 in retrieval; this is not the place to
   discover that, and skipping is strictly safer than writing a guess.
3. **It does not import `rag.vectors`.** The packing below is deliberately a local copy.
   A migration is a statement about a schema at a point in time and must keep producing the
   same bytes forever; importing the live codec would mean a future format change silently
   rewrites what this revision does to a database being upgraded from scratch today.
   `tests/unit/test_vectors.py` asserts this copy and `rag.vectors` still agree, so the
   duplication is checked rather than merely hoped for.

Rows are converted in keyset batches over the primary key. `document_chunks` is the largest
table in the schema — one row per ~1200 characters of every document ever uploaded — so
loading it in one statement is the difference between a migration that runs and one that
exhausts memory on a machine nobody was watching.

Revision ID: 20260813a001
Revises: 20260805a003
Create Date: 2026-08-13 00:00:00.000000

"""
from typing import Sequence, Union
import array
import json
import math
import sys

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


revision: str = "20260813a001"
down_revision: Union[str, None] = "20260805a003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# How many chunks are read, converted and written per round trip. At 1536 dimensions a
# batch of 500 is ~3 MB of blob in flight, which is a comfortable statement size for MySQL's
# default 64 MB max_allowed_packet with room to spare for a wider vector.
BATCH_SIZE = 500

_LITTLE_ENDIAN = sys.byteorder == "little"


def _pack(values) -> bytes:
    """Frozen copy of `rag.vectors.pack_embedding`. See point 3 in the docstring above."""
    packed = array.array("f", values)
    if not _LITTLE_ENDIAN:  # pragma: no cover - no big-endian CI
        packed.byteswap()
    return packed.tobytes()


def _convert(json_text: str):
    """(blob, norm) for one stored JSON vector, or None if it is not one."""
    try:
        values = json.loads(json_text)
    except (TypeError, ValueError):
        return None
    if not isinstance(values, list) or not values:
        return None
    try:
        blob = _pack(values)
    except (TypeError, ValueError, OverflowError):
        return None
    # The norm is taken from the round-tripped float32 values, not from the parsed decimals,
    # so the denominator matches the numerator that retrieval will actually read. Computing
    # it from the wider input leaves cosines fractionally above 1.0, and a similarity
    # ceiling that can be exceeded is not a ceiling.
    stored = array.array("f")
    stored.frombytes(blob)
    if not _LITTLE_ENDIAN:  # pragma: no cover
        stored.byteswap()
    norm = math.sqrt(sum(value * value for value in stored))
    if not math.isfinite(norm):
        return None
    return blob, norm


def _chunks_table() -> sa.Table:
    """A minimal table definition, not the ORM model.

    The model describes today's schema; this revision has to keep working after the model
    grows a column, loses one, or renames the ones it does not touch.
    """
    return sa.table(
        "document_chunks",
        sa.column("id", sa.String),
        sa.column("embedding_json", sa.Text),
        sa.column("embedding_vector", sa.LargeBinary),
        sa.column("embedding_norm", sa.Float),
    )


def _backfill() -> None:
    bind = op.get_bind()
    chunks = _chunks_table()
    last_id = ""

    while True:
        rows = bind.execute(
            sa.select(chunks.c.id, chunks.c.embedding_json)
            .where(
                chunks.c.embedding_json.isnot(None),
                chunks.c.embedding_vector.is_(None),
                chunks.c.id > last_id,
            )
            .order_by(chunks.c.id)
            .limit(BATCH_SIZE)
        ).fetchall()
        if not rows:
            break
        last_id = rows[-1][0]

        updates = []
        for chunk_id, json_text in rows:
            converted = _convert(json_text)
            if converted is None:
                continue
            blob, norm = converted
            # `pk` rather than `id`: a bindparam sharing a column's name collides with the
            # column in the SET clause and updates the primary key to itself.
            updates.append({"pk": chunk_id, "vector": blob, "norm": norm})
        if updates:
            bind.execute(
                sa.update(chunks)
                .where(chunks.c.id == sa.bindparam("pk"))
                .values(
                    embedding_vector=sa.bindparam("vector"),
                    embedding_norm=sa.bindparam("norm"),
                ),
                updates,
            )


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column(
            "embedding_vector",
            sa.LargeBinary().with_variant(mysql.MEDIUMBLOB(), "mysql"),
            nullable=True,
        ),
    )
    op.add_column(
        "document_chunks",
        # Float(53) is DOUBLE on MySQL. A 4-byte FLOAT would carry ~7 digits, which is
        # enough for a norm but leaves the division at the edge of the precision the
        # similarity floors in `rag/service.py` are stated to.
        sa.Column("embedding_norm", sa.Float(precision=53), nullable=True),
    )
    _backfill()


def downgrade() -> None:
    # Lossless: `embedding_json` was never emptied, so every vector this revision converted
    # is still in the column the previous code reads.
    op.drop_column("document_chunks", "embedding_norm")
    op.drop_column("document_chunks", "embedding_vector")
