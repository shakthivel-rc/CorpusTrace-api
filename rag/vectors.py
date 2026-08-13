"""Storage codec for chunk embeddings — packing, unpacking, and nothing else.

WHY THIS MODULE EXISTS.

Vectors used to be stored the way `terms_json` is: a JSON array of floats in a TEXT column.
That was a reasonable first shape — it needed no new type, no new table and no migration —
but profiling put the cost somewhere unhelpful. On a 500-chunk base at 1536 dimensions,
`json.loads` of `embedding_json` was **78% of the whole semantic retrieval cost** (315 ms of
405 ms). The arithmetic was never the problem; parsing seven kilobytes of decimal text per
chunk, per question, was. No amount of tuning the cosine touches a number like that.

So the vector is now `array('f')` bytes — IEEE 754 binary32, little-endian, no separators.
`frombytes` is a length check and a memcpy where `json.loads` was a character-by-character
parse building a Python float object per dimension.

Three properties are worth stating plainly, because each one is load-bearing:

* **It is a format change, not a semantic one.** The stored numbers mean exactly what they
  meant: an unnormalized embedding, compared by cosine. Nothing about ranking, blending or
  the evidence gate moves because of this file.
* **It is more precise than what it replaces, in the only sense that matters.** The JSON
  path wrote `round(value, 6)` — an absolute error of up to 5e-7 at every magnitude.
  binary32 carries ~7 significant digits, so across the [-1, 1] values an embedding
  contains its worst-case error is several times smaller (measured on a 768-dimension
  vector: 1.2e-7 against 5.0e-7). Per *element* the two trade wins, because a number that
  already had six decimal digits survives rounding exactly and does not survive float32
  exactly; per *format* the bound is what counts. Converting an existing stored vector
  moves its cosine by ~3e-9, against similarity floors stated to two decimal places — which
  is what makes the backfill safe to run over production data.
* **Both formats stay readable.** `load_embedding` prefers the blob and falls back to the
  JSON text, so a chunk indexed before this existed keeps answering questions with no
  backfill, and the backfill that does exist stays reversible (see the migration).

Pure by design: no SQLAlchemy import, no `rag.service` import. Every function here takes
values and returns values, so the format can be tested without a database, exactly as
`rag/chunking.py` is tested without one.
"""
from __future__ import annotations

import array
import json
import math
import sys
from typing import Sequence

# The one format constant. `array('f')` is a C float, which is binary32 on every platform
# CPython supports; the assertion is enforced at decode time rather than at import, because
# an import-time failure here would take the whole API down over a check that has never
# failed anywhere.
VECTOR_TYPECODE = "f"
VECTOR_ITEM_SIZE = 4

# Bytes are written little-endian regardless of host, so a database dump restored on a
# big-endian machine still decodes. Every platform this runs on today is little-endian, so
# this branch is free in practice and correct in principle.
_HOST_IS_LITTLE_ENDIAN = sys.byteorder == "little"

# `math.sumprod` landed in 3.12 and is a C-level fused loop; the generator fallback exists
# so this module imports on 3.10/3.11, which `scripts/bootstrap.sh` will accept with a note.
try:  # pragma: no cover - the branch taken depends on the interpreter, not the input
    from math import sumprod as sumprod
except ImportError:  # pragma: no cover
    def sumprod(left, right):  # type: ignore[misc]
        return math.fsum(a * b for a, b in zip(left, right))


def pack_embedding(values: Sequence[float]) -> bytes:
    """Serialize a vector to packed little-endian binary32.

    Raises `ValueError` on anything that is not a sequence of finite numbers, because the
    caller is about to write this into a column that retrieval will trust.
    """
    try:
        packed = array.array(VECTOR_TYPECODE, values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"vector is not a sequence of floats: {exc}") from exc
    if not _HOST_IS_LITTLE_ENDIAN:  # pragma: no cover - no big-endian CI
        packed.byteswap()
    return packed.tobytes()


def unpack_embedding(blob: bytes) -> array.array:
    """Decode packed binary32 back to a float sequence.

    Returns an `array('f')`, not a list. It is a genuine sequence — `len`, indexing and
    iteration all work, and `math.sumprod` consumes it directly — while allocating one
    buffer instead of N boxed floats. Building the list is most of what the JSON path was
    paying for, so materializing one here would hand the cost straight back.
    """
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise ValueError(f"expected bytes, got {type(blob).__name__}")
    decoded = array.array(VECTOR_TYPECODE)
    if decoded.itemsize != VECTOR_ITEM_SIZE:  # pragma: no cover - not reachable on CPython
        raise ValueError(f"platform float is {decoded.itemsize} bytes, not {VECTOR_ITEM_SIZE}")
    if len(blob) % VECTOR_ITEM_SIZE:
        # A truncated write, or a column that holds something that is not a vector at all.
        # Decoding the whole prefix would produce a shorter vector that then loses to a
        # dimension check somewhere far away from the actual corruption.
        raise ValueError(f"blob of {len(blob)} bytes is not a whole number of float32 values")
    decoded.frombytes(bytes(blob))
    if not _HOST_IS_LITTLE_ENDIAN:  # pragma: no cover - no big-endian CI
        decoded.byteswap()
    return decoded


def load_embedding(vector_blob: bytes | None, json_text: str | None) -> Sequence[float] | None:
    """The dual-read: a chunk's vector in whichever format it was stored in, or None.

    **This is the only place that knows both formats exist**, and the only place allowed to
    swallow a decode failure. Retrieval reads rows written years apart by two different
    code paths, and a chunk it cannot decode is a chunk that scores nothing — never an
    exception on a chat request, which this backend has no global handler for.

    The blob wins when both are present. That is what the backfill relies on: it writes the
    binary form and *keeps* the JSON, so the migration stays reversible, and reads still
    only ever pay for one of them.
    """
    if vector_blob:
        try:
            return unpack_embedding(vector_blob)
        except ValueError:
            return None
    if json_text:
        try:
            values = json.loads(json_text)
        except (TypeError, ValueError):
            return None
        if not isinstance(values, list):
            # A JSON object or scalar. Historically this fell through to a cosine that
            # raised on the first multiplication.
            return None
        try:
            # Validated in C, and returned in the same shape as the binary path so callers
            # never have to care which branch produced it. Doubles here rather than floats
            # because these values were written as decimal text and should not acquire a
            # rounding step on the way out of a format they are only being read from.
            return array.array("d", values)
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def vector_norm(values: Sequence[float]) -> float:
    """The Euclidean norm, used as the denominator of every cosine against this vector.

    Stored alongside the vector so retrieval computes it once at ingest instead of once per
    chunk per question — with N chunks and Q questions that is N×Q square roots and 2N×Q
    multiplications traded for N.
    """
    return math.sqrt(sumprod(values, values))


def encode_embedding(values: Sequence[float]) -> tuple[bytes, float, int]:
    """The three columns a chunk stores for one vector: (blob, norm, dimensions).

    The norm is computed from the *decoded* bytes rather than from `values`, so it is the
    norm of what was actually stored. Computing it from the float64 input instead would
    leave the denominator very slightly inconsistent with the numerator that later reads
    the float32 column, which is how a cosine ends up at 1.0000000001 and a similarity
    ceiling silently stops being a ceiling.
    """
    blob = pack_embedding(values)
    stored = unpack_embedding(blob)
    return blob, vector_norm(stored), len(stored)
