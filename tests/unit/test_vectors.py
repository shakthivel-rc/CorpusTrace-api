"""The embedding storage codec — `rag/vectors.py`.

WHAT IS AT STAKE HERE.

This module is the only thing standing between a column of bytes and a cosine similarity.
A cosine is a well-behaved-looking float in [-1, 1]: if the decoder hands back the wrong
numbers, nothing raises, nothing looks wrong, and an unrelated passage is ranked first and
cited in an answer. So the tests below are mostly about the two ways that can happen —
losing precision on the way in, and accepting garbage on the way out.

The format changed from a JSON array of decimals to packed binary32 because parsing the
former was 78% of semantic retrieval's cost. A format change to a column that already holds
production data has exactly two obligations, and both are pinned below: it must be able to
read what is already there, and converting what is already there must not change any answer.
"""
import array
import importlib.util
import json
import math
import struct
from pathlib import Path

import pytest

from rag.vectors import (
    VECTOR_ITEM_SIZE,
    encode_embedding,
    load_embedding,
    pack_embedding,
    sumprod,
    unpack_embedding,
    vector_norm,
)

pytestmark = pytest.mark.unit

# A vector with the awkward values in it: signs, a zero, a subnormal-ish magnitude, and
# numbers that have no exact binary representation.
SAMPLE = [0.1, -0.2, 0.0, 0.3333333, -0.99999, 1.0, -1.0, 5e-8]


def _pseudo_random(index: int) -> float:
    """A deterministic value in (-1, 1) with far more than six decimal digits.

    Spelled out rather than seeded from `random` so the precision assertions below are
    reproducible across interpreter versions — CPython's Mersenne Twister is stable, but
    the properties being pinned here are about a storage format and should not depend on
    that being true.
    """
    scrambled = math.sin(index * 12.9898) * 43758.5453123
    return (scrambled - math.floor(scrambled)) * 2.0 - 1.0


class TestRoundTrip:
    def test_values_survive_a_round_trip_within_float32_precision(self):
        decoded = unpack_embedding(pack_embedding(SAMPLE))
        assert len(decoded) == len(SAMPLE)
        for original, restored in zip(SAMPLE, decoded):
            assert restored == pytest.approx(original, rel=1e-6, abs=1e-9)

    def test_the_blob_is_four_bytes_per_dimension(self):
        """The size claim the change was made for: 3 KB at 768 dimensions against ~7 KB of
        decimal text. If this ever stops holding, the storage win has quietly gone."""
        assert len(pack_embedding([0.0] * 768)) == 768 * VECTOR_ITEM_SIZE == 3072

    def test_the_encoding_is_little_endian_binary32(self):
        """Pinned against `struct` rather than against `array` round-tripping itself, which
        would pass on a big-endian machine while writing bytes no other machine can read.
        A database dump is portable; this format has to be too."""
        assert pack_embedding([1.0, -2.0]) == struct.pack("<2f", 1.0, -2.0)

    def test_an_empty_vector_round_trips_to_empty(self):
        assert len(unpack_embedding(pack_embedding([]))) == 0

    def test_decoding_returns_a_sequence_not_a_list(self):
        """`array('f')` is one buffer; a list is N boxed Python floats. Materializing the
        list is most of what the JSON path was being charged for, so returning one here
        would hand the cost straight back while looking like a tidy-up."""
        decoded = unpack_embedding(pack_embedding(SAMPLE))
        assert isinstance(decoded, array.array)
        assert len(decoded) == len(SAMPLE) and decoded[0] == pytest.approx(0.1)
        assert sumprod(decoded, decoded) > 0  # consumable by the dot product as-is


class TestPrecisionAgainstWhatItReplaced:
    """binary32 is not a downgrade from the JSON it replaces, and that is a measurable
    claim rather than an opinion: the old writer did `round(value, 6)`.

    The claim is about the WORST case, not about every value. A number that already had six
    or fewer decimal digits — 0.1, 1e-4 — survived rounding exactly and does not survive
    float32 exactly, so per-element the two formats trade wins. What decides the precision
    of a *format* is its bound: rounding to six places admits an absolute error up to 5e-7
    at every magnitude, while binary32 carries ~7 significant digits, so on the [-1, 1]
    values an embedding actually contains its error is several times smaller. Below is that
    bound measured on a realistic vector, plus the only consequence anyone cares about —
    the effect on a cosine.
    """

    def test_the_worst_case_error_is_smaller_than_the_format_it_replaces(self):
        values = [_pseudo_random(index) for index in range(768)]
        legacy = [round(value, 6) for value in values]
        packed = unpack_embedding(pack_embedding(values))

        worst_legacy = max(abs(a - b) for a, b in zip(values, legacy))
        worst_packed = max(abs(a - b) for a, b in zip(values, packed))

        assert worst_legacy == pytest.approx(5e-7, abs=1e-8)
        assert worst_packed < worst_legacy

    def test_converting_a_legacy_vector_does_not_move_its_cosine(self):
        """The property the backfill rests on, and the one that makes it safe to run
        against production data. Similarity floors in `rag/service.py` are stated to two
        decimal places; a conversion that shifted the third would silently re-decide
        borderline questions, and nothing would look wrong. Measured shift: ~3e-9.
        """
        legacy_query = [round(_pseudo_random(index), 6) for index in range(768)]
        legacy_chunk = [round(_pseudo_random(index + 1000), 6) for index in range(768)]
        packed_query = unpack_embedding(pack_embedding(legacy_query))
        packed_chunk = unpack_embedding(pack_embedding(legacy_chunk))

        def cosine(left, right):
            return sumprod(left, right) / (vector_norm(left) * vector_norm(right))

        assert cosine(packed_query, packed_chunk) == pytest.approx(
            cosine(legacy_query, legacy_chunk), abs=1e-7
        )


class TestEncodeEmbedding:
    def test_it_returns_the_three_columns_a_chunk_stores(self):
        blob, norm, dimensions = encode_embedding([3.0, 4.0])
        assert blob == struct.pack("<2f", 3.0, 4.0)
        assert norm == pytest.approx(5.0)
        assert dimensions == 2

    def test_the_norm_is_of_the_stored_values_not_the_input(self):
        """The subtle one. Computing the norm from the float64 input leaves a denominator
        very slightly inconsistent with the numerator that later reads the float32 column —
        which is how a self-similarity comes out at 1.0000000001 and a similarity ceiling
        stops being a ceiling."""
        blob, norm, _ = encode_embedding(SAMPLE)
        stored = unpack_embedding(blob)

        assert norm == vector_norm(stored)
        self_similarity = sumprod(stored, stored) / (norm * norm)
        assert self_similarity <= 1.0

    @pytest.mark.parametrize("bad", [["a"], [None], [{}], "not a vector", [complex(1, 2)]])
    def test_a_non_numeric_vector_is_rejected_loudly(self, bad):
        """The write side raises where the read side returns None, and the asymmetry is
        deliberate: a bad write is a bug in ingestion that should stop the document, while a
        bad read is data already on disk that must not stop a chat request."""
        with pytest.raises(ValueError):
            pack_embedding(bad)


class TestLoadEmbeddingReadsBothFormats:
    def test_the_binary_column_is_preferred_when_both_are_present(self):
        """A backfilled row holds both. Reading the JSON would reintroduce the entire cost
        the change was made to remove, on exactly the rows it was made for."""
        loaded = load_embedding(pack_embedding([1.0, 2.0]), json.dumps([9.0, 9.0]))
        assert list(loaded) == [1.0, 2.0]

    def test_a_legacy_json_row_still_decodes(self):
        """The compatibility guarantee: no backfill is required for correctness, only for
        speed. A base embedded before the format change keeps answering questions."""
        assert list(load_embedding(None, json.dumps([1.5, -2.5]))) == [1.5, -2.5]

    def test_an_unembedded_chunk_is_none_rather_than_empty(self):
        """The overwhelmingly common case — every chunk in every lexical-only base. None
        means "no vector"; an empty sequence would mean "a zero-dimensional vector" and
        would sail into a dimension check instead of being skipped."""
        assert load_embedding(None, None) is None

    @pytest.mark.parametrize(
        "blob,text",
        [
            (b"\x01\x02\x03", None),                      # not a whole number of float32s
            (b"", "not json at all"),                     # empty blob falls through, text is junk
            (None, "not json at all"),
            (None, json.dumps({"vector": [1.0]})),        # an object, not an array
            (None, json.dumps(3.5)),                      # a bare scalar
            (None, json.dumps(["a", "b"])),               # right shape, wrong contents
            (None, json.dumps([1.0, None])),              # one poisoned element
            (42, None),                                   # not bytes at all
        ],
    )
    def test_unreadable_storage_is_none_and_never_raises(self, blob, text):
        """Every one of these decodes on a chat request, and this backend has no global
        exception handler — a raise here is a bare text/plain 500 with no envelope."""
        assert load_embedding(blob, text) is None

    def test_a_truncated_blob_is_rejected_rather_than_silently_shortened(self):
        """Decoding the whole prefix would produce a valid-looking shorter vector, which
        then loses to a dimension check somewhere far away from the actual corruption."""
        full = pack_embedding([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            unpack_embedding(full[:-1])
        assert load_embedding(full[:-1], None) is None


class TestVectorNorm:
    def test_it_is_the_euclidean_length(self):
        assert vector_norm([3.0, 4.0]) == pytest.approx(5.0)

    def test_a_zero_vector_has_a_zero_norm_rather_than_raising(self):
        """Some providers return an all-zero vector for an empty or unsupported passage.
        `_cosine` checks for this; it must get a number to check."""
        assert vector_norm([0.0, 0.0, 0.0]) == 0.0

    def test_it_matches_the_naive_computation(self):
        assert vector_norm(SAMPLE) == pytest.approx(math.sqrt(sum(v * v for v in SAMPLE)))


def _migration_module():
    """The backfill revision, loaded by path — its filename is not a valid module name."""
    path = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / "20260813a001_pack_chunk_embeddings_as_binary.py"
    )
    spec = importlib.util.spec_from_file_location("_backfill_revision", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheMigrationAgreesWithTheCodec:
    """The backfill deliberately inlines its own packer instead of importing this module.

    A migration is a statement about a schema at a point in time and has to keep producing
    the same bytes forever, so importing the live codec would let a future format change
    silently rewrite what that revision does to a database being upgraded from scratch
    today. The duplication is the correct call — but a duplicate nobody checks is just two
    things that disagree later, so it is checked here, at the only moment the divergence
    would be introduced.
    """

    def test_the_inlined_packer_produces_identical_bytes(self):
        migration = _migration_module()
        assert migration._pack(SAMPLE) == pack_embedding(SAMPLE)

    def test_the_inlined_converter_produces_the_same_blob_and_norm(self):
        migration = _migration_module()
        blob, norm = migration._convert(json.dumps(SAMPLE))
        expected_blob, expected_norm, _ = encode_embedding(SAMPLE)

        assert blob == expected_blob
        assert norm == pytest.approx(expected_norm, rel=1e-12)

    @pytest.mark.parametrize(
        "text", ["not json", json.dumps({}), json.dumps([]), json.dumps(["a"]), json.dumps(1.0)]
    )
    def test_the_converter_skips_what_it_cannot_read(self, text):
        """A row it cannot decode keeps exactly what it had. It was already scoring 0.0 in
        retrieval; a migration is not the place to discover that, and skipping is strictly
        safer than writing a guess."""
        assert _migration_module()._convert(text) is None
