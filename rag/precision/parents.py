"""Parent-chunk recovery — stage 9 of the high-precision pipeline.

THE SHAPE OF THE HIERARCHY HERE. The brief's Document -> Section -> Parent -> Child chain is
real, but only the child level is stored: `document_chunks` rows ARE the children, and they
must stay exactly as they are, because every other RAG mode retrieves from them. Re-chunking
a corpus into smaller children to serve this one mode would silently change what Contextual
Hybrid, Fusion, GraphRAG, Corrective, Multi-Modal and Agentic all return from the same
knowledge base — which is the one thing this work is not allowed to do.

So the parent is a WINDOW over consecutive siblings of the same document
(`metadata.parent_key`), materialised here on demand. That gives the property the stage
exists for — a child selected for a precise match is returned with the surrounding text that
makes it mean something — without a migration, a re-upload, or a single byte of change to
what any other mode sees.

The child remains the citation anchor. `chunk_id`, `page_start` and `char_start` on the
returned result are the child's, so the evidence panel highlights the passage that actually
matched rather than a paragraph it was near (CLAUDE.md §20).
"""
from __future__ import annotations

from rag.precision.types import ChunkLike


def _join(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    return f"{left} {right}" if not left.endswith(" ") else f"{left}{right}"


def recover_parent(
    chunk: ChunkLike,
    family: list[str],
    by_id: dict[str, ChunkLike],
    *,
    max_chars: int,
) -> tuple[str | None, tuple[str, ...]]:
    """`(parent_context, sibling_ids)` for one child, or `(None, ())` when there is none.

    Expansion is SYMMETRIC and alternates outward from the child — one sibling before, one
    after, and so on — so a child in the middle of its window gets context from both sides
    and one at an edge is not padded from the only direction available. The budget is
    characters, not siblings, because the point is "enough surrounding text to preserve
    meaning" and a window of three 200-character chunks is not the same amount of context
    as three 1200-character ones.

    Returns None when the child is its own family. An empty string would read downstream as
    "there is a parent and it is blank"; None says the child stands alone, which for a
    single-chunk document is the truth.

    `max_chars` bounds the sibling TEXT, so the joined result can exceed it by up to one
    separator per sibling used. That is deliberate rather than overlooked: budgeting the
    separators would mean truncating a passage to make room for a space.
    """
    if not family or len(family) <= 1:
        return None, ()

    try:
        position = family.index(chunk.id)
    except ValueError:
        return None, ()

    budget = max(max_chars - len(chunk.content or ""), 0)
    if budget <= 0:
        return None, ()

    before: list[str] = []
    after: list[str] = []
    used: list[str] = []
    left, right = position - 1, position + 1
    take_left = True

    while budget > 0 and (left >= 0 or right < len(family)):
        index = None
        if take_left and left >= 0:
            index = left
            left -= 1
        elif right < len(family):
            index = right
            right += 1
        elif left >= 0:
            index = left
            left -= 1
        take_left = not take_left
        if index is None:
            continue

        sibling = by_id.get(family[index])
        if sibling is None:
            continue
        text = (sibling.content or "").strip()
        if not text:
            continue
        if len(text) > budget:
            # Truncate toward the child: a preceding sibling contributes its END, a
            # following one its START, so the text nearest the match is what survives.
            text = text[-budget:] if index < position else text[:budget]
        budget -= len(text)
        # Recorded with its position so the returned ids can be put back into READING order
        # below. Collected in outward-walk order they came back as (c1, c3, c0, c4), which
        # reads as a bug in any trace or provenance panel that displays them.
        used.append((index, sibling.id))
        (before if index < position else after).append(text)

    if not used:
        return None, ()
    used.sort(key=lambda item: item[0])

    # `before` was collected walking backwards, so it is reversed into reading order.
    ordered = list(reversed(before)) + [(chunk.content or "").strip()] + after
    context = ""
    for piece in ordered:
        context = _join(context, piece)
    return context.strip(), tuple(sibling_id for _index, sibling_id in used)
