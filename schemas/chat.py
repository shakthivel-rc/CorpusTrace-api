from pydantic import BaseModel, Field


class ResourceRenameRequest(BaseModel):
    # Bounded to the column width: `resources.resource_name` is VARCHAR(255) and MySQL
    # raises error 1406 rather than truncating, which would surface as an unhandled 500.
    resource_name: str = Field(..., min_length=1, max_length=255)


class PrecisionTraceRequest(BaseModel):
    """A question to run through the High-Precision pipeline for diagnosis only.

    The observability surface for that mode: it returns every stage's input and output —
    the normalized query, the expanded terms, the filters, the candidate counts, the
    reranker scores, the dedup and MMR decisions and the final chunks — so a retrieval
    result can be explained without re-deriving it by hand.

    `overrides` is a sparse `PrecisionConfig` patch, so one knob can be tried against a real
    knowledge base without changing the deployment's configuration for everyone. Unknown
    keys are ignored by `PrecisionConfig.with_overrides` rather than rejected, because this
    is a diagnostic and a typo in it should not be a 422.

    Nothing here is persisted: no chat history row, no activity log, no answer.
    """

    query: str = Field(..., min_length=1, max_length=2000)
    overrides: dict | None = None
