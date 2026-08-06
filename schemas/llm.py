from pydantic import BaseModel, Field


class LlmCredentialUpsert(BaseModel):
    api_key: str | None = Field(default=None, min_length=1)
    base_url: str | None = Field(default=None, max_length=512)
    refresh_models: bool = True


class LlmPreferenceUpdate(BaseModel):
    provider: str = Field(..., min_length=1, max_length=50)
    model_id: str = Field(..., min_length=1, max_length=255)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=32000)
    top_p: float | None = Field(default=None, ge=0, le=1)
    # Tri-state on purpose: None means "leave it as it is", so saving a model from the
    # dropdown cannot turn synthesis back on behind the user's back.
    llm_enabled: bool | None = Field(default=None)
