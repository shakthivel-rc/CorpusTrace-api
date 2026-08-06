from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from schemas.llm import LlmCredentialUpsert, LlmPreferenceUpdate
from schemas.response import ErrorResponse, SuccessResponse
from services.llm_provider import (
    LlmProviderError,
    delete_provider_credential,
    list_models_for_provider,
    list_provider_options,
    refresh_provider_models,
    set_user_preference,
    test_provider_connection,
    upsert_provider_credential,
)


def _user_id(request) -> str | None:
    user_payload = getattr(request.state, "user", None)
    return user_payload.get("sub") if user_payload else None


def _unauthenticated():
    return JSONResponse(status_code=401, content=ErrorResponse(status_code=401, message="Not authenticated").model_dump())


def _error(exc: LlmProviderError):
    body = ErrorResponse(status_code=exc.status_code, message=str(exc)).model_dump()
    return JSONResponse(status_code=exc.status_code, content=body)


def get_llm_providers(request, db: Session):
    user_id = _user_id(request)
    if not user_id:
        return _unauthenticated()
    return SuccessResponse(status_code=200, message="LLM providers fetched", data=list_provider_options(db, user_id))


def save_llm_credential(provider: str, payload: LlmCredentialUpsert, request, db: Session):
    user_id = _user_id(request)
    if not user_id:
        return _unauthenticated()
    try:
        result = upsert_provider_credential(db, user_id, provider, payload.api_key, payload.base_url, payload.refresh_models)
    except LlmProviderError as exc:
        return _error(exc)
    return SuccessResponse(status_code=200, message="LLM provider credential saved", data=result)


def remove_llm_credential(provider: str, request, db: Session):
    user_id = _user_id(request)
    if not user_id:
        return _unauthenticated()
    try:
        result = delete_provider_credential(db, user_id, provider)
    except LlmProviderError as exc:
        return _error(exc)
    return SuccessResponse(status_code=200, message="LLM provider credential removed", data=result)


def get_llm_models(provider: str, force_refresh: bool, request, db: Session):
    user_id = _user_id(request)
    if not user_id:
        return _unauthenticated()
    try:
        result = list_models_for_provider(db, user_id, provider, force_refresh)
    except LlmProviderError as exc:
        return _error(exc)
    return SuccessResponse(status_code=200, message="LLM models fetched", data=result)


def refresh_llm_models(provider: str, request, db: Session):
    user_id = _user_id(request)
    if not user_id:
        return _unauthenticated()
    try:
        result = refresh_provider_models(db, user_id, provider)
        db.commit()
    except LlmProviderError as exc:
        db.rollback()
        return _error(exc)
    return SuccessResponse(status_code=200, message="LLM models refreshed", data=result)


def save_llm_preference(payload: LlmPreferenceUpdate, request, db: Session):
    user_id = _user_id(request)
    if not user_id:
        return _unauthenticated()
    sent_params = bool({"temperature", "max_tokens", "top_p"} & payload.model_fields_set)
    try:
        result = set_user_preference(
            db,
            user_id,
            payload.provider,
            payload.model_id,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            top_p=payload.top_p,
            update_generation_params=sent_params,
            llm_enabled=payload.llm_enabled,
        )
    except LlmProviderError as exc:
        return _error(exc)
    return SuccessResponse(status_code=200, message="LLM preference saved", data=result)


def test_llm_connection(provider: str, request, db: Session):
    user_id = _user_id(request)
    if not user_id:
        return _unauthenticated()
    try:
        result = test_provider_connection(db, user_id, provider)
    except LlmProviderError as exc:
        return _error(exc)
    return SuccessResponse(status_code=200, message="LLM provider connection tested", data=result)
