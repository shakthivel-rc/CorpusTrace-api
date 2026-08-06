from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from controllers.llm_controller import (
    get_llm_models,
    get_llm_providers,
    refresh_llm_models,
    remove_llm_credential,
    save_llm_credential,
    save_llm_preference,
    test_llm_connection,
)
from db.session import get_db
from dependencies.auth import check_permissions
from schemas.llm import LlmCredentialUpsert, LlmPreferenceUpdate


router = APIRouter(prefix="/llm", tags=["LLM Providers"])


@router.get("/providers", dependencies=[Depends(check_permissions(["ai_access"]))])
def providers(request: Request, db: Session = Depends(get_db)):
    return get_llm_providers(request, db)


@router.put("/providers/{provider}/credential", dependencies=[Depends(check_permissions(["ai_access"]))])
def upsert_credential(provider: str, payload: LlmCredentialUpsert, request: Request, db: Session = Depends(get_db)):
    return save_llm_credential(provider, payload, request, db)


@router.delete("/providers/{provider}/credential", dependencies=[Depends(check_permissions(["ai_access"]))])
def delete_credential(provider: str, request: Request, db: Session = Depends(get_db)):
    return remove_llm_credential(provider, request, db)


@router.get("/providers/{provider}/models", dependencies=[Depends(check_permissions(["ai_access"]))])
def models(provider: str, request: Request, force_refresh: bool = Query(False), db: Session = Depends(get_db)):
    return get_llm_models(provider, force_refresh, request, db)


@router.post("/providers/{provider}/models/refresh", dependencies=[Depends(check_permissions(["ai_access"]))])
def refresh_models(provider: str, request: Request, db: Session = Depends(get_db)):
    return refresh_llm_models(provider, request, db)


@router.post("/providers/{provider}/test", dependencies=[Depends(check_permissions(["ai_access"]))])
def test_connection(provider: str, request: Request, db: Session = Depends(get_db)):
    return test_llm_connection(provider, request, db)


@router.put("/preferences", dependencies=[Depends(check_permissions(["ai_access"]))])
def preference(payload: LlmPreferenceUpdate, request: Request, db: Session = Depends(get_db)):
    return save_llm_preference(payload, request, db)
