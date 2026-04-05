from fastapi import APIRouter

from app.logging import log

router = APIRouter()


@router.get("/health")
def health():
    log.debug("health.check")
    return {"status": "ok"}
