from fastapi import APIRouter, UploadFile, Depends
from starlette.concurrency import run_in_threadpool

from . import services as vision_services
from .state import session

from app.users.database import get_db

router = APIRouter(prefix="/vision", tags=["vision"])


@router.post("/identify")
async def identify(image: UploadFile, db=Depends(get_db)):

    # Called by ARC every time a face is freshly detected.
    # Send the photo as multipart/form-data, field name "image".

    if session.should_reuse_cache():
        return {"user_id": session.active_user_id, "cached": True}

    image_bytes = await image.read()

    # face_recognition is CPU-bound — run_in_threadpool keeps it from
    # blocking the event loop while Piper TTS is also running.
    result = await run_in_threadpool(vision_services.identify_face, db, image_bytes)

    if result:
        session.set_identity(result["user_id"])
        return {**result, "cached": False}

    return {"user_id": None, "cached": False}


@router.post("/lost")
async def lost():

    # Call this from ARC's "on tracking end" event so the flicker-tolerance
    # logic in state.py knows when the face actually disappeared.

    session.face_lost()
    return {"ok": True}


@router.post("/enroll")
async def enroll(user_id: int, image: UploadFile, db=Depends(get_db)):

    # Admin-only registration step. Call once per reference photo
    # (recommended: 3-5 calls per person, different angles/lighting).

    image_bytes = await image.read()
    success = await run_in_threadpool(vision_services.enroll_face, db, user_id, image_bytes)
    return {"success": success}