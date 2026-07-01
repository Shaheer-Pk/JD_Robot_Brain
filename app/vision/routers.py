from fastapi import APIRouter, UploadFile, Depends
from starlette.concurrency import run_in_threadpool

from . import services as vision_services

from app.users.database import get_db

router = APIRouter(prefix="/vision", tags=["vision"])


@router.post("/enroll")
async def enroll(user_id: int, image: UploadFile, db=Depends(get_db)):

    # Registration step. Call once per reference photo
    # (recommended: 3-5 calls per person, different angles/lighting).

    image_bytes = await image.read()
    success = await run_in_threadpool(vision_services.enroll_face, db, user_id, image_bytes)
    return {"success": success}

# /identify and /lost were removed from before as we shift to live stream
# now does continuous face detection and recognition itself