import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, BackgroundTasks

from app.vision.state import session
from app.vision.frame_processor import process_frame
from app.vision.recognition_services import attempt_identification

router = APIRouter(prefix="/vision", tags=["vision-stream"])


@router.post("/stream")
async def receive_frame(frame: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    contents = await frame.read()
    np_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"status": "bad_frame"}

    face_present, mouth_is_open = process_frame(image)

    if face_present:
        session.on_face_detected()
        if mouth_is_open is not None:
            session.update_mouth_state(mouth_is_open)

        if session.recognition_status == "pending":
            background_tasks.add_task(attempt_identification, contents)

    else:
        session.on_face_lost()

    return {
        "face_present": face_present,
        "is_listening": session.is_listening,
        "mouth_closed_seconds": session.seconds_mouth_closed(),
        "recognition_status": session.recognition_status,
    }