"""
app/vision/routers.py — Continuous frame-ingestion endpoint. C# posts to
POST /vision/stream repeatedly, once per captured camera frame, at
whatever cadence the camera/skill produces them — NOT a one-shot "take a
photo" call. Currently "Option A" per vision-handoff.md: every frame gets
decoded and run through mediapipe unconditionally (no ARC-side
pre-filtering of frames-with-no-face — that's the contingent Option B,
gated on Test Protocol A's outcome, not yet decided).

Per-request work:
  1. Decode multipart upload -> raw JPEG bytes -> cv2.imdecode -> pixel array.
  2. process_frame() (frame_processor.py, mediapipe FaceMesh) in a
     threadpool -- CPU-bound and blocking, same category of cost as
     Whisper in hearing/; MUST stay wrapped in run_in_threadpool or this
     async route freezes the whole event loop for the call's duration
     (this was a live, unfixed bug in the original teammate version --
     see hearing-feature.md issue #4 for the identical failure shape,
     already fixed there once).
  3. Face present -> session.on_face_detected() (may start a fresh
     recognition cycle or preserve an ongoing one -- see state.py) ->
     mouth state updated if available -> IF status is still "pending"
     after that call, schedule attempt_identification() as a
     BackgroundTasks callback. NOT awaited, NOT run inline -- it executes
     AFTER this response has already been returned, in its own thread.
     Face absent -> session.on_face_lost().
  4. Return a status snapshot (face_present, is_listening,
     mouth_closed_seconds, recognition_status) reflecting state as of
     BEFORE any background task just scheduled in this same call has run.

This route never returns identity/profile information, and never talks to
brain/ directly. Recognition results only ever get written into `session`
(app/vision/state.py) by the background attempt_identification() call, on
its own throttled schedule (RETRY_INTERVAL_SECONDS + an in-progress lock
-- real identification work happens far less often than frames arrive).
Whoever needs identity (brain/routers.py) reads session.get_profile()
independently, whenever a chat request happens to land -- there is no
direct call chain from this endpoint to brain/. C# is expected to poll
this endpoint every frame and react to recognition_status changing across
successive responses, not expect a complete answer from any single call.

Output Enforcement:
    response_model=StreamStatusResponse (vision/schemas.py) is now enforced on
    this route -- the return value must match that shape exactly, which is
    also why the bad-frame path changed: it used to return a differently-
    shaped {"status": "bad_frame"} dict as a normal 200, which would now fail
    response_model validation outright. It's now HTTPException(400, ...)
    instead -- a real contract change for whatever's posting frames: a
    malformed/undecodable frame used to come back as a 200 with a distinct
    shape, now comes back as a 400 error. Not yet reflected in any C#-side
    error handling, this needs its own handling when the C# frame-posting code is
    written.
"""

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from starlette.concurrency import run_in_threadpool

from app.vision.state import session
from app.vision.services.frame_processor import process_frame
from app.vision.services.recognition_services import attempt_identification
from app.vision.schemas import StreamStatusResponse

router = APIRouter()


@router.post("/stream")
async def receive_frame(frame: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    contents = await frame.read()
    np_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code = 400, detail = "Could not decode frame")

    face_present, mouth_is_open = await run_in_threadpool(process_frame, image)

    if face_present:
        session.on_face_detected()
        if mouth_is_open is not None:
            session.update_mouth_state(mouth_is_open)

        if session.get_status() == "pending":
            background_tasks.add_task(attempt_identification, contents)

    else:
        session.on_face_lost()
        session.expire_stale_session_if_needed()

    return StreamStatusResponse (
        face_present = face_present,
        is_listening = session.get_is_listening(),
        mouth_closed_seconds = session.seconds_mouth_closed(),
        recognition_status = session.get_status(),
    )