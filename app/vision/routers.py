"""
app/vision/routers.py — Continuous frame-ingestion endpoint. C# posts to
POST /vision/stream repeatedly, once per captured camera frame, at
whatever cadence the camera/skill produces them — NOT a one-shot "take a
photo" call. Currently "Option A" per vision-handoff.md: every frame that
clears the throttle below gets decoded and run through mediapipe
unconditionally (no ARC-side pre-filtering of frames-with-no-face — that's
the contingent Option B, gated on Test Protocol A's outcome, not yet
decided).

THROTTLE — added this session: incoming frames arrive at ~15-30fps (camera-
driven, uncontrollable at the source — confirmed via Synthiam's own
release notes, the JPEG stream engine has no fps limiter). Under
sustained real-camera load this created measurable event-loop contention
(GIL + raw CPU competition on a no-GPU i5) that showed up as inflated,
occasionally severe (~11s observed once) latency on completely unrelated
/brain/chat Gemini calls — root-caused via controlled before/after timing
comparisons, not guessed. Fix: a simple time-based gate limits actual
processing to ~5fps; frames arriving faster than that are skipped
entirely — no decode, no mediapipe, no session state mutation of any
kind — and get back a response reflecting session state exactly as it
stood after the last frame that WAS processed. This is a deliberate
hard skip, not a partial/best-effort one: a skipped frame contributes
NOTHING to face-presence or mouth-state tracking; only frames that clear
the throttle move those forward. See tasks.md for the fuller incident
account.

Per-request work (unchanged below the throttle check):
  1. Decode multipart upload -> raw JPEG bytes -> cv2.imdecode -> pixel array.
  2. process_frame() (frame_processor.py, mediapipe FaceMesh) in a
     threadpool -- CPU-bound and blocking, same category of cost as
     Whisper in hearing/; MUST stay wrapped in run_in_threadpool or this
     async route freezes the whole event loop for the call's duration.
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
    instead. A skipped-frame response uses this exact same enforced shape
    too — C# never sees a distinct "skipped" signal, deliberately, per
    this session's decision: it's a plain 200 that C#'s existing
    success-path handling (which never inspects the body) already treats
    correctly with zero changes needed on that side.
"""

import time as time_module  # aliased to avoid clashing with state.py's `from time import time` pattern if this file is ever merged/refactored near it
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from starlette.concurrency import run_in_threadpool

from app.vision.state import session
from app.vision.services.frame_processor import process_frame
from app.vision.services.recognition_services import attempt_identification
from app.vision.schemas import StreamStatusResponse

router = APIRouter()

# --- Throttle state — module-level, single-producer assumption ---
# Safe ONLY because exactly one source (CameraFrameUploader.cs, itself
# gated by its own _uploadInProgress guard) posts frames one at a time,
# never concurrently. If this route is ever hit by multiple simultaneous
# callers (e.g. a burst test script instead of the real ARC pipeline),
# this becomes a real race — acceptable here because that's not this
# route's actual traffic pattern, but do not copy this pattern
# elsewhere without re-checking that assumption holds.
TARGET_FPS = 5
MIN_FRAME_INTERVAL_SECONDS = 1.0 / TARGET_FPS
_last_processed_time = 0.0
_last_known_face_present = False

# StreamStatusResponse is enforced for this router endpoint.
# See app/vision/schemas.py for more information on StreamStatusReponse.
@router.post("/stream", response_model=StreamStatusResponse)
async def receive_frame(frame: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    global _last_processed_time, _last_known_face_present

    contents = await frame.read()  # always drain the upload body, skip or not

    now = time_module.monotonic()  # monotonic, not time.time() — immune to
    # system clock adjustments (NTP sync, etc.) that could otherwise cause
    # a spurious backward jump and break this elapsed-time check.

    if now - _last_processed_time < MIN_FRAME_INTERVAL_SECONDS:
        # Hard skip — no decode, no mediapipe, no session mutation at all.
        # face_present is held at whatever the last PROCESSED frame found;
        # every other field is read fresh from session since those are
        # cheap, already-locked, and reflect true current state regardless
        # of whether this particular frame was processed.
        return StreamStatusResponse(
            face_present=_last_known_face_present,
            is_listening=session.get_is_listening(),
            mouth_closed_seconds=session.seconds_mouth_closed(),
            recognition_status=session.get_status(),
        )

    _last_processed_time = now

    np_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode frame")

    face_present, mouth_is_open = await run_in_threadpool(process_frame, image)
    _last_known_face_present = face_present  # remember for the next skipped frames

    if face_present:
        session.on_face_detected()
        if mouth_is_open is not None:
            session.update_mouth_state(mouth_is_open)

        if session.get_status() == "pending":
            background_tasks.add_task(attempt_identification, contents)

    else:
        session.on_face_lost()
        session.expire_stale_session_if_needed()

    return StreamStatusResponse(
        face_present=face_present,
        is_listening=session.get_is_listening(),
        mouth_closed_seconds=session.seconds_mouth_closed(),
        recognition_status=session.get_status(),
    )