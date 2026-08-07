"""
app/vision/services/recognition_services.py — Runs ONE identification
attempt against a single captured frame's raw JPEG bytes. Always invoked
as a FastAPI BackgroundTasks callback from vision/routers.py's /stream
endpoint — never called directly from a request handler, never awaited,
nothing reads its return value. All effects happen through mutating
`session` (app/vision/state.py); the outcome surfaces on the NEXT frame's
response, not this call's own.

Does nothing unless BOTH gates pass:
  1. session.get_status() == "pending" — someone's actually mid-
     recognition; a no-op if already "identified" or "guest".
  2. session.start_recognition_attempt() — an in-progress + min-interval
     (RETRY_INTERVAL_SECONDS) lock, preventing overlapping attempts and
     preventing a new attempt firing on every single frame (frames can
     arrive faster than one embedding extraction can finish).
Designed to be invoked far more often (once per frame with a face
present) than it actually does real work.

db_gen = get_db(); db = next(db_gen) — get_db() (app/users/database.py)
is a generator-based FastAPI dependency, normally driven automatically by
FastAPI via Depends(get_db) inside a real request (FastAPI calls next()
once to get the connection before the route runs, and once more after it
returns to trigger get_db's own `finally: conn.close()`). This function
runs OUTSIDE any request context, so that automatic machinery doesn't
apply — the two next() calls here manually replicate what FastAPI would
have done: the first runs get_db() up to `yield conn` and returns the
live connection; the second (in finally) resumes past that yield,
triggering cleanup, and is expected to raise StopIteration once the
generator ends — caught and ignored, not an error.

On match: session.mark_identified(...), return immediately.
On no match: recognition_attempts increments. On the FIRST miss only
(attempts == 1), queue_stall_phrase() is set so JD can say something like
"one sec" while still trying (Python->ARC channel for this — not yet
wired). Once attempts >= MAX_RECOGNITION_ATTEMPTS (state.py, =3),
mark_guest() fires — count-based, not time-based, deliberately; see
state.py's own docstring for why wall-clock time was rejected.
"""

from app.vision.state import session, RETRY_INTERVAL_SECONDS, MAX_RECOGNITION_ATTEMPTS
from app.vision.services.identification_services import identify_face
from app.users.database import get_db

CONFIDENCE_THRESHOLD = 0.2  # empirically tuned against one enrolled user
# (6 embeddings): genuine matches scored 0.3-0.6, lookalike strangers
# scored 0.1-0.12. Re-validate once a second real user is enrolled —
# small-sample thresholds can compress with more real variation in the DB.


def attempt_identification(image_bytes: bytes):
    if session.get_status() != "pending":
        return

    if not session.start_recognition_attempt(min_interval=RETRY_INTERVAL_SECONDS):
        return

    db_gen = get_db()
    db = next(db_gen)
    try:
        result = identify_face(db, image_bytes)

        # `result is None` MUST be checked first, combined via `or`, so
        # Python's short-circuit evaluation never touches result.confidence
        # on a None result (AttributeError otherwise). This branch covers
        # BOTH "no face/embedding extracted at all" AND "a face was found
        # but confidence too low to trust" — both correctly fall through
        # to attempts-then-guest logic. A prior version of this function
        # returned early on None specifically, which silently skipped
        # incrementing attempts — unmatched guests were re-attempted
        # indefinitely instead of ever reaching MAX_RECOGNITION_ATTEMPTS
        # and falling back to guest. Fixed this session.
        if result is None or result.confidence <= CONFIDENCE_THRESHOLD:
            attempts = session.increment_recognition_attempts()
            
            print(f"[VISION RECOGNITION] Couldn't recognize the face, this is attempt{attempts}:")

            if attempts == 1:
                session.queue_stall_phrase()

            if attempts >= MAX_RECOGNITION_ATTEMPTS:
                session.mark_guest()
        else:
            session.mark_identified(result.user_id, result)

    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass
        session.finish_recognition_attempt()