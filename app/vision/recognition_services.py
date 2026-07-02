from app.vision.state import session
from app.vision.services import identify_face
from app.users.database import get_db

RETRY_INTERVAL_SECONDS = 1.5       # don't re-attempt more often than this
RECOGNITION_TIMEOUT_SECONDS = 9.0  # give up and default to guest after this long


def attempt_identification(image_bytes: bytes):
    """
    One identification attempt against a live frame's raw JPEG bytes.
    Runs as a FastAPI background task — updates session directly,
    doesn't return anything the caller needs to act on.
    """
    if session.recognition_status != "pending":
        return  # already resolved, or no face present anymore

    if not session.start_recognition_attempt(min_interval=RETRY_INTERVAL_SECONDS):
        return  # too soon since last try, or one is already running

    # get_db() is a generator (used elsewhere via Depends). Outside of a
    # request, we pull the session out manually and advance it again at
    # the end to trigger its own cleanup/close logic.
    db_gen = get_db()
    db = next(db_gen)
    try:
        result = identify_face(db, image_bytes)  # your existing full pipeline

        if result is not None:
            session.mark_identified(result["user_id"], result)
            return

        attempts = session.increment_recognition_attempts()
        if attempts == 1:
            # First miss — queue the stall phrase. Step 5 (Python -> ARC
            # channel) is what actually makes ARC speak it.
            session.queue_stall_phrase()

        elapsed = session.seconds_since_detected()
        if elapsed is not None and elapsed > RECOGNITION_TIMEOUT_SECONDS:
            session.mark_guest()

    finally:
        try:
            next(db_gen)  # lets get_db's own cleanup (closing the session) run
        except StopIteration:
            pass
        session.finish_recognition_attempt()