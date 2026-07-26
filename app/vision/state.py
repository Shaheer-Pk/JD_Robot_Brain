"""
app/vision/state.py — Single shared in-memory session state answering
"who does JD currently see and recognize." One instance (`session`),
process-lifetime, imported by both vision/routers.py (frame ingestion)
and vision/services/recognition_services.py (background identification).

Every field is read/written through a threading.Lock (self._lock) because
two different threads touch this object concurrently: the request thread
handling each POST /vision/stream call, and the background-task thread(s)
Starlette spins up for attempt_identification(). All external reads MUST
go through the locked getters (get_status(), get_is_listening(),
get_profile()) — never read self.recognition_status / self.is_listening /
self.user_profile directly from outside this file. Not defensive
paranoia: a single attribute read is atomic under CPython's GIL, but a
caller reading two related fields in two separate statements can still
observe a torn, contradictory snapshot if a writer thread runs in
between — the getters make each external read one atomic operation.

State machine — recognition_status:
  "idle"       -> no face currently in frame, nothing in progress
  "pending"    -> a face is present, identification attempt(s) in flight
  "identified" -> matched to a known user_id, active_user_id/user_profile set
  "guest"      -> MAX_RECOGNITION_ATTEMPTS failed attempts, gave up
   NOTE:
        'idle' is a guard-rail which runs only once, at the very beginning,
        later on nothing triggers it back and states cycle between
        'pending', 'identified', 'guest'
Flicker tolerance (FLICKER_TOLERANCE = 2s): mediapipe can lose a face for
a frame or two on ordinary head movement, not because the person actually
left. _should_preserve_session_locked() checks whether the face was lost
under FLICKER_TOLERANCE seconds ago AND we haven't already dropped to
"idle" — if so, on_face_detected() treats it as the SAME ongoing session
(recognition_attempts, active identity, etc. untouched) instead of
starting a fresh pending cycle. This covers an already-identified person
flickering (don't re-run recognition) AND a still-pending/guest person
flickering (don't reset their attempt count and get stuck in "pending"
indefinitely on repeated flicker — this was a confirmed bug in the
original design; see vision-handoff.md).

first_detected_at is set ONLY when a genuinely new face starts a session
(never touched during a preserved/flickered one) — kept for future
logging/debugging even though the recognition-timeout guard no longer
reads it (that logic is count-based now, below).

MAX_RECOGNITION_ATTEMPTS (=3), not time-based: recognition_services.py
gives up and calls mark_guest() once recognition_attempts >= 3 failed
tries. Chosen over wall-clock time because thread-scheduling delay under
load makes elapsed real time a poor proxy for "how many attempts actually
ran" — count is what "max 3 attempts" is actually supposed to mean.

RETRY_INTERVAL_SECONDS (=1.5): throttles how often attempt_identification
does real work, via start_recognition_attempt()/finish_recognition_attempt()
— an in-progress + min-interval gate preventing overlapping attempts and
preventing a new attempt firing on every single incoming frame.
"""

from time import time
import threading
from app.shared.memory import memory_session

from app.vision.schemas import VisionProfile

FLICKER_TOLERANCE = 10          # seconds — brief disappearance still counts as the same person
RETRY_INTERVAL_SECONDS = 0.5   # only re-attempt recognition after this much time
MAX_RECOGNITION_ATTEMPTS = 2   # give up and default to guest after this many failed attempts

SESSION_PRIVACY_TIMEOUT_SECONDS = 10  # deliberately separate from
# FLICKER_TOLERANCE — see rationale in vision-feature.md. Flicker tolerance
# answers "is this the same brief tracking blip"; this answers "has enough
# time passed that we assume the person genuinely left, for privacy reasons."
# Conflating them would silently couple two unrelated tuning knobs together.


class FaceSessionState:
    def __init__(self):
        self._lock = threading.Lock()
        self.active_user_id = None
        self.user_profile = None
        self.lost_at = None

        self.face_present = False
        self.recognition_status = "idle"        # "idle" | "pending" | "identified" | "guest"
        self.recognition_attempts = 0
        self.first_detected_at = None

        self.recognition_in_progress = False
        self.last_attempt_at = None
        self.stall_phrase_queued = False

        self.mouth_closed_since = None
        self.is_listening = False

    def on_face_detected(self):
        with self._lock:
            if self.face_present:
                return

            self.face_present = True

            if self._should_preserve_session_locked():
                # Same ongoing recognition (or already-identified person),
                # brief flicker — do not reset anything, do not touch
                # first_detected_at (it reflects when THIS session actually
                # started, not when it happened to flicker back).
                self.lost_at = None

                 # DEBUGGING
                print("[VISION STATE] Cache is STILL preserving previous session")

                return

            # Genuinely new face — start a fresh identification cycle.
            self.first_detected_at = time()
            self.recognition_status = "pending"
            self.recognition_attempts = 0
            self.active_user_id = None
            self.user_profile = None
            self.recognition_in_progress = False
            self.last_attempt_at = None
            self.stall_phrase_queued = False

            # DEBUGGING
            print("[VISION STATE] On Face Detected Was Triggered as a GENUINELY NEW FACE")

    def on_face_lost(self):
        with self._lock:
            if (self.face_present == False):    # Face is still lost
                 # DEBUGGING
                print("[VISION STATE] The face is GENUINELY lost (self.face_present is False)")

                return
            self.face_present = False
            self.lost_at = time()

            # DEBUGGING
            print("[VISION STATE] On Face Lost Was Triggered")
            # active_user_id / user_profile / first_detected_at deliberately
            # kept — flicker tolerance needs them if this person reappears soon.

    def _should_preserve_session_locked(self) -> bool:
        if self.lost_at is None:
            return False
        if self.recognition_status == "idle":
            return False
        return (time() - self.lost_at) < FLICKER_TOLERANCE
    
    def expire_stale_session_if_needed(self):
        should_expire = False
        with self._lock:
            if self.recognition_status != "idle" and self.lost_at is not None:
                if (time() - self.lost_at) >= SESSION_PRIVACY_TIMEOUT_SECONDS:
                    # DEBUGGING
                    print("[VISION STATE] Stale Session has been expired (SESSION_PRIVACY_TIMEOUT exceeded)")
                    self.active_user_id = None
                    self.user_profile = None
                    self.recognition_status = "idle"
                    self.recognition_attempts = 0
                    self.first_detected_at = None
                    should_expire = True

        # Called OUTSIDE state.py's own lock, deliberately — avoids nesting two
        # different locks (state's + memory's) in one call stack. Currently
        # safe either way since nothing calls in the reverse direction
        # (memory → state), but keeping locks un-nested is the more robust
        # habit regardless — it removes the need to ever reason about lock
        # ordering here at all.
        if should_expire:
            memory_session.reset()

            # DEBUGGING
            print("[VISION STATE] Memory has been reset successfully by expire_stale_session_if_needed()")

    def mark_identified(self, user_id: int, profile: VisionProfile):
        with self._lock:
            self.active_user_id = user_id
            self.user_profile = profile
            self.recognition_status = "identified"
            self.lost_at = None

    def mark_guest(self):
        with self._lock:
            self.active_user_id = None
            self.user_profile = None
            self.recognition_status = "guest"

    def increment_recognition_attempts(self):
        with self._lock:
            self.recognition_attempts += 1
            return self.recognition_attempts

    def seconds_since_detected(self):
        # Unused in current guard logic — kept for future logging/debugging.
        # first_detected_at is preserved across flicker, so this reflects
        # true session age, not time-since-last-flicker.
        with self._lock:
            if self.first_detected_at is None:
                return None
            return time() - self.first_detected_at

    def start_recognition_attempt(self, min_interval=RETRY_INTERVAL_SECONDS):
        with self._lock:
            if self.recognition_in_progress:
                return False
            if self.last_attempt_at is not None and (time() - self.last_attempt_at) < min_interval:
                return False
            self.recognition_in_progress = True
            self.last_attempt_at = time()
            return True

    def finish_recognition_attempt(self):
        with self._lock:
            self.recognition_in_progress = False

    def queue_stall_phrase(self):
        with self._lock:
            self.stall_phrase_queued = True

    def pop_stall_phrase(self):
        with self._lock:
            if self.stall_phrase_queued:
                self.stall_phrase_queued = False
                return True
            return False

    def update_mouth_state(self, is_open: bool):
        with self._lock:
            if is_open:
                self.mouth_closed_since = None
            elif self.mouth_closed_since is None:
                self.mouth_closed_since = time()

    def seconds_mouth_closed(self):
        with self._lock:
            if self.mouth_closed_since is None:
                return 0
            return time() - self.mouth_closed_since

    def start_listening(self):
        with self._lock:
            self.is_listening = True
            self.mouth_closed_since = None

    def stop_listening(self):
        with self._lock:
            self.is_listening = False

    # ---- locked getters — all reads from outside this class go through
    #      these, never raw attribute access ----

    def get_status(self):
        with self._lock:
            return self.recognition_status

    def get_is_listening(self):
        with self._lock:
            return self.is_listening

    def get_profile(self) -> VisionProfile | None:
        with self._lock:
            return self.user_profile


session = FaceSessionState()