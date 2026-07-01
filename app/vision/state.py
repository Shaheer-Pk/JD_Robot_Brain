from time import time
import threading

FLICKER_TOLERANCE = 2          # seconds — brief disappearance still counts as the same person
RETRY_INTERVAL_SECONDS = 1.5   # only re-attempt recognition after this much time


class FaceSessionState:
    """
    Tracks who the robot currently sees, recognition progress, mouth state,
    and listening status. Single shared instance, all in memory.
    """

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

    # ---- face presence (called every frame from the stream) ----

    def on_face_detected(self):
        with self._lock:
            if self.face_present:
                return  # still the same ongoing detection, nothing to do

            self.face_present = True
            self.first_detected_at = time()

            if self._should_reuse_cache_locked():
                # Same person as a moment ago (brief flicker, e.g. they
                # turned their head) — keep their identity, skip recognition.
                self.recognition_status = "identified"
                self.lost_at = None
                return

            # Genuinely new face — start a fresh identification cycle.
            self.recognition_status = "pending"
            self.recognition_attempts = 0
            self.active_user_id = None
            self.user_profile = None
            self.recognition_in_progress = False
            self.last_attempt_at = None
            self.stall_phrase_queued = False

    def on_face_lost(self):
        # Called automatically now from the live stream (no face found
        # in the latest frame) — replaces ARC's old "on tracking end" call.
        with self._lock:
            self.face_present = False
            self.first_detected_at = None
            self.lost_at = time()
            # active_user_id / user_profile are deliberately kept — flicker
            # tolerance needs them in case this same person reappears soon.

    def _should_reuse_cache_locked(self) -> bool:
        if self.active_user_id is None or self.lost_at is None:
            return False
        return (time() - self.lost_at) < FLICKER_TOLERANCE

    # ---- recognition outcome ----

    def mark_identified(self, user_id, profile):
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
        with self._lock:
            if self.first_detected_at is None:
                return None
            return time() - self.first_detected_at

    # ---- recognition attempt locking (prevents overlapping attempts) ----

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

    # ---- stall phrase ("user unidentified, one sec") ----

    def queue_stall_phrase(self):
        with self._lock:
            self.stall_phrase_queued = True

    def pop_stall_phrase(self):
        """Returns True once, then clears the flag, so it only fires once."""
        with self._lock:
            if self.stall_phrase_queued:
                self.stall_phrase_queued = False
                return True
            return False

    # ---- mouth tracking ----

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

    # ---- listening flag ----

    def start_listening(self):
        with self._lock:
            self.is_listening = True
            self.mouth_closed_since = None

    def stop_listening(self):
        with self._lock:
            self.is_listening = False


# Single shared instance — there's only one camera, one robot, one "current person"
session = FaceSessionState()