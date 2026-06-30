import time
import threading

class VisionState:
    """
    Live, in-memory state of what the robot currently sees.
    One shared instance for the whole app. Resets on backend restart
    (no persistence needed).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.face_present = False           # is anyone in frame right now
            self.user_id = None                 # recognized user id, None if not known yet
            self.user_profile = None             # cached dict from get_user_profile()
            self.recognition_status = "idle"     # "idle" | "pending" | "identified" | "guest"
            self.recognition_attempts = 0
            self.first_detected_at = None        # timestamp face first appeared
            self.mouth_closed_since = None       # timestamp mouth became closed, None if open
            self.is_listening = False
            self.recognition_in_progress = False  # is an attempt running right now
            self.last_attempt_at = None            # timestamp of the last attempt
            self.stall_phrase_queued = False       # should ARC say the "unidentified" line

    # ---- face presence / recognition ----

    def on_face_detected(self):
        with self._lock:
            if not self.face_present:
                self.face_present = True
                self.first_detected_at = time.time()
                self.recognition_status = "pending"
                self.recognition_attempts = 0
                self.user_id = None
                self.user_profile = None
                self.recognition_in_progress = False
                self.last_attempt_at = None
                self.stall_phrase_queued = False

    def on_face_lost(self):
        with self._lock:
            self.face_present = False
            self.first_detected_at = None
            # NOTE: we deliberately do NOT clear user_id/profile here.
            # Flicker tolerance (a few seconds of grace before treating
            # someone as "really gone") will live in the service layer,
            # not in this state object.

    def mark_identified(self, user_id, profile):
        with self._lock:
            self.user_id = user_id
            self.user_profile = profile
            self.recognition_status = "identified"

    def mark_guest(self):
        with self._lock:
            self.user_id = None
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
            return time.time() - self.first_detected_at

    def start_recognition_attempt(self, min_interval=1.5):
        """
        Returns True if a new recognition attempt is allowed to start
        right now: nothing else already running, and enough time has
        passed since the last try. Call finish_recognition_attempt()
        when done, in a finally block.
        """
        with self._lock:
            if self.recognition_in_progress:
                return False
            if self.last_attempt_at is not None and (time.time() - self.last_attempt_at) < min_interval:
                return False
            self.recognition_in_progress = True
            self.last_attempt_at = time.time()
            return True

    def finish_recognition_attempt(self):
        with self._lock:
            self.recognition_in_progress = False

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
                self.mouth_closed_since = time.time()

    def seconds_mouth_closed(self):
        with self._lock:
            if self.mouth_closed_since is None:
                return 0
            return time.time() - self.mouth_closed_since

    # ---- listening flag ----

    def start_listening(self):
        with self._lock:
            self.is_listening = True
            self.mouth_closed_since = None

    def stop_listening(self):
        with self._lock:
            self.is_listening = False


# Single shared instance — every part of the vision module imports this,
# not the class itself.
vision_state = VisionState()