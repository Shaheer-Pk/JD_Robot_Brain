from time import time

FLICKER_TOLERANCE = 2  # tune for face loss time tolerance (seconds)


class FaceSessionState:
    # Tracks "is this still the same person, or did someone new just arrive".

    def __init__(self):
        self.active_user_id = None
        self.lost_at = None

    def face_lost(self):
        # Call this on ARC's 'on tracking end' event in camera skill
        self.lost_at = time()

    def should_reuse_cache(self) -> bool:
        #  Call this on ARC's 'on tracking start' event in camera, before re-running recognition
        if self.active_user_id is None or self.lost_at is None:
            return False
        return (time() - self.lost_at) < FLICKER_TOLERANCE

    def set_identity(self, user_id):
        self.active_user_id = user_id
        self.lost_at = None


# Single shared instance, amke sure there's only one camera, one robot and one "current person"
session = FaceSessionState()