"""
app/emotion/state.py

[MODEL] Holds JD's current mood as plain data - a single shared object,
since there's only one robot with one mood at a time.

This file deliberately does NOT know the actual mood math - decay rate,
event penalties, band names, hysteresis. All of that logic belongs in
services.py, which is the ONE place that reads emotion_profile.json.
Keeping the numbers/config out of this file means the formula can only
ever be written in one place, so a future edit can't accidentally create
two slightly different copies of the same calculation.

A threading.Lock is used because FastAPI can be handling more than one
request at the same time. Without a lock, two chat replies landing at
almost the same instant could each read the mood, work out a new value,
and write it back a fraction of a second apart - silently losing one of
the two updates. Same reasoning as vision/state.py's FaceSessionState.
"""

from time import time
import threading


class MoodState:
    """
    Raw mood data for JD's simulated emotional state. Every read and
    write goes through the same lock, including the "last shown preset"
    value - even though picking a preset name is really services.py's
    job, it lives here too so there's only ONE lock guarding ALL of
    JD's mood-related data, instead of two separate locks that could
    each be updated out of step with the other.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # The mood dial itself: -1.0 (angry) through +1.0 (excited).
        # Starts at 0.0 - neutral, same as a freshly booted robot with no conversation history yet.
        self.mood_value: float = 0.0

        # Timestamp of the last time mood_value was written. services.py
        # uses this to work out "how much time has passed since we last
        # touched this number" for the decay calculation.
        self.last_updated: float = time()

        # Timestamp of the last time ANY chat turn completed successfully,
        # regardless of whether it changed the mood number. Deliberately
        # separate from last_updated - a neutral-tone reply still counts
        # as "JD is not idle" for sleepy mode checking
        self.last_interaction: float = time()

        # Whatever preset name (e.g. "happy") was last handed out by
        # services.py's get_current_preset(). Only used so the hysteresis
        # buffer has something to compare the current value against -
        # this file doesn't interpret what the name means.
        self.last_shown_preset: str | None = None

    def read(self):
        """
        Returns a (mood_value, last_updated) snapshot with no decay math
        applied - services.py is responsible for that calculation.
        Locked so this can never return a half-written value if another
        request is updating it at the exact same moment.
        """
        with self._lock:
            return self.mood_value, self.last_updated

    def write(self, new_value: float):
        """
        Overwrites mood_value with an already-computed new value (decay
        + event delta already applied by services.py before calling
        this), and stamps both last_updated (for decay) and 
        last_interaction (for sleepy mode checking) as right now 
        """
        with self._lock:
            self.mood_value = new_value
            now = time()
            self.last_updated = now
            self.last_interaction = now

    # def touch_interaction_only(self):
    #     """
    #     Resets the idle/sleepy timer WITHOUT touching mood_value. Used
    #     when a chat turn completed but had no effect on the mood number
    #     (a neutral-tone reply) - nothing about JD's mood changes, but JD
    #     is clearly still NOT idle, so the sleepy clock still resets.
    #     """
    #     with self._lock:
    #         self.last_interaction = time()
    #         self.last_updated = time()      # Done for decay during neutral conversation

    def seconds_since_interaction(self) -> float:
        """Used by services.py to check the 5-minute sleepy threshold."""
        with self._lock:
            return time() - self.last_interaction

    """ GETTERS and SETTERS for last preset needed for the Hysterisis buffer (no flicker moods)"""
    def get_last_shown_preset(self) -> str | None:
        with self._lock:
            return self.last_shown_preset

    def set_last_shown_preset(self, preset_name: str):
        with self._lock:
            self.last_shown_preset = preset_name


# Single shared instance - there's only one robot, one mood at a time
mood_state = MoodState()