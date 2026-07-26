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

------------------------------------------------------------------------
WHY read() + write() ALONE ARE NOT ENOUGH (added this session)
------------------------------------------------------------------------
read() and write() are each individually safe - neither can ever return
or store a half-written value. But they are two SEPARATE lock
acquisitions. Nothing stops another thread from acquiring the lock in
the GAP between a caller's read() and that same caller's write():

    Thread A: read()  -> lock acquired, value returned, lock RELEASED
    Thread B: read()  -> lock acquired, value returned, lock RELEASED  <- sees A's stale value!
    Thread A: ...computes new value based on what it read...
    Thread B: ...computes new value based on what IT read...
    Thread A: write(new_value)  -> saves
    Thread B: write(new_value)  -> OVERWRITES A's write. A's update just vanished.

This is a classic "lost update" race. It could not happen when this
project had only one caller at a time (one conversation turn in flight
at once), which is why it went unnoticed originally. It became a real,
live risk once TWO independent, genuinely concurrent callers were added
this session: apply_event() can now race against itself under rapid
back-to-back requests, and get_current_preset() now races against
itself because it is polled by BOTH a 5-second C#-side timer AND a
fire-and-forget call fired every time JD finishes speaking.

THE FIX: apply() and decide_preset() below wrap an entire
read-compute-write (or read-decide-write) SEQUENCE inside ONE lock
acquisition, so there is no gap for another thread to sneak into. They
do this using a CALLBACK - a function handed in by services.py, called
BY this file, WHILE the lock is held.

Why a callback, and not just writing the math directly in this file:
this file is explicitly not allowed to know mood math (decay formula,
event deltas, band names, hysteresis buffer - see file docstring above).
But the lock MUST be held for the entire duration of the computation for
it to be atomic. The only way to satisfy both constraints at once - lock
spans the computation, but the computation's logic lives in services.py,
not here - is for services.py to hand this file a ready-to-call function
(the callback), and let THIS file decide exactly when to call it (i.e.
right after acquiring the lock, before releasing it). This file never
reads the callback's internals; it just calls it at the right moment
and uses whatever it returns.

Concretely, "handing over a function" means passing it WITHOUT
parentheses - e.g. `mood_state.apply(compute)`, not
`mood_state.apply(compute())`. No parentheses means "here is the
function itself, don't run it yet." Parentheses mean "run it right
now." services.py builds the function and hands it over unrun; THIS
file is the one that adds the parentheses - see apply()'s body below,
`compute_fn(self.mood_value, self.last_updated)` - and that call only
ever happens inside the `with self._lock:` block, which is the entire
point.
------------------------------------------------------------------------
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

    def apply(self, compute_fn):
        """
        Atomic "read the current mood, compute a new value from it, save
        the new value" - all as ONE uninterruptible operation, fixing the
        lost-update race described in the module docstring above.

        compute_fn is a CALLBACK: a function object, handed in by
        services.py, that this method calls itself, from INSIDE the
        `with self._lock:` block below. services.py never touches
        self._lock directly - it just writes an ordinary function that
        takes (raw_mood_value, last_updated_timestamp) and returns the
        new mood value it wants saved. This file has no idea what math
        happens inside compute_fn - decay formula, event deltas,
        clamping, all of that stays entirely in services.py, exactly as
        this file's docstring requires. All this method knows is: call
        it once, while locked, save whatever it returns.

        Example of what services.py passes in (see services.py's
        apply_event() for the real version):

            def compute(raw_value, last_updated):
                current = _decayed_value(raw_value, last_updated)
                return max(-1.0, min(1.0, current + delta))

            mood_state.apply(compute)   # note: `compute`, NOT `compute()`
                                         # - we are handing over the
                                         # function itself, unrun. This
                                         # method is the one that actually
                                         # calls it, below.

        Why this closes the race: because compute_fn(...) is called
        WHILE self._lock is held, no other thread can acquire the lock
        (and therefore no other thread can read self.mood_value or call
        apply()/read()/write() themselves) until this entire
        read-compute-write sequence has finished and the `with` block
        has exited. There is no gap left for a second thread to sneak a
        stale read into.
        """
        with self._lock:
            # compute_fn is called HERE - this is the one and only
            # moment its body actually executes, and it is guaranteed to
            # happen while self._lock is held (we are inside the `with`
            # block). self.mood_value / self.last_updated are read
            # directly here (not via self.read(), which would try to
            # acquire self._lock a second time and DEADLOCK - a plain
            # threading.Lock cannot be acquired twice by the same thread).
            new_value = compute_fn(self.mood_value, self.last_updated)

            now = time()
            self.mood_value = new_value
            self.last_updated = now
            self.last_interaction = now
            return new_value

    def decide_preset(self, decision_fn):
        """
        Atomic version of "work out which preset (angry/annoyed/.../
        sleepy) JD should currently show, possibly updating
        last_shown_preset in the process" - fixing the same category of
        lost-update race as apply() above, now applied to
        get_current_preset()'s multi-step decision (sleepy check, band
        lookup, hysteresis comparison, last_shown_preset update), which
        previously touched self._lock four to five separate times with
        no atomicity across the whole decision.

        This became a real risk (not just theoretical) once
        get_current_preset() started being polled from more than one
        place concurrently - a 5-second C#-side timer, AND a
        fire-and-forget call fired every time JD finishes speaking.

        decision_fn is a CALLBACK, same mechanism as apply() above: a
        function services.py hands in, called by THIS method, from
        inside the lock. It receives everything it needs as plain
        arguments (mood_value, last_updated, last_interaction,
        last_shown_preset) and must return a
        (preset_to_report, new_last_shown_preset_to_save) tuple.

        CRITICAL RULE for whatever function is passed in here: because
        it runs WHILE self._lock is already held, it must NEVER call
        any of this class's OTHER locked methods (read(), write(),
        seconds_since_interaction(), get_last_shown_preset(), etc.) -
        doing so would try to acquire self._lock a second time from the
        same thread and DEADLOCK PERMANENTLY (a plain threading.Lock is
        not reentrant). It must only use the raw values it was handed as
        arguments, plus plain functions like time() or
        _decayed_value() that do not touch self._lock at all. See
        services.py's get_current_preset() for the real implementation
        of this rule.
        """
        with self._lock:
            preset, new_last_shown = decision_fn(
                self.mood_value,
                self.last_updated,
                self.last_interaction,
                self.last_shown_preset,
            )
            self.last_shown_preset = new_last_shown
            return preset

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