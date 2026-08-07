import threading


class PoseState:
    """
    Holds JD's current tracked physical pose as a single value — NOT a
    history, unlike ConversationMemory. Exactly two valid values in this
    project's current design: "standing" or "sitting". Starts at
    "standing" at process boot (matches JD's real default resting frame).

    This class owns ONLY state management — one value, one lock. It has
    zero knowledge of Gemini, prompt formatting, verify_actions()'s
    simulation logic, or robot_profile.json's action metadata. That
    reasoning belongs entirely in brain/services.py and brain/routers.py.

    Deliberately given its OWN dedicated lock, separate from
    ConversationMemory's lock in app/shared/memory.py, even though both
    are brain-adjacent state. Pose and conversation text are unrelated
    concerns — locking them together would mean a pose read has to wait
    behind an unrelated memory write for no reason. Mirrors the same
    reasoning app/emotion/state.py's MoodState already established for
    why mood value and last-shown-preset share ONE lock (they're the
    same concern) — pose and conversation text are not the same concern,
    so they do not share a lock, even living in the same project area.

    CRITICAL, GENUINE LIMITATION — read before trusting this class's
    value as physical fact. This tracked pose is a PREDICTION, not a
    confirmed hardware readout. There is no confirmation channel
    anywhere in this pipeline (dispatch to ARC is fire-and-forget via
    VariableManager + watcher-script polling — see arc-csharp.md) that
    tells Python whether a dispatched action actually executed
    successfully on real hardware. set_pose() is called by
    brain/routers.py based on what verify_actions() SIMULATED would
    happen if the verified batch executes as expected — not based on
    any real confirmation that it did. A dropped dispatch, a stalled
    AutoPositionActionWait mid-sequence, or a human physically moving
    JD between turns will silently desync this value from JD's real
    physical pose, with no detection mechanism for that drift anywhere
    in this codebase. The pose guard-rail built on top of this value
    (in brain/services.py's verify_actions()) is only as trustworthy as
    this assumption — treat it as raising the safety ceiling, not as a
    guarantee.

    Owned by app/brain/, NOT app/shared/ — per jd-context-main.md's
    app/shared/ ownership test ("would this still make sense with its
    current caller stripped away"), tracked_pose has exactly one writer
    (brain/routers.py, post-verification) and one reader
    (brain/services.py, prompt injection). No cross-module reader exists
    the way vision/state.py or app/shared/memory.py's memory_session
    genuinely have. If a future feature needs cross-module pose access,
    revisit this placement at that time — don't assume it in advance.
    """

    def __init__(self, initial_pose: str = "standing"):
        self._lock = threading.Lock()
        self._pose = initial_pose

    def get_pose(self) -> str:
        with self._lock:
            # A bare str is immutable — no copy-on-read concern here,
            # unlike ConversationMemory.get_history()'s list-reference
            # problem. Returning self._pose directly is safe.
            return self._pose

    def set_pose(self, new_pose: str) -> None:
        with self._lock:
            self._pose = new_pose


# Module-level singleton, same instantiate-once-at-import pattern as
# app/shared/memory.py's memory_session. Imported by brain/routers.py
# (read before the Gemini call, write after verification) and passed
# into brain/services.py as a plain parameter — services.py does NOT
# import pose_session directly, per the earlier confirmed decision to
# keep pose-context flowing the same way mood_context already does.
pose_session = PoseState(initial_pose="standing")