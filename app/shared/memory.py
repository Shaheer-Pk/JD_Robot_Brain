import threading

class ConversationMemory:
    """
    Holds a sliding window of conversational turns for a single in-progress
    session. Each turn is a dict shaped {"user": str, "jd": str}.

    This class owns ONLY state management. It knows nothing about Gemini,
    prompt formatting, or how the data will eventually be sent to an LLM.
    That translation step belongs in brain/services.py.

    reset() and seed() are intentionally unused right now. They exist as
    forward-compatible hooks for the future face-recognition trigger
    (reset on new face) and DB-loaded history (seed on recognized user).

    ACTION-HISTORY DESIGN NOTE — added this session, NO STRUCTURAL CHANGE
    MADE. As of this session, brain/routers.py records executed physical
    action history by APPENDING a human-readable note directly onto the
    `jd_text` string BEFORE calling add_turn() — e.g. "Sure!" becomes
    "Sure! [JD physically performed: StandFromSit, Wave]". This class's
    shape and add_turn()'s 2-argument signature are UNCHANGED — action
    history is fused into the existing `jd` field, not stored as a third
    field.

    This was a deliberate, discussed tradeoff, not an oversight:
    - Chosen because Gemini's own Content/Part API has no separate
      channel for "action" data distinct from spoken text anyway — a
      3rd field would still collapse into one concatenated string by
      the time it reaches Gemini's prompt, just built one layer later
      (in services.py instead of here).
    - Chosen because it required zero changes to this class's signature,
      zero changes to get_history()'s callers, and zero changes to
      services.py's history-reconstruction loop — smallest possible
      blast radius on the one file this project's own history
      (refinement-optimization.md) already flagged as too risky to
      touch carelessly close to a deadline.

    KNOWN, ACCEPTED DOWNSIDE — worth reading before extending this
    further: because the action note is permanently fused into
    `jd_text`, a stored turn cannot be cleanly split back into "what JD
    said" vs. "what JD did" for any future purpose (e.g. separate
    display, separate logging, separate analytics) without fragile
    string-parsing.

    RECOMMENDATION FOR FUTURE SCALABILITY, IF THIS PROJECT CONTINUES
    PAST ITS CURRENT DEMO SCOPE: break `_history` into a genuine 3-field
    shape — {"user": str, "jd": str, "actions": list[str] | None} — with
    add_turn() gaining a third optional parameter, and let
    services.py's Content-building loop concatenate `jd` + `actions`
    into the single string Gemini's API requires AT THAT POINT, not
    here. This restores a clean separation of concerns at the storage
    layer, at the cost of updating every caller of add_turn() and
    get_history() — a real but bounded, well-understood migration,
    deliberately deferred rather than attempted under this session's
    time constraints.
    """

    def __init__(self, max_turns: int):
        self._lock = threading.Lock()
        self._max_turns = max_turns
        self._history: list[dict[str, str]] = []

    def add_turn(self, user_text: str, jd_text: str) -> None:
        with self._lock:
            # Append the new turn unconditionally — this is the only place
            # in the whole codebase where a turn gets constructed, so the
            # {"user", "jd"} shape can never drift or go out of sync.
            self._history.append({"user": user_text, "jd": jd_text})

            # Trim exactly one turn from the front if we've gone over the cap.
            # No loop needed — at most one turn can ever be over the limit,
            # since add_turn() only ever adds one turn per call.
            if len(self._history) > self._max_turns:
                self._history.pop(0)

    def get_history(self) -> list[dict[str, str]]:
        with self._lock:
            # Return a COPY, not the live internal list. If this returned
            # self.history directly, the lock only protects the moment of
            # the return statement — the caller then holds a raw reference
            # to the SAME mutable list. If brain/services.py starts
            # iterating that returned list while another thread's
            # add_turn()/reset() mutates it concurrently, you get either a
            # "list changed size during iteration" crash or silently
            # inconsistent data — completely outside the lock's protection,
            # because the lock has already been released by the time
            # iteration happens.

            # Raw structured data only. No formatting, no Gemini-specific shape.
            return list(self._history)

    def reset(self) -> None:
        with self._lock:
            # Future hook: called when a NEW recognized/unrecognized face
            # is detected, wiping the current user's in-session memory.
            self._history = []

    def seed(self, past_turns: list[dict[str, str]]) -> None:
        with self._lock:
            # Future hook: called when a recognized user's last few turns
            # are pulled from the database at the start of a new session.
            self._history = list(past_turns)

memory_session = ConversationMemory(max_turns=10)