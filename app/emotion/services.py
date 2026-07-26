"""
app/emotion/services.py

[CONTROLLER-ADJACENT SERVICE LOGIC] All the actual "thinking" behind
JD's simulated mood lives here: the decay formula, event penalties/
bonuses, the sleepy override, hysteresis, and mapping a raw mood number
to an eye preset name. state.py deliberately holds no logic - only raw
numbers behind a lock - so this file is the ONLY place these rules are
written down. If a number ever needs retuning (e.g. the decay rate),
there is exactly one place to change it.
"""

import json
import os
from time import time

from app.emotion.state import mood_state

# Load emotion_profile.json once at module import time, same pattern
# brain/services.py already uses for robot_profile.json. Never re-read
# per request - if these numbers need retuning, restart the server.
with open(os.path.join(os.path.dirname(__file__), "emotion_profile.json"), "r") as f:
    _profile = json.load(f)

# Done to make changing these values easy (just edit emotion_profile.json)
REPEAT_PENALTY = _profile["repeat_penalty"]
RUDE_PENALTY = _profile["rude_penalty"]
NICE_BONUS = _profile["nice_bonus"]
DECAY_PER_MINUTE = _profile["decay_per_minute"]
HYSTERESIS_BUFFER = _profile["hysteresis_buffer"]       # prevent flickering state of mood/light 
SLEEPY_IDLE_MINUTES = _profile["sleepy_idle_minutes"]
BANDS = _profile["bands"]                  # e.g. {"angry": [-1.0, -0.6], ...}
EYE_PRESETS = _profile["eye_presets"]       # e.g. {"angry": {"color": ..., "action_name": ...}}


def _decayed_value(raw_value: float, last_updated: float) -> float:
    """
    Works out what the mood value SHOULD be right now, given how much
    time has passed since it was last written - no background timer
    needed. Instead of a clock constantly ticking the value down, we just do
    the math whenever someone actually communicates with the robot?"

    Moves the value toward 0.0 (neutral) at a fixed rate per minute, but
    never overshoots past neutral in either direction.
    """
    elapsed_minutes = (time() - last_updated) / 60.0
    decay_amount = DECAY_PER_MINUTE * elapsed_minutes         # Lazy decay

    if raw_value > 0:       # Check if its more to the excited state
        return max(0.0, raw_value - decay_amount)
    elif raw_value < 0:     # Check if its more to the angry state
        return min(0.0, raw_value + decay_amount)
    return 0.0              # Incase of complete neutral


def get_current_mood_value() -> float:
    """
    Public function for "what is JD's mood right now, decay included?"
    without changing anything in state.py. Exists so other files (like
    routers.py) never need to reach into the private _decayed_value
    helper directly - there is exactly one way to ask this question.
    """
    raw_value, last_updated = mood_state.read()    
    # Apply lazy decay here to give accurate mood value after natural decay
    return _decayed_value(raw_value, last_updated)      


def apply_event(is_repeat: bool, user_tone: str):
    """
    Called once per completed chat turn - and ONLY if Gemini's call
    actually succeeded. A failed call should never move JD's mood, same
    rule the conversation memory feature already follows for not
    recording phantom exchanges.

    Order of operations matters:
      1. Work out what the mood has ALREADY decayed to since it was last
         touched (so a repeat-question 10 minutes after the last chat
         isn't judged against a 10-minute-stale number).
      2. Add whatever just happened on top of that already-decayed value or 
         just add the decay in case of neutral conversation
      3. Clamp the result so it can never go past -1.0 or +1.0.
      4. Save it to update mood value - which also resets the sleepy timer.

    user_tone is expected to be exactly one of "nice", "rude", "neutral" -
    enforced upstream by Gemini's locked response schema, so no other
    value should EVER reach this function.

    ------------------------------------------------------------------
    ATOMICITY FIX (this session): read + compute + write via a CALLBACK
    ------------------------------------------------------------------
    Previously this function called mood_state.read(), did math, then
    separately called mood_state.write(new_value) - two separate lock
    acquisitions with an unprotected gap between them. Two overlapping
    calls to apply_event() (e.g. two rapid requests) could both read()
    the SAME starting value before either had written back, and the
    second write() would silently overwrite the first - a "lost update"
    race. See state.py's module docstring for a fully traced-out example
    of exactly this happening.

    The fix: delta is computed first (pure math, no shared state
    involved, safe to do outside any lock). Then a small function -
    `compute` - is defined right here, INSIDE apply_event(), and handed
    over to mood_state.apply(compute) UNRUN (no parentheses after
    `compute` in the call below - we are passing the function itself,
    not its result). mood_state.apply() is the one that actually calls
    `compute(...)`, and it does so from inside its own `with
    self._lock:` block - see state.py. That means the entire
    decay-then-add-delta-then-clamp computation now happens while the
    lock is held, closing the gap that caused the race.

    `compute` can still see `delta` even though it gets CALLED later,
    from a different file (state.py) - this is a Python "closure":
    `compute` remembers the variables from its surrounding scope
    (apply_event's local `delta`) at the moment it was defined, and
    keeps that value attached to itself no matter where it's later
    invoked from. You do not need to pass delta in as an argument.
    """
    # delta only depends on is_repeat/user_tone, both already fully known
    # here - no shared/lockable state involved, so this part is fine to
    # compute before touching mood_state at all.
    delta = 0.0
    # Add more conditions in case of more triggers like is_repeat
    if is_repeat:
        delta += REPEAT_PENALTY
    if user_tone == "rude":
        delta += RUDE_PENALTY
    elif user_tone == "nice":
        delta += NICE_BONUS
    # user_tone == "neutral" contributes nothing - deliberate no-op.

    def compute(raw_value: float, last_updated: float) -> float:
        """
        The callback. Defined fresh every time apply_event() runs (cheap
        in Python - this is normal, not wasteful). Takes the RAW,
        not-yet-decayed mood_value and its last_updated timestamp -
        exactly what mood_state.apply() will hand it, straight from
        MoodState's own attributes, taken WHILE the lock is held.

        Must NOT call any mood_state method itself (read/write/etc.) -
        it runs from inside mood_state.apply()'s lock, and this class's
        lock is not reentrant; calling another locked method here would
        deadlock permanently. It only uses the two plain values it was
        handed, plus the already-computed `delta` from the enclosing
        scope (via the closure explained above) and the module-level
        _decayed_value() helper (which does not touch the lock at all).
        """
        current_value = _decayed_value(raw_value, last_updated)
        new_value = current_value + delta
        return max(-1.0, min(1.0, new_value))  # clamp to the dial's range

    # Hand the function itself to apply() - note: `compute`, NOT
    # `compute()`. apply() will call it, exactly once, from inside its
    # lock, and save whatever it returns. See state.py's apply() for
    # the receiving side of this call.
    mood_state.apply(compute)


def _band_for_value(value: float) -> str:
    """
    Finds which named band (angry/annoyed/neutral/happy/excited) a raw
    mood number falls into, using emotion_profile.json's band ranges.
    Deliberately does NOT apply hysteresis here - that needs to know the
    PREVIOUSLY shown preset, which is one level of context this function
    has no reason to carry. Incase of lets say -0.6 where it is true for 
    both "angry" and "annoyed" it will be set to "angry" as that is checked
    first in the bands (dependent on dictionary insertion order)
    """
    for band_name, (low, high) in BANDS.items():
        if low <= value <= high:
            return band_name
    return "neutral"  # safe fallback - should never actually be reached


def get_current_preset() -> str:
    """
    The single function anything outside this module should call to ask
    "what should JD's eyes show right now?". Handles, in order:
      1. The sleepy override - ignores the mood number completely, since
         sleepy is a separate switch, not a point on the mood dial.
      2. The hysteresis buffer - so a value sitting right on a boundary
         doesn't flicker rapidly between two presets.

    ------------------------------------------------------------------
    ATOMICITY FIX (this session): whole decision via a CALLBACK
    ------------------------------------------------------------------
    Previously this function made 4-5 SEPARATE calls into mood_state
    (seconds_since_interaction, read (via get_current_mood_value),
    get_last_shown_preset, set_last_shown_preset) - each individually
    lock-safe, but with no atomicity across the whole decision. This was
    harmless while only one conversation turn could ever be in flight at
    a time, but became a real risk this session: get_current_preset() is
    now polled from TWO independent, genuinely concurrent places on the
    C# side (a 5-second timer, and a fire-and-forget call fired every
    time JD finishes speaking) - so two calls can now genuinely overlap
    and interleave badly, producing a wrong/flickery hysteresis decision.

    Fix: same pattern as apply_event() above. `decision` is defined
    below and handed to mood_state.decide_preset(decision) UNRUN.
    decide_preset() calls it exactly once, from inside its own lock,
    passing in the raw snapshot values it needs as plain arguments. The
    entire sleepy-check + band-lookup + hysteresis-compare + "what do we
    save as last_shown_preset" decision now happens as one uninterrupted
    unit - no other thread can read or write mood_state's fields midway
    through this decision anymore.
    """

    def decision(raw_value: float, last_updated: float, last_interaction: float, last_shown):
        """
        The callback. Receives everything as plain values, taken by
        decide_preset() straight from MoodState's attributes while its
        lock is held. Must return (preset_to_report,
        new_last_shown_preset_to_save).

        CRITICAL: this function runs FROM INSIDE mood_state's lock (see
        decide_preset() in state.py). It must never call
        mood_state.read(), mood_state.seconds_since_interaction(),
        mood_state.get_last_shown_preset(), or any other locked
        MoodState method - MoodState's lock is not reentrant, so doing
        so would deadlock this entire endpoint permanently, silently,
        the very first time it's hit. That's why the sleepy check below
        uses bare time() math on the last_interaction argument instead
        of calling the existing seconds_since_interaction() helper -
        that helper internally re-acquires the lock, which is exactly
        what must be avoided here.
        """
        # Sleepy check happens first and overrides everything below.
        # By design, sleepy mode isn't part of the -1.0 to 1.0 scale at
        # all. (Reimplemented inline with bare time() math rather than
        # calling mood_state.seconds_since_interaction() - see the
        # CRITICAL note above for why.)
        if (time() - last_interaction) >= (SLEEPY_IDLE_MINUTES * 60):
            return "sleepy", "sleepy"

        current_value = _decayed_value(raw_value, last_updated)  # accurate mood with lazy decay
        candidate = _band_for_value(current_value)                # candidate band

        # Only actually switch presets if the value has moved past the
        # boundary by more than the buffer amount. Otherwise, keep
        # showing whatever was shown last time, even if the raw number
        # has crossed into the next band's range by only a small amount
        # e.g. 0.21.
        if last_shown is not None and last_shown != "sleepy" and candidate != last_shown:
            boundary_low, boundary_high = BANDS[last_shown]
            still_within_buffer = (
                (boundary_low - HYSTERESIS_BUFFER) <= current_value <= (boundary_high + HYSTERESIS_BUFFER)
            )
            if still_within_buffer:  # e.g. 0.21 - still within buffer + band range
                # Report the OLD preset, and explicitly re-save it as
                # last_shown too (not strictly required since it's
                # unchanged, but keeps the return shape uniform/obvious).
                return last_shown, last_shown

        # Value crossed the buffer + band range - candidate becomes the
        # new preset, and also what gets saved as last_shown_preset.
        return candidate, candidate

    # Hand the function itself to decide_preset() - note: `decision`,
    # NOT `decision()`. decide_preset() will call it, exactly once, from
    # inside its lock, save the returned last_shown_preset, and give us
    # back just the preset to report. See state.py's decide_preset().
    return mood_state.decide_preset(decision)


def get_eye_preset_details(preset_name: str) -> dict:
    """
    Looks up the color + ARC "Action" name for a given preset, straight
    from emotion_profile.json. Used by the dev-only test route now, and
    later by whatever code sends this to the C# skill once the Python ->
    ARC reverse channel exists.

    IMPORTANT: action_name must match, character-for-character, the name
    of the Action built inside ARC's RGB Animator editor. That matching
    is a coordination point with whoever builds the ARC side - nothing
    here can detect a typo'd mismatch on its own.
    """
    return EYE_PRESETS.get(preset_name, EYE_PRESETS["neutral"])