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
    """
    raw_value, last_updated = mood_state.read()         # Read current mood
    current_value = _decayed_value(raw_value, last_updated)     # Add decay based on time elapsed

    # Change for mode represented by delta
    delta = 0.0
    # Add more conditions in case of more triggers like is_repeat
    if is_repeat:
        delta += REPEAT_PENALTY
    if user_tone == "rude":
        delta += RUDE_PENALTY
    elif user_tone == "nice":
        delta += NICE_BONUS
    # user_tone == "neutral" contributes nothing - deliberate no-op.

    # Removed a check for delta==0.0 as we want the natural decay to
    # happen even in the case of a neutral conversation the same way

    new_value = current_value + delta           # Update the new mood value (decay + change)
    new_value = max(-1.0, min(1.0, new_value))  # clamp to the dial's range

    mood_state.write(new_value)     # Updates the last interaction and last update here


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
    """
    # Sleepy check happens first and overrides everything below
    # By design, sleepy mode isn't part of the -1.0 to 1.0 scale at all.
    if mood_state.seconds_since_interaction() >= (SLEEPY_IDLE_MINUTES * 60):
        mood_state.set_last_shown_preset("sleepy")
        return "sleepy"

    current_value = get_current_mood_value()        # returns accurate current mood with lazy decay
    candidate = _band_for_value(current_value)      # candidate band where we should set

    last_shown = mood_state.get_last_shown_preset()     # getting the last preset for hysterisis

    # Only actually switch presets if the value has moved past the
    # boundary by more than the buffer amount. Otherwise, keep showing
    # whatever was shown last time, even if the raw number has crossed
    # into the next band's range by only a small amount e.g 0.21.
    if last_shown is not None and last_shown != "sleepy" and candidate != last_shown:
        boundary_low, boundary_high = BANDS[last_shown]     # Get bands boundries
        # Check if the slightly changed value is in the buffer + band range
        still_within_buffer = (
            (boundary_low - HYSTERESIS_BUFFER) <= current_value <= (boundary_high + HYSTERESIS_BUFFER)
        )
        if still_within_buffer:     # if value like 0.21 (within buffer + band range)
            return last_shown       # give previous preset

    # If value crossed buffer + band range then set candidate band as new preset
    mood_state.set_last_shown_preset(candidate)     
    return candidate            # return candidate


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