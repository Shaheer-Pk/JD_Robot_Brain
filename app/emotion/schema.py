"""
[MODEL] Pydantic schema for the emotion module's live production route,
GET /emotion/state - polled by ARC's MoodPoller.cs (see
emotion-arc-csharp.md). Kept separate from services.py so anything
reading this shape doesn't need to pull in any of the actual
mood-calculation logic.
"""

#Using pydantic to enfore data validation (return correct datatype)
from pydantic import BaseModel

class EmotionStateResponse(BaseModel):
    """
    What GET /emotion/state returns. Originally built to verify the mood
    math through a browser/Swagger UI; now also the real, live contract
    ARC polls to drive JD's RGB eyes. A Python -> ARC push design
    (reverse HttpListener channel) was considered and rejected in favor
    of this pull/poll model - see emotion-arc-csharp.md.
    """

    preset: str          # e.g. "happy", "angry", "sleepy"
    color: str           # e.g. "green" - from emotion_profile.json
    action_name: str     # must match an ARC RGB Animator "Action" name EXACTLY
    mood_value: float    # the raw, decay-applied dial value, -1.0 to 1.0
    is_sleepy: bool      # flag to check if robot should go in sleepy mode