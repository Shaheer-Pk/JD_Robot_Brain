"""
app/emotion/schema.py

[MODEL] Pydantic schema for the emotion module's dev-only test route.
Kept separate from services.py so anything reading this shape doesn't
need to pull in any of the actual mood-calculation logic.
"""

#Using pydantic to enfore data validation (return correct datatype)
from pydantic import BaseModel

class EmotionStateResponse(BaseModel):
    """
    What GET /emotion/state returns. Lets us verify the mood math works
    correctly through a browser or FastAPI's /docs Swagger UI, before
    ARC can receive pushed commands at all (the Python -> ARC reverse
    channel doesn't exist yet).
    """

    preset: str          # e.g. "happy", "angry", "sleepy"
    color: str           # e.g. "green" - from emotion_profile.json
    action_name: str     # must match an ARC RGB Animator "Action" name EXACTLY
    mood_value: float    # the raw, decay-applied dial value, -1.0 to 1.0
    is_sleepy: bool      # flag to check if robot should go in sleepy mode