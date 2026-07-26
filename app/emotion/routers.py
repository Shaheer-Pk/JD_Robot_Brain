"""
app/emotion/routers.py

[CONTROLLER] Dev-only route for the emotion module. Right now this is
the ONLY way to see JD's mood - the Python -> ARC reverse channel
doesn't exist yet (see arc-csharp.md / jd-context-main.md). Once that
channel is built, whatever get_current_preset() returns gets pushed to
ARC's C# skill instead of (or as well as) sitting behind this route.
"""

import asyncio

from fastapi import APIRouter

from app.emotion.services import get_current_preset, get_eye_preset_details, get_current_mood_value
from app.emotion.schema import EmotionStateResponse

router = APIRouter()


@router.get("/state", response_model=EmotionStateResponse)
async def get_emotion_state():
    """
    Returns JD's current mood preset, its color, and the ARC action name
    it should trigger - all computed fresh on every call from the decay
    math in services.py. Nothing here is cached.
    """

    # Run independent synchronous calls concurrently in worker threads
    preset_task = asyncio.to_thread(get_current_preset)
    current_value_task = asyncio.to_thread(get_current_mood_value)

    # Wait for both to finish in parallel
    preset, current_value = await asyncio.gather(preset_task, current_value_task)

    # Run details after preset is known
    details = await asyncio.to_thread(get_eye_preset_details, preset)
    

    return EmotionStateResponse(
        preset=preset,
        color=details["color"],
        action_name=details["action_name"],
        mood_value=round(current_value, 3),
        is_sleepy=(preset == "sleepy"),
    )   