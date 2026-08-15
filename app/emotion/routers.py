"""
app/emotion/routers.py

[CONTROLLER] Live, production route for the emotion module - polled by
ARC's MoodPoller.cs every 5 seconds AND fired on-demand right after
every conversational turn (see emotion-arc-csharp.md). This is NOT a
dev-only test route.

A Python -> ARC reverse-channel design (Python actively pushing mood
updates into ARC via an inbound HttpListener) was considered and
explicitly REJECTED - see emotion-arc-csharp.md's "Rejected: the
Reverse-Channel / HttpListener Design" section for the full rationale.
What was actually built instead is a plain polling model: ARC pulls
from this route on a timer plus an on-demand trigger; Python never
pushes. This keeps the project's networking direction uniformly
C# -> Python, unchanged from every other feature.
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

    # IMPORTANT NOTE:
        # SEQUENTIAL, not asyncio.gather - deliberate, fixed this session.
        # get_current_preset() and get_current_mood_value() are each
        # individually atomic (MoodState.apply()/decide_preset()), but
        # running them CONCURRENTLY via gather made them two independent,
        # interleavable lock acquisitions with no atomicity relative to
        # EACH OTHER: an apply_event() write landing between them could
        # produce one response where mood_value and preset reflect two
        # different underlying mood states. Sequential awaits close that
        # window. Deliberate perf-for-correctness trade - two sequential
        # thread-pool hops instead of two parallel ones, immaterial cost
        # given both are near-instant lock-guarded reads. Do not "optimize"
        # this back to gather() without re-introducing the race.
    preset = await asyncio.to_thread(get_current_preset)
    current_value = await asyncio.to_thread(get_current_mood_value)

    details = await asyncio.to_thread(get_eye_preset_details, preset)

    return EmotionStateResponse(
        preset=preset,
        color=details["color"],
        action_name=details["action_name"],
        mood_value=round(current_value, 3),
        is_sleepy=(preset == "sleepy"),
    )