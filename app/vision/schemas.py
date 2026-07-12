"""
app/vision/schemas.py — Pydantic schemas for vision/. Two genuinely
different contracts, kept in one file for now (no vision/schemas/ folder
yet — split apart per jd-context-main.md's flat-by-default rule if a
third schema shows up).

RecognitionStatus — single source of truth for state.py's recognition_status
values (idle/pending/identified/guest). Defined once here rather than as
a bare string comment duplicated across files.

StreamStatusResponse — the HTTP response body for POST /vision/stream
(vision/routers.py), wired as that route's response_model. What C#
actually receives on every frame POST — presence/listening/mouth/status
ONLY, never identity.

VisionProfile — the in-process (NEVER serialized over HTTP) contract
between vision's recognition pipeline and brain/. Built by identify_face()
(vision/services/identification_services.py), stored via
session.mark_identified() (vision/state.py), read via
session.get_profile() from brain/routers.py. Enforced as a real model, not
a raw dict, so a future shape change happens visibly here, with a reason
— not as a key silently added at some call site with nothing to check it.
"""

from typing import Literal
from pydantic import BaseModel

RecognitionStatus = Literal["idle", "pending", "identified", "guest"]


class StreamStatusResponse(BaseModel):
    face_present: bool
    is_listening: bool
    mouth_closed_seconds: float
    recognition_status: RecognitionStatus


class VisionProfile(BaseModel):
    user_id: int
    name: str
    persona: str
    memory_context: str | None
    confidence: float