"""
app/brain/routers.py — The HTTP entrypoint the C# skill hits after
hearing finishes transcribing an utterance (POST /brain/chat, per
hearing-feature.md's pipeline). Ties conversation (Gemini), identity
(vision), and speech (Piper TTS) into one request/response cycle: text
in, synthesized WAV audio out.

Identity comes from app.vision.state.session, NOT from anything the
client sends — ChatRequest carries only {"text": str}, deliberately
unchanged. The C# skill has no way to know who's currently recognized
(that state lives entirely server-side, inside the same Python process
vision runs in) — this route reads session.get_profile() itself at
request time rather than requiring identity to be passed over the wire
and duplicated into a second place that could drift out of sync with
vision's own state.

profile is None in two cases NOT distinguished here, on purpose: truly
unrecognized (recognition_status == "guest") AND still mid-recognition
(recognition_status == "pending", not yet resolved). A chat request
arriving during the pending window gets the guest persona for that one
turn even if the person would have been correctly identified a moment
later — accepted tradeoff, not a bug (confirmed decision — not worth
holding/delaying a chat request on an in-flight background recognition
attempt).

When profile exists, only name and persona are pulled into
custom_personality — NOT the full vision dict (which also carries
user_id, memory_context, confidence). Deliberate narrowing at the module
boundary: brain/services.py should depend on "a name and a persona
string," not on vision's internal return shape — if identify_face's
return shape changes later, brain/ doesn't silently break.
"""

from fastapi import APIRouter
from fastapi.responses import Response
from app.brain.schemas import ChatRequest
from app.brain.services import get_llm_response, text_to_speech
from app.vision.state import session

router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    profile = session.get_profile()
    custom_personality = None
    if profile is not None:
        custom_personality = {"name": profile.name, "persona": profile.persona}

    llm_response = await get_llm_response(request.text, custom_personality=custom_personality)
    audio_bytes = await text_to_speech(llm_response)
    return Response(content=audio_bytes, media_type="audio/wav")