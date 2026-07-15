from fastapi import APIRouter
from fastapi.responses import Response
from app.brain.schemas import ChatRequest
from app.brain.services import get_llm_response, text_to_speech
from app.emotion.services import apply_event, get_current_preset   # read + write

router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    current_mood = get_current_preset()         # Read mood before Gemini call
    spoken_text, is_repeat, user_tone = await get_llm_response(request.text, mood_context=current_mood)
    # apply event from emotions module for mood to affect the response (write after response)
    apply_event(is_repeat, user_tone)     
    audio_bytes = await text_to_speech(spoken_text)         # spoken text -> audio bytes
    return Response(content=audio_bytes, media_type="audio/wav")    # wav format through piper tts (for elevenlab its mpeg)
