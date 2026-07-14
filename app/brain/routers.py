from fastapi import APIRouter
from fastapi.responses import Response
from app.brain.schemas import ChatRequest
from app.brain.services import get_llm_response, text_to_speech
from app.emotion.services import apply_event        # Get the mood to show emotion in response  

router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    spoken_text, is_repeat, user_tone = await get_llm_response(request.text)
    # apply event from emotions module for mood to affect the response
    apply_event(is_repeat, user_tone)   
    audio_bytes = await text_to_speech(spoken_text)         # spoken text -> audio bytes
    return Response(content=audio_bytes, media_type="audio/wav")    # wav format through piper tts (for elevenlab its mpeg)
