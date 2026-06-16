from fastapi import APIRouter
from fastapi.responses import Response
from app.brain.schemas import ChatRequest
from app.brain.services import get_llm_response, text_to_speech

router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    llm_response = await get_llm_response(request.text)
    audio_bytes = await text_to_speech(llm_response)
    return Response(content=audio_bytes, media_type="audio/wav")    # wav format through piper tts (for elevenlab its mpeg)