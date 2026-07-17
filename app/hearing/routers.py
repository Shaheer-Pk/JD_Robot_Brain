import asyncio

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.hearing.schemas import TranscribeResponse
from app.hearing.services import transcribe_audio

router = APIRouter()


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)):
    """
    Receives a raw WAV file upload from the C# skill.
    Runs Silero VAD to confirm it's real speech.
    If confirmed, runs Faster Whisper and returns the transcript.
    If noise/silence, returns is_speech=False with no transcript.

    The C# caller should check is_speech before deciding whether
    to forward the transcript to /brain/chat.
    """

    # Read the raw bytes from the uploaded file
    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received.")

    try:
        is_speech, transcript = await asyncio.to_thread(transcribe_audio, audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription pipeline failed: {str(e)}")

    return TranscribeResponse(
        transcript=transcript,
        is_speech=is_speech
    )