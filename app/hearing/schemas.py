from pydantic import BaseModel


class TranscribeResponse(BaseModel):
    """
    Returned by POST /hearing/transcribe.

    transcript  — the spoken text Whisper decoded, or None if Silero VAD
                  determined the audio was noise/silence (not real speech).
    is_speech   — explicit boolean so the C# caller can cheaply check whether
                  it should bother calling /brain/chat at all, without having
                  to inspect the transcript field for None.
    """
    transcript: str | None = None       # Transcript can be either a string or nothing at all
    is_speech: bool