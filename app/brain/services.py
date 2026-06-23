import os
import json
import io
import wave
# import httpx          # Use for elevenlabs text_to_speech

from piper import PiperVoice
from google import genai
from google.genai import types

# Importing from our env file (hidden)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# Load the Piper voice once at module load time, same pattern as the Gemini client
piper_voice = PiperVoice.load("voices/en_US-danny-low.onnx")

# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")          # Uncomment when using elevenlabs and not piper
# ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# Load robot_profile.json once at module load time, never re-read per request
# os.path.dirname(__file__) is an inbuilt python vari which fetches file path relatively so for this its something like
# D:\ARC JD Project\JD_Robot_Brain\app\brain\services.py
# Then when we do os.path.dirname we cut off servicies.py at the end to get:
# D:\ARC JD Project\JD_Robot_Brain\app\brain
# And then with os.path.join we get 
# D:\ARC JD Project\JD_Robot_Brain\app\brain\robot_profile.json
# This helps in dynamically searching for robot_profile.json specifically in brain independent from which folder uvicorn is run from
with open(os.path.join(os.path.dirname(__file__), "robot_profile.json"), "r") as f:
    _profile = json.load(f)

# Key:Value pairs from \brain\robot_profile.json
IDENTITY_AND_CAPABILITIES = _profile["identity_and_capabilities"]
DEFAULT_GUEST_PERSONA = _profile["default_guest_persona"]


async def get_llm_response(text: str, custom_personality: str | None = None) -> str:
    persona = custom_personality if custom_personality else DEFAULT_GUEST_PERSONA
    system_prompt = IDENTITY_AND_CAPABILITIES + "\n\n" + persona        # Evaluated at runtime (based on face recognition future work)

    response = await client.aio.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
        )
    )
    return response.text

# UNCOMMENT if you want to use eleven labs
# async def text_to_speech(text: str) -> bytes:
#     url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
#     headers = {
#         "xi-api-key": ELEVENLABS_API_KEY,
#         "Content-Type": "application/json"
#     }
#     payload = {"text": text}

#     async with httpx.AsyncClient() as client:
#         response = await client.post(url, json=payload, headers=headers)
#         response.raise_for_status()
#         return response.content

# Piper version
async def text_to_speech(text: str) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        piper_voice.synthesize_wav(text, wav_file)
    return buffer.getvalue()