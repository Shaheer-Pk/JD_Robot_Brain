import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = "You are JD, a friendly campus assistant robot. Keep your answers short, clear and conversational. Never mention being an AI or a language model."

async def get_llm_response(text: str) -> str:
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        )
    )
    return response.text