import os
from google import genai

# Creates one single Gemini client at module load time (We reuse it forever)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

async def get_llm_response(text: str) -> str:
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=text
    )
    return response.text