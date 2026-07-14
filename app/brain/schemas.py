# Validating data-type so we only take string as input and give string as an output

from pydantic import BaseModel
from typing import Literal

class ChatRequest(BaseModel):
    text: str

# Potentially dead code not confirmed though yet — endpoint returns raw
# audio bytes via Response(), not this model
class ChatResponse(BaseModel):
    response: str

"""
Change our Gemini response format from a string to a JSON with a "response" 
in string that will become audio bytes and an "is_repeat" boolean check 
to see if the question is a repeat (made JD's mood a bit angry) and check the 
user tone (literal strings) to change JD's mood accordingly.
"""
class LLMTurnResult(BaseModel):
    response: str
    is_repeat: bool
    user_tone: Literal["nice", "rude", "neutral"]