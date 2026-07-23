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

action: NEW — Gemini's freeform guess at a physical action/soundboard
keyword that fits this moment, or None if nothing fits. Deliberately NOT
a Literal enum, even though the whole list is finite and known — the list
is long (~38 entries) and will grow, and constraining it at the schema
level would mean every new action requires a schema edit + redeploy on
top of the robot_profile.json edit. Free text here is caught by
verify_action()'s dict-based guardrail in services.py instead — Gemini is
never trusted to have generated a real, executable action on its own.
"""
class LLMTurnResult(BaseModel):
    response: str
    is_repeat: bool
    user_tone: Literal["nice", "rude", "neutral"]
    action: str | None