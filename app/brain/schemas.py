# Validating data-type so we only take string as input and give string as an output

from pydantic import BaseModel
from typing import Literal

class ChatRequest(BaseModel):
    text: str

"""
Change our Gemini response format from a string to a JSON with a "response" 
in string that will become audio bytes and an "is_repeat" boolean check 
to see if the question is a repeat (made JD's mood a bit angry) and check the 
user tone (literal strings) to change JD's mood accordingly.

actions: CHANGED this session — was a single `action: str | None`, now a
LIST. Gemini can now propose more than one physical action per turn (e.g.
"stand up and do a silly dance" needs ["StandFromSit", "Shimmy"], not one
or the other — a single-action field structurally could not represent
this, and was the root cause of a real, hardware-observed bug: only one
of the two requested actions could ever be sent, so the wrong one fired
against the wrong starting pose). Still deliberately NOT a Literal enum
or a list of an enum, for the same reason as before - the whitelist has
22 real entries as of this session (see robot_profile.json) and will
grow, and a schema-level constraint would mean every new action needs a
schema edit + redeploy on top of the robot_profile.json edit.
Free text here is caught by verify_actions()'s
dict-based guardrail in services.py instead, per-item — Gemini is never
trusted to have generated real, executable actions on its own, and one
hallucinated entry in the list no longer invalidates the rest of it (see
verify_actions() in services.py).

null or an empty list both mean "no action this turn" — Gemini is
instructed to use either; verify_actions() treats both identically.
"""
class LLMTurnResult(BaseModel):
    response: str
    is_repeat: bool
    user_tone: Literal["nice", "rude", "neutral"]
    actions: list[str] | None