# Validating data-type so we only take string as input and give string as an output

from pydantic import BaseModel

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    response: str