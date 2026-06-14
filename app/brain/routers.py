from fastapi import APIRouter
from app.brain.schemas import ChatRequest, ChatResponse
from app.brain.services import get_llm_response

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    llm_response = await get_llm_response(request.text)
    return ChatResponse(response=llm_response)