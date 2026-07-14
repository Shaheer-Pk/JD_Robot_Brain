from fastapi import FastAPI
from dotenv import load_dotenv
from app.vision.routers import router as vision_router
from app.vision import stream_routes

load_dotenv()       # Load env first because router tries to access api key on start-up

from app.brain.routers import router as brain_router
from app.hearing.routers import router as hearing_router
from app.emotion.routers import router as emotion_router

app = FastAPI(
    title = "JD Robot Backend",
    version = "0.1.0"
)

app.include_router(vision_router)
app.include_router(stream_routes.router)
app.include_router(brain_router, prefix = "/brain", tags = ["Brain"])
app.include_router(hearing_router, prefix = "/hearing", tags = ["Hearing"])
app.include_router(emotion_router, prefix = "/emotion", tags = ["Emotion"])