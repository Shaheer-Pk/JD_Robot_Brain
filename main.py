from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()       # Load env first because router tries to access api key on start-up
import os
print("KEY:", os.getenv("GEMINI_API_KEY"))

from app.brain.routers import router as brain_router

app = FastAPI(
    title = "JD Robot Backend",
    version = "0.1.0"
)

app.include_router(brain_router, prefix = "/brain", tags = ["Brain"])