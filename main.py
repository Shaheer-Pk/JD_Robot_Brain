from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()       # Load env first because router tries to access api key on start-up

from app.brain.routers import router as brain_router
from app.hearing.routers import router as hearing_router
from app.vision.routers import router as vision_router
#from app.vision import stream_routes
from app.users.routers import router as user_router

app = FastAPI(
    title = "JD Robot Backend",
    version = "0.1.0"
)

app.include_router(vision_router, prefix = "/vision", tags = ["Vision"])
#app.include_router(stream_routes.router, prefix = "/vision", tags = ["Vision"])
app.include_router(user_router, prefix = "/users", tags = ["Users"])
app.include_router(brain_router, prefix = "/brain", tags = ["Brain"])
app.include_router(hearing_router, prefix = "/hearing", tags = ["Hearing"])
