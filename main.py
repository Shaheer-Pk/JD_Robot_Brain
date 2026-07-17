import os

# 1. CRITICAL: This MUST be set before any AI imports
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from dotenv import load_dotenv

load_dotenv()       # Load env first because router tries to access api key on start-up

from app.brain.routers import router as brain_router
from app.hearing.routers import router as hearing_router
from app.emotion.routers import router as emotion_router
from app.vision.routers import router as vision_router
from app.users.routers import router as user_router

app = FastAPI(
    title = "TESTING_DEBUG_MODE",
    version = "0.1.0"
)

app.include_router(vision_router, prefix = "/vision", tags = ["Vision"])
app.include_router(user_router, prefix = "/users", tags = ["Users"])
app.include_router(brain_router, prefix = "/brain", tags = ["Brain"])
app.include_router(hearing_router, prefix = "/hearing", tags = ["Hearing"])
app.include_router(emotion_router, prefix = "/emotion", tags = ["Emotion"])


'''
Code in this section is to fix openapi formatting which breaks the line
'photos: list[UploadFile] = File(...)' in users/routers.py due to swaggerUI
being outdated with respect to FastAPI.
We fix this issue, by overriding openapi.json created and updating the modern FastAPI
inferences back to old inferences (fastapi 0.129 version) with which swaggerUI was compatible
'''

def patch_openapi_for_file_arrays(schema: dict | list):
    """
    Recursively walk the OpenAPI schema dict.
    Whenever we find {"contentMediaType": "application/octet-stream"},
    we replace it with {"format": "binary"} so Swagger UI renders file pickers.
    """
    if isinstance(schema, dict):
        # The specific bug is Swagger UI failing to render arrays of files.
        # FastAPI 0.129.1+ sets `contentMediaType` for files.
        if schema.get("contentMediaType") == "application/octet-stream":
            del schema["contentMediaType"]
            schema["format"] = "binary"
            # Swagger also expects the type to be string for binary formats
            schema["type"] = "string"
        
        # Continue recursively through all dictionary values
        for key, value in schema.items():
            patch_openapi_for_file_arrays(value)
            
    elif isinstance(schema, list):
        # Continue recursively through all list items
        for item in schema:
            patch_openapi_for_file_arrays(item)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )

    patch_openapi_for_file_arrays(openapi_schema)

    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Final injection line to log our changes in openapi.json
# This injection works because openapi.json is on called when we do
# */docs in the browser tab
app.openapi = custom_openapi
