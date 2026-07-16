import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(key: str = Security(api_key_header)) -> str:
    # Read fresh from os.environ on every call — deliberately NOT a
    # module-level constant. A module-level read would freeze whatever
    # value existed at import time, which depends on whether load_dotenv()
    # in main.py already ran before this module got imported. That ordering
    # is not something this file controls or should have to trust — see
    # main.py's import-order fix in the same change for the historical bug
    # this avoids repeating.
    expected_key = os.getenv("JD_API_KEY")

    if key is None or key != expected_key:
        # Same exception, same message, regardless of whether the header
        # was missing entirely or present with a wrong value — distinguishing
        # the two would let an attacker confirm the header name is correct
        # and only the value is wrong.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
        )
    return key