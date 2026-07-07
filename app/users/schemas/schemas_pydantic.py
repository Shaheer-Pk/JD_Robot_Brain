"""
Pydantic request models for app/users/routers.py.

Covers ONLY the pure-JSON endpoints: edit-profile, wipe-embeddings,
delete-account. Enroll and add-embedding endpoints do NOT use these —
they carry binary photo data and use FastAPI's Form(...)/File(...)/
UploadFile parameters instead of a BaseModel, per the locked multipart
decision (real binary transfer, no base64 inflation, consistent with
hearing/'s existing multipart WAV upload pattern).
"""

from pydantic import BaseModel


class PasswordAuthRequest(BaseModel):
    """
    Shared auth shape for every password-gated endpoint that needs no
    other input: wipe-embeddings, delete-account. Never resolves a
    user_id itself — routers.py's _authenticate() helper does that by
    checking `password` against every row get_users_by_name(name)
    returns, first match wins. The client never supplies a user_id.
    """
    name: str
    password: str


class EditProfileRequest(PasswordAuthRequest):
    """
    Extends PasswordAuthRequest with the fields being changed.
    `name`/`password` (inherited) are LOGIN credentials, not the new
    name — deliberately named new_name/new_persona below so a client
    can never confuse "who I'm logging in as" with "what I'm changing
    my name to" in the same payload.

    Both optional — mirrors update_user's partial-update contract.
    memory_context is NOT here and never will be via this endpoint;
    it's system-owned (see users_dao.update_user's docstring).
    """
    new_name: str | None = None
    new_persona: str | None = None