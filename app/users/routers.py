"""
app/users/routers.py — 5 endpoints, per this session's locked roadmap.

Route summary:
  1. POST   /users/enroll             — no auth, multipart (name/persona/password as Form, photos as File)
  2. PATCH  /users/me                 — password-gated, JSON (EditProfileRequest)
  3. POST   /users/me/embeddings      — password-gated, multipart (name/password as Form, photo as File)
  4. DELETE /users/me/embeddings      — password-gated, JSON (PasswordAuthRequest) — wipes ALL embeddings
  5. DELETE /users/me                 — password-gated, JSON (PasswordAuthRequest) — deletes the account

Shared auth pipeline (routes 2, 3, 4, 5): client sends name+password,
_authenticate() resolves it to a verified user_id via get_users_by_name
+ verify_password (first match wins on name collision — documented
accepted risk, see jd-context-main.md). No route ever accepts a
client-supplied user_id directly.

Multipart, not base64-JSON, for routes 1 and 3 — locked decision this
session: real binary transfer avoids the 33% base64 size tax and an
encode/decode CPU cycle on hardware already documented as CPU-bound
with no GPU, and it matches the multipart pattern hearing/ already
uses successfully for WAV upload. See routers using Form/UploadFile,
not a Pydantic BaseModel, for these two routes specifically.

Route handlers are plain `def`, not `async def`, DELIBERATELY. FastAPI
runs synchronous route functions in an external threadpool
automatically — this offloads the blocking sqlite3 calls and the
blocking extract_embedding() call (CPU-bound, same category of
operation as Whisper/Silero in hearing/) without needing an explicit
asyncio.to_thread wrapper. This is functionally the same fix documented
in hearing-feature.md's "Runtime Issues Found" #4, applied via a
different, framework-native mechanism instead of an explicit wrapper.
If any handler here is ever changed to `async def`, every blocking
call inside it (sqlite3, extract_embedding) MUST be wrapped in
asyncio.to_thread or it will freeze the event loop exactly like the
bug already found and fixed in hearing/routers.py.

extract_embedding — CONFIRMED against the real file this session:
app/shared/extract_embedding.py, signature
extract_embedding(image_bytes: bytes) -> list[float] | None, returning
None when no face is detected in the photo (DeepFace.represent raises
ValueError internally, caught and converted to None). Both call sites
below (enroll, add-embedding) handle the None case explicitly — see
each route's own comments for the exact behavior, which differs
between the two (per-photo skip for enroll's multiple photos vs.
outright rejection for add-embedding's single photo).

API-key gating is also NOT wired here — locked roadmap scope for a
future chat (see jd-context-main.md, item #6 from this session's open
questions). No dependency for it exists yet.
"""

import sqlite3

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.shared.auth import hash_password, verify_password
from app.shared.extract_embedding import extract_embedding  # confirmed against real file this session
from app.users.database import get_db
from app.users.schemas.schemas_pydantic import EditProfileRequest, PasswordAuthRequest
from app.users.services.embeddings_dao import (
    count_embeddings_for_user,
    delete_all_embeddings_for_user,
    delete_embedding,
    insert_embedding,
    list_embeddings_for_user,
)
from app.users.services.users_dao import (
    delete_user,
    get_users_by_name,
    insert_user,
    update_user,
)

router = APIRouter()

FIFO_CAP = 8  # hard cap on stored embeddings per user; evict oldest at/over this count before inserting


def _authenticate(conn: sqlite3.Connection, name: str, password: str) -> int:
    """
    Resolves name+password to a verified user_id. Never trusts a
    client-supplied user_id. Iterates every row get_users_by_name
    returns (names are not unique — see jd-context-main.md), checking
    verify_password against each row's hash; first match wins. Raises
    401 if no row matches, rather than leaking whether the name exists
    at all (single generic message either way).
    """
    for row in get_users_by_name(conn, name):
        if verify_password(password, row["password_hash"]):
            return row["id"]
    raise HTTPException(status_code=401, detail="Invalid name or password.")


# ---------------------------------------------------------------------------
# Route 1 — Enrollment. No password gate (no account exists yet).
# ---------------------------------------------------------------------------
@router.post("/enroll")
def enroll_user(
    name: str = Form(...),
    persona: str = Form(...),
    password: str = Form(...),
    photos: list[UploadFile] = File(...),
    conn: sqlite3.Connection = Depends(get_db),
):
    if not photos:
        raise HTTPException(status_code=400, detail="At least one enrollment photo is required.")

    if not persona.strip():
        raise HTTPException(status_code=400, detail="Persona cannot be blank.")

    # Extract every embedding BEFORE opening any DB write — per the
    # original locked roadmap ordering ("extract all embeddings fully
    # first, then open one connection, write both tables, commit-or-
    # rollback atomically"). A bad photo fails here, before the DB is
    # ever touched, not mid-transaction.
    #
    # extract_embedding returns None when no face was detected in a
    # given photo. Per-photo handling, decided this session: skip the
    # bad photo and continue (NOT reject the whole batch — rejecting
    # everyone's enrollment over one blurry photo out of several is
    # bad UX), UNLESS every single photo comes back None, in which
    # case there would be zero embeddings to attach to the new
    # account — a face-recognition feature with nothing to recognize,
    # a broken account by construction. That case hard-rejects with no
    # user ever created.
    embeddings: list[list[float]] = []
    rejected_filenames: list[str] = []
    for photo in photos:
        raw_bytes = photo.file.read()
        result = extract_embedding(raw_bytes)
        if result is None:
            rejected_filenames.append(photo.filename)
            continue
        embeddings.append(result)

    if not embeddings:
        raise HTTPException(
            status_code=400,
            detail="No usable photos — none had a detectable face. Please try enrolling again.",
        )

    hashed_password = hash_password(password)  # auth.py validates length, raises ValueError if out of bounds

    try:
        user_id = insert_user(
            conn,
            name=name,
            persona=persona,
            memory_context=None,  # no history yet — this is a brand new account
            password_hash=hashed_password,
        )
        for embedding in embeddings:
            insert_embedding(conn, user_id, embedding)
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Failed to enroll user. Please try again.")
    except Exception:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Unexpected error during enrollment.")

    return {
        "user_id": user_id,
        "name": name,
        "embeddings_stored": len(embeddings),
        "rejected_photos": rejected_filenames,
    }


# ---------------------------------------------------------------------------
# Route 2 — Edit profile. Password-gated. Name/persona only, partial update.
# ---------------------------------------------------------------------------
@router.patch("/me")
def edit_profile(
    request: EditProfileRequest,
    conn: sqlite3.Connection = Depends(get_db),
):
    user_id = _authenticate(conn, request.name, request.password)

    try:
        updated = update_user(conn, user_id, name=request.new_name, persona=request.new_persona)
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Failed to update profile. Please try again.")
    except Exception:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Unexpected error updating profile.")

    return {"updated": updated}


# ---------------------------------------------------------------------------
# Route 3 — Add embedding. Password-gated. FIFO eviction at cap, one txn.
# ---------------------------------------------------------------------------
@router.post("/me/embeddings")
def add_embedding(
    name: str = Form(...),
    password: str = Form(...),
    photo: UploadFile = File(...),
    conn: sqlite3.Connection = Depends(get_db),
):
    user_id = _authenticate(conn, name, password)

    raw_bytes = photo.file.read()
    embedding = extract_embedding(raw_bytes)

    if embedding is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in that photo. Please try again with a clearer photo.",
        )

    try:
        # Count BEFORE inserting — this is the corrected ordering from
        # this session's stress-test. Counting after the insert would
        # always read a number one higher than the true pre-insert
        # state, converging the effective cap to 7, not 8 (see this
        # session's transcript — "count-after-insert" bug). Evicting
        # before inserting also means order-of-operations between
        # evict/insert cannot matter: whichever runs first, the final
        # row set is identical (also verified this session).
        current_count = count_embeddings_for_user(conn, user_id)
        if current_count >= FIFO_CAP:
            oldest_id = list_embeddings_for_user(conn, user_id)[0]["id"]
            delete_embedding(conn, oldest_id)
        insert_embedding(conn, user_id, embedding)
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Failed to add embedding. Please try again.")
    except Exception:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Unexpected error adding embedding.")

    return {"status": "embedding added"}


# ---------------------------------------------------------------------------
# Route 4 — Wipe ALL embeddings for the user. Password-gated.
# ---------------------------------------------------------------------------
@router.delete("/me/embeddings")
def wipe_embeddings(
    request: PasswordAuthRequest,
    conn: sqlite3.Connection = Depends(get_db),
):
    user_id = _authenticate(conn, request.name, request.password)

    try:
        deleted_count = delete_all_embeddings_for_user(conn, user_id)
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Failed to wipe embeddings. Please try again.")
    except Exception:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Unexpected error wiping embeddings.")

    return {"deleted_count": deleted_count}


# ---------------------------------------------------------------------------
# Route 5 — Delete account entirely. Password-gated. Cascades to embeddings.
# ---------------------------------------------------------------------------
@router.delete("/me")
def delete_account(
    request: PasswordAuthRequest,
    conn: sqlite3.Connection = Depends(get_db),
):
    user_id = _authenticate(conn, request.name, request.password)

    try:
        deleted = delete_user(conn, user_id)
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete account. Please try again.")
    except Exception:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Unexpected error deleting account.")

    return {"deleted": deleted}