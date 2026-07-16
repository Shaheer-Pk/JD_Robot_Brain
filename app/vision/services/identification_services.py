"""
app/vision/services/identification_services.py — The "whose face is this"
pipeline: raw photo bytes in, a user profile dict (or None) out. Pure
functions, no state, called by recognition_services.py.

find_best_match(conn, new_embedding): full-table-scan comparison against
every stored embedding across every user, via embeddings_dao.
load_face_embeds() (one query, deliberately unfiltered by user — see that
function's own docstring). Each stored embedding was written as a
JSON-encoded string (embeddings_dao.insert_embedding does json.dumps
before INSERT — SQLite has no native array/vector column), so on read,
json.loads() MUST run before np.array() to turn it back into an actual
numeric vector — skip this and every distance calculation silently breaks
instead of raising. Metric is euclidean_l2 (np.linalg.norm of the
difference), matching Facenet512's own normalization. THRESHOLD = 1.04 is
the published tuned cutoff for this model+metric pair (the original
teammate code had 1.4 — that's the model's untuned default, left
uncorrected). Below threshold = confident match; nothing close enough
returns (None, best_distance), meaning "no match."

identify_face(conn, image_bytes): orchestrates the pipeline —
extract_embedding() (app/shared/extract_embedding.py, DeepFace/Facenet512,
returns None if no face detected — checked, short-circuits here) ->
find_best_match() -> if matched, users_dao.get_user_profile() for the
actual name/persona/memory_context. The get_user_profile() None-check
guards a real, if narrow, race: find_best_match can return a user_id that
gets deleted (DELETE /users/me) before this immediately-following lookup
runs — treated as "no match" instead of letting a None-subscript crash
propagate unwatched out of a background task.

Return shape on success:
    {"user_id": int, "name": str, "persona": str,
     "memory_context": str, "confidence": float}
confidence = round(1 - distance, 2) — NOT a calibrated probability, can go
negative for a bad match (distances range roughly 0-2 for this
model/metric) — included for debugging/logging only.

Crosses the vision -> users module boundary here (embeddings_dao,
users_dao) — the ONLY file in vision/ that does. Every other vision file
reaches users/ data exclusively through this module's two functions, per
jd-context-main.md's vertical-slice rule.
"""

import json
import numpy as np

from app.shared.extract_embedding import extract_embedding
from app.users.services import embeddings_dao, users_dao
from app.vision.schemas import VisionProfile

THRESHOLD = 1.04  # tuned value for Facenet512 + euclidean_l2


def find_best_match(conn, new_embedding):
    stored = embeddings_dao.load_face_embeds(conn)

    best_user_id = None
    best_distance = float("inf")

    for record in stored:
        stored_embedding = np.array(json.loads(record["embedding"]))
        distance = np.linalg.norm(new_embedding - stored_embedding)
        if distance < best_distance:
            best_distance = distance
            best_user_id = record["user_id"]

    if best_distance < THRESHOLD:
        return best_user_id, best_distance
    return None, best_distance


def identify_face(conn, image_bytes: bytes) -> VisionProfile | None:
    embedding = extract_embedding(image_bytes)
    if embedding is None:
        return None

    user_id, distance = find_best_match(conn, embedding)
    if user_id is None:
        return None

    profile = users_dao.get_user_profile(conn, user_id)
    if profile is None:
        # User matched at embedding-scan time but was deleted before this
        # lookup ran (race, not hypothetical) — treat as no match.
        return None

    return VisionProfile(
        user_id=profile["id"],
        name=profile["name"],
        persona=profile["persona"],
        memory_context=profile["memory_context"],
        confidence=round(1 - distance, 2),
    )