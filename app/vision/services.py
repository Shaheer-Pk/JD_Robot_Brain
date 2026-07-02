import io
import numpy as np
from PIL import Image
from deepface import DeepFace

from .models import FaceEmbedding

# defines get_user_profile(db, user_id)
from app.users import services as user_services
EMBEDDING_MODEL = "Facenet512"

THRESHOLD = 1.4  # starting point, tune after testing with real enrolled users

 
def extract_embedding(image_bytes: bytes):
    
    # Turns a photo (raw bytes) into a 512-number fingerprint.
    # Returns None if no face was found in the photo.
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)  # store image as number array in variable, same as before
 
    try:
        result = DeepFace.represent(
            img_path=image_np,            # DeepFace accepts a numpy array directly here
            model_name=EMBEDDING_MODEL,
            detector_backend="retinaface",
            enforce_detection=True,       # makes it throw an error if no face is found
        )
    except ValueError:
        # This is DeepFace's way of saying "no face found in this photo"
        return None
 
    raw_embedding = np.array(result[0]["embedding"])
    normalized = raw_embedding / np.linalg.norm(raw_embedding)
 
    return normalized.tolist()


def find_best_match(db, new_embedding):
    """
    Compares the new embedding against every stored embedding in the DB.
    Returns (user_id, distance) of the closest match, or (None, distance)
    if nothing was close enough (guest mode)
    """
    stored = db.query(FaceEmbedding).all()

    best_user_id = None
    best_distance = float("inf")

    for record in stored:
        distance = np.linalg.norm(new_embedding - np.array(record.embedding))
        if distance < best_distance:
            best_distance = distance
            best_user_id = record.user_id

    # TEMPORARY Print statement for testing
    print(f"Best match: user_id={best_user_id}, distance={best_distance}")

    if best_distance < THRESHOLD:
        return best_user_id, best_distance
    return None, best_distance


def identify_face(db, image_bytes: bytes):

    # Full pipeline: photo -> embedding -> match -> user profile.

    embedding = extract_embedding(image_bytes)
    if embedding is None:
        return None

    user_id, distance = find_best_match(db, embedding)
    if user_id is None:
        return None

    # Crosses into the users module here — vision only ever talks to users
    # through this one function, per the vertical-slice rule.
    profile = user_services.get_user_profile(db, user_id)

    return {
        "user_id": profile["id"],
        "name": profile["name"],
        "persona": profile["persona"],          # Add memory context from table if needed here
        "confidence": round(1 - distance, 2),
    }


# One-time registration step. Run once per reference photo
# (call this 3-5 times per person with different photos).
def enroll_face(db, user_id: int, image_bytes: bytes) -> bool:

    embedding = extract_embedding(image_bytes)
    if embedding is None:
        return False
    
    db.add(FaceEmbedding(user_id=user_id, embedding=embedding))
    db.commit()
    return True