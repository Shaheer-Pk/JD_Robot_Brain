import io
import numpy as np
from PIL import Image
from deepface import DeepFace

EMBEDDING_MODEL = "Facenet512"


def extract_embedding(image_bytes: bytes) -> list[float] | None:
    """
    Turns a photo (raw bytes) into a 512-number fingerprint, normalized
    to unit length (required for the euclidean_l2 distance metric used
    downstream in vision/services.py's find_best_match — see this
    session's threshold analysis: DeepFace's own tuned euclidean_l2
    threshold for Facenet512 assumes normalized vectors, 1.04, not the
    unverified 1.4 the original code shipped with).

    Lives in app/shared/, not app/vision/, because both users/ (enrollment,
    customization photo uploads) and vision/ (live recognition) call this
    — no single domain owns it. Pure function: image bytes in, vector out,
    zero DB access, zero knowledge of which domain is calling it.

    Returns None if no face was found in the photo.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    try:
        result = DeepFace.represent(
            img_path=image_np,
            model_name=EMBEDDING_MODEL,
            detector_backend="retinaface",
            enforce_detection=True,
        )
    except ValueError:
        return None

    raw_embedding = np.array(result[0]["embedding"])
    normalized = raw_embedding / np.linalg.norm(raw_embedding)
    return normalized.tolist()