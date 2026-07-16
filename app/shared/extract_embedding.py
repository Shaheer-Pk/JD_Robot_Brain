import os

# 1. CRITICAL: This MUST be set before any AI imports
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import cv2
import traceback

from PIL import Image, ImageOps
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
    # DEBUGGING PORTION
    print(f"DEBUG: Received {len(image_bytes)} bytes.")
    
    if len(image_bytes) < 1000:
        print("DEBUG: ALERT! Less than 1kb of data. This is likely NOT an image file.")
        return None
    # DEBUGGING PORTION

    # ------- PIL LIBRARY ----
    # # 1. Open the raw bytes into a PIL Image
    # image = Image.open(io.BytesIO(image_bytes))
    
    # # 2. Fix sideways/upside-down rotation from smartphone EXIF metadata
    # image = ImageOps.exif_transpose(image)
    
    # # 3. Convert to a standard RGB array
    # image = image.convert("RGB")
    # image_np = np.array(image)
    
    # # 4. FIX: DeepFace/OpenCV expects BGR format. 
    # # Slice the array to reverse the last dimension (swapping Red and Blue).
    # # NOTE: this isn't contingous in memory yet
    # image_bgr = image_np[:, :, ::-1]

    # # 5. Force the array to be contiguous in memory for OpenCV
    # image_bgr_contiguous = np.ascontiguousarray(image_bgr)

    # ----- OPENCV LIBRARY ----
    # 1. Convert the raw Python bytes into a 1-dimensional NumPy array of 8-bit integers
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # 2. Decode the array into an image using OpenCV natively in memory.
    # cv2.IMREAD_COLOR natively handles JPEG decoding and EXIF rotation exactly
    # as cv2.imread() would from a hard drive, directly outputting a contiguous BGR array.
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image_bgr is None:
        # Failsafe: if the bytes weren't a valid image file, OpenCV returns None
        return None

    try:
        result = DeepFace.represent(
            img_path=image_bgr,
            model_name=EMBEDDING_MODEL,
            detector_backend="retinaface",
            enforce_detection=True,
        )

        # DEBUGGING PORTION
        # Now we can see EXACTLY what it found, including the confidence score
        print("\n--- DEEPFACE RAW OUTPUT ---")
        print(f"Face Confidence: {result[0].get('face_confidence', 'N/A')}")
        print(f"Facial Area Bounding Box: {result[0].get('facial_area', 'N/A')}")
        print("---------------------------\n")
        # DEBUGGING PORTION

    except ValueError as e:
        # If it still throws a ValueError (which it shouldn't with enforce_detection=True),
        # this will print the exact string DeepFace generated.
        print(f"\n--- VALUE ERROR CAUGHT ---")
        print(f"Exact Error Message: {str(e)}")
        print("--- FULL STACK TRACE ---")
        traceback.print_exc()
        print("--------------------------\n")
        return None
    
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR ---")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return None

    raw_embedding = np.array(result[0]["embedding"])
    normalized = raw_embedding / np.linalg.norm(raw_embedding)
    return normalized.tolist()