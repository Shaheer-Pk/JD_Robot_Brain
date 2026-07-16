import math
import cv2
import mediapipe as mp  # type: ignore
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

_face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Landmark indices for mouth measurement (fixed positions in mediapipe's
# face mesh topology — these numbers don't change between people/frames).
_UPPER_LIP = 13
_LOWER_LIP = 14
_MOUTH_LEFT = 78
_MOUTH_RIGHT = 308

# Starting point only — Needs tuning against your actual webcam/lighting/distance.
MOUTH_OPEN_RATIO_THRESHOLD = 0.35


def _distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def process_frame(image: np.ndarray):
    """
    Runs face mesh on a single frame.

    Returns:
        face_present (bool): was a face found in this frame
        mouth_is_open (bool | None): None if no face was found
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = _face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return False, None

    landmarks = results.multi_face_landmarks[0].landmark

    mouth_height = _distance(landmarks[_UPPER_LIP], landmarks[_LOWER_LIP])
    mouth_width = _distance(landmarks[_MOUTH_LEFT], landmarks[_MOUTH_RIGHT])

    if mouth_width == 0:
        return True, None

    ratio = mouth_height / mouth_width
    mouth_is_open = ratio > MOUTH_OPEN_RATIO_THRESHOLD

    return True, mouth_is_open