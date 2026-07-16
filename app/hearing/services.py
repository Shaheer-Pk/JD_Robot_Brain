import io
import numpy as np
import soundfile as sf
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Model loading — both models load ONCE at module import time, not per request.
# Loading a model on every request would add 3-5 seconds of latency each time.
# ---------------------------------------------------------------------------

# Silero VAD — tiny model, loads in under a second
_vad_model = load_silero_vad()

# Faster Whisper — base model on CPU.
# compute_type="int8" means it uses 8-bit integer math instead of 32-bit floats.
# Cuts memory usage and speeds up inference on CPU with negligible accuracy loss.
_whisper_model = WhisperModel("base", device="cpu", compute_type="int8")


# ---------------------------------------------------------------------------
# Silero requires audio at exactly 16000 Hz, mono, float32.
# C# will send us a WAV file so soundfile handles the decoding automatically,
# but we still need to enforce the sample rate and channel requirements.
# ---------------------------------------------------------------------------
_SILERO_SAMPLE_RATE = 16000
_VAD_THRESHOLD = 0.5       # Probability above which Silero considers it speech
_MIN_SPEECH_DURATION = 0.3 # Seconds — ignore clips shorter than this (coughs, clicks)


def transcribe_audio(audio_bytes: bytes) -> tuple[bool, str | None]:
    """
    Full pipeline: raw WAV bytes in, (is_speech, transcript) tuple out.

    Steps:
    1. Decode WAV bytes into a numpy float32 array via soundfile
    2. Resample to 16000 Hz if C# sent us a different rate
    3. Run Silero VAD — if no speech detected, return early (False, None)
    4. Run Faster Whisper on the confirmed speech audio
    5. Return (True, transcribed_text)

    Returns:
        (False, None)        — audio was noise/silence, discard it
        (True, "some text")  — confirmed speech, transcribed successfully
    """

    # --- Step 1: Decode WAV bytes ---
    # soundfile reads directly from a BytesIO buffer, no temp file needed.
    # It returns a numpy array of samples and the original sample rate.
    audio_buffer = io.BytesIO(audio_bytes)
    audio_array, sample_rate = sf.read(audio_buffer, dtype="float32")

    # --- Step 2: Enforce mono ---
    # If C# sends stereo (2 channels), average them down to mono.
    # Silero and Whisper both expect a single channel 1D array.
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    # --- Step 3: Resample to 16000 Hz if needed ---
    # NAudio in C# will capture at the system default (usually 44100 or 48000 Hz).
    # Silero strictly requires 16000 Hz so we resample if necessary.
    if sample_rate != _SILERO_SAMPLE_RATE:
        # Simple linear interpolation resample.
        # Not the highest quality but fast and sufficient for speech.
        target_length = int(len(audio_array) * _SILERO_SAMPLE_RATE / sample_rate)
        audio_array = np.interp(
            np.linspace(0, len(audio_array) - 1, target_length),
            np.arange(len(audio_array)),
            audio_array
        )

    # --- Step 4: Silero VAD check ---
    # Silero expects a torch tensor, not a numpy array.
    # .float() forces the tensor to be standard Float (float32), preventing Double errors!
    audio_tensor = torch.from_numpy(audio_array).float()

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        _vad_model,
        sampling_rate=_SILERO_SAMPLE_RATE,
        threshold=_VAD_THRESHOLD,
        min_speech_duration_ms=int(_MIN_SPEECH_DURATION * 1000)
    )

    # If Silero found zero speech segments, this is noise. Discard.
    if not speech_timestamps:
        return False, None

    # --- Step 5: Faster Whisper transcription ---
    # WhisperModel.transcribe() accepts a numpy array directly.
    # beam_size=5 is the default — good balance of accuracy vs speed.
    # language="en" skips language detection, saving ~0.5s per request.
    segments, _ = _whisper_model.transcribe(
        audio_array,
        beam_size=5,
        language="en"
    )

    # segments is a generator — join all segments into one clean string.
    transcript = " ".join(segment.text.strip() for segment in segments).strip()

    # Edge case: Whisper returned empty text even though Silero detected speech.
    # Treat as noise.
    if not transcript:
        return False, None

    return True, transcript