# JD Robot Brain

Python/FastAPI backend powering JD, an EZ-Robot JD humanoid running as a
conversational AI companion. Handles speech transcription, LLM
orchestration (Gemini), face recognition, simulated mood, text-to-speech,
and verified physical-action dispatch. Designed to run alongside a
companion C# ARC skill ([`JdMegaMind`](https://github.com/Shaheer-Pk/Robot-Skills-Csharp)) that handles the
robot-facing I/O — camera, mic, and physical action execution.

This repo is one half of a two-repo system. **It does nothing on its own
without JD's hardware and the ARC skill sending it requests** — see that
repo's README for the other half.

## Architecture

Vertically-sliced modules, each owning one concern:

| Module | Responsibility |
|---|---|
| `brain/` | LLM orchestration (Gemini), persona/mood/pose prompt assembly, physical-action verification, TTS handoff |
| `hearing/` | Voice activity detection + local speech-to-text (Whisper) |
| `vision/` | Face detection/recognition (mediapipe + DeepFace), in-memory identity session |
| `emotion/` | Simulated mood engine, decay/hysteresis math, RGB-eye-color state |
| `users/` | SQLite user profiles, face-embedding storage, enrollment |
| `shared/` | Cross-module code with no single domain owner (conversation memory, auth, embedding extraction) |

Full architectural rationale and decision history live in this project's
internal documentation; this README covers what's needed to run it.

## Requirements

- Python **3.12.10**
- **Windows 11** — developed and tested exclusively on Windows 11.
  Portability to Linux/macOS is untested; nothing here is deliberately
  Windows-only, but no guarantees are made either way.
- ~3.3 GB free disk space for the Python environment (see below)

## Setup

```bash
git clone <repo-url>
cd JD_Robot_Brain
python -m venv venv_bridge
venv_bridge\Scripts\activate
pip install -r requirements.txt
```

**Expect the installed environment to take up ~3.3 GB on disk.** The
bulk comes from `torch` (CPU build), `deepface` (pulls TensorFlow/Keras
transitively), and `mediapipe`. Budget accordingly.

**NOTE** mediapipe is not being utilized in this project build currently. It was future spec'd to replace the wakeword with lip tracking movements. However, that approach is **incomplete**

### Environment variables

Copy `.env.example` to `.env` and fill in your own key:


`JD_API_KEY` also appears in `.env.example` as a placeholder for a
planned API-key-gating feature. **It is not currently enforced anywhere
in the codebase** — setting it has no effect yet. See Known Limitations.

### Voice model (TTS)

Text-to-speech uses [Piper TTS](https://github.com/rhasspy/piper). The
default voice model (`en-US-cori-medium.onnx` + `.onnx.json`) is
committed under `voices/` — no separate download needed to run as-is.

To use a different voice, download any Piper-compatible voice pair from
Piper's own voice repository and swap the files in `voices/`, updating
the load path in `brain/services.py` accordingly.

### First run

On first use, `faster-whisper`, `mediapipe`, and `deepface` will each
download their own model weights on demand (not part of the `pip
install` step above). Expect a one-time delay of several hundred MB
worth of downloads the first time each subsystem actually runs.

## Running

From the project root, with the venv activated:

```bash
uvicorn main:app 
```

Serves on `127.0.0.1:8000` by default. The companion ARC skill expects
to reach this over the network (via mDNS hostname in the reference
setup) — see that repo's README for how to point it at this backend.

## Known Limitations

**API-key gating and HTTPS are designed but not enforced.** A
`get_api_key` dependency exists in `shared/security.py` but is not
attached to any route, and the server runs over plain HTTP. As shipped,
any device on the same network can reach every endpoint — including
`/brain/chat`, which can trigger real physical robot movement — with no
authentication. Treat this as a closed local-network-only setup until
this is built out.

**Persistent (cross-session) memory is not implemented.** The memory
system was designed with two layers in mind: an in-session, in-RAM
layer, and a persistent, cross-session layer. Only the first was built —
conversation history lives in a locked in-memory object and is lost on
every backend restart. Concretely:
- The `users` table has a `memory_context` column reserved for
  per-user persistent memory. No code currently reads or writes it —
  it's schema-only.
- There is no SQLite (or other) persistence of conversation turns at
  all. `seed()`/`reset()` hooks exist on the in-memory store for this
  purpose but are unused.

**Pose tracking assumes JD starts standing.** JD's tracked pose (used
to guard against physically unsafe action sequences) is initialized to
`"standing"` at process start, with no hardware confirmation channel. If
JD is actually powered on or connected while seated, the tracked pose is
wrong from the first turn onward, and the safety guard-rail built on top
of it can make incorrect decisions. **Known flaw** — always start JD
standing before running a session.

**Multi-person tracking is unsolved.** The vision pipeline tracks at
most one face at a time by design; two people in frame simultaneously
have no defined behavior beyond "whichever face the detector happens to
rank first."