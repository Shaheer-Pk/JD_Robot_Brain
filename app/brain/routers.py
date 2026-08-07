"""
app/brain/routers.py — The HTTP entrypoint the C# skill hits after
hearing finishes transcribing an utterance (POST /brain/chat, per
hearing-feature.md's pipeline). Ties conversation (Gemini), identity
(vision), pose tracking, and speech (Piper TTS) into one request/
response cycle: text in, synthesized WAV audio out.

Identity comes from app.vision.state.session, NOT from anything the
client sends — ChatRequest carries only {"text": str}, deliberately
unchanged. This route reads session.get_profile() itself at request
time rather than requiring identity to be passed over the wire.

profile is None in two cases NOT distinguished here, on purpose: truly
unrecognized (recognition_status == "guest") AND still mid-recognition
(recognition_status == "pending"). Accepted tradeoff, not a bug.

When profile exists, only name and persona are pulled into
custom_personality — NOT the full vision dict.

CHANGED this session — pose tracking added. See app/brain/pose_state.py
for the full rationale (single tracked value, own lock, prediction-not-
confirmation limitation). This route now:
  1. Reads JD's current tracked pose BEFORE the Gemini call, feeds it
     into get_llm_response() as pose_context (same pattern as
     mood_context).
  2. Passes that same pose into verify_actions() as current_pose, so
     the guard-rail can simulate pose transitions through the batch.
  3. Writes memory_session.add_turn() HERE, after verify_actions()
     resolves — MOVED from services.py's get_llm_response(), which
     used to call it before verification ever ran. See
     services.py's get_llm_response() docstring/comments for why that
     ordering was wrong once action history needed to be recorded.
  4. Writes the new tracked pose back via pose_session.set_pose(),
     ONLY when at least one action survived verification.
"""
import asyncio
import json  # for serializing the verified action triple into a response header
import time  # for perf_counter — high-resolution, monotonic, immune to system clock adjustments


from fastapi import APIRouter
from fastapi.responses import Response
from app.brain.schemas import ChatRequest
from app.brain.services import get_llm_response, text_to_speech, verify_actions
from app.brain.pose_state import pose_session   # NEW this session — read + write
from app.emotion.services import apply_event, get_current_preset   # read + write
from app.shared.memory import memory_session   # NEW import here — add_turn() now called from this file
from app.vision.state import session

router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    # Load custom_personality at runtime according to
    # Vision/schemas VisionProfile
    profile = session.get_profile()
    custom_personality = None
    if profile is not None:
        custom_personality = {"name": profile.name, "persona": profile.persona}
    
    # Read mood before Gemini call
    current_mood = get_current_preset()

    # NEW this session — read JD's tracked pose before the Gemini call,
    # same "read fresh per-request" pattern as mood above. See
    # pose_state.py — this is a PREDICTED pose, not hardware-confirmed.
    current_pose = pose_session.get_pose()

    # DEBUG: T0 — immediately before the Gemini call
    t0 = time.perf_counter()

    # Brain/services Gemini call — action_keywords is an ORDERED LIST
    # (or None/empty). NEW this session: pose_context is now passed
    # alongside mood_context, so Gemini's prompt includes JD's current
    # pose and the full per-action pose table (see services.py).
    spoken_text, is_repeat, user_tone, action_keywords = await get_llm_response(
        request.text,
        custom_personality=custom_personality,
        mood_context=current_mood,
        pose_context=current_pose,
    )

    # DEBUG: T1 — immediately after Gemini returns.
    t1 = time.perf_counter()
    word_count = len(spoken_text.split())
    print(f"[TIMING] Gemini call: {t1 - t0:.3f}s | response word count: {word_count} | turn history length check happens inside get_llm_response, not visible here")

    # apply event from emotions module for mood to affect the response (write after response)
    apply_event(is_repeat, user_tone)

    # CHANGED this session — verify_actions() now takes current_pose and
    # returns a 3-tuple: (verified_action_triples, verified_action_names,
    # final_pose). See services.py's verify_actions() docstring for the
    # full pose-guard-rail mechanism, including the truncation behavior
    # on a pose mismatch (deliberately more costly than a hallucinated
    # keyword — see that docstring's "asymmetry" example).
    verified_actions, verified_action_names, final_pose = verify_actions(action_keywords, current_pose)

    # DEBUG: T2 — immediately after verify_actions. T2-T1 should be near-zero.
    t2 = time.perf_counter()
    print(f"[TIMING] verify_actions: {t2 - t1:.3f}s")

    # NEW this session — build the action-history note that gets fused
    # into what's stored in conversation memory (Option A, per this
    # session's design discussion — see memory.py's docstring for the
    # full rationale and the deliberate future-scalability caveat).
    # Uses verified_action_NAMES, never raw ARC triples — Gemini should
    # never see gadget/cmd plumbing echoed back at it from its own
    # memory.
    if verified_actions is not None:
        action_note = f" [JD physically performed: {', '.join(verified_action_names)}]"
    else:
        action_note = ""

    # NEW this session — memory_session.add_turn() now called HERE, not
    # inside services.py's get_llm_response(). This is deliberately
    # AFTER verify_actions() has resolved, so memory only ever records
    # actions that actually survived verification — a hallucinated or
    # pose-rejected action leaves ZERO trace in memory, exactly as if
    # Gemini never requested it (per this session's "invisible on
    # rejection" decision).
    memory_session.add_turn(request.text, spoken_text + action_note)

    # NEW this session — write the predicted new pose back, ONLY if
    # something actually survived verification (nothing survived means
    # nothing executed, so pose cannot have changed — verify_actions()
    # already returns current_pose unchanged in that case, but the
    # explicit guard here keeps the "why are we writing" reasoning
    # visible at the call site too).
    if verified_actions is not None:
        pose_session.set_pose(final_pose)
        print(f"[Brain] Tracked pose updated to (simulated, unconfirmed): '{final_pose}'")

    # Take above spoken_text and perform Piper TTS
    # Offloaded to bg thread so it doesnt conflict with
    # Mood GET endpoint or vision POST endpoint when working
    # in parallel.     
    audio_bytes = await asyncio.to_thread(text_to_speech, spoken_text)         # spoken text -> audio bytes

    # DEBUG: T3 — immediately after Piper returns.
    t3 = time.perf_counter()
    print(f"[TIMING] text_to_speech (Piper): {t3 - t2:.3f}s")

    response = Response(content=audio_bytes, media_type="audio/wav")    # wav format through piper tts (for elevenlab its mpeg)

    # DEBUG: T_total — full route duration, for a sanity-check cross-reference
    print(f"[TIMING] TOTAL route time: {t3 - t0:.3f}s")

    # Actions are still carried out-of-band from the audio body via the
    # same custom X-JD-Action response header. Shape unchanged — a JSON
    # array of [gadget, cmd, param] triples, in execution order. Header
    # still OMITTED ENTIRELY when there are no verified actions.
    if verified_actions is not None:
        response.headers["X-JD-Action"] = json.dumps(verified_actions)
        print(f"[Brain] X-JD-Action header set: {response.headers['X-JD-Action']}")
    else:
        print("[Brain] No X-JD-Action header sent this turn.")

    return response