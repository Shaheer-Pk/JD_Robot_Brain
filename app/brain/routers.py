"""
app/brain/routers.py — The HTTP entrypoint the C# skill hits after
hearing finishes transcribing an utterance (POST /brain/chat, per
hearing-feature.md's pipeline). Ties conversation (Gemini), identity
(vision), and speech (Piper TTS) into one request/response cycle: text
in, synthesized WAV audio out.

Identity comes from app.vision.state.session, NOT from anything the
client sends — ChatRequest carries only {"text": str}, deliberately
unchanged. The C# skill has no way to know who's currently recognized
(that state lives entirely server-side, inside the same Python process
vision runs in) — this route reads session.get_profile() itself at
request time rather than requiring identity to be passed over the wire
and duplicated into a second place that could drift out of sync with
vision's own state.

profile is None in two cases NOT distinguished here, on purpose: truly
unrecognized (recognition_status == "guest") AND still mid-recognition
(recognition_status == "pending", not yet resolved). A chat request
arriving during the pending window gets the guest persona for that one
turn even if the person would have been correctly identified a moment
later — accepted tradeoff, not a bug (confirmed decision — not worth
holding/delaying a chat request on an in-flight background recognition
attempt).

When profile exists, only name and persona are pulled into
custom_personality — NOT the full vision dict (which also carries
user_id, memory_context, confidence). Deliberate narrowing at the module
boundary: brain/services.py should depend on "a name and a persona
string," not on vision's internal return shape — if identify_face's
return shape changes later, brain/ doesn't silently break.
"""
import asyncio
import json  # for serializing the verified action triple into a response header
import time  # for perf_counter — high-resolution, monotonic, immune to system clock adjustments


from fastapi import APIRouter
from fastapi.responses import Response
from app.brain.schemas import ChatRequest
from app.brain.services import get_llm_response, text_to_speech, verify_actions
from app.emotion.services import apply_event, get_current_preset   # read + write
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

    # DEBUG: T0 — immediately before the Gemini call
    t0 = time.perf_counter()

    # Brain/services Gemini call — action_keywords is now an ORDERED LIST
    # (or None/empty), not a single keyword. See schemas.py's
    # LLMTurnResult.actions and services.py's action_guidance for why.
    spoken_text, is_repeat, user_tone, action_keywords = await get_llm_response(request.text,
                                                               custom_personality=custom_personality,
                                                               mood_context=current_mood)

    # DEBUG: T1 — immediately after Gemini returns. T1-T0 isolates the FULL
    # Gemini round-trip, including any silent SDK-level retries — this
    # is the number that tells us if Gemini/network is the culprit.
    t1 = time.perf_counter()
    word_count = len(spoken_text.split())
    print(f"[TIMING] Gemini call: {t1 - t0:.3f}s | response word count: {word_count} | turn history length check happens inside get_llm_response, not visible here")

    # apply event from emotions module for mood to affect the response (write after response)
    apply_event(is_repeat, user_tone)

    # CHANGED — the guardrail, now plural. verified_actions is either
    # None (no action wanted this turn, or every requested keyword was
    # invalid) or an ORDERED LIST of [gadget, cmd, param] triples, each
    # correctly cased, ready for ARC to execute blindly and sequentially,
    # in this exact order, with zero validation on its side. A single
    # hallucinated keyword no longer costs the whole batch — see
    # verify_actions()'s docstring in services.py.
    verified_actions = verify_actions(action_keywords)

    # DEBUG: T2 — immediately after verify_actions. T2-T1 should be near-zero.
    # This is the control that proves/disproves action verification
    # as a cost center, instead of us just assuming it's negligible.
    t2 = time.perf_counter()
    print(f"[TIMING] verify_actions: {t2 - t1:.3f}s")

    # Take above spoken_text and perform Piper TTS
    # Offloaded to bg thread so it doesnt conflict with
    # Mood GET endpoint or vision POST endpoint when working
    # in parallel.     
    audio_bytes = await asyncio.to_thread(text_to_speech, spoken_text)         # spoken text -> audio bytes

    # DEBUG: T3 — immediately after Piper returns. T3-T2 isolates TTS synthesis
    # time specifically, separate from everything above it.
    t3 = time.perf_counter()
    print(f"[TIMING] text_to_speech (Piper): {t3 - t2:.3f}s")

    response = Response(content=audio_bytes, media_type="audio/wav")    # wav format through piper tts (for elevenlab its mpeg)

    # DEBUG: T_total — full route duration, for a sanity-check cross-reference
    # against whatever Swagger's own request duration shows you.
    print(f"[TIMING] TOTAL route time: {t3 - t0:.3f}s")

    # CHANGED — actions are still carried out-of-band from the audio body
    # via the same custom X-JD-Action response header (deliberately kept
    # in the existing /brain/chat response rather than moved to a
    # separate endpoint — actions have no independent existence outside a
    # single Gemini turn, unlike mood, which is why mood got its own
    # GET /emotion/state endpoint and actions did not).
    #
    # The header value is now a JSON array OF ARRAYS — a list of
    # [gadget, cmd, param] triples instead of one bare triple — e.g.
    # [["Auto Position","AutoPositionActionWait","StandFromSit"],
    #  ["Auto Position","AutoPositionActionWait","Wave"]]
    # json.dumps handles the extra nesting level with zero extra work;
    # C# already parses this with JArray.Parse, which handles nested
    # arrays exactly as well as flat ones — only the C#-side loop over
    # the outer array is new, not the parsing mechanism itself.
    #
    # Header is still OMITTED ENTIRELY when there are no verified actions
    # — verify_actions() already collapses "Gemini wanted nothing" and
    # "every requested keyword was invalid" to the same None, so this
    # check needs no changes beyond the variable's new name.
    if verified_actions is not None:
        response.headers["X-JD-Action"] = json.dumps(verified_actions)
        print(f"[Brain] X-JD-Action header set: {response.headers['X-JD-Action']}")
    else:
        print("[Brain] No X-JD-Action header sent this turn.")

    return response