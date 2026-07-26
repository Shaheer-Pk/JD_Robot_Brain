import os
import json
import io
import wave
# import httpx          # Use for elevenlabs text_to_speech

from piper import PiperVoice
from google import genai
from google.genai import types

from app.brain.schemas import LLMTurnResult
from app.shared.memory import memory_session

# Importing from our env file (hidden)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# Load the Piper voice once at module load time, same pattern as the Gemini client
piper_voice = PiperVoice.load("voices/en-US-cori-medium.onnx")

# Uncomment when using elevenlabs and not piper
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")          
# ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# Load robot_profile.json once at module load time, never re-read per request
# os.path.dirname(__file__) is an inbuilt python vari which fetches file path relatively so for this its something like
# D:\ARC JD Project\JD_Robot_Brain\app\brain\services.py
# Then when we do os.path.dirname we cut off servicies.py at the end to get:
# D:\ARC JD Project\JD_Robot_Brain\app\brain
# And then with os.path.join we get 
# D:\ARC JD Project\JD_Robot_Brain\app\brain\robot_profile.json
# This helps in dynamically searching for robot_profile.json specifically in brain independent from which folder uvicorn is run from
with open(os.path.join(os.path.dirname(__file__), "robot_profile.json"), "r") as f:
    _profile = json.load(f)

# Key:Value pairs from \brain\robot_profile.json
IDENTITY_AND_CAPABILITIES = _profile["identity_and_capabilities"]
DEFAULT_GUEST_PERSONA = _profile["default_guest_persona"]
TONE_CLASSIFICATION_GUIDANCE = _profile["tone_classification_guidance"]
MOOD_INFLUENCE_GUIDANCE = _profile["mood_influence_guidance"]

# NEW — the physical action / soundboard whitelist. Loaded once at module
# import time, same pattern as everything else above. This is the single
# source of truth for both (a) what gets shown to Gemini in the system
# prompt, and (b) what verify_action() checks Gemini's response against —
# deliberately the SAME dict for both purposes, not two separately
# maintained lists, per this project's established "two sources of truth
# is always a bug" principle.
POSSIBLE_ACTIONS = _profile["possible_action_list"]

# A second, normalized-key version of the SAME dict, built once at import
# time — NOT rebuilt per-request. Keys are lowercased/stripped for
# case-insensitive, whitespace-tolerant matching against whatever free
# text Gemini actually returns (Gemini is not guaranteed to echo back
# the exact casing/spacing of a keyword it saw in a paragraph of prompt
# text). Values are left completely untouched — they still hold the
# real, correctly-cased triples ARC needs. Only the lookup KEY is
# normalized; the data being matched TO is never altered.
_NORMALIZED_ACTIONS = {k.strip().lower(): v for k, v in POSSIBLE_ACTIONS.items()}

# Built once — the plain keyword list, comma-joined, injected into every
# system prompt. Gemini only ever sees action NAMES here, never the
# underlying gadget/cmd ARC plumbing, which is meaningless to an LLM
# deciding "does a physical action fit this moment."
ACTION_LIST_TEXT = ", ".join(POSSIBLE_ACTIONS.keys())


async def get_llm_response(text: str, custom_personality: dict | None = None, mood_context: str | None = "neutral") -> tuple[str, bool, str, list[str] | None]:
    # In case vision was able to recognize and 
    # send custom_personality over to brain/routers.py
    # Otherwise router keeps this as none, which means load
    # default guest persona from robot_profile.json
    if custom_personality:
        persona = f"You are talking with {custom_personality['name']}. Behave towards them as follows: {custom_personality['persona']}"
    else:
        persona = DEFAULT_GUEST_PERSONA

    mood_line = f"JD's current mood is: {mood_context if mood_context else 'neutral'}." # neutral mood fallback

    # CHANGED this session — instructs Gemini it may now return MULTIPLE
    # actions, in the order they should execute, not just one. Built from
    # the same POSSIBLE_ACTIONS dict verify_actions() checks against
    # below, so the prompt and the guardrail can never silently drift out
    # of sync with each other.
    #
    # The ordering/prerequisite instruction below is a real, deliberate
    # mitigation for the exact bug this feature was built to fix
    # (Gemini asking for a pose-dependent action like a dance without
    # first requesting the stand-up action it physically requires) — it
    # is a PROMPT-LEVEL nudge only, not a guarantee. verify_actions()
    # does NOT reject or reorder based on pose correctness; there is no
    # server-side pose-prerequisite enforcement yet (tracking JD's
    # current resting pose and feeding it back into this prompt was
    # discussed and deliberately parked as separate, not-yet-built future
    # work — do not assume this comment means it is already handled).
    action_guidance = (
        f"If one or more physical actions or sounds genuinely fit this moment, "
        f"choose from this list: {ACTION_LIST_TEXT}. Return them as a list of "
        f"keywords in the actions field, IN THE ORDER they should execute. "
        f"Some actions require a specific starting pose (e.g. a standing dance "
        f"cannot run from a seated position) — if the action you want requires "
        f"a pose JD may not currently be in, include the necessary pose-change "
        f"action(s) first in the list. If none fit, or you are unsure, return "
        f"null or an empty list. Do not invent an action name that is not in "
        f"this list."
    )

    system_prompt = IDENTITY_AND_CAPABILITIES + "\n\n" + persona + "\n\n" + TONE_CLASSIFICATION_GUIDANCE + "\n\n" + MOOD_INFLUENCE_GUIDANCE + "\n\n" + mood_line + "\n\n" + action_guidance
    # Evaluated at runtime (based on face recognition)       
    # Commented this because made memory_session a single sharabale session to be utilized
    # over all modules as now vision/state.py will reset memory on_face_lost()

    # Transform our internal {"user": ..., "jd": ...} turn dicts into the
    # role-tagged types.Content objects Gemini's contents parameter expects.
    # This translation lives here, not in ConversationMemory, because the
    # memory class has zero knowledge of Gemini's prompt schema — it just
    # stores turns. If the LLM provider or prompt format ever changes,
    # only this function needs to change, not the memory class.
    contents: list[types.Content] = []
    for turn in memory_session.get_history():
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=turn["user"])])     # Here 'user' is the key in dict in app/shared/memory.py        
        )
        contents.append(
            types.Content(role="model", parts=[types.Part.from_text(text=turn["jd"])])      # Same as 'user'
        )

    # The new incoming question is appended last, after all prior history.
    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=text)])
    )

    response = await client.aio.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=LLMTurnResult
)
    )

    result: LLMTurnResult = response.parsed     # parse once, store in a variable (use LLMTurnResult class schema)

    # DEBUG — permanent diagnostic, same convention as vision/state.py's
    # debug prints. Shows exactly what Gemini returned before any
    # verification touches it — critical for telling "Gemini never wanted
    # any action" apart from "Gemini wanted some, but one or all got
    # rejected." !r gives you the repr() so None prints as None and an
    # empty list prints as [] (visibly distinct from a populated list).
    print(f"[Brain] Gemini raw actions returned: {result.actions!r}")

    # Store this exchange AFTER a successful response, so a failed/errored
    # call never gets recorded as if JD actually said something.
    memory_session.add_turn(text, result.response)


    # return the response that will be used by the robot to be spoken to the user,
    # the repeat check and user tone to change JD's mood accordingly, and NOW
    # also the raw, ORDERED list of action keywords Gemini picked (or None/
    # empty) — unverified at this point, verify_actions() (below) is what
    # actually validates each item.
    return result.response, result.is_repeat, result.user_tone, result.actions


def verify_actions(action_keywords: list[str] | None) -> list[list[str]] | None:
    """
    CHANGED this session — was verify_action() (singular), checking one
    keyword. Now verify_actions() (plural): checks an ORDERED LIST of
    keywords, since Gemini can now request more than one action per turn
    (e.g. stand up, then dance). This is still the ONLY place a
    Gemini-suggested action is checked against the real whitelist —
    routers.py trusts whatever this function returns without
    re-validating, exactly as before.

    PER-ITEM verification, NOT all-or-nothing (confirmed decision this
    session): if the list contains one hallucinated/invalid keyword among
    otherwise-valid ones, that ONE item is dropped and every other valid
    item in the list is still kept, in its original relative order. A
    single bad entry must not cost JD a perfectly good, verified
    sequence — e.g. ["StandFromSit", "HallucinatedNonsense", "Wave"]
    verifies to [StandFromSit's triple, Wave's triple], not None. This
    mirrors the existing single-action design's philosophy at the
    per-item level: a malformed/unmatched entry collapses to "that one
    thing doesn't happen," never a whole-request failure.

    Returns None (not an empty list) when nothing survives verification
    — either Gemini returned null/an empty list to begin with, or every
    item it did return was invalid. routers.py's existing
    `if verified_actions is not None` check (see routers.py) then behaves
    identically to the old single-action code path — no header sent.

    Normalization: strips whitespace and lowercases before comparing each
    keyword against _NORMALIZED_ACTIONS, since Gemini is a language model
    producing free text, not a strict enum picker — exact-match-only
    would silently reject valid actions over trivial casing/whitespace
    differences. Unchanged from the original single-action design.

    NOTE — pose/sequencing correctness (e.g. "don't dance before you
    stand up") is NOT enforced here. This function only checks "is this
    keyword a real, whitelisted action" — it has no concept of JD's
    current physical pose and does not reorder, insert, or reject items
    based on prerequisites. That reasoning is currently pushed entirely
    onto Gemini via the action_guidance prompt text (see
    get_llm_response() above) — a real, known gap, not an oversight.
    Actually PREVENTING servo damage from a genuinely wrong pose sequence
    still rests on your own manual/testing verification of what
    sequences you allow through, not on this function.
    """
    if not action_keywords:
        # Covers both None and an empty list — Gemini is instructed to
        # use either for "no action this turn" (see action_guidance).
        print("[Verify] No actions requested — Gemini returned null/empty list.")
        return None

    verified: list[list[str]] = []
    for keyword in action_keywords:
        if keyword is None:
            continue

        normalized = keyword.strip().lower()

        # Same O(1) lookup against the pre-normalized dict as the
        # original single-action design — just called once per item now.
        result = _NORMALIZED_ACTIONS.get(normalized)

        if result is None:
            print(f"[Verify] REJECTED — '{keyword}' (normalized: '{normalized}') not found in whitelist. Dropped; rest of the batch still evaluated.")
            continue

        print(f"[Verify] ACCEPTED — '{keyword}' matched -> {result}")
        verified.append(result)

    if not verified:
        print("[Verify] No valid actions survived verification for this turn.")
        return None

    return verified


# UNCOMMENT if you want to use eleven labs
# async def text_to_speech(text: str) -> bytes:
#     url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
#     headers = {
#         "xi-api-key": ELEVENLABS_API_KEY,
#         "Content-Type": "application/json"
#     }
#     payload = {"text": text}

#     async with httpx.AsyncClient() as client:
#         response = await client.post(url, json=payload, headers=headers)
#         response.raise_for_status()
#         return response.content

# Piper version
async def text_to_speech(text: str) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        piper_voice.synthesize_wav(text, wav_file)
    return buffer.getvalue()