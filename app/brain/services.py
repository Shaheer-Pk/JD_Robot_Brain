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

# The physical action / soundboard whitelist. Loaded once at module
# import time, same pattern as everything else above. This is the single
# source of truth for (a) what gets shown to Gemini in the system
# prompt, (b) what verify_actions() checks Gemini's response against,
# and (c) what pose metadata the pose guard-rail simulates against —
# THREE consumers of one dict, not three separately maintained lists.
#
# CHANGED this session — each value is no longer a flat [gadget, cmd,
# param] triple. Each value is now a dict:
#   {"triple": [gadget, cmd, param], "requires_pose": str, "results_pose": str}
# Every one of the 22 real entries carries real, non-null pose values —
# there is no "pose doesn't matter for this action" case left in this
# whitelist as of this session (see robot_profile.json).
POSSIBLE_ACTIONS = _profile["possible_action_list"]
 
# A second, normalized-key version of the SAME dict, built once at import
# time — NOT rebuilt per-request. Keys are lowercased/stripped for
# case-insensitive, whitespace-tolerant matching against whatever free
# text Gemini actually returns. Values are left completely untouched —
# still the real nested {"triple", "requires_pose", "results_pose"}
# dicts ARC/the pose guard-rail need. Only the lookup KEY is normalized.
_NORMALIZED_ACTIONS = {k.strip().lower(): v for k, v in POSSIBLE_ACTIONS.items()}
 
# Built once — the plain keyword list, comma-joined, injected into every
# system prompt. Kept as-is (unchanged from before) as a quick-scan list
# — ACTION_POSE_TABLE_TEXT below is the detailed version with pose data.
ACTION_LIST_TEXT = ", ".join(POSSIBLE_ACTIONS.keys())
 
# NEW this session — a readable, per-action pose table, rendered ONCE at
# import time by reading POSSIBLE_ACTIONS. This is DERIVED TEXT, not a
# second data source: nothing in this codebase ever looks anything up
# inside this string, the same way nothing looks anything up inside
# ACTION_LIST_TEXT above. It exists solely so Gemini can read each
# action's pose requirements in the prompt. If robot_profile.json
# changes, this is automatically correct on next server restart with
# zero manual edits here — same "single source of truth, multiple
# derived views" pattern already established by ACTION_LIST_TEXT and
# _NORMALIZED_ACTIONS.
_pose_table_lines = [
    f"{name} - requires: {meta['requires_pose']}, results in: {meta['results_pose']}"
    for name, meta in POSSIBLE_ACTIONS.items()
]
ACTION_POSE_TABLE_TEXT = "\n".join(_pose_table_lines)
 
 
# get_llm_response is kept async def because it awaits
# gemini response. As a result this method SHOULD NOT BE
# WRAPPED in a asyncio.to_thread() block in brain/routers.py
# just like text_to_speech has been wrapped and is kept as a plain
# 'def' (not async def).
async def get_llm_response(
    text: str,
    custom_personality: dict | None = None,
    mood_context: str | None = "neutral",
    pose_context: str | None = "standing",
) -> tuple[str, bool, str, list[str] | None]:
    # In case vision was able to recognize and 
    # send custom_personality over to brain/routers.py
    # Otherwise router keeps this as none, which means load
    # default guest persona from robot_profile.json
    if custom_personality:
        persona = f"You are talking with {custom_personality['name']}. Behave towards them as follows: {custom_personality['persona']}"
    else:
        persona = DEFAULT_GUEST_PERSONA
 
    mood_line = f"JD's current mood is: {mood_context if mood_context else 'neutral'}." # neutral mood fallback
 
    # CHANGED this session — pose_context now flows the same way
    # mood_context already does: read fresh in routers.py before this
    # call, passed in as a plain parameter, never imported directly by
    # this module (keeps services.py from acquiring a second direct
    # dependency on brain-owned mutable state beyond what it already
    # reads via parameters).
    pose_line = f"JD's current pose is: {pose_context if pose_context else 'standing'}."
 
    # CHANGED this session — instructs Gemini it may return MULTIPLE
    # actions, in the order they should execute, AND now gives Gemini
    # the full per-action pose table so it can reason about sequencing
    # itself, before the guard-rail ever has to intervene. This is a
    # real, functioning mitigation as of this session — NOT just a
    # prompt-level nudge with no backstop. verify_actions() (below)
    # independently re-checks and enforces pose correctness on the
    # server side, regardless of what Gemini decides — see that
    # function's docstring for the full guard-rail mechanism. The
    # prompt guidance below reduces how OFTEN the guard-rail needs to
    # actually intervene; it does not replace the guard-rail.
    action_guidance = (
        f"If one or more physical actions or sounds genuinely fit this moment, "
        f"choose from this list: {ACTION_LIST_TEXT}. Return them as a list of "
        f"keywords in the actions field, IN THE ORDER they should execute. "
        f"If none fit, or you are unsure, return null or an empty list. Do not "
        f"invent an action name that is not in this list.\n\n"
        f"{pose_line}\n\n"
        f"Each action below requires JD to be in a specific pose before it can "
        f"run, and leaves JD in a specific pose afterward:\n"
        f"{ACTION_POSE_TABLE_TEXT}\n\n"
        f"Actions execute in the order you list them, and JD's pose updates "
        f"after each one completes — when building your batch, check each "
        f"action's required pose against what JD's pose will actually be at "
        f"that point in the sequence, not just JD's pose right now. If a "
        f"needed pose doesn't match, insert the correct transition action "
        f"first, before the action that needs it."
    )
 
    system_prompt = IDENTITY_AND_CAPABILITIES + "\n\n" + persona + "\n\n" + TONE_CLASSIFICATION_GUIDANCE + "\n\n" + MOOD_INFLUENCE_GUIDANCE + "\n\n" + mood_line + "\n\n" + action_guidance
 
    # Transform our internal {"user": ..., "jd": ...} turn dicts into the
    # role-tagged types.Content objects Gemini's contents parameter expects.
    contents: list[types.Content] = []
    for turn in memory_session.get_history():
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=turn["user"])])
        )
        contents.append(
            types.Content(role="model", parts=[types.Part.from_text(text=turn["jd"])])
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
    # verification touches it.
    print(f"[Brain] Gemini raw actions returned: {result.actions!r}")
 
    # CHANGED this session — memory_session.add_turn() call REMOVED from
    # here. Previously this fired unconditionally right after the Gemini
    # response, BEFORE verify_actions() ever ran (verify_actions() is
    # only ever called later, in routers.py). That ordering meant memory
    # could never distinguish "Gemini requested a real action" from
    # "Gemini hallucinated a keyword that got rejected" — both would
    # look identical to a future turn reading memory back, if action
    # data were ever recorded here. Since Layer 1 (recording verified
    # action history into memory text) now depends on knowing the
    # POST-VERIFICATION result, this call MUST happen after
    # verify_actions() resolves — which this function has no visibility
    # into. add_turn() now lives in routers.py instead, called once,
    # after verify_actions() returns. See routers.py and
    # memory.py's docstring for the full rationale.
    #
    # ORIGINAL LINE, now deleted from this function:
    #   memory_session.add_turn(text, result.response)
 
    # return the response that will be used by the robot to be spoken to the user,
    # the repeat check and user tone to change JD's mood accordingly, and
    # the raw, ORDERED list of action keywords Gemini picked (or None/
    # empty) — unverified at this point, verify_actions() (below) is what
    # actually validates each item.
    return result.response, result.is_repeat, result.user_tone, result.actions
 
 
def verify_actions(
    action_keywords: list[str] | None, current_pose: str
) -> tuple[list[list[str]] | None, list[str], str]:
    """
    Checks an ORDERED LIST of keywords Gemini requested against the real
    whitelist AND against JD's tracked physical pose. This is the ONLY
    place a Gemini-suggested action is checked before real dispatch —
    routers.py trusts whatever this function returns without
    re-validating.
 
    CHANGED this session — this function now does TWO independent kinds
    of verification per item, not one, and they behave DIFFERENTLY on
    failure. Read both carefully, they are not interchangeable:
 
    1. HALLUCINATION CHECK (unchanged from before): is this keyword a
       real entry in the whitelist at all? If not, that ONE item is
       dropped, and the rest of the batch is still evaluated normally,
       against an UNCHANGED simulated pose. A single hallucinated
       keyword never costs the rest of the batch.
 
    2. POSE CHECK (NEW this session — the actual hardware-safety guard-
       rail this session was built for): does this real, whitelisted
       action's `requires_pose` match JD's CURRENTLY SIMULATED pose at
       this exact point in the batch? The simulated pose starts at
       `current_pose` (JD's real tracked pose entering this turn) and
       updates after every ACCEPTED action to that action's
       `results_pose` — so a later item in the same batch is checked
       against the pose JD would be in AFTER earlier accepted items ran,
       not against the turn's starting pose. If a real action's
       required pose does NOT match the simulated pose at that point,
       that item is rejected AND EVERY ITEM AFTER IT IN THE BATCH IS
       DISCARDED TOO (truncation) — nothing after a broken link in a
       sequential, order-dependent chain can be trusted to start from
       the pose it assumed. This is a DELIBERATE ASYMMETRY versus
       the hallucination case above: a bad pose match is more costly
       than a bad keyword, on purpose, because pose is safety-critical
       and keyword validity is not.
 
    CONCRETE EXAMPLE OF THE ASYMMETRY: JD is standing.
    ["Wave", "HallucinatedNonsense", "Bow"] -> hallucination only ->
    verifies to [Wave's triple, Bow's triple] — the bad item is skipped,
    everything else survives.
    ["Wave", "SitDown", "Pushups"] -> Wave accepted (standing->standing),
    SitDown accepted (standing->sitting, simulated pose now "sitting"),
    Pushups REJECTED (requires "standing", simulated pose is "sitting")
    -> TRUNCATED -> verifies to [Wave's triple, SitDown's triple] only.
    Pushups does not fire. This is intentional, physically-motivated
    behavior, not a bug — see tasks.md Part 0/0.5 for the hardware fall
    this exists to prevent.
 
    GENUINE LIMITATION, not fixed by this function: the simulated pose
    this function reasons over is a PREDICTION based on `current_pose`,
    which is itself an unconfirmed, un-verified-by-hardware value (see
    pose_state.py's docstring). This function has no way to know if a
    PRIOR turn's action actually executed on real hardware — it can only
    trust that it did. This guard-rail raises the safety ceiling; it is
    not an absolute physical guarantee.
 
    Returns a 3-tuple: (verified_action_triples, verified_action_names,
    final_simulated_pose).
    - verified_action_triples: list[list[str]] | None — the [gadget,
      cmd, param] triples surviving verification, in original order, or
      None if nothing survived (Gemini requested nothing, or every item
      was rejected). Unchanged shape/meaning from before — this is what
      routers.py sends to ARC via the X-JD-Action header.
    - verified_action_names: list[str] — the KEYWORD NAMES (e.g.
      "StandFromSit") corresponding 1:1 with verified_action_triples, in
      the same order. NEW this session — needed so routers.py can build
      a human-readable action note for conversation memory (see
      memory.py's docstring) without exposing raw ARC triples into
      Gemini-readable memory text. Empty list if nothing survived.
    - final_simulated_pose: str — the pose JD is predicted to be in
      after every surviving action in the returned list executes. Equal
      to `current_pose` unchanged if nothing survived (nothing executed,
      so pose cannot have changed). routers.py writes this into
      pose_state.py's PoseState via set_pose() when actions survived.
 
    Normalization: strips whitespace and lowercases before comparing
    each keyword against _NORMALIZED_ACTIONS, since Gemini is a language
    model producing free text, not a strict enum picker. Unchanged from
    the original design.
    """
    if not action_keywords:
        # Covers both None and an empty list — Gemini is instructed to
        # use either for "no action this turn."
        print("[Verify] No actions requested — Gemini returned null/empty list.")
        return None, [], current_pose
 
    verified_triples: list[list[str]] = []
    verified_names: list[str] = []
    simulated_pose = current_pose
 
    for keyword in action_keywords:
        if keyword is None:
            continue
 
        normalized = keyword.strip().lower()
        meta = _NORMALIZED_ACTIONS.get(normalized)
 
        # --- Hallucination check ---
        if meta is None:
            print(f"[Verify] REJECTED (unknown keyword) — '{keyword}' not found in whitelist. Dropped; rest of batch still evaluated against unchanged simulated pose '{simulated_pose}'.")
            continue
 
        # --- Pose check, against the DYNAMIC simulated pose, not the
        # turn's original starting pose ---
        requires = meta["requires_pose"]
        if requires != simulated_pose:
            print(f"[Verify] REJECTED (pose mismatch) — '{keyword}' requires pose '{requires}' but simulated pose at this point in the batch is '{simulated_pose}'. TRUNCATING remainder of batch — nothing after this item will be evaluated.")
            break  # Option C: truncate, do not continue evaluating the rest
 
        # --- Accepted: record it and advance the simulated pose ---
        verified_triples.append(meta["triple"])
        verified_names.append(keyword)
        simulated_pose = meta["results_pose"]
        print(f"[Verify] ACCEPTED — '{keyword}' matched -> {meta['triple']} | simulated pose now '{simulated_pose}'")
 
    if not verified_triples:
        print("[Verify] No valid actions survived verification for this turn.")
        return None, [], current_pose
 
    return verified_triples, verified_names, simulated_pose
 
 
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
# Not kept 'async def' because there is no genuine await
# present here and in brain/routers.py its wrapped in
# asyncio.to_thread() which expects SYNCHRONUS BLOCKING CALLS
# Not async def
def text_to_speech(text: str) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        piper_voice.synthesize_wav(text, wav_file)
    return buffer.getvalue()