"""
app/emotion/triggers.py

Deliberately empty for now. Room to grow: future mood triggers beyond
"is_repeat" and "user_tone" - e.g. reacting to a recognized returning
user once the vision module is wired into the brain's prompt, or a
teaching-mode detector - should get added as new functions here, not
crammed into services.py. Keeps services.py focused on the core mood
math (decay, clamping, presets) instead of growing indefinitely every
time someone thinks of a new thing that should affect JD's mood.

Nothing to import from this file yet - it exists as a marked landing
spot for that future work.
"""