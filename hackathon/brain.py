"""Brain module: Claude API calls, context assembly, insights loading.

Biometric context comes entirely from Daniel's insights.json file.
The agent never touches raw data — Daniel's pipeline handles that.
"""

import os
import json
from pathlib import Path
from datetime import datetime

import anthropic
from dotenv import load_dotenv

from hackathon.database import (
    get_conversation_history,
    get_or_create_profile,
    save_message,
)

# Load env
load_dotenv(Path(__file__).parent / ".env")

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

PROMPTS_DIR = Path(__file__).parent / "prompts"
DATA_DIR = Path(__file__).parent / "data"

# Cache for Daniel's insights file (hot-reload via mtime)
_insights_cache = {"data": None, "mtime": 0}
_calendar_cache = {"data": None, "mtime": 0}


def _load_system_prompt() -> str:
    return (PROMPTS_DIR / "system_prompt.md").read_text()


def _load_insights() -> dict | None:
    """Load Daniel's insights.json with mtime-based cache. Hot-reloads when file changes."""
    path = DATA_DIR / "insights.json"
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if mtime > _insights_cache["mtime"]:
        _insights_cache["data"] = json.loads(path.read_text())
        _insights_cache["mtime"] = mtime
    return _insights_cache["data"]


def _load_calendar() -> dict | None:
    """Load calendar.json with mtime-based cache."""
    path = DATA_DIR / "calendar.json"
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if mtime > _calendar_cache["mtime"]:
        _calendar_cache["data"] = json.loads(path.read_text())
        _calendar_cache["mtime"] = mtime
    return _calendar_cache["data"]


def _build_calendar_context() -> str:
    """Assemble calendar context from calendar.json."""
    cal = _load_calendar()
    if not cal:
        return ""

    items = cal.get("items", [])
    if not items:
        return ""

    lines = ["## Upcoming Calendar"]
    for item in items:
        start = item["start"]["dateTime"]
        summary = item["summary"]
        desc = item.get("description", "")
        location = item.get("location", "")
        line = f"- **{start}** -- {summary}"
        if location:
            line += f" @ {location}"
        lines.append(line)
        if desc:
            lines.append(f"  Context: {desc}")
    return "\n".join(lines)


def _build_biometric_context() -> str:
    """Assemble biometric context from Daniel's insights.json."""
    insights = _load_insights()
    if not insights:
        return "## Biometric Data\nNo biometric data loaded yet. If the user asks about their data, let them know you're waiting for their first test results."

    parts = ["## Biometric Insights (from Pison data pipeline)"]
    parts.append(json.dumps(insights, indent=2))
    return "\n".join(parts)


def _build_profile_context(phone: str) -> str:
    """Build user profile context block."""
    profile = get_or_create_profile(phone)
    lines = ["## User Profile"]
    if profile.get("name"):
        lines.append(f"- Name: {profile['name']}")
    if profile.get("why_category"):
        lines.append(f"- Why they test: {profile['why_category']}")
    if profile.get("routine_anchor"):
        lines.append(f"- Routine anchor: {profile['routine_anchor']}")
    if profile.get("identity_segment"):
        lines.append(f"- Identity: {profile['identity_segment']}")
    if profile.get("goals"):
        lines.append(f"- Goals: {profile['goals']}")
    lines.append(f"- Streak: {profile.get('streak_count', 0)} days")
    lines.append(f"- Relationship stage: {profile.get('relationship_stage', 'translator')}")
    lines.append(f"- Current time: {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}")
    return "\n".join(lines)


def _count_messages(phone: str) -> int:
    """Count total messages exchanged to determine relationship stage."""
    history = get_conversation_history(phone, limit=9999)
    return len(history)


def get_response(phone: str, user_message: str) -> str:
    """Generate a response from the nervous system agent."""

    # Save the incoming message
    save_message(phone, "user", user_message)

    # Build full system prompt with context
    system_prompt = _load_system_prompt()
    profile_context = _build_profile_context(phone)
    biometric_context = _build_biometric_context()
    calendar_context = _build_calendar_context()

    # Determine relationship stage
    msg_count = _count_messages(phone)
    if msg_count <= 5:
        stage = "translator"
        stage_note = "You are in TRANSLATOR mode (early relationship). Report data clearly and simply. Don't assume familiarity."
    elif msg_count <= 20:
        stage = "pattern_finder"
        stage_note = "You are in PATTERN FINDER mode. Start connecting dots and noticing trends."
    elif msg_count <= 50:
        stage = "anticipator"
        stage_note = "You are in ANTICIPATOR mode. Make predictions based on established patterns."
    else:
        stage = "integrated"
        stage_note = "You are in INTEGRATED AWARENESS mode. You and the user think as one system."

    full_system = f"""{system_prompt}

---

# CURRENT CONTEXT

{profile_context}

{biometric_context}

{calendar_context}

## Relationship Stage
{stage_note}

## Message Count
This is message #{msg_count + 1} in your conversation. Adjust depth accordingly.
"""

    # Build conversation history for Claude
    history = get_conversation_history(phone, limit=50)
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Current message is already in history from save_message above,
    # but let's make sure it's there
    if not messages or messages[-1]["content"] != user_message:
        messages.append({"role": "user", "content": user_message})

    # Call Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,  # Keep responses short like texts
        system=full_system,
        messages=messages,
    )

    assistant_message = response.content[0].text

    # Save response
    save_message(phone, "assistant", assistant_message)

    return assistant_message
