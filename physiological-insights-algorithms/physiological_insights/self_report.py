"""Parse self-reported stress/sleepiness/sharpness from freetext test comments."""

import re
import pandas as pd
import numpy as np

# Patterns handle common misspellings and variants found in the data
_STRESS_RE = re.compile(r"stress[:\s]*(\d+\.?\d*)\s*/\s*10", re.IGNORECASE)
_SLEEPINESS_RE = re.compile(
    r"(?:sleep(?:iness)?|sleapiness|sleepiness)[:\s]*(\d+\.?\d*)\s*/\s*10",
    re.IGNORECASE,
)
_SHARPNESS_RE = re.compile(
    r"(?:(?:subjective\s+)?sharpness)[:\s]*(\d+\.?\d*)\s*/\s*10",
    re.IGNORECASE,
)

# Strip the numeric rating lines to extract the freetext context
_RATING_LINE_RE = re.compile(
    r"^.*(?:stress|sleep(?:iness)?|sleapiness|sharpness)[:\s]*\d+\.?\d*\s*/\s*10.*$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_float(pattern: re.Pattern, text: str) -> float | None:
    m = pattern.search(text)
    if m:
        return float(m.group(1))
    return None


def _extract_context_note(text: str) -> str:
    """Return the freetext portion of a comment with rating lines removed."""
    cleaned = _RATING_LINE_RE.sub("", text)
    lines = [ln.strip() for ln in cleaned.strip().splitlines() if ln.strip()]
    note = " ".join(lines)
    # Collapse references like "same as test about a minute ago"
    if len(note) > 200:
        note = note[:200] + "..."
    return note


def parse_comment(text: str) -> dict:
    """Parse a single comment string into structured fields."""
    if not text or pd.isna(text):
        return {"stress": None, "sleepiness": None, "sharpness": None, "context_note": ""}
    text = str(text)
    return {
        "stress": _extract_float(_STRESS_RE, text),
        "sleepiness": _extract_float(_SLEEPINESS_RE, text),
        "sharpness": _extract_float(_SHARPNESS_RE, text),
        "context_note": _extract_context_note(text),
    }


def parse_all_comments(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed self-report columns to the test-results DataFrame."""
    parsed = df["comment"].apply(parse_comment).apply(pd.Series)
    for col in ("stress", "sleepiness", "sharpness"):
        df[col] = pd.to_numeric(parsed[col], errors="coerce")
    df["context_note"] = parsed["context_note"]
    return df
