"""Tier 2: LLM analyst — compress analysis_output.json into agent_briefing.json."""

import json
import os
from pathlib import Path


_PROMPT_PATH = Path(__file__).parent / "prompts" / "analyst_system.txt"


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text()


def _call_openai(packet_json: str, system_prompt: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": packet_json},
        ],
        temperature=0.0,
        max_tokens=1500,
    )
    return response.choices[0].message.content


def _call_anthropic(packet_json: str, system_prompt: str, model: str) -> str:
    from anthropic import Anthropic
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1500,
        system=system_prompt,
        messages=[
            {"role": "user", "content": packet_json},
        ],
        temperature=0.0,
    )
    return response.content[0].text


def generate_briefing(packet: dict, provider: str = "openai", model: str = "gpt-4o-mini") -> dict:
    """Call the analyst LLM to compress the full analysis into a compact briefing."""
    system_prompt = _load_system_prompt()

    slim_packet = dict(packet)

    if "circadian_profile" in slim_packet and slim_packet["circadian_profile"]:
        circ = dict(slim_packet["circadian_profile"])
        circ.pop("fitted_curve", None)
        slim_packet["circadian_profile"] = circ

    if "sleep_sessions" in slim_packet and slim_packet["sleep_sessions"]:
        ss = dict(slim_packet["sleep_sessions"])
        ss.pop("nights", None)
        slim_packet["sleep_sessions"] = ss

    packet_json = json.dumps(slim_packet, indent=2, default=str)

    if provider == "openai":
        raw = _call_openai(packet_json, system_prompt, model)
    elif provider == "anthropic":
        raw = _call_anthropic(packet_json, system_prompt, model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    # Parse the response — strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        briefing = json.loads(text)
    except json.JSONDecodeError:
        briefing = {
            "briefing_version": "1.0",
            "error": "LLM response was not valid JSON",
            "raw_response": text[:2000],
        }

    return briefing
