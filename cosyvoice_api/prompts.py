"""CosyVoice3 prompt formatting and synthesis parameter validation."""

from __future__ import annotations

import math
from typing import Any

from cosyvoice_api.config import END_OF_PROMPT, MAX_TTS_SPEED, MIN_TTS_SPEED, SYSTEM_PROMPT


def format_prompt_text(text: str) -> str:
    """Wrap plain text with CosyVoice3's system prompt and end-of-prompt marker."""
    cleaned = text.strip()
    if END_OF_PROMPT in cleaned:
        return cleaned
    return f"{SYSTEM_PROMPT}{END_OF_PROMPT}{cleaned}"


def format_instruction(instruction: str) -> str:
    """Wrap a plain style instruction in CosyVoice3's expected instruct2 format."""
    cleaned = instruction.strip()
    if END_OF_PROMPT in cleaned:
        return cleaned
    return f"{SYSTEM_PROMPT} {cleaned}{END_OF_PROMPT}"


def parse_speed(raw_speed: Any) -> float:
    """Parse and validate a speech-rate multiplier within [MIN_TTS_SPEED, MAX_TTS_SPEED]."""
    if isinstance(raw_speed, bool):
        raise ValueError("'speed' must be a number.")

    try:
        speed = float(raw_speed)
    except (TypeError, ValueError):
        raise ValueError("'speed' must be a number.") from None

    if not math.isfinite(speed):
        raise ValueError("'speed' must be a finite number.")

    if not MIN_TTS_SPEED <= speed <= MAX_TTS_SPEED:
        raise ValueError(f"'speed' must be between {MIN_TTS_SPEED:g} and {MAX_TTS_SPEED:g}.")

    return speed
