from __future__ import annotations

import math

import pytest

from cosyvoice_api.prompts import format_instruction, format_prompt_text, parse_speed


def test_format_prompt_text_wraps_plain_text() -> None:
    assert format_prompt_text("  Hello world  ") == (
        "You are a helpful assistant.<|endofprompt|>Hello world"
    )


def test_format_prompt_text_preserves_existing_marker() -> None:
    raw = "Custom system.<|endofprompt|>Keep as-is"
    assert format_prompt_text(f"  {raw}  ") == raw


def test_format_instruction_wraps_plain_instruction() -> None:
    assert format_instruction("  Speak cheerfully.  ") == (
        "You are a helpful assistant. Speak cheerfully.<|endofprompt|>"
    )


def test_format_instruction_preserves_existing_marker() -> None:
    raw = "You are a helpful assistant. Whisper.<|endofprompt|>"
    assert format_instruction(raw) == raw


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.5, 0.5),
        (1, 1.0),
        (1.25, 1.25),
        ("1.75", 1.75),
        (2.0, 2.0),
    ],
)
def test_parse_speed_valid(raw: object, expected: float) -> None:
    assert parse_speed(raw) == expected


@pytest.mark.parametrize(
    "invalid",
    [
        True,
        False,
        None,
        "fast",
        math.nan,
        math.inf,
        -math.inf,
        0.49,
        2.01,
        -1.0,
    ],
)
def test_parse_speed_rejects_invalid(invalid: object) -> None:
    with pytest.raises(ValueError):
        parse_speed(invalid)
