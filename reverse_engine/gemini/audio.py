"""Gemini-powered audio analysis."""

from pathlib import Path

from ..models import AudioEvent
from .client import GeminiClient, parse_json_response

AUDIO_PROMPT = """\
Analyze this audio track and classify the audio events you hear.
Return a JSON object with an "events" array. Each event should have:
- label: category (music, silence, laughter, applause, crowd_noise, engine_sound, siren, speech, ambient, other)
- start: start time in seconds
- end: end time in seconds
- confidence: 0-1 confidence score

Cover the full duration. Return valid JSON only. No markdown fences.
"""


def analyze_audio(
    client: GeminiClient,
    audio_path: Path,
) -> list[AudioEvent]:
    """Analyze audio using Gemini.

    Args:
        client: GeminiClient instance.
        audio_path: Path to audio file (typically the no_vocals stem).

    Returns:
        List of AudioEvent models.
    """
    raw = client.analyze_audio(audio_path=audio_path, prompt=AUDIO_PROMPT)
    data = parse_json_response(raw)

    events = data if isinstance(data, list) else data.get("events", [])

    return [
        AudioEvent(
            label=e["label"],
            start=e["start"],
            end=e["end"],
            confidence=e.get("confidence", 0.0),
        )
        for e in events
    ]
