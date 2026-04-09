"""Export video insights as JSON and SRT."""

import json
from pathlib import Path

from .models import Transcript, VideoInsights


def export_json(insights: VideoInsights, output_path: Path) -> Path:
    """Export insights as a formatted JSON file.

    Args:
        insights: The complete VideoInsights model.
        output_path: Where to write the JSON file.

    Returns:
        The output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(insights.model_dump_json())
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    return output_path


def export_srt(transcript: Transcript | None, output_path: Path) -> Path:
    """Export transcript as SRT subtitle file.

    Args:
        transcript: Transcript model with segments.
        output_path: Where to write the SRT file.

    Returns:
        The output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not transcript or not transcript.segments:
        output_path.write_text("")
        return output_path

    lines: list[str] = []
    for i, seg in enumerate(transcript.segments, start=1):
        start_srt = _seconds_to_srt_time(seg.start)
        end_srt = _seconds_to_srt_time(seg.end)
        lines.append(f"{i}")
        lines.append(f"{start_srt} --> {end_srt}")
        lines.append(seg.text)
        lines.append("")

    output_path.write_text("\n".join(lines))
    return output_path


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
