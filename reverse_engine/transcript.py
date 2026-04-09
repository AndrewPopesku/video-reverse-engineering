"""Speech-to-text transcription using faster-whisper."""

from pathlib import Path

from faster_whisper import WhisperModel

from .models import Transcript, TranscriptSegment, TranscriptWord


def transcribe(
    audio_path: Path,
    model_size: str = "small",
) -> Transcript:
    """Transcribe audio to text with word-level timestamps.

    Args:
        audio_path: Path to audio file (WAV).
        model_size: Whisper model size (tiny, base, small, medium, large-v3).

    Returns:
        Transcript with segments and word-level timestamps.
    """
    model = WhisperModel(model_size)

    segments_iter, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
    )

    segments = []
    for seg in segments_iter:
        words = [
            TranscriptWord(
                start=w.start,
                end=w.end,
                word=w.word.strip(),
                confidence=w.probability,
            )
            for w in (seg.words or [])
        ]

        segments.append(
            TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                words=words,
                confidence=_logprob_to_confidence(seg.avg_logprob),
            )
        )

    return Transcript(
        language=info.language,
        segments=segments,
    )


def _logprob_to_confidence(logprob: float) -> float:
    """Convert average log probability to a 0-1 confidence score."""
    import math

    return math.exp(logprob)
