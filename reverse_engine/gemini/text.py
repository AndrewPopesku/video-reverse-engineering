"""Gemini-powered text analysis of transcripts."""

from ..models import (
    EmotionSegment,
    Keyword,
    NamedEntity,
    TimeRange,
    Topic,
    Transcript,
)
from .client import GeminiClient, parse_json_response

TEXT_PROMPT = """\
Analyze the following video transcript and return a JSON object with:

- entities: list of named entities, each with "text", "category" (PERSON, BRAND, LOCATION, PRODUCT, EVENT), and "timestamps" [start, end] of first mention
- keywords: list of important keywords, each with "text", "relevance" (0-1), and "count"
- topics: list of inferred topics, each with "name", "description", "confidence" (0-1), and "related_keywords" list
- emotions: list of emotional segments, each with "start", "end", "emotion" (joy, anger, sadness, fear, surprise, neutral), and "confidence" (0-1)
- summary: 2-3 sentence summary of the video content

Return valid JSON only. No markdown fences.
"""


def analyze_transcript(
    client: GeminiClient,
    transcript: Transcript,
) -> dict:
    """Analyze transcript text using Gemini.

    Returns:
        Dict with keys: entities, keywords, topics, emotions, summary.
    """
    empty_result = {
        "entities": [],
        "keywords": [],
        "topics": [],
        "emotions": [],
        "summary": "",
    }

    if not transcript.segments:
        return empty_result

    raw = client.analyze_text(text=transcript.full_text, prompt=TEXT_PROMPT)
    data = parse_json_response(raw)

    entities = [
        NamedEntity(
            text=e["text"],
            category=e["category"],
            instances=(
                [TimeRange(start=e["timestamps"][0], end=e["timestamps"][1])]
                if e.get("timestamps") and len(e["timestamps"]) == 2
                else []
            ),
        )
        for e in data.get("entities", [])
    ]

    keywords = [
        Keyword(
            text=k["text"],
            relevance=k.get("relevance", 0.0),
            count=k.get("count", 0),
        )
        for k in data.get("keywords", [])
    ]

    topics = [
        Topic(
            name=t["name"],
            description=t.get("description", ""),
            confidence=t.get("confidence", 0.0),
            related_keywords=t.get("related_keywords", []),
        )
        for t in data.get("topics", [])
    ]

    emotions = [
        EmotionSegment(
            start=em["start"],
            end=em["end"],
            emotion=em["emotion"],
            confidence=em.get("confidence", 0.0),
        )
        for em in data.get("emotions", [])
    ]

    summary = data.get("summary", "")

    return {
        "entities": entities,
        "keywords": keywords,
        "topics": topics,
        "emotions": emotions,
        "summary": summary,
    }
