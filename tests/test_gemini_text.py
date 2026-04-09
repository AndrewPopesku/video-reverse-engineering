"""Tests for Gemini text analysis (Phase 3)."""

import json
from unittest.mock import MagicMock

import pytest

from reverse_engine.gemini.text import analyze_transcript
from reverse_engine.models import (
    EmotionSegment,
    Keyword,
    NamedEntity,
    Topic,
    Transcript,
    TranscriptSegment,
)


@pytest.fixture
def sample_transcript():
    return Transcript(
        language="en",
        segments=[
            TranscriptSegment(start=0.0, end=5.0, text="Tim is working on a BMW engine."),
            TranscriptSegment(start=5.0, end=10.0, text="The car was found in a barn in Germany."),
        ],
    )


@pytest.fixture
def sample_gemini_response():
    return json.dumps({
        "entities": [
            {"text": "Tim", "category": "PERSON", "timestamps": [0.0, 5.0]},
            {"text": "BMW", "category": "BRAND", "timestamps": [0.0, 5.0]},
            {"text": "Germany", "category": "LOCATION", "timestamps": [5.0, 10.0]},
        ],
        "keywords": [
            {"text": "engine", "relevance": 0.95, "count": 3},
            {"text": "barn", "relevance": 0.7, "count": 1},
        ],
        "topics": [
            {
                "name": "Automotive Restoration",
                "description": "Restoring vintage cars",
                "confidence": 0.92,
                "related_keywords": ["engine", "car", "restoration"],
            }
        ],
        "emotions": [
            {"start": 0.0, "end": 5.0, "emotion": "neutral", "confidence": 0.8},
            {"start": 5.0, "end": 10.0, "emotion": "surprise", "confidence": 0.7},
        ],
        "summary": "A mechanic named Tim works on restoring a BMW engine found in a German barn.",
    })


class TestAnalyzeTranscript:
    def test_returns_structured_insights(self, sample_transcript, sample_gemini_response):
        mock_client = MagicMock()
        mock_client.analyze_text.return_value = sample_gemini_response

        result = analyze_transcript(mock_client, sample_transcript)

        assert "entities" in result
        assert "keywords" in result
        assert "topics" in result
        assert "emotions" in result
        assert "summary" in result

    def test_extracts_entities(self, sample_transcript, sample_gemini_response):
        mock_client = MagicMock()
        mock_client.analyze_text.return_value = sample_gemini_response

        result = analyze_transcript(mock_client, sample_transcript)

        entities = result["entities"]
        assert len(entities) == 3
        assert isinstance(entities[0], NamedEntity)
        names = {e.text for e in entities}
        assert "Tim" in names
        assert "BMW" in names
        assert "Germany" in names

    def test_extracts_keywords(self, sample_transcript, sample_gemini_response):
        mock_client = MagicMock()
        mock_client.analyze_text.return_value = sample_gemini_response

        result = analyze_transcript(mock_client, sample_transcript)

        keywords = result["keywords"]
        assert len(keywords) == 2
        assert isinstance(keywords[0], Keyword)
        assert keywords[0].text == "engine"

    def test_extracts_topics(self, sample_transcript, sample_gemini_response):
        mock_client = MagicMock()
        mock_client.analyze_text.return_value = sample_gemini_response

        result = analyze_transcript(mock_client, sample_transcript)

        topics = result["topics"]
        assert len(topics) == 1
        assert topics[0].name == "Automotive Restoration"

    def test_extracts_emotions(self, sample_transcript, sample_gemini_response):
        mock_client = MagicMock()
        mock_client.analyze_text.return_value = sample_gemini_response

        result = analyze_transcript(mock_client, sample_transcript)

        emotions = result["emotions"]
        assert len(emotions) == 2
        assert emotions[0].emotion == "neutral"
        assert emotions[1].emotion == "surprise"

    def test_extracts_summary(self, sample_transcript, sample_gemini_response):
        mock_client = MagicMock()
        mock_client.analyze_text.return_value = sample_gemini_response

        result = analyze_transcript(mock_client, sample_transcript)
        assert "BMW" in result["summary"]

    def test_empty_transcript(self):
        mock_client = MagicMock()
        empty = Transcript(language="en", segments=[])

        result = analyze_transcript(mock_client, empty)
        assert result["entities"] == []
        assert result["summary"] == ""
