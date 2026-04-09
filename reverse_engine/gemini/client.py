"""Gemini API client wrapper for video insight extraction."""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai


def parse_json_response(raw: str) -> dict | list:
    """Parse JSON from Gemini response, stripping markdown fences if present."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if not text:
        return {}
    return json.loads(text)

load_dotenv()


class GeminiClient:
    """Thin wrapper around the Gemini API for multimodal analysis."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
    ):
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set and no api_key provided"
            )

        self._client = genai.Client(api_key=resolved_key)
        self.model = model

    def analyze_images(
        self,
        images: list[Path],
        prompt: str,
    ) -> str:
        """Send images to Gemini for visual analysis.

        Args:
            images: List of image file paths.
            prompt: The analysis prompt.

        Returns:
            Raw response text from Gemini.
        """
        parts: list = []
        for img_path in images:
            parts.append(
                genai.types.Part.from_bytes(
                    data=img_path.read_bytes(),
                    mime_type="image/jpeg",
                )
            )
        parts.append(prompt)

        response = self._client.models.generate_content(
            model=self.model,
            contents=parts,
        )
        return response.text

    def analyze_text(
        self,
        text: str,
        prompt: str,
    ) -> str:
        """Send text to Gemini for analysis.

        Args:
            text: The text content to analyze.
            prompt: The analysis prompt.

        Returns:
            Raw response text from Gemini.
        """
        content = f"{prompt}\n\n---\n\n{text}"

        response = self._client.models.generate_content(
            model=self.model,
            contents=content,
        )
        return response.text

    def analyze_audio(
        self,
        audio_path: Path,
        prompt: str,
    ) -> str:
        """Send audio to Gemini for analysis.

        Args:
            audio_path: Path to audio file.
            prompt: The analysis prompt.

        Returns:
            Raw response text from Gemini.
        """
        audio_part = genai.types.Part.from_bytes(
            data=audio_path.read_bytes(),
            mime_type="audio/wav",
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=[audio_part, prompt],
        )
        return response.text
