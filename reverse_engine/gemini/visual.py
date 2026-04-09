"""Gemini-powered visual analysis of keyframes."""

from pathlib import Path

from ..models import BrandMention, FaceGroup, OcrResult, SceneLabel, TimeRange
from .client import GeminiClient, parse_json_response

VISUAL_PROMPT = """\
Analyze each keyframe image. For each one, return a JSON object with:
- scene_index: the index (0-based, matching image order)
- caption: one-sentence description of what's happening
- objects: list of visible objects and actions
- ocr_text: any on-screen text (titles, signs, lower thirds, watermarks). Empty string if none.
- brands: list of visible brand logos or product names
- people: list of people visible, each with "description" (clothing, appearance) and "appears_in_scenes" (list of scene indices where this same person appears)
- shot_type: one of "close-up", "medium", "wide", "extreme-wide"
- mood: one-word mood/atmosphere

Return valid JSON with a top-level "scenes" array. No markdown fences.
"""


def analyze_keyframes(
    client: GeminiClient,
    keyframe_paths: list[Path],
    scenes: list[tuple[float, float]],
) -> dict:
    """Analyze keyframes using Gemini vision.

    Returns:
        Dict with keys: labels, ocr, brands, faces.
    """
    empty_result = {"labels": [], "ocr": [], "brands": [], "faces": []}

    if not keyframe_paths:
        return empty_result

    raw = client.analyze_images(images=keyframe_paths, prompt=VISUAL_PROMPT)
    data = parse_json_response(raw)

    labels: list[SceneLabel] = []
    ocr_results: list[OcrResult] = []
    all_brands: dict[str, list[TimeRange]] = {}
    people_map: dict[str, dict] = {}

    for scene_data in data.get("scenes", []):
        idx = scene_data["scene_index"]

        labels.append(
            SceneLabel(
                scene_index=idx,
                caption=scene_data.get("caption", ""),
                objects=scene_data.get("objects", []),
                shot_type=scene_data.get("shot_type", ""),
                mood=scene_data.get("mood", ""),
            )
        )

        ocr_text = scene_data.get("ocr_text", "")
        if ocr_text:
            ocr_results.append(
                OcrResult(
                    text=ocr_text,
                    scene_index=idx,
                    confidence=0.9,
                )
            )

        for brand_name in scene_data.get("brands", []):
            if idx < len(scenes):
                start, end = scenes[idx]
                time_range = TimeRange(start=start, end=end)
                all_brands.setdefault(brand_name, []).append(time_range)

        for person in scene_data.get("people", []):
            desc = person.get("description", "")
            if desc not in people_map:
                people_map[desc] = {
                    "description": desc,
                    "scene_indices": set(),
                }
            for si in person.get("appears_in_scenes", []):
                people_map[desc]["scene_indices"].add(si)

    brands = [
        BrandMention(name=name, source="visual", instances=instances)
        for name, instances in all_brands.items()
    ]

    faces = []
    for i, (desc, info) in enumerate(people_map.items()):
        appearances = []
        for si in sorted(info["scene_indices"]):
            if si < len(scenes):
                start, end = scenes[si]
                appearances.append(TimeRange(start=start, end=end))
        faces.append(
            FaceGroup(
                id=f"person_{i + 1}",
                description=info["description"],
                appearances=appearances,
            )
        )

    return {
        "labels": labels,
        "ocr": ocr_results,
        "brands": brands,
        "faces": faces,
    }
