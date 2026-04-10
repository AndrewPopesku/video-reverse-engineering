"""Evaluate a human-written description against extracted VideoInsights."""

from typing import Literal

from pydantic import BaseModel, ValidationError

from .gemini.client import GeminiClient, parse_json_response
from .models import VideoInsights

ClaimStatus = Literal["matched", "partial", "unmatched", "unverifiable"]
Verdict = Literal["match", "partial", "mismatch"]

_TRANSCRIPT_CHAR_CAP = 6000
_TOP_N = 15


class MatchClaim(BaseModel):
    claim: str
    status: ClaimStatus
    evidence: str


class MatchResult(BaseModel):
    score: float
    verdict: Verdict
    summary: str
    claims: list[MatchClaim] = []
    missing_from_video: list[str] = []
    extra_in_video: list[str] = []


_PROMPT = """You are an impartial video-vs-description judge.

INPUTS:
- DESCRIPTION: the claim a human wrote about a video.
- EVIDENCE: structured facts extracted from the actual video (summary, scene
  captions, transcript, brands, topics, audio events). Treat EVIDENCE as ground
  truth. Do not invent facts beyond it.

TASK:
1. Decompose DESCRIPTION into atomic claims (one testable statement each).
2. For each claim, set status:
   - "matched"       — directly supported by EVIDENCE.
   - "partial"       — loosely supported or only generically true.
   - "unmatched"     — contradicted or absent.
   - "unverifiable"  — out of scope for the available evidence.
   Cite the smallest evidence snippet that justifies the judgment.
3. Compute score = (matched + 0.5*partial) / (matched + partial + unmatched).
   Exclude unverifiable from the denominator. If denominator is 0, score = 0.
4. verdict: "match" if score >= 0.8, "partial" if >= 0.5, else "mismatch".
5. missing_from_video: description items not in EVIDENCE.
6. extra_in_video: salient EVIDENCE items the description omits.

Return ONLY JSON (no markdown fences, no commentary) matching exactly:
{
  "score": float,
  "verdict": "match" | "partial" | "mismatch",
  "summary": string,
  "claims": [{"claim": string, "status": "matched"|"partial"|"unmatched"|"unverifiable", "evidence": string}],
  "missing_from_video": [string],
  "extra_in_video": [string]
}
"""


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[:cap].rstrip() + " …[truncated]"


def _build_evidence_brief(insights: VideoInsights) -> str:
    """Render VideoInsights as a compact, token-efficient evidence brief."""
    lines: list[str] = []

    if insights.summary:
        lines.append("## Pipeline summary")
        lines.append(insights.summary.strip())
        lines.append("")

    if insights.labels:
        lines.append("## Scene labels")
        for label in insights.labels:
            objs = ", ".join(label.objects[:8]) if label.objects else ""
            shot = f" [{label.shot_type}]" if label.shot_type else ""
            mood = f" (mood: {label.mood})" if label.mood else ""
            obj_part = f" — objects: {objs}" if objs else ""
            lines.append(
                f"- scene {label.scene_index}{shot}{mood}: {label.caption}{obj_part}"
            )
        lines.append("")

    if insights.transcript and insights.transcript.full_text.strip():
        lines.append("## Transcript")
        lines.append(_truncate(insights.transcript.full_text.strip(), _TRANSCRIPT_CHAR_CAP))
        lines.append("")

    if insights.brands:
        lines.append("## Brands")
        for b in insights.brands[:_TOP_N]:
            lines.append(f"- {b.name} ({b.source})")
        lines.append("")

    if insights.topics:
        lines.append("## Topics")
        for t in insights.topics[:_TOP_N]:
            desc = f" — {t.description}" if t.description else ""
            lines.append(f"- {t.name}{desc}")
        lines.append("")

    if insights.keywords:
        lines.append("## Keywords")
        kws = [k.text for k in insights.keywords[:_TOP_N]]
        lines.append(", ".join(kws))
        lines.append("")

    if insights.entities:
        lines.append("## Named entities")
        for e in insights.entities[:_TOP_N]:
            lines.append(f"- {e.category}: {e.text}")
        lines.append("")

    if insights.audio_events:
        lines.append("## Audio events")
        for a in insights.audio_events[:_TOP_N]:
            lines.append(f"- {a.label} ({a.start:.1f}s–{a.end:.1f}s)")
        lines.append("")

    if insights.ocr:
        lines.append("## On-screen text")
        for o in insights.ocr[:_TOP_N]:
            snippet = o.text.strip().replace("\n", " ")
            if snippet:
                lines.append(f"- scene {o.scene_index}: {snippet}")
        lines.append("")

    return "\n".join(lines).strip()


def evaluate_description(
    insights: VideoInsights,
    description: str,
    client: GeminiClient | None = None,
) -> MatchResult:
    """Judge how well a human description matches extracted video insights."""
    description = description.strip()
    if not description:
        raise ValueError("description must not be empty")

    evidence = _build_evidence_brief(insights)
    if not evidence:
        raise ValueError("insights produced no evidence to evaluate against")

    judge = client or GeminiClient()

    payload = (
        f"DESCRIPTION:\n{description}\n\n"
        f"EVIDENCE:\n{evidence}\n"
    )

    raw = judge.analyze_text(payload, _PROMPT)

    try:
        parsed = parse_json_response(raw)
    except Exception as exc:
        head = (raw or "")[:500]
        raise ValueError(
            f"Failed to parse judge response as JSON: {exc}\n---\n{head}"
        ) from exc

    if not isinstance(parsed, dict):
        head = (raw or "")[:500]
        raise ValueError(f"Expected JSON object from judge, got {type(parsed).__name__}\n---\n{head}")

    try:
        return MatchResult.model_validate(parsed)
    except ValidationError as exc:
        head = (raw or "")[:500]
        raise ValueError(f"Judge response did not match MatchResult schema: {exc}\n---\n{head}") from exc
