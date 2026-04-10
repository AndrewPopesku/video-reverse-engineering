"""CLI: evaluate how well a human-written description matches a video.

Usage:
    python examples/evaluate_video.py --description "A short promo for ..."
    python examples/evaluate_video.py --description-file desc.txt
    python examples/evaluate_video.py --insights ./output/insights.json --description-file desc.txt
"""

import argparse
import json
import sys
from pathlib import Path

from reverse_engine import MatchResult, evaluate_description
from reverse_engine.gemini.client import GeminiClient
from reverse_engine.models import VideoInsights


_STATUS_ICON = {
    "matched": "[+]",
    "partial": "[~]",
    "unmatched": "[-]",
    "unverifiable": "[?]",
}

_EXIT_CODE = {
    "match": 0,
    "partial": 1,
    "mismatch": 2,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--insights",
        default="./output/insights.json",
        help="Path to insights.json produced by the pipeline.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--description", help="Description string.")
    group.add_argument(
        "--description-file",
        help="Path to a file containing the description.",
    )
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw MatchResult JSON to stdout instead of a report.",
    )
    return parser.parse_args()


def _load_insights(path: Path) -> VideoInsights:
    if not path.exists():
        raise SystemExit(f"insights file not found: {path}")
    return VideoInsights.model_validate_json(path.read_text())


def _load_description(args: argparse.Namespace) -> str:
    if args.description:
        return args.description
    path = Path(args.description_file)
    if not path.exists():
        raise SystemExit(f"description file not found: {path}")
    return path.read_text()


def _print_report(result: MatchResult) -> None:
    verdict_label = result.verdict.upper()
    print(f"Verdict: {verdict_label} (score: {result.score:.2f})")
    if result.summary:
        print(result.summary)
    print()

    if result.claims:
        print("Claims:")
        for c in result.claims:
            icon = _STATUS_ICON.get(c.status, "[?]")
            evidence = c.evidence.strip().replace("\n", " ")
            if len(evidence) > 160:
                evidence = evidence[:157] + "..."
            print(f"  {icon} {c.claim}")
            if evidence:
                print(f"      evidence: {evidence}")
        print()

    if result.missing_from_video:
        print("Missing from video:")
        for item in result.missing_from_video:
            print(f"  - {item}")
        print()

    if result.extra_in_video:
        print("Extra in video (not in description):")
        for item in result.extra_in_video:
            print(f"  - {item}")
        print()


def main() -> int:
    args = _parse_args()
    insights = _load_insights(Path(args.insights))
    description = _load_description(args)

    client = GeminiClient(model=args.gemini_model)
    result = evaluate_description(insights, description, client=client)

    if args.json:
        json.dump(result.model_dump(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_report(result)

    return _EXIT_CODE.get(result.verdict, 2)


if __name__ == "__main__":
    raise SystemExit(main())
