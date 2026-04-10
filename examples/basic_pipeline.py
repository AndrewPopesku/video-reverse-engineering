"""Basic pipeline example: extract insights from a video file.

Usage:
    python examples/basic_pipeline.py <path-to-video>
"""

import sys

from reverse_engine import reverse_engineer


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "./video.mp4"

    result = reverse_engineer(
        video_path,
        output_dir="./output",
        insights={"transcript", "visual", "text", "audio"},
        whisper_model="small",
        gemini_model="gemini-2.5-flash",
    )

    print(f"\nVideo: {result.video.duration}s, {len(result.scenes)} scenes")

    if result.transcript:
        print(f"Transcript: {len(result.transcript.segments)} segments ({result.transcript.language})")

    if result.labels:
        print(f"Scene labels: {len(result.labels)}")
        for label in result.labels[:3]:
            print(f"  [{label.scene_index}] {label.caption}")

    if result.entities:
        print(f"Named entities: {len(result.entities)}")
        for e in result.entities[:5]:
            print(f"  {e.category}: {e.text}")

    if result.brands:
        print(f"Brands: {[b.name for b in result.brands]}")

    if result.topics:
        print(f"Topics: {[t.name for t in result.topics]}")

    if result.keywords:
        print(f"Keywords: {[k.text for k in result.keywords[:10]]}")

    if result.summary:
        print(f"\nSummary: {result.summary}")


if __name__ == "__main__":
    main()
