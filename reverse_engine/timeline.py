from pathlib import Path

import opentimelineio as otio


def build_timeline(
    video_path: Path,
    scenes: list[tuple[float, float]],
    stems: dict[str, Path],
    fps: float,
    total_duration: float | None = None,
) -> otio.schema.Timeline:
    """Build a multi-track OTIO timeline from scene cuts and audio stems.

    Args:
        video_path: Path to the source video file.
        scenes: List of (start_seconds, end_seconds) tuples.
        stems: Dict mapping stem name to WAV path (e.g. {"vocals": ..., "no_vocals": ...}).
        fps: Video framerate.
        total_duration: Total video duration in seconds (used for audio available_range).
    """
    timeline = otio.schema.Timeline(name=Path(video_path).stem)

    if total_duration is None and scenes:
        total_duration = scenes[-1][1]

    # -- Video track with scene cuts --
    video_track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)

    video_ref = otio.schema.ExternalReference(
        target_url=str(Path(video_path).resolve()),
        available_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, fps),
            duration=otio.opentime.RationalTime.from_seconds(total_duration, fps),
        ),
    )

    for i, (start, end) in enumerate(scenes):
        duration = end - start
        clip = otio.schema.Clip(
            name=f"scene_{i + 1:03d}",
            media_reference=video_ref.deepcopy(),
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime.from_seconds(start, fps),
                duration=otio.opentime.RationalTime.from_seconds(duration, fps),
            ),
        )
        video_track.append(clip)

    timeline.tracks.append(video_track)

    # -- Audio tracks for each stem --
    # Use video fps for timeline rate (FCP XML requires SMPTE-compatible rates)
    for stem_name, stem_path in stems.items():
        audio_track = otio.schema.Track(
            name=stem_name, kind=otio.schema.TrackKind.Audio
        )

        audio_ref = otio.schema.ExternalReference(
            target_url=str(Path(stem_path).resolve()),
            available_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, fps),
                duration=otio.opentime.RationalTime.from_seconds(
                    total_duration, fps
                ),
            ),
        )

        clip = otio.schema.Clip(
            name=stem_name,
            media_reference=audio_ref,
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, fps),
                duration=otio.opentime.RationalTime.from_seconds(
                    total_duration, fps
                ),
            ),
        )
        audio_track.append(clip)
        timeline.tracks.append(audio_track)

    return timeline


def export_fcp_xml(timeline: otio.schema.Timeline, output_path: Path) -> Path:
    """Export an OTIO timeline as FCP 7 XML."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    otio.adapters.write_to_file(timeline, str(output_path), adapter_name="fcp_xml")
    return output_path
