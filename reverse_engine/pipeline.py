"""Main orchestration pipeline for video indexing."""

from pathlib import Path

from .download import download_video
from .gemini.audio import analyze_audio
from .gemini.client import GeminiClient
from .gemini.text import analyze_transcript
from .gemini.visual import analyze_keyframes
from .keyframes import extract_keyframes
from .models import Scene, VideoInsights, VideoMeta
from .report import export_json, export_srt
from .scenes import detect_scenes
from .stems import extract_audio, separate_stems
from .timeline import build_timeline, export_fcp_xml
from .transcript import transcribe


DEFAULT_INSIGHTS = frozenset({"transcript", "visual", "text", "audio"})


def reverse_engineer(
    source: str,
    output_dir: str = "./output",
    scene_threshold: float = 27.0,
    insights: set[str] | None = None,
    gemini_api_key: str | None = None,
    gemini_model: str = "gemini-2.5-flash",
    whisper_model: str = "small",
) -> VideoInsights:
    """Run the full video indexing pipeline.

    Args:
        source: A YouTube URL or local file path.
        output_dir: Directory for all outputs.
        scene_threshold: Sensitivity for scene cut detection (lower = more cuts).
        insights: Set of insight modules to run. Options: "transcript", "visual",
                  "text", "audio". Defaults to all. Pass empty set to skip AI analysis.
        gemini_api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
        gemini_model: Gemini model to use.
        whisper_model: Whisper model size for transcription.

    Returns:
        VideoInsights with all extracted insights.
    """
    enabled = insights if insights is not None else DEFAULT_INSIGHTS
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Get the video file
    source_path = Path(source)
    if source_path.exists():
        video_path = source_path
    else:
        print("[1] Downloading video...")
        video_path = download_video(source, out)

    # Step 2: Detect scenes
    print("[2] Detecting scenes...")
    scene_tuples, fps = detect_scenes(video_path, threshold=scene_threshold)
    print(f"    Found {len(scene_tuples)} scenes")

    # Step 3: Extract keyframes
    print("[3] Extracting keyframes...")
    keyframe_dir = out / "keyframes"
    keyframe_paths = extract_keyframes(video_path, scene_tuples, keyframe_dir)

    scenes = [
        Scene(
            index=i,
            start=start,
            end=end,
            keyframe_path=keyframe_paths[i] if i < len(keyframe_paths) else None,
        )
        for i, (start, end) in enumerate(scene_tuples)
    ]

    # Step 4: Extract audio
    print("[4] Extracting audio...")
    audio_path = extract_audio(video_path, out)

    # Step 5: Separate stems
    print("[5] Separating audio stems...")
    stems = separate_stems(audio_path, out)

    # Step 6: Build video metadata
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    total_duration = scene_tuples[-1][1] if scene_tuples else 0.0
    video_meta = VideoMeta(
        path=video_path,
        duration=total_duration,
        fps=fps,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()

    result = VideoInsights(video=video_meta, scenes=scenes)

    # Step 7: Transcribe (local, no Gemini needed)
    if "transcript" in enabled:
        print("[6] Transcribing audio...")
        result.transcript = transcribe(stems["vocals"], model_size=whisper_model)

    # Steps 8-10: Gemini analysis (only if any Gemini insight is enabled)
    gemini_insights = {"visual", "text", "audio"} & enabled
    if gemini_insights:
        print("[7] Running Gemini analysis...")
        client = GeminiClient(api_key=gemini_api_key, model=gemini_model)

        if "visual" in enabled:
            print("    Analyzing keyframes...")
            visual = analyze_keyframes(client, keyframe_paths, scene_tuples)
            result.labels = visual["labels"]
            result.ocr = visual["ocr"]
            result.brands = visual["brands"]
            result.faces = visual["faces"]

        if "text" in enabled and result.transcript:
            print("    Analyzing transcript...")
            text_insights = analyze_transcript(client, result.transcript)
            result.entities = text_insights["entities"]
            result.keywords = text_insights["keywords"]
            result.topics = text_insights["topics"]
            result.emotions = text_insights["emotions"]
            result.summary = text_insights["summary"]

        if "audio" in enabled:
            print("    Analyzing audio events...")
            result.audio_events = analyze_audio(client, stems["no_vocals"])

    # Step 11: Build and export timeline
    print("[8] Building timeline...")
    timeline = build_timeline(video_path, scene_tuples, stems, fps, total_duration)
    export_fcp_xml(timeline, out / "timeline.xml")

    # Step 12: Export reports
    print("[9] Exporting reports...")
    export_json(result, out / "insights.json")
    if result.transcript:
        export_srt(result.transcript, out / "subtitles.srt")

    print("Done!")
    return result
