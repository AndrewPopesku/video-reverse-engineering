from pathlib import Path

import yt_dlp


def download_video(url: str, output_dir: Path) -> Path:
    """Download a video from a URL and return the path to the MP4 file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(output_dir / "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # yt-dlp may change extension after merge
        video_path = Path(filename).with_suffix(".mp4")

    if not video_path.exists():
        raise FileNotFoundError(f"Downloaded video not found at {video_path}")

    return video_path
