import subprocess
import sys
from pathlib import Path


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """Extract audio from a video file as WAV using ffmpeg."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = output_dir / "audio.wav"

    subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            "-y", str(audio_path),
        ],
        check=True,
        capture_output=True,
    )

    return audio_path


def separate_stems(audio_path: Path, output_dir: Path) -> dict[str, Path]:
    """Separate audio into vocal and non-vocal stems using Demucs.

    Returns:
        Dict with keys 'vocals' and 'no_vocals' mapping to WAV file paths.
    """
    output_dir = Path(output_dir)

    subprocess.run(
        [
            sys.executable, "-m", "demucs",
            "--two-stems=vocals",
            "-o", str(output_dir),
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )

    # Demucs outputs to: output_dir/htdemucs/<stem_name>/{vocals,no_vocals}.wav
    stem_name = audio_path.stem
    demucs_dir = output_dir / "htdemucs" / stem_name

    vocals = demucs_dir / "vocals.wav"
    no_vocals = demucs_dir / "no_vocals.wav"

    if not vocals.exists() or not no_vocals.exists():
        raise FileNotFoundError(
            f"Demucs output not found in {demucs_dir}. "
            f"Contents: {list(demucs_dir.iterdir()) if demucs_dir.exists() else 'dir missing'}"
        )

    return {"vocals": vocals, "no_vocals": no_vocals}
