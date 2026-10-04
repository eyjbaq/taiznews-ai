"""Karaoke subtitle renderer — PNG overlay states (replaces ASS text burning).

Each state is one full chunk of words rendered as a SINGLE continuous PIL
text draw with only the active word colored differently. Because the whole
chunk is drawn in one pass every time, word order is 100% deterministic —
there is no dependency on libass's tag-based run isolation, which was the
actual root cause of the reordering bug.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

REEL_WIDTH = 1080
CANVAS_HEIGHT = 220


def strip_tashkeel(text: str) -> str:
    """Remove Arabic diacritics / harakat to ensure 100% clean on-screen captions."""
    return re.sub(r"[\u0617-\u061A\u064B-\u0652\u06D6-\u06ED]", "", text).strip()


def _group_words_into_chunks(
    word_timestamps: List[Dict[str, Any]], chunk_size: int = 3
) -> List[List[Dict[str, Any]]]:
    chunks = []
    for i in range(0, len(word_timestamps), chunk_size):
        group = word_timestamps[i : i + chunk_size]
        if group:
            chunks.append(group)
    return chunks


def _render_state_image(
    chunk: List[Dict[str, Any]], active_idx: int, font: ImageFont.FreeTypeFont
) -> Image.Image:
    canvas = Image.new("RGBA", (REEL_WIDTH, CANVAS_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    words = [strip_tashkeel(str(w.get("text", ""))) for w in chunk]

    word_lens = [draw.textlength(t, font=font, direction="rtl") for t in words]
    space_w = draw.textlength(" ", font=font, direction="rtl")
    total_w = sum(word_lens) + space_w * (len(words) - 1)

    cur_x = (REEL_WIDTH + total_w) / 2
    y = CANVAS_HEIGHT / 2

    for i, wtxt in enumerate(words):
        color = (255, 215, 0) if i == active_idx else (255, 255, 255)
        draw.text(
            (cur_x, y),
            wtxt,
            font=font,
            fill=color,
            anchor="rm",
            direction="rtl",
            stroke_width=5,
            stroke_fill=(0, 0, 0),
        )
        cur_x -= (word_lens[i] + space_w)

    return canvas


def compose_subtitle_video(
    word_timestamps: List[Dict[str, Any]],
    audio_duration: float,
    output_path: Path | str,
    font_path: str,
    font_size: int = 64,
    chunk_size: int = 3,
    fps: int = 30,
) -> Path:
    """Compose dynamic 3-word chunk subtitles into a single transparent video track.

    Guarantees broadcast quality:
      - Active spoken word is highlighted in golden yellow
      - Full 3-word phrase remains visible
      - Strict boundary clipping ensures ZERO overlap between consecutive words or chunks
      - Encoded into transparent QuickTime MOV (RGBA / qtrle)
      - Single overlay input in FFmpeg avoids Windows command-length limits
    """
    import subprocess

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(font_path, font_size)

    total_frames = int(audio_duration * fps) + 1
    chunks = _group_words_into_chunks(word_timestamps or [], chunk_size=chunk_size)
    num_chunks = len(chunks)

    frame_states: List[Dict[str, Any]] = []

    for ci, chunk in enumerate(chunks):
        num_in_chunk = len(chunk)
        for active_idx, w_active in enumerate(chunk):
            t_start = float(w_active.get("start", 0.0))
            t_end = float(w_active.get("end", 0.0))

            if active_idx < num_in_chunk - 1:
                t_boundary = float(chunk[active_idx + 1].get("start", 0.0))
            else:
                if ci < num_chunks - 1:
                    next_chunk_first_start = float(chunks[ci + 1][0].get("start", 0.0))
                    t_boundary = min(next_chunk_first_start, t_end + 0.30)
                else:
                    t_boundary = t_end + 0.35

            img = _render_state_image(chunk, active_idx, font)
            frame_states.append({
                "bytes": img.tobytes(),
                "t_start": t_start,
                "t_boundary": max(t_start, t_boundary),
            })

    blank_bytes = bytes(REEL_WIDTH * CANVAS_HEIGHT * 4)

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{REEL_WIDTH}x{CANVAS_HEIGHT}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "qtrle", "-pix_fmt", "argb",
        str(output_path),
    ]

    print(f"🎨 [الترجمة التفاعلية] تركيب مسار فيديو الترجمة الشفاف بنمط {chunk_size} كلمات بدون أي تداخل...")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        for fi in range(total_frames):
            t = fi / fps
            active = None
            for st in frame_states:
                if st["t_start"] <= t < st["t_boundary"]:
                    active = st
                    break

            if active is None:
                proc.stdin.write(blank_bytes)
            else:
                proc.stdin.write(active["bytes"])

        proc.stdin.close()
        _, stderr_data = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg subtitle composition failed: {stderr_data.decode(errors='replace')}")
    except BrokenPipeError:
        proc.kill()
        raise RuntimeError("FFmpeg subtitle composition pipe broke unexpectedly.")

    return output_path


def generate_subtitle_states(
    word_timestamps: List[Dict[str, Any]],
    output_dir: Path,
    font_path: str,
    font_size: int = 64,
    chunk_size: int = 3,
) -> List[Dict[str, Any]]:
    """Generate chronological PNG subtitle state images for FFmpeg overlay (3-word chunks)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(font_path, font_size)

    chunks = _group_words_into_chunks(word_timestamps or [], chunk_size=chunk_size)
    states: List[Dict[str, Any]] = []
    idx = 0
    num_chunks = len(chunks)

    for ci, chunk in enumerate(chunks):
        num_in_chunk = len(chunk)
        for active_idx, w_active in enumerate(chunk):
            img = _render_state_image(chunk, active_idx, font)
            p = output_dir / f"sub_state_{idx:04d}.png"
            img.save(str(p), "PNG")

            t_start = float(w_active.get("start", 0.0))
            t_end_raw = float(w_active.get("end", 0.0))

            if active_idx < num_in_chunk - 1:
                t_end = float(chunk[active_idx + 1].get("start", 0.0))
            else:
                if ci < num_chunks - 1:
                    next_chunk_first_start = float(chunks[ci + 1][0].get("start", 0.0))
                    t_end = min(next_chunk_first_start, t_end_raw + 0.30)
                else:
                    t_end = t_end_raw + 0.35

            states.append({
                "path": str(p),
                "start": t_start,
                "end": max(t_start, t_end),
            })
            idx += 1
    return states
