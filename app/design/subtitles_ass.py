"""Karaoke subtitle renderer — PNG overlay states (replaces ASS text burning).

Each state is one full chunk of words rendered as a SINGLE continuous PIL
text draw with only the active word colored differently. Because the whole
chunk is drawn in one pass every time, word order is 100% deterministic —
there is no dependency on libass's tag-based run isolation, which was the
actual root cause of the reordering bug.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

REEL_WIDTH = 1080
CANVAS_HEIGHT = 220


def _group_words_into_chunks(
    word_timestamps: List[Dict[str, Any]], chunk_size: int = 4
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

    words = [str(w.get("text", "")).strip() for w in chunk]
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


def generate_subtitle_states(
    word_timestamps: List[Dict[str, Any]],
    output_dir: Path,
    font_path: str,
    font_size: int = 64,
    chunk_size: int = 4,
) -> List[Dict[str, Any]]:
    """Generate chronological PNG subtitle state images for FFmpeg overlay.

    Returns:
        List of dicts: [{"path": str, "start": float, "end": float}, ...]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(font_path, font_size)

    chunks = _group_words_into_chunks(word_timestamps or [], chunk_size=chunk_size)
    states: List[Dict[str, Any]] = []
    idx = 0
    for chunk in chunks:
        num_in_chunk = len(chunk)
        for active_idx, w_active in enumerate(chunk):
            img = _render_state_image(chunk, active_idx, font)
            p = output_dir / f"sub_state_{idx:04d}.png"
            img.save(str(p), "PNG")

            # Extend active word duration until next word begins to eliminate gaps/flickering
            if active_idx < num_in_chunk - 1:
                t_end = max(float(w_active.get("end", 0.0)), float(chunk[active_idx + 1].get("start", 0.0)))
            else:
                t_end = float(w_active.get("end", 0.0)) + 0.20

            states.append({
                "path": str(p),
                "start": float(w_active.get("start", 0.0)),
                "end": t_end,
            })
            idx += 1
    return states
