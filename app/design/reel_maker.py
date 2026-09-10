"""News Reel video composer using native FFmpeg subprocess and ASS subtitles.

Features:
  - Multi-scene storyboarding with cinematic Ken Burns motion via FFmpeg zoompan
  - Burned ASS karaoke subtitles with explicit Unicode bidi overrides (RLE/PDF)
  - Persistent urgent news badge ('🔴 عاجل | تعز نيوز') and dark contrast gradient
  - Background breaking news tension music mixed via FFmpeg aloop & amix at 12%
  - High-performance native encoding without MoviePy overhead
  - Vertical 9:16 format (1080x1920) optimized for Facebook & Instagram Reels
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

from app.design.subtitles_ass import generate_subtitle_states

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "generated_reels"
DEFAULT_AUDIO_DIR = BASE_DIR / "assets" / "audio"
DEFAULT_FONTS_DIR = BASE_DIR / "assets" / "fonts"

# Reel dimensions (vertical 9:16)
REEL_WIDTH = 1080
REEL_HEIGHT = 1920


def _get_audio_duration(audio_path: Union[str, Path]) -> float:
    """Retrieve audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


def _get_background_music() -> Optional[Path]:
    """Locate background news music track in assets/audio/ if present."""
    audio_dir = Path(DEFAULT_AUDIO_DIR)
    if not audio_dir.exists():
        return None
    for ext in ("*.wav", "*.mp3", "*.aac", "*.ogg"):
        for f in audio_dir.glob(ext):
            if f.stat().st_size > 5000:
                return f
    return None


def _get_font(size: int = 38) -> ImageFont.FreeTypeFont:
    """Load Arabic font for badge rendering."""
    fonts_dir = Path(DEFAULT_FONTS_DIR)
    for font_name in ("Cairo-Bold.ttf", "Cairo[slnt,wght].ttf", "Almarai-Bold.ttf", "Tajawal-Bold.ttf"):
        p = fonts_dir / font_name
        if p.exists() and p.stat().st_size > 10000:
            return ImageFont.truetype(str(p), size)
    for win_p in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf"):
        if Path(win_p).exists():
            return ImageFont.truetype(win_p, size)
    return ImageFont.load_default()


def _create_top_badge(width: int = REEL_WIDTH) -> Image.Image:
    """Pre-render the persistent top urgent news badge once as RGBA."""
    badge_img = Image.new("RGBA", (width, 240), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge_img)
    badge_font = _get_font(38)

    label = "عاجل | تعز نيوز"
    bbox = draw.textbbox((0, 0), label, font=badge_font, direction="rtl")
    lw = bbox[2] - bbox[0]
    lh = bbox[3] - bbox[1]

    dot_r = 7
    dot_gap = 14
    pad_x, pad_y = 35, 14
    box_w = lw + (dot_r * 2) + dot_gap + (pad_x * 2)
    box_h = lh + (pad_y * 2)
    box_x = (width - box_w) // 2
    box_y = 95

    # Rounded red pill badge
    draw.rounded_rectangle(
        [box_x, box_y, box_x + box_w, box_y + box_h],
        radius=16,
        fill=(220, 20, 35, 255),
        outline=(255, 255, 255, 255),
        width=2,
    )

    content_w = lw + dot_gap + (dot_r * 2)
    start_x = (width + content_w) // 2
    center_y = (box_y + box_y + box_h) // 2 - 2

    # Draw white indicator dot on the right
    dot_cx = start_x - dot_r
    draw.ellipse(
        [dot_cx - dot_r, center_y - dot_r, dot_cx + dot_r, center_y + dot_r],
        fill=(255, 255, 255, 255),
    )

    # Draw text to the left of the dot
    text_rx = dot_cx - dot_r - dot_gap
    draw.text(
        (text_rx, center_y),
        label,
        font=badge_font,
        fill=(255, 255, 255, 255),
        anchor="rm",
        direction="rtl",
    )
    return badge_img


def _create_bottom_gradient(width: int = REEL_WIDTH, height: int = 650) -> Image.Image:
    """Create a smooth dark gradient for subtitle contrast at the bottom."""
    grad = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(grad)
    for y in range(height):
        alpha = int(210 * (y / height) ** 1.6)
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
    return grad


def _render_scene_clip_ffmpeg(
    image_path: Union[str, Path],
    duration: float,
    output_clip_path: Path,
    scene_index: int,
    fps: int = 24,
    zoom_factor: float = 1.14,
) -> Path:
    """Render a single scene image into a video clip with Ken Burns motion using FFmpeg zoompan."""
    total_frames = max(1, int(fps * duration))
    mode = scene_index % 4

    if mode == 0:
        # Zoom in towards center
        zoom_expr = f"min({zoom_factor:.2f}, 1.0 + {zoom_factor - 1.0:.2f}*on/{total_frames})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif mode == 1:
        # Zoom out from center
        zoom_expr = f"max(1.0, {zoom_factor:.2f} - {zoom_factor - 1.0:.2f}*on/{total_frames})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif mode == 2:
        # Zoom in with subtle upward tilt
        zoom_expr = f"min({zoom_factor:.2f}, 1.0 + {zoom_factor - 1.0:.2f}*on/{total_frames})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = f"(ih/2-(ih/zoom/2)) - (0.04*ih*on/{total_frames})"
    else:
        # Zoom out with subtle downward tilt
        zoom_expr = f"max(1.0, {zoom_factor:.2f} - {zoom_factor - 1.0:.2f}*on/{total_frames})"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = f"(ih/2-(ih/zoom/2)) + (0.04*ih*on/{total_frames})"

    vf_filter = (
        f"scale={REEL_WIDTH}:{REEL_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={REEL_WIDTH}:{REEL_HEIGHT},"
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':"
        f"d=1:s={REEL_WIDTH}x{REEL_HEIGHT}:fps={fps},setsar=1"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        str(image_path),
        "-vf",
        vf_filter,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-t",
        f"{duration:.3f}",
        str(output_clip_path),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg Ken Burns failed for scene {scene_index}: {res.stderr[-400:]}")

    return output_clip_path


def compose_news_reel(
    image_paths: Union[str, List[str]],
    audio_path: str,
    output_dir: Path | str | None = None,
    word_timestamps: Optional[List[Dict[str, Any]]] = None,
    headline: str = "",
    category: str = "",
    slug: str = "reel",
    zoom_factor: float = 1.14,
    fps: int = 24,
) -> Optional[str]:
    """Compose a multi-scene news reel video with kinetic karaoke subtitles using native FFmpeg.

    Args:
        image_paths: Single image path or list of paths (4-5 scenes).
        audio_path: Path to narration audio.
        output_dir: Output folder.
        word_timestamps: Exact word timestamps from TTS for karaoke subtitles.
        headline: Optional news headline for context.
        category: News category.
        slug: Output filename slug.
        zoom_factor: Motion headroom scale.
        fps: Frames per second.

    Returns:
        Path to output .mp4 video file, or None on failure.
    """
    # 1. Validate inputs
    if isinstance(image_paths, (str, Path)):
        raw_images = [str(image_paths)]
    else:
        raw_images = [str(p) for p in image_paths if p]

    valid_images = [p for p in raw_images if Path(p).exists()]
    if not valid_images:
        LOGGER.error("No valid image files provided for reel.")
        print("❌ لم يتم العثور على أي ملفات صور صالحة لتركيب الريلز.")
        return None

    aud_file = Path(audio_path)
    if not aud_file.exists():
        LOGGER.error("Audio file not found: %s", aud_file)
        print(f"❌ لم يتم العثور على ملف الصوت: {aud_file}")
        return None

    try:
        total_duration = _get_audio_duration(aud_file)
    except Exception as dur_err:
        LOGGER.error("Could not determine audio duration: %s", dur_err)
        print(f"❌ تعذر استخراج مدة الملف الصوتي: {dur_err}")
        return None

    if total_duration < 1.0:
        LOGGER.warning("Audio duration too short (%.2fs), skipping reel composition.", total_duration)
        return None

    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    output_file = out_dir / f"{slug}_{timestamp}.mp4"
    temp_dir = out_dir / f"_temp_render_{timestamp}_{os.getpid()}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    num_scenes = len(valid_images)
    scene_duration = total_duration / float(num_scenes)

    print(f"🎬 جاري تركيب مقطع الريلز المتكامل عبر FFmpeg ({num_scenes} مشاهد + ترجمة متحركة + موسيقى خلفية)...")

    try:
        # 2. Pre-render persistent top badge & bottom gradient overlays
        badge_img = _create_top_badge(width=REEL_WIDTH)
        badge_path = temp_dir / "_badge_overlay.png"
        badge_img.save(str(badge_path), "PNG")

        gradient_img = _create_bottom_gradient(width=REEL_WIDTH, height=650)
        gradient_path = temp_dir / "_gradient_overlay.png"
        gradient_img.save(str(gradient_path), "PNG")

        # 3. Render scene clips concurrently with Ken Burns motion
        scene_clip_paths: List[Path] = []
        for idx in range(num_scenes):
            scene_clip_paths.append(temp_dir / f"_scene_clip_{idx}.mp4")

        def _render_worker(idx_and_img: Tuple[int, str]) -> Path:
            s_idx, img_p = idx_and_img
            cur_dur = (
                total_duration - (scene_duration * (num_scenes - 1))
                if s_idx == num_scenes - 1
                else scene_duration
            )
            return _render_scene_clip_ffmpeg(
                image_path=img_p,
                duration=cur_dur,
                output_clip_path=scene_clip_paths[s_idx],
                scene_index=s_idx,
                fps=fps,
                zoom_factor=zoom_factor,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_scenes, 4)) as pool:
            list(pool.map(_render_worker, enumerate(valid_images)))

        # 4. Stitch scene clips using FFmpeg concat demuxer (-c copy)
        concat_manifest = temp_dir / "concat_manifest.txt"
        with open(concat_manifest, "w", encoding="utf-8") as f:
            for c in scene_clip_paths:
                f.write(f"file '{str(c.resolve()).replace(chr(92), '/')}'\n")

        stitched_video = temp_dir / "stitched_raw.mp4"
        cmd_concat = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_manifest),
            "-c",
            "copy",
            str(stitched_video),
        ]
        res_concat = subprocess.run(cmd_concat, capture_output=True, text=True)
        if res_concat.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed: {res_concat.stderr[-400:]}")

        # 5. Generate PNG subtitle state overlays with 100% deterministic word ordering
        sub_states_dir = temp_dir / "subtitle_states"
        cairo_font_path = DEFAULT_FONTS_DIR / "Cairo-Bold.ttf"
        font_file_str = str(cairo_font_path if cairo_font_path.exists() else _get_font(64).path)

        subtitle_states = generate_subtitle_states(
            word_timestamps=word_timestamps or [],
            output_dir=sub_states_dir,
            font_path=font_file_str,
            font_size=64,
            chunk_size=4,
        )

        # 6. Build FFmpeg input list and filter-complex overlay chain
        cmd_inputs = [
            "-i", str(stitched_video),    # [0:v]
            "-i", str(aud_file),          # [1:a]
            "-i", str(gradient_path),     # [2:v]
            "-i", str(badge_path),        # [3:v]
        ]
        bg_music_file = _get_background_music()
        has_bgm = bool(bg_music_file and bg_music_file.exists())
        if has_bgm:
            cmd_inputs += ["-i", str(bg_music_file)]  # [4:a]
            bgm_idx = 4
            first_sub_idx = 5
        else:
            bgm_idx = None
            first_sub_idx = 4

        for st in subtitle_states:
            cmd_inputs += ["-i", st["path"]]

        # Video overlay chain: Gradient -> Top Badge -> Subtitle States
        filter_parts = [
            "[0:v][2:v]overlay=0:1920-650[v_grad]",
            "[v_grad][3:v]overlay=0:0[v_badge]",
        ]
        cur_v = "v_badge"

        sub_y = 1450
        for i, st in enumerate(subtitle_states):
            in_idx = first_sub_idx + i
            out_v = f"v_sub{i}"
            t_start = st["start"]
            t_end = st["end"]
            filter_parts.append(
                f"[{cur_v}][{in_idx}:v]overlay=0:{sub_y}:enable='between(t,{t_start:.3f},{t_end:.3f})'[{out_v}]"
            )
            cur_v = out_v

        final_v_label = cur_v

        # Audio filter: Mix BGM at 12% if available
        if has_bgm:
            print(f"🎵 تم دمج الموسيقى التصويرية للأخبار بنجاح: {bg_music_file.name}")
            filter_parts.append(
                f"[{bgm_idx}:a]aloop=loop=-1:size=2e+09,atrim=0:{total_duration:.3f},"
                f"volume=0.12[bgm]"
            )
            filter_parts.append(
                f"[1:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
            audio_map = "[aout]"
        else:
            audio_map = "1:a"

        full_filter = ";".join(filter_parts)
        cmd_final = [
            "ffmpeg",
            "-y",
            *cmd_inputs,
            "-filter_complex",
            full_filter,
            "-map",
            f"[{final_v_label}]",
            "-map",
            audio_map,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-t",
            f"{total_duration:.3f}",
            "-movflags",
            "+faststart",
            str(output_file),
        ]

        print(f"📹 جاري تصدير الفيديو النهائي عبر FFmpeg ({total_duration:.1f} ثانية, {num_scenes} مشاهد, {fps} FPS)...")
        res_final = subprocess.run(cmd_final, capture_output=True, text=True)
        if res_final.returncode != 0:
            raise RuntimeError(f"FFmpeg final composition failed: {res_final.stderr[-500:]}")

        if output_file.exists() and output_file.stat().st_size > 10000:
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"✅ تم تصدير مقطع الريلز الاحترافي بنجاح ({file_size_mb:.1f} MB, {total_duration:.1f}s): {output_file.name}")
            LOGGER.info("News reel saved via native FFmpeg: %s (%.1f MB)", output_file, file_size_mb)
            return str(output_file)
        else:
            print("❌ ملف الفيديو الناتج فارغ أو تالف.")
            return None

    except Exception as exc:
        LOGGER.error("Failed to compose news reel: %s", exc)
        print(f"❌ خطأ أثناء تركيب مقطع الريلز: {exc}")
        return None

    finally:
        # 7. Safe temporary directory cleanup
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
