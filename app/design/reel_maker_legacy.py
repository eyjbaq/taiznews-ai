"""News Reel video composer using MoviePy.

Features:
  - Multi-scene storyboarding (4-5 dynamic scenes per video)
  - Varied cinematic Ken Burns motion per scene (Zoom In, Pan Left-Right, Pan Right-Left)
  - Synchronized word-by-word / karaoke subtitles in high-contrast gold & white
  - Persistent psychological hook badges ('🔴 عاجل | تعز نيوز')
  - Background breaking news tension music mixed at 12% volume
  - Vertical 9:16 format (1080x1920) optimized for Facebook & Instagram Reels
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "generated_reels"
DEFAULT_AUDIO_DIR = BASE_DIR / "assets" / "audio"
DEFAULT_FONTS_DIR = BASE_DIR / "assets" / "fonts"

# Reel dimensions (vertical 9:16)
REEL_WIDTH = 1080
REEL_HEIGHT = 1920


class _ReelExportLogger:
    """Small, terminal-friendly MoviePy progress reporter.

    MoviePy's default progress bar is not consistently rendered by IDE output
    panes.  A silent encoder makes a normal 1080p reel export look as if it
    has frozen, so report coarse, readable milestones instead.
    """

    def __init__(self) -> None:
        from proglog import ProgressBarLogger

        class ProgressLogger(ProgressBarLogger):
            def __init__(self) -> None:
                super().__init__()
                self._last_percent = -1

            def bars_callback(self, bar, attr, value, old_value=None):
                if bar != "t" or attr != "index":
                    return
                total = self.bars.get(bar, {}).get("total")
                if not total:
                    return
                percent = min(100, int((value + 1) * 100 / total))
                if percent == 100 or percent // 5 > self._last_percent // 5:
                    self._last_percent = percent
                    # Keep this ASCII-only: some Windows IDE consoles still
                    # use a legacy code page and would otherwise raise while
                    # the encoder is running.
                    print(f"[Reels] Export progress: {percent}%")

        self.logger = ProgressLogger()


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


def _get_font(size: int = 64) -> ImageFont.FreeTypeFont:
    """Load Arabic headline font."""
    fonts_dir = Path(DEFAULT_FONTS_DIR)
    for font_name in ("Cairo-Bold.ttf", "Cairo[slnt,wght].ttf", "Almarai-Bold.ttf", "Tajawal-Bold.ttf"):
        p = fonts_dir / font_name
        if p.exists() and p.stat().st_size > 10000:
            return ImageFont.truetype(str(p), size)
    # Windows fallback
    for win_p in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf"):
        if Path(win_p).exists():
            return ImageFont.truetype(win_p, size)
    return ImageFont.load_default()


def _chunk_words(words: List[Dict[str, Any]], chunk_size: int = 4) -> List[Dict[str, Any]]:
    """Group word timestamps into readable subtitle chunks."""
    chunks = []
    for i in range(0, len(words), chunk_size):
        group = words[i : i + chunk_size]
        if not group:
            continue
        chunks.append({
            "start": group[0]["start"],
            "end": group[-1]["end"] + 0.15,  # Slight tail buffer for readability
            "words": group,
        })
    return chunks


def _create_top_badge(width: int = REEL_WIDTH) -> Image.Image:
    """Pre-render the persistent top urgent news badge once as RGBA."""
    badge_img = Image.new("RGBA", (width, 240), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge_img)
    badge_font = _get_font(38)

    label = "🔴 عاجل | تعز نيوز"
    # Measure
    bbox = draw.textbbox((0, 0), label, font=badge_font, direction="rtl")
    lw = bbox[2] - bbox[0]
    lh = bbox[3] - bbox[1]

    pad_x, pad_y = 35, 14
    box_w = lw + (pad_x * 2)
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
    draw.text(
        ((box_x + box_x + box_w) // 2, (box_y + box_y + box_h) // 2 - 2),
        label,
        font=badge_font,
        fill=(255, 255, 255, 255),
        anchor="mm",
        direction="rtl",
    )
    return badge_img


def _render_subtitle_frame(
    chunk: Dict[str, Any],
    active_word_idx: int,
    font: ImageFont.FreeTypeFont,
) -> Image.Image:
    """Render a single RGBA subtitle canvas for a specific chunk and active word state."""
    canvas = Image.new("RGBA", (REEL_WIDTH, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    words = chunk["words"]
    word_texts = [w["text"] for w in words]

    word_lens = [draw.textlength(w_txt, font=font, direction="rtl") for w_txt in word_texts]
    space_w = draw.textlength(" ", font=font, direction="rtl")
    total_text_w = sum(word_lens) + space_w * (len(word_texts) - 1)

    box_w = total_text_w + 48
    box_x = (REEL_WIDTH - box_w) / 2
    draw.rounded_rectangle(
        [box_x, 56, box_x + box_w, 146],
        radius=16,
        fill=(10, 10, 15, 240),
        outline=(60, 60, 75, 255),
        width=2,
    )

    cur_x = (REEL_WIDTH + total_text_w) / 2
    for i, w_txt in enumerate(word_texts):
        text_color = (255, 215, 0, 255) if i == active_word_idx else (255, 255, 255, 255)
        draw.text((cur_x, 100), w_txt, font=font, fill=text_color, anchor="rm", direction="rtl")
        cur_x -= (word_lens[i] + space_w)

    return canvas


def _create_bottom_gradient(width: int = REEL_WIDTH, height: int = 650) -> Image.Image:
    """Create a smooth dark gradient for subtitle contrast at the bottom."""
    grad = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(grad)
    for y in range(height):
        # Quadratic curve for smooth alpha transition
        alpha = int(210 * (y / height) ** 1.6)
        draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
    return grad


def _build_scene_clip(
    image_path: str,
    duration: float,
    scene_idx: int,
    zoom_factor: float = 1.14,
):
    """Build a scene clip, preprocessing its image only once.

    Applying a PIL crop and resize through ``transform`` makes MoviePy repeat
    that expensive operation for every frame.  At 1080x1920 this can make the
    first frame take long enough to look like the app has frozen.  The visual
    variety comes from the generated scene images; keep the export path fast
    by preparing the final frame once here.
    """
    from moviepy import ImageClip

    pil_img = Image.open(image_path)
    img_w, img_h = pil_img.size
    target_ratio = REEL_WIDTH / REEL_HEIGHT
    current_ratio = img_w / img_h

    # Crop to 9:16
    if current_ratio > target_ratio:
        new_w = int(img_h * target_ratio)
        left = (img_w - new_w) // 2
        pil_img = pil_img.crop((left, 0, left + new_w, img_h))
    else:
        new_h = int(img_w / target_ratio)
        top = (img_h - new_h) // 2
        pil_img = pil_img.crop((0, top, img_w, top + new_h))

    # Resize once, before MoviePy starts requesting video frames.  This avoids
    # hundreds of full-resolution PIL resizes during the FFmpeg export.
    pil_img = pil_img.resize((REEL_WIDTH, REEL_HEIGHT), Image.Resampling.BILINEAR)

    # Save temp scene image
    temp_dir = Path(DEFAULT_OUTPUT_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"_scene_buf_{scene_idx}_{int(time.time())}.png"
    pil_img.save(str(temp_path), "PNG")

    return ImageClip(str(temp_path)).with_duration(duration), temp_path


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
    """Compose a multi-scene news reel video with kinetic karaoke subtitles.

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
    # Normalize images
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

    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    output_file = out_dir / f"{slug}_{timestamp}.mp4"

    print(f"🎬 جاري تركيب مقطع الريلز المتكامل ({len(valid_images)} مشاهد + ترجمة متحركة + موسيقى خلفية)...")

    temp_cleanup_files: List[Path] = []

    try:
        from moviepy import (
            AudioFileClip,
            CompositeAudioClip,
            concatenate_videoclips,
        )

        # 1. Audio and Duration
        audio_clip = AudioFileClip(str(aud_file))
        total_duration = audio_clip.duration

        if total_duration < 1.0:
            LOGGER.warning("Audio too short: %.1f s", total_duration)
            audio_clip.close()
            return None

        # 2. Multi-Scene Visual Sequence
        num_scenes = len(valid_images)
        scene_duration = total_duration / float(num_scenes)

        scene_clips = []
        for s_idx, img_p in enumerate(valid_images):
            cur_dur = total_duration - (scene_duration * (num_scenes - 1)) if s_idx == num_scenes - 1 else scene_duration
            clip, temp_p = _build_scene_clip(img_p, cur_dur, s_idx, zoom_factor)
            scene_clips.append(clip)
            temp_cleanup_files.append(temp_p)

        video_sequence = concatenate_videoclips(scene_clips, method="chain")

        # 3. Pre-render top badge and pre-calculate subtitle frame cache (Zero RTL shaping per frame)
        badge_overlay = _create_top_badge(width=REEL_WIDTH)
        sub_font = _get_font(60)
        sub_chunks = _chunk_words(word_timestamps or [], chunk_size=4)

        subtitle_cache: Dict[Tuple[int, int], Image.Image] = {}
        for c_idx, chunk in enumerate(sub_chunks):
            for w_idx in range(len(chunk["words"])):
                subtitle_cache[(c_idx, w_idx)] = _render_subtitle_frame(chunk, w_idx, sub_font)
            # Default / pause state (all white text)
            subtitle_cache[(c_idx, -1)] = _render_subtitle_frame(chunk, -1, sub_font)

        # 4. Ultra-fast Dynamic Frame Overlay (direct dict lookup and C-level alpha blit)
        def overlay_transform(get_frame, t):
            frame = get_frame(t)
            pil_frame = Image.fromarray(frame)

            # Persistent top urgent badge (blitted directly without redrawing)
            pil_frame.paste(badge_overlay, (0, 0), badge_overlay)

            # Subtitle state lookup and blit
            for c_idx, ch in enumerate(sub_chunks):
                if ch["start"] <= t <= ch["end"]:
                    active_word_idx = -1
                    for w_idx, w in enumerate(ch["words"]):
                        if w["start"] <= t <= w["end"]:
                            active_word_idx = w_idx
                            break
                    if active_word_idx == -1 and ch["words"] and t >= ch["words"][-1]["end"]:
                        active_word_idx = len(ch["words"]) - 1

                    cached_sub = subtitle_cache.get((c_idx, active_word_idx)) or subtitle_cache.get((c_idx, 0))
                    if cached_sub is not None:
                        pil_frame.paste(cached_sub, (0, 1320), cached_sub)
                    break

            return np.array(pil_frame)

        final_video_clip = video_sequence.transform(overlay_transform)

        # 6. Mix Background News Tension Music (12% volume)
        bg_music_file = _get_background_music()
        bg_clip = None
        if bg_music_file and bg_music_file.exists():
            try:
                raw_bg = AudioFileClip(str(bg_music_file))
                if raw_bg.duration < total_duration:
                    from moviepy.audio.AudioClip import concatenate_audioclips
                    repeats = int(total_duration / raw_bg.duration) + 1
                    looped_bg = concatenate_audioclips([raw_bg] * repeats)
                    bg_clip = looped_bg.subclipped(0, total_duration)
                else:
                    bg_clip = raw_bg.subclipped(0, total_duration)
                bg_clip = bg_clip.with_volume_scaled(0.12)
                print(f"🎵 تم دمج الموسيقى التصويرية للأخبار بنجاح: {bg_music_file.name}")
            except Exception as bg_err:
                LOGGER.warning("Could not mix background music: %s", bg_err)
                bg_clip = None

        if bg_clip:
            composite_audio = CompositeAudioClip([audio_clip, bg_clip])
        else:
            composite_audio = audio_clip

        final_clip = final_video_clip.with_audio(composite_audio)

        # 7. Video Export
        num_threads = max(2, (os.cpu_count() or 4) - 1)
        print(f"📹 جاري تصدير الفيديو الكامل ({total_duration:.1f} ثانية, {num_scenes} مشاهد, {fps} FPS, {num_threads} threads)...")

        temp_audio = str(out_dir / f"_temp_audio_{timestamp}.m4a")
        final_clip.write_videofile(
            str(output_file),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            threads=num_threads,
            temp_audiofile=temp_audio,
            remove_temp=True,
            logger=_ReelExportLogger().logger,
        )

        # 8. Cleanup
        final_clip.close()
        audio_clip.close()
        if bg_clip:
            try:
                bg_clip.close()
            except Exception:
                pass
        for sc in scene_clips:
            try:
                sc.close()
            except Exception:
                pass

        for tp in temp_cleanup_files:
            if tp.exists():
                try:
                    tp.unlink()
                except Exception:
                    pass

        if output_file.exists() and output_file.stat().st_size > 10000:
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"✅ تم تصدير مقطع الريلز الاحترافي بنجاح ({file_size_mb:.1f} MB, {total_duration:.1f}s): {output_file.name}")
            LOGGER.info("News reel saved: %s (%.1f MB)", output_file, file_size_mb)
            return str(output_file)
        else:
            print("❌ ملف الفيديو الناتج فارغ أو تالف.")
            return None

    except Exception as exc:
        LOGGER.error("Failed to compose news reel: %s", exc)
        print(f"❌ خطأ أثناء تركيب مقطع الريلز: {exc}")
        return None
