"""Interactive Broadcast Map News Reel Composer (Production).

Produces 1080x1920 vertical news reels featuring:
  - Satellite or Dark base map of Taiz governorate with Web-Mercator framing
  - Animated high-altitude zoom-in & orbital sweep around the target district
  - Glowing neon district boundaries with laser perimeter trace & spotlight dimming
  - Optional tactical military animations (fighter jets, airstrikes, armor, clash line)
  - Opening hook headline card, channel logo, urgent badge, and RTL karaoke subtitles
  - Native FFmpeg piping + background tension music bed (Meta Reels ready)
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from app.design.tts_generator import _get_audio_duration_ffprobe, generate_news_audio
from app.processing.safe_text import obfuscate_for_facebook_display

from .base_map import (
    GAZETTEER,
    H,
    W,
    BaseMap,
    _bounce,
    _draw_attribution,
    _draw_top_badge,
    _ease,
    _fmt_ar,
    ensure_base_map,
    get_map_font,
    load_base_map_from_meta,
    lookup_location,
)
from .district_boundaries import DISTRICT_MANAGER
from .tactical_military import TacticalMilitaryDirector

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "data" / "generated_reels"
AUDIO_DIR = BASE_DIR / "data" / "generated_audio"
DATA_IMAGES_DIR = BASE_DIR / "data" / "generated_images"
LOGO_PATH = ASSETS_DIR / "templates" / "taiz_logo.png"


def _clean_arabic_for_subtitles(word: str) -> str:
    """Strip diacritics and surrounding punctuation for clean subtitles."""
    cleaned = re.sub(r"[\u0617-\u061A\u064B-\u0652\u06D6-\u06ED]", "", word)
    return cleaned.strip(" \t\n\r،.؟!-:;()[]\"'")


async def _fallback_edge_audio_async(
    text_vocalized: str,
    output_audio_path: Path,
    voice: str = "ar-BH-AliNeural",
    rate: str = "+8%",
) -> List[Dict[str, Any]]:
    """Last-resort Edge-TTS synthesizer if primary TTS rotation fails."""
    import edge_tts

    raw_words = [w for w in text_vocalized.strip().split() if w]
    communicate = edge_tts.Communicate(
        text=text_vocalized,
        voice=voice,
        rate=rate,
        pitch="+0Hz",
        volume="+10%",
        boundary="WordBoundary",
    )
    temp_out = output_audio_path.with_suffix(".tmp.mp3")
    word_timestamps: List[Dict[str, Any]] = []

    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_out, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                idx = len(word_timestamps)
                w_text = raw_words[idx] if idx < len(raw_words) else chunk.get("text", "")
                clean_w = _clean_arabic_for_subtitles(w_text)
                word_timestamps.append({
                    "text": clean_w or w_text,
                    "start": round(chunk["offset"] / 10_000_000, 3),
                    "end": round((chunk["offset"] + chunk["duration"]) / 10_000_000, 3),
                })

    if temp_out.exists() and temp_out.stat().st_size > 1000:
        if output_audio_path.exists():
            output_audio_path.unlink()
        temp_out.rename(output_audio_path)
        return word_timestamps
    return []


def _synthesize_map_narration(
    narration_clean: str,
    narration_vocalized: str,
    slug: str,
    preferred_voice_key: Optional[str] = None,
) -> Tuple[Optional[Path], List[Dict[str, Any]], float]:
    """Generate speech narration via production TTS rotation with Edge-TTS fallback."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    primary_text = (narration_clean or narration_vocalized).strip()
    vocalized_text = (narration_vocalized or primary_text).strip()

    tts_path, word_timestamps = generate_news_audio(
        text=primary_text,
        output_dir=AUDIO_DIR,
        slug=f"map_tts_{slug[:24]}",
        vocalized_fallback_text=vocalized_text,
        preferred_voice_key=preferred_voice_key,
    )

    if tts_path and Path(tts_path).exists() and Path(tts_path).stat().st_size > 1000:
        audio_file = Path(tts_path)
        duration = _get_audio_duration_ffprobe(audio_file)
        if duration > 0.5:
            return audio_file, word_timestamps, duration

    # Fallback direct Edge-TTS if production rotation failed completely
    fallback_path = AUDIO_DIR / f"map_tts_{slug[:24]}_{int(time.time())}.mp3"
    loop = asyncio.new_event_loop()
    try:
        timestamps = loop.run_until_complete(
            _fallback_edge_audio_async(vocalized_text, fallback_path)
        )
        if fallback_path.exists() and fallback_path.stat().st_size > 1000:
            duration = _get_audio_duration_ffprobe(fallback_path)
            if duration > 0.5:
                return fallback_path, timestamps, duration
    except Exception as exc:
        LOGGER.warning("Fallback Edge-TTS failed for map reel: %s", exc)
    finally:
        loop.close()

    return None, [], 0.0


def _create_bottom_gradient(width: int = W, height: int = 600) -> Image.Image:
    """Pre-render vertical dark gradient for subtitle contrast."""
    grad = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(grad)
    for y in range(height):
        ratio = y / height
        alpha = int(220 * (ratio ** 1.8))
        d.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
    return grad


def _draw_hook_card(layer: Image.Image, hook_text: str, alpha: int = 255) -> None:
    """Render high-impact Hook Headline Card during opening seconds with safe auto-wrap and OCR defense."""
    if not hook_text or alpha <= 0:
        return
    safe_hook = obfuscate_for_facebook_display(hook_text)
    d = ImageDraw.Draw(layer)
    max_text_w = W - 180

    font_size = 54
    font = get_map_font(font_size)
    words = safe_hook.strip().split()

    # Step down font size slightly if single-line or wrapped exceeds safe bounds
    lines: List[str] = []
    for candidate_size in (54, 48, 42):
        font = get_map_font(candidate_size)
        shaped_full, dir_full = _fmt_ar(safe_hook.strip())
        bb_full = d.textbbox((0, 0), shaped_full, font=font, direction=dir_full)
        if (bb_full[2] - bb_full[0]) <= max_text_w:
            lines = [safe_hook.strip()]
            break

        # Wrap into balanced lines (max 2 lines)
        lines = []
        curr: List[str] = []
        for w in words:
            test = " ".join(curr + [w])
            sh, dr = _fmt_ar(test)
            bb = d.textbbox((0, 0), sh, font=font, direction=dr)
            if (bb[2] - bb[0]) <= max_text_w or not curr:
                curr.append(w)
            else:
                lines.append(" ".join(curr))
                curr = [w]
        if curr:
            lines.append(" ".join(curr))
        if len(lines) <= 2:
            break
    lines = lines[:2]

    line_widths: List[int] = []
    line_heights: List[int] = []
    for ln in lines:
        sh, dr = _fmt_ar(ln)
        bb = d.textbbox((0, 0), sh, font=font, direction=dr)
        line_widths.append(bb[2] - bb[0])
        line_heights.append(bb[3] - bb[1])

    tw = max(line_widths) if line_widths else 300
    line_step = (max(line_heights) if line_heights else 48) + 16
    total_th = line_step * len(lines)
    pw = min(W - 80, tw + 80)
    ph = total_th + 34
    x0, y0 = (W - pw) // 2, 280

    d.rounded_rectangle(
        [x0, y0, x0 + pw, y0 + ph],
        radius=20,
        fill=(18, 22, 28, int(alpha * 0.92)),
        outline=(255, 45, 65, alpha),
        width=3,
    )
    d.rectangle([x0, y0 + 10, x0 + 10, y0 + ph - 10], fill=(255, 45, 65, alpha))

    start_y = y0 + (ph - total_th) // 2 + line_step // 2 - 4
    for idx, ln in enumerate(lines):
        sh, dr = _fmt_ar(ln)
        d.text(
            (W // 2, start_y + idx * line_step),
            sh,
            font=font,
            fill=(255, 255, 255, alpha),
            anchor="mm",
            direction=dr,
        )


def _draw_tactical_reticle(
    layer: Image.Image,
    cx: float,
    cy: float,
    t: float,
    size: float = 170,
    color: Tuple[int, int, int] = (255, 45, 65),
    alpha: int = 220,
) -> None:
    """Render tactical target lock brackets and rotating radar crosshair."""
    if alpha <= 0:
        return
    d = ImageDraw.Draw(layer)
    hs = size / 2
    arm = size * 0.24
    col = (color[0], color[1], color[2], alpha)
    w = 2

    d.line([(cx - hs, cy - hs), (cx - hs + arm, cy - hs)], fill=col, width=w)
    d.line([(cx - hs, cy - hs), (cx - hs, cy - hs + arm)], fill=col, width=w)
    d.line([(cx + hs, cy - hs), (cx + hs - arm, cy - hs)], fill=col, width=w)
    d.line([(cx + hs, cy - hs), (cx + hs, cy - hs + arm)], fill=col, width=w)
    d.line([(cx - hs, cy + hs), (cx - hs + arm, cy + hs)], fill=col, width=w)
    d.line([(cx - hs, cy + hs), (cx - hs, cy + hs - arm)], fill=col, width=w)
    d.line([(cx + hs, cy + hs), (cx + hs - arm, cy + hs)], fill=col, width=w)
    d.line([(cx + hs, cy + hs), (cx + hs, cy + hs - arm)], fill=col, width=w)

    angle = (t * 2.6) % (2 * math.pi)
    rx = cx + math.cos(angle) * (hs - 12)
    ry = cy + math.sin(angle) * (hs - 12)
    d.line([(cx, cy), (rx, ry)], fill=(0, 255, 220, int(alpha * 0.60)), width=2)

    d.line([(cx - 7, cy), (cx + 7, cy)], fill=(255, 255, 255, alpha), width=1)
    d.line([(cx, cy - 7), (cx, cy + 7)], fill=(255, 255, 255, alpha), width=1)


def _draw_channel_logo(layer: Image.Image, alpha: int = 255) -> None:
    """Render official channel logo badge at top-left with frosted glass styling."""
    if alpha <= 0 or not LOGO_PATH.exists():
        return
    try:
        logo_im = Image.open(LOGO_PATH).convert("RGBA")
        logo_w, logo_h = 100, 100
        logo_resized = logo_im.resize((logo_w, logo_h), Image.Resampling.LANCZOS)

        x0, y0 = 45, 55
        pad = 8
        bw, bh = logo_w + pad * 2, logo_h + pad * 2

        d = ImageDraw.Draw(layer)
        d.rounded_rectangle(
            [x0, y0, x0 + bw, y0 + bh],
            radius=20,
            fill=(12, 16, 24, int(alpha * 0.88)),
            outline=(255, 255, 255, int(alpha * 0.60)),
            width=2,
        )

        if alpha < 255:
            r, g, b, a = logo_resized.split()
            a = a.point(lambda p: int(p * (alpha / 255.0)))
            logo_resized = Image.merge("RGBA", (r, g, b, a))

        layer.paste(logo_resized, (x0 + pad, y0 + pad), mask=logo_resized)
    except Exception as exc:
        LOGGER.warning("Could not draw channel logo: %s", exc)


def _draw_location_chip(layer: Image.Image, text: str, alpha: int = 255) -> None:
    """Render location tag chip in hybrid field scenes."""
    if not text or alpha <= 0:
        return
    font = get_map_font(30)
    clean_text = text.replace("📍", "").strip()
    shaped, direction = _fmt_ar(clean_text)
    d = ImageDraw.Draw(layer)
    bb = d.textbbox((0, 0), shaped, font=font, direction=direction)
    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
    pw, ph = bw + 74, bh + 20
    x0, y0 = W - pw - 45, 140
    d.rounded_rectangle(
        [x0, y0, x0 + pw, y0 + ph],
        radius=14,
        fill=(15, 20, 26, int(alpha * 0.88)),
        outline=(255, 255, 255, int(alpha * 0.6)),
        width=2,
    )
    dot_cx = x0 + 26
    dot_cy = y0 + ph // 2
    d.ellipse([dot_cx - 7, dot_cy - 7, dot_cx + 7, dot_cy + 7], fill=(255, 45, 65, alpha))
    d.ellipse([dot_cx - 3, dot_cy - 3, dot_cx + 3, dot_cy + 3], fill=(255, 255, 255, alpha))
    d.text(
        (x0 + 44 + bw // 2, y0 + ph // 2 - 2),
        shaped,
        font=font,
        fill=(255, 255, 255, alpha),
        anchor="mm",
        direction=direction,
    )


def _draw_mini_radar_pip(
    layer: Image.Image,
    base_map: BaseMap,
    district_feats: List[dict],
    lat: float,
    lon: float,
    t: float,
    pip_size: int = 200,
) -> None:
    """Render circular Mini-Map Radar PIP chip in top-left corner."""
    px, py = base_map.to_img_px(lat, lon)
    pip_scale = 1.3
    vw, vh = pip_size / pip_scale, pip_size / pip_scale
    box = (px - vw / 2, py - vh / 2, px + vw / 2, py + vh / 2)

    pip_crop = base_map.image.transform((pip_size, pip_size), Image.EXTENT, box, resample=Image.BICUBIC).convert("RGBA")
    pip_layer = Image.new("RGBA", (pip_size, pip_size), (0, 0, 0, 0))
    DISTRICT_MANAGER.draw_district_boundary(
        layer=pip_layer,
        features=district_feats,
        base_map=base_map,
        view_box=box,
        scale=pip_scale,
        progress=1.0,
        glow_color=(255, 50, 70),
        fill_color=(255, 30, 50),
        pulse_phase=0.5 + 0.5 * math.sin(t * 6.0),
    )

    p_draw = ImageDraw.Draw(pip_layer)
    cx, cy = pip_size / 2, pip_size / 2
    angle = (t * 2.2) % (2 * math.pi)
    rx = cx + math.cos(angle) * (pip_size / 2 - 8)
    ry = cy + math.sin(angle) * (pip_size / 2 - 8)
    p_draw.line([(cx, cy), (rx, ry)], fill=(0, 255, 200, 160), width=2)
    p_draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 45, 65, 255), outline=(255, 255, 255, 255), width=2)

    pip_combined = Image.alpha_composite(pip_crop, pip_layer)
    mask = Image.new("L", (pip_size, pip_size), 0)
    ImageDraw.Draw(mask).ellipse([2, 2, pip_size - 2, pip_size - 2], fill=255)

    dest_x, dest_y = 45, 65
    layer.paste(pip_combined, (dest_x, dest_y), mask)

    d = ImageDraw.Draw(layer)
    d.ellipse([dest_x, dest_y, dest_x + pip_size, dest_y + pip_size], outline=(255, 45, 65, 230), width=3)
    d.ellipse([dest_x - 3, dest_y - 3, dest_x + pip_size + 3, dest_y + pip_size + 3], outline=(255, 45, 65, 60), width=3)


def _draw_subtitles_karaoke(
    layer: Image.Image,
    word_timestamps: List[Dict[str, Any]],
    current_time: float,
    y_pos: int = 1540,
    chunk_size: int = 4,
) -> None:
    """Render kinetic subtitles with active spoken word highlighted in Gold."""
    if not word_timestamps:
        return

    active_word_idx = -1
    for i, w in enumerate(word_timestamps):
        if w["start"] <= current_time <= w["end"] + 0.15:
            active_word_idx = i
            break

    if active_word_idx == -1:
        for i in range(len(word_timestamps) - 1):
            if word_timestamps[i]["end"] <= current_time <= word_timestamps[i + 1]["start"]:
                active_word_idx = i
                break

    if active_word_idx == -1:
        return

    chunk_start = (active_word_idx // chunk_size) * chunk_size
    chunk = word_timestamps[chunk_start : chunk_start + chunk_size]
    active_sub_idx = active_word_idx - chunk_start

    d = ImageDraw.Draw(layer)
    words_raw = [
        obfuscate_for_facebook_display(_clean_arabic_for_subtitles(str(w.get("text", ""))) or str(w.get("text", "")))
        for w in chunk
    ]
    words_fmt = [_fmt_ar(w) for w in words_raw]

    # Auto-fit font size so 4-word chunks never exceed horizontal margins
    font = get_map_font(56)
    space_w = 18
    word_widths: List[int] = []
    tot_w = 0
    for candidate_size in (56, 48, 42):
        font = get_map_font(candidate_size)
        sp_bb = d.textbbox((0, 0), " ", font=font, direction="rtl")
        space_w = max(12, sp_bb[2] - sp_bb[0])
        word_widths = []
        for shaped, direction in words_fmt:
            bb = d.textbbox((0, 0), shaped, font=font, direction=direction)
            word_widths.append(bb[2] - bb[0])
        tot_w = sum(word_widths) + space_w * max(0, len(words_fmt) - 1)
        if tot_w <= W - 120:
            break

    ref_shaped, ref_dir = _fmt_ar("أبجد تعز")
    sample_bb = d.textbbox((0, 0), ref_shaped, font=font, direction=ref_dir)
    th = sample_bb[3] - sample_bb[1]

    pad_x, pad_y = 44, 22
    bx0, by0 = (W - tot_w) // 2 - pad_x, y_pos - pad_y
    d.rounded_rectangle(
        [bx0, by0, bx0 + tot_w + pad_x * 2, by0 + th + pad_y * 2],
        radius=20,
        fill=(10, 14, 22, 225),
        outline=(255, 255, 255, 60),
        width=2,
    )

    right_edge = (W + tot_w) // 2
    cur_x = right_edge

    for i, (shaped, direction) in enumerate(words_fmt):
        w_w = word_widths[i]
        is_active = i == active_sub_idx
        text_color = (255, 215, 0, 255) if is_active else (255, 255, 255, 235)

        shadow_offset = 3 if is_active else 2
        for dx, dy in [
            (-shadow_offset, 0), (shadow_offset, 0), (0, -shadow_offset),
            (0, shadow_offset), (shadow_offset, shadow_offset),
        ]:
            d.text(
                (cur_x + dx, y_pos + th // 2 - 4 + dy),
                shaped,
                font=font,
                fill=(0, 0, 0, 240),
                anchor="rm",
                direction=direction,
            )

        if is_active:
            d.rounded_rectangle(
                [cur_x - w_w - 6, y_pos - 8, cur_x + 6, y_pos + th + 8],
                radius=8,
                fill=(255, 215, 0, 45),
            )

        d.text(
            (cur_x, y_pos + th // 2 - 4),
            shaped,
            font=font,
            fill=text_color,
            anchor="rm",
            direction=direction,
        )
        cur_x -= (w_w + space_w)


def _draw_progress_bar(layer: Image.Image, progress: float) -> None:
    """Render animated 6px progress bar along the bottom edge."""
    d = ImageDraw.Draw(layer)
    bar_y = H - 8
    d.line([(0, bar_y), (W, bar_y)], fill=(255, 255, 255, 50), width=6)
    fill_w = int(W * max(0.0, min(1.0, progress)))
    if fill_w > 0:
        d.line([(0, bar_y), (fill_w, bar_y)], fill=(255, 45, 65, 255), width=6)


def compose_interactive_news_reel(
    district_name: str = "المخا",
    label: str = "مديرية المخا",
    hook_text: str = "تطورات متسارعة في الساحل الغربي",
    location_chip_text: str = "المخا | غرب تعز",
    narration_vocalized: str = "",
    narration_clean: str = "",
    fps: int = 24,
    slug: str = "news_reel_map",
    map_style: str = "satellite",
    mode: str = "full_map",
    military: bool = False,
    output_dir: Optional[Path] = None,
    preferred_voice_key: Optional[str] = None,
) -> Optional[Path]:
    """Compose and export a 1080x1920 interactive satellite/tactical map news reel."""
    print("\n" + "=" * 70)
    print(f"🗺️ [خريطة تفاعلية] بدء تركيب ريلز الخريطة: [{district_name}] (النمط: {map_style} | عسكري: {military})")
    print("=" * 70)

    # 1. Load or auto-build Base Map
    base_map_png = ensure_base_map(style=map_style)
    if not base_map_png or not base_map_png.exists():
        LOGGER.error("Base map asset unavailable.")
        print("❌ تعذر العثور على خريطة الأساس لتعز.")
        return None
    base_map = load_base_map_from_meta(base_map_png)

    # 2. Lookup Coordinates & District Polygon Features
    district_feats = DISTRICT_MANAGER.find_district(district_name or label)
    centroid = DISTRICT_MANAGER.get_district_centroid(district_feats)
    if centroid:
        lat, lon = centroid
    else:
        coords = lookup_location(district_name) or lookup_location(label)
        lat, lon = coords if coords else (13.5795, 44.0209)
    print(f"📍 الإحداثيات المستهدفة: ({lat:.4f}, {lon:.4f}) | المعالم الجغرافية: {len(district_feats)}")

    # 3. Synthesize Speech Narration via production TTS rotation
    audio_path, word_timestamps, total_duration = _synthesize_map_narration(
        narration_clean=narration_clean,
        narration_vocalized=narration_vocalized,
        slug=slug,
        preferred_voice_key=preferred_voice_key,
    )
    if not audio_path or total_duration <= 0.5:
        LOGGER.error("Failed to generate speech audio for map reel.")
        print("❌ تعذر توليد التعليق الصوتي لريلز الخريطة.")
        return None

    print(f"⏱️ مدة التعليق الصوتي: {total_duration:.2f} ثانية | الكلمات المتزامنة: {len(word_timestamps)}")

    min_video_duration = 25.0
    if total_duration < min_video_duration:
        print(f"⏳ تمديد مدة الفيديو إلى الحد الأدنى ({min_video_duration:.1f} ثانية) لاستكمال الحركات التكتيكية.")
        total_duration = min_video_duration

    # 4. Prepare Hybrid visuals if hybrid mode requested
    docu_images: List[Image.Image] = []
    if mode == "hybrid":
        img_candidates = sorted(DATA_IMAGES_DIR.glob("reel_scene_*.jpg"), key=lambda p: p.stat().st_size, reverse=True)
        if not img_candidates:
            img_candidates = sorted(DATA_IMAGES_DIR.glob("*.jpg"), key=lambda p: p.stat().st_size, reverse=True)
        for p in img_candidates[:2]:
            try:
                im = Image.open(p).convert("RGB")
                if im.size != (W, H):
                    im = im.resize((W, H), Image.BICUBIC)
                docu_images.append(im)
            except Exception:
                pass
        while len(docu_images) < 2:
            docu_images.append(Image.new("RGB", (W, H), (22, 28, 36)))

    bottom_gradient = _create_bottom_gradient(W, 600)
    map_hook_duration = min(3.8, total_duration * 0.35)
    field_split = map_hook_duration + (total_duration - map_hook_duration) / 2.0

    out_dir = Path(output_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    safe_slug = re.sub(r"[^\w\-]+", "_", slug).strip("_") or "map_reel"
    temp_dir = out_dir / f"_temp_map_{timestamp}_{os.getpid()}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_video_raw = temp_dir / "map_raw.mp4"
    final_video_path = out_dir / f"{safe_slug}_{timestamp}.mp4"

    total_frames = int(total_duration * fps)
    print(f"🎥 جاري معالجة وتصدير {total_frames} إطار خريطة تفاعلية (1080x1920 @ {fps}fps)...")

    px, py = base_map.to_img_px(lat, lon)
    bw, bh = base_map.image.size
    start_scale = max(W / bw, H / bh)
    target_cx, target_cy, target_scale = DISTRICT_MANAGER.calculate_optimal_framing(
        district_feats, base_map, screen_w=W, screen_h=H, fill_ratio_w=0.82, fill_ratio_h=0.52
    )
    end_scale = target_scale
    start_center = (bw / 2, bh / 2)
    end_center = (target_cx, target_cy)
    military_director = TacticalMilitaryDirector((px, py)) if military else None

    cmd_raw = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-g", str(fps * 2), "-keyint_min", str(fps),
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-pix_fmt", "yuv420p", "-r", str(fps), "-an",
        str(temp_video_raw),
    ]
    proc = subprocess.Popen(cmd_raw, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        for f in range(total_frames):
            t = f / fps
            progress = t / total_duration

            if mode == "full_map":
                if t < 3.2:
                    k = _ease(t / 2.8)
                    scale = start_scale + (end_scale - start_scale) * k
                    cx = start_center[0] + (end_center[0] - start_center[0]) * k
                    cy = start_center[1] + (end_center[1] - start_center[1]) * k
                else:
                    orbit_t = t - 3.2
                    orbit_blend = _ease(min(1.0, orbit_t / 1.5))
                    orbit_angle = orbit_t * 0.40
                    cx = end_center[0] + (40.0 * orbit_blend) * math.sin(orbit_angle)
                    cy = end_center[1] + (26.0 * orbit_blend) * math.cos(orbit_angle)
                    scale = end_scale * (1.0 + 0.04 * math.sin(orbit_angle * 1.4) * orbit_blend)

                vw, vh = W / scale, H / scale
                cx = min(max(cx, vw / 2), bw - vw / 2)
                cy = min(max(cy, vh / 2), bh - vh / 2)
                box = (cx - vw / 2, cy - vh / 2, cx + vw / 2, cy + vh / 2)

                frame = base_map.image.transform((W, H), Image.EXTENT, box, resample=Image.BICUBIC).convert("RGBA")
                sx = (px - box[0]) * scale
                sy = (py - box[1]) * scale

                if district_feats and t >= 1.2:
                    spotlight_a = int(120 * _ease((t - 1.2) / 1.4))
                    DISTRICT_MANAGER.draw_district_spotlight(frame, district_feats, base_map, box, scale, dim_alpha=spotlight_a)

                layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))

                if district_feats and t >= 0.4:
                    if t < 2.8:
                        trace_prog = _ease((t - 0.4) / 2.4)
                        DISTRICT_MANAGER.draw_district_boundary_animated(
                            layer, district_feats, base_map, box, scale,
                            trace_progress=trace_prog, glow_alpha=0.0, pulse_phase=0.0, draw_spark=True,
                        )
                    else:
                        pulse = 0.5 + 0.5 * math.sin(t * 5.0)
                        glow_prog = _ease((t - 2.8) / 0.6)
                        DISTRICT_MANAGER.draw_district_boundary_animated(
                            layer, district_feats, base_map, box, scale,
                            trace_progress=1.0, glow_alpha=glow_prog, pulse_phase=pulse, draw_spark=False,
                        )

                if t >= 2.4:
                    ld = ImageDraw.Draw(layer)
                    for phase in (0.0, 0.5):
                        p_rad = ((t - 2.4) / 1.1 + phase) % 1.0
                        rad = 25 + 160 * p_rad
                        a_rad = int(220 * (1 - p_rad))
                        ld.ellipse(
                            [sx - rad, sy - rad * 0.55, sx + rad, sy + rad * 0.55],
                            outline=(255, 60, 60, a_rad),
                            width=6,
                        )

                if t >= 2.8:
                    reticle_a = int(220 * _ease((t - 2.8) / 0.5))
                    _draw_tactical_reticle(layer, sx, sy, t, size=170, alpha=reticle_a)

                if military and military_director:
                    military_director.render(layer, box, scale, t)

                if t >= 0.2:
                    top_a = int(255 * _ease((t - 0.2) / 0.4))
                    _draw_top_badge(layer, top_a)
                    _draw_channel_logo(layer, top_a)

                if t <= 3.8:
                    card_a = int(255 * _ease((t - 0.3) / 0.4))
                    if t > 3.2:
                        card_a = int(255 * (1.0 - _ease((t - 3.2) / 0.6)))
                    if card_a > 0:
                        _draw_hook_card(layer, hook_text, card_a)

                frame = Image.alpha_composite(frame, layer)
                _draw_attribution(frame, base_map.attribution)

            else:
                if t < map_hook_duration:
                    k = _ease(t / 1.3)
                    scale = start_scale + (end_scale - start_scale) * k
                    cx = start_center[0] + (end_center[0] - start_center[0]) * k
                    cy = start_center[1] + (end_center[1] - start_center[1]) * k
                    vw, vh = W / scale, H / scale
                    cx = min(max(cx, vw / 2), bw - vw / 2)
                    cy = min(max(cy, vh / 2), bh - vh / 2)
                    box = (cx - vw / 2, cy - vh / 2, cx + vw / 2, cy + vh / 2)

                    frame = base_map.image.transform((W, H), Image.EXTENT, box, resample=Image.BICUBIC).convert("RGBA")
                    sx = (px - box[0]) * scale
                    sy = (py - box[1]) * scale

                    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                    if district_feats and t >= 0.2:
                        bound_p = _ease((t - 0.2) / 0.8)
                        pulse = 0.5 + 0.5 * math.sin(t * 6.0)
                        DISTRICT_MANAGER.draw_district_boundary(
                            layer, district_feats, base_map, box, scale, progress=bound_p, pulse_phase=pulse
                        )
                    if t >= 1.7:
                        ld = ImageDraw.Draw(layer)
                        for phase in (0.0, 0.5):
                            p_rad = ((t - 1.7) / 1.0 + phase) % 1.0
                            rad = 20 + 150 * p_rad
                            a_rad = int(200 * (1 - p_rad))
                            ld.ellipse(
                                [sx - rad, sy - rad * 0.55, sx + rad, sy + rad * 0.55],
                                outline=(255, 60, 60, a_rad),
                                width=6,
                            )
                    if t >= 0.3:
                        _draw_hook_card(layer, hook_text, int(255 * _ease((t - 0.3) / 0.4)))
                    if t >= 0.2:
                        top_a = int(255 * _ease((t - 0.2) / 0.4))
                        _draw_top_badge(layer, top_a)
                        _draw_channel_logo(layer, top_a)

                    frame = Image.alpha_composite(frame, layer)
                    _draw_attribution(frame, base_map.attribution)
                else:
                    active_img = docu_images[0] if t < field_split else docu_images[1]
                    scene_t = (t - map_hook_duration) if t < field_split else (t - field_split)
                    scene_dur = (field_split - map_hook_duration) if t < field_split else (total_duration - field_split)
                    motion_k = min(1.0, scene_t / max(0.1, scene_dur))

                    zoom_k = 1.05 + 0.08 * motion_k
                    pan_x = int(W * 0.04 * motion_k)
                    vw, vh = int(W / zoom_k), int(H / zoom_k)
                    crop_x0 = max(0, min(W - vw, (W - vw) // 2 + pan_x))
                    crop_y0 = max(0, min(H - vh, (H - vh) // 2))
                    frame = active_img.crop((crop_x0, crop_y0, crop_x0 + vw, crop_y0 + vh)).resize((W, H), Image.BICUBIC).convert("RGBA")

                    frame.paste(bottom_gradient, (0, H - 600), bottom_gradient)
                    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                    _draw_top_badge(layer, 255)
                    _draw_location_chip(layer, location_chip_text, 255)
                    _draw_mini_radar_pip(layer, base_map, district_feats, lat, lon, t, pip_size=190)
                    frame = Image.alpha_composite(frame, layer)

            ov_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            _draw_subtitles_karaoke(ov_layer, word_timestamps, t, y_pos=1540, chunk_size=4)
            _draw_progress_bar(ov_layer, progress)
            frame = Image.alpha_composite(frame, ov_layer)

            proc.stdin.write(frame.convert("RGB").tobytes())

        if proc.stdin:
            proc.stdin.flush()
            proc.stdin.close()
        proc.stdin = None  # Avoid POSIX flush-of-closed-file error in communicate()

        _, stderr_data = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg raw map video failed: {stderr_data.decode(errors='replace')[-400:]}")

        # 5. Audio Mixing with Background Tension Music
        print("🎵 جاري دمج التعليق الصوتي مع الموسيقى الخلفية لريلز الخريطة...")
        bgm_path = ASSETS_DIR / "audio" / "news_bed.wav"
        fade_start = max(0.0, total_duration - 1.5)

        if bgm_path.exists() and bgm_path.stat().st_size > 5000:
            cmd_mux = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(temp_video_raw),
                "-i", str(audio_path),
                "-i", str(bgm_path),
                "-filter_complex",
                f"[1:a]aresample=48000,apad,atrim=0:{total_duration:.3f},volume=1.0[a_voice];"
                f"[2:a]aresample=48000,aloop=loop=-1:size=2e+09,atrim=0:{total_duration:.3f},volume=0.12[a_bgm];"
                f"[a_voice][a_bgm]amix=inputs=2:duration=first:dropout_transition=2,afade=t=out:st={fade_start:.2f}:d=1.5[a_out]",
                "-map", "0:v",
                "-map", "[a_out]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                "-t", f"{total_duration:.3f}",
                "-movflags", "+faststart",
                str(final_video_path),
            ]
        else:
            cmd_mux = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(temp_video_raw),
                "-i", str(audio_path),
                "-filter_complex",
                f"[1:a]aresample=48000,apad,atrim=0:{total_duration:.3f},afade=t=out:st={fade_start:.2f}:d=1.5[a_out]",
                "-map", "0:v",
                "-map", "[a_out]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                "-t", f"{total_duration:.3f}",
                "-movflags", "+faststart",
                str(final_video_path),
            ]

        mux_res = subprocess.run(cmd_mux, capture_output=True, text=True)
        if mux_res.returncode != 0:
            raise RuntimeError(f"FFmpeg audio mux failed: {mux_res.stderr[-400:]}")

        if final_video_path.exists() and final_video_path.stat().st_size > 10000:
            size_mb = final_video_path.stat().st_size / (1024 * 1024)
            print(f"✅ تم إنتاج ريلز الخريطة التفاعلية بنجاح ({size_mb:.1f} MB, {total_duration:.1f}s): {final_video_path.name}")
            return final_video_path
        return None

    except Exception as exc:
        if proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.stdin = None
        proc.kill()
        LOGGER.error("Interactive map reel composition failed: %s", exc)
        print(f"❌ خطأ أثناء تركيب ريلز الخريطة التفاعلية: {exc}")
        return None

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
