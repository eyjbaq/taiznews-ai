"""AI Image Generator using Pollinations.ai for news reel backgrounds.

Generates photorealistic vertical (1080x1920) images based on an English prompt
returned by Gemini's editorial engine. Images are saved locally for reel composition.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, ImageFilter

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "generated_images"

POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt"

# Reels dimensions (vertical 9:16)
REEL_WIDTH = 1080
REEL_HEIGHT = 1920
SUPERSAMPLE_FACTOR = 1.4


def generate_ai_image(
    prompt: str,
    output_dir: Path | str | None = None,
    width: int = REEL_WIDTH,
    height: int = REEL_HEIGHT,
    slug: str = "reel_bg",
    max_retries: int = 3,
    timeout: int = 120,
) -> str | None:
    """Generate a photorealistic image via Pollinations.ai, supersampled and enhanced.

    Args:
        prompt: English description of the desired image.
        output_dir: Directory to save the generated image.
        width: Image width in pixels (default 1080).
        height: Image height in pixels (default 1920).
        slug: Filename slug prefix.
        max_retries: Number of retry attempts on failure.
        timeout: HTTP request timeout in seconds.

    Returns:
        Absolute path to the saved image file, or None on failure.
    """
    if not prompt or not prompt.strip():
        LOGGER.warning("Empty image prompt provided, skipping AI image generation.")
        print("⚠️ لم يتم توفير وصف لتوليد الصورة، تخطي توليد صورة AI.")
        return None

    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Request at supersampled resolution with enhance=true for richer details
    gen_width = int(width * SUPERSAMPLE_FACTOR)
    gen_height = int(height * SUPERSAMPLE_FACTOR)
    encoded_prompt = quote(prompt.strip(), safe="")
    url = f"{POLLINATIONS_BASE_URL}/{encoded_prompt}?width={gen_width}&height={gen_height}&model=flux&nologo=true&enhance=true"

    print(f"🎨 جاري توليد صورة AI واقعية عبر Pollinations (Flux Engine - {gen_width}x{gen_height} + enhance)...")

    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                wait_time = 5 * attempt
                print(f"🔄 إعادة المحاولة ({attempt}/{max_retries}) بعد {wait_time} ثوانٍ...")
                time.sleep(wait_time)

            response = requests.get(url, timeout=timeout, stream=True)

            if response.status_code != 200:
                LOGGER.warning(
                    "Pollinations API returned HTTP %d on attempt %d",
                    response.status_code, attempt,
                )
                continue

            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type:
                LOGGER.warning(
                    "Unexpected content type from Pollinations: %s", content_type
                )
                continue

            # Read the full image content
            image_data = response.content
            if len(image_data) < 5000:
                LOGGER.warning(
                    "Image data too small (%d bytes), likely an error response.",
                    len(image_data),
                )
                continue

            # Save initial image
            timestamp = int(time.time())
            output_file = out_dir / f"{slug}_{timestamp}.png"
            with open(output_file, "wb") as f:
                f.write(image_data)

            # Post-processing: downscale with LANCZOS and subtle unsharp mask for maximum visual clarity
            try:
                with Image.open(output_file) as img:
                    img = img.convert("RGB")
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
                    img.save(output_file, "PNG", optimize=True)
            except Exception as post_err:
                LOGGER.warning("Post-processing image %s failed: %s", output_file.name, post_err)

            file_size_kb = output_file.stat().st_size / 1024
            print(f"✅ تم توليد وتوضيح صورة AI بنجاح ({file_size_kb:.0f} KB): {output_file.name}")
            LOGGER.info("AI image saved to: %s (%d KB)", output_file, file_size_kb)
            return str(output_file)

        except requests.exceptions.Timeout:
            LOGGER.warning("Pollinations request timed out on attempt %d", attempt)
            print(f"⏳ انتهت مهلة الاتصال بـ Pollinations (محاولة {attempt}/{max_retries})...")
        except requests.exceptions.ConnectionError as ce:
            LOGGER.warning("Connection error to Pollinations on attempt %d: %s", attempt, ce)
            print(f"🌐 خطأ في الاتصال بـ Pollinations (محاولة {attempt}/{max_retries})...")
        except Exception as exc:
            LOGGER.error("Unexpected error generating AI image: %s", exc)
            print(f"❌ خطأ غير متوقع أثناء توليد الصورة: {exc}")
            break

    print("❌ تعذر توليد صورة AI بعد جميع المحاولات.")
    return None


def generate_ai_images(
    prompts: list[str],
    output_dir: Path | str | None = None,
    width: int = REEL_WIDTH,
    height: int = REEL_HEIGHT,
    slug_prefix: str = "reel_scene",
    max_workers: int = 3,
) -> list[str]:
    """Generate multiple AI storyboard scenes concurrently via Pollinations Flux while preserving scene order."""
    if not prompts:
        return []

    print(f"🎨 [Storyboarding] جاري توليد {len(prompts)} مشاهد مصورة عبر Flux Engine (بالتوازي - {max_workers} مسارات)...")

    results: dict[int, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, p in enumerate(prompts):
            if idx > 0:
                time.sleep(1.2)  # Stagger submission to prevent instant 429 bursts
            fut = executor.submit(
                generate_ai_image,
                prompt=p,
                output_dir=output_dir,
                width=width,
                height=height,
                slug=f"{slug_prefix}_{idx + 1}",
            )
            future_to_idx[fut] = idx
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                path = future.result()
                if path:
                    results[idx] = path
            except Exception as exc:
                LOGGER.warning("Scene %d generation failed: %s", idx, exc)

    # Maintain strict original storyboard scene order (1 -> 5)
    ordered_paths = [results[i] for i in sorted(results.keys())]
    print(f"🎬 اكتمل توليد {len(ordered_paths)}/{len(prompts)} مشاهد مصورة متتالية للريلز.")
    return ordered_paths

