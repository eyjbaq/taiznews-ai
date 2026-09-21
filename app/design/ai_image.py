"""AI Image Generator for news reel backgrounds with Cloudflare Workers AI & Pollinations Smart Fallback.

Primary Engine:
  - Cloudflare Workers AI REST API (@cf/black-forest-labs/flux-2-klein-4b) for ultra-fast,
    high-detail photojournalistic news imagery.
Fallback Engine:
  - Pollinations.ai (Flux Engine, supersampled and enhanced) when Cloudflare credentials
    are absent or when the API quota is exhausted / rate-limited.
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from PIL import Image, ImageFilter

# Console encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "generated_images"

# Cloudflare Workers AI configuration
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_FLUX_MODEL = os.getenv(
    "CLOUDFLARE_FLUX_MODEL",
    "@cf/black-forest-labs/flux-2-klein-4b",
)

# Pollinations configuration
POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt"

# Reels dimensions (vertical 9:16)
REEL_WIDTH = 1080
REEL_HEIGHT = 1920
SUPERSAMPLE_FACTOR = 1.4

# Journalistic documentary negative reinforcement suffix
DOCUMENTARY_STYLE_SUFFIX = (
    ", raw documentary photojournalism, authentic Associated Press news dispatch style, 35mm lens, natural daylight, "
    "gritty authentic texture, realistic Taiz Yemen environment --no sunset, no golden hour, no studio portrait, "
    "no cinematic rim light, no CGI, no 3D render, no digital painting, no fantasy, no watermark, no logo"
)


def _generate_via_cloudflare(
    prompt: str,
    output_file: Path,
    width: int = REEL_WIDTH,
    height: int = REEL_HEIGHT,
    timeout: int = 30,
) -> bool:
    """Generate image via Cloudflare Workers AI REST API."""
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", CLOUDFLARE_ACCOUNT_ID)
    api_token = os.getenv("CLOUDFLARE_API_TOKEN", CLOUDFLARE_API_TOKEN)

    if not account_id or not api_token:
        LOGGER.debug("Cloudflare Workers AI credentials missing, skipping primary engine.")
        return False

    full_prompt = prompt.strip() + DOCUMENTARY_STYLE_SUFFIX
    model = CLOUDFLARE_FLUX_MODEL
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "User-Agent": "TaizNews-AI-Production/2.1",
    }

    try:
        # flux-2-klein-4b expects multipart/form-data with prompt
        resp = requests.post(
            url,
            headers=headers,
            files={"prompt": (None, full_prompt)},
            timeout=timeout,
        )

        if resp.status_code == 400 and "multipart" not in resp.text.lower():
            # Fallback for models expecting application/json
            resp = requests.post(
                url,
                headers=headers,
                json={"prompt": full_prompt},
                timeout=timeout,
            )

        if resp.status_code != 200:
            LOGGER.warning("Cloudflare AI returned HTTP %d: %s", resp.status_code, resp.text[:200])
            return False

        # Parse base64 from JSON or direct binary
        raw_bytes: bytes | None = None
        content_type = resp.headers.get("content-type", "")

        if "application/json" in content_type:
            data = resp.json()
            b64_img = data.get("result", {}).get("image")
            if b64_img:
                raw_bytes = base64.b64decode(b64_img)
        elif len(resp.content) > 10000:
            raw_bytes = resp.content

        if not raw_bytes or len(raw_bytes) < 5000:
            LOGGER.warning("Cloudflare returned empty/truncated image payload.")
            return False

        # Save and post-process to exact reel dimensions (9:16)
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img = img.convert("RGB")
            # Crop to 9:16 aspect ratio then resize
            im_w, im_h = img.size
            target_ratio = width / float(height)
            current_ratio = im_w / float(im_h)
            if current_ratio > target_ratio:
                new_w = int(im_h * target_ratio)
                left = (im_w - new_w) // 2
                cropped = img.crop((left, 0, left + new_w, im_h))
            else:
                new_h = int(im_w / target_ratio)
                top = (im_h - new_h) // 2
                cropped = img.crop((0, top, im_w, top + new_h))

            final_img = cropped.resize((width, height), Image.Resampling.LANCZOS)
            final_img = final_img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=100, threshold=3))
            final_img.save(output_file, "JPEG", quality=92, optimize=True)

        return True

    except Exception as exc:
        LOGGER.warning("Cloudflare Workers AI generation failed: %s", exc)
        return False


def _generate_via_pollinations(
    prompt: str,
    output_file: Path,
    width: int = REEL_WIDTH,
    height: int = REEL_HEIGHT,
    max_retries: int = 2,
    timeout: int = 40,
) -> bool:
    """Generate image via Pollinations.ai (Flux Engine) as reliable free fallback."""
    gen_width = int(width * SUPERSAMPLE_FACTOR)
    gen_height = int(height * SUPERSAMPLE_FACTOR)
    full_prompt = prompt.strip() + DOCUMENTARY_STYLE_SUFFIX
    encoded_prompt = quote(full_prompt, safe="")
    url = f"{POLLINATIONS_BASE_URL}/{encoded_prompt}?width={gen_width}&height={gen_height}&model=flux&nologo=true&enhance=true"

    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                time.sleep(2 * attempt)

            response = requests.get(url, timeout=timeout, stream=True)
            if response.status_code != 200 or len(response.content) < 5000:
                continue

            with Image.open(io.BytesIO(response.content)) as img:
                img = img.convert("RGB")
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=110, threshold=3))
                img.save(output_file, "JPEG", quality=90, optimize=True)
            return True

        except Exception as exc:
            LOGGER.debug("Pollinations attempt %d error: %s", attempt, exc)

    return False


def generate_ai_image(
    prompt: str,
    output_dir: Path | str | None = None,
    width: int = REEL_WIDTH,
    height: int = REEL_HEIGHT,
    slug: str = "reel_scene",
    max_retries: int = 2,
    timeout: int = 30,
) -> str | None:
    """Generate a photojournalistic news image using Cloudflare Workers AI with automatic Pollinations fallback.

    Pipeline:
      1. Try Cloudflare Workers AI (Flux-2-Klein-4B).
      2. If Cloudflare fails or quota exhausted, fall back to Pollinations Flux.
    """
    if not prompt or not prompt.strip():
        LOGGER.warning("Empty image prompt provided, skipping AI image generation.")
        return None

    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    output_file = out_dir / f"{slug}_{timestamp}.jpg"

    # 1. Primary Engine: Cloudflare Workers AI
    print(f"🎨 جاري توليد الصورة عبر محرك Cloudflare Workers AI (Flux)...")
    if _generate_via_cloudflare(prompt, output_file, width=width, height=height, timeout=timeout):
        file_size_kb = output_file.stat().st_size / 1024
        print(f"✅ تم توليد الصورة عبر Cloudflare AI بنجاح ({file_size_kb:.0f} KB): {output_file.name}")
        return str(output_file)

    # 2. Fallback Engine: Pollinations Flux
    print(f"🔄 [التبديل التلقائي] تعذر التوليد عبر Cloudflare AI، جاري التبديل للمحرك الاحتياطي (Pollinations Flux)...")
    if _generate_via_pollinations(prompt, output_file, width=width, height=height, max_retries=max_retries, timeout=timeout):
        file_size_kb = output_file.stat().st_size / 1024
        print(f"✅ تم توليد الصورة عبر المحرك الاحتياطي بنجاح ({file_size_kb:.0f} KB): {output_file.name}")
        return str(output_file)

    print(f"❌ فشل توليد الصورة عبر جميع المحركات المتاحة.")
    return None


def generate_ai_images(
    prompts: list[str],
    output_dir: Path | str | None = None,
    width: int = REEL_WIDTH,
    height: int = REEL_HEIGHT,
    slug_prefix: str = "reel_scene",
    max_workers: int = 3,
) -> list[str]:
    """Generate multiple storyboard scene images concurrently while preserving exact scene order."""
    if not prompts:
        return []

    print(f"🎨 [Storyboarding] جاري توليد {len(prompts)} مشاهد مصورة للريلز (Cloudflare AI مع بديل Pollinations)...")

    results: dict[int, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, p in enumerate(prompts):
            if idx > 0:
                time.sleep(1.0)  # Stagger slightly to avoid rate limit spikes
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
    ordered_paths = [results[i] for i in sorted(results.keys()) if i in results]
    print(f"🎬 اكتمل توليد {len(ordered_paths)}/{len(prompts)} مشاهد مصورة للريلز بنجاح.")
    return ordered_paths
