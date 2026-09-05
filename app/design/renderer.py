"""High-impact visual breaking news card renderer for Taiz News.

Features an iconic red background, white typography, framed 'عاجل' badge,
and custom calligraphic 'تعز' emblem matching professional news network standards.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont, features

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_BIDI_RESHAPER = True
except ImportError:
    HAS_BIDI_RESHAPER = False

from app.ai.models import EditorialPost
from app.collector.base import Article

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FONTS_DIR = BASE_DIR / "assets" / "fonts"
DEFAULT_TEMPLATES_DIR = BASE_DIR / "assets" / "templates"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "generated_images"

FONT_DOWNLOAD_URLS = [
    "https://raw.githubusercontent.com/google/fonts/main/ofl/cairo/Cairo%5Bslnt%2Cwght%5D.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/almarai/Almarai-Bold.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/notokufiarabic/NotoKufiArabic%5Bwght%5D.ttf",
]

# Iconic Urgent News Red
NEWS_RED = (222, 29, 46)


class NewsCardRenderer:
    """Renders high-resolution 'عاجل' breaking news cards in signature red & white."""

    def __init__(
        self,
        fonts_dir: Optional[Path] = None,
        templates_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        width: int = 1080,
        height: int = 1350,
    ) -> None:
        self.fonts_dir = Path(fonts_dir or DEFAULT_FONTS_DIR)
        self.templates_dir = Path(templates_dir or DEFAULT_TEMPLATES_DIR)
        self.output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
        self.width = width
        self.height = height

        self.fonts_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.font_path = self._ensure_font()
        self.logo_path = self._ensure_logo()

    def _ensure_font(self) -> Path:
        """Ensure a bold Arabic headline font exists locally."""
        for font_name in ("Cairo-Bold.ttf", "Cairo[slnt,wght].ttf", "Almarai-Bold.ttf", "NotoKufiArabic.ttf"):
            p = self.fonts_dir / font_name
            if p.exists() and p.stat().st_size > 10000:
                return p

        # Check any existing ttf font
        for existing in self.fonts_dir.glob("*.ttf"):
            if existing.stat().st_size > 10000:
                return existing

        # Download from Google Fonts
        primary = self.fonts_dir / "Cairo-Bold.ttf"
        for url in FONT_DOWNLOAD_URLS:
            try:
                LOGGER.info("Downloading Arabic font from %s...", url)
                res = requests.get(url, timeout=20)
                if res.status_code == 200 and len(res.content) > 10000:
                    with open(primary, "wb") as f:
                        f.write(res.content)
                    return primary
            except Exception as err:
                LOGGER.warning("Could not download font from %s: %s", url, err)

        # Windows font fallback
        for win_path in [Path("C:/Windows/Fonts/arialbd.ttf"), Path("C:/Windows/Fonts/segoeuib.ttf")]:
            if win_path.exists():
                return win_path

        raise FileNotFoundError("Could not locate a suitable Arabic TTF font.")

    def _ensure_logo(self) -> Optional[Path]:
        """Ensure the 'تعز' transparent logo exists."""
        logo_path = self.templates_dir / "taiz_logo.png"
        if logo_path.exists() and logo_path.stat().st_size > 500:
            return logo_path
        return None

    def _format_arabic(self, text: str) -> Tuple[str, str]:
        """Format Arabic text for Pillow rendering based on Raqm/HarfBuzz capabilities."""
        if features.check("raqm"):
            return text, "rtl"
        elif HAS_BIDI_RESHAPER:
            try:
                reshaped = arabic_reshaper.reshape(text)
                return get_display(reshaped), "ltr"
            except Exception:
                return text, "rtl"
        return text, "rtl"

    def _wrap_headline(
        self,
        draw: ImageDraw.ImageDraw,
        headline: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
        direction: str,
    ) -> list[str]:
        """Wrap headline words to achieve balanced, impactful line lengths."""
        words = headline.split()
        lines: list[str] = []
        current: list[str] = []

        for word in words:
            candidate = " ".join(current + [word])
            bbox = draw.textbbox((0, 0), candidate, font=font, direction=direction)
            line_w = bbox[2] - bbox[0]
            if line_w <= max_width or not current:
                current.append(word)
            else:
                lines.append(" ".join(current))
                current = [word]

        if current:
            lines.append(" ".join(current))
        return lines

    def render_card(
        self,
        editorial_post: EditorialPost,
        output_path: Optional[str] = None,
        article: Optional[Article] = None,
    ) -> str:
        """Render the 'عاجل' breaking news card with pure red background, bold headline, and Taiz logo."""
        width, height = self.width, self.height
        image = Image.new("RGB", (width, height), NEWS_RED)
        draw = ImageDraw.Draw(image)

        # 1. Top Framed 'عاجل' Box
        urgent_label, urgent_dir = self._format_arabic("عاجل")
        font_urgent = ImageFont.truetype(str(self.font_path), 64)

        bbox_urgent = draw.textbbox((0, 0), urgent_label, font=font_urgent, direction=urgent_dir)
        uw = bbox_urgent[2] - bbox_urgent[0]
        uh = bbox_urgent[3] - bbox_urgent[1]

        pad_x = 55
        pad_y = 20
        box_w = uw + (pad_x * 2)
        box_h = uh + (pad_y * 2)
        box_x0 = (width - box_w) // 2
        box_y0 = 135
        box_x1 = box_x0 + box_w
        box_y1 = box_y0 + box_h

        border_thickness = 8
        for i in range(border_thickness):
            draw.rectangle(
                [box_x0 + i, box_y0 + i, box_x1 - i, box_y1 - i],
                outline=(255, 255, 255),
            )

        draw.text(
            ((box_x0 + box_x1) // 2, (box_y0 + box_y1) // 2 - 4),
            urgent_label,
            font=font_urgent,
            fill=(255, 255, 255),
            anchor="mm",
            direction=urgent_dir,
        )

        # 2. Headline with enhanced, bold and large font sizing
        words = editorial_post.headline.split()
        words_count = len(words)

        if words_count <= 8:
            title_size = 92
            line_height = 136
        elif words_count <= 13:
            title_size = 82
            line_height = 124
        elif words_count <= 18:
            title_size = 74
            line_height = 112
        else:
            title_size = 66
            line_height = 100

        font_title = ImageFont.truetype(str(self.font_path), title_size)
        formatted_headline, head_dir = self._format_arabic(editorial_post.headline)
        max_w = width - 140  # 940px wide for expansive, bold headlines

        lines = self._wrap_headline(draw, formatted_headline, font_title, max_w, head_dir)
        total_text_h = len(lines) * line_height

        # Position centered vertically between the top box and bottom logo
        start_y = 680 - (total_text_h // 2)

        for i, line in enumerate(lines):
            ly = start_y + (i * line_height)
            draw.text(
                (width // 2, ly),
                line,
                font=font_title,
                fill=(255, 255, 255),
                anchor="mm",
                direction=head_dir,
            )

        # 3. Bottom 'تعز' Calligraphic Logo
        if self.logo_path and self.logo_path.exists():
            try:
                logo_img = Image.open(self.logo_path).convert("RGBA")
                logo_w = 175
                logo_h = int(logo_img.height * (logo_w / logo_img.width))
                logo_resized = logo_img.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
                logo_x = (width - logo_w) // 2
                logo_y = height - logo_h - 105
                image.paste(logo_resized, (logo_x, logo_y), mask=logo_resized)
            except Exception as logo_err:
                LOGGER.warning("Could not paste logo image: %s", logo_err)
                self._draw_fallback_logo(draw, width, height)
        else:
            self._draw_fallback_logo(draw, width, height)

        # 4. Save output
        if not output_path:
            timestamp = int(time.time())
            slug = (article.id if article else f"card_{timestamp}")[:16]
            output_file = self.output_dir / f"card_{slug}_{timestamp}.png"
        else:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

        image.save(output_file, "PNG", quality=95)
        LOGGER.info("Saved breaking news card to: %s", output_file)
        return str(output_file)

    def _draw_fallback_logo(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        """Vector fallback emblem for 'تعز' if template image is unavailable."""
        font_logo = ImageFont.truetype(str(self.font_path), 52)
        cx, cy, r = width // 2, height - 160, 65
        for i in range(4):
            draw.ellipse([cx - r + i, cy - r + i, cx + r - i, cy + r - i], outline=(255, 255, 255))
        taiz_label, taiz_dir = self._format_arabic("تعز")
        draw.text((cx, cy - 3), taiz_label, font=font_logo, fill=(255, 255, 255), anchor="mm", direction=taiz_dir)
