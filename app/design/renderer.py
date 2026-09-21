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
from app.design.ai_image import generate_ai_image

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FONTS_DIR = BASE_DIR / "assets" / "fonts"
DEFAULT_TEMPLATES_DIR = BASE_DIR / "assets" / "templates"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "generated_images"

FONT_DOWNLOAD_URLS = [
    "https://raw.githubusercontent.com/google/fonts/main/ofl/almarai/Almarai-Bold.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/cairo/Cairo%5Bslnt%2Cwght%5D.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/notokufiarabic/NotoKufiArabic%5Bwght%5D.ttf",
]

# Iconic Urgent News Red
NEWS_RED = (182, 18, 26)


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
        for font_name in ("Almarai-Bold.ttf", "Cairo-Bold.ttf", "Cairo[slnt,wght].ttf", "NotoKufiArabic.ttf"):
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
        bg_image_path: Optional[str] = None,
        badge_text: Optional[str] = None,
    ) -> str:
        """Render high-impact breaking news card matching Taiz News visual identity.

        Components:
          1. AI Generated / Photographic Background (1080x1350)
          2. Tactical grid overlay & subtle broadcast dark gradients
          3. Top-left channel bug container (Taiz emblem + 'تعز نيوز')
          4. White pill badge on the right ('مصادر محلية:' / 'عاجل:') with red chevrons
          5. Signature crimson red rounded card with bold right-aligned headline
          6. Bottom social media handle & broadcast icons ('TaizNews')
        """
        width, height = self.width, self.height

        # 1. Resolve Background Image
        resolved_bg: Optional[Path] = None
        if bg_image_path and Path(bg_image_path).exists():
            resolved_bg = Path(bg_image_path)
        elif editorial_post.image_prompt_en:
            try:
                slug = (article.id if article else f"card_{int(time.time())}")[:16]
                generated_bg = generate_ai_image(
                    prompt=editorial_post.image_prompt_en,
                    width=width,
                    height=height,
                    slug=f"card_bg_{slug}",
                )
                if generated_bg and Path(generated_bg).exists():
                    resolved_bg = Path(generated_bg)
            except Exception as bg_err:
                LOGGER.warning("Could not generate AI background image for card: %s", bg_err)

        if resolved_bg:
            try:
                img = Image.open(resolved_bg).convert("RGB")
                target_ratio = width / height
                src_ratio = img.width / img.height
                if src_ratio > target_ratio:
                    new_w = int(img.height * target_ratio)
                    left = (img.width - new_w) // 2
                    img = img.crop((left, 0, left + new_w, img.height))
                else:
                    new_h = int(img.width / target_ratio)
                    top = (img.height - new_h) // 2
                    img = img.crop((0, top, img.width, top + new_h))
                base_image = img.resize((width, height), Image.Resampling.LANCZOS)
            except Exception as load_err:
                LOGGER.warning("Failed loading background image %s: %s", resolved_bg, load_err)
                base_image = Image.new("RGB", (width, height), (14, 20, 32))
        else:
            base_image = Image.new("RGB", (width, height), (14, 20, 32))

        # 2. Overlays: Tactical Grid + Gradients
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)

        # Tactical grid
        grid_step = 70
        for x in range(0, width, grid_step):
            ov_draw.line([(x, 0), (x, height)], fill=(255, 255, 255, 10), width=1)
        for y in range(0, height, grid_step):
            ov_draw.line([(0, y), (width, y)], fill=(255, 255, 255, 10), width=1)

        # Smooth top gradient (y=0 to 240)
        for y in range(0, 240):
            alpha = int((1 - (y / 240)) * 90)
            ov_draw.line([(0, y), (width, y)], fill=(8, 12, 20, alpha))

        # Deep bottom gradient (y=500 to height)
        for y in range(500, height):
            prog = (y - 500) / (height - 500)
            alpha = int((prog ** 1.5) * 255)
            ov_draw.line([(0, y), (width, y)], fill=(8, 12, 20, min(255, alpha)))

        base_image.paste(overlay, (0, 0), mask=overlay)
        draw = ImageDraw.Draw(base_image)

        # 3. Top-Left Station Bug Container
        bug_x0, bug_y0, bug_w, bug_h = 65, 60, 135, 150
        container = Image.new("RGBA", (bug_w, bug_h), (0, 0, 0, 0))
        cdraw = ImageDraw.Draw(container)
        cdraw.rounded_rectangle(
            [0, 0, bug_w - 1, bug_h - 1],
            radius=18,
            fill=(10, 14, 22, 190),
            outline=(255, 255, 255, 75),
            width=2,
        )
        base_image.paste(container, (bug_x0, bug_y0), mask=container)

        if self.logo_path and self.logo_path.exists():
            try:
                logo_img = Image.open(self.logo_path).convert("RGBA")
                logo_resized = logo_img.resize((82, 82), Image.Resampling.LANCZOS)
                base_image.paste(logo_resized, (bug_x0 + (bug_w - 82) // 2, bug_y0 + 12), mask=logo_resized)
            except Exception as logo_err:
                LOGGER.warning("Could not paste logo in container: %s", logo_err)

        station_text, station_dir = self._format_arabic("تعز نيوز")
        font_bug = ImageFont.truetype(str(self.font_path), 24)
        draw.text(
            (bug_x0 + bug_w // 2, bug_y0 + 125),
            station_text,
            font=font_bug,
            fill=(255, 255, 255),
            anchor="mm",
            direction=station_dir,
        )

        # 4. Headline Formatting & Red Box Calculation
        words = editorial_post.headline.split()
        words_count = len(words)
        if words_count <= 8:
            title_size = 56
            line_height = 82
        elif words_count <= 14:
            title_size = 50
            line_height = 74
        elif words_count <= 20:
            title_size = 46
            line_height = 68
        else:
            title_size = 40
            line_height = 60

        font_title = ImageFont.truetype(str(self.font_path), title_size)
        formatted_headline, head_dir = self._format_arabic(editorial_post.headline)
        max_text_w = 860

        lines = self._wrap_headline(draw, formatted_headline, font_title, max_text_w, head_dir)

        pad_y = 38
        box_x0 = 65
        box_x1 = 1015
        box_y1 = 1225
        box_h = len(lines) * line_height + (pad_y * 2)
        box_y0 = box_y1 - box_h

        # Draw Signature Red Box
        red_box = Image.new("RGBA", (box_x1 - box_x0, box_h), (0, 0, 0, 0))
        rdraw = ImageDraw.Draw(red_box)
        rdraw.rounded_rectangle(
            [0, 0, box_x1 - box_x0 - 1, box_h - 1],
            radius=22,
            fill=(*NEWS_RED, 255),
        )
        base_image.paste(red_box, (box_x0, box_y0), mask=red_box)

        # 5. White Pill Badge (Right-aligned above the red box)
        if not badge_text:
            if "عاجل" in editorial_post.headline or editorial_post.importance_score >= 85:
                badge_text = "عاجل:"
            else:
                badge_text = "مصادر محلية:"

        formatted_badge, badge_dir = self._format_arabic(badge_text)
        font_badge = ImageFont.truetype(str(self.font_path), 32)
        bb_badge = draw.textbbox((0, 0), formatted_badge, font=font_badge, direction=badge_dir)
        bw = bb_badge[2] - bb_badge[0]

        badge_h = 56
        badge_w = bw + 80
        badge_x1 = box_x1
        badge_x0 = badge_x1 - badge_w
        badge_y1 = box_y0 - 8
        badge_y0 = badge_y1 - badge_h

        badge_img = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
        bdraw = ImageDraw.Draw(badge_img)
        bdraw.rounded_rectangle([0, 0, badge_w - 1, badge_h - 1], radius=15, fill=(255, 255, 255, 255))
        base_image.paste(badge_img, (badge_x0, badge_y0), mask=badge_img)

        # Draw red chevrons '«' on the right side of the badge
        chev_x = badge_x1 - 25
        chev_y = (badge_y0 + badge_y1) // 2
        for offset in (0, 11):
            cx = chev_x - offset
            draw.line([(cx + 4, chev_y - 8), (cx - 4, chev_y), (cx + 4, chev_y + 8)], fill=NEWS_RED, width=4)

        draw.text(
            (badge_x1 - 48, chev_y - 2),
            formatted_badge,
            font=font_badge,
            fill=(18, 22, 30),
            anchor="rm",
            direction=badge_dir,
        )

        # 6. Headline Text inside Red Box
        text_right_x = box_x1 - 48
        start_text_y = box_y0 + pad_y + (line_height // 2)
        for i, line in enumerate(lines):
            ly = start_text_y + (i * line_height)
            draw.text(
                (text_right_x, ly),
                line,
                font=font_title,
                fill=(255, 255, 255),
                anchor="rm",
                direction=head_dir,
            )

        # 7. Bottom Social Bar
        social_y = 1262
        icons_x = 75
        font_sicon = ImageFont.truetype(str(self.font_path), 14)
        cx = icons_x + 12
        for label in ["f", "𝕏", "ig", "▶", "●"]:
            draw.ellipse(
                [cx - 11, social_y + 16 - 11, cx + 11, social_y + 16 + 11],
                fill=(255, 255, 255, 30),
                outline=(255, 255, 255, 140),
                width=1,
            )
            draw.text((cx, social_y + 15), label, font=font_sicon, fill=(255, 255, 255, 220), anchor="mm")
            cx += 28

        font_social = ImageFont.truetype(str(self.font_path), 23)
        draw.text((cx + 8, social_y + 16), "TaizNews", font=font_social, fill=(240, 245, 255), anchor="lm")

        # 8. Save output
        if not output_path:
            timestamp = int(time.time())
            slug = (article.id if article else f"card_{timestamp}")[:16]
            output_file = self.output_dir / f"card_{slug}_{timestamp}.png"
        else:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

        base_image.save(output_file, "PNG", quality=95)
        LOGGER.info("Saved breaking news card to: %s", output_file)
        return str(output_file)
