"""Base map loader, Web-Mercator projection, Arabic shaping, and shared drawing helpers."""

from __future__ import annotations

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont, features

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
MAPS_DIR = BASE_DIR / "assets" / "maps"
FONTS_DIR = BASE_DIR / "assets" / "fonts"
FONT_PATH = FONTS_DIR / "Cairo-Bold.ttf"
TILE_CACHE_DIR = BASE_DIR / "data" / "tile_cache"

W, H = 1080, 1920
TILE = 256
DEFAULT_BBOX = (13.95, 43.05, 12.55, 44.55)

ESRI_SATELLITE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_ATTRIBUTION = "Source: Esri, Maxar, Earthstar Geographics"

# Gazetteer for Taiz city & districts (lat, lon)
GAZETTEER: Dict[str, Tuple[float, float]] = {
    "تعز": (13.5795, 44.0209),
    "مدينة تعز": (13.5795, 44.0209),
    "المخا": (13.3205, 43.2500),
    "الحوبان": (13.6267, 44.0722),
    "التعزية": (13.6267, 44.0722),
    "جبل صبر": (13.5186, 44.0083),
    "صبر الموادم": (13.5186, 44.0083),
    "الضباب": (13.5350, 43.9550),
    "جبل حبشي": (13.4950, 43.8900),
    "التربة": (13.2186, 44.1350),
    "حيفان": (13.2833, 44.2833),
    "مقبنة": (13.6700, 43.7500),
    "الوازعية": (13.1667, 43.7500),
    "ذباب": (12.9333, 43.4167),
    "باب المندب": (12.6600, 43.4400),
    "الصلو": (13.3667, 44.2167),
    "المعافر": (13.3833, 43.9167),
    "المواسط": (13.2800, 44.0900),
    "الشمايتين": (13.2500, 44.0833),
    "سامع": (13.3500, 44.1167),
    "صالة": (13.5850, 44.0400),
    "المظفر": (13.5780, 44.0050),
    "القاهرة": (13.5800, 44.0150),
    "موزع": (13.3100, 43.5100),
    "المسراخ": (13.4500, 44.0200),
    "دمنة خدير": (13.4450, 44.1850),
    "خدير": (13.4450, 44.1850),
    "ماوية": (13.5900, 44.3200),
    "شرعب": (13.7300, 43.8700),
    "شرعب الرونة": (13.7300, 43.8700),
    "شرعب السلام": (13.7500, 44.0000),
}


def get_map_font(size: int = 32) -> ImageFont.FreeTypeFont:
    """Load bold Arabic font safely across Windows and Linux runners."""
    for font_name in ("Cairo-Bold.ttf", "Cairo[slnt,wght].ttf", "Almarai-Bold.ttf", "Tajawal-Bold.ttf"):
        p = FONTS_DIR / font_name
        if p.exists() and p.stat().st_size > 10000:
            return ImageFont.truetype(str(p), size)
    for fallback in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
    ):
        if fallback.exists():
            return ImageFont.truetype(str(fallback), size)
    return ImageFont.load_default()


def lookup_location(name: str) -> Optional[Tuple[float, float]]:
    """Lookup approximate (lat, lon) coordinates in Taiz governorate."""
    clean_name = (name or "").strip()
    if not clean_name:
        return None
    if clean_name in GAZETTEER:
        return GAZETTEER[clean_name]
    for key, coords in GAZETTEER.items():
        if key in clean_name or clean_name in key:
            return coords
    return None


def latlon_to_world_px(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    """Convert geographic (lat, lon) to Web-Mercator pixel coordinates at `zoom`."""
    n = TILE * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


class BaseMap:
    """A stitched base image paired with the world-pixel offset of its top-left corner."""

    def __init__(
        self,
        image: Image.Image,
        zoom: int,
        origin_px: Tuple[float, float],
        attribution: str = "© OpenStreetMap contributors",
    ) -> None:
        self.image = image
        self.zoom = zoom
        self.origin = origin_px
        self.attribution = attribution

    def to_img_px(self, lat: float, lon: float) -> Tuple[float, float]:
        wx, wy = latlon_to_world_px(lat, lon, self.zoom)
        return wx - self.origin[0], wy - self.origin[1]


def load_base_map_from_meta(png_path: Path) -> BaseMap:
    """Load pre-rendered base map image and its paired JSON metadata."""
    png_path = Path(png_path)
    meta = json.loads(png_path.with_suffix(".json").read_text(encoding="utf-8"))
    img = Image.open(png_path).convert("RGB")
    return BaseMap(img, int(meta["zoom"]), tuple(meta["origin_px"]), meta.get("attribution", ""))


def _build_esri_satellite_map(
    out_png: Path,
    zoom: int = 12,
    bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
) -> Path:
    """Download and stitch Esri World Imagery satellite tiles (no API key needed)."""
    lat_t, lon_l, lat_b, lon_r = bbox
    x0, y0 = latlon_to_world_px(lat_t, lon_l, zoom)
    x1, y1 = latlon_to_world_px(lat_b, lon_r, zoom)
    ox, oy = int(math.floor(x0)), int(math.floor(y0))
    ex, ey = int(math.ceil(x1)), int(math.ceil(y1))

    tx0, ty0 = ox // TILE, oy // TILE
    tx1, ty1 = (ex - 1) // TILE, (ey - 1) // TILE
    coords = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]

    cache = TILE_CACHE_DIR / "esri-satellite" / str(zoom)
    cache.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "TaizNews-AI/2.1 (map-reel-builder)"

    def fetch_tile(c: Tuple[int, int]) -> Tuple[Tuple[int, int], Path]:
        tx, ty = c
        p = cache / f"{tx}_{ty}.png"
        if p.exists() and p.stat().st_size > 100:
            return c, p
        url = ESRI_SATELLITE_URL.format(z=zoom, x=tx, y=ty)
        for attempt in range(1, 4):
            try:
                resp = session.get(url, timeout=15)
                if resp.status_code == 200 and len(resp.content) > 100:
                    p.write_bytes(resp.content)
                    time.sleep(0.03)
                    return c, p
            except requests.RequestException:
                pass
            time.sleep(1.0 * attempt)
        raise RuntimeError(f"Failed downloading map tile {zoom}/{tx}/{ty}")

    LOGGER.info("Building satellite base map (%d tiles) at %s...", len(coords), out_png)
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE))
    with ThreadPoolExecutor(max_workers=6) as pool:
        for (tx, ty), p in pool.map(fetch_tile, coords):
            with Image.open(p) as tile_img:
                mosaic.paste(tile_img.convert("RGB"), ((tx - tx0) * TILE, (ty - ty0) * TILE))

    left, top = ox - tx0 * TILE, oy - ty0 * TILE
    final = mosaic.crop((left, top, left + (ex - ox), top + (ey - oy)))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    final.save(out_png, optimize=True)
    meta = {
        "zoom": zoom,
        "origin_px": [ox, oy],
        "size": list(final.size),
        "bbox": list(bbox),
        "provider": "esri-satellite",
        "attribution": ESRI_ATTRIBUTION,
    }
    out_png.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_png


def ensure_base_map(style: str = "satellite") -> Optional[Path]:
    """Locate the cached base map PNG+JSON or auto-build satellite tiles if absent."""
    sat_png = MAPS_DIR / "taiz_satellite_z12.png"
    sat_json = MAPS_DIR / "taiz_satellite_z12.json"
    dark_png = MAPS_DIR / "taiz_z12.png"
    dark_json = MAPS_DIR / "taiz_z12.json"

    if style == "satellite":
        if sat_png.exists() and sat_json.exists() and sat_png.stat().st_size > 50000:
            return sat_png
        try:
            print("🛰️ [خريطة الأساس] بناء خريطة الأقمار الصناعية لتعز تلقائياً (Esri World Imagery)...")
            return _build_esri_satellite_map(sat_png)
        except Exception as err:
            LOGGER.warning("Could not auto-build satellite map (%s), falling back to dark map.", err)
            if dark_png.exists() and dark_json.exists():
                return dark_png
            return None

    if dark_png.exists() and dark_json.exists():
        return dark_png
    if sat_png.exists() and sat_json.exists():
        return sat_png
    return None


def _ease(t: float) -> float:
    """Smoothstep ease-in-out in [0, 1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def _bounce(t: float) -> float:
    """0..1 with slight overshoot settle for pin drop."""
    t = max(0.0, min(1.0, t))
    if t < 0.7:
        return (t / 0.7) ** 2
    k = (t - 0.7) / 0.3
    return 1.0 - 0.12 * math.sin(k * math.pi)


def _fmt_ar(text: str) -> Tuple[str, str]:
    """Shape Arabic text safely depending on Pillow Raqm/HarfBuzz availability."""
    if features.check("raqm"):
        return text, "rtl"
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text)), "ltr"
    except Exception:
        return text, "rtl"


def _draw_pin(layer: Image.Image, cx: float, tip_y: float, scale: float = 1.0) -> None:
    d = ImageDraw.Draw(layer)
    r = 46 * scale
    head_cy = tip_y - 98 * scale
    d.polygon(
        [(cx - r * 0.8, head_cy + r * 0.55), (cx + r * 0.8, head_cy + r * 0.55), (cx, tip_y)],
        fill=(214, 24, 36, 255),
    )
    d.ellipse(
        [cx - r, head_cy - r, cx + r, head_cy + r],
        fill=(214, 24, 36, 255),
        outline=(255, 255, 255, 255),
        width=int(6 * scale),
    )
    d.ellipse(
        [cx - r * 0.38, head_cy - r * 0.38, cx + r * 0.38, head_cy + r * 0.38],
        fill=(255, 255, 255, 255),
    )


def _draw_top_badge(layer: Image.Image, alpha: int = 255) -> None:
    """Render urgent news channel badge at top right."""
    if alpha <= 0:
        return
    badge_font = get_map_font(32)
    d = ImageDraw.Draw(layer)
    badge_text = "عاجل | تعز نيوز"
    shaped, direction = _fmt_ar(badge_text)
    bb = d.textbbox((0, 0), shaped, font=badge_font, direction=direction)
    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
    pw, ph = bw + 70, bh + 26
    x0, y0 = W - pw - 45, 65
    d.rounded_rectangle(
        [x0, y0, x0 + pw, y0 + ph],
        radius=16,
        fill=(210, 20, 32, alpha),
        outline=(255, 255, 255, int(alpha * 0.9)),
        width=2,
    )
    dot_r = 6
    dot_cx = x0 + 26
    dot_cy = y0 + ph // 2
    d.ellipse([dot_cx - dot_r, dot_cy - dot_r, dot_cx + dot_r, dot_cy + dot_r], fill=(255, 255, 255, alpha))
    d.text(
        (x0 + (pw + 26) // 2, y0 + ph // 2 - 2),
        shaped,
        font=badge_font,
        fill=(255, 255, 255, alpha),
        anchor="mm",
        direction=direction,
    )


def _draw_attribution(img: Image.Image, text: str) -> None:
    """Credit text required by tile provider."""
    if not text:
        return
    d = ImageDraw.Draw(img, "RGBA")
    font = get_map_font(22)
    d.text((W - 24, H - 24), text, font=font, fill=(255, 255, 255, 170), anchor="rb")
