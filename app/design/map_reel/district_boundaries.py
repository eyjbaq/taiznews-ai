"""Taiz District Boundaries Loader & High-End Visual Renderer.

Loads district polygons from assets/maps/taiz_districts.json and provides:
  1. Multi-lingual fuzzy matching (Arabic district names & landmarks -> GeoJSON features).
  2. Geographic coordinate to map pixel conversion.
  3. News-broadcast boundary rendering (spotlight dimming + animated laser trace + neon glow).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
PRIMARY_DATA_PATH = BASE_DIR / "assets" / "maps" / "taiz_districts.json"
FALLBACK_DATA_PATH = BASE_DIR / "sandbox" / "map" / "data" / "taiz_districts.json"

# Comprehensive Arabic & English alias map for Taiz districts
DISTRICT_ALIASES: Dict[str, str] = {
    # Al Makha / Western Coast
    "المخا": "Al Makha",
    "مخا": "Al Makha",
    "ميناء المخا": "Al Makha",
    "الساحل الغربي": "Al Makha",
    "al makha": "Al Makha",
    "al mokha": "Al Makha",
    "mokha": "Al Makha",

    # Maqbanah
    "مقبنة": "Maqbanah",
    "البرح": "Maqbanah",
    "شمير": "Maqbanah",
    "maqbanah": "Maqbanah",

    # Bab Al-Mandab / Dhubab
    "ذباب": "Dhubab",
    "باب المندب": "Dhubab",
    "dhubab": "Dhubab",

    # Mawza'
    "موزع": "Mawza'",
    "mawza": "Mawza'",

    # Al Wazi'yah
    "الوازعية": "Al Wazi'yah",
    "al waziyah": "Al Wazi'yah",

    # Jabal Habashi / Al Dhabab
    "جبل حبشي": "Jabal Habashi",
    "الضباب": "Jabal Habashi",
    "حبشي": "Jabal Habashi",
    "jabal habashi": "Jabal Habashi",

    # Sabir
    "صبر": "Sabir Al Mawadim",
    "جبل صبر": "Sabir Al Mawadim",
    "صبر الموادم": "Sabir Al Mawadim",
    "الموادم": "Sabir Al Mawadim",
    "sabir": "Sabir Al Mawadim",

    # Mashra'ah Wa Hadnan
    "مشرعة وحدنان": "Mashr'ah Wa Hadnan",
    "مشرعة": "Mashr'ah Wa Hadnan",

    # Al Misrakh
    "المسراخ": "Al Misrakh",
    "طالوق": "Al Misrakh",

    # Al Ma'afer / Al Nashamah
    "المعافر": "Al Ma'afer",
    "النشمة": "Al Ma'afer",
    "al maafer": "Al Ma'afer",

    # Al Mawasit / Al Ain
    "المواسط": "Al Mawasit",
    "العين": "Al Mawasit",

    # Ash Shamayatayn / At Turbah
    "الشمايتين": "Ash Shamayatayn",
    "التربة": "Ash Shamayatayn",
    "ذبحان": "Ash Shamayatayn",
    "turbah": "Ash Shamayatayn",

    # As Silw
    "الصلو": "As Silw",
    "silw": "As Silw",

    # Hayfan
    "حيفان": "Hayfan",
    "الأعبوس": "Hayfan",
    "الأثاور": "Hayfan",
    "hayfan": "Hayfan",

    # Dimnat Khadir
    "دمنة خدير": "Dimnat Khadir",
    "خدير": "Dimnat Khadir",

    # Sami'
    "سامع": "Sami'",

    # City districts
    "مدينة تعز": "CITY_COMBINED",
    "تعز": "CITY_COMBINED",
    "المظفر": "Al Mudhaffar",
    "القاهرة": "Al Qahirah",
    "صالة": "Salah",

    # At Ta'iziyah / Al Hawban
    "الحوبان": "At Ta'iziyah",
    "التعزية": "At Ta'iziyah",
    "at taiziyah": "At Ta'iziyah",

    # Mawiyah
    "ماوية": "Mawiyah",

    # Shar'ab
    "شرعب الرونة": "Shar'ab Ar Rawnah",
    "شرعب السلام": "Shar'ab As Salam",
    "شرعب": "Shar'ab Ar Rawnah",
}


class DistrictBoundaryManager:
    """Manages spatial lookup and visual drawing of district boundaries."""

    def __init__(self, geojson_path: Optional[Path] = None) -> None:
        if geojson_path:
            self.geojson_path = Path(geojson_path)
        elif PRIMARY_DATA_PATH.exists():
            self.geojson_path = PRIMARY_DATA_PATH
        else:
            self.geojson_path = FALLBACK_DATA_PATH
        self.features_by_name: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.geojson_path.exists():
            LOGGER.warning("Districts GeoJSON not found at: %s", self.geojson_path)
            return

        with open(self.geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for feat in data.get("features", []):
            name = feat.get("properties", {}).get("shapeName", "").strip()
            if name:
                self.features_by_name[name.lower()] = feat

    def find_district(self, query: str) -> List[dict]:
        """Find matching GeoJSON district feature(s) from Arabic or English name."""
        clean = (query or "").strip().lower()
        if not clean:
            return []

        canonical = DISTRICT_ALIASES.get(query.strip()) or DISTRICT_ALIASES.get(clean)
        if not canonical:
            # Try substring match against aliases (longer keys first)
            for alias in sorted(DISTRICT_ALIASES.keys(), key=len, reverse=True):
                if alias.lower() in clean:
                    canonical = DISTRICT_ALIASES[alias]
                    break

        if canonical == "CITY_COMBINED":
            res = []
            for city_d in ("al mudhaffar", "al qahirah", "salah"):
                if city_d in self.features_by_name:
                    res.append(self.features_by_name[city_d])
            return res

        if canonical:
            c_key = canonical.lower()
            if c_key in self.features_by_name:
                return [self.features_by_name[c_key]]

        for k, feat in self.features_by_name.items():
            if clean in k or k in clean:
                return [feat]

        return []

    def extract_polygons(self, feature: dict) -> List[List[Tuple[float, float]]]:
        """Extract lists of (lon, lat) rings from Polygon or MultiPolygon geometry."""
        geom = feature.get("geometry", {})
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])

        polygons: List[List[Tuple[float, float]]] = []
        if gtype == "Polygon":
            for ring in coords:
                polygons.append([(float(pt[0]), float(pt[1])) for pt in ring])
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    polygons.append([(float(pt[0]), float(pt[1])) for pt in ring])
        return polygons

    def get_district_centroid(self, features: List[dict]) -> Optional[Tuple[float, float]]:
        """Return (lat, lon) geographic center of the district features."""
        if not features:
            return None
        for feat in features:
            props = feat.get("properties", {})
            if "center" in props and len(props["center"]) == 2:
                c = props["center"]
                return float(c[0]), float(c[1])
        lats, lons = [], []
        for feat in features:
            for ring in self.extract_polygons(feat):
                for lon, lat in ring:
                    lats.append(lat)
                    lons.append(lon)
        if lats and lons:
            return sum(lats) / len(lats), sum(lons) / len(lons)
        return None

    def calculate_optimal_framing(
        self,
        features: List[dict],
        base_map: Any,
        screen_w: int = 1080,
        screen_h: int = 1920,
        fill_ratio_w: float = 0.82,
        fill_ratio_h: float = 0.52,
    ) -> Tuple[float, float, float]:
        """Calculate optimal center (cx, cy) in image pixels and optimal zoom scale to frame the district."""
        bw, bh = base_map.image.size
        start_scale = max(screen_w / bw, screen_h / bh)
        if not features:
            return bw / 2, bh / 2, start_scale * 1.5

        all_px, all_py = [], []
        for feat in features:
            for ring in self.extract_polygons(feat):
                for lon, lat in ring:
                    ix, iy = base_map.to_img_px(lat, lon)
                    all_px.append(ix)
                    all_py.append(iy)

        if not all_px:
            return bw / 2, bh / 2, start_scale * 1.5

        min_x, max_x = min(all_px), max(all_px)
        min_y, max_y = min(all_py), max(all_py)

        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        pw = max(10.0, max_x - min_x)
        ph = max(10.0, max_y - min_y)

        safe_w = screen_w * fill_ratio_w
        safe_h = screen_h * fill_ratio_h

        optimal_scale = min(safe_w / pw, safe_h / ph)
        optimal_scale = max(start_scale * 1.15, min(optimal_scale, 2.5))
        return cx, cy, optimal_scale

    def draw_district_spotlight(
        self,
        layer: Image.Image,
        features: List[dict],
        base_map: Any,
        view_box: Tuple[float, float, float, float],
        scale: float,
        dim_alpha: int = 150,
    ) -> None:
        """Dim everything outside the district to highlight the targeted area."""
        if not features or dim_alpha <= 0:
            return

        W, H = layer.size
        box_x0, box_y0 = view_box[0], view_box[1]

        mask = Image.new("L", (W, H), min(255, max(0, dim_alpha)))
        m_draw = ImageDraw.Draw(mask)
        has_poly = False

        for feat in features:
            for ring in self.extract_polygons(feat):
                screen_pts = [
                    ((base_map.to_img_px(lat, lon)[0] - box_x0) * scale,
                     (base_map.to_img_px(lat, lon)[1] - box_y0) * scale)
                    for lon, lat in ring
                ]
                if len(screen_pts) >= 3:
                    m_draw.polygon(screen_pts, fill=0)
                    has_poly = True

        if has_poly:
            blurred_mask = mask.filter(ImageFilter.BoxBlur(4))
            dim_layer = Image.new("RGBA", (W, H), (4, 8, 14, 255))
            layer.paste(dim_layer, (0, 0), blurred_mask)

    def draw_district_boundary_animated(
        self,
        layer: Image.Image,
        features: List[dict],
        base_map: Any,
        view_box: Tuple[float, float, float, float],
        scale: float,
        trace_progress: float = 1.0,
        glow_alpha: float = 1.0,
        pulse_phase: float = 0.0,
        glow_color: Tuple[int, int, int] = (255, 45, 65),
        fill_color: Optional[Tuple[int, int, int]] = (255, 30, 50),
        draw_spark: bool = True,
    ) -> None:
        """Render animated laser perimeter trace or glowing multi-layer neon boundary."""
        if not features or (trace_progress <= 0.0 and glow_alpha <= 0.0):
            return

        box_x0, box_y0 = view_box[0], view_box[1]
        draw = ImageDraw.Draw(layer)

        for feat in features:
            for ring in self.extract_polygons(feat):
                screen_pts: List[Tuple[float, float]] = []
                for lon, lat in ring:
                    img_x, img_y = base_map.to_img_px(lat, lon)
                    screen_pts.append(((img_x - box_x0) * scale, (img_y - box_y0) * scale))

                if len(screen_pts) < 3:
                    continue

                all_pts = screen_pts + [screen_pts[0]]

                if trace_progress < 1.0:
                    dists = [0.0]
                    for i in range(len(all_pts) - 1):
                        d = math.hypot(all_pts[i + 1][0] - all_pts[i][0], all_pts[i + 1][1] - all_pts[i][1])
                        dists.append(dists[-1] + d)

                    total_len = dists[-1]
                    target_d = max(0.0, min(1.0, trace_progress)) * total_len

                    sub_pts = []
                    lead_pt = None
                    for i in range(len(dists) - 1):
                        if dists[i] <= target_d <= dists[i + 1]:
                            seg_len = dists[i + 1] - dists[i]
                            frac = (target_d - dists[i]) / seg_len if seg_len > 0 else 0.0
                            lx = all_pts[i][0] + (all_pts[i + 1][0] - all_pts[i][0]) * frac
                            ly = all_pts[i][1] + (all_pts[i + 1][1] - all_pts[i][1]) * frac
                            sub_pts = all_pts[: i + 1] + [(lx, ly)]
                            lead_pt = (lx, ly)
                            break

                    if len(sub_pts) >= 2:
                        draw.line(sub_pts, fill=(glow_color[0], glow_color[1], glow_color[2], 140), width=10)
                        draw.line(sub_pts, fill=(255, 255, 255, 240), width=4)

                    if lead_pt and draw_spark:
                        sp_x, sp_y = lead_pt
                        draw.ellipse(
                            [sp_x - 22, sp_y - 22, sp_x + 22, sp_y + 22],
                            fill=(glow_color[0], glow_color[1], glow_color[2], 190),
                        )
                        draw.ellipse([sp_x - 8, sp_y - 8, sp_x + 8, sp_y + 8], fill=(255, 255, 255, 255))
                else:
                    g_factor = max(0.0, min(1.0, glow_alpha))
                    pulse = 0.82 + 0.18 * pulse_phase

                    if fill_color:
                        f_alpha = int(62 * g_factor * pulse)
                        draw.polygon(screen_pts, fill=(fill_color[0], fill_color[1], fill_color[2], f_alpha))

                    draw.line(
                        all_pts,
                        fill=(glow_color[0], glow_color[1], glow_color[2], int(75 * g_factor * pulse)),
                        width=18,
                    )
                    draw.line(
                        all_pts,
                        fill=(glow_color[0], glow_color[1], glow_color[2], int(165 * g_factor * pulse)),
                        width=8,
                    )
                    draw.line(all_pts, fill=(255, 255, 255, int(255 * g_factor)), width=3)

    def draw_district_boundary(
        self,
        layer: Image.Image,
        features: List[dict],
        base_map: Any,
        view_box: Tuple[float, float, float, float],
        scale: float,
        progress: float = 1.0,
        glow_color: Tuple[int, int, int] = (255, 45, 65),
        fill_color: Optional[Tuple[int, int, int]] = (255, 30, 50),
        pulse_phase: float = 0.0,
    ) -> None:
        """Backward-compatible wrapper calling draw_district_boundary_animated."""
        self.draw_district_boundary_animated(
            layer=layer,
            features=features,
            base_map=base_map,
            view_box=view_box,
            scale=scale,
            trace_progress=progress,
            glow_alpha=progress,
            pulse_phase=pulse_phase,
            glow_color=glow_color,
            fill_color=fill_color,
        )


DISTRICT_MANAGER = DistrictBoundaryManager()
