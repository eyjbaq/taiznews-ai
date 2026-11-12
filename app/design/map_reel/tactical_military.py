"""Tactical Military & Conflict Visual Graphics for News Maps."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from PIL import Image, ImageDraw

from .base_map import _fmt_ar, get_map_font


def draw_explosion(
    layer: Image.Image,
    cx: float,
    cy: float,
    progress: float,
    max_radius: float = 130.0,
    label: Optional[str] = "قصف مباشر",
) -> None:
    """Render a dynamic multi-layered explosion with fiery shockwaves and fireball."""
    if progress < 0.0 or progress > 1.0:
        return

    d = ImageDraw.Draw(layer)
    p = progress

    ring_r = 15.0 + (max_radius * 1.1) * (p ** 0.7)
    ring_alpha = int(220 * (1.0 - p))
    if ring_alpha > 5:
        d.ellipse(
            [cx - ring_r, cy - ring_r * 0.65, cx + ring_r, cy + ring_r * 0.65],
            outline=(255, 90, 30, ring_alpha),
            width=5,
        )
        inner_r = ring_r * 0.65
        inner_a = int(255 * (1.0 - p ** 0.5))
        d.ellipse(
            [cx - inner_r, cy - inner_r * 0.65, cx + inner_r, cy + inner_r * 0.65],
            outline=(255, 230, 80, inner_a),
            width=3,
        )

    if p < 0.75:
        fire_p = p / 0.75
        fire_r = max_radius * 0.45 * math.sin(fire_p * math.pi)
        fire_a = int(255 * (1.0 - fire_p * 0.7))

        num_spikes = 10
        rot_offset = p * 1.2
        for i in range(num_spikes):
            angle = rot_offset + i * (2 * math.pi / num_spikes)
            sp_len = fire_r * (1.4 if i % 2 == 0 else 0.95)
            sp_x = cx + math.cos(angle) * sp_len
            sp_y = cy + math.sin(angle) * sp_len * 0.75
            d.line([(cx, cy), (sp_x, sp_y)], fill=(255, 50, 10, fire_a), width=6)
            d.line(
                [(cx, cy), (sp_x * 0.85 + cx * 0.15, sp_y * 0.85 + cy * 0.15)],
                fill=(255, 210, 40, fire_a),
                width=3,
            )

        d.ellipse(
            [cx - fire_r * 0.6, cy - fire_r * 0.45, cx + fire_r * 0.6, cy + fire_r * 0.45],
            fill=(255, 80, 20, fire_a),
            outline=(255, 240, 120, fire_a),
            width=3,
        )
        d.ellipse(
            [cx - fire_r * 0.25, cy - fire_r * 0.18, cx + fire_r * 0.25, cy + fire_r * 0.18],
            fill=(255, 255, 230, fire_a),
        )

    if label and 0.15 < p < 0.95:
        tag_alpha = int(255 * math.sin(((p - 0.15) / 0.8) * math.pi))
        font = get_map_font(22)
        clean_label = label.replace("💥", "").replace("⚠️", "").strip()
        shaped, direction = _fmt_ar(clean_label)
        bb = d.textbbox((0, 0), shaped, font=font, direction=direction)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        bx, by = cx - tw // 2 - 20, cy - max_radius * 0.85 - 28
        d.rounded_rectangle(
            [bx, by, bx + tw + 40, by + th + 12],
            radius=8,
            fill=(24, 10, 10, int(tag_alpha * 0.92)),
            outline=(255, 60, 40, tag_alpha),
            width=2,
        )
        d.ellipse(
            [bx + 12, by + (th + 12) // 2 - 4, bx + 20, by + (th + 12) // 2 + 4],
            fill=(255, 45, 65, tag_alpha),
        )
        d.text(
            (bx + 26 + tw // 2, by + th // 2 + 4),
            shaped,
            font=font,
            fill=(255, 230, 80, tag_alpha),
            anchor="mm",
            direction=direction,
        )


def draw_fighter_jet(
    layer: Image.Image,
    cx: float,
    cy: float,
    heading_deg: float,
    afterburner_phase: float = 0.0,
    scale: float = 1.0,
    label: Optional[str] = "طيران حربي",
) -> None:
    """Render a tactical military delta-wing jet with glowing afterburner exhaust."""
    d = ImageDraw.Draw(layer)
    rad = math.radians(heading_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    def rot(lx: float, ly: float) -> Tuple[float, float]:
        return cx + (lx * cos_a - ly * sin_a) * scale, cy + (lx * sin_a + ly * cos_a) * scale

    body_pts = [
        rot(48, 0), rot(22, 6), rot(5, 48), rot(-6, 44), rot(-14, 15),
        rot(-38, 12), rot(-46, 26), rot(-48, 0), rot(-46, -26), rot(-38, -12),
        rot(-14, -15), rot(-6, -44), rot(5, -48), rot(22, -6),
    ]

    flame_len = -48 - (24 + 14 * math.sin(afterburner_phase * 10.0))
    flame_tip = rot(flame_len, 0)
    d.polygon([rot(-46, -8), flame_tip, rot(-46, 8)], fill=(0, 220, 255, 230))
    d.polygon([rot(-46, -4), rot(flame_len * 0.7, 0), rot(-46, 4)], fill=(255, 255, 255, 255))

    shadow_offset = 24 * scale
    d.polygon([(x + 8, y + shadow_offset) for (x, y) in body_pts], fill=(0, 0, 0, 75))
    d.polygon(body_pts, fill=(235, 240, 250, 250), outline=(255, 45, 65, 255))
    d.polygon([rot(26, 0), rot(14, 4), rot(4, 0), rot(14, -4)], fill=(40, 140, 220, 255), outline=(255, 255, 255, 220))

    if label:
        font = get_map_font(20)
        clean_label = label.replace("✈️", "").strip()
        shaped, direction = _fmt_ar(clean_label)
        lx, ly = cx, cy - 55 * scale
        bb = d.textbbox((0, 0), shaped, font=font, direction=direction)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        d.rounded_rectangle(
            [lx - tw // 2 - 14, ly - th // 2 - 4, lx + tw // 2 + 14, ly + th // 2 + 4],
            radius=6,
            fill=(10, 15, 22, 215),
            outline=(0, 220, 255, 190),
            width=1,
        )
        d.text((lx, ly), shaped, font=font, fill=(255, 255, 255, 240), anchor="mm", direction=direction)


def draw_tank(
    layer: Image.Image,
    cx: float,
    cy: float,
    heading_deg: float,
    firing_phase: float = 0.0,
    scale: float = 1.0,
    label: Optional[str] = "دبابات ومدرعات",
) -> None:
    """Render a tactical ground combat armored tank with firing muzzle flash."""
    d = ImageDraw.Draw(layer)
    rad = math.radians(heading_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    def rot(lx: float, ly: float) -> Tuple[float, float]:
        return cx + (lx * cos_a - ly * sin_a) * scale, cy + (lx * sin_a + ly * cos_a) * scale

    d.ellipse([cx - 45 * scale, cy + 8 * scale, cx + 45 * scale, cy + 35 * scale], fill=(0, 0, 0, 90))

    for track_y in (-24, 24):
        track_box = [rot(-38, track_y - 7), rot(38, track_y - 7), rot(38, track_y + 7), rot(-38, track_y + 7)]
        d.polygon(track_box, fill=(35, 42, 52, 255), outline=(255, 190, 40, 220))
        for wx in (-26, -13, 0, 13, 26):
            w_pt = rot(wx, track_y)
            d.ellipse([w_pt[0] - 3, w_pt[1] - 3, w_pt[0] + 3, w_pt[1] + 3], fill=(180, 190, 205, 255))

    hull_pts = [rot(-34, -18), rot(34, -18), rot(40, -10), rot(40, 10), rot(34, 18), rot(-34, 18)]
    d.polygon(hull_pts, fill=(55, 68, 85, 255), outline=(20, 26, 34, 255), width=2)

    barrel_tip = rot(58, 0)
    barrel_base = rot(10, 0)
    d.line([barrel_base, barrel_tip], fill=(220, 225, 235, 255), width=int(6 * scale))

    turret_r = 18 * scale
    d.ellipse(
        [cx - turret_r, cy - turret_r, cx + turret_r, cy + turret_r],
        fill=(70, 85, 105, 255),
        outline=(255, 190, 40, 255),
        width=2,
    )
    hatch_pt = rot(-4, 0)
    d.ellipse([hatch_pt[0] - 6, hatch_pt[1] - 6, hatch_pt[0] + 6, hatch_pt[1] + 6], fill=(30, 38, 48, 255))

    flash_val = math.sin(firing_phase * 6.0)
    if flash_val > 0.75:
        f_strength = (flash_val - 0.75) / 0.25
        flash_cx, flash_cy = barrel_tip
        flash_r = 22 * f_strength * scale
        d.ellipse(
            [flash_cx - flash_r, flash_cy - flash_r * 0.7, flash_cx + flash_r, flash_cy + flash_r * 0.7],
            fill=(255, 240, 80, int(240 * f_strength)),
            outline=(255, 60, 20, 255),
            width=2,
        )
        d.ellipse(
            [flash_cx - flash_r * 0.4, flash_cy - flash_r * 0.3, flash_cx + flash_r * 0.4, flash_cy + flash_r * 0.3],
            fill=(255, 255, 255, 255),
        )

    if label:
        font = get_map_font(20)
        clean_label = label.replace("🛡️", "").strip()
        shaped, direction = _fmt_ar(clean_label)
        lx, ly = cx, cy + 46 * scale
        bb = d.textbbox((0, 0), shaped, font=font, direction=direction)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        d.rounded_rectangle(
            [lx - tw // 2 - 14, ly - th // 2 - 4, lx + tw // 2 + 14, ly + th // 2 + 4],
            radius=6,
            fill=(10, 15, 22, 215),
            outline=(255, 190, 40, 200),
            width=1,
        )
        d.text((lx, ly), shaped, font=font, fill=(255, 220, 80, 255), anchor="mm", direction=direction)


def draw_clash_frontline(
    layer: Image.Image,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    t: float,
    label: Optional[str] = "خط اشتباكات ناري",
) -> None:
    """Render an energetic flashing contested frontline with clash sparks."""
    d = ImageDraw.Draw(layer)
    x1, y1 = p1
    x2, y2 = p2

    pulse = 0.5 + 0.5 * math.sin(t * 8.0)
    line_alpha = int(180 + 75 * pulse)

    d.line([p1, p2], fill=(255, 45, 65, int(60 * pulse)), width=16)
    d.line([p1, p2], fill=(255, 45, 65, line_alpha), width=6)
    d.line([p1, p2], fill=(255, 255, 200, 240), width=2)

    frac = 0.5 + 0.45 * math.sin(t * 3.5)
    spark_x = x1 + (x2 - x1) * frac
    spark_y = y1 + (y2 - y1) * frac

    spark_r = 14 + 6 * math.sin(t * 12.0)
    d.ellipse(
        [spark_x - spark_r, spark_y - spark_r, spark_x + spark_r, spark_y + spark_r],
        fill=(255, 220, 50, 230),
        outline=(255, 50, 30, 255),
        width=2,
    )
    d.ellipse([spark_x - 5, spark_y - 5, spark_x + 5, spark_y + 5], fill=(255, 255, 255, 255))

    if label:
        font = get_map_font(22)
        mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0 - 24
        clean_label = label.replace("⚔️", "").strip()
        shaped, direction = _fmt_ar(clean_label)
        bb = d.textbbox((0, 0), shaped, font=font, direction=direction)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        d.rounded_rectangle(
            [mid_x - tw // 2 - 16, mid_y - th // 2 - 4, mid_x + tw // 2 + 16, mid_y + th // 2 + 4],
            radius=6,
            fill=(18, 12, 14, 220),
            outline=(255, 45, 65, 220),
            width=2,
        )
        d.text((mid_x, mid_y), shaped, font=font, fill=(255, 255, 255, 255), anchor="mm", direction=direction)


class TacticalMilitaryDirector:
    """Coordinates multi-stage animated military clash & airstrike events on the map."""

    def __init__(self, target_img_pt: Tuple[float, float]) -> None:
        self.tx, self.ty = target_img_pt

    def render(
        self,
        layer: Image.Image,
        box: Tuple[float, float, float, float],
        scale: float,
        t: float,
    ) -> None:
        """Render all active military operations mapped to the current camera viewport."""
        box_x0, box_y0 = box[0], box[1]

        def to_screen(ix: float, iy: float) -> Tuple[float, float]:
            return (ix - box_x0) * scale, (iy - box_y0) * scale

        # Event 1: Fighter Jet Airstrike & Impact
        if 3.0 <= t <= 7.2:
            jet_p = (t - 3.0) / 4.2
            jet_ix = (self.tx - 480) + 1050 * jet_p
            jet_iy = (self.ty - 380) + 720 * jet_p
            jet_sx, jet_sy = to_screen(jet_ix, jet_iy)
            heading = math.degrees(math.atan2(720, 1050))
            draw_fighter_jet(
                layer,
                cx=jet_sx,
                cy=jet_sy,
                heading_deg=heading,
                afterburner_phase=t,
                scale=1.05,
                label="مقاتلة F-15 | غارة جوية",
            )

        strike_sx, strike_sy = to_screen(self.tx + 50, self.ty - 60)
        if 5.1 <= t <= 7.8:
            exp_p = (t - 5.1) / 2.7
            draw_explosion(layer, strike_sx, strike_sy, exp_p, max_radius=150, label="غارة جوية | استهداف مباشر")

        # Event 2: Ground Armor & Contested Frontline Clash
        if t >= 4.0:
            fl_p1 = to_screen(self.tx - 240, self.ty + 160)
            fl_p2 = to_screen(self.tx + 140, self.ty + 260)
            draw_clash_frontline(layer, fl_p1, fl_p2, t, label="محور اشتباكات عنيفة")

            tank_sx, tank_sy = to_screen(self.tx + 220, self.ty + 220)
            draw_tank(layer, tank_sx, tank_sy, heading_deg=-155, firing_phase=t, scale=1.0, label="قوات برية ومدرعات")

        if t >= 6.8:
            cycle = (t - 6.8) % 2.5
            if cycle < 1.6:
                sh_p = cycle / 1.6
                sh_sx, sh_sy = to_screen(self.tx - 110, self.ty + 190)
                draw_explosion(layer, sh_sx, sh_sy, sh_p, max_radius=90, label="قصف مدفعي")

        # Event 3: Secondary Sortie & Precision Strike (for 25s+ reels)
        if 14.0 <= t <= 18.5:
            sort_p = (t - 14.0) / 4.5
            s_ix = (self.tx - 520) + 1100 * sort_p
            s_iy = (self.ty + 350) - 620 * sort_p
            s_sx, s_sy = to_screen(s_ix, s_iy)
            s_heading = math.degrees(math.atan2(-620, 1100))
            draw_fighter_jet(
                layer,
                cx=s_sx,
                cy=s_sy,
                heading_deg=s_heading,
                afterburner_phase=t * 1.3,
                scale=1.0,
                label="طلعة جوية ثانية | إسناد ناري",
            )

        if 16.2 <= t <= 19.2:
            s_exp_p = (t - 16.2) / 3.0
            s_target_sx, s_target_sy = to_screen(self.tx - 60, self.ty - 100)
            draw_explosion(layer, s_target_sx, s_target_sy, s_exp_p, max_radius=135, label="استهداف تعزيزات العدو")
