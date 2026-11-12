"""Autonomous Central Brain for self-learning, quota budgeting, and hourly editorial planning."""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.processing.history import (
    DEFAULT_HISTORY_PATH,
    YEMEN_TZ,
    get_today_publish_counts,
    infer_post_type,
    load_published_history,
    save_published_history,
)
from app.publisher.facebook import FacebookPublisher

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY_PATH = BASE_DIR / "data" / "brain_memory.json"
DEFAULT_VOICE_STATE_PATH = BASE_DIR / "data" / "voice_state.json"

# Operating windows in Yemen local time (UTC+3)
PEAK_HOURS = set(range(12, 16)) | set(range(18, 24))  # 12:00-15:59 & 18:00-23:59
QUIET_HOURS = set(range(2, 6))                        # 02:00-05:59 AM

DEFAULT_DAILY_CAPS: Dict[str, int] = {
    "max_daily_reels": 4,
    "max_daily_photos": 10,
    "max_daily_text_satire": 8,
    "max_daily_yemen_trends": 2,
}

DEFAULT_FORMATS = ("text_satire", "photo_card", "standard_reel", "map_reel")
DEFAULT_CATEGORIES = ("عاجل", "أمن", "خدمات", "اقتصاد", "سياسة", "مجتمع")
DEFAULT_VOICES = ("adam", "maryam", "saleh", "ali", "zariyah")
ACTIVE_VOICES = ("maryam", "saleh", "ali", "zariyah", "adam")

VOICE_ALIASES: Dict[str, str] = {
    "adam": "adam",
    "pninz6obpgdqgcfmajgb": "adam",
    "maryam": "maryam",
    "ar-ye-maryamneural": "maryam",
    "saleh": "saleh",
    "ar-ye-salehneural": "saleh",
    "ali": "ali",
    "ar-bh-alineural": "ali",
    "zariyah": "zariyah",
    "ar-sa-zariyahneural": "zariyah",
}


def _empty_perf_bucket() -> Dict[str, Any]:
    return {
        "count": 0,
        "total_engagement": 0,
        "avg_engagement": 0.0,
        "weight": 1.0,
    }


def _parse_iso_utc(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


class CentralBrain:
    """Self-learning orchestrator managing engagement feedback, daily quotas, and hourly run plans."""

    def __init__(
        self,
        memory_path: Path | str = DEFAULT_MEMORY_PATH,
        history_path: Path | str = DEFAULT_HISTORY_PATH,
        voice_state_path: Path | str = DEFAULT_VOICE_STATE_PATH,
    ) -> None:
        self.memory_path = Path(memory_path)
        self.history_path = Path(history_path)
        self.voice_state_path = Path(voice_state_path)
        self._ensure_memory()

    def _build_default_memory(self) -> Dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "version": 1,
            "created_at": now_iso,
            "updated_at": now_iso,
            "last_sync_at": None,
            "total_posts_analyzed": 0,
            "daily_caps": dict(DEFAULT_DAILY_CAPS),
            "format_performance": {k: _empty_perf_bucket() for k in DEFAULT_FORMATS},
            "category_performance": {k: _empty_perf_bucket() for k in DEFAULT_CATEGORIES},
            "voice_performance": {k: _empty_perf_bucket() for k in DEFAULT_VOICES},
            "hourly_engagement": {},
            "learnings_summary": "الذاكرة في وضع التهيئة الأولي؛ بانتظار جمع قياسات التفاعل من فيسبوك.",
        }

    def _ensure_memory(self) -> None:
        if not self.memory_path.exists():
            self.save_memory(self._build_default_memory())

    def load_memory(self) -> Dict[str, Any]:
        """Load brain memory from JSON, filling any missing schema defaults safely."""
        defaults = self._build_default_memory()
        if not self.memory_path.exists():
            self.save_memory(defaults)
            return defaults

        try:
            raw = json.loads(self.memory_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raw = {}
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Could not parse brain memory file, reinitializing: %s", exc)
            self.save_memory(defaults)
            return defaults

        modified = False
        for top_key, default_val in defaults.items():
            if top_key not in raw:
                raw[top_key] = default_val
                modified = True

        # Ensure required subkeys exist inside caps and performance dictionaries
        if not isinstance(raw.get("daily_caps"), dict):
            raw["daily_caps"] = dict(DEFAULT_DAILY_CAPS)
            modified = True
        else:
            for cap_k, cap_v in DEFAULT_DAILY_CAPS.items():
                if cap_k not in raw["daily_caps"]:
                    raw["daily_caps"][cap_k] = cap_v
                    modified = True

        for group_key, expected_keys in (
            ("format_performance", DEFAULT_FORMATS),
            ("category_performance", DEFAULT_CATEGORIES),
            ("voice_performance", DEFAULT_VOICES),
        ):
            group = raw.setdefault(group_key, {})
            if not isinstance(group, dict):
                raw[group_key] = {k: _empty_perf_bucket() for k in expected_keys}
                modified = True
                continue
            for k in expected_keys:
                if k not in group or not isinstance(group[k], dict):
                    group[k] = _empty_perf_bucket()
                    modified = True

        if modified:
            self.save_memory(raw)

        return raw

    def save_memory(self, memory: Dict[str, Any]) -> None:
        """Persist brain memory atomically with UTF-8 encoding."""
        try:
            memory["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self.memory_path.with_suffix(".tmp")
            temp_file.write_text(json.dumps(memory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            temp_file.replace(self.memory_path)
        except OSError as exc:
            LOGGER.error("Failed to save brain memory to %s: %s", self.memory_path, exc)

    @staticmethod
    def _normalize_voice_key(voice_id: str) -> str:
        clean = (voice_id or "").strip().lower()
        if not clean:
            return ""
        return VOICE_ALIASES.get(clean, clean)

    @staticmethod
    def _update_perf_bucket(
        group: Dict[str, Dict[str, Any]],
        key: str,
        new_score: int,
        old_score: Optional[int],
    ) -> None:
        if not key:
            return
        bucket = group.setdefault(key, _empty_perf_bucket())
        if old_score is None:
            bucket["count"] = int(bucket.get("count", 0)) + 1
            bucket["total_engagement"] = max(0, int(bucket.get("total_engagement", 0)) + new_score)
        else:
            delta = new_score - int(old_score)
            bucket["total_engagement"] = max(0, int(bucket.get("total_engagement", 0)) + delta)
            if int(bucket.get("count", 0)) == 0:
                bucket["count"] = 1

        count = int(bucket["count"])
        bucket["avg_engagement"] = round(bucket["total_engagement"] / count, 2) if count > 0 else 0.0

    @staticmethod
    def _recalculate_weights(group: Dict[str, Dict[str, Any]]) -> None:
        observed = [float(b.get("avg_engagement", 0.0)) for b in group.values() if int(b.get("count", 0)) > 0]
        global_avg = (sum(observed) / len(observed)) if observed else 0.0
        for bucket in group.values():
            count = int(bucket.get("count", 0))
            if count <= 0 or global_avg <= 0.0:
                bucket["weight"] = 1.0
            else:
                ratio = float(bucket.get("avg_engagement", 0.0)) / global_avg
                bucket["weight"] = round(min(2.0, max(0.5, ratio)), 3)

    @staticmethod
    def _top_performer(group: Dict[str, Dict[str, Any]], min_count: int = 1) -> Optional[Tuple[str, float]]:
        candidates = [
            (k, float(v.get("avg_engagement", 0.0)))
            for k, v in group.items()
            if int(v.get("count", 0)) >= min_count
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])

    def _build_learnings_summary(self, memory: Dict[str, Any]) -> str:
        top_fmt = self._top_performer(memory.get("format_performance", {}))
        top_cat = self._top_performer(memory.get("category_performance", {}))
        top_voice = self._top_performer(memory.get("voice_performance", {}))
        analyzed = int(memory.get("total_posts_analyzed", 0))
        if analyzed == 0:
            return "الذاكرة في وضع التهيئة الأولي؛ بانتظار جمع قياسات التفاعل من فيسبوك."
        parts = [f"تم تحليل {analyzed} منشوراً"]
        if top_fmt:
            parts.append(f"أفضل صيغة: {top_fmt[0]} (متوسط {top_fmt[1]:.1f})")
        if top_cat:
            parts.append(f"أعلى تصنيف تفاعلاً: {top_cat[0]} ({top_cat[1]:.1f})")
        if top_voice:
            parts.append(f"أفضل صوت: {top_voice[0]} ({top_voice[1]:.1f})")
        return " | ".join(parts)

    def sync_and_learn_from_facebook(
        self,
        facebook_publisher: Optional[FacebookPublisher] = None,
        max_requests: int = 5,
        dry_run: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Fetch recent post metrics from Facebook and update self-learning averages."""
        publisher = facebook_publisher or FacebookPublisher(dry_run=dry_run)
        if dry_run or getattr(publisher, "dry_run", False):
            return {"synced_count": 0, "requests_made": 0, "skipped_reason": "dry_run"}
        if not getattr(publisher, "access_token", ""):
            return {"synced_count": 0, "requests_made": 0, "skipped_reason": "missing_token"}

        history = load_published_history(self.history_path)
        if not history:
            return {"synced_count": 0, "requests_made": 0, "skipped_reason": "empty_history"}

        now_utc = datetime.now(timezone.utc)
        memory = self.load_memory()
        requests_made = 0
        synced_count = 0
        history_modified = False

        for entry in history[:20]:
            if requests_made >= max_requests:
                break

            post_id = str(entry.get("post_id") or "").strip()
            if not post_id or "mock" in post_id.lower():
                continue

            pub_dt = _parse_iso_utc(entry.get("published_at"))
            if not pub_dt or (now_utc - pub_dt) <= timedelta(hours=2):
                continue

            updated_dt = _parse_iso_utc(entry.get("metrics_updated_at"))
            if updated_dt and (now_utc - updated_dt) <= timedelta(hours=6):
                continue

            requests_made += 1
            metrics = publisher.fetch_post_metrics(post_id)
            if not metrics:
                continue

            old_metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else None
            old_score = int(old_metrics.get("engagement_score", 0)) if old_metrics is not None else None
            new_score = int(metrics.get("engagement_score", 0))

            post_type = infer_post_type(entry)
            category = str(entry.get("category") or "مجتمع").strip() or "مجتمع"
            voice_key = self._normalize_voice_key(str(entry.get("voice_id") or ""))
            hour_ye = f"{pub_dt.astimezone(YEMEN_TZ).hour:02d}"

            self._update_perf_bucket(memory["format_performance"], post_type, new_score, old_score)
            self._update_perf_bucket(memory["category_performance"], category, new_score, old_score)
            if voice_key:
                self._update_perf_bucket(memory["voice_performance"], voice_key, new_score, old_score)
            self._update_perf_bucket(memory.setdefault("hourly_engagement", {}), hour_ye, new_score, old_score)

            if old_score is None:
                memory["total_posts_analyzed"] = int(memory.get("total_posts_analyzed", 0)) + 1

            entry["metrics"] = metrics
            entry["metrics_updated_at"] = now_utc.isoformat()
            history_modified = True
            synced_count += 1

        if history_modified:
            for group_key in ("format_performance", "category_performance", "voice_performance"):
                self._recalculate_weights(memory[group_key])
            memory["last_sync_at"] = now_utc.isoformat()
            memory["learnings_summary"] = self._build_learnings_summary(memory)
            save_published_history(history, self.history_path)
            self.save_memory(memory)

        return {
            "synced_count": synced_count,
            "requests_made": requests_made,
            "skipped_reason": None,
        }

    def _read_last_voice_key(self) -> str:
        """Read normalized key of the last used voice from voice_state.json."""
        try:
            if self.voice_state_path.exists():
                state = json.loads(self.voice_state_path.read_text(encoding="utf-8"))
                raw_last = str(state.get("last_voice_key") or state.get("last_voice_id") or "")
                return self._normalize_voice_key(raw_last)
        except (OSError, json.JSONDecodeError):
            pass
        return ""

    @staticmethod
    def _next_in_rotation(candidates: List[str], last_key: str, fallback: str) -> str:
        """Return the next candidate after last_key in list order."""
        if not candidates:
            return fallback
        if last_key in candidates:
            return candidates[(candidates.index(last_key) + 1) % len(candidates)]
        return candidates[0]

    def _select_preferred_voice(self, memory: Dict[str, Any]) -> str:
        """Select voice using initial exploration (untried count==0) then 70% exploit / 30% explore."""
        has_elevenlabs = bool(os.getenv("ELEVENLABS_API_KEY", "").strip()) and (
            os.getenv("ENABLE_ELEVENLABS", "true").strip().lower() not in ("false", "0", "no")
        )
        available_voices = [v for v in ACTIVE_VOICES if v != "adam" or has_elevenlabs]
        if not available_voices:
            available_voices = ["maryam", "saleh", "ali", "zariyah"]

        voice_perf = memory.get("voice_performance", {})
        last_key = self._read_last_voice_key()

        # Phase 1: Initial exploration — prioritize untried voices (count == 0) in rotation
        untried = [v for v in available_voices if int(voice_perf.get(v, {}).get("count", 0)) == 0]
        if untried:
            if last_key in available_voices and len(untried) > 1:
                start_idx = (available_voices.index(last_key) + 1) % len(available_voices)
                rotated_order = available_voices[start_idx:] + available_voices[:start_idx]
                for v in rotated_order:
                    if v in untried:
                        return v
            return untried[0]

        # Phase 2: Exploitation (70%) vs Exploration (30%)
        best_voice = max(
            available_voices,
            key=lambda v: (
                float(voice_perf.get(v, {}).get("weight", 1.0)),
                float(voice_perf.get(v, {}).get("avg_engagement", 0.0)),
            ),
        )
        if random.random() < 0.70:
            return best_voice

        alternatives = [v for v in available_voices if v != best_voice]
        return self._next_in_rotation(alternatives, last_key, best_voice)

    def _sort_formats_by_memory(
        self,
        formats: List[str],
        memory: Dict[str, Any],
    ) -> List[str]:
        """Preserve strategic priority order on ties, nudging higher-performing formats forward."""
        fmt_perf = memory.get("format_performance", {})
        base_order = {fmt: idx for idx, fmt in enumerate(formats)}
        return sorted(
            formats,
            key=lambda fmt: (
                -float(fmt_perf.get(fmt, {}).get("weight", 1.0)),
                base_order.get(fmt, 99),
            ),
        )

    def _build_directives_prompt(self, plan: Dict[str, Any], memory: Dict[str, Any]) -> str:
        """Build a concise Arabic strategic brief for Gemini editorial prompting."""
        hour_ye = plan["current_hour_ye"]
        if plan["is_quiet_hour"]:
            window_label = "ساعات الفجر الميتة (02:00 - 05:59 بتوقيت اليمن)"
        elif plan["is_peak_hour"]:
            window_label = "وقت ذروة المشاهدة والتفاعل بتوقيت اليمن"
        else:
            window_label = "ساعات النهار العادية (خارج ذروة الريلز)"

        counts = plan["today_counts"]
        caps = plan["daily_caps"]
        formats_str = "، ".join(plan["allowed_formats"]) if plan["allowed_formats"] else "لا توجد صيغ متاحة (تخطي)"
        yemen_note = (
            "مسموح بحد أقصى خبر ترند يمني واحد خارج تعز إن كان مهماً."
            if plan["allow_yemen_trend"]
            else "ممنوع نشر أخبار من خارج تعز في هذه الجولة."
        )

        lines = [
            f"توجيهات العقل المركزي (الساعة {hour_ye:02d}:00 بتوقيت اليمن — {window_label}):",
            f"- الحالة: {plan['decision']} | الحد الأدنى لدرجة الأهمية: {plan['min_importance_score']}/100",
            f"- الصيغ المسموحة حسب الأولوية: [{formats_str}]",
            (
                f"- استهلاك اليوم: الريلز {counts['total_reels']}/{caps['max_daily_reels']} | "
                f"البطاقات {counts['photo_card']}/{caps['max_daily_photos']} | "
                f"النصوص الساخرة {counts['text_satire']}/{caps['max_daily_text_satire']} | "
                f"ترند اليمن {counts['yemen_trend_count']}/{caps['max_daily_yemen_trends']}"
            ),
            f"- نطاق التغطية: {yemen_note}",
            f"- الصوت المقترح للريلز: {plan['preferred_voice']}",
            f"- خلاصة التعلم الذاتي: {memory.get('learnings_summary', '')}",
        ]
        return "\n".join(lines)

    def plan_hourly_run(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Evaluate Yemen local time, daily quotas, and learned weights to plan the run."""
        memory = self.load_memory()
        caps = memory.get("daily_caps", DEFAULT_DAILY_CAPS)
        max_reels = int(caps.get("max_daily_reels", 4))
        max_photos = int(caps.get("max_daily_photos", 10))
        max_satire = int(caps.get("max_daily_text_satire", 8))
        max_yemen = int(caps.get("max_daily_yemen_trends", 2))

        now_ye = (now or datetime.now(YEMEN_TZ)).astimezone(YEMEN_TZ)
        hour = now_ye.hour
        is_peak_hour = hour in PEAK_HOURS
        is_quiet_hour = hour in QUIET_HOURS

        today_counts = get_today_publish_counts(self.history_path)
        remaining_quotas = {
            "reels_remaining": max(0, max_reels - today_counts["total_reels"]),
            "photos_remaining": max(0, max_photos - today_counts["photo_card"]),
            "text_satire_remaining": max(0, max_satire - today_counts["text_satire"]),
            "yemen_trends_remaining": max(0, max_yemen - today_counts["yemen_trend_count"]),
        }

        reels_rem = remaining_quotas["reels_remaining"]
        photos_rem = remaining_quotas["photos_remaining"]
        satire_rem = remaining_quotas["text_satire_remaining"]
        yemen_rem = remaining_quotas["yemen_trends_remaining"]

        allowed_formats: List[str] = []
        allow_urgent_override = False

        if is_quiet_hour:
            # Dawn quiet hours (02:00-05:59): skip routine publishing; allow urgent >= 92 only
            if photos_rem > 0:
                allowed_formats.append("photo_card")
            if reels_rem > 0:
                allowed_formats.append("standard_reel")
            should_run = False
            allow_urgent_override = len(allowed_formats) > 0
            decision = "URGENT_ONLY" if allow_urgent_override else "SKIP"
            min_importance_score = 92
            allow_yemen_trend = False
        elif is_peak_hour and reels_rem > 0:
            # Peak hours with remaining reel budget: prioritize video reels first
            reel_tier = self._sort_formats_by_memory(["standard_reel", "map_reel"], memory)
            secondary_tier: List[str] = []
            if photos_rem > 0:
                secondary_tier.append("photo_card")
            if satire_rem > 0:
                secondary_tier.append("text_satire")
            secondary_tier = self._sort_formats_by_memory(secondary_tier, memory)
            allowed_formats = reel_tier + secondary_tier
            should_run = True
            decision = "PUBLISH"
            min_importance_score = 65
            allow_yemen_trend = yemen_rem > 0
        else:
            # Off-peak hours OR daily reel cap reached (4/4): focus on photo_card & text_satire
            non_reel: List[str] = []
            if photos_rem > 0:
                non_reel.append("photo_card")
            if satire_rem > 0:
                non_reel.append("text_satire")
            allowed_formats = self._sort_formats_by_memory(non_reel, memory)
            should_run = len(allowed_formats) > 0
            decision = "PUBLISH" if should_run else "SKIP"
            min_importance_score = 70
            allow_yemen_trend = yemen_rem > 0 and should_run

        top_fmt = self._top_performer(memory.get("format_performance", {}))
        top_cat = self._top_performer(memory.get("category_performance", {}))
        preferred_voice = self._select_preferred_voice(memory)

        plan: Dict[str, Any] = {
            "should_run": should_run,
            "decision": decision,
            "allow_urgent_override": allow_urgent_override,
            "current_time_ye": now_ye.isoformat(),
            "current_hour_ye": hour,
            "is_peak_hour": is_peak_hour,
            "is_quiet_hour": is_quiet_hour,
            "allowed_formats": allowed_formats,
            "allow_yemen_trend": allow_yemen_trend,
            "max_yemen_trends_this_run": 1 if allow_yemen_trend else 0,
            "preferred_voice": preferred_voice,
            "min_importance_score": min_importance_score,
            "emergency_reel_min_score": 95,
            "today_counts": today_counts,
            "daily_caps": dict(caps),
            "remaining_quotas": remaining_quotas,
            "top_performing_format": top_fmt[0] if top_fmt else None,
            "top_performing_category": top_cat[0] if top_cat else None,
            "brain_directives_prompt": "",
        }
        plan["brain_directives_prompt"] = self._build_directives_prompt(plan, memory)
        return plan
