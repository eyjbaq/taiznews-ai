"""History manager for tracking published Facebook posts to avoid repetition."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.collector.base import Article
from app.ai.models import EditorialPost

LOGGER = logging.getLogger(__name__)

DEFAULT_HISTORY_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "published_posts.json"
YEMEN_TZ = timezone(timedelta(hours=3))
MAX_HISTORY_ENTRIES = 100


def load_published_history(path: Path = DEFAULT_HISTORY_PATH) -> List[Dict[str, Any]]:
    """Load list of published posts from history JSON."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as exc:
        LOGGER.warning("Could not read published history from %s: %s", path, exc)
    return []


def save_published_history(history: List[Dict[str, Any]], path: Path = DEFAULT_HISTORY_PATH) -> None:
    """Persist published post history safely with UTF-8 encoding."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(history[:MAX_HISTORY_ENTRIES], f, indent=2, ensure_ascii=False)
        temp_file.replace(path)
    except Exception as exc:
        LOGGER.error("Failed to write published history to %s: %s", path, exc)


def infer_post_type(entry: Dict[str, Any]) -> str:
    """Resolve post_type for both Phase 1+ entries and legacy history records."""
    explicit = str(entry.get("post_type") or "").strip()
    if explicit:
        return explicit
    post_url = str(entry.get("post_url") or "").lower()
    if "/reel/" in post_url or "watch/?v=" in post_url:
        return "standard_reel"
    return "photo_card"


def record_published_post(
    post: EditorialPost,
    article: Article,
    fb_result: Optional[Dict[str, Any]] = None,
    path: Path = DEFAULT_HISTORY_PATH,
    post_type: str = "photo_card",
    voice_id: str = "",
    scope: str = "taiz",
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a newly published post with format, voice, scope, and metrics fields."""
    history = load_published_history(path)
    now_iso = datetime.now(timezone.utc).isoformat()

    normalized_scope = (scope or "taiz").strip().lower()
    if normalized_scope not in ("taiz", "yemen_trend"):
        normalized_scope = "yemen_trend" if normalized_scope == "yemen" else "taiz"

    entry = {
        "published_at": now_iso,
        "headline": post.headline.strip(),
        "body": post.body.strip(),
        "category": post.category,
        "post_type": (post_type or "photo_card").strip(),
        "voice_id": (voice_id or "").strip(),
        "scope": normalized_scope,
        "metrics": metrics,
        "metrics_updated_at": now_iso if metrics is not None else None,
        "article_title": article.title,
        "article_url": article.url,
        "article_source": article.source,
        "post_id": (fb_result or {}).get("post_id", ""),
        "photo_id": (fb_result or {}).get("id", ""),
        "post_url": (fb_result or {}).get("url", ""),
    }

    # Keep latest entries at the front and retain up to MAX_HISTORY_ENTRIES (100)
    history.insert(0, entry)
    history = history[:MAX_HISTORY_ENTRIES]

    save_published_history(history, path)
    LOGGER.info("Recorded post to published history (%s): %s", entry["post_type"], post.headline)


def get_today_publish_counts(path: Path = DEFAULT_HISTORY_PATH) -> Dict[str, int]:
    """Count posts published during the current Yemen day (UTC+3) by post_type and scope."""
    counts: Dict[str, int] = {
        "text_satire": 0,
        "photo_card": 0,
        "standard_reel": 0,
        "map_reel": 0,
        "total_reels": 0,
        "yemen_trend_count": 0,
        "total_posts": 0,
    }
    today_ye = datetime.now(YEMEN_TZ).date()
    history = load_published_history(path)

    for entry in history:
        pub_str = entry.get("published_at")
        if not pub_str:
            continue
        try:
            pub_dt = datetime.fromisoformat(str(pub_str))
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            if pub_dt.astimezone(YEMEN_TZ).date() != today_ye:
                continue
        except (TypeError, ValueError):
            continue

        ptype = infer_post_type(entry)
        counts[ptype] = counts.get(ptype, 0) + 1
        if ptype in ("standard_reel", "map_reel"):
            counts["total_reels"] += 1

        scope = str(entry.get("scope") or "taiz").strip().lower()
        if scope in ("yemen", "yemen_trend"):
            counts["yemen_trend_count"] += 1

        counts["total_posts"] += 1

    return counts


def get_recent_published_summary(limit: int = 5, path: Path = DEFAULT_HISTORY_PATH) -> str:
    """Generate a formatted Arabic summary of recent posts to pass into Gemini context."""
    history = load_published_history(path)
    if not history:
        return "لا توجد منشورات سابقة مسجلة حتى الآن."

    lines = []
    for idx, item in enumerate(history[:limit], start=1):
        lines.append(f"{idx}. عنوان المنشور: {item.get('headline', '')}")
        lines.append(f"   المحتوى المنشور: {item.get('body', '')[:140]}...")

    return "\n".join(lines)
