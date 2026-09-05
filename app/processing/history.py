"""History manager for tracking published Facebook posts to avoid repetition."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.collector.base import Article
from app.ai.models import EditorialPost

LOGGER = logging.getLogger(__name__)

DEFAULT_HISTORY_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "published_posts.json"


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


def record_published_post(
    post: EditorialPost,
    article: Article,
    fb_result: Optional[Dict[str, Any]] = None,
    path: Path = DEFAULT_HISTORY_PATH,
) -> None:
    """Save a newly published post to the history log."""
    history = load_published_history(path)
    now_iso = datetime.now(timezone.utc).isoformat()

    entry = {
        "published_at": now_iso,
        "headline": post.headline.strip(),
        "body": post.body.strip(),
        "category": post.category,
        "article_title": article.title,
        "article_url": article.url,
        "article_source": article.source,
        "post_id": (fb_result or {}).get("post_id", ""),
        "photo_id": (fb_result or {}).get("id", ""),
        "post_url": (fb_result or {}).get("url", ""),
    }

    # Keep latest entries at the front
    history.insert(0, entry)
    # Retain the last 50 entries
    history = history[:50]

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        LOGGER.info("Recorded post to published history: %s", post.headline)
    except Exception as exc:
        LOGGER.error("Failed to write published history to %s: %s", path, exc)


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
