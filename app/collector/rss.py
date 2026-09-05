"""RSS/Atom collector for configured Yemeni news sources."""

from __future__ import annotations

import calendar
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import feedparser
import requests
import yaml
from dateutil import parser as date_parser

from .base import Article, article_id_for_url

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sources.yaml"
RSS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*;q=0.8",
}


def load_sources(config_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load and validate the RSS source list from YAML."""

    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    sources = config.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("sources.yaml must contain a list named 'sources'")

    valid_sources = []
    for source in sources:
        if not isinstance(source, dict) or not source.get("name") or not source.get("feed_url"):
            LOGGER.warning("Skipping malformed RSS source configuration: %r", source)
            continue
        valid_sources.append(source)
    return valid_sources


def load_settings(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the optional settings section from the source configuration."""

    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    return config.get("settings", {})


def _published_datetime(entry: Any) -> datetime:
    """Extract an aware UTC datetime from a feed entry."""

    for parsed_field in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed_value = getattr(entry, parsed_field, None)
        if parsed_value:
            return datetime.fromtimestamp(calendar.timegm(parsed_value), tz=timezone.utc)

    for text_field in ("published", "updated", "created", "date"):
        text_value = getattr(entry, text_field, None)
        if text_value:
            try:
                parsed_date = date_parser.parse(str(text_value))
                return parsed_date.replace(tzinfo=timezone.utc) if parsed_date.tzinfo is None else parsed_date.astimezone(timezone.utc)
            except (TypeError, ValueError, OverflowError):
                LOGGER.debug("Could not parse feed date %r", text_value)

    return datetime.now(timezone.utc)


def _entry_image_url(entry: Any) -> Optional[str]:
    """Find a usable image URL across common RSS/Atom extensions."""

    for field_name in ("media_content", "media_thumbnail"):
        media_items = getattr(entry, field_name, None) or []
        if media_items and isinstance(media_items, list):
            candidate = media_items[0].get("url")
            if candidate:
                return str(candidate)

    for enclosure in getattr(entry, "enclosures", None) or []:
        candidate = enclosure.get("href") or enclosure.get("url")
        media_type = str(enclosure.get("type", ""))
        if candidate and (media_type.startswith("image/") or not media_type):
            return str(candidate)

    return None


def _entry_to_article(entry: Any, source_name: str) -> Optional[Article]:
    title = str(getattr(entry, "title", "")).strip()
    url = str(getattr(entry, "link", "")).strip()
    if not title or not url:
        return None

    description = getattr(entry, "summary", None) or getattr(entry, "description", None) or ""
    return Article(
        id=article_id_for_url(url),
        title=title,
        url=url,
        source=source_name,
        published_at=_published_datetime(entry),
        description=str(description),
        language="ar",
        image_url=_entry_image_url(entry),
    )


def fetch_rss_articles(
    config_path: Optional[Path] = None,
    timeout: int = 20,
    max_entries_per_source: int = 30,
) -> List[Article]:
    """Fetch configured RSS feeds and map entries to :class:`Article` objects.

    A broken or unavailable feed is logged and skipped so other sources can
    still contribute news during the dry run.
    """

    articles: List[Article] = []
    for source in load_sources(config_path):
        source_name = str(source["name"])
        feed_url = str(source["feed_url"])
        try:
            response = requests.get(feed_url, headers=RSS_HEADERS, timeout=timeout)
            if response.status_code != 200:
                LOGGER.warning("RSS source failed (%s): HTTP status %d", source_name, response.status_code)
                continue

            content_type = response.headers.get("Content-Type", "").lower()
            text_prefix = response.text.lstrip()[:100].lower()
            if (
                not any(sub in content_type for sub in ("xml", "rss", "atom"))
                or text_prefix.startswith("<!doctype html")
                or text_prefix.startswith("<html")
            ):
                LOGGER.warning("RSS source skipped (%s): HTML response received instead of XML", source_name)
                continue

            parsed_feed = feedparser.parse(response.content)
            if getattr(parsed_feed, "bozo", False) and not getattr(parsed_feed, "entries", []):
                raise ValueError(f"invalid or empty feed ({parsed_feed.bozo_exception})")

            source_articles = []
            for entry in parsed_feed.entries[:max_entries_per_source]:
                article = _entry_to_article(entry, source_name)
                if article:
                    source_articles.append(article)
            LOGGER.info("RSS source %s: %d articles", source_name, len(source_articles))
            articles.extend(source_articles)
        except (requests.RequestException, ValueError, UnicodeError) as exc:
            LOGGER.warning("RSS source failed (%s): %s", source_name, exc)

    return articles
