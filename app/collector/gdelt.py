"""Collector for the public GDELT DOC 2.0 Article List API."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests

from .base import Article, article_id_for_url

LOGGER = logging.getLogger(__name__)
GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_QUERY = '(تعز OR الحوبان OR حيفان OR مقبنة OR المخا OR Taiz)'
GDELT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}


def _parse_gdelt_date(value: Any) -> datetime:
    if value:
        try:
            parsed = datetime.strptime(str(value), "%Y%m%d%H%M%S")
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            LOGGER.debug("Could not parse GDELT date %r", value)
    return datetime.now(timezone.utc)


def fetch_gdelt_articles(
    query: str = DEFAULT_QUERY,
    maxrecords: int = 35,
    timeout: int = 20,
) -> List[Article]:
    """Query GDELT DOC 2.0 and return its article records as ``Article`` objects."""

    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": maxrecords,
        "format": "json",
    }
    try:
        response = requests.get(
            GDELT_ENDPOINT,
            params=params,
            headers=GDELT_HEADERS,
            timeout=timeout,
        )
        if response.status_code == 429:
            LOGGER.debug("GDELT rate limit reached; continuing without GDELT articles")
            return []
        response.raise_for_status()
        payload: Dict[str, Any] = response.json()
    except (requests.RequestException, ValueError) as exc:
        LOGGER.debug("GDELT request failed; continuing without GDELT articles: %s", exc)
        return []

    articles: List[Article] = []
    for record in payload.get("articles", []) or []:
        url = str(record.get("url") or record.get("url_mobile") or "").strip()
        title = str(record.get("title") or "").strip()
        if not url or not title:
            continue

        domain = str(record.get("domain") or "GDELT")
        articles.append(
            Article(
                id=article_id_for_url(url),
                title=title,
                url=url,
                source=f"GDELT / {domain}",
                published_at=_parse_gdelt_date(record.get("seendate")),
                description=str(record.get("snippet") or record.get("description") or ""),
                language=str(record.get("language") or "ar"),
                image_url=record.get("socialimage") or None,
            )
        )

    LOGGER.info("GDELT: %d articles", len(articles))
    return articles
