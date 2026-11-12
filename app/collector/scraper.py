"""Lightweight HTML article scraper using standard library html.parser for top news candidates."""

from __future__ import annotations

import html
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import requests

from .base import Article

LOGGER = logging.getLogger(__name__)

SCRAPER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
}

MAX_FULL_TEXT_CHARS = 1500
MIN_PARAGRAPH_CHARS = 41
WHITESPACE_RE = re.compile(r"\s+")

BOILERPLATE_MARKERS = (
    "جميع الحقوق محفوظة",
    "حقوق النشر",
    "سياسة الخصوصية",
    "ملفات تعريف الارتباط",
    "اشترك في",
    "تابعونا على",
    "شروط الاستخدام",
    "all rights reserved",
    "privacy policy",
    "cookie policy",
)

IGNORED_CONTAINER_TAGS = frozenset({"script", "style", "noscript", "nav", "footer", "aside", "header", "form"})


class _ArticleHTMLParser(HTMLParser):
    """Extract meta image (og:image / twitter:image) and clean article <p> paragraphs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.og_image: Optional[str] = None
        self.twitter_image: Optional[str] = None
        self.paragraphs: List[str] = []
        self._ignored_depth = 0
        self._in_p = False
        self._current_p_chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        if t in IGNORED_CONTAINER_TAGS:
            self._ignored_depth += 1
            return

        if t == "meta":
            attr_map = {k.lower(): (v or "").strip() for k, v in attrs if k}
            prop = (attr_map.get("property") or attr_map.get("name") or "").lower()
            content = attr_map.get("content") or ""
            if content:
                if prop in ("og:image", "og:image:secure_url") and not self.og_image:
                    self.og_image = content
                elif prop in ("twitter:image", "twitter:image:src") and not self.twitter_image:
                    self.twitter_image = content
            return

        if t == "p" and self._ignored_depth == 0:
            self._in_p = True
            self._current_p_chunks = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in IGNORED_CONTAINER_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return

        if t == "p" and self._in_p:
            self._in_p = False
            raw_text = html.unescape("".join(self._current_p_chunks))
            cleaned = WHITESPACE_RE.sub(" ", raw_text).strip()
            self._current_p_chunks = []
            if len(cleaned) >= MIN_PARAGRAPH_CHARS and not _is_boilerplate(cleaned):
                self.paragraphs.append(cleaned)

    def handle_data(self, data: str) -> None:
        if self._in_p and self._ignored_depth == 0:
            self._current_p_chunks.append(data)


def _is_boilerplate(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in BOILERPLATE_MARKERS):
        return len(text) < 180
    return False


def _scrape_single_article(article: Article, timeout: int = 8) -> Article:
    """Fetch HTML for one article and return an updated copy with full_text and image_url."""
    url = (article.url or "").strip()
    if not url or not url.startswith(("http://", "https://")):
        return article

    try:
        response = requests.get(url, headers=SCRAPER_HEADERS, timeout=timeout)
        if response.status_code != 200:
            LOGGER.debug("Scraper skipped %s: HTTP %d", url, response.status_code)
            return article

        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and not any(t in content_type for t in ("html", "text", "xml")):
            return article

        # Fix missing charset headers common on Arabic sites
        if not response.encoding or response.encoding.lower() in ("iso-8859-1", "latin-1"):
            response.encoding = response.apparent_encoding or "utf-8"

        parser = _ArticleHTMLParser()
        parser.feed(response.text[:400_000])
        parser.close()

        # Deduplicate extracted paragraphs while preserving order and limiting total chars
        unique_paragraphs: List[str] = []
        seen_p: set[str] = set()
        total_len = 0
        for p in parser.paragraphs:
            if p in seen_p:
                continue
            seen_p.add(p)
            sep_len = 1 if unique_paragraphs else 0
            if total_len + sep_len + len(p) > MAX_FULL_TEXT_CHARS:
                remaining = MAX_FULL_TEXT_CHARS - total_len - sep_len
                if remaining > 60:
                    trimmed = p[:remaining].rsplit(" ", 1)[0].strip()
                    if trimmed:
                        unique_paragraphs.append(trimmed)
                break
            unique_paragraphs.append(p)
            total_len += sep_len + len(p)

        full_text = "\n".join(unique_paragraphs).strip()
        resolved_image = article.image_url
        if not resolved_image:
            raw_img = parser.og_image or parser.twitter_image
            if raw_img:
                resolved_image = urljoin(url, raw_img)

        update_fields = {}
        if full_text:
            update_fields["full_text"] = full_text
        if resolved_image and resolved_image != article.image_url:
            update_fields["image_url"] = resolved_image

        if not update_fields:
            return article

        if hasattr(article, "model_copy"):
            return article.model_copy(update=update_fields)
        return article.copy(update=update_fields)
    except Exception as exc:
        LOGGER.debug("Scraper failed quietly for %s: %s", url, exc)
        return article


def enrich_top_candidates(
    articles: List[Article],
    max_candidates: int = 6,
    timeout: int = 8,
) -> List[Article]:
    """Concurrently scrape full_text and meta image_url for only the top `max_candidates` articles."""
    if not articles or max_candidates <= 0:
        return list(articles)

    head = articles[:max_candidates]
    tail = articles[max_candidates:]

    with ThreadPoolExecutor(max_workers=4) as executor:
        enriched_head = list(executor.map(lambda art: _scrape_single_article(art, timeout=timeout), head))

    return enriched_head + tail
