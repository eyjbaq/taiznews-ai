"""Text cleanup for canonical articles."""

from __future__ import annotations

import html
import re
from typing import Iterable, List

from app.collector.base import Article

TAG_RE = re.compile(r"<[^>]*>")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    """Remove HTML tags/entities and collapse whitespace."""

    text = html.unescape(value or "")
    text = TAG_RE.sub(" ", text)
    text = text.replace("\xa0", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_article(article: Article) -> Article:
    """Return a copy of an article with normalized textual fields."""

    values = {
        "title": clean_text(article.title),
        "description": clean_text(article.description),
    }
    if hasattr(article, "model_copy"):
        return article.model_copy(update=values)
    return article.copy(update=values)


def normalize_articles(articles: Iterable[Article]) -> List[Article]:
    return [normalize_article(article) for article in articles]
