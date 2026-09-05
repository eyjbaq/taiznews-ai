"""Persistent URL and in-session title deduplication."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Set

from app.collector.base import Article

DEFAULT_PROCESSED_PATH = Path(__file__).resolve().parents[2] / "data" / "processed_urls.json"
WORD_RE = re.compile(r"\w+", re.UNICODE)


def load_processed_urls(path: Path = DEFAULT_PROCESSED_PATH) -> Set[str]:
    """Load processed URLs, accepting both the initial list and old map formats."""

    file_path = Path(path)
    if not file_path.exists():
        return set()
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if isinstance(data, list):
        return {str(url) for url in data if url}
    if isinstance(data, dict):
        return {str(url) for url in data.keys() if url}
    return set()


def is_processed(url: str, processed_urls: Set[str]) -> bool:
    return url.strip() in processed_urls


def _title_words(title: str) -> Set[str]:
    return {word.lower() for word in WORD_RE.findall(title or "") if len(word) > 1}


def jaccard_similarity(first_title: str, second_title: str) -> float:
    first_words = _title_words(first_title)
    second_words = _title_words(second_title)
    if not first_words or not second_words:
        return 0.0
    return len(first_words & second_words) / len(first_words | second_words)


def deduplicate_articles(articles: Iterable[Article], similarity_threshold: float = 0.75) -> List[Article]:
    """Remove duplicate URLs and near-identical titles within this run."""

    unique: List[Article] = []
    seen_urls: Set[str] = set()
    for article in articles:
        if article.url in seen_urls:
            continue
        if any(jaccard_similarity(article.title, existing.title) >= similarity_threshold for existing in unique):
            continue
        seen_urls.add(article.url)
        unique.append(article)
    return unique


def save_processed_urls(urls: Iterable[str], path: Path = DEFAULT_PROCESSED_PATH) -> None:
    """Merge URLs into the persistent JSON list."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    all_urls = load_processed_urls(file_path)
    all_urls.update(url.strip() for url in urls if url and url.strip())
    file_path.write_text(json.dumps(sorted(all_urls), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
