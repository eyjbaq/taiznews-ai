"""Persistent URL, history-based filtering, and multi-source story clustering."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from app.collector.base import Article
from app.processing.relevance import normalize_arabic

DEFAULT_PROCESSED_PATH = Path(__file__).resolve().parents[2] / "data" / "processed_urls.json"
WORD_RE = re.compile(r"\w+", re.UNICODE)
MAX_CLUSTER_SUMMARY_CHARS = 1200

RAW_STOP_WORDS = (
    "في", "من", "على", "عن", "إلى", "الى", "بعد", "قبل", "مع", "بين",
    "عاجل", "تفاصيل", "شاهد", "فيديو", "بالصور", "اليمن", "اليمنية",
    "مصادر", "يكشف", "يعلن", "تعلن", "خلال", "حول", "أكثر", "اكثر",
    "جديد", "اليوم", "الآن", "الان",
)
STOP_WORDS: Set[str] = {normalize_arabic(w) for w in RAW_STOP_WORDS if normalize_arabic(w)}


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
    """Extract normalized Arabic/English tokens from an article title."""
    normalized = normalize_arabic(title or "")
    return {word for word in WORD_RE.findall(normalized) if len(word) > 1 and word not in STOP_WORDS}


def _jaccard_word_sets(first_words: Set[str], second_words: Set[str]) -> float:
    if not first_words or not second_words:
        return 0.0
    return len(first_words & second_words) / len(first_words | second_words)


def jaccard_similarity(first_title: str, second_title: str) -> float:
    return _jaccard_word_sets(_title_words(first_title), _title_words(second_title))


def filter_against_published_history(
    articles: Iterable[Article],
    history_entries: List[Dict[str, Any]],
    similarity_threshold: float = 0.55,
) -> List[Article]:
    """Exclude candidate articles whose titles match recently published stories locally."""
    if not history_entries:
        return list(articles)

    historical_word_sets: List[Set[str]] = []
    for entry in history_entries:
        if not isinstance(entry, dict):
            continue
        for field_name in ("article_title", "headline"):
            words = _title_words(str(entry.get(field_name) or ""))
            if words:
                historical_word_sets.append(words)

    if not historical_word_sets:
        return list(articles)

    fresh: List[Article] = []
    for article in articles:
        cand_words = _title_words(article.title)
        if any(
            _jaccard_word_sets(cand_words, hist_words) >= similarity_threshold
            for hist_words in historical_word_sets
        ):
            continue
        fresh.append(article)
    return fresh


def _build_cluster_summary(primary: Article, members: List[Article]) -> str:
    """Merge unique headlines and descriptions from secondary cluster members up to 1200 chars."""
    if len(members) <= 1:
        return ""

    primary_desc = (primary.description or "").strip()
    seen_snippets: Set[str] = {normalize_arabic(primary.title)}
    lines: List[str] = []

    for member in members:
        if member is primary:
            continue
        title_clean = (member.title or "").strip()
        desc_clean = (member.description or "").strip()
        title_norm = normalize_arabic(title_clean)

        snippet_parts: List[str] = []
        if title_clean and title_norm not in seen_snippets:
            seen_snippets.add(title_norm)
            snippet_parts.append(title_clean)

        if desc_clean and desc_clean != primary_desc and normalize_arabic(desc_clean) not in seen_snippets:
            seen_snippets.add(normalize_arabic(desc_clean))
            snippet_parts.append(desc_clean)

        if snippet_parts:
            lines.append(f"[{member.source}] " + " — ".join(snippet_parts))

    combined = "\n".join(lines).strip()
    if len(combined) <= MAX_CLUSTER_SUMMARY_CHARS:
        return combined
    return combined[: MAX_CLUSTER_SUMMARY_CHARS - 3].rsplit(" ", 1)[0].strip() + "..."


def cluster_articles(articles: Iterable[Article], similarity_threshold: float = 0.33) -> List[Article]:
    """Group similar multi-source stories into clusters and boost corroboration metadata."""
    clusters: List[Dict[str, Any]] = []
    seen_urls: Set[str] = set()

    for article in articles:
        clean_url = article.url.strip()
        if clean_url in seen_urls:
            continue
        seen_urls.add(clean_url)

        title_words = _title_words(article.title)
        matched_cluster: Dict[str, Any] | None = None
        for cluster in clusters:
            if any(
                _jaccard_word_sets(title_words, m_words) >= similarity_threshold
                for m_words in cluster["word_sets"]
            ):
                matched_cluster = cluster
                break

        if matched_cluster is not None:
            matched_cluster["members"].append(article)
            matched_cluster["word_sets"].append(title_words)
        else:
            clusters.append({
                "members": [article],
                "word_sets": [title_words],
            })

    clustered_results: List[Article] = []
    for cluster in clusters:
        members: List[Article] = cluster["members"]
        primary = max(
            members,
            key=lambda a: (len((a.description or "").strip()), 1 if a.image_url else 0),
        )

        unique_sources: List[str] = []
        inherited_image = primary.image_url
        for member in [primary] + [m for m in members if m is not primary]:
            src = (member.source or "").strip()
            if src and src not in unique_sources:
                unique_sources.append(src)
            if not inherited_image and member.image_url:
                inherited_image = member.image_url

        update_fields = {
            "corroboration_count": max(1, len(unique_sources)),
            "corroborating_sources": unique_sources,
            "cluster_summary": _build_cluster_summary(primary, members),
            "image_url": inherited_image,
        }
        if hasattr(primary, "model_copy"):
            clustered_results.append(primary.model_copy(update=update_fields))
        else:
            clustered_results.append(primary.copy(update=update_fields))

    return clustered_results


def deduplicate_articles(articles: Iterable[Article], similarity_threshold: float = 0.75) -> List[Article]:
    """Backward-compatible wrapper preserving deduplicate_articles via story clustering."""
    return cluster_articles(articles, similarity_threshold=similarity_threshold)


def save_processed_urls(urls: Iterable[str], path: Path = DEFAULT_PROCESSED_PATH) -> None:
    """Merge URLs into the persistent JSON list."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    all_urls = load_processed_urls(file_path)
    all_urls.update(url.strip() for url in urls if url and url.strip())
    file_path.write_text(json.dumps(sorted(all_urls), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
