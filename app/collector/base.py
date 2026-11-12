"""Shared article model and helpers used by all collectors."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Article(BaseModel):
    """Canonical representation of a news article."""

    id: str = Field(..., description="Stable hash derived from the original URL")
    title: str
    url: str
    source: str
    published_at: datetime
    description: str = ""
    language: str = "ar"
    image_url: Optional[str] = None
    scope: str = "taiz"
    corroboration_count: int = 1
    corroborating_sources: List[str] = Field(default_factory=list)
    cluster_summary: str = ""
    full_text: str = ""

    def rich_context_text(self) -> str:
        """Return the richest combined text context available for AI editorial evaluation."""
        parts: List[str] = []
        full = (self.full_text or "").strip()
        cluster = (self.cluster_summary or "").strip()
        desc = (self.description or "").strip()

        if full:
            parts.append(full)
        if cluster and cluster not in full:
            parts.append(f"تغطية المصادر الأخرى:\n{cluster}" if parts else cluster)
        if not parts and desc:
            parts.append(desc)
        return "\n\n".join(parts)


def article_id_for_url(url: str) -> str:
    """Return a stable SHA-256 identifier for a URL."""

    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()
