"""Shared article model and helpers used by all collectors."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Optional

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


def article_id_for_url(url: str) -> str:
    """Return a stable SHA-256 identifier for a URL."""

    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()
