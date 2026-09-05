"""Local relevance filter for Taiz city and governorate."""

from __future__ import annotations

import re
from typing import Iterable, List

from app.collector.base import Article

TAIZ_KEYWORDS = (
    "تعز",
    "الحوبان",
    "التربة",
    "المخا",
    "المعافر",
    "المواسط",
    "صبر الموادم",
    "مشرعة وحدنان",
    "حدنان",
    "صالة",
    "المظفر",
    "القاهرة",
    "الشمايتين",
    "جبل حبشي",
    "مقبنة",
    "الوازعية",
    "موزع",
    "حيفان",
    "ماوية",
    "الضباب",
    "هيجة العبد",
    "النشمة",
    "شرعب الرونة",
    "شرعب السلام",
    "شرعب",
    "الصلو",
    "سامع",
    "دمنة خدير",
    "خدير",
    "المسراخ",
    "باب المندب",
    "ذباب",
    "قهبان",
    "الكدحة",
    "كلابة",
    "عصيفرة",
    "بير باشا",
    "الجحملية",
    "الاربعين",
    "الدفاع الجوي",
    "الشقب",
    "الاحكوم",
    "وادي القاضي",
    "محور تعز",
    "محافظة تعز",
    "مدينة تعز",
    # English names for international sources and GDELT
    "taiz",
    "taizz",
)

ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
PUNCTUATION_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text:
    - Convert (إ, أ, آ, ٱ) -> ا
    - Convert ة -> ه
    - Convert ى -> ي
    - Remove tashkeel (diacritics) and tatweel (ـ)
    """
    if not text:
        return ""
    cleaned = text.lower()
    cleaned = ARABIC_DIACRITICS_RE.sub("", cleaned)
    cleaned = cleaned.replace("ـ", "")
    cleaned = PUNCTUATION_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[إأآٱ]", "ا", cleaned)
    cleaned = cleaned.replace("ة", "ه")
    cleaned = cleaned.replace("ى", "ي")
    cleaned = cleaned.replace("ؤ", "و").replace("ئ", "ي")
    return re.sub(r"\s+", " ", cleaned).strip()


def matched_keywords(article: Article) -> List[str]:
    search_text = normalize_arabic(f"{article.title} {article.description or ''}")
    matched = []
    for keyword in TAIZ_KEYWORDS:
        norm_kw = normalize_arabic(keyword)
        if not norm_kw:
            continue
        if " " in norm_kw:
            if norm_kw in search_text:
                matched.append(keyword)
        elif norm_kw.isascii():
            if re.search(r"\b" + re.escape(norm_kw) + r"\b", search_text):
                matched.append(keyword)
        else:
            pref = r"(?:^|\s)(?:و|ف|ب|ل|ك|وب|ول|فل)?"
            if not norm_kw.startswith("ال"):
                pref += r"(?:ال)?"
            pat = pref + re.escape(norm_kw) + r"(?:ي|يه)?(?:$|\s)"
            if re.search(pat, search_text):
                matched.append(keyword)
    return matched


def is_relevant(article: Article) -> bool:
    """Return true when title or short text mentions Taiz or a local area."""

    return bool(matched_keywords(article))


def filter_relevant(articles: Iterable[Article]) -> List[Article]:
    return [article for article in articles if is_relevant(article)]
