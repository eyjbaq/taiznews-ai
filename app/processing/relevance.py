"""Local relevance filter for Taiz, Yemen national trend hunter, and smart article scoring."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Iterable, List, Tuple

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

YEMEN_TREND_KEYWORDS = (
    "اليمن",
    "صنعاء",
    "عدن",
    "مأرب",
    "الحديدة",
    "حضرموت",
    "شبوة",
    "المهرة",
    "الضالع",
    "لحج",
    "أبين",
    "صعدة",
    "البيضاء",
    "إب",
    "ذمار",
    "حجة",
    "عمران",
    "الجوف",
    "سقطرى",
    "البحر الأحمر",
    "الريال اليمني",
    "البنك المركزي",
    "مجلس القيادة",
    "العليمي",
    "الحوثي",
    "الحوثيين",
    "الحكومة اليمنية",
    "المرتبات",
    "مرتبات",
    "الرواتب",
    "أسعار الصرف",
    "سعر الصرف",
    "الكهرباء",
    "المشتقات النفطية",
    "مطار صنعاء",
    "مطار عدن",
    "ميناء الحديدة",
)

OTHER_YEMEN_GOVERNORATES = (
    "صنعاء",
    "عدن",
    "مأرب",
    "الحديدة",
    "حضرموت",
    "شبوة",
    "المهرة",
    "الضالع",
    "لحج",
    "أبين",
    "صعدة",
    "البيضاء",
    "إب",
    "ذمار",
    "حجة",
    "عمران",
    "الجوف",
    "سقطرى",
    "ريمة",
    "المحويت",
)

URGENT_SCORE_KEYWORDS = (
    "عاجل", "مصرع", "سيطرة", "اشتباكات", "مواجهات", "غارات", "طيران",
    "معارك", "هجوم", "كارثة", "نداء إنساني", "تسلل", "استهداف", "شهداء",
    "انفجار", "حادث", "طريق", "سيول", "انهيار", "جريمة", "ضبط", "إحباط",
    "توتر", "احتجاجات", "فتح طريق", "أزمة",
)

VISUAL_AND_ECONOMIC_KEYWORDS = (
    "غارات", "حريق", "سيول", "اشتباكات", "طريق", "شاحنات", "موكب",
    "انفجار", "مسيرة", "سد", "مدرعات", "حصار", "ميناء", "المخا", "الحوبان", "جبل صبر",
    "العملة", "الريال", "الصرف", "مرتبات", "المرتبات", "الرواتب", "الكهرباء",
    "المشتقات", "الوقود", "البنك المركزي", "مطار", "سفينة",
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


def _match_keyword_list(
    raw_text: str,
    search_text: str,
    keywords: Tuple[str, ...],
) -> List[str]:
    """Match normalized Arabic/English keywords against article text safely."""
    raw_clean = ARABIC_DIACRITICS_RE.sub("", raw_text or "").replace("ـ", "")
    matched: List[str] = []

    for keyword in keywords:
        # Guard against false positives for 'إب' matching 'باب' or 'الأب'
        if keyword == "إب":
            ibb_raw_pat = r"(?:^|[\s،.:\-\"'()])(?:و|ف|ب|ل|ك)?إب(?:$|[\s،.:\-\"'()!؟])"
            if re.search(ibb_raw_pat, raw_clean) or "محافظه اب" in search_text or "مدينه اب" in search_text:
                matched.append(keyword)
            continue

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
            suffix = r"(?:ين|ون|يه)?" if norm_kw.endswith("ي") else r"(?:ي|يه|يين|يون)?"
            pat = pref + re.escape(norm_kw) + suffix + r"(?:$|\s)"
            if re.search(pat, search_text):
                matched.append(keyword)

    return matched


def matched_keywords(article: Article) -> List[str]:
    """Return local Taiz keywords matched in the article title or description."""
    raw_text = f"{article.title} {article.description or ''}"
    search_text = normalize_arabic(raw_text)
    return _match_keyword_list(raw_text, search_text, TAIZ_KEYWORDS)


def is_relevant(article: Article) -> bool:
    """Return true when title or short text mentions Taiz or a local area."""
    title_raw = article.title or ""
    title_norm = normalize_arabic(title_raw)
    if (
        _match_keyword_list(title_raw, title_norm, OTHER_YEMEN_GOVERNORATES)
        and not _match_keyword_list(title_raw, title_norm, TAIZ_KEYWORDS)
    ):
        return False
    return bool(matched_keywords(article))


def filter_relevant(articles: Iterable[Article]) -> List[Article]:
    """Filter articles relevant to Taiz governorate."""
    return [article for article in articles if is_relevant(article)]


def matched_yemen_keywords(article: Article) -> List[str]:
    """Return national Yemen trend keywords matched in the article."""
    raw_text = f"{article.title} {article.description or ''}"
    search_text = normalize_arabic(raw_text)
    return _match_keyword_list(raw_text, search_text, YEMEN_TREND_KEYWORDS)


def is_yemen_trend_relevant(article: Article) -> bool:
    """Return True if the article is NOT about Taiz directly but covers a major Yemen trend."""
    if is_relevant(article):
        return False
    return bool(matched_yemen_keywords(article))


def filter_fresh_articles(
    articles: Iterable[Article],
    max_age_hours: float = 24.0,
) -> List[Article]:
    """Keep only articles published within the last `max_age_hours` (strict freshness gate)."""
    now = datetime.now(timezone.utc)
    fresh: List[Article] = []
    for article in articles:
        pub = article.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        else:
            pub = pub.astimezone(timezone.utc)
        age_hours = (now - pub).total_seconds() / 3600.0
        if age_hours <= max_age_hours:
            fresh.append(article)
    return fresh


def filter_yemen_trend(articles: Iterable[Article]) -> List[Article]:
    """Select national Yemen trend articles outside Taiz and tag them with scope='yemen_trend'."""
    results: List[Article] = []
    for article in articles:
        if is_yemen_trend_relevant(article):
            if hasattr(article, "model_copy"):
                results.append(article.model_copy(update={"scope": "yemen_trend"}))
            else:
                results.append(article.copy(update={"scope": "yemen_trend"}))
    return results


def score_article_smart(article: Article) -> float:
    """Score an article using capped urgency, visual/economic impact, corroboration, scope, and freshness."""
    score = 0.0
    title_norm = normalize_arabic(article.title)
    desc_norm = normalize_arabic(article.description or "")

    urgent_points = 0.0
    for kw in URGENT_SCORE_KEYWORDS:
        kw_norm = normalize_arabic(kw)
        if kw_norm in title_norm:
            urgent_points += 35.0
        elif kw_norm in desc_norm:
            urgent_points += 15.0
        if urgent_points >= 50.0:
            break
    score += min(50.0, urgent_points)

    for vkw in VISUAL_AND_ECONOMIC_KEYWORDS:
        vkw_norm = normalize_arabic(vkw)
        if vkw_norm in title_norm or vkw_norm in desc_norm:
            score += 20.0
            break

    corroboration = max(1, int(getattr(article, "corroboration_count", 1) or 1))
    if corroboration > 1:
        score += (corroboration - 1) * 25.0

    if getattr(article, "scope", "taiz") == "taiz":
        if "تعز" in title_norm:
            score += 40.0
        elif "تعز" in desc_norm:
            score += 20.0

    now = datetime.now(timezone.utc)
    pub = article.published_at
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    else:
        pub = pub.astimezone(timezone.utc)
    age_hours = max(0.1, (now - pub).total_seconds() / 3600.0)
    score += max(0.0, 50.0 - (age_hours * 2.0))
    if age_hours > 12.0:
        score -= 25.0

    return score


score_article = score_article_smart
