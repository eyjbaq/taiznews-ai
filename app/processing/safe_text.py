"""Facebook Algorithm Defense Shield & Broadcast Speech Sanitizer.

This module replaces sensitive political and military trigger words with eloquent,
broadcast-standard Arabic journalistic phrasing across all media outputs (Facebook captions,
card headlines, video subtitles, and spoken audio TTS).

All dot-tatweel cutting ('ـ.ـ') has been eliminated in favor of clean journalistic synonyms
with full support for vocalized Arabic diacritics and strict hashtag preservation.
"""

from __future__ import annotations

import re
from typing import List, Pattern, Tuple

# Tashkeel / diacritics / tatweel pattern matching any vowel mark between base Arabic letters
_TASHKEEL_RE = r"[\u0610-\u061A\u0640\u064B-\u065F\u0670]*"


def _build_arabic_pattern(term: str) -> str:
    """Build a regex pattern that matches the Arabic term allowing arbitrary diacritics and letter variants."""
    parts: List[str] = []
    for ch in term:
        if ch == " ":
            parts.append(r"\s+")
        elif ch in ("ا", "أ", "إ", "آ"):
            parts.append(r"[اأإآ]" + _TASHKEEL_RE)
        elif ch in ("ي", "ى"):
            parts.append(r"[يى]" + _TASHKEEL_RE)
        elif ch in ("ة", "ه"):
            parts.append(r"[ةه]" + _TASHKEEL_RE)
        else:
            parts.append(re.escape(ch) + _TASHKEEL_RE)
    return "".join(parts)


def _p(term: str) -> str:
    return _build_arabic_pattern(term)


# Ordered list of replacement rules (compound phrases and prepositions first, single words second)
# Format: (compiled_regex, replacement_str)
_RULES_RAW: List[Tuple[str, str]] = [
    # 1. Preposition + Militia + Houthi compounds (e.g. من مليشيا الحوثي -> من مسلحي الجماعة)
    (
        r"(?<![ء-ي\w])(?P<prep>" + _p("من") + r"|" + _p("على") + r"|" + _p("عن") + r"|" + _p("ضد") + r")\s+"
        + r"(?:" + _p("ميليشيا") + r"|" + _p("مليشيا") + r"|" + _p("ميليشيات") + r"|" + _p("مليشيات") + r")\s+"
        + r"(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r"|" + _p("الحوثيون") + r")(?![ء-ي\w])",
        r"\g<prep> مسلحي الجماعة",
    ),

    # 2. Prefixed Militia + Houthi (e.g. ولمليشيا الحوثي / لمليشيا الحوثي / بمليشيا الحوثي)
    (
        r"(?<![ء-ي\w])" + _p("ولمليشيا") + r"\s+(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r"|" + _p("الحوثيون") + r")(?![ء-ي\w])",
        "ولمسلحي الجماعة",
    ),
    (
        r"(?<![ء-ي\w])" + _p("ولميليشيا") + r"\s+(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r"|" + _p("الحوثيون") + r")(?![ء-ي\w])",
        "ولمسلحي الجماعة",
    ),
    (
        r"(?<![ء-ي\w])" + _p("لمليشيا") + r"\s+(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r"|" + _p("الحوثيون") + r")(?![ء-ي\w])",
        "لمسلحي الجماعة",
    ),
    (
        r"(?<![ء-ي\w])" + _p("لميليشيا") + r"\s+(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r"|" + _p("الحوثيون") + r")(?![ء-ي\w])",
        "لمسلحي الجماعة",
    ),
    (
        r"(?<![ء-ي\w])" + _p("بمليشيا") + r"\s+(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r"|" + _p("الحوثيون") + r")(?![ء-ي\w])",
        "بمسلحي الجماعة",
    ),
    (
        r"(?<![ء-ي\w])" + _p("بميليشيا") + r"\s+(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r"|" + _p("الحوثيون") + r")(?![ء-ي\w])",
        "بمسلحي الجماعة",
    ),

    # 3. Standard Militia + Houthi compounds (subject / starting sentence)
    (
        r"(?<![ء-ي\w])(?:و|ف)?" + _p("مليشيا") + r"\s+(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r"|" + _p("الحوثيون") + r")(?![ء-ي\w])",
        "مسلحو الجماعة",
    ),
    (
        r"(?<![ء-ي\w])(?:و|ف)?" + _p("ميليشيا") + r"\s+(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r"|" + _p("الحوثيون") + r")(?![ء-ي\w])",
        "مسلحو الجماعة",
    ),
    (
        r"(?<![ء-ي\w])(?:و|ف)?" + _p("مليشيات") + r"\s+(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r"|" + _p("الحوثيون") + r")(?![ء-ي\w])",
        "مسلحو الجماعة",
    ),
    (
        r"(?<![ء-ي\w])(?:و|ف)?" + _p("ميليشيات") + r"\s+(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r"|" + _p("الحوثيون") + r")(?![ء-ي\w])",
        "مسلحو الجماعة",
    ),

    # 4. Forces / Elements compounds
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفكل]?)" + _p("قوات") + r"\s+(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r")(?![ء-ي\w])",
        r"\g<pfx>قوات الجماعة",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفكل]?)" + _p("عناصر") + r"\s+(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r")(?![ء-ي\w])",
        r"\g<pfx>عناصر الجماعة",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفكل]?)" + _p("مسلحي") + r"\s+(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r")(?![ء-ي\w])",
        r"\g<pfx>مسلحي الجماعة",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفكل]?)" + _p("مسلحو") + r"\s+(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r")(?![ء-ي\w])",
        r"\g<pfx>مسلحو الجماعة",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفكل]?)" + _p("قيادات") + r"\s+(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r")(?![ء-ي\w])",
        r"\g<pfx>قيادات الجماعة",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفكل]?)" + _p("قادة") + r"\s+(?:" + _p("الحوثي") + r"|" + _p("الحوثيين") + r")(?![ء-ي\w])",
        r"\g<pfx>قادة الجماعة",
    ),

    # 5. Compound Airstrikes (غارات جوية / غارة جوية)
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفب]?)" + _p("غارات جوية") + r"(?![ء-ي\w])",
        r"\g<pfx>ضربات جوية",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفب]?)" + _p("غارة جوية") + r"(?![ء-ي\w])",
        r"\g<pfx>ضربة جوية",
    ),

    # 6. Prefixed Houthi forms
    (r"(?<![ء-ي\w])" + _p("والحوثيون") + r"(?![ء-ي\w])", "ومسلحو الجماعة"),
    (r"(?<![ء-ي\w])" + _p("فالحوثيون") + r"(?![ء-ي\w])", "فمسلحو الجماعة"),
    (r"(?<![ء-ي\w])" + _p("والحوثيين") + r"(?![ء-ي\w])", "ومسلحي الجماعة"),
    (r"(?<![ء-ي\w])" + _p("فالحوثيين") + r"(?![ء-ي\w])", "فمسلحي الجماعة"),
    (r"(?<![ء-ي\w])" + _p("وللحوثيين") + r"(?![ء-ي\w])", "ولمسلحي الجماعة"),
    (r"(?<![ء-ي\w])" + _p("للحوثيين") + r"(?![ء-ي\w])", "لمسلحي الجماعة"),
    (r"(?<![ء-ي\w])" + _p("وللحوثي") + r"(?![ء-ي\w])", "وللجماعة"),
    (r"(?<![ء-ي\w])" + _p("للحوثي") + r"(?![ء-ي\w])", "للجماعة"),
    (r"(?<![ء-ي\w])" + _p("وبالحوثيين") + r"(?![ء-ي\w])", "وبمسلحي الجماعة"),
    (r"(?<![ء-ي\w])" + _p("بالحوثيين") + r"(?![ء-ي\w])", "بمسلحي الجماعة"),
    (r"(?<![ء-ي\w])" + _p("وبالحوثي") + r"(?![ء-ي\w])", "وبالجماعة"),
    (r"(?<![ء-ي\w])" + _p("بالحوثي") + r"(?![ء-ي\w])", "بالجماعة"),
    (r"(?<![ء-ي\w])" + _p("والحوثي") + r"(?![ء-ي\w])", "والجماعة"),
    (r"(?<![ء-ي\w])" + _p("فالحوثي") + r"(?![ء-ي\w])", "فالجماعة"),

    # 7. Base Houthi forms
    (r"(?<![ء-ي\w])" + _p("الحوثيون") + r"(?![ء-ي\w])", "مسلحو الجماعة"),
    (r"(?<![ء-ي\w])" + _p("الحوثيين") + r"(?![ء-ي\w])", "مسلحي الجماعة"),
    (r"(?<![ء-ي\w])" + _p("الحوثي") + r"(?![ء-ي\w])", "الجماعة"),

    # 8. Indefinite Houthi forms (حوثيون / حوثيين / حوثية / حوثي)
    (r"(?<![ء-ي\w])(?P<pfx>[وف]?)" + _p("حوثيون") + r"(?![ء-ي\w])", r"\g<pfx>مسلحو الجماعة"),
    (r"(?<![ء-ي\w])(?P<pfx>[وف]?)" + _p("حوثيين") + r"(?![ء-ي\w])", r"\g<pfx>مسلحي الجماعة"),
    (r"(?<![ء-ي\w])(?P<pfx>[وفب]?)" + _p("حوثية") + r"(?![ء-ي\w])", r"\g<pfx>مسلحة"),
    (r"(?<![ء-ي\w])(?P<pfx>[وفب]?)" + _p("حوثي") + r"(?![ء-ي\w])", r"\g<pfx>مسلح"),

    # 9. Standalone Militias (مليشيا / ميليشيا / مليشيات / ميليشيات)
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفبل]?)" + _p("مليشيات") + r"(?![ء-ي\w])",
        r"\g<pfx>تشكيلات مسلحة",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفبل]?)" + _p("ميليشيات") + r"(?![ء-ي\w])",
        r"\g<pfx>تشكيلات مسلحة",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفبل]?)" + _p("مليشيا") + r"(?![ء-ي\w])",
        r"\g<pfx>تشكيلات مسلحة",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفبل]?)" + _p("ميليشيا") + r"(?![ء-ي\w])",
        r"\g<pfx>تشكيلات مسلحة",
    ),

    # 10. Casualties (مقتل / مصرع / قتلى / قتيل)
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفب]?)" + _p("مقتل") + r"(?![ء-ي\w])",
        r"\g<pfx>سقوط",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفب]?)" + _p("مصرع") + r"(?![ء-ي\w])",
        r"\g<pfx>سقوط",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفب]?)" + _p("قتلى") + r"(?![ء-ي\w])",
        r"\g<pfx>ضحايا",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفب]?)" + _p("قتيل") + r"(?![ء-ي\w])",
        r"\g<pfx>ضحية",
    ),

    # 11. Airstrikes & Shelling (غارات / غارة / قصف)
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفب]?)" + _p("غارات") + r"(?![ء-ي\w])",
        r"\g<pfx>ضربات",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفب]?)" + _p("غارة") + r"(?![ء-ي\w])",
        r"\g<pfx>ضربة",
    ),
    (
        r"(?<![ء-ي\w])" + _p("بالقصف") + r"(?![ء-ي\w])",
        "بالاستهداف",
    ),
    (
        r"(?<![ء-ي\w])" + _p("للقصف") + r"(?![ء-ي\w])",
        "للاستهداف",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وف]?)" + _p("قصف") + r"(?![ء-ي\w])",
        r"\g<pfx>استهداف",
    ),

    # 12. Explosions & Assassinations
    (
        r"(?<![ء-ي\w])(?P<pfx>[وف]?)" + _p("انفجارات") + r"(?![ء-ي\w])",
        r"\g<pfx>حادث ميداني",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وف]?)" + _p("انفجار") + r"(?![ء-ي\w])",
        r"\g<pfx>دوي",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفب]?)" + _p("اغتيالات") + r"(?![ء-ي\w])",
        r"\g<pfx>استهدافات",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وفب]?)" + _p("اغتيال") + r"(?![ء-ي\w])",
        r"\g<pfx>استهداف",
    ),

    # 13. Suicide / Terror terms
    (
        r"(?<![ء-ي\w])(?P<pfx>[وف]?)" + _p("انتحاريين") + r"(?![ء-ي\w])",
        r"\g<pfx>مهاجمين",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وف]?)" + _p("انتحاري") + r"(?![ء-ي\w])",
        r"\g<pfx>مهاجم",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وف]?)" + _p("انتحارية") + r"(?![ء-ي\w])",
        r"\g<pfx>هجومية",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وف]?)" + _p("إرهابيين") + r"(?![ء-ي\w])",
        r"\g<pfx>مسلحين",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وف]?)" + _p("إرهابية") + r"(?![ء-ي\w])",
        r"\g<pfx>مسلحة",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وف]?)" + _p("إرهابي") + r"(?![ء-ي\w])",
        r"\g<pfx>مسلح",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وف]?)" + _p("ارهابيين") + r"(?![ء-ي\w])",
        r"\g<pfx>مسلحين",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وف]?)" + _p("ارهابية") + r"(?![ء-ي\w])",
        r"\g<pfx>مسلحة",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وف]?)" + _p("ارهابي") + r"(?![ء-ي\w])",
        r"\g<pfx>مسلح",
    ),

    # 14. Snipers
    (
        r"(?<![ء-ي\w])(?P<pfx>[وف]?)" + _p("قناصة") + r"(?![ء-ي\w])",
        r"\g<pfx>نيران مباشرة",
    ),
    (
        r"(?<![ء-ي\w])(?P<pfx>[وف]?)" + _p("قناص") + r"(?![ء-ي\w])",
        r"\g<pfx>مطلق نار",
    ),
]

_COMPILED_RULES: List[Tuple[Pattern[str], str]] = [
    (re.compile(pattern), repl) for pattern, repl in _RULES_RAW
]


def sanitize_sensitive_terms(text: str) -> str:
    """Sanitize Arabic text by replacing sensitive political/military trigger words with broadcast-safe alternatives.

    Key guarantees:
      1. Zero dot-tatweel cutting ('ـ.ـ') — words are replaced cleanly with natural Arabic journalistic terms.
      2. Handles both unvocalized and fully vocalized Arabic text (tashkeel/harakat).
      3. Compound phrases are evaluated before individual words.
      4. Preserves hashtags (#...) completely intact so Facebook indexing and reach remain unaffected.
    """
    if not text:
        return ""

    # 1. Protect all hashtags from modification by stashing them into unique placeholders
    saved_hashtags: List[str] = []

    def _stash_hashtag(match: re.Match[str]) -> str:
        saved_hashtags.append(match.group(0))
        return f"__SAFE_HTAG_{len(saved_hashtags) - 1}__"

    protected_text = re.sub(r"#[^\s#]+", _stash_hashtag, text)

    # 2. Strip any legacy dot-tatweel artifacts if present in source text
    protected_text = re.sub(r"ـ\s*\.\s*ـ", "", protected_text)
    protected_text = re.sub(r"(?<=[ء-ي])\.(?=[ء-ي])", "", protected_text)

    # 3. Apply compiled regex substitution rules in priority order
    cleaned = protected_text
    for pattern, repl in _COMPILED_RULES:
        cleaned = pattern.sub(repl, cleaned)

    # 4. Clean up any accidental double spaces resulting from substitutions
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # 5. Restore protected hashtags
    for idx, ht in enumerate(saved_hashtags):
        cleaned = cleaned.replace(f"__SAFE_HTAG_{idx}__", ht)

    return cleaned


def obfuscate_for_facebook_display(text: str) -> str:
    """Public entry point for Facebook post captions, card titles, and video subtitles.

    Aliases directly to sanitize_sensitive_terms without dot-cutting to ensure clean, shadowban-free display.
    """
    return sanitize_sensitive_terms(text)


def sanitize_for_tts_audio(text: str) -> str:
    """Public entry point for spoken audio TTS.

    Replaces sensitive terms and strips any lingering tatweel or punctuation artifacts for fluent speech synthesis.
    """
    if not text:
        return ""
    sanitized = sanitize_sensitive_terms(text)
    # Strip tatweel for audio engines
    sanitized = re.sub(r"ـ+", "", sanitized)
    return re.sub(r"[ \t]+", " ", sanitized).strip()
