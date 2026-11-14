"""Facebook Algorithm Defense Shield & Broadcast Speech Sanitizer.

1. obfuscate_for_facebook_display(text: str) -> str:
   Cuts sensitive political/military terms with a subtle dash-dot 'ـ.ـ' so human readers
   understand them effortlessly while Facebook automated NLP classifiers & OCR text extractors
   fail to trigger shadowbans or reach suppression. Hashtags are preserved intact.

2. sanitize_for_tts_audio(text: str) -> str:
   Replaces high-risk terms with eloquent, broadcast-standard Arabic journalistic phrasing
   for TTS narration, ensuring smooth spoken cadence without spelling out punctuation.
"""

from __future__ import annotations

import re
from typing import Dict, List, Pattern, Tuple

# Mapping of sensitive Arabic roots/words to their obfuscated visual form.
# Ordered from longest/most specific to shortest to prevent prefix collisions.
SENSITIVE_DISPLAY_MAP: List[Tuple[str, str]] = [
    # Houthi movements & affiliations
    ("الحوثيين", "الحـ.ـوثيين"),
    ("الحوثيون", "الحـ.ـوثيون"),
    ("الحوثية", "الحـ.ـوثية"),
    ("الحوثي", "الحـ.ـوثي"),
    ("حوثيين", "حـ.ـوثيين"),
    ("حوثيون", "حـ.ـوثيون"),
    ("حوثية", "حـ.ـوثية"),
    ("حوثي", "حـ.ـوثي"),

    # Militias & armed groups
    ("ميليشيات", "ميليـ.ـشيات"),
    ("ميليشيا", "ميليـ.ـشيا"),
    ("مليشيات", "مليـ.ـشيات"),
    ("مليشيا", "مليـ.ـشيا"),

    # Airstrikes & artillery
    ("غارات", "غـ.ـارات"),
    ("غارة", "غـ.ـارة"),
    ("انفجارات", "انفـ.ـجارات"),
    ("انفجار", "انفـ.ـجار"),
    ("قصف", "قـ.ـصف"),
    ("صواريخ", "صـ.ـواريخ"),
    ("صاروخ", "صـ.ـاروخ"),

    # Casualties & deaths
    ("اغتيالات", "اغتيـ.ـالات"),
    ("اغتيال", "اغتيـ.ـال"),
    ("شهداء", "شـ.ـهداء"),
    ("شهيد", "شـ.ـهيد"),
    ("مصرع", "مصـ.ـرع"),
    ("مقتل", "مقـ.ـتل"),
    ("قتلى", "قتـ.ـلى"),
    ("قتيل", "قتـ.ـيل"),

    # Clashes & combat
    ("اشتباكات", "اشتبـ.ـاكات"),
    ("اشتباك", "اشتبـ.ـاك"),
    ("معارك", "معـ.ـارك"),
    ("معركة", "معـ.ـركة"),
    ("القتال", "القـ.ـتال"),
    ("قتال", "قـ.ـتال"),

    # Snipers & terrorism
    ("انتحاريين", "انتحـ.ـاريين"),
    ("انتحاري", "انتحـ.ـاري"),
    ("انتحارية", "انتحـ.ـارية"),
    ("إرهابيين", "إرهـ.ـابيين"),
    ("ارهابيين", "ارهـ.ـابيين"),
    ("إرهابية", "إرهـ.ـابية"),
    ("ارهابية", "ارهـ.ـابية"),
    ("إرهابي", "إرهـ.ـابي"),
    ("ارهابي", "ارهـ.ـابي"),
    ("قناصة", "قنـ.ـاصة"),
    ("قناص", "قنـ.ـاص"),
]

# Prefixes commonly attached to Arabic words (e.g. و، فـ، بـ، كـ، لـ، للـ، والـ، بالـ)
_PREFIX_PATTERN = r"(?<![ء-ي\w])(?P<pfx>[وفكلبل]|لل|وال|فال|بال|كال)?"


def _build_display_regexes() -> List[Tuple[Pattern[str], str]]:
    compiled: List[Tuple[Pattern[str], str]] = []
    for word, obf in SENSITIVE_DISPLAY_MAP:
        pattern = re.compile(_PREFIX_PATTERN + re.escape(word) + r"(?![ء-ي\w])")
        compiled.append((pattern, obf))
    return compiled


_COMPILED_DISPLAY_RULES: List[Tuple[Pattern[str], str]] = _build_display_regexes()


def obfuscate_for_facebook_display(text: str) -> str:
    """Obfuscate sensitive military/political terms for Facebook display and OCR protection.

    Rules:
      - Inserts 'ـ.ـ' into sensitive words.
      - Preserves prefixes (و، ب، ل، ال...).
      - Strict exception: Hashtags (words starting with #) are NEVER modified to preserve FB clickability.
    """
    if not text:
        return ""

    # Split text while keeping whitespace and hashtag tokens intact
    tokens = re.split(r"(\s+)", text)
    result_tokens: List[str] = []

    for token in tokens:
        if not token or token.isspace():
            result_tokens.append(token)
            continue

        # If token is a hashtag (starts with # or contains #), keep it intact
        if token.startswith("#") or "#" in token:
            result_tokens.append(token)
            continue

        # Apply obfuscation rules
        modified = token
        for pattern, obf_replacement in _COMPILED_DISPLAY_RULES:
            def _repl(match: re.Match[str]) -> str:
                pfx = match.group("pfx") or ""
                return f"{pfx}{obf_replacement}"

            modified = pattern.sub(_repl, modified)

        result_tokens.append(modified)

    return "".join(result_tokens)


# Replacement rules for speech synthesis (TTS) to maintain smooth eloquence and pass FB audio ASR checks.
# Multi-word replacements first, then single-word.
TTS_AUDIO_REPLACEMENTS: List[Tuple[Pattern[str], str]] = [
    (re.compile(r"(?<![ء-ي\w])(?:ميليشيا|مليشيا)\s+الحوثي(?:ين|ون)?(?![ء-ي\w])"), "مسلحي الجماعة"),
    (re.compile(r"(?<![ء-ي\w])(?:وميليشيا|ومليشيا)\s+الحوثي(?:ين|ون)?(?![ء-ي\w])"), "ومسلحي الجماعة"),
    (re.compile(r"(?<![ء-ي\w])(?:بميليشيا|بمليشيا)\s+الحوثي(?:ين|ون)?(?![ء-ي\w])"), "بمسلحي الجماعة"),
    (re.compile(r"(?<![ء-ي\w])(?:لميليشيا|لمليشيا)\s+الحوثي(?:ين|ون)?(?![ء-ي\w])"), "لمسلحي الجماعة"),

    (re.compile(r"(?<![ء-ي\w])الحوثيين(?![ء-ي\w])"), "مسلحي الجماعة"),
    (re.compile(r"(?<![ء-ي\w])والحوثيين(?![ء-ي\w])"), "ومسلحي الجماعة"),
    (re.compile(r"(?<![ء-ي\w])للحوثيين(?![ء-ي\w])"), "لمسلحي الجماعة"),
    (re.compile(r"(?<![ء-ي\w])بالحوثيين(?![ء-ي\w])"), "بمسلحي الجماعة"),

    (re.compile(r"(?<![ء-ي\w])الحوثي(?![ء-ي\w])"), "الجماعة"),
    (re.compile(r"(?<![ء-ي\w])والحوثي(?![ء-ي\w])"), "والجماعة"),
    (re.compile(r"(?<![ء-ي\w])للحوثي(?![ء-ي\w])"), "للجماعة"),
    (re.compile(r"(?<![ء-ي\w])بالحوثي(?![ء-ي\w])"), "بالجماعة"),

    (re.compile(r"(?<![ء-ي\w])مقتل(?![ء-ي\w])"), "سقوط"),
    (re.compile(r"(?<![ء-ي\w])ومقتل(?![ء-ي\w])"), "وسقوط"),
    (re.compile(r"(?<![ء-ي\w])بمقتل(?![ء-ي\w])"), "بسقوط"),
    (re.compile(r"(?<![ء-ي\w])مصرع(?![ء-ي\w])"), "سقوط"),
    (re.compile(r"(?<![ء-ي\w])ومصرع(?![ء-ي\w])"), "وسقوط"),

    (re.compile(r"(?<![ء-ي\w])قتلى(?![ء-ي\w])"), "ضحايا"),
    (re.compile(r"(?<![ء-ي\w])وقتلى(?![ء-ي\w])"), "وضحايا"),

    (re.compile(r"(?<![ء-ي\w])انتحاري(?![ء-ي\w])"), "مهاجم"),
    (re.compile(r"(?<![ء-ي\w])وانتحاري(?![ء-ي\w])"), "ومهاجم"),
    (re.compile(r"(?<![ء-ي\w])انتحارية(?![ء-ي\w])"), "هجومية"),

    (re.compile(r"(?<![ء-ي\w])إرهابي(?![ء-ي\w])"), "مسلح"),
    (re.compile(r"(?<![ء-ي\w])ارهابي(?![ء-ي\w])"), "مسلح"),
    (re.compile(r"(?<![ء-ي\w])وإرهابي(?![ء-ي\w])"), "ومسلح"),
    (re.compile(r"(?<![ء-ي\w])وارهابي(?![ء-ي\w])"), "ومسلح"),
    (re.compile(r"(?<![ء-ي\w])إرهابية(?![ء-ي\w])"), "مسلحة"),
    (re.compile(r"(?<![ء-ي\w])ارهابية(?![ء-ي\w])"), "مسلحة"),
    (re.compile(r"(?<![ء-ي\w])وإرهابية(?![ء-ي\w])"), "ومسلحة"),
    (re.compile(r"(?<![ء-ي\w])وارهابية(?![ء-ي\w])"), "ومسلحة"),
]


def sanitize_for_tts_audio(text: str) -> str:
    """Sanitize Arabic text specifically for TTS voice generation.

    1. Removes any 'ـ.ـ' or tatweel (ـ) so audio engines never pronounce dots.
    2. Swaps high-risk military/political triggers with eloquent journalistic phrasing.
    """
    if not text:
        return ""

    # First: remove any obfuscation artifacts (ـ.ـ, dots between Arabic letters, or tatweel)
    cleaned = re.sub(r"ـ\s*\.\s*ـ", "", text)
    cleaned = re.sub(r"(?<=[ء-ي])\.(?=[ء-ي])", "", cleaned)
    cleaned = re.sub(r"ـ+", "", cleaned)

    # Second: apply safe audio substitutions
    for pattern, replacement in TTS_AUDIO_REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)

    return re.sub(r"\s+", " ", cleaned).strip()
