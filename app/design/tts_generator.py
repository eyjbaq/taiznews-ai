"""Text-to-Speech engine using ElevenLabs API with seamless Edge-TTS Smart Fallback.

Primary Engine:
  - ElevenLabs REST API (eleven_turbo_v2_5) for hyper-realistic Arabic broadcast anchor voices.
  - Male rotation: Adam (pNInz6obpgDQGcFmaJgB) & Liam (TX3LPaxmHKxFdv7VOQHJ).
  - Female voice: Sarah (EXAVITQu4vr4xnSDxMaL).
Fallback Engine:
  - Microsoft Edge TTS (free, no API key required):
    - Male: Jamal Moroccan Anchor (ar-MA-JamalNeural)
    - Female: Maryam Yemeni Anchor (ar-YE-MaryamNeural)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "generated_audio"

# ElevenLabs engine toggle: when False, ElevenLabs API is bypassed and Gemini TTS is used directly
ENABLE_ELEVENLABS: bool = False
if os.getenv("ENABLE_ELEVENLABS", "").lower() in ("true", "1"):
    ENABLE_ELEVENLABS = True

# ElevenLabs configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Primary ElevenLabs Authorized Voices:
# Male Voices: Adam and Liam (alternated for male turns)
VOICE_MALE_ADAM = os.getenv("ELEVENLABS_VOICE_ADAM", "pNInz6obpgDQGcFmaJgB")
VOICE_MALE_LIAM = os.getenv("ELEVENLABS_VOICE_LIAM", "TX3LPaxmHKxFdv7VOQHJ")

# Female Voice: Sarah
VOICE_FEMALE_SARAH = os.getenv("ELEVENLABS_VOICE_SARAH", "EXAVITQu4vr4xnSDxMaL")

AUTHORIZED_VOICE_IDS = [VOICE_MALE_ADAM, VOICE_FEMALE_SARAH]
MALE_VOICE_IDS = [VOICE_MALE_ADAM, VOICE_MALE_LIAM]

# Primary Model: eleven_turbo_v2_5 (50% cheaper credits, fast generation)
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")

# Gemini TTS Models Cascading Priority Order (Single-attempt fallback: 3.8 -> 3.1 -> 2.5 -> 3.8 Lite)
GEMINI_TTS_CASCADING_MODELS = [
    "gemini-3.8-flash-tts",
    "gemini-3.1-flash-tts",
    "gemini-2.5-flash-tts",
    "gemini-2.5-flash-preview-tts",
    "gemini-3.8-flash-lite-tts",
]

# Gemini Authorized Voices:
# Female: Kore (كوري)
# Male: Orus (أوروس) & Charon (شارون)
GEMINI_VOICE_FEMALE_KORE = "Kore"
GEMINI_VOICE_MALE_ORUS = "Orus"
GEMINI_VOICE_MALE_CHARON = "Charon"

# Edge-TTS Fallback Voices
EDGE_VOICE_MALE = "ar-MA-JamalNeural"      # Jamal - Moroccan male news presenter
EDGE_VOICE_FEMALE = "ar-YE-MaryamNeural"   # Maryam - Yemeni female news presenter
DEFAULT_EDGE_VOICE = os.getenv("TTS_VOICE", EDGE_VOICE_MALE)

# Voice alternation state file
_VOICE_STATE_FILE = BASE_DIR / "data" / "voice_state.json"


def _read_voice_state() -> dict[str, Any]:
    """Read persistent voice state JSON file."""
    try:
        if _VOICE_STATE_FILE.exists():
            import json
            return json.loads(_VOICE_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _get_last_gender() -> str:
    """Retrieve the last used gender from state file."""
    state = _read_voice_state()
    gender = state.get("last_gender")
    if not gender:
        last_id = state.get("last_voice_id")
        if last_id == VOICE_FEMALE_SARAH:
            return "female"
        return "male"
    return gender


def _get_last_voice_id() -> Optional[str]:
    """Retrieve the last used voice ID from state file."""
    state = _read_voice_state()
    return state.get("last_voice_id")


def _get_next_voice() -> tuple[str, str, str]:
    """Alternate between male and female for each run.

    When it is male turn, alternate between Adam and Liam (checking which was used last).
    When it is female turn, use Sarah.

    Returns:
        tuple of (voice_id, voice_name, gender) where gender is 'male' or 'female'.
    """
    import json

    state = _read_voice_state()
    last_gender = state.get("last_gender")
    last_male_id = state.get("last_male_voice_id")
    last_voice_id = state.get("last_voice_id")

    if not last_gender and last_voice_id:
        last_gender = "female" if last_voice_id == VOICE_FEMALE_SARAH else "male"

    # Alternate gender
    if last_gender == "male":
        next_gender = "female"
        next_voice = VOICE_FEMALE_SARAH
        next_name = "سارة (Sarah)"
        chosen_male_id = last_male_id
    else:
        next_gender = "male"
        # Alternate between Adam and Liam
        if last_male_id == VOICE_MALE_ADAM:
            next_voice = VOICE_MALE_LIAM
            next_name = "ليام (Liam)"
        else:
            next_voice = VOICE_MALE_ADAM
            next_name = "آدم (Adam)"
        chosen_male_id = next_voice

    # Save state
    try:
        _VOICE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        new_state = {
            "last_gender": next_gender,
            "last_voice_id": next_voice,
            "last_male_voice_id": chosen_male_id,
            "last_voice_name": next_name,
        }
        _VOICE_STATE_FILE.write_text(
            json.dumps(new_state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        LOGGER.warning("Failed to save voice state: %s", exc)

    return next_voice, next_name, next_gender


def _get_next_gemini_voice() -> tuple[str, str, str]:
    """Alternate between female (Kore) and male (alternating between Orus & Charon) for Gemini TTS.

    Returns:
        tuple of (voice_id, voice_name, gender) where gender is 'male' or 'female'.
    """
    import json

    state = _read_voice_state()
    last_gender = state.get("last_gemini_gender") or state.get("last_gender", "male")
    last_male_id = state.get("last_gemini_male_voice_id")

    # Alternate gender
    if last_gender == "male":
        next_gender = "female"
        next_voice = GEMINI_VOICE_FEMALE_KORE
        next_name = "كوري (Kore)"
        chosen_male_id = last_male_id or GEMINI_VOICE_MALE_ORUS
    else:
        next_gender = "male"
        # Alternate between Orus and Charon
        if last_male_id == GEMINI_VOICE_MALE_ORUS:
            next_voice = GEMINI_VOICE_MALE_CHARON
            next_name = "شارون (Charon)"
        else:
            next_voice = GEMINI_VOICE_MALE_ORUS
            next_name = "أوروس (Orus)"
        chosen_male_id = next_voice

    # Save state
    try:
        _VOICE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        new_state = dict(state)
        new_state.update({
            "last_gender": next_gender,
            "last_gemini_gender": next_gender,
            "last_voice_id": next_voice,
            "last_voice_name": next_name,
            "last_gemini_male_voice_id": chosen_male_id,
        })
        _VOICE_STATE_FILE.write_text(
            json.dumps(new_state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        LOGGER.warning("Failed to save voice state: %s", exc)

    return next_voice, next_name, next_gender


# Edge-TTS Arabic broadcast voices
ARABIC_VOICES = [
    "ar-BH-AliNeural",        # Bahraini male - authoritative, broadcast anchor (Ali from Bahrain)
    "ar-YE-SalehNeural",      # Yemeni male - authentic Yemeni news presenter
    "ar-AE-HamdanNeural",     # Emirati male - confident and crisp
    "ar-EG-ShakirNeural",     # Egyptian male - professional broadcaster
    "ar-YE-MaryamNeural",     # Yemeni female - clear news presenter
    "ar-SA-ZariyahNeural",    # Saudi female - smooth broadcast tone
]
DEFAULT_EDGE_VOICE = os.getenv("TTS_VOICE", ARABIC_VOICES[0])


def _clean_word_for_subtitles(word: str) -> str:
    """Remove Arabic diacritics (tashkeel) and punctuation from a word."""
    cleaned = re.sub(r"[\u0617-\u061A\u064B-\u0652\u06D6-\u06ED]", "", word)
    return cleaned.strip(" \t\n\r،.؟!-:;()[]\"'")


def _get_audio_duration_ffprobe(audio_path: Path | str) -> float:
    """Retrieve audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return 0.0


def _generate_linear_word_timestamps(text: str, duration: float) -> list[dict[str, Any]]:
    """Evenly distribute words across the audio duration for clean subtitle sync."""
    words = [w for w in text.strip().split() if w]
    if not words or duration <= 0:
        return []

    total_words = len(words)
    pad_start = 0.15
    usable_duration = max(duration - 0.3, 0.5)
    word_duration = usable_duration / total_words

    timestamps: list[dict[str, Any]] = []
    for i, w in enumerate(words):
        start = pad_start + (i * word_duration)
        end = start + word_duration
        cleaned = _clean_word_for_subtitles(w)
        timestamps.append({
            "text": cleaned or w,
            "start": round(start, 3),
            "end": round(end, 3),
        })
    return timestamps


def _extract_words_from_alignment(alignment: dict[str, Any], raw_text: str, audio_duration: float = 0.0) -> list[dict[str, Any]]:
    """Extract word-level timestamps from ElevenLabs character-level alignment."""
    chars = alignment.get("characters", [])
    starts = alignment.get("character_start_times_seconds", [])
    ends = alignment.get("character_end_times_seconds", [])

    if not chars or not starts or not ends:
        return []

    words_from_text = [w for w in raw_text.strip().split() if w]
    char_words: list[dict[str, Any]] = []
    curr_chars: list[str] = []
    word_start: float | None = None
    prev_end: float = 0.0

    for c, s, e in zip(chars, starts, ends):
        if c.isspace():
            if curr_chars:
                char_words.append({
                    "start": round(word_start if word_start is not None else s, 3),
                    "end": round(prev_end, 3),
                })
                curr_chars = []
                word_start = None
        else:
            if word_start is None:
                word_start = s
            curr_chars.append(c)
            prev_end = e

    if curr_chars:
        char_words.append({
            "start": round(word_start if word_start is not None else 0.0, 3),
            "end": round(prev_end, 3),
        })

    # If alignment word count differs too much from text (>30%), alignment is unreliable
    if char_words and words_from_text:
        mismatch_ratio = abs(len(char_words) - len(words_from_text)) / len(words_from_text)
        if mismatch_ratio > 0.30:
            LOGGER.warning(
                "Alignment mismatch: %d aligned vs %d text words (%.0f%%). Using linear fallback.",
                len(char_words), len(words_from_text), mismatch_ratio * 100,
            )
            return []

    last_align_end = char_words[-1]["end"] if char_words else 0.0
    remaining_count = max(len(words_from_text) - len(char_words), 0)
    if remaining_count > 0 and audio_duration > last_align_end:
        remaining_dur = audio_duration - last_align_end - 0.1  # small end padding
        per_word = max(remaining_dur / remaining_count, 0.15)
    else:
        per_word = 0.35  # absolute fallback (shouldn't normally reach here)

    result: list[dict[str, Any]] = []
    for i, w in enumerate(words_from_text):
        if i < len(char_words):
            s_time = char_words[i]["start"]
            e_time = char_words[i]["end"]
        else:
            offset = i - len(char_words)
            s_time = round(last_align_end + (offset * per_word), 3)
            e_time = round(s_time + per_word, 3)

        result.append({
            "text": _clean_word_for_subtitles(w) or w,
            "start": s_time,
            "end": e_time,
        })
    return result


def _generate_tts_elevenlabs(
    text: str,
    output_path: Path,
    voice_id: Optional[str] = None,
    timeout: int = 40,
) -> tuple[bool, list[dict[str, Any]], str]:
    """Generate audio via ElevenLabs REST API using eleven_turbo_v2_5."""
    api_key = os.getenv("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY)
    if not api_key:
        return False, [], "male"

    if voice_id:
        selected_voice = voice_id
        gender = "female" if voice_id == VOICE_FEMALE_SARAH else "male"
        voice_label = f"🎙️ {selected_voice[:8]}"
    else:
        selected_voice, voice_name, gender = _get_next_voice()
        gender_icon = "👨 ذكر" if gender == "male" else "👩 أنثى"
        voice_label = f"{gender_icon} - {voice_name}"

    print(f"🎤 المذيع المختار لهذا الريلز: {voice_label} ({selected_voice[:8]}...)")
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.38,
            "similarity_boost": 0.80,
            "style": 0.25,
            "use_speaker_boost": True,
            "speed": 1.05,
        },
        "speed": 1.05,
    }

    current_voice = selected_voice
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{current_voice}/with-timestamps"

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)

        if resp.status_code != 200:
            # Fallback to standard endpoint if with-timestamps is unavailable
            std_url = f"https://api.elevenlabs.io/v1/text-to-speech/{current_voice}"
            resp = requests.post(std_url, headers=headers, json=payload, timeout=timeout)

        if resp.status_code != 200:
            LOGGER.warning("ElevenLabs API returned HTTP %d: %s", resp.status_code, resp.text[:200])
            return False, [], gender

        content_type = resp.headers.get("content-type", "")
        word_timestamps: list[dict[str, Any]] = []
        alignment: dict[str, Any] = {}

        if "application/json" in content_type:
            data = resp.json()
            b64_audio = data.get("audio_base64", "")
            if b64_audio:
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(b64_audio))
            alignment = data.get("alignment", {})
        else:
            with open(output_path, "wb") as f:
                f.write(resp.content)

        if not output_path.exists() or output_path.stat().st_size < 1000:
            return False, [], gender

        duration = _get_audio_duration_ffprobe(output_path)

        # Extract alignment AFTER knowing actual audio duration
        if alignment:
            word_timestamps = _extract_words_from_alignment(alignment, text, duration)
        if not word_timestamps and duration > 0:
            word_timestamps = _generate_linear_word_timestamps(text, duration)

        return True, word_timestamps, gender

    except Exception as exc:
        LOGGER.warning("ElevenLabs generation error: %s", exc)
        return False, [], gender


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop that works in all contexts."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


async def _generate_tts_edge_async(
    text: str,
    output_path: str,
    voice: str = DEFAULT_EDGE_VOICE,
    rate: str = "+10%",
    pitch: str = "+0Hz",
    volume: str = "+10%",
) -> tuple[bool, list[dict[str, Any]]]:
    """Internal async function to stream Edge-TTS audio and capture word timestamps."""
    try:
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
            boundary="WordBoundary",
        )
        words: list[dict[str, Any]] = []
        with open(output_path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    clean_word = _clean_word_for_subtitles(chunk["text"])
                    words.append({
                        "start": round(chunk["offset"] / 10_000_000, 3),
                        "end": round((chunk["offset"] + chunk["duration"]) / 10_000_000, 3),
                        "text": clean_word or chunk["text"],
                    })
        return True, words
    except Exception as exc:
        LOGGER.error("edge-tts generation failed: %s", exc)
        return False, []


def _run_async_edge_tts(
    text: str, output_path: str, voice: str, rate: str, pitch: str, volume: str = "+10%"
) -> tuple[bool, list[dict[str, Any]]]:
    """Helper to run async Edge-TTS in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _generate_tts_edge_async(text, output_path, voice, rate, pitch, volume)
        )
    finally:
        loop.close()


def _generate_tts_gemini(
    text: str,
    output_path: Path,
    voice_id: Optional[str] = None,
) -> tuple[bool, list[dict[str, Any]], str]:
    """Synthesize speech using Google Gemini Flash TTS with single-attempt model cascading.

    Prompt is sent STRICTLY verbatim with no instruction wrapping.
    Speed is adjusted slightly via FFmpeg (atempo=1.06) for crisp broadcast delivery.
    Word timestamps are evenly distributed across actual audio duration.
    """
    from google import genai
    from google.genai import types

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_VOICE_LAB_KEY")
    if not gemini_key:
        LOGGER.error("GEMINI_API_KEY is not set.")
        return False, [], "male"

    if voice_id:
        selected_voice = voice_id
        gender = "female" if voice_id == GEMINI_VOICE_FEMALE_KORE else "male"
        voice_label = f"🎙️ {selected_voice}"
    else:
        selected_voice, voice_name, gender = _get_next_gemini_voice()
        gender_icon = "👨 ذكر" if gender == "male" else "👩 أنثى"
        voice_label = f"{gender_icon} - {voice_name}"

    print(f"🎤 المذيع المختار لهذا الريلز (Gemini): {voice_label} ({selected_voice})")

    try:
        client = genai.Client(api_key=gemini_key)
    except Exception as init_err:
        LOGGER.error("Failed to initialize Google GenAI client: %s", init_err)
        return False, [], gender

    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=selected_voice
                )
            )
        ),
    )

    response = None

    for target_model in GEMINI_TTS_CASCADING_MODELS:
        print(f"🤖 [Gemini TTS] محاولة التوليد عبر النموذج: [{target_model}] (محاولة واحدة فقط)...")
        try:
            # Strictly send raw text verbatim without instructions
            response = client.models.generate_content(
                model=target_model,
                contents=text,
                config=config,
            )
            if (
                response
                and response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
            ):
                print(f"✅ نجح التوليد الصوتي بنجاح باستخدام النموذج: [{target_model}]")
                break
            else:
                print(f"⚠️ النموذج [{target_model}] أعاد استجابة فارغة. الانتقال الفوري للنموذج التالي...")
        except Exception as api_err:
            err_str = str(api_err)
            err_lower = err_str.lower()
            if "429" in err_str or "resource_exhausted" in err_lower or "quota" in err_lower:
                print(f"⛔ تم بلوغ حد الحصة للنموذج [{target_model}] (429 Quota Exceeded).")
            elif "404" in err_str or "not_found" in err_lower:
                print(f"⚠️ النموذج [{target_model}] غير متاح (404 Not Found).")
            else:
                print(f"⚠️ خطأ أثناء الطلب بالنموذج [{target_model}]: {err_str[:120]}.")
            print("🔄 جاري الانتقال الفوري للنموذج التالي في القائمة...")
            continue

    if not response or not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
        LOGGER.error("Gemini TTS returned no audio candidates across all cascading models.")
        return False, [], gender

    try:
        candidate = response.candidates[0]
        part = candidate.content.parts[0]
        if not hasattr(part, "inline_data") or not part.inline_data or not part.inline_data.data:
            LOGGER.error("Gemini TTS candidate contains no audio data.")
            return False, [], gender

        raw_pcm_data = part.inline_data.data
        temp_pcm = output_path.with_suffix(".raw.pcm")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_pcm, "wb") as f:
            f.write(raw_pcm_data)

        # Convert raw PCM (s16le 24000Hz mono) to mp3 with atempo=1.06
        cmd = [
            "ffmpeg", "-y",
            "-f", "s16le",
            "-ar", "24000",
            "-ac", "1",
            "-i", str(temp_pcm),
            "-af", "atempo=1.06",
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            str(output_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if temp_pcm.exists():
            temp_pcm.unlink(missing_ok=True)

        if res.returncode != 0 or not output_path.exists() or output_path.stat().st_size < 1000:
            LOGGER.error("FFmpeg PCM conversion failed: %s", res.stderr[-300:] if res else "")
            return False, [], gender

        duration = _get_audio_duration_ffprobe(output_path)
        word_timestamps = _generate_linear_word_timestamps(text, duration)
        return True, word_timestamps, gender

    except Exception as exc:
        LOGGER.error("Gemini TTS synthesis processing error: %s", exc)
        return False, [], gender


def generate_news_audio(
    text: str,
    output_dir: Path | str | None = None,
    voice: str = DEFAULT_EDGE_VOICE,
    slug: str = "news_tts",
    rate: str = "+10%",
    pitch: str = "+0Hz",
    volume: str = "+10%",
    vocalized_fallback_text: Optional[str] = None,
) -> tuple[Optional[str], list[dict[str, Any]]]:
    """Generate Arabic TTS audio and word timestamps.

    Pipeline:
      1. Primary (if ENABLE_ELEVENLABS=True): ElevenLabs REST API.
      2. Primary (if ENABLE_ELEVENLABS=False, or ElevenLabs fails): Google Gemini Flash TTS
         - Alternates between Female (Kore) and Male (Orus / Charon).
         - Cascades across models: 3.8 -> 3.1 -> 2.5 -> 3.8 Lite (1 attempt each).
      3. Fallback: Edge-TTS (ar-MA-JamalNeural / ar-YE-MaryamNeural) if all upstream engines fail.
    """
    if not text or not text.strip():
        LOGGER.warning("Empty text provided for TTS generation.")
        return None, []

    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    output_file = out_dir / f"{slug}_{timestamp}.mp3"
    clean_text = _prepare_text_for_narration(text)
    gender = "male"

    # 1. ElevenLabs REST API (bypassed if ENABLE_ELEVENLABS is False)
    if ENABLE_ELEVENLABS:
        print(f"🎙️ جاري توليد التعليق الصوتي الإخباري عبر محرك ElevenLabs API ({ELEVENLABS_MODEL_ID})...")
        success, word_timestamps, gender = _generate_tts_elevenlabs(clean_text, output_file)
        if success and output_file.exists() and output_file.stat().st_size > 1000:
            file_size_kb = output_file.stat().st_size / 1024
            print(f"✅ تم توليد التعليق الصوتي عبر ElevenLabs بنجاح ({file_size_kb:.0f} KB) وتحديد {len(word_timestamps)} كلمة!")
            return str(output_file), word_timestamps
        print("🔄 [التبديل التلقائي] تعذر التوليد عبر ElevenLabs، جاري الانتقال إلى محرك Gemini TTS...")
    else:
        print("⚡ [محرك ElevenLabs معطل - ENABLE_ELEVENLABS=False] جاري الانتقال مباشرة إلى محرك Gemini TTS...")

    # 2. Gemini Flash TTS Engine
    print("🎙️ جاري توليد التعليق الصوتي الإخباري عبر محرك Google Gemini Flash TTS...")
    success, word_timestamps, gender = _generate_tts_gemini(clean_text, output_file)
    if success and output_file.exists() and output_file.stat().st_size > 1000:
        file_size_kb = output_file.stat().st_size / 1024
        print(f"✅ تم توليد التعليق الصوتي عبر Gemini TTS بنجاح ({file_size_kb:.0f} KB) وتحديد {len(word_timestamps)} كلمة!")
        return str(output_file), word_timestamps

    # 3. Fallback Engine: Microsoft Edge-TTS
    print("🔄 [التبديل التلقائي] تعذر التوليد عبر Gemini TTS، جاري التبديل للمحرك الاحتياطي (Edge-TTS)...")
    if gender == "female":
        edge_voice = EDGE_VOICE_FEMALE  # ar-YE-MaryamNeural
        print(f"🎤 [Edge-TTS البديل] مذيعة: مريم ({edge_voice})")
    else:
        edge_voice = EDGE_VOICE_MALE    # ar-MA-JamalNeural
        print(f"🎤 [Edge-TTS البديل] مذيع: جمال المغربي ({edge_voice})")

    try:
        loop = _get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(_run_async_edge_tts, clean_text, str(output_file), edge_voice, rate, pitch, volume)
                success, word_timestamps = future.result(timeout=60)
        else:
            success, word_timestamps = loop.run_until_complete(
                _generate_tts_edge_async(clean_text, str(output_file), edge_voice, rate, pitch, volume)
            )
    except Exception as exc:
        LOGGER.error("Edge-TTS generation failed: %s", exc)
        return None, []

    if success and output_file.exists() and output_file.stat().st_size > 1000:
        file_size_kb = output_file.stat().st_size / 1024
        print(f"✅ تم توليد التعليق الصوتي عبر المحرك الاحتياطي Edge-TTS ({file_size_kb:.0f} KB) وتحديد {len(word_timestamps)} كلمة.")
        return str(output_file), word_timestamps

    print("❌ تعذر توليد التعليق الصوتي عبر جميع المحركات.")
    return None, []


def _prepare_text_for_narration(text: str) -> str:
    """Clean and prepare Arabic text for natural TTS narration."""
    clean = text.strip()
    clean = re.sub(r"#\S+", "", clean)
    clean = re.sub(r"https?://\S+", "", clean)
    clean = re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
        r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
        r"\U0000FE00-\U0000FE0F\U0000200D]+",
        "",
        clean,
    )
    # Strip any source attribution phrases so news starts directly
    clean = re.sub(r"نقلا[ً]? عن [^\s،.]+( [^\s،.]+)?", "", clean)
    clean = re.sub(r"بحسب [^\s،.]+( [^\s،.]+)?", "", clean)
    clean = re.sub(r"وفقا[ً]? ل[^\s،.]+( [^\s،.]+)?", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean
