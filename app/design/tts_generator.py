"""Text-to-Speech engine using ElevenLabs API with seamless Edge-TTS Smart Fallback.

Primary Voice Alternation:
  - Turn 1: آدم (Adam) - ElevenLabs (Male: pNInz6obpgDQGcFmaJgB)
    * Text is sent UNVOCALIZED (بدون تشكيل) to save credits and maintain natural brisk delivery.
  - Turn 2: مريم اليمنية (Maryam) - Microsoft Edge TTS (Female: ar-YE-MaryamNeural)
    * Text is sent FULLY VOCALIZED (مشكول بالكامل) with accurate Arabic diacritics for broadcast eloquence.

Fallback Engine:
  - Automatic fallback between Adam and Maryam if either engine is temporarily unavailable.
  - Optional Google Gemini Flash TTS fallback if enabled.
"""

from __future__ import annotations

import asyncio
import base64
import json
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

# ElevenLabs engine toggle
ENABLE_ELEVENLABS: bool = True
if os.getenv("ENABLE_ELEVENLABS", "").lower() in ("false", "0", "no"):
    ENABLE_ELEVENLABS = False

# ElevenLabs configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Primary ElevenLabs Authorized Voices:
VOICE_MALE_ADAM = os.getenv("ELEVENLABS_VOICE_ADAM", "pNInz6obpgDQGcFmaJgB")

# Edge-TTS Broadcast Voices:
EDGE_VOICE_FEMALE_MARYAM = "ar-YE-MaryamNeural"   # Maryam - Yemeni female news presenter
EDGE_VOICE_MALE_JAMAL = "ar-MA-JamalNeural"       # Jamal - Moroccan male news presenter
DEFAULT_EDGE_VOICE = os.getenv("TTS_VOICE", EDGE_VOICE_FEMALE_MARYAM)

# Two-Voice Alternation: Adam (Male/ElevenLabs) <-> Maryam (Female/Edge-TTS)
VOICE_ROTATION_LIST = [
    {
        "id": "adam",
        "engine": "elevenlabs",
        "voice_id": VOICE_MALE_ADAM,
        "name": "آدم (Adam)",
        "gender": "male",
    },
    {
        "id": "maryam",
        "engine": "edge",
        "voice_id": EDGE_VOICE_FEMALE_MARYAM,
        "name": "مريم اليمنية (Maryam)",
        "gender": "female",
    },
]

# Primary ElevenLabs Model: eleven_multilingual_v2
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

# Gemini TTS toggle (disabled by default)
ENABLE_GEMINI_TTS: bool = os.getenv("ENABLE_GEMINI_TTS", "false").lower() in ("true", "1")

# Gemini TTS Models Cascading Priority Order (active only if ENABLE_GEMINI_TTS is True)
GEMINI_TTS_CASCADING_MODELS = [
    "gemini-3.8-flash-tts",
    "gemini-3.1-flash-tts",
    "gemini-2.5-flash-tts",
    "gemini-2.5-flash-preview-tts",
    "gemini-3.8-flash-lite-tts",
]

# Gemini Authorized Voices:
GEMINI_VOICE_FEMALE_KORE = "Kore"
GEMINI_VOICE_MALE_ORUS = "Orus"
GEMINI_VOICE_MALE_CHARON = "Charon"

# Voice alternation state file
_VOICE_STATE_FILE = BASE_DIR / "data" / "voice_state.json"


def _read_voice_state() -> dict[str, Any]:
    """Read persistent voice state JSON file."""
    try:
        if _VOICE_STATE_FILE.exists():
            return json.loads(_VOICE_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _get_next_rotated_voice() -> dict[str, Any]:
    """Rotate sequentially between Adam (ElevenLabs) and Maryam (Edge-TTS)."""
    state = _read_voice_state()
    last_index = state.get("last_voice_rotation_index", -1)
    next_index = (last_index + 1) % len(VOICE_ROTATION_LIST)

    chosen = VOICE_ROTATION_LIST[next_index]

    state["last_voice_rotation_index"] = next_index
    state["last_voice_id"] = chosen["voice_id"]
    state["last_gender"] = chosen["gender"]
    state["last_voice_name"] = chosen["name"]

    try:
        _VOICE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _VOICE_STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("Could not persist voice state: %s", exc)

    return chosen


def _get_next_voice() -> tuple[str, str, str]:
    """Backwards compatibility helper: returns the next voice tuple."""
    chosen = _get_next_rotated_voice()
    return chosen["voice_id"], chosen["name"], chosen["gender"]


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


def _extract_timestamps_with_whisper(
    audio_path: Path,
    original_text: str,
) -> list[dict[str, Any]]:
    """Extract precise word timestamps from audio using faster-whisper STT."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        LOGGER.warning("faster-whisper not installed, falling back to linear timestamps.")
        return []

    try:
        import numpy as np

        print("🎯 [Whisper Sync] جاري تحليل الصوت لاستخراج توقيتات دقيقة لكل كلمة...")
        model = WhisperModel("tiny", compute_type="int8", device="cpu")

        # Bypass av.open() which fails with newer PyAV versions by decoding with ffmpeg directly
        cmd = [
            "ffmpeg", "-nostdin", "-threads", "0", "-i", str(audio_path),
            "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "-"
        ]
        audio_data = subprocess.run(cmd, capture_output=True, check=True).stdout
        audio_array = np.frombuffer(audio_data, np.int16).flatten().astype(np.float32) / 32768.0

        segments, info = model.transcribe(
            audio_array,
            word_timestamps=True,
            language="ar",
        )

        whisper_words: list[dict[str, Any]] = []
        for segment in segments:
            if segment.words:
                for w in segment.words:
                    cleaned = _clean_word_for_subtitles(w.word)
                    if cleaned:
                        whisper_words.append({
                            "text": cleaned,
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                        })

        if not whisper_words:
            return []

        original_words = [_clean_word_for_subtitles(w) for w in original_text.split() if w.strip()]
        original_words = [w for w in original_words if w]

        if not original_words:
            return whisper_words

        import difflib
        whisper_texts = [w["text"] for w in whisper_words]
        matcher = difflib.SequenceMatcher(None, whisper_texts, original_words)

        result: list[dict[str, Any]] = [{"text": w, "start": None, "end": None} for w in original_words]

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal' or (tag == 'replace' and (i2 - i1) == (j2 - j1)):
                for w_idx, o_idx in zip(range(i1, i2), range(j1, j2)):
                    result[o_idx]["start"] = whisper_words[w_idx]["start"]
                    result[o_idx]["end"] = whisper_words[w_idx]["end"]

        # Linear interpolation for any unmatched missing words
        i = 0
        n = len(result)
        while i < n:
            if result[i]["start"] is None:
                start_missing = i
                while i < n and result[i]["start"] is None:
                    i += 1
                end_missing = i
                count = end_missing - start_missing

                prev_time = result[start_missing - 1]["end"] if start_missing > 0 and result[start_missing - 1]["end"] is not None else 0.15
                next_time = result[end_missing]["start"] if end_missing < n and result[end_missing]["start"] is not None else None

                if next_time is None:
                    for j in range(count):
                        result[start_missing + j]["start"] = round(prev_time + (j * 0.35), 3)
                        result[start_missing + j]["end"] = round(prev_time + ((j + 1) * 0.35), 3)
                else:
                    gap = next_time - prev_time
                    if gap <= 0:
                        gap = 0.1 * count
                    time_per_word = gap / count
                    for j in range(count):
                        w_start = prev_time + (j * time_per_word)
                        w_end = prev_time + ((j + 1) * time_per_word)
                        result[start_missing + j]["start"] = round(w_start, 3)
                        result[start_missing + j]["end"] = round(w_end, 3)
            else:
                i += 1

        print(f"✅ [Whisper Sync] تم استخراج وتصحيح توقيتات {len(result)} كلمة باحترافية!")
        return result

    except Exception as exc:
        LOGGER.warning("Whisper timestamp extraction failed: %s", exc)
        return []


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
        remaining_dur = audio_duration - last_align_end - 0.1
        per_word = max(remaining_dur / remaining_count, 0.15)
    else:
        per_word = 0.35

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
    """Generate audio via ElevenLabs REST API using natural broadcast settings."""
    api_key = os.getenv("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY)
    if not api_key:
        return False, [], "male"

    selected_voice = voice_id or VOICE_MALE_ADAM
    gender = "male"
    voice_label = "👨 ذكر - آدم (Adam)" if selected_voice == VOICE_MALE_ADAM else f"🎙️ {selected_voice[:8]}"

    print(f"🎤 المذيع المختار لهذا الريلز: {voice_label} ({selected_voice[:8]}...)")
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "User-Agent": "TaizNews/2.0",
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.50,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
            "speed": 1.15,
        },
        "speed": 1.15,
    }

    current_voice = selected_voice
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{current_voice}/with-timestamps"

    try:
        session = requests.Session()
        resp = session.post(url, headers=headers, json=payload, timeout=timeout)

        if resp.status_code != 200:
            std_url = f"https://api.elevenlabs.io/v1/text-to-speech/{current_voice}"
            resp = session.post(std_url, headers=headers, json=payload, timeout=timeout)

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
    """Synthesize speech using Google Gemini Flash TTS with single-attempt model cascading."""
    from google import genai
    from google.genai import types

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_VOICE_LAB_KEY")
    if not gemini_key:
        LOGGER.error("GEMINI_API_KEY is not set.")
        return False, [], "male"

    selected_voice = voice_id or GEMINI_VOICE_MALE_ORUS
    gender = "female" if selected_voice == GEMINI_VOICE_FEMALE_KORE else "male"
    voice_label = f"🎙️ {selected_voice}"

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
        except Exception as api_err:
            LOGGER.warning("Gemini model %s error: %s", target_model, api_err)
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
        word_timestamps = _extract_timestamps_with_whisper(output_path, text)
        if not word_timestamps:
            print("⚠️ [Fallback] تعذر استخراج التوقيتات عبر Whisper، جاري استخدام التوزيع الخطي البديل...")
            word_timestamps = _generate_linear_word_timestamps(text, duration)
        return True, word_timestamps, gender

    except Exception as exc:
        LOGGER.error("Gemini TTS synthesis processing error: %s", exc)
        return False, [], gender


# ==============================================================================
# Text Cleaning & Tashkeel Handling
# ==============================================================================

def _prepare_vocalized_text(text: str) -> str:
    """Clean and prepare Arabic text preserving 100% full tashkeel / diacritics for Maryam (Edge-TTS)."""
    if not text:
        return ""
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


def _prepare_unvocalized_text(text: str) -> str:
    """Clean Arabic text stripping all full tashkeel (protecting 'تَعِز') for Adam (ElevenLabs)."""
    if not text:
        return ""
    clean = _prepare_vocalized_text(text)

    # Protect 'تَعِز'
    protected_words = {
        "تَعِز": "___PROTECTED_TAIZ___",
        "تَعِزّ": "___PROTECTED_TAIZ___",
    }
    for word, placeholder in protected_words.items():
        clean = clean.replace(word, placeholder)

    # Strip excessive Arabic diacritics
    clean = re.sub(r"[\u0617-\u061A\u064B-\u0652\u06D6-\u06ED]", "", clean)
    clean = clean.replace("___PROTECTED_TAIZ___", "تَعِز")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _prepare_text_for_narration(text: str) -> str:
    """Backward compatibility helper: cleans text preserving vocalization."""
    return _prepare_vocalized_text(text)


# ==============================================================================
# Public API: generate_news_audio
# ==============================================================================

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
    """Generate Arabic TTS audio and word timestamps with Adam (ElevenLabs) / Maryam (Edge-TTS) rotation.

    Voice & Text Rules:
      - Turn 1: آدم (Adam) - ElevenLabs (Male): text sent UNVOCALIZED (بدون تشكيل) to save credits and ensure brisk natural flow.
      - Turn 2: مريم اليمنية (Maryam) - Edge-TTS (Female): text sent FULLY VOCALIZED (مشكول بالكامل) for classical Arabic eloquence.
      - Fallback: Gracefully switches between Adam and Maryam if an engine is unavailable, with Gemini TTS backup.
    """
    if not text or not text.strip():
        LOGGER.warning("Empty text provided for TTS generation.")
        return None, []

    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    output_file = out_dir / f"{slug}_{timestamp}.mp3"

    # 1. Select next voice from rotation (Adam <-> Maryam)
    voice_info = _get_next_rotated_voice()
    target_engine = voice_info["engine"]
    target_voice_id = voice_info["voice_id"]
    voice_name = voice_info["name"]
    gender = voice_info["gender"]

    print(f"🎙️ [الدورة الصوتية] المذيع المختار لهذا الفيديو: {voice_name} ({target_voice_id}) [المحرك: {target_engine.upper()}]")

    # 2. Prepare text specifically for the selected voice
    if target_engine == "elevenlabs":
        # Adam: Plain unvocalized text
        narration_text = _prepare_unvocalized_text(text)
        print("📝 [تجهيز النص] إرسال النص إلى آدم (ElevenLabs) بدون تشكيل لحفظ الرصيد وسرعة الإلقاء.")
    else:
        # Maryam: Fully vocalized text with all tashkeel
        raw_vocalized = vocalized_fallback_text or text
        narration_text = _prepare_vocalized_text(raw_vocalized)
        print("📝 [تجهيز النص] إرسال النص إلى مريم (Edge-TTS) مشكولاً بالكامل (تشكيل فصيح).")

    success = False
    word_timestamps = []

    # 3. Execute primary engine
    if target_engine == "elevenlabs" and ENABLE_ELEVENLABS:
        print(f"🎙️ جاري توليد التعليق الصوتي عبر محرك ElevenLabs API ({ELEVENLABS_MODEL_ID})...")
        success, word_timestamps, gender = _generate_tts_elevenlabs(
            text=narration_text,
            output_path=output_file,
            voice_id=target_voice_id,
        )
        if not success or not output_file.exists() or output_file.stat().st_size < 1000:
            print("🔄 [التبديل التلقائي] تعذر التوليد عبر ElevenLabs، الانتقال الفوري للبديل مريم اليمنية (Edge-TTS)...")
            target_engine = "edge"
            target_voice_id = EDGE_VOICE_FEMALE_MARYAM
            # Switch to vocalized text for Maryam
            raw_vocalized = vocalized_fallback_text or text
            narration_text = _prepare_vocalized_text(raw_vocalized)

    if target_engine == "edge" or not success:
        voice_label = "مريم اليمنية (أنثى)" if target_voice_id == EDGE_VOICE_FEMALE_MARYAM else f"{target_voice_id}"
        print(f"🎙️ [Edge-TTS] جاري التوليد بصوت: {voice_label} ({target_voice_id})...")
        try:
            loop = _get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(_run_async_edge_tts, narration_text, str(output_file), target_voice_id, rate, pitch, volume)
                    success, word_timestamps = future.result(timeout=60)
            else:
                success, word_timestamps = loop.run_until_complete(
                    _generate_tts_edge_async(narration_text, str(output_file), target_voice_id, rate, pitch, volume)
                )
        except Exception as exc:
            LOGGER.error("Edge-TTS generation failed: %s", exc)
            success = False

        if not success and ENABLE_ELEVENLABS:
            print("🔄 [التبديل الاحتياطي] تعذر توليد Edge-TTS، التبديل إلى ElevenLabs بصوت آدم...")
            unvoc_text = _prepare_unvocalized_text(text)
            success, word_timestamps, gender = _generate_tts_elevenlabs(
                text=unvoc_text,
                output_path=output_file,
                voice_id=VOICE_MALE_ADAM,
            )

        if not success and ENABLE_GEMINI_TTS:
            print("🔄 [بديل الطوارئ] تجربة Gemini TTS...")
            success, word_timestamps, gender = _generate_tts_gemini(
                narration_text, output_file, voice_id=GEMINI_VOICE_MALE_ORUS
            )

    if success and output_file.exists() and output_file.stat().st_size > 1000:
        file_size_kb = output_file.stat().st_size / 1024
        print(f"✅ تم توليد التعليق الصوتي بنجاح ({file_size_kb:.0f} KB) وتحديد {len(word_timestamps)} كلمة متزامنة!")
        return str(output_file), word_timestamps

    print("❌ تعذر توليد التعليق الصوتي عبر جميع المحركات المتاحة.")
    return None, []
