"""Text-to-Speech engine using Microsoft Edge TTS for Arabic news narration.

Converts editorial post body text into a high-quality Arabic .mp3 audio file
using the edge-tts library (free, no API key required).
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import os
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "generated_audio"

# Arabic news broadcast voices
ARABIC_VOICES = [
    "ar-SA-HamedNeural",      # Saudi male - authoritative, deep broadcast anchor
    "ar-YE-SalehNeural",      # Yemeni male - authentic Yemeni news presenter
    "ar-AE-HamdanNeural",     # Emirati male - confident and crisp
    "ar-EG-ShakirNeural",     # Egyptian male - professional broadcaster
    "ar-YE-MaryamNeural",     # Yemeni female - clear news presenter
    "ar-SA-ZariyahNeural",    # Saudi female - smooth broadcast tone
]

DEFAULT_VOICE = os.getenv("TTS_VOICE", ARABIC_VOICES[0])


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop that works in all contexts."""
    try:
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


async def _generate_tts_async(
    text: str,
    output_path: str,
    voice: str = DEFAULT_VOICE,
    rate: str = "+2%",
    pitch: str = "-2Hz",
    volume: str = "+15%",
) -> tuple[bool, list[dict[str, Any]]]:
    """Internal async function to stream TTS audio and capture word timestamps."""
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
                    words.append({
                        "start": chunk["offset"] / 10_000_000,
                        "end": (chunk["offset"] + chunk["duration"]) / 10_000_000,
                        "text": chunk["text"],
                    })
        return True, words
    except Exception as exc:
        LOGGER.error("edge-tts generation failed: %s", exc)
        return False, []


def generate_news_audio(
    text: str,
    output_dir: Path | str | None = None,
    voice: str = DEFAULT_VOICE,
    slug: str = "news_tts",
    rate: str = "+2%",
    pitch: str = "-2Hz",
    volume: str = "+15%",
) -> tuple[Optional[str], list[dict[str, Any]]]:
    """Generate Arabic TTS audio and word timestamps for synchronized subtitle animation.

    Args:
        text: The Arabic news text to narrate.
        output_dir: Output directory.
        voice: Voice name.
        slug: Filename prefix.
        rate: Speech rate adjustment.
        pitch: Pitch adjustment.
        volume: Volume adjustment.

    Returns:
        Tuple of (audio_file_path, list of word timestamp dicts).
    """
    if not text or not text.strip():
        LOGGER.warning("Empty text provided for TTS generation.")
        print("⚠️ لم يتم توفير نص للتحويل الصوتي.")
        return None, []

    out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    output_file = out_dir / f"{slug}_{timestamp}.mp3"

    print(f"🎙️ جاري توليد التعليق الصوتي وتوقيت الكلمات بصوت [{voice}]...")

    # Clean the text for better narration
    clean_text = _prepare_text_for_narration(text)
    word_timestamps: list[dict[str, Any]] = []
    success = False

    # Run the async TTS generation
    try:
        loop = _get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(_run_async_tts, clean_text, str(output_file), voice, rate, pitch, volume)
                success, word_timestamps = future.result(timeout=60)
        else:
            success, word_timestamps = loop.run_until_complete(
                _generate_tts_async(clean_text, str(output_file), voice, rate, pitch, volume)
            )
    except Exception as exc:
        LOGGER.error("TTS generation failed: %s", exc)
        print(f"❌ فشل التوليد الصوتي: {exc}")
        return None, []

    if success and output_file.exists() and output_file.stat().st_size > 1000:
        file_size_kb = output_file.stat().st_size / 1024
        print(f"✅ تم توليد التعليق الصوتي ({file_size_kb:.0f} KB) وتحديد توقيت {len(word_timestamps)} كلمة بنجاح!")
        return str(output_file), word_timestamps
    else:
        print("❌ فشل في توليد ملف صوتي صالح.")
        # Try fallback voices
        for fallback_voice in ARABIC_VOICES:
            if fallback_voice == voice:
                continue
            print(f"🔄 تجربة صوت بديل: [{fallback_voice}]...")
            try:
                loop = _get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(_run_async_tts, clean_text, str(output_file), fallback_voice, rate, pitch, volume)
                        success, word_timestamps = future.result(timeout=60)
                else:
                    success, word_timestamps = loop.run_until_complete(
                        _generate_tts_async(clean_text, str(output_file), fallback_voice, rate, pitch, volume)
                    )
                if success and output_file.exists() and output_file.stat().st_size > 1000:
                    file_size_kb = output_file.stat().st_size / 1024
                    print(f"✅ تم توليد التعليق الصوتي بالصوت البديل ({file_size_kb:.0f} KB) وتوقيت {len(word_timestamps)} كلمة.")
                    return str(output_file), word_timestamps
            except Exception:
                continue

        print("❌ تعذر توليد التعليق الصوتي بجميع الأصوات المتاحة.")
        return None, []


def _run_async_tts(
    text: str, output_path: str, voice: str, rate: str, pitch: str, volume: str = "+15%"
) -> tuple[bool, list[dict[str, Any]]]:
    """Helper to run async TTS in a new event loop (for thread execution)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _generate_tts_async(text, output_path, voice, rate, pitch, volume)
        )
    finally:
        loop.close()



def _prepare_text_for_narration(text: str) -> str:
    """Clean and prepare Arabic text for natural TTS narration."""
    import re

    clean = text.strip()

    # Remove hashtags
    clean = re.sub(r"#\S+", "", clean)

    # Remove URLs
    clean = re.sub(r"https?://\S+", "", clean)

    # Remove emoji
    clean = re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
        r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
        r"\U0000FE00-\U0000FE0F\U0000200D]+",
        "",
        clean,
    )

    # Collapse multiple spaces and newlines
    clean = re.sub(r"\s+", " ", clean).strip()

    # Add slight pause markers for better narration flow
    clean = clean.replace(".", ".\n")
    clean = clean.replace("،", "،\n")

    return clean
