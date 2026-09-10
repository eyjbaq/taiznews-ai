"""Local dry-run test for the News Reels Engine.

Runs a complete single-article pipeline:
  1. Fetch news (Phase 1)
  2. Gemini editorial processing (Phase 2)
  3. Generate AI background image via Pollinations (NEW)
  4. Generate TTS narration via edge-tts (NEW)
  5. Compose reel video via MoviePy (NEW)

All outputs saved locally. NO Facebook publishing. NO git operations.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Force DRY_RUN mode for safety
os.environ["DRY_RUN"] = "true"

from app.collector.base import Article
from app.collector.gdelt import fetch_gdelt_articles
from app.collector.rss import load_settings, load_sources, fetch_rss_articles
from app.processing.deduplicate import (
    DEFAULT_PROCESSED_PATH,
    deduplicate_articles,
    load_processed_urls,
)
from app.processing.normalize import normalize_articles
from app.processing.relevance import filter_relevant
from app.processing.history import get_recent_published_summary
from app.ai.editorial import process_article
from app.ai.quota_manager import QUOTA_MANAGER
from app.design.ai_image import generate_ai_image, generate_ai_images
from app.design.tts_generator import generate_news_audio
from app.design.reel_maker import compose_news_reel
from app.design.renderer import NewsCardRenderer
from app.publisher.facebook import format_facebook_caption

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "sources.yaml"
LOGGER = logging.getLogger(__name__)


def score_article(article: Article) -> float:
    """Score an article locally based on breaking urgency, Taiz relevance, and freshness."""
    score = 0.0
    title_lower = article.title.lower()
    desc_lower = (article.description or "").lower()

    urgent_keywords = [
        "عاجل", "مصرع", "سيطرة", "اشتباكات", "مواجهات", "غارات", "طيران",
        "معارك", "هجوم", "كارثة", "نداء إنساني", "تسلل", "استهداف", "شهداء"
    ]
    for kw in urgent_keywords:
        if kw in title_lower:
            score += 35.0
        elif kw in desc_lower:
            score += 15.0

    if "تعز" in title_lower:
        score += 40.0
    elif "تعز" in desc_lower:
        score += 20.0

    now = datetime.now(timezone.utc)
    pub = article.published_at
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    age_hours = max(0.1, (now - pub).total_seconds() / 3600.0)
    score += max(0.0, 50.0 - (age_hours * 2.0))

    return score


def run_reels_test() -> None:
    """Run a local dry-run test of the full Reels pipeline for one article."""

    print("\n" + "=" * 70)
    print("🎬 === TaizNews AI | News Reels Engine — اختبار محلي (Dry Run) === 🎬")
    print("=" * 70)

    # ─── Phase 1: News Collection ───
    print("\n📡 المرحلة 1: جلب الأخبار...")
    settings = load_settings(CONFIG_PATH)
    timeout = int(settings.get("request_timeout_seconds", 20))
    max_entries = int(settings.get("max_entries_per_source", 30))
    minimum_reliability = float(settings.get("min_reliability", 80))

    source_reliability: Dict[str, float] = {
        str(source["name"]): float(source.get("reliability", 0))
        for source in load_sources(CONFIG_PATH)
    }
    source_reliability["GDELT"] = 80

    with ThreadPoolExecutor(max_workers=3) as executor:
        rss_future = executor.submit(fetch_rss_articles, CONFIG_PATH, timeout, max_entries)
        gdelt_future = executor.submit(fetch_gdelt_articles, '(تعز OR الحوبان OR حيفان OR مقبنة OR المخا OR Taiz)', 35, timeout)
        rss_articles = rss_future.result()
        gdelt_articles = gdelt_future.result()

    fetched = rss_articles + gdelt_articles
    normalized = normalize_articles(fetched)

    def article_reliability(article: Article) -> float:
        if article.source.startswith("GDELT"):
            return source_reliability["GDELT"]
        return source_reliability.get(article.source, 0)

    reliable = [a for a in normalized if article_reliability(a) >= minimum_reliability]
    processed_urls = load_processed_urls(DEFAULT_PROCESSED_PATH)
    unseen = [a for a in reliable if a.url not in processed_urls]
    relevant = filter_relevant(unseen)
    qualified = deduplicate_articles(relevant)
    qualified.sort(key=score_article, reverse=True)

    print(f"📊 إجمالي الأخبار المجلوبة: {len(fetched)}")
    print(f"📊 الأخبار الخاصة بتعز: {len(relevant)}")
    print(f"📊 الأخبار المؤهلة: {len(qualified)}")

    if not qualified:
        print("\n⚠️ لا توجد أخبار مؤهلة جديدة لاختبار الريلز.")
        return

    # ─── Phase 2: Gemini Editorial Processing (first qualified article only) ───
    print(f"\n📊 حالة النماذج: {QUOTA_MANAGER.get_status_summary()}")

    target_article = qualified[0]
    print(f"\n📰 الخبر المختار للاختبار:")
    print(f"   العنوان: {target_article.title}")
    print(f"   المصدر: {target_article.source}")
    print(f"   الرابط: {target_article.url}")
    print(f"   نقاط الأهمية: {score_article(target_article):.1f}")

    print(f"\n🤖 المرحلة 2: الصياغة التحريرية عبر Gemini...")
    recent_posts_context = get_recent_published_summary(limit=5)
    post = process_article(target_article, recent_posts_context=recent_posts_context)

    if not post:
        print("❌ تعذرت الصياغة التحريرية. إيقاف الاختبار.")
        return

    if not post.should_publish:
        print(f"❌ الخبر مرفوض تحريرياً: {post.rejection_reason}")
        print("🔄 جاري تجربة الخبر التالي...")

        for fallback in qualified[1:3]:
            print(f"\n📰 تجربة الخبر البديل: {fallback.title}")
            post = process_article(fallback, recent_posts_context=recent_posts_context)
            if post and post.should_publish:
                target_article = fallback
                break
        else:
            print("❌ لم يتم العثور على خبر مؤهل للنشر. إيقاف الاختبار.")
            return

    print("\n" + "-" * 55)
    print(f"✅ الخبر معتمد تحريرياً!")
    print(f"📰 العنوان: {post.headline}")
    print(f"📝 المتن: {post.body}")
    print(f"🏷️ الوسوم: {' '.join(post.hashtags)}")
    print(f"📸 وصف الصورة (EN): {post.image_prompt_en}")

    # ─── Phase 3a: Static News Card (existing) ───
    print(f"\n🖼️ المرحلة 3a: تصميم البطاقة الإخبارية الثابتة...")
    card_renderer = NewsCardRenderer()
    try:
        card_path = card_renderer.render_card(post, article=target_article)
        print(f"✅ بطاقة الخبر: {card_path}")
    except Exception as e:
        print(f"⚠️ تعذر تصميم البطاقة: {e}")
        card_path = None

    # ─── Phase 3b: Multi-Scene AI Images (Storyboarding) ───
    print(f"\n🎨 المرحلة 3b: توليد مشاهد القصة البصرية عبر Pollinations Flux...")
    prompts = post.image_prompts_en if hasattr(post, "image_prompts_en") and post.image_prompts_en else []
    if not prompts and post.image_prompt_en:
        prompts = [post.image_prompt_en]
    if not prompts:
        prompts = [
            f"Establishing wide shot of Yemeni mountain city Taiz {post.category}, ancient stone houses, dramatic clouds, cinematic lighting, 8k",
            f"Photojournalism of armed military technical pickup truck on rugged mountain road in Taiz Yemen, distant smoke, 35mm lens",
            f"Yemeni soldiers in uniform scanning the horizon from rocky ridge overlooking Taiz valley, dramatic natural lighting",
            f"Scenic golden hour view over Mount Sabir ridges in Taiz Yemen, cinematic documentary photojournalism",
        ]

    scene_images = generate_ai_images(
        prompts=prompts,
        slug_prefix=f"reel_scene_{target_article.id[:8]}",
    )

    if not scene_images:
        print("❌ تعذر توليد صور المشاهد. إيقاف مسار الريلز.")
        return

    # ─── Phase 3c: TTS Narration + Word Timestamps ───
    print(f"\n🎙️ المرحلة 3c: توليد التعليق الصوتي وتوقيت الكلمات...")
    first_para = post.body.split("\n\n")[0].strip()
    narration_text = f"{post.headline}. {first_para}"
    audio_path, word_timestamps = generate_news_audio(
        text=narration_text,
        slug=f"reel_tts_{target_article.id[:8]}",
    )

    if not audio_path:
        print("❌ تعذر توليد الصوت. إيقاف مسار الريلز.")
        return

    # ─── Phase 3d: Reel Video Composition ───
    print(f"\n🎬 المرحلة 3d: تركيب مقطع الريلز المتكامل (مشاهد + صوت + كاريوكي)...")
    reel_path = compose_news_reel(
        image_paths=scene_images,
        audio_path=audio_path,
        word_timestamps=word_timestamps,
        headline=post.headline,
        category=post.category,
        slug=f"reel_{target_article.id[:8]}",
    )

    # ─── Final Summary ───
    print("\n" + "=" * 70)
    print("📊 === ملخص اختبار محرك الريلز الإخباري ===")
    print("=" * 70)
    print(f"📰 الخبر: {post.headline}")
    print(f"📝 المتن: {post.body[:100]}...")
    print(f"📸 وصف الصورة: {post.image_prompt_en[:80]}...")

    if card_path:
        print(f"\n🖼️ بطاقة الخبر الثابتة:")
        print(f"   {card_path}")

    print(f"\n🎨 صورة خلفية AI:")
    print(f"   {ai_image_path}")

    print(f"\n🎙️ التعليق الصوتي:")
    print(f"   {audio_path}")

    if reel_path:
        print(f"\n🎬 مقطع الريلز النهائي:")
        print(f"   {reel_path}")
        print(f"\n🎉 ✅ نجح الاختبار بالكامل! يمكنك فتح ملف الفيديو ومشاهدته محلياً.")
    else:
        print(f"\n❌ فشل تركيب مقطع الريلز.")

    print(f"\n📊 استهلاك النماذج: {QUOTA_MANAGER.get_status_summary()}")
    print("=" * 70)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run_reels_test()


if __name__ == "__main__":
    main()
