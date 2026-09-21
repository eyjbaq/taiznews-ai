"""Command-line entry point for the Phase 1 Taiz news engine."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.collector.base import Article
from app.collector.gdelt import fetch_gdelt_articles
from app.collector.rss import load_settings, load_sources, fetch_rss_articles
from app.processing.deduplicate import (
    DEFAULT_PROCESSED_PATH,
    deduplicate_articles,
    load_processed_urls,
    save_processed_urls,
)
from app.processing.normalize import normalize_articles
from app.processing.relevance import filter_relevant
from app.processing.history import record_published_post, get_recent_published_summary
from app.ai.editorial import process_article
from app.ai.models import EditorialPost
from app.ai.quota_manager import QUOTA_MANAGER
from app.design.renderer import NewsCardRenderer
from app.design.ai_image import generate_ai_image, generate_ai_images
from app.design.tts_generator import generate_news_audio
from app.design.reel_maker import compose_news_reel
from app.publisher.facebook import FacebookPublisher, format_facebook_caption, format_facebook_reel_caption


CONFIG_PATH = Path(__file__).resolve().parent / "config" / "sources.yaml"
LOGGER = logging.getLogger(__name__)


def score_article(article: Article) -> float:
    """Score an article locally based on breaking urgency, Taiz relevance, and freshness."""
    score = 0.0
    title_lower = article.title.lower()
    desc_lower = (article.description or "").lower()

    # Breaking and viral news keywords
    urgent_keywords = [
        "عاجل", "مصرع", "سيطرة", "اشتباكات", "مواجهات", "غارات", "طيران",
        "معارك", "هجوم", "كارثة", "نداء إنساني", "تسلل", "استهداف", "شهداء",
        "انفجار", "حادث", "طريق", "سيول", "انهيار", "جريمة", "ضبط", "إحباط",
        "توتر", "احتجاجات", "فتح طريق", "أزمة"
    ]
    for kw in urgent_keywords:
        if kw in title_lower:
            score += 35.0
        elif kw in desc_lower:
            score += 15.0

    # High visual storytelling potential keywords (allows rich photojournalism)
    visual_keywords = [
        "غارات", "حريق", "سيول", "اشتباكات", "طريق", "شاحنات", "موكب",
        "انفجار", "مسيرة", "سد", "مدرعات", "حصار", "ميناء", "المخا", "الحوبان", "جبل صبر"
    ]
    for vkw in visual_keywords:
        if vkw in title_lower or vkw in desc_lower:
            score += 20.0
            break

    # Taiz prominence
    if "تعز" in title_lower:
        score += 40.0
    elif "تعز" in desc_lower:
        score += 20.0

    # Freshness / Recency weight (newer = higher score)
    now = datetime.now(timezone.utc)
    pub = article.published_at
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    age_hours = max(0.1, (now - pub).total_seconds() / 3600.0)
    score += max(0.0, 50.0 - (age_hours * 2.0))

    return score


def _format_date(article: Article) -> str:
    date = article.published_at
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    return date.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _create_reel_for_article(
    post: EditorialPost,
    article: Article,
) -> Optional[str]:
    """Create a high-impact news reel using AI-generated storyboard images and Ken Burns motion."""
    # 1. Prepare narration text: vocalized_content ensures 100% accurate Arabic diacritics and pronunciation
    narration_text = (
        getattr(post, "vocalized_content", "")
        or getattr(post, "clean_content", "")
        or getattr(post, "body", "")
    ).strip()

    if not narration_text:
        narration_text = f"{post.headline}. {post.body}"

    audio_path, word_timestamps = generate_news_audio(
        text=narration_text,
        slug=f"reel_tts_{(article.id if article else 'taiz')[:8]}",
        vocalized_fallback_text=getattr(post, "vocalized_content", ""),
    )
    if not audio_path:
        LOGGER.error("Failed to generate TTS audio for reel: %s", article.title)
        return None

    # 2. AI Image Storyboard Mode (Cloudflare Workers AI with Pollinations fallback)
    print(f"\n[🎨 نمط الصور: توليد مشاهد بصرية عبر Cloudflare Workers AI وبديل Pollinations]...")
    prompts = post.image_prompts_en if hasattr(post, "image_prompts_en") and post.image_prompts_en else []
    if not prompts and post.image_prompt_en:
        prompts = [post.image_prompt_en]
    if not prompts:
        prompts = [
            f"Dramatic wide shot of heavy smoke rising from military clashes near Yemeni city Taiz, armored vehicles on dusty road, AP photojournalism, 35mm lens, natural harsh daylight",
            f"Military technical pickup truck with mounted heavy machine gun racing on mountain road in Taiz Yemen, soldiers aboard, dust cloud, Reuters war photography",
            f"Ground level close-up of battle aftermath on a Yemeni street in Taiz, shell casings scattered, damaged concrete wall with bullet holes, gritty photojournalism",
            f"Yemeni civilians fleeing with belongings through a damaged narrow street in Taiz, ambulance in background, chaotic atmosphere, documentary photography",
            f"Aftermath scene of military checkpoint in Taiz Yemen, armored vehicle parked near damaged building, soldiers standing guard, cautious calm, no sunset",
        ]

    print(f"📸 أوصاف المشاهد المصورة المولدة ({len(prompts)} مشاهد):")
    for pi, p_txt in enumerate(prompts, start=1):
        print(f"   [{pi}] {p_txt[:95]}...")

    scene_images = generate_ai_images(
        prompts=prompts,
        slug_prefix=f"reel_scene_{(article.id if article else 'taiz')[:8]}",
    )
    if scene_images and audio_path:
        reel_path = compose_news_reel(
            image_paths=scene_images,
            audio_path=audio_path,
            word_timestamps=word_timestamps,
            headline=post.headline,
            category=post.category,
            source_attribution="",
            slug=f"reel_{(article.id if article else 'taiz')[:8]}",
        )
        if reel_path:
            print(f"\n🎉 🎬 مقطع الريلز السينمائي النهائي المصمم بنجاح:\n{reel_path}")
            return reel_path

    return None


def run_pipeline(
    dry_run: bool = False,
    max_posts: Optional[int] = None,
) -> List[Article]:
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

    reliable = [article for article in normalized if article_reliability(article) >= minimum_reliability]

    processed_urls = load_processed_urls(DEFAULT_PROCESSED_PATH)
    unseen = [article for article in reliable if article.url not in processed_urls]
    relevant = filter_relevant(unseen)
    qualified = deduplicate_articles(relevant)

    # Smart Ranking: Sort qualified articles so the most urgent and fresh Taiz news is first
    qualified.sort(key=score_article, reverse=True)

    print("\n=== taiznews-ai | News Reels Engine ===")
    if dry_run:
        print("⚠️ وضع الفحص المحلي (DRY RUN): مفعل — لن يتم النشر على فيسبوك وسيتم حفظ كل المخرجات محلياً.")
    print(f"إجمالي الأخبار المجلوبة: {len(fetched)}")

    if fetched:
        print("\n--- عينة من العناوين المجلوبة (أول 5 عناوين) ---")
        for index, article in enumerate(fetched[:5], start=1):
            print(f"[{index}] ({article.source}) {article.title}")

    print(f"\nبعد فلتر الموثوقية والروابط السابقة: {len(unseen)}")
    print(f"الأخبار الخاصة بتعز: {len(relevant)}")
    print(f"الأخبار المؤهلة بعد إزالة التكرار: {len(qualified)}")

    editorial_posts = []
    if qualified:
        print("\n--- الأخبار المرشحة مرتبة حسب الأهمية وحداثة الحدث ---")
        for index, article in enumerate(qualified[:5], start=1):
            print(f"{index}. [نقاط الأهمية: {score_article(article):.1f}] [{article.source}] {article.title}")
            print(f"   الرابط: {article.url}")

        print("\n" + "=" * 65)
        print("=== taiznews-ai | Phase 2, 3 & 4: Editorial, Design & Publishing ===")
        print("=" * 65)
        print(f"📊 حالة النماذج المتاحة: {QUOTA_MANAGER.get_status_summary()}")

        card_renderer = NewsCardRenderer()
        facebook_publisher = FacebookPublisher()
        # Target posts per run (defaults to 2: 1 photo card + 1 news reel)
        max_publish_count = max_posts if max_posts is not None else int(os.getenv("MAX_POSTS_PER_RUN", "2"))
        recent_posts_context = get_recent_published_summary(limit=5)
        published_posts = []
        editorial_posts = []

        # Smart Token-Saving loop: Only evaluate candidates until target posts are published
        for index, article in enumerate(qualified, start=1):
            if len(published_posts) >= max_publish_count:
                print(f"\n🎯 اكتمل نشر المنشورات المطلوبة لهذه الجولة ({len(published_posts)}/{max_publish_count}). التوقف لترشيد التوكن والحصص.")
                break

            print(f"\n[جاري تقييم الخبر المرشح #{index}/{len(qualified)} عبر Gemini]:")
            print(f"📰 {article.title}")

            post = process_article(article, recent_posts_context=recent_posts_context)
            if not post:
                print(f"⚠️ تعذر التحرير الذكي للخبر: {article.title}")
                continue

            status_str = "✅ معتمد للنشر" if post.should_publish else "❌ مرفوض من النشر"
            relevance_str = "نعم" if post.is_relevant_to_taiz else "لا"

            print("\n" + "-" * 55)
            print(f"📌 تقييم المنشور التحريري:")
            print(f"• قرار النشر التحريري: {status_str}")
            print(f"• صلة مباشرة بمحافظة تعز: {relevance_str}")
            print(f"• التصنيف: {post.category}")
            print(f"• درجة الأهمية: {post.importance_score}/100")
            if not post.should_publish and post.rejection_reason:
                print(f"• سبب الاستبعاد: {post.rejection_reason}")

            if not post.should_publish:
                # Mark as processed so we don't re-evaluate a rejected news item
                save_processed_urls([article.url], DEFAULT_PROCESSED_PATH)
                print(f"⏩ الانتقال للخبر المرشح التالي...")
                print("-" * 55)
                continue

            editorial_posts.append((article, post))

            print(f"\n📰 العنوان المصاغ:\n{post.headline}")
            display_content = post.clean_content or post.body
            print(f"\n📝 النص الجاهز للمنشور:\n{display_content}")
            print(f"\n🏷️ الوسوم:\n{' '.join(post.hashtags)}")

            caption = format_facebook_caption(post, article)

            # Check if this is Post #1 (Photo Card) or Post #2 (News Reel)
            if len(published_posts) == 0:
                # ─── المنشور الأول: نشر فوري لبطاقة الخبر والنص ───
                print(f"\n[🚀 المنشور الأول: نشر فوري لبطاقة الخبر والنص على فيسبوك]...")
                image_path = None
                try:
                    image_path = card_renderer.render_card(post, article=article)
                    if image_path:
                        print(f"🖼️ بطاقة الخبر المصممة:\n{image_path}")
                except Exception as img_err:
                    LOGGER.error("Failed rendering card for %s: %s", article.title, img_err)

                if not image_path:
                    print("⚠️ تعذر تصميم بطاقة الخبر، تخطي هذا المنشور...")
                    continue

                pub_result = None
                if dry_run:
                    print("🧪 [DRY RUN] تم تجاوز النشر على فيسبوك بنجاح.")
                    pub_result = {"id": f"dry_run_{int(time.time())}", "mode": "photo"}
                else:
                    pub_result = facebook_publisher.publish_photo(
                        image_path,
                        caption,
                        scheduled_publish_time=None,  # نشر فوري 100% بدون أي جدولة
                    )

                if pub_result:
                    record_published_post(post, article, pub_result)
                    save_processed_urls([article.url], DEFAULT_PROCESSED_PATH)
                    published_posts.append((article, post, pub_result, "photo"))
                    recent_posts_context = get_recent_published_summary(limit=5)
                    print(f"🎉 تم تسجيل المنشور الأول (البطاقة والنص) بنجاح!")
                else:
                    if facebook_publisher.last_error_code in (190, 102, 10):
                        print("\n" + "=" * 65)
                        print(f"🛑 إيقاف فوري للدورة: تعذر النشر بسبب خطأ تصريح/صلاحية فيسبوك (Error {facebook_publisher.last_error_code})!")
                        print("=" * 65)
                        break

            else:
                # ─── المنشور الثاني: نشر فوري لمقطع ريلز سينمائي ───
                print(f"\n[🎬 المنشور الثاني: إنشاء ونشر مقطع ريلز سينمائي فورياً]...")
                reel_path = None
                try:
                    reel_path = _create_reel_for_article(post, article)
                except Exception as reel_err:
                    LOGGER.error("Failed creating news reel: %s", reel_err)
                    print(f"⚠️ تعذر إنشاء مقطع الريلز: {reel_err}")

                pub_result = None
                if dry_run:
                    print("🧪 [DRY RUN] تم تجاوز النشر على فيسبوك بنجاح.")
                    pub_result = {"id": f"dry_run_{int(time.time())}", "mode": "reel"}
                elif reel_path:
                    reel_caption = format_facebook_reel_caption(post, article)
                    pub_result = facebook_publisher.publish_reel(
                        video_path=reel_path,
                        caption=reel_caption,
                        title=post.headline,
                    )
                else:
                    # Fallback to card photo if reel creation failed
                    print("🔄 استخدام بطاقة الخبر البديلة للنشر الفوري بدلاً من الريلز المتعثر...")
                    image_path = card_renderer.render_card(post, article=article)
                    if image_path:
                        pub_result = facebook_publisher.publish_photo(
                            image_path,
                            caption,
                            scheduled_publish_time=None,
                        )

                if pub_result:
                    record_published_post(post, article, pub_result)
                    save_processed_urls([article.url], DEFAULT_PROCESSED_PATH)
                    published_posts.append((article, post, pub_result, "reel" if reel_path else "photo"))
                    recent_posts_context = get_recent_published_summary(limit=5)
                    print(f"🎉 تم تسجيل المنشور الثاني بنجاح!")
                else:
                    if facebook_publisher.last_error_code in (190, 102, 10):
                        print("\n" + "=" * 65)
                        print(f"🛑 إيقاف فوري للدورة: تعذر النشر بسبب خطأ تصريح/صلاحية فيسبوك (Error {facebook_publisher.last_error_code})!")
                        print("=" * 65)
                        break

            print("-" * 55)
            if len(published_posts) >= max_publish_count:
                print(f"🎯 اكتمل نشر المنشورات المطلوبة ({len(published_posts)}/{max_publish_count}). إنهاء الدورة بنجاح.")
                break

        # If only 1 post was published and we have an approved story, also create its reel so we always produce 1 Photo + 1 Reel
        if len(published_posts) == 1 and len(editorial_posts) > 0:
            article, post = editorial_posts[0]
            print("\n" + "=" * 55)
            print(f"🎬 [توليد ريلز إضافي] لم يتوفر خبر ثانٍ مؤهل، جاري تحويل الخبر الأول إلى مقطع ريلز فورياً...")
            reel_caption = format_facebook_reel_caption(post, article)
            reel_path = None
            try:
                reel_path = _create_reel_for_article(post, article)
                pub_result = None
                if dry_run:
                    print("🧪 [DRY RUN] تم تجاوز النشر على فيسبوك بنجاح.")
                    pub_result = {"id": f"dry_run_{int(time.time())}", "mode": "reel"}
                elif reel_path:
                    pub_result = facebook_publisher.publish_reel(
                        video_path=reel_path,
                        caption=reel_caption,
                        title=post.headline,
                    )
                if pub_result:
                    published_posts.append((article, post, pub_result, "reel"))
                    print("🎉 تم تسجيل مقطع ريلز الخبر الأول بنجاح!")
            except Exception as e_reel:
                LOGGER.warning("Extra reel creation failed: %s", e_reel)

        approved_count = len(editorial_posts)
        cards_count = sum(1 for item in published_posts if len(item) > 3 and item[3] == "photo")
        reels_count = sum(1 for item in published_posts if len(item) > 3 and item[3] == "reel")
        print(f"\n📊 الحصيلة التحريرية: {approved_count} منشور معتمد.")
        print(f"🎨 الحصيلة الميدانية: {cards_count} بطاقة خبر + {reels_count} مقطع ريلز سينمائي.")
        print(f"📊 استهلاك النماذج اليوم: {QUOTA_MANAGER.get_status_summary()}")
    else:
        print("\nلا توجد أخبار مؤهلة جديدة في هذه الجولة.")

    return qualified


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dry-run execution and max-posts."""
    parser = argparse.ArgumentParser(
        description="TaizNews AI — Automated News Reels & Publishing Engine"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.getenv("DRY_RUN", "false").lower() in ("true", "1", "yes"),
        help="Local testing mode: skip publishing to Facebook and keep all generated assets locally",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=int(os.getenv("MAX_POSTS_PER_RUN", "2")),
        help="Maximum number of posts to publish/process per run (default: 2)",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    args = parse_args()
    run_pipeline(
        dry_run=args.dry_run,
        max_posts=args.max_posts,
    )


if __name__ == "__main__":
    main()


