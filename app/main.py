"""Command-line entry point for the Phase 1 Taiz news engine."""

from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

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
from app.ai.quota_manager import QUOTA_MANAGER
from app.design.renderer import NewsCardRenderer
from app.publisher.facebook import FacebookPublisher, format_facebook_caption

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "sources.yaml"
LOGGER = logging.getLogger(__name__)


def score_article(article: Article) -> float:
    """Score an article locally based on breaking urgency, Taiz relevance, and freshness."""
    score = 0.0
    title_lower = article.title.lower()
    desc_lower = (article.description or "").lower()

    # Breaking news keywords
    urgent_keywords = [
        "عاجل", "مصرع", "سيطرة", "اشتباكات", "مواجهات", "غارات", "طيران",
        "معارك", "هجوم", "كارثة", "نداء إنساني", "تسلل", "استهداف", "شهداء"
    ]
    for kw in urgent_keywords:
        if kw in title_lower:
            score += 35.0
        elif kw in desc_lower:
            score += 15.0

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


def run_pipeline() -> List[Article]:
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

    print("\n=== taiznews-ai | Phase 1: News Engine (Dry Run) ===")
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
        max_publish_count = int(os.getenv("MAX_POSTS_PER_RUN", "2"))
        recent_posts_context = get_recent_published_summary(limit=5)
        published_posts = []

        # Smart Token-Saving loop: Only evaluate candidates until target posts are published/scheduled
        for index, article in enumerate(qualified, start=1):
            if len(published_posts) >= max_publish_count:
                print(f"\n🎯 اكتمل نشر/جدولة المنشورات المطلوبة لهذه الجولة ({len(published_posts)}/{max_publish_count}). التوقف لترشيد التوكن والحصص.")
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

            # Render image card
            image_path = None
            try:
                image_path = card_renderer.render_card(post, article=article)
            except Exception as img_err:
                LOGGER.error("Failed rendering card for %s: %s", article.title, img_err)

            editorial_posts.append((article, post, image_path))

            print(f"\n📰 العنوان المصاغ:\n{post.headline}")
            print(f"\n📝 النص الجاهز للمنشور:\n{post.body}")
            print(f"\n🏷️ الوسوم:\n{' '.join(post.hashtags)}")
            if image_path:
                print(f"\n🖼️ بطاقة الخبر المصممة:\n{image_path}")

            # Phase 4: Facebook Publishing (First post immediate, second post scheduled 1 hour later)
            caption = format_facebook_caption(post, article)
            if len(published_posts) == 0:
                schedule_time = None
                print(f"\n[🚀 النشر المباشر الفوري على فيسبوك للخبر #{index}]...")
            else:
                schedule_time = int(time.time()) + 3600
                print(f"\n[⏰ جدولة النشر على فيسبوك لبعد ساعة لتفادي التكرار والسبام]...")

            pub_result = facebook_publisher.publish_photo(
                image_path,
                caption,
                scheduled_publish_time=schedule_time,
            )
            if pub_result:
                record_published_post(post, article, pub_result)
                save_processed_urls([article.url], DEFAULT_PROCESSED_PATH)
                published_posts.append((article, post, pub_result))
                recent_posts_context = get_recent_published_summary(limit=5)
                if schedule_time:
                    print(f"🎉 تم جدولة المنشور الثاني بنجاح لينشر تلقائياً بعد ساعة!")
                else:
                    print(f"🎉 تم نشر المنشور الأول فورياً بنجاح!")
            else:
                if facebook_publisher.last_error_code in (190, 102, 10):
                    print("\n" + "=" * 65)
                    print(f"🛑 إيقاف فوري للدورة: تعذر النشر بسبب خطأ تصريح/صلاحية فيسبوك (Error {facebook_publisher.last_error_code})!")
                    print(f"🛑 تفاصيل: الرمز (Access Token) منتهي الصلاحية أو غير صالح.")
                    print(f"🛑 تم إيقاف باقي المنشورات فوراً لحماية رصيد وحصص Gemini من الاستهلاك بلا جدوى.")
                    print("=" * 65)
                    break

            print("-" * 55)
            if len(published_posts) >= max_publish_count:
                print(f"🎯 اكتمل نشر/جدولة المنشورات المطلوبة ({len(published_posts)}/{max_publish_count}). إنهاء الدورة بنجاح.")
                break

        approved_count = len(editorial_posts)
        cards_count = sum(1 for _, _, img in editorial_posts if img)
        print(f"\n📊 الحصيلة التحريرية: {approved_count} منشور معتمد.")
        print(f"🎨 الحصيلة التصميمية: تم توليد {cards_count} بطاقة إخبارية احترافية.")
        print(f"🚀 الحصيلة الميدانية: تم نشر {len(published_posts)} منشور على فيسبوك بنجاح.")
        print(f"📊 استهلاك النماذج اليوم: {QUOTA_MANAGER.get_status_summary()}")
    else:
        print("\nلا توجد أخبار مؤهلة جديدة في هذه الجولة.")

    return qualified


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run_pipeline()


if __name__ == "__main__":
    main()
