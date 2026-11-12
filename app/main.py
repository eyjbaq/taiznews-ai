"""Command-line entry point and CentralBrain orchestrator for taiznews-ai."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ai.editorial import evaluate_and_draft_batch
from app.ai.models import EditorialPost
from app.ai.quota_manager import QUOTA_MANAGER
from app.brain.central_brain import CentralBrain
from app.collector.base import Article
from app.collector.gdelt import fetch_gdelt_articles
from app.collector.rss import fetch_rss_articles, load_settings, load_sources
from app.collector.scraper import enrich_top_candidates
from app.design.ai_image import generate_ai_images
from app.design.map_reel import compose_interactive_news_reel
from app.design.reel_maker import compose_news_reel
from app.design.renderer import NewsCardRenderer
from app.design.tts_generator import generate_news_audio, get_last_used_voice_key
from app.processing.deduplicate import (
    DEFAULT_PROCESSED_PATH,
    cluster_articles,
    filter_against_published_history,
    load_processed_urls,
    save_processed_urls,
)
from app.processing.history import (
    get_recent_published_summary,
    load_published_history,
    record_published_post,
)
from app.processing.normalize import normalize_articles
from app.processing.relevance import (
    filter_fresh_articles,
    filter_relevant,
    filter_yemen_trend,
    score_article_smart,
)
from app.publisher.facebook import (
    FacebookPublisher,
    format_facebook_caption,
    format_facebook_reel_caption,
    format_map_reel_caption,
    format_satirical_caption,
)

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "sources.yaml"
LOGGER = logging.getLogger(__name__)
QUIET_HOUR_LOCAL_SCORE_THRESHOLD = 105.0
SECOND_POST_DELAY_SECONDS = 35 * 60
FATAL_FB_ERROR_CODES = (190, 102, 10)

# Backwards compatibility alias
score_article = score_article_smart


def _create_reel_for_article(
    post: EditorialPost,
    article: Article,
    preferred_voice: Optional[str] = None,
) -> Optional[str]:
    """Create a standard vertical news reel using TTS voice routing and AI storyboard scenes."""
    narration_text = (
        getattr(post, "vocalized_content", "")
        or getattr(post, "clean_content", "")
        or getattr(post, "body", "")
    ).strip()
    if not narration_text:
        narration_text = f"{post.headline}. {post.body}".strip()

    audio_path, word_timestamps = generate_news_audio(
        text=narration_text,
        slug=f"reel_tts_{(article.id if article else 'taiz')[:8]}",
        vocalized_fallback_text=getattr(post, "vocalized_content", ""),
        preferred_voice_key=preferred_voice,
    )
    if not audio_path:
        LOGGER.error("Failed to generate TTS audio for standard reel: %s", article.title)
        return None

    prompts = list(getattr(post, "image_prompts_en", []) or [])
    if not prompts and post.image_prompt_en:
        prompts = [post.image_prompt_en]
    if not prompts:
        prompts = [
            "Editorial breaking news street scene in Taiz Yemen, AP photojournalism, natural daylight",
            "Yemeni city overview looking toward Mount Sabir in Taiz, documentary wide shot",
            "Close-up of crowded market street in Taiz Yemen, realistic photojournalism",
            "Local traffic and urban life in Yemeni city district, atmospheric light",
        ]

    print(f"\n[🎨 توليد {len(prompts)} مشاهد بصرية للريلز]...")
    scene_images = generate_ai_images(
        prompts=prompts,
        slug_prefix=f"reel_scene_{(article.id if article else 'taiz')[:8]}",
    )
    if not scene_images:
        LOGGER.error("Failed to generate storyboard images for reel: %s", article.title)
        return None

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
        print(f"🎉 تم تصميم مقطع الريلز القياسي بنجاح: {reel_path}")
    return reel_path


def _print_brain_plan_summary(
    run_plan: Dict[str, Any],
    sync_stats: Dict[str, Any],
) -> None:
    """Print structured telemetry for the CentralBrain hourly strategy."""
    today = run_plan.get("today_counts", {})
    caps = run_plan.get("daily_caps", {})
    rem = run_plan.get("remaining_quotas", {})
    formats = run_plan.get("allowed_formats") or []

    if run_plan.get("is_quiet_hour"):
        window_state = "ساعات الفجر الميتة (عاجل فقط)"
    elif run_plan.get("is_peak_hour"):
        window_state = "ذروة التفاعل والمشاهدة"
    else:
        window_state = "ساعات النهار العادية"

    print("\n" + "=" * 68)
    print("🧠 === taiznews-ai | خطة العقل المركزي لهذه الساعة ===")
    print("=" * 68)
    print(f"🕒 التوقيت المحلي في اليمن: {run_plan.get('current_time_ye')} ({window_state})")
    print(
        f"📡 مزامنة تفاعلات فيسبوك: تم تحديث {sync_stats.get('synced_count', 0)} منشور "
        f"(سبب التخطي إن وُجد: {sync_stats.get('skipped_reason') or 'لا يوجد'})"
    )
    print(f"⚖️ القرار المبدئي للعقل المركزي: {run_plan.get('decision')}")
    print(
        f"📊 الحصص المتبقية اليوم: ريلز {rem.get('reels_remaining', 0)}/{caps.get('max_daily_reels', 4)} | "
        f"بطاقات {rem.get('photos_remaining', 0)}/{caps.get('max_daily_photos', 10)} | "
        f"ساخر {rem.get('text_satire_remaining', 0)}/{caps.get('max_daily_text_satire', 8)} | "
        f"ترند اليمن {rem.get('yemen_trends_remaining', 0)}/{caps.get('max_daily_yemen_trends', 2)}"
    )
    print(f"📈 إجمالي المنشور اليوم: {today.get('total_posts', 0)}")
    print(f"🎬 الصيغ المسموحة بالترتيب: {formats or 'لا توجد (تخطي)'}")
    print(f"🎙️ الصوت المقترح للريلز: {run_plan.get('preferred_voice', 'maryam')}")
    print("=" * 68)


def _dry_run_fallback_result(
    dry_run: bool,
    fmt: str,
    preview_label: str,
    caption: str,
    asset_path: Optional[str] = None,
    sched_ts: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Return a clean dry-run mock payload if FB credentials are absent locally."""
    if not dry_run:
        return None
    sched_info = f" [مجدول: {sched_ts}]" if sched_ts else " [فوري]"
    print(f"\n🧪 [DRY RUN - {preview_label}]{sched_info}")
    if asset_path:
        print(f"📎 الملف المنتج: {asset_path}")
    print(f"📝 النص/الوصف:\n{caption}")
    return {"id": f"dry_run_{fmt}_{int(time.time())}", "post_id": f"dry_run_{fmt}_{int(time.time())}", "mode": fmt}


def _publish_photo_with_fallback(
    post: EditorialPost,
    article: Article,
    card_renderer: NewsCardRenderer,
    facebook_publisher: FacebookPublisher,
    dry_run: bool,
    sched_ts: Optional[int],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Publish as hybrid photo card, falling back to plain text satire if rendering fails."""
    caption = format_facebook_caption(post, article)
    image_path: Optional[str] = None
    try:
        image_path = card_renderer.render_card(post, article=article)
        if image_path:
            print(f"🖼️ بطاقة الخبر المصممة: {image_path}")
    except Exception as exc:
        LOGGER.error("Card rendering failed for %s: %s", article.title, exc)

    if image_path:
        res = facebook_publisher.publish_photo(image_path, caption, scheduled_publish_time=sched_ts)
        if res:
            return res, "photo_card"
        if dry_run:
            return _dry_run_fallback_result(dry_run, "photo_card", "بطاقة إخبارية", caption, asset_path=image_path, sched_ts=sched_ts), "photo_card"
        return None, "photo_card"

    print("🔄 تعذر تصميم بطاقة الخبر، التحول التلقائي إلى منشور نصي...")
    sat_caption = format_satirical_caption(post)
    res = facebook_publisher.publish_text(sat_caption, scheduled_publish_time=sched_ts)
    if not res and dry_run:
        res = _dry_run_fallback_result(dry_run, "text_satire", "منشور نصي بديل", sat_caption, sched_ts=sched_ts)
    return res, "text_satire"


def _dispatch_selected_post(
    post: EditorialPost,
    article: Article,
    preferred_voice: Optional[str],
    card_renderer: NewsCardRenderer,
    facebook_publisher: FacebookPublisher,
    dry_run: bool,
    sched_ts: Optional[int],
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """Produce media (if needed) and publish according to post.chosen_format with safe fallbacks."""
    chosen = (post.chosen_format or "photo_card").strip()
    print(f"\n🚀 [التنفيذ والتوجيه] الصيغة المختارة: [{chosen}] | النطاق: [{post.scope}]")
    print(f"📰 العنوان: {post.headline}")

    if chosen == "text_satire":
        caption = format_satirical_caption(post)
        res = facebook_publisher.publish_text(caption, scheduled_publish_time=sched_ts)
        if not res and dry_run:
            res = _dry_run_fallback_result(dry_run, "text_satire", "منشور ساخر", caption, sched_ts=sched_ts)
        return res, "text_satire", ""

    if chosen == "photo_card":
        res, final_fmt = _publish_photo_with_fallback(
            post, article, card_renderer, facebook_publisher, dry_run, sched_ts
        )
        return res, final_fmt, ""

    if chosen == "standard_reel":
        reel_path: Optional[str] = None
        try:
            reel_path = _create_reel_for_article(post, article, preferred_voice=preferred_voice)
        except Exception as exc:
            LOGGER.error("Standard reel synthesis failed for '%s': %s", article.title, exc)
            print(f"⚠️ تعذر إنشاء الريلز القياسي: {exc}")

        if reel_path and Path(reel_path).exists():
            used_voice = get_last_used_voice_key()
            reel_caption = format_facebook_reel_caption(post, article)
            res = facebook_publisher.publish_reel(
                video_path=reel_path,
                caption=reel_caption,
                title=post.headline,
                scheduled_publish_time=sched_ts,
            )
            if not res and dry_run:
                res = _dry_run_fallback_result(
                    dry_run, "standard_reel", "ريلز قياسي", reel_caption, asset_path=str(reel_path), sched_ts=sched_ts
                )
            return res, "standard_reel", used_voice

        print("🔄 فشل إنشاء فيديو الريلز القياسي؛ التحول التلقائي لبطاقة مصورة أو نص...")
        res, final_fmt = _publish_photo_with_fallback(
            post, article, card_renderer, facebook_publisher, dry_run, sched_ts
        )
        return res, final_fmt, ""

    if chosen == "map_reel":
        map_reel_path: Optional[Path] = None
        try:
            map_reel_path = compose_interactive_news_reel(
                district_name=post.district_name or "مدينة تعز",
                label=post.map_label or post.district_name or "تعز",
                hook_text=post.hook_text or post.headline,
                location_chip_text=post.location_chip_text or post.district_name or "تعز",
                narration_vocalized=post.vocalized_content or post.clean_content,
                narration_clean=post.clean_content or post.body,
                slug=f"map_reel_{(article.id if article else 'taiz')[:8]}",
                map_style="satellite",
                mode="full_map",
                military=bool(post.is_military),
                preferred_voice_key=preferred_voice,
            )
        except Exception as exc:
            LOGGER.error("Map reel synthesis failed for '%s': %s", article.title, exc)
            print(f"⚠️ تعذر إنشاء ريلز الخريطة التفاعلية: {exc}")

        if map_reel_path and Path(map_reel_path).exists():
            used_voice = get_last_used_voice_key()
            short_title, map_caption = format_map_reel_caption(post, article)
            res = facebook_publisher.publish_reel(
                video_path=str(map_reel_path),
                caption=map_caption,
                title=short_title,
                scheduled_publish_time=sched_ts,
            )
            if not res and dry_run:
                res = _dry_run_fallback_result(
                    dry_run, "map_reel", "ريلز خريطة تفاعلية", map_caption, asset_path=str(map_reel_path), sched_ts=sched_ts
                )
            return res, "map_reel", used_voice

        print("🔄 تعذر إنشاء ريلز الخريطة التفاعلية؛ التحول التلقائي لبطاقة مصورة أو نص...")
        res, final_fmt = _publish_photo_with_fallback(
            post, article, card_renderer, facebook_publisher, dry_run, sched_ts
        )
        return res, final_fmt, ""

    # Defensive fallback for any unexpected format string
    res, final_fmt = _publish_photo_with_fallback(
        post, article, card_renderer, facebook_publisher, dry_run, sched_ts
    )
    return res, final_fmt, ""


def _resolve_max_selections(run_plan: Dict[str, Any], max_posts: Optional[int]) -> int:
    """Allow 2 posts only during peak hours when daily output is below half capacity, else 1."""
    today_counts = run_plan.get("today_counts", {})
    caps = run_plan.get("daily_caps", {})
    total_cap = sum(
        int(caps.get(k, 0))
        for k in ("max_daily_reels", "max_daily_photos", "max_daily_text_satire")
    )
    today_total = int(today_counts.get("total_posts", 0))

    if run_plan.get("is_peak_hour") and today_total < (total_cap / 2.0):
        target = 2
    else:
        target = 1

    if max_posts is not None:
        target = max(0, min(target, int(max_posts)))
    return target


def run_pipeline(
    dry_run: bool = False,
    max_posts: Optional[int] = None,
    enable_map_reel: bool = True,
    map_schedule_minutes: int = 60,
) -> List[Article]:
    """Execute the autonomous CentralBrain hourly pipeline."""
    # Step 1: Initialize FacebookPublisher + CentralBrain & Sync Self-Learning
    facebook_publisher = FacebookPublisher(dry_run=dry_run)
    brain = CentralBrain()
    sync_stats = brain.sync_and_learn_from_facebook(facebook_publisher, dry_run=dry_run)
    run_plan = brain.plan_hourly_run()

    if not enable_map_reel:
        run_plan["allowed_formats"] = [f for f in run_plan.get("allowed_formats", []) if f != "map_reel"]

    _print_brain_plan_summary(run_plan, sync_stats)

    if dry_run:
        print("⚠️ وضع الفحص المحلي (DRY RUN): مفعل — لن يتم النشر على فيسبوك ولن تُحفظ الروابط في السجل.")

    if run_plan.get("decision") == "SKIP":
        print("⏭️ قرار العقل المركزي لهذه الساعة: SKIP (اكتمال الحصص اليومية أو عدم وجود صيغ متاحة). إنهاء الجولة.")
        return []

    # Step 2: Concurrent Collection + Dual-Track Filtering (Taiz & Yemen Trend)
    settings = load_settings(CONFIG_PATH)
    timeout = int(settings.get("request_timeout_seconds", 20))
    max_entries = int(settings.get("max_entries_per_source", 30))
    minimum_reliability = float(settings.get("min_reliability", 80))

    source_reliability: Dict[str, float] = {
        str(source["name"]): float(source.get("reliability", 0))
        for source in load_sources(CONFIG_PATH)
    }
    source_reliability["GDELT"] = 80.0

    with ThreadPoolExecutor(max_workers=2) as executor:
        rss_future = executor.submit(fetch_rss_articles, CONFIG_PATH, timeout, max_entries)
        gdelt_future = executor.submit(fetch_gdelt_articles, "(تعز OR الحوبان OR حيفان OR مقبنة OR المخا OR Taiz)", 35, timeout)
        rss_articles = rss_future.result()
        gdelt_articles = gdelt_future.result()

    fetched = rss_articles + gdelt_articles
    normalized = normalize_articles(fetched)

    def _article_reliability(article: Article) -> float:
        if article.source.startswith("GDELT"):
            return source_reliability["GDELT"]
        return source_reliability.get(article.source, 0.0)

    reliable = [a for a in normalized if _article_reliability(a) >= minimum_reliability]
    processed_urls = load_processed_urls(DEFAULT_PROCESSED_PATH)
    unseen = [a for a in reliable if a.url not in processed_urls]
    fresh_unseen = filter_fresh_articles(unseen, max_age_hours=24.0)

    taiz_raw = filter_relevant(fresh_unseen)
    yemen_raw = filter_yemen_trend(fresh_unseen) if run_plan.get("allow_yemen_trend", False) else []

    history_entries = load_published_history()
    taiz_fresh = filter_against_published_history(taiz_raw, history_entries)
    yemen_fresh = filter_against_published_history(yemen_raw, history_entries) if yemen_raw else []

    taiz_qualified = cluster_articles(taiz_fresh)
    yemen_qualified = cluster_articles(yemen_fresh) if yemen_fresh else []

    taiz_qualified.sort(key=score_article_smart, reverse=True)
    yemen_qualified.sort(key=score_article_smart, reverse=True)

    print(
        f"\n📥 الجمع والفلترة: إجمالي={len(fetched)} | موثوق وغير معالج={len(unseen)} | "
        f"حديث (<24 ساعة)={len(fresh_unseen)} | "
        f"مسار تعز (بعد التجميع والتاريخ)={len(taiz_qualified)} | "
        f"ترند اليمن={len(yemen_qualified)}"
    )

    if taiz_qualified:
        print("\n--- أفضل مرشحي مسار تعز ---")
        for i, art in enumerate(taiz_qualified[:5], start=1):
            print(f"  {i}. [{score_article_smart(art):.1f}] (مصادر:{art.corroboration_count}) [{art.source}] {art.title}")
    if yemen_qualified:
        print("\n--- أفضل مرشحي ترند اليمن ---")
        for i, art in enumerate(yemen_qualified[:3], start=1):
            print(f"  {i}. [{score_article_smart(art):.1f}] (مصادر:{art.corroboration_count}) [{art.source}] {art.title}")

    # Step 3: Local Pre-Gemini Token Conservation Gatekeeper
    if not taiz_qualified and not yemen_qualified:
        print("\n🛑 لا توجد أخبار جديدة مؤهلة في تعز أو ترند اليمن بعد الفلترة المحلية — الخروج دون استهلاك Gemini.")
        return []

    if run_plan.get("is_quiet_hour") or run_plan.get("decision") == "URGENT_ONLY":
        top_taiz_score = score_article_smart(taiz_qualified[0]) if taiz_qualified else 0.0
        if top_taiz_score < QUIET_HOUR_LOCAL_SCORE_THRESHOLD:
            print(
                f"\n🌙 [بوابة ترشيد الفجر] أعلى خبر محلي سجل {top_taiz_score:.1f} نقطة "
                f"(أقل من الحد الطارئ {QUIET_HOUR_LOCAL_SCORE_THRESHOLD:.1f}). التوقف لتوفير التوكنات للنهار."
            )
            return []
        print(f"\n🚨 [اختراق طارئ لساعات الفجر] خبر تعزي فائق الأهمية ({top_taiz_score:.1f} نقطة) — متابعة المعالجة!")

    # Step 4: Enrich Top Candidates & Single-Call Batch Triage
    top_taiz = enrich_top_candidates(taiz_qualified[:5], max_candidates=5) if taiz_qualified else []
    top_yemen = (
        enrich_top_candidates(yemen_qualified[:2], max_candidates=2)
        if (yemen_qualified and run_plan.get("allow_yemen_trend", False))
        else []
    )
    batch_articles: List[Article] = list(top_taiz) + list(top_yemen)

    max_selections = _resolve_max_selections(run_plan, max_posts)
    recent_posts_context = get_recent_published_summary(limit=5)
    print(f"\n📊 حالة نماذج Gemini المتاحة: {QUOTA_MANAGER.get_status_summary()}")
    print(f"🎯 الحد الأقصى للاختيار في هذه الجولة: {max_selections}")

    batch_resp = evaluate_and_draft_batch(
        taiz_candidates=top_taiz,
        yemen_candidates=top_yemen,
        run_plan=run_plan,
        recent_posts_context=recent_posts_context,
        max_selections=max_selections,
    )

    if batch_resp is None:
        print("\n⚠️ تعذر إتمام استدعاء التحرير الدفعي عبر Gemini.")
        return taiz_qualified + yemen_qualified

    if batch_resp.decision != "PUBLISH" or not batch_resp.selected_posts:
        print(f"\n⏭️ [قرار التحرير الدفعي: SKIP] السبب: {batch_resp.skip_reason or 'لا يوجد خبر يستحق النشر'}")
        if not dry_run and batch_articles:
            save_processed_urls([a.url for a in batch_articles], DEFAULT_PROCESSED_PATH)
            print(f"💾 تم حفظ {len(batch_articles)} روابط مستبعدة في سجل المعالجة لمنع إعادتها في الساعة القادمة.")
        return taiz_qualified + yemen_qualified

    # Resolve selected (article, post) pairs preserving candidate index order
    selected_pairs: List[Tuple[Article, EditorialPost]] = []
    for post in batch_resp.selected_posts:
        idx = post.candidate_index - 1
        if 0 <= idx < len(batch_articles):
            selected_pairs.append((batch_articles[idx], post))

    # Step 5: Execute & Route by Format
    card_renderer = NewsCardRenderer()
    preferred_voice = run_plan.get("preferred_voice")
    published_urls: Set[str] = set()
    published_summary: List[Tuple[str, str, str]] = []
    fatal_fb_auth_error = False

    for seq_idx, (article, post) in enumerate(selected_pairs):
        sched_ts: Optional[int] = (
            int(time.time()) + SECOND_POST_DELAY_SECONDS
            if len(selected_pairs) >= 2 and seq_idx > 0
            else None
        )

        pub_result, actual_fmt, used_voice_key = _dispatch_selected_post(
            post=post,
            article=article,
            preferred_voice=preferred_voice,
            card_renderer=card_renderer,
            facebook_publisher=facebook_publisher,
            dry_run=dry_run,
            sched_ts=sched_ts,
        )

        if pub_result:
            published_urls.add(article.url)
            published_summary.append((actual_fmt, post.scope, used_voice_key))
            if not dry_run:
                record_published_post(
                    post,
                    article,
                    pub_result,
                    post_type=actual_fmt,
                    voice_id=used_voice_key,
                    scope=post.scope,
                )
                save_processed_urls([article.url], DEFAULT_PROCESSED_PATH)
            print(f"✅ تم اعتماد وتسجيل المنشور #{seq_idx + 1} بنجاح ({actual_fmt} | النطاق: {post.scope}).")
        elif facebook_publisher.last_error_code in FATAL_FB_ERROR_CODES:
            print("\n" + "=" * 65)
            print(f"🛑 إيقاف فوري للدورة: خطأ صلاحيات فيسبوك (Error {facebook_publisher.last_error_code})!")
            print("=" * 65)
            fatal_fb_auth_error = True
            break

    # Save evaluated-but-unpicked candidate URLs so next hour doesn't waste tokens on them
    if not dry_run and not fatal_fb_auth_error:
        remaining_batch_urls = [a.url for a in batch_articles if a.url not in published_urls]
        if remaining_batch_urls:
            save_processed_urls(remaining_batch_urls, DEFAULT_PROCESSED_PATH)

    print("\n" + "=" * 68)
    print(f"🏁 انتهاء جولة الساعة بنجاح — تم إنتاج/نشر {len(published_summary)} منشور:")
    for fmt, scope, voice in published_summary:
        voice_tag = f" | صوت: {voice}" if voice else ""
        print(f"   • الصيغة: {fmt} | النطاق: {scope}{voice_tag}")
    print(f"📊 استهلاك نماذج Gemini اليوم: {QUOTA_MANAGER.get_status_summary()}")
    print("=" * 68)

    return taiz_qualified + yemen_qualified


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dry-run execution and max-posts."""
    parser = argparse.ArgumentParser(
        description="TaizNews AI — Autonomous CentralBrain News Engine"
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
        default=None,
        help="Optional upper bound on posts published in this run",
    )
    parser.add_argument(
        "--skip-map-reel",
        action="store_true",
        default=os.getenv("ENABLE_MAP_REEL", "true").lower() in ("false", "0", "no"),
        help="Disable interactive map reels for this run",
    )
    parser.add_argument(
        "--map-schedule-minutes",
        type=int,
        default=int(os.getenv("MAP_REEL_SCHEDULE_MINUTES", "60")),
        help="Retained for CLI compatibility",
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
        enable_map_reel=not args.skip_map_reel,
        map_schedule_minutes=args.map_schedule_minutes,
    )


if __name__ == "__main__":
    main()
