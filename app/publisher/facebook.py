"""Facebook Graph API publisher for Taiz News cards and articles."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv

from app.collector.base import Article
from app.ai.models import EditorialPost
from app.processing.safe_text import obfuscate_for_facebook_display

load_dotenv()
LOGGER = logging.getLogger(__name__)

GRAPH_API_VERSION = "v19.0"


def format_facebook_caption(post: EditorialPost, article: Optional[Article] = None) -> str:
    """Format the headline, body, interactive CTA, and hashtags for Facebook with algorithm defense."""
    raw_headline = post.headline.strip()
    safe_headline = obfuscate_for_facebook_display(raw_headline)

    body = post.body.strip()
    body = re.sub(r"نقلا[ً]? عن [^\s،.]+( [^\s،.]+)?", "", body)
    body = re.sub(r"بحسب [^\s،.]+( [^\s،.]+)?", "", body)
    body = re.sub(r"\s+", " ", body).strip()
    safe_body = obfuscate_for_facebook_display(body)

    tags: list[str] = []
    for h in post.hashtags:
        clean_h = h.strip()
        if not clean_h:
            continue
        if not clean_h.startswith("#"):
            clean_h = f"#{clean_h}"
        if clean_h not in tags:
            tags.append(clean_h)

    for core_tag in ["#تعز", "#تعز_نيوز", "#اليمن"]:
        if core_tag not in tags:
            tags.append(core_tag)

    interactive_prompt = "💬 شاركنا رأيك في التعليقات.. كيف ترى هذه التطورات؟"

    lines = [
        f"🔴 {safe_headline}",
        "",
        safe_body,
        "",
        interactive_prompt,
        "",
        " ".join(tags),
    ]
    return "\n".join(lines).strip()


def format_satirical_caption(post: EditorialPost) -> str:
    """Format a conversational Yemeni satirical text post with algorithm defense."""
    raw_text = (
        getattr(post, "satirical_text", "")
        or getattr(post, "clean_content", "")
        or getattr(post, "body", "")
        or ""
    ).strip()
    clean = re.sub(r"^[🔴🚨\s]*(?:عاجل\s*[:.\-–—]*\s*)?", "", raw_text)
    clean = re.sub(r"نقلا[ً]? عن [^\s،.]+( [^\s،.]+)?", "", clean)
    clean = re.sub(r"بحسب [^\s،.]+( [^\s،.]+)?", "", clean)
    clean = re.sub(r"وفقا[ً]? ل[^\s،.]+( [^\s،.]+)?", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    safe_clean = obfuscate_for_facebook_display(clean)

    tags: list[str] = []
    for h in getattr(post, "hashtags", []) or []:
        clean_h = re.sub(r"\s+", "_", str(h).strip())
        if not clean_h or clean_h == "#":
            continue
        if not clean_h.startswith("#"):
            clean_h = f"#{clean_h}"
        if clean_h not in tags:
            tags.append(clean_h)

    region_tag = "#تعز" if getattr(post, "scope", "taiz") == "taiz" else "#اليمن"
    if "#تعز" not in tags and "#اليمن" not in tags:
        tags.append(region_tag)
    if "#تعز_نيوز" not in tags:
        tags.append("#تعز_نيوز")

    if len(tags) > 4:
        essential = [t for t in tags if t in ("#تعز", "#اليمن", "#تعز_نيوز")]
        extras = [t for t in tags if t not in essential]
        tags = (extras[: max(0, 4 - len(essential))] + essential)[:4]

    return f"{safe_clean}\n\n{' '.join(tags)}".strip()


def format_facebook_reel_caption(post: EditorialPost, article: Optional[Article] = None) -> str:
    """Format an engaging Facebook Reel caption strictly under 400 characters with viral hashtags.

    Avoids repeating the entire spoken script, providing a punchy hook, teaser,
    and high-impact hashtags for optimal algorithmic reach.
    """
    headline = post.headline.strip()
    headline = re.sub(r"^(🔴\s*)+", "", headline).strip()
    safe_headline = obfuscate_for_facebook_display(headline)

    # 1. Extract a concise, intriguing teaser (1 sentence) rather than the whole spoken body
    body_text = (getattr(post, "clean_content", "") or getattr(post, "body", "")).strip()
    body_text = re.sub(r"نقلا[ً]? عن [^\s،.]+( [^\s،.]+)?", "", body_text)
    body_text = re.sub(r"بحسب [^\s،.]+( [^\s،.]+)?", "", body_text)
    body_text = re.sub(r"\s+", " ", body_text).strip()

    sentences = [s.strip() for s in body_text.replace("!", ".").replace("؟", ".").split(".") if len(s.strip()) > 10]
    teaser = sentences[0] if sentences else ""

    cta = "تابع التفاصيل الميدانية في الفيديو ⬇️"

    # 2. Build high-impact hashtags
    base_tags = ["#تعز", "#اليمن", "#أخبار_تعز", "#تعز_نيوز", "#Reels"]
    post_tags = []
    for h in post.hashtags:
        clean_h = h.strip()
        if not clean_h:
            continue
        if not clean_h.startswith("#"):
            clean_h = f"#{clean_h}"
        post_tags.append(clean_h)

    all_tags = []
    for tag in post_tags + base_tags:
        if tag not in all_tags:
            all_tags.append(tag)

    hashtags_str = " ".join(all_tags)

    if teaser and teaser.lower() in headline.lower():
        teaser = ""

    safe_teaser = obfuscate_for_facebook_display(teaser) if teaser else ""

    lines = [f"🔴 {safe_headline}"]
    if safe_teaser:
        lines.append(f"{safe_teaser}.. {cta}")
    else:
        lines.append(cta)
    lines.append("")
    lines.append(hashtags_str)

    caption = "\n".join(lines).strip()

    # Prune hashtags if total length exceeds 385 characters
    while len(caption) >= 385 and len(all_tags) > 3:
        all_tags.pop()
        hashtags_str = " ".join(all_tags)
        lines[-1] = hashtags_str
        caption = "\n".join(lines).strip()

    # If still >= 395 chars, cleanly trim teaser
    if len(caption) >= 395 and teaser:
        over = len(caption) - 375
        trimmed_teaser = teaser[:-over].rsplit(" ", 1)[0] + "..."
        lines[1] = f"{trimmed_teaser}.. {cta}"
        caption = "\n".join(lines).strip()

    return caption


def shorten_headline(headline: str, max_chars: int = 75) -> str:
    """Clean and shorten a news headline on a word boundary for high-impact Reels display."""
    text = (headline or "").strip()
    text = re.sub(r"^[🔴🚨📍⚡💥\s]+", "", text)
    text = re.sub(r"^(?:عاجل|بالصور|فيديو|شاهد)\s*[:.\-–—]*\s*", "", text)
    text = re.sub(r"نقلا[ً]? عن [^\s،.]+( [^\s،.]+)?", "", text)
    text = re.sub(r"بحسب [^\s،.]+( [^\s،.]+)?", "", text)
    text = re.sub(r"وفقا[ً]? ل[^\s،.]+( [^\s،.]+)?", "", text)
    text = re.sub(r"\s+", " ", text).strip(" .،:;-–—")

    if len(text) <= max_chars:
        return text

    truncated = text[: max_chars - 1].rsplit(" ", 1)[0].strip(" .،:;-–—")
    trailing_stops = {"في", "من", "على", "عن", "إلى", "و", "وسط", "بعد", "قبل", "مع", "بين"}
    words = truncated.split()
    while len(words) > 3 and words[-1] in trailing_stops:
        words.pop()
    clean_short = " ".join(words).strip(" .،:;-–—")
    return clean_short or text[:max_chars].strip()


def format_map_reel_caption(story: Any, article: Optional[Article] = None) -> Tuple[str, str]:
    """Build a shortened punchy title and viral-ready caption (<390 chars) for Map Reels.

    Returns:
        (short_title, full_caption)
    """
    short_title = shorten_headline(getattr(story, "headline", "") or "", max_chars=75)
    if not short_title and article:
        short_title = shorten_headline(article.title, max_chars=75)
    safe_short_title = obfuscate_for_facebook_display(short_title)

    # Extract concise curiosity-inducing teaser distinct from headline
    raw_teaser = (getattr(story, "teaser", "") or "").strip()
    if not raw_teaser:
        body_text = (getattr(story, "clean_content", "") or "").strip()
        sentences = [s.strip() for s in body_text.replace("!", ".").replace("؟", ".").split(".") if len(s.strip()) > 10]
        raw_teaser = sentences[0] if sentences else ""

    raw_teaser = re.sub(r"^[🔴🚨📍⚡💥\s]+", "", raw_teaser)
    raw_teaser = re.sub(r"نقلا[ً]? عن [^\s،.]+( [^\s،.]+)?", "", raw_teaser)
    raw_teaser = re.sub(r"بحسب [^\s،.]+( [^\s،.]+)?", "", raw_teaser)
    raw_teaser = re.sub(r"\s+", " ", raw_teaser).strip(" .،")
    if len(raw_teaser) > 125:
        raw_teaser = raw_teaser[:122].rsplit(" ", 1)[0].strip(" .،")

    if raw_teaser and (raw_teaser.lower() in short_title.lower() or short_title.lower() in raw_teaser.lower()):
        raw_teaser = ""

    safe_teaser = obfuscate_for_facebook_display(raw_teaser) if raw_teaser else ""

    cta = "🗺️ شاهد التفاصيل الميدانية على الخريطة التفاعلية ⬇️"

    # Build normalized district & topical hashtags
    district_raw = (getattr(story, "district_name", "") or "").strip()
    district_clean = re.sub(r"[^\w\u0600-\u06FF\s]+", "", district_raw).strip()
    district_tag = f"#{district_clean.replace(' ', '_')}" if district_clean else ""

    post_tags: list[str] = []
    if district_tag and district_tag != "#تعز":
        post_tags.append(district_tag)

    for h in getattr(story, "hashtags", []) or []:
        clean_h = re.sub(r"\s+", "_", str(h).strip())
        if not clean_h or clean_h == "#":
            continue
        if not clean_h.startswith("#"):
            clean_h = f"#{clean_h}"
        if clean_h not in post_tags:
            post_tags.append(clean_h)

    base_tags = ["#تعز", "#أخبار_تعز", "#تعز_نيوز", "#اليمن", "#Reels"]
    all_tags: list[str] = []
    for tag in post_tags + base_tags:
        if tag not in all_tags:
            all_tags.append(tag)
    all_tags = all_tags[:7]

    lines = [f"🔴 {safe_short_title}"]
    if safe_teaser:
        lines.append(f"{safe_teaser}.. {cta}")
    else:
        lines.append(cta)
    lines.append("")
    lines.append(" ".join(all_tags))

    caption = "\n".join(lines).strip()

    while len(caption) >= 385 and len(all_tags) > 3:
        all_tags.pop()
        lines[-1] = " ".join(all_tags)
        caption = "\n".join(lines).strip()

    if len(caption) >= 395 and raw_teaser:
        over = len(caption) - 375
        trimmed = raw_teaser[:-over].rsplit(" ", 1)[0].strip(" .،") if over < len(raw_teaser) else ""
        lines[1] = f"{trimmed}.. {cta}" if trimmed else cta
        caption = "\n".join(lines).strip()

    return short_title, caption


class FacebookPublisher:
    """Publishes news cards with captions to a Facebook Page via Graph API."""

    def __init__(
        self,
        page_id: Optional[str] = None,
        access_token: Optional[str] = None,
        dry_run: Optional[bool] = None,
    ) -> None:
        self.page_id = page_id or os.getenv("FACEBOOK_PAGE_ID", "").strip()
        self.access_token = access_token or os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN", "").strip()

        env_dry = os.getenv("DRY_RUN", "false").strip().lower()
        self.dry_run = dry_run if dry_run is not None else (env_dry in ("true", "1", "yes"))
        self.last_error_code: Optional[int] = None
        self.last_error_message: Optional[str] = None

    def _handle_api_error(self, result: Dict[str, Any], raw_text: str, action_label: str) -> None:
        """Record and log Facebook Graph API error details consistently."""
        err = result.get("error", {}) if isinstance(result, dict) else {}
        err_msg = err.get("message", raw_text)
        try:
            err_code: Any = int(err.get("code", 0))
        except (ValueError, TypeError):
            err_code = err.get("code", "Unknown")
        err_subcode = err.get("error_subcode", "")
        self.last_error_code = err_code
        self.last_error_message = err_msg
        LOGGER.error("Facebook %s error (%s): %s", action_label, err_code, err_msg)
        print(f"❌ فشل {action_label} على فيسبوك! (كود الخطأ: {err_code}, الفرعي: {err_subcode})")
        print(f"⚠️ تفاصيل الخطأ من فيسبوك: {err_msg}")

    def publish_photo(
        self,
        image_path: Path | str,
        caption: str,
        scheduled_publish_time: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Upload a photo with caption to the page's feed, either immediately or scheduled."""
        path = Path(image_path)
        if not path.exists():
            LOGGER.error("Image file does not exist: %s", path)
            print(f"❌ لم يتم العثور على ملف الصورة: {path}")
            return None

        if not self.page_id or not self.access_token:
            LOGGER.error("Missing FACEBOOK_PAGE_ID or FACEBOOK_PAGE_ACCESS_TOKEN.")
            print("❌ تعذر النشر: بيانات فيسبوك (FACEBOOK_PAGE_ID أو FACEBOOK_PAGE_ACCESS_TOKEN) غير مكتملة في ملف .env")
            return None

        if self.dry_run:
            sched_str = f" [مجدول لوقت: {scheduled_publish_time}]" if scheduled_publish_time else " [نشر فوري]"
            print(f"\n[وضع المحاكاة DRY_RUN=true]{sched_str} لن يتم إرسال طلب فعلي لفيسبوك.")
            print(f"📸 مسار الصورة: {path}")
            print(f"📝 النص البرمجي للنشر:\n{caption}")
            return {"id": "mock_photo_id", "post_id": f"{self.page_id}_mock_post_id", "url": f"https://www.facebook.com/{self.page_id}"}

        url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{self.page_id}/photos"
        data = {
            "caption": caption,
            "access_token": self.access_token,
        }

        if scheduled_publish_time:
            data["published"] = "false"
            data["scheduled_publish_time"] = str(int(scheduled_publish_time))
            print(f"\n⏰ جاري جدولة نشر البطاقة والخبر على فيسبوك بعد ساعة (Page ID: {self.page_id})...")
        else:
            data["published"] = "true"
            print(f"\n🚀 جاري رفع البطاقة ونشر الخبر فورياً على صفحة فيسبوك (Page ID: {self.page_id})...")

        try:
            with open(path, "rb") as img_file:
                files = {
                    "source": (path.name, img_file, "image/png"),
                }
                response = requests.post(url, data=data, files=files, timeout=45)

            result = response.json()

            if response.status_code == 200:
                photo_id = result.get("id")
                post_id = result.get("post_id") or photo_id
                post_url = f"https://www.facebook.com/{post_id}"

                print("=" * 60)
                if scheduled_publish_time:
                    print("🎉 تم جدولة نشر الخبر والبطاقة بنجاح على فيسبوك!")
                else:
                    print("🎉 تم نشر الخبر والبطاقة بنجاح فورياً على فيسبوك!")
                print(f"🆔 معرف الصورة (Photo ID): {photo_id}")
                print(f"📌 معرف المنشور (Post ID): {post_id}")
                print(f"🔗 رابط المنشور: {post_url}")
                print("=" * 60)

                result["url"] = post_url
                self.last_error_code = None
                self.last_error_message = None
                return result
            else:
                self._handle_api_error(result, response.text, "النشر")
                return None

        except Exception as exc:
            LOGGER.error("Failed to publish to Facebook: %s", exc)
            print(f"❌ استثناء أثناء الاتصال بـ Facebook Graph API: {exc}")
            return None

    def publish_reel(
        self,
        video_path: Path | str,
        caption: str,
        title: str = "",
        scheduled_publish_time: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Upload and publish (or schedule) a vertical video as a Reel on the Facebook Page via Graph API."""
        path = Path(video_path)
        if not path.exists():
            LOGGER.error("Video file does not exist: %s", path)
            print(f"❌ لم يتم العثور على ملف الفيديو: {path}")
            return None

        if not self.page_id or not self.access_token:
            LOGGER.error("Missing FACEBOOK_PAGE_ID or FACEBOOK_PAGE_ACCESS_TOKEN.")
            print("❌ تعذر النشر: بيانات فيسبوك (FACEBOOK_PAGE_ID أو FACEBOOK_PAGE_ACCESS_TOKEN) غير مكتملة في ملف .env")
            return None

        if self.dry_run:
            if scheduled_publish_time:
                sched_dt = datetime.fromtimestamp(int(scheduled_publish_time), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                sched_str = f" [مجدول للنشر عند: {sched_dt} (Unix: {int(scheduled_publish_time)})]"
            else:
                sched_str = " [نشر فوري]"
            print(f"\n[وضع المحاكاة DRY_RUN=true]{sched_str} لن يتم إرسال طلب نشر ريلز فعلي لفيسبوك.")
            print(f"📹 مسار ملف الريلز: {path}")
            if title:
                print(f"🏷️ عنوان الريلز: {title}")
            print(f"📝 النص المرافق للريلز:\n{caption}")
            res: Dict[str, Any] = {
                "id": "mock_reel_id",
                "post_id": f"{self.page_id}_mock_reel_post_id",
                "url": "https://www.facebook.com/reel/mock_reel_id",
            }
            if scheduled_publish_time:
                res["scheduled_publish_time"] = int(scheduled_publish_time)
            return res

        file_size = path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        if scheduled_publish_time:
            sched_dt = datetime.fromtimestamp(int(scheduled_publish_time), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            print(f"\n⏰ جاري رفع مقطع الريلز وجدولته للنشر عند {sched_dt} على صفحة فيسبوك (Page ID: {self.page_id}, الحجم: {file_size_mb:.1f} MB)...")
        else:
            print(f"\n🚀 جاري رفع مقطع الريلز ونشره على صفحة فيسبوك (Page ID: {self.page_id}, الحجم: {file_size_mb:.1f} MB)...")

        # Approach 1: Dedicated Facebook Reels API (/{page_id}/video_reels)
        try:
            init_url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{self.page_id}/video_reels"
            init_payload = {
                "upload_phase": "start",
                "access_token": self.access_token,
            }
            init_resp = requests.post(init_url, json=init_payload, timeout=30)
            init_data = init_resp.json()

            if init_resp.status_code == 200 and "upload_url" in init_data:
                video_id = init_data.get("video_id")
                upload_url = init_data.get("upload_url")
                print(f"📡 تم تهيئة جلسة الريلز (Video ID: {video_id}). جاري رفع الملف الثنائي...")

                # Step 2: Binary upload
                upload_headers = {
                    "Authorization": f"OAuth {self.access_token}",
                    "file_size": str(file_size),
                    "offset": "0",
                    "Content-Type": "application/octet-stream",
                }
                with open(path, "rb") as vf:
                    upload_resp = requests.post(upload_url, data=vf, headers=upload_headers, timeout=180)
                if upload_resp.status_code >= 400:
                    LOGGER.warning("video_reels binary upload returned HTTP %d: %s", upload_resp.status_code, upload_resp.text[:200])

                # Step 3: Finish and Publish / Schedule
                finish_payload: Dict[str, Any] = {
                    "upload_phase": "finish",
                    "video_id": video_id,
                    "description": caption,
                    "title": title or caption[:80],
                    "access_token": self.access_token,
                }
                if scheduled_publish_time:
                    finish_payload["video_state"] = "SCHEDULED"
                    finish_payload["scheduled_publish_time"] = int(scheduled_publish_time)
                else:
                    finish_payload["video_state"] = "PUBLISHED"

                finish_resp = requests.post(init_url, json=finish_payload, timeout=45)
                finish_data = finish_resp.json()

                if finish_resp.status_code == 200 and finish_data.get("success", False):
                    reel_url = f"https://www.facebook.com/reel/{video_id}"
                    print("=" * 60)
                    if scheduled_publish_time:
                        sched_dt = datetime.fromtimestamp(int(scheduled_publish_time), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                        print(f"⏰ تم رفع وجدولة مقطع الريلز بنجاح على فيسبوك للنشر عند: {sched_dt}")
                    else:
                        print("🎉 تم نشر مقطع الريلز (Reel) بنجاح على فيسبوك عبر Video Reels API!")
                    print(f"🆔 معرف الريلز: {video_id}")
                    print(f"🔗 رابط الريلز: {reel_url}")
                    print("=" * 60)
                    self.last_error_code = None
                    self.last_error_message = None
                    out: Dict[str, Any] = {"id": video_id, "post_id": video_id, "url": reel_url}
                    if scheduled_publish_time:
                        out["scheduled_publish_time"] = int(scheduled_publish_time)
                    return out
                else:
                    LOGGER.warning("video_reels finish returned: %s, falling back to /videos...", finish_data)
            else:
                LOGGER.warning("video_reels init returned: %s, falling back to /videos...", init_data)
        except Exception as reel_exc:
            LOGGER.warning("video_reels endpoint exception: %s, falling back to /videos...", reel_exc)

        # Approach 2 (Fallback): Standard /{page_id}/videos endpoint (FB auto-detects 9:16 vertical as Reels)
        try:
            print("🔄 جاري الرفع عبر نقطة النهاية الموثوقة (/{page_id}/videos)...")
            fallback_url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{self.page_id}/videos"
            with open(path, "rb") as vf:
                files = {"source": (path.name, vf, "video/mp4")}
                data: Dict[str, Any] = {
                    "description": caption,
                    "title": title or caption[:80],
                    "access_token": self.access_token,
                }
                if scheduled_publish_time:
                    data["published"] = "false"
                    data["scheduled_publish_time"] = str(int(scheduled_publish_time))
                resp = requests.post(fallback_url, data=data, files=files, timeout=180)

            result = resp.json()
            if resp.status_code == 200 and "id" in result:
                vid_id = result.get("id")
                post_url = f"https://www.facebook.com/watch/?v={vid_id}"
                print("=" * 60)
                if scheduled_publish_time:
                    sched_dt = datetime.fromtimestamp(int(scheduled_publish_time), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                    print(f"⏰ تم رفع وجدولة الفيديو/الريلز بنجاح على فيسبوك للنشر عند: {sched_dt}")
                else:
                    print("🎉 تم نشر مقطع الريلز بنجاح على صفحة فيسبوك!")
                print(f"🆔 معرف الفيديو/الريلز: {vid_id}")
                print(f"🔗 الرابط: {post_url}")
                print("=" * 60)
                self.last_error_code = None
                self.last_error_message = None
                result["url"] = post_url
                if scheduled_publish_time:
                    result["scheduled_publish_time"] = int(scheduled_publish_time)
                return result
            else:
                self._handle_api_error(result, resp.text, "نشر الريلز")
                return None
        except Exception as exc:
            LOGGER.error("Failed to publish video to Facebook: %s", exc)
            print(f"❌ استثناء أثناء نشر الفيديو على فيسبوك: {exc}")
            return None

    def publish_text(
        self,
        message: str,
        scheduled_publish_time: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Publish (or schedule) a plain text post directly to the Facebook Page feed."""
        clean_message = (message or "").strip()
        if not clean_message:
            LOGGER.error("Empty text message provided for Facebook feed publishing.")
            print("❌ تعذر النشر: نص المنشور فارغ.")
            return None

        if not self.page_id or not self.access_token:
            LOGGER.error("Missing FACEBOOK_PAGE_ID or FACEBOOK_PAGE_ACCESS_TOKEN.")
            print("❌ تعذر النشر: بيانات فيسبوك (FACEBOOK_PAGE_ID أو FACEBOOK_PAGE_ACCESS_TOKEN) غير مكتملة في ملف .env")
            return None

        if self.dry_run:
            sched_str = f" [مجدول لوقت: {scheduled_publish_time}]" if scheduled_publish_time else " [نشر فوري]"
            print(f"\n[وضع المحاكاة DRY_RUN=true]{sched_str} لن يتم إرسال منشور نصي فعلي لفيسبوك.")
            print(f"📝 نص المنشور:\n{clean_message}")
            res: Dict[str, Any] = {
                "id": f"{self.page_id}_mock_text_id",
                "post_id": f"{self.page_id}_mock_text_id",
                "url": f"https://www.facebook.com/{self.page_id}",
            }
            if scheduled_publish_time:
                res["scheduled_publish_time"] = int(scheduled_publish_time)
            return res

        url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{self.page_id}/feed"
        data: Dict[str, Any] = {
            "message": clean_message,
            "access_token": self.access_token,
        }

        if scheduled_publish_time:
            data["published"] = "false"
            data["scheduled_publish_time"] = str(int(scheduled_publish_time))
            print(f"\n⏰ جاري جدولة المنشور النصي على فيسبوك (Page ID: {self.page_id})...")
        else:
            data["published"] = "true"
            print(f"\n🚀 جاري نشر المنشور النصي فورياً على صفحة فيسبوك (Page ID: {self.page_id})...")

        try:
            response = requests.post(url, data=data, timeout=30)
            result = response.json()

            if response.status_code == 200 and result.get("id"):
                post_id = result.get("id")
                post_url = f"https://www.facebook.com/{post_id}"

                print("=" * 60)
                if scheduled_publish_time:
                    print("🎉 تم جدولة المنشور النصي بنجاح على فيسبوك!")
                else:
                    print("🎉 تم نشر المنشور النصي بنجاح فورياً على فيسبوك!")
                print(f"📌 معرف المنشور (Post ID): {post_id}")
                print(f"🔗 رابط المنشور: {post_url}")
                print("=" * 60)

                result["post_id"] = post_id
                result["url"] = post_url
                if scheduled_publish_time:
                    result["scheduled_publish_time"] = int(scheduled_publish_time)
                self.last_error_code = None
                self.last_error_message = None
                return result

            self._handle_api_error(result, response.text, "نشر النص")
            return None
        except Exception as exc:
            LOGGER.error("Failed to publish text post to Facebook: %s", exc)
            print(f"❌ استثناء أثناء نشر النص على فيسبوك: {exc}")
            return None

    def fetch_post_metrics(self, post_id: str) -> Optional[Dict[str, int]]:
        """Fetch engagement metrics from Facebook Graph API with Reel video-node fallback."""
        clean_id = (post_id or "").strip()
        if self.dry_run or not self.access_token or not clean_id:
            return None

        url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{clean_id}"
        primary_params = {
            "fields": "reactions.summary(true),comments.summary(true),shares",
            "access_token": self.access_token,
        }

        try:
            response = requests.get(url, params=primary_params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "error" not in data:
                    reactions = int(((data.get("reactions") or {}).get("summary") or {}).get("total_count", 0) or 0)
                    comments = int(((data.get("comments") or {}).get("summary") or {}).get("total_count", 0) or 0)
                    shares = int((data.get("shares") or {}).get("count", 0) or 0)
                    return {
                        "reactions": reactions,
                        "comments": comments,
                        "shares": shares,
                        "engagement_score": reactions + (comments * 3) + (shares * 5),
                    }
                LOGGER.debug("Facebook metrics error payload for %s: %s", clean_id, data)
                return None

            if response.status_code >= 400:
                LOGGER.debug(
                    "Primary metrics failed for %s (HTTP %d); trying Reel likes/comments fallback...",
                    clean_id,
                    response.status_code,
                )
                fallback_params = {
                    "fields": "likes.summary(true),comments.summary(true)",
                    "access_token": self.access_token,
                }
                fb_resp = requests.get(url, params=fallback_params, timeout=15)
                if fb_resp.status_code == 200:
                    fb_data = fb_resp.json()
                    if isinstance(fb_data, dict) and "error" not in fb_data:
                        likes = int(((fb_data.get("likes") or {}).get("summary") or {}).get("total_count", 0) or 0)
                        comments = int(((fb_data.get("comments") or {}).get("summary") or {}).get("total_count", 0) or 0)
                        return {
                            "reactions": likes,
                            "comments": comments,
                            "shares": 0,
                            "engagement_score": likes + (comments * 3),
                        }
                LOGGER.debug(
                    "Fallback metrics failed for %s (HTTP %d): %s",
                    clean_id,
                    fb_resp.status_code,
                    fb_resp.text[:200],
                )
                return None

            LOGGER.debug(
                "Could not fetch Facebook metrics for %s (HTTP %d): %s",
                clean_id,
                response.status_code,
                response.text[:200],
            )
            return None
        except Exception as exc:
            LOGGER.debug("Failed fetching Facebook metrics for %s: %s", clean_id, exc)
            return None


def publish_text_post(
    message: str,
    page_id: Optional[str] = None,
    access_token: Optional[str] = None,
    dry_run: Optional[bool] = None,
    scheduled_publish_time: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Helper function to publish or schedule a plain text post."""
    publisher = FacebookPublisher(page_id=page_id, access_token=access_token, dry_run=dry_run)
    return publisher.publish_text(message, scheduled_publish_time=scheduled_publish_time)


def publish_news_card(
    image_path: Path | str,
    caption: str,
    page_id: Optional[str] = None,
    access_token: Optional[str] = None,
    dry_run: Optional[bool] = None,
    scheduled_publish_time: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Helper function to publish a news card with caption."""
    publisher = FacebookPublisher(page_id=page_id, access_token=access_token, dry_run=dry_run)
    return publisher.publish_photo(image_path, caption, scheduled_publish_time=scheduled_publish_time)


def publish_news_reel(
    video_path: Path | str,
    caption: str,
    title: str = "",
    page_id: Optional[str] = None,
    access_token: Optional[str] = None,
    dry_run: Optional[bool] = None,
    scheduled_publish_time: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Helper function to publish or schedule a news reel video."""
    publisher = FacebookPublisher(page_id=page_id, access_token=access_token, dry_run=dry_run)
    return publisher.publish_reel(
        video_path,
        caption,
        title=title,
        scheduled_publish_time=scheduled_publish_time,
    )
