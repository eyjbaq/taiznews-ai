"""Facebook Graph API publisher for Taiz News cards and articles."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv

from app.collector.base import Article
from app.ai.models import EditorialPost

load_dotenv()
LOGGER = logging.getLogger(__name__)

GRAPH_API_VERSION = "v19.0"


def format_facebook_caption(post: EditorialPost, article: Optional[Article] = None) -> str:
    """Format the headline, body, and hashtags for Facebook without links or external attribution."""
    lines = [
        f"🔴 {post.headline.strip()}",
        "",
        post.body.strip(),
        "",
        " ".join(h.strip() for h in post.hashtags if h.strip()),
    ]
    return "\n".join(lines).strip()


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
                err = result.get("error", {})
                err_msg = err.get("message", response.text)
                try:
                    err_code = int(err.get("code", 0))
                except (ValueError, TypeError):
                    err_code = err.get("code", "Unknown")
                err_subcode = err.get("error_subcode", "")
                self.last_error_code = err_code
                self.last_error_message = err_msg
                LOGGER.error("Facebook API error (%s): %s", err_code, err_msg)
                print(f"❌ فشل النشر على فيسبوك! (كود الخطأ: {err_code}, الفرعي: {err_subcode})")
                print(f"⚠️ تفاصيل الخطأ من فيسبوك: {err_msg}")
                return None

        except Exception as exc:
            LOGGER.error("Failed to publish to Facebook: %s", exc)
            print(f"❌ استثناء أثناء الاتصال بـ Facebook Graph API: {exc}")
            return None


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

