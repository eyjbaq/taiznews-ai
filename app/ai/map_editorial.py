"""Gemini AI Editorial Engine for Interactive Satellite & Tactical Map News Reels.

Adapts incoming Taiz news articles into verified district-bound stories with:
  - Strict Taiz district mapping (matching assets/maps/taiz_districts.json)
  - 65-85 word structured narration (>= 25s broadcast reel duration)
  - Dual text outputs: clean_content (subtitles) + narration_vocalized (TTS)
  - Tactical military flag (is_military) to trigger jet/explosion/armor overlays
  - Shortened punchy headline, engaging caption teaser, and targeted hashtags
  - Full integration with QUOTA_MANAGER & RATE_LIMITER
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.collector.base import Article
from .editorial import MAX_RETRIES_PER_MODEL, build_editorial_prompt, get_genai_client
from .models import EditorialPost
from .quota_manager import QUOTA_MANAGER, RATE_LIMITER

LOGGER = logging.getLogger(__name__)

SUPPORTED_TAIZ_DISTRICTS: List[str] = [
    "المخا",
    "مقبنة",
    "جبل صبر",
    "التعزية",
    "الحوبان",
    "ذباب",
    "موزع",
    "الوازعية",
    "جبل حبشي",
    "حيفان",
    "الصلو",
    "الشمايتين",
    "شرعب الرونة",
    "شرعب السلام",
    "المظفر",
    "القاهرة",
    "صالة",
    "ماوية",
    "سامع",
    "المسراخ",
    "دمنة خدير",
    "المعافر",
    "المواسط",
    "الضباب",
    "التربة",
    "مدينة تعز",
]


class MapNewsStory(BaseModel):
    """Structured editorial story tailored for interactive map video production."""

    is_relevant_to_taiz: bool = Field(
        description="هل الخبر يخص محافظة أو مدينة تعز أو مديرياتها بشكل مباشر"
    )
    should_publish: bool = Field(
        description="القرار التحريري النهائي (True إذا كان الخبر مؤكداً وموثوقاً ومهماً للنشر)"
    )
    rejection_reason: str = Field(
        default="",
        description="سبب استبعاد الخبر إذا كان should_publish = False",
    )
    importance_score: int = Field(
        default=80,
        description="درجة أهمية الخبر لأهالي تعز من 1 إلى 100",
    )
    district_name: str = Field(
        description="اسم المديرية بدقة في محافظة تعز حصراً من القائمة المعتمدة (مثل: مقبنة، المخا، جبل صبر، الحوبان، حيفان، الضباب، المظفر، الشمايتين، ذباب، ماوية، مدينة تعز)"
    )
    label: str = Field(
        description="عنوان تعريفي جغرافي للمديرية والقطاع (مثال: مديرية مقبنة | جبهة البرح، مديرية المخا | الساحل الغربي)"
    )
    location_chip_text: str = Field(
        description="وسم موقع موجز (مثال: مقبنة | غرب تعز، المخا | الساحل الغربي، الحوبان | شرق تعز)"
    )
    headline: str = Field(
        description="عنوان ريلز قصير وجذاب ومباشر من 6 إلى 10 كلمات كحد أقصى (بدون كلمة عاجل أو نقلاً عن) يشد انتباه القارئ على فيسبوك فوراً"
    )
    hook_text: str = Field(
        description="عنوان مختصر جداً يظهر على شاشة الخريطة في أول 3 ثوانٍ لا يتجاوز 6 كلمات (مثال: تصعيد ميداني في مقبنة)"
    )
    teaser: str = Field(
        description="جملة تشويقية مكثفة من 10 إلى 16 كلمة لوصف الريلز تثير فضول المشاهد دون تكرار العنوان حرفياً"
    )
    category: str = Field(
        description="تصنيف الحدث الإخباري: عاجل / أمن / خدمات / طقس / محليات / إنساني / اقتصاد"
    )
    is_military: bool = Field(
        description="True فقط إذا كان الخبر عسكرياً (اشتباكات، قصف، غارات، تعزيزات، انتشار أمني). False للأخبار المدنية أو الخدمية أو الطقس"
    )
    clean_content: str = Field(
        description="متن الخبر الصحفي بالكامل بدون أي تشكيل أو حركات (65 إلى 85 كلمة بالضبط) والدخول في الحدث مباشرة دون إسناد"
    )
    narration_vocalized: str = Field(
        description="نفس متن الخبر الصحفي تماماً وبنفس الكلمات والترتيب (65 إلى 85 كلمة) مشكولاً بالكامل بالتشكيل الصحيح للنطق الإذاعي الفصيح"
    )
    source_attribution: str = Field(
        default="",
        description="إسناد مختصر للمصدر الأصلي (مثال: المشهد اليمني، كريتر سكاي، وكالة سبأ)",
    )
    hashtags: List[str] = Field(
        default_factory=list,
        description="قائمة 4 إلى 6 وسوم نشطة تشمل #تعز ووسم المديرية (مثل #المخا أو #مقبنة) والموضوع",
    )

    def to_editorial_post(self) -> EditorialPost:
        """Convert to EditorialPost for history tracking and summary compatibility."""
        return EditorialPost(
            is_relevant_to_taiz=self.is_relevant_to_taiz,
            category=self.category,
            importance_score=self.importance_score,
            headline=self.headline,
            clean_content=self.clean_content,
            vocalized_content=self.narration_vocalized,
            body=self.clean_content,
            source_attribution=self.source_attribution,
            hashtags=self.hashtags,
            should_publish=self.should_publish,
            rejection_reason=self.rejection_reason,
            image_prompt_en="",
            image_prompts_en=[],
            video_keywords_en=[],
        )


MAP_EDITORIAL_SYSTEM_PROMPT = f"""أنت رئيس تحرير النشرات الجغرافية والميدانية المصورة لصفحة وقناة "تعز نيوز".
مهمتك تقييم الخبر الوارد وتحويله إلى قصة إخبارية متكاملة مخصصة للعرض على خريطة الأقمار الصناعية التفاعلية لمديريات محافظة تعز.

══════════════════════════════════════════════════════════════════════
القواعد الإلزامية الصارمة للنشرة الخرائطية (Map Reel Rules):
══════════════════════════════════════════════════════════════════════
1. التقييم التحريري ومنع التكرار:
   - تأكد أن الخبر يخص محافظة تعز أو يؤثر عليها مباشرة (is_relevant_to_taiz = True).
   - إذا كان الخبر روتينياً أو مكرراً لنفس الحدث في قائمة المنشورات السابقة دون تطور ميداني جديد، اجعل should_publish = False.
   - لا تختلق أي أرقام أو تفاصيل غير واردة في الخبر.

2. التحديد الجغرافي الصارم لمديرية في تعز (Geographic Precision):
   - يجب تعيين حقل district_name حصراً من إحدى هذه المديريات والمناطق المدعومة على الخريطة:
     {", ".join(SUPPORTED_TAIZ_DISTRICTS)}.
   - إذا ذكر الخبر منطقة فرعية (مثل البرح -> مقبنة، الساحل الغربي -> المخا، باب المندب -> ذباب، عقبة منيف/التعزية -> الحوبان، هيجة العبد -> الشمايتين أو التربة، جبل صبر -> جبل صبر، وسط المدينة -> مدينة تعز أو المظفر/القاهرة/صالة)، فاختر المديرية المطابقة بدقة.

3. تجهيز عنوان النشر والوصف الجذاب لفيسبوك ريلز (Social Packaging):
   - headline: عنوان قصير وحاد وجذاب من 6 إلى 10 كلمات فقط (بحد أقصى 75 حرفاً)، خالٍ من الحشو أو عبارة "عاجل:" أو الإسناد اللفظي، يلتقط جوهر الحدث ويثير الفضول المهني.
   - hook_text: عنوان مكثف جداً يظهر على الخريطة أول 3 ثوانٍ (3 إلى 6 كلمات فقط).
   - teaser: جملة تشويقية ذكية (10 إلى 16 كلمة) تكشف زاوية مثيرة أو أثراً ميدانياً وتدفع المشاهد لمشاهدة الخريطة التفاعلية دون تكرار كلمات العنوان.
   - hashtags: من 4 إلى 6 وسوم دقيقة تشمل #تعز ووسم المديرية (مثل #مقبنة أو #المخا أو #الحوبان) ووسم الموضوع.

4. شرط المدة الزمنية الحاسم (Reel Duration >= 25 seconds):
   - طول متن الخبر (clean_content و narration_vocalized) يجب أن يتراوح بدقة بين 65 و 85 كلمة!
   - ادخل في صلب الخبر مباشرة (يُمنع منعاً باتاً قول "نقلاً عن" أو "بحسب" أو ذكر أسماء المواقع داخل المتن أو العنوان).
   - قسّم المتن بانسيابية صحفية على 3 محاور: الحدث ونقطة وقوعه في المديرية -> التفاصيل الميدانية على الأرض -> الموقف الراهن أو الإجراءات المتخذة.

5. فصل النص المقروء عن النص الصوتي المشكول:
   - clean_content: بدون أي تشكيل أو حركات نهائياً (لشريط الترجمة على الشاشة).
   - narration_vocalized: نفس الكلمات والترتيب تماماً لكن مشكولاً بالكامل بالتشكيل الإعرابي الصحيح.

6. الطبيعة التكتيكية (is_military):
   - True إذا كان الخبر عسكرياً أو أمنياً ميدانياً (اشتباكات، قصف، غارات، تعزيزات، إحباط تسلل).
   - False إذا كان الخبر خدمياً، مدنياً، اقتصادياً، إنسانياً، أو طقساً.
"""


def process_article_for_map(
    article: Article,
    client: Optional[genai.Client] = None,
    model_name: Optional[str] = None,
    recent_posts_context: Optional[str] = None,
) -> Optional[MapNewsStory]:
    """Evaluate and transform an Article into a MapNewsStory using QuotaManager & RateLimiter."""
    try:
        active_client = client or get_genai_client()
        prompt = build_editorial_prompt(article, recent_posts_context=recent_posts_context)

        config = types.GenerateContentConfig(
            system_instruction=MAP_EDITORIAL_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=MapNewsStory,
            temperature=0.25,
        )

        if model_name:
            candidates = [model_name] + [m for m in QUOTA_MANAGER.get_available_models() if m != model_name]
        else:
            candidates = QUOTA_MANAGER.get_available_models()

        if not candidates:
            print("⚠️ جميع نماذج Gemini المعتمدة استنفدت حصتها أو محظورة للـ 24 ساعة الحالية.")
            return None

        for target_model in candidates:
            for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
                try:
                    RATE_LIMITER.wait_for_slot(target_model)
                    if attempt == 1:
                        print(f"🗺️🤖 [محرر الخريطة] إرسال الخبر إلى Gemini عبر النموذج [{target_model}]...")
                    else:
                        print(f"🔄 [محرر الخريطة] إعادة المحاولة بالنموذج [{target_model}] (محاولة {attempt}/{MAX_RETRIES_PER_MODEL})...")

                    response = active_client.models.generate_content(
                        model=target_model,
                        contents=prompt,
                        config=config,
                    )

                    story: Optional[MapNewsStory] = None
                    if hasattr(response, "parsed") and isinstance(response.parsed, MapNewsStory):
                        story = response.parsed
                    elif response.text:
                        story = MapNewsStory.model_validate_json(response.text)

                    if story:
                        used = QUOTA_MANAGER.record_success(target_model)
                        limit = QUOTA_MANAGER._load_data().get(target_model, {}).get("daily_limit", 20)
                        LOGGER.info("Map story processed with %s (Usage: %d/%d)", target_model, used, limit)
                        return story

                    LOGGER.warning("Gemini returned empty map response for '%s' on %s", article.title, target_model)
                    break

                except Exception as api_err:
                    err_str = str(api_err)
                    err_lower = err_str.lower()

                    is_quota_exhausted = (
                        ("429" in err_str and any(q in err_lower for q in [
                            "exceeded your current quota", "quota", "resource_exhausted", "billing details", "check your plan"
                        ]))
                        or "resource_exhausted" in err_lower
                        or "quota_exceeded" in err_lower
                    )
                    is_not_found = "404" in err_str or "not_found" in err_lower or "is not found" in err_lower
                    is_transient_error = (
                        "503" in err_str
                        or "unavailable" in err_lower
                        or "high demand" in err_lower
                        or "spikes in demand" in err_lower
                        or "try again later" in err_lower
                        or "500" in err_str
                        or "internal" in err_lower
                        or "timeout" in err_lower
                    )

                    if is_quota_exhausted:
                        QUOTA_MANAGER.mark_exhausted(target_model, reason="429 You exceeded your current quota")
                        print(f"⛔ تم بلوغ الحد الأقصى للنموذج [{target_model}] (429 Quota Exceeded). التبديل للنموذج التالي...")
                        break
                    elif is_not_found:
                        QUOTA_MANAGER.mark_exhausted(target_model, reason="404 Model Not Found")
                        print(f"⚠️ النموذج [{target_model}] غير متاح (404). التبديل للنموذج البديل...")
                        break
                    elif is_transient_error:
                        if attempt < MAX_RETRIES_PER_MODEL:
                            backoff = 6 * attempt
                            print(f"⏳ ضغط مؤقت على خوادم Google [{target_model}]. انتظار {backoff} ثوانٍ...")
                            time.sleep(backoff)
                            continue
                        break
                    else:
                        LOGGER.error("Map editorial API error on %s: %s", target_model, api_err)
                        if attempt < MAX_RETRIES_PER_MODEL:
                            time.sleep(5)
                            continue
                        break

        LOGGER.error("All candidate models failed for map article: '%s'", article.title)
        return None

    except Exception as exc:
        LOGGER.error("Map editorial processing failed for '%s': %s", article.title, exc)
        return None
