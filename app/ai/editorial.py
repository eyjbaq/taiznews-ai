"""AI Editorial engine for evaluating, verifying, and drafting Taiz news."""

from __future__ import annotations

import logging
import os
import time
import warnings
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.collector.base import Article
from .models import EditorialPost
from .quota_manager import QUOTA_MANAGER, RATE_LIMITER

# Suppress harmless internal SDK function calling warnings
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()
LOGGER = logging.getLogger(__name__)

EDITORIAL_SYSTEM_PROMPT = """أنت محرر صحفي ذكي ومحترف لصفحة إخبارية مستقلة وموثوقة على فيسبوك باسم "تعز نيوز".
مهمتك تقييم الخبر الوارد، والتأكد من صلته بمحافظة تعز، وصياغة منشور إخباري رصين وجذاب وموضوعي.

القواعد التحريرية الصارمة والإلزامية:
1. منع اختلاق الحقائق منعاً باتاً (Never invent facts):
   - لا تضف أي أرقام، تواريخ، أسماء أشخاص، جهات، أو تفاصيل غير موجودة إطلاقاً في المدخلات الواردة إليك.
   - التزم بالحقائق المؤكدة في نص الخبر فقط دون أي اجتهاد أو هلوسة.
2. الحيادية والموضوعية الصحفية:
   - الصياغة بلغة عربية صحفية فصيحة، رصينة، ومحايدة تماماً، بعيداً عن أي تحيز سياسي أو طائفي أو أسلوب دعائي.
3. منع التضخيم وعناوين الإثارة (No Clickbait):
   - صياغة عنوان مباشر وواضح يعكس جوهر الحدث بمهنية وبدون علامات تعجب أو تهويل مبالغ فيه.
4. الإسناد والتحفظ:
   - إسناد الخبر بوضوح إلى مصدره الأصلي (مثال: "نقلاً عن الجزيرة"، "بحسب وكالة سبأ").
   - إذا كان الخبر غير مكتمل أو منقولاً عن جهة غير رسمية، استخدم صيغ التحفظ الصحفي ("أفادت تقارير"، "بحسب مصادر محلية").
5. القرار التحريري وتصنيف المنشور:
   - is_relevant_to_taiz: تكون True فقط إذا كان الخبر يخص تعز (المدينة، المديريات، الأرياف، الساحل الغربي) أو يؤثر عليها مباشرة.
   - should_publish: القرار التحريري النهائي (True فقط إذا كان الخبر ذا صلة بتعز، موثوقاً، ومفيداً للمتابعين).
   - rejection_reason: في حال should_publish = False، اذكر سبب الاستبعاد بدقة واختصار.
   - category: اختر واحداً من (أمن / مجتمع / خدمات / طقس / عاجل / محليات / صحة / اقتصاد).
   - importance_score: تقييم رقمي من 1 إلى 100 لأهمية الحدث لأهالي تعز.
   - body: متن المنشور من فقرة إلى فقرتين، جاهز ومناسب للقراءة على فيسبوك.
   - hashtags: قائمة وسوم مخصصة ذات صلة مثل #تعز #أخبار_تعز مع اسم المديرية إن ذكرت.
   - image_prompts_en: قائمة إلزامية من 4 إلى 5 أوصاف دقيقة ومتتابعة باللغة الإنجليزية (Sequential Storyboard Scenes) لتحويل الخبر إلى قصة بصرية سينمائية واقعية تحاكي مقطع ريلز حي في تعز أو اليمن:
     * يجب أن تنقل الأوصاف الخمسة المشاهد عبر تسلسل الحدث (Cinematic 5-Scene Storyboard):
       1. المشهد الأول (The Hook): لقطة افتتاحية عامة درامية للموقع في تعز (جبال شاهقة، وادٍ، قلعة القاهرة، أو قرية حجرية تحت سماء عاصفة).
       2. المشهد الثاني (The Core Action): تجسيد صريح لجوهر الحدث (آليات عسكرية أو أطقم أو مدرعات تسير على طريق ترابي، أو شاحنات إغاثة، أو سيول تجرف صخوراً).
       3. المشهد الثالث (The Human / Ground Perspective): لقطة مقربة أو زاوية منخفضة حركية (جنود بالعتاد الكامل يرصدون الأفق بين الصخور، أو مواطنون يمنيون، أو آثار غبار وركام).
       4. المشهد الرابع (The Environmental Context): لقطة للأجواء المحيطة والأثر الميداني (أعمدة دخان تتصاعد في الأفق البعيد، سحب داكنة، أو تضاريس وعرة تعكس صرامة الموقف).
       5. المشهد الخامس (Cinematic Resolution): لقطة سينمائية ختامية واسعة مع إضاءة الغروب (Golden hour) فوق قمم جبال تعز (جبل صبر) تعطي هيبة للخبر.
     * المعايير الفنية الصارمة:
       - أسلوب التصوير الصحفي الوثائقي العالمي (Associated Press / Reuters photojournalism, 35mm lens, f/2.8, photorealistic, gritty authentic texture, dramatic cinematic lighting, 8k).
       - الهوية اليمنية المباشرة (Taiz Yemen architecture, rugged Yemeni terrain, Mount Sabir).
       - الأمان: تجنب الدماء المسفوكة الصريحة أو الجثث، لكن الآليات والمدرعات والجنود والدخان والغبار والدمار مسموحة ومطلوبة جداً.
   - image_prompt_en: ضع هنا المشهد الأول أو الأبرز كخلاصة سريعة.
"""


def get_genai_client(api_key: Optional[str] = None) -> genai.Client:
    """Initialize and return the official Google GenAI client."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")
    return genai.Client(api_key=key)


def build_editorial_prompt(article: Article, recent_posts_context: Optional[str] = None) -> str:
    """Format an article's raw metadata and body into an editorial evaluation prompt with anti-duplicate context."""
    description_text = article.description.strip() if article.description else "لا يوجد وصف إضافي متوفر."
    prompt = f"""يرجى تقييم الخبر التالي وصياغته تحريرياً وفق التعليمات المحددة:

- المصدر: {article.source}
- العنوان الأصلي: {article.title}
- الرابط: {article.url}
- تاريخ النشر: {article.published_at.strftime('%Y-%m-%d %H:%M UTC')}

تفاصيل الخبر / الملخص:
{description_text}
"""
    if recent_posts_context and recent_posts_context != "لا توجد منشورات سابقة مسجلة حتى الآن.":
        prompt += f"""
قائمة المنشورات التي نُشرت مؤخراً على الصفحة (قاعدة منع التكرار الصارمة):
{recent_posts_context}

تنبيه حاسم: إذا كان الخبر الحالي أعلاه يتناول نفس الحدث أو نفس المعطيات التي نُشرت بالفعل في المنشورات السابقة دون أي تطور ميداني أو مستجد حقيقي، فيجب عليك رفض النشر وجعل should_publish = False واذكر في rejection_reason: "تكرار لحدث تم نشره مؤخراً على الصفحة".
"""
    return prompt


MAX_RETRIES_PER_MODEL = 3


def process_article(
    article: Article,
    client: Optional[genai.Client] = None,
    model_name: Optional[str] = None,
    recent_posts_context: Optional[str] = None,
) -> Optional[EditorialPost]:
    """Pass a qualified Article to Gemini using available models with fallback, retry, and rate limiting."""
    try:
        active_client = client or get_genai_client()
        prompt = build_editorial_prompt(article, recent_posts_context=recent_posts_context)

        config = types.GenerateContentConfig(
            system_instruction=EDITORIAL_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=EditorialPost,
            temperature=0.2,
        )

        # Build prioritized list of candidate models
        if model_name:
            candidates = [model_name] + [m for m in QUOTA_MANAGER.get_available_models() if m != model_name]
        else:
            candidates = QUOTA_MANAGER.get_available_models()

        if not candidates:
            print("⚠️ جميع نماذج Gemini المعتمدة استنفدت حصتها أو محظورة للـ 24 ساعة الحالية.")
            return None

        for target_model in candidates:
            success = False
            for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
                try:
                    # Enforce per-model rate limit (Standard: 3 RPM / Lite: 10 RPM)
                    RATE_LIMITER.wait_for_slot(target_model)

                    if attempt == 1:
                        print(f"🤖 إرسال الخبر إلى Gemini عبر النموذج [{target_model}]...")
                    else:
                        print(f"🔄 إعادة المحاولة بالنموذج [{target_model}] (محاولة {attempt}/{MAX_RETRIES_PER_MODEL})...")

                    response = active_client.models.generate_content(
                        model=target_model,
                        contents=prompt,
                        config=config,
                    )

                    post: Optional[EditorialPost] = None
                    if hasattr(response, "parsed") and isinstance(response.parsed, EditorialPost):
                        post = response.parsed
                    elif response.text:
                        post = EditorialPost.model_validate_json(response.text)

                    if post:
                        used = QUOTA_MANAGER.record_success(target_model)
                        limit = QUOTA_MANAGER._load_data().get(target_model, {}).get("daily_limit", 20)
                        LOGGER.info("Successfully processed with %s (Usage: %d/%d)", target_model, used, limit)
                        return post

                    LOGGER.warning("Gemini returned empty response for '%s' on %s", article.title, target_model)
                    break  # Move to next model if empty response

                except Exception as api_err:
                    err_str = str(api_err)
                    err_lower = err_str.lower()

                    # 1. Quota / Daily Limit Exceeded (429 with quota message)
                    is_quota_exhausted = (
                        ("429" in err_str and any(q in err_lower for q in [
                            "exceeded your current quota", "quota", "resource_exhausted", "billing details", "check your plan"
                        ]))
                        or "resource_exhausted" in err_lower
                        or "quota_exceeded" in err_lower
                    )

                    # 2. Model Not Found (404)
                    is_not_found = "404" in err_str or "not_found" in err_lower or "is not found" in err_lower

                    # 3. Temporary Server Demand Spikes (503 High Demand, 500, Spikes in demand)
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
                        print(f"⛔ تم بلوغ الحد الأقصى للنموذج [{target_model}] (429 Quota Exceeded).")
                        print(f"⏰ تم حفظ توقيت الاستنفاد، وسيتجدد النموذج تلقائياً بعد 24 ساعة.")
                        print(f"🔄 جاري الانتقال الفوري للنموذج التالي في القائمة...")
                        break  # Break retry loop, switch to next model

                    elif is_not_found:
                        QUOTA_MANAGER.mark_exhausted(target_model, reason="404 Model Not Found")
                        print(f"⚠️ النموذج [{target_model}] غير متاح (404 Not Found).")
                        print(f"🔄 جاري التبديل إلى النموذج البديل...")
                        break  # Break retry loop, switch to next model

                    elif is_transient_error:
                        if attempt < MAX_RETRIES_PER_MODEL:
                            backoff = 6 * attempt
                            print(
                                f"⏳ ضغط مؤقت على خوادم Google للنموذج [{target_model}] (503 High Demand / Spikes in demand)."
                            )
                            print(
                                f"🔄 هذا ليس استنفاداً للحصة. انتظار {backoff} ثوانٍ وإعادة المحاولة بنفس النموذج (محاولة {attempt}/{MAX_RETRIES_PER_MODEL})..."
                            )
                            time.sleep(backoff)
                            continue  # Retry same model!
                        else:
                            print(
                                f"⚠️ استمرار ضغط الخوادم على النموذج [{target_model}] بعد {MAX_RETRIES_PER_MODEL} محاولات. التبديل للنموذج التالي لهذا الخبر..."
                            )
                            break  # Switch to next model for this article

                    else:
                        LOGGER.error("API error with model %s: %s", target_model, api_err)
                        print(f"⚠️ خطأ أثناء الطلب بالنموذج [{target_model}]: {api_err}")
                        if attempt < MAX_RETRIES_PER_MODEL:
                            time.sleep(5)
                            continue
                        break

        LOGGER.error("All candidate models failed for article: '%s'", article.title)
        return None

    except Exception as exc:
        LOGGER.error("AI editorial processing failed for '%s': %s", article.title, exc)
        return None
