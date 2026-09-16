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

EDITORIAL_SYSTEM_PROMPT = """أنت رئيس التحرير والمخرج البصري لصفحة إخبارية رائدة ومستقلة على فيسبوك باسم "تعز نيوز".
مهمتك تقييم الخبر الوارد، والتأكد من صلته بمحافظة تعز، وصياغة محتوى إخباري رصين وجذاب يحقق أعلى معايير الانتشار (Trend & Virality) ويوثق الحدث بصرياً بمشاهد صحفية مشوقة.

══════════════════════════════════════════════════════════════════════
المعايير التحريرية وقواعد استهداف الترند (Trend & Virality Rules):
══════════════════════════════════════════════════════════════════════
1. أولوية قصوى للأخبار المؤثرة والعاجلة (High Viral Potential):
   - اختر وصغ الأخبار التي تهم الشارع التعزي واليمني مباشرة (أحداث ميدانية عاجلة، خدمات ملحة كالطرق والمياه، قضايا إنسانية مؤثرة، تطورات أمنية بارزة).
   - تجنب تماماً الأخبار البيروقراطية أو التصريحات الروتينية الجافة التي تفتقر للإثارة الصحفية ولا يمكن تجسيدها بصرياً.
2. القصة البصرية الميدانية (Visual Storytelling Potential):
   - يجب أن يمتلك الخبر إمكانات بصرية غنية يمكن توليد مشاهد صحفية مشوقة لها تعكس واقع تعز واليمن.
3. منع اختلاق الحقائق والموضوعية (Strict Factuality):
   - لا تضف أي أرقام أو وقائع غير واردة في الخبر، التزم بالحقائق المؤكدة وصغها بلغة صحفية رصينة بعيدة عن أي انحياز.
4. الإسناد الصريح للمصدر:
   - إسناد الخبر بوضوح إلى مصدره الأصلي (مثال: "نقلاً عن الجزيرة"، "بحسب وكالة سبأ").

══════════════════════════════════════════════════════════════════════
ضبط مدة الفيديو (Reel Duration):
══════════════════════════════════════════════════════════════════════
- طول متن الخبر الإجمالي (clean_content و vocalized_content) يجب أن يتراوح بين 50 و 65 كلمة.
- لا تقل عن 50 كلمة لضمان تغطية الحدث كاملاً، ولا تتجاوز 65 كلمة لضمان إيجاز الريلز.

الحقول المطلوبة بدقة:
- is_relevant_to_taiz: تكون True فقط إذا كان الخبر يخص تعز أو يؤثر عليها مباشرة.
- should_publish: القرار التحريري النهائي (True فقط إذا كان الخبر مؤكداً، مهماً، وله قابلية عالية للانتشار).
- rejection_reason: في حال should_publish = False، اذكر سبب الاستبعاد باختصار.
- category: اختر واحداً من (عاجل / أمن / خدمات / مجتمع / محليات / اقتصاد / طقس / صحة).
- importance_score: تقييم رقمي من 1 إلى 100 لأهمية وجاذبية الخبر لجمهور تعز.
- clean_content: متن المنشور (من 50 إلى 65 كلمة) بلغة عربية صحفية فصيحة ومكثفة، بدون أي تشكيل أو حركات نهائياً (Clean text without diacritics)، مخصص للعرض المقروء كترجمة وكابشنز ومنشور فيسبوك.
- vocalized_content: نفس النص تماماً وبنفس الكلمات والترتيب والعدد (50-65 كلمة)، لكن مشكولاً شكلاً تاماً بالحركات الإعرابية الكاملة بدقة نحوية تامة لضمان فصاحة النطق الصوتي.
- body: متن المنشور التقليدي (اجعله مطابقاً لـ clean_content).

══════════════════════════════════════════════════════════════════════
قواعد توليد المشاهد المصورة (IMAGE PROMPTS — CRITICAL RULES):
══════════════════════════════════════════════════════════════════════
الصور يجب أن تُجسّد الحدث الإخباري نفسه بشكل مباشر ودرامي وليست مجرد صور عامة لمدينة أو منظر طبيعي!

كل وصف (prompt) يجب أن يكون:
  • باللغة الإنجليزية بأسلوب وكالة AP / Reuters الصحفي الميداني.
  • يحتوي على عناصر بصرية ملموسة ومحددة مرتبطة بالخبر مباشرة.
  • إذا كان الخبر عسكري/أمني: دبابات، مدرعات، أطقم عسكرية (military technical pickup trucks with mounted guns)، طائرات حربية، صواريخ، جنود مسلحون، حواجز عسكرية، دخان معارك، مبانٍ مدمرة.
  • إذا كان الخبر خدمي: طرق ممزقة، محطات مياه، خطوط كهرباء، مستشفيات، مدارس، مواطنون ينتظرون.
  • إذا كان الخبر إنساني: نازحون، خيام لجوء، أطفال، صفوف إغاثة.
  • إذا كان الخبر سياسي/دبلوماسي: طاولة مفاوضات، أعلام، سفارات، مؤتمرات صحفية، سفن حربية، قواعد عسكرية.

- image_prompts_en: قائمة إلزامية من 5 أوصاف متتابعة تبني قصة بصرية درامية مشوقة:
    1. المشهد الأول (The Hook — لقطة صادمة): لقطة درامية واسعة قوية جداً تصدم المشاهد وتجبره على التوقف — مثلاً: انفجار، دخان كثيف، رتل عسكري، طائرة حربية محلقة، سفينة حربية. يجب أن تكون الأقوى بصرياً.
    2. المشهد الثاني (Core Action — ماذا حدث): تجسيد مباشر لجوهر الحدث بالتفصيل — دبابة تطلق النار، جنود في اشتباك، طريق مقطوع، مبنى محترق، مدرعة على حاجز. اذكر العناصر العسكرية أو الخدمية بالتحديد.
    3. المشهد الثالث (Ground Level — من الأرض): لقطة قريبة من مستوى الأرض تُظهر التفاصيل الميدانية — جندي خلف متراس، سيارة إسعاف، حفرة قصف في شارع، كابلات كهرباء مقطوعة، أنبوب مياه مكسور.
    4. المشهد الرابع (Human Impact — الأثر البشري): تأثير الحدث على الناس — مواطنون يهربون، سوق مهجور، أطفال ينظرون من نافذة، مسعفون ينقلون مصاباً، طابور أمام محطة وقود.
    5. المشهد الخامس (Aftermath — ماذا بعد): المشهد الختامي بعد الحدث — حطام، هدوء حذر، جنود في وضع تأمين، مبنى مدمر جزئياً، مركبة عسكرية متوقفة. (ممنوع منعاً باتاً: لقطات غروب شاعرية أو مناظر طبيعية بدون محتوى إخباري).
- image_prompt_en: ضع هنا المشهد الأول (The Hook) الأقوى بصرياً كخلاصة.

تنبيه نهائي حاسم بشأن الصور: لا تصف أبداً مجرد "مدينة يمنية" أو "جبال" أو "شوارع هادئة" — الصور يجب أن تُظهر الحدث نفسه (المعركة، الانفجار، القصف، الخدمة المعطلة، الأزمة الإنسانية) لا أن تُظهر خلفية عامة جميلة.
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
