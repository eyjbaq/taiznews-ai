"""AI Editorial engine for evaluating, verifying, and drafting Taiz news."""

from __future__ import annotations

import logging
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.collector.base import Article
from app.processing.relevance import score_article_smart
from .models import BatchEditorialResponse, EditorialPost
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
4. الدخول المباشر في صلب الخبر (Direct News Delivery — ممنوع الإسناد اللفظي):
   - ممنوع منعاً باتاً ذكر عبارات مثل "نقلاً عن..." أو "بحسب..." أو "وفقاً لـ..." أو ذكر أسماء المواقع أو المصادر داخل متن الخبر (clean_content و vocalized_content) أو العنوان نهائياً.
   - ادخل في الحدث والوقائع مباشرة بصياغة صحفية رصينة وقوية (مثال: "اندلعت اشتباكات عنيفة..." بدلاً من "أفادت مصادر نقلاً عن...").

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
- vocalized_content: نفس النص تماماً وبنفس الكلمات والترتيب والعدد (50-65 كلمة)، لكن مشكولاً بتشكيل ذكي مخفف للكلمات الصعبة والملتبسة فقط (مثل: الأفعال المبنية للمجهول، الكلمات التي قد تلتبس قراءتها، أسماء الأعلام والمناطق المحلية بتعز كـ "لِتَعِزّ" و"جِرْدَادْ" و"رَاسِنْ"، والأعداد المركبة)، مع ترك الكلمات الشائعة البسيطة بدون أي تشكيل لتوفير الحروف وتسريع النطق.
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
T = TypeVar("T")

BATCH_BRAIN_SYSTEM_PROMPT = """أنت رئيس التحرير التنفيذي وصاحب القلم الساخر لمنصة "تعز نيوز" (Taiz News).
مهمتك في هذه الجولة فحص قائمة الأخبار المرشحة دفعة واحدة، وتنفيذ خطة العقل المركزي بدقة صارمة: عند توفر أخبار مؤهلة، قم باختيار وصياغة منشورين متميزين من خبرين مرشحين مختلفين تماماً (منشور مرئي رئيسي + منشور يمني ساخر إلزامي)، أو إعلان التخطي (SKIP) فقط في حال انعدام الأخبار المؤهلة أو تكرارها الكامل.

══════════════════════════════════════════════════════════════════════
القواعد الصارمة لاتخاذ القرار والتحرير (Central Brain Rules):
══════════════════════════════════════════════════════════════════════

1) قرار النشر أو الصمت (PUBLISH vs SKIP):
- راجع الحد الأدنى للأهمية المطلوب في الخطة (min_importance_score).
- إذا كانت جميع الأخبار المرشحة دون المستوى، أو روتينية مملة، أو مكررة لما نُشر سابقاً في سجل النشر الحديث، فاتخذ فوراً قرار:
  decision = "SKIP" مع كتابة سبب واضح في skip_reason وترك selected_posts فارغة [].
- عندما تكون هناك أخبار مؤهلة و max_selections >= 2 مع توفر مرشحين فأكثر، يجب عليك إلزامياً اختيار وصياغة **منشورين مختلفين من خبرين مختلفين**:
  * المنشور الأول: منشور مرئي رئيسي (standard_reel أو map_reel أو photo_card).
  * المنشور الثاني: منشور يمني ساخر إلزامي (text_satire) يختار حدثاً خدمياً أو معيشياً أو سياسياً آخر.

2) قاعدة ترند اليمن (yemen_trend) وأولوية تعز:
- الأولوية الدائمة لأخبار تعز (scope == "taiz") إذا تقاربت الأهمية الصحفية.
- يُسمح باختيار خبر واحد فقط كحد أقصى من نطاق yemen_trend (وفقط إذا كانت allow_yemen_trend = True في الخطة)، وبشرط أن يكون حدثاً يمنياً كبيراً يهم عموم اليمنيين (اقتصاد، عملة، خدمات كبرى، قرار مصيري، حدث أمني/سياسي كبير).
- ممنوع منعاً باتاً اختيار صيغة "map_reel" لأي خبر نطاقه yemen_trend لأن الخريطة التفاعلية مخصصة حصراً لمديريات محافظة تعز.

3) اختيار الصيغة وتوزيع المنشورين (chosen_format حصراً من قائمة allowed_formats):
عند اختيار منشورين (max_selections >= 2)، وزع الصيغتين كالتالي:
- المنشور 1 (المرئي الرئيسي): اختر الصيغة الأنسب من بين ("standard_reel" أو "map_reel" أو "photo_card"):
  * "map_reel" (للأحداث الميدانية/الأمنية/الخدمية بتعز المحددة جغرافياً بمديرية معينة): املأ district_name و map_label و hook_text و location_chip_text و image_prompts_en.
  * "standard_reel" (للأخبار الدرامية ذات القصة البصرية القوية في تعز أو ترند اليمن): املأ headline و clean_content و vocalized_content و 5 مشاهد في image_prompts_en بالإنجليزية خالية من النصوص.
  * "photo_card" (بطاقة إخبارية مصورة عاجلة وسريعة): املأ headline و clean_content و image_prompt_en (وصف مشهد واقعي بالإنجليزية لتوليد خلفية بالذكاء الاصطناعي 100% دون نصوص داخل الصورة).
- المنشور 2 (المنشور اليمني الساخر الإلزامي): اختر "text_satire" لخبر مرشح آخر مختلف.

4) هندسة شخصية "الكاتب السياسي اليمني الساخر" (عند اختيار chosen_format == "text_satire"):
- اكتب المنشور في حقل satirical_text (وكرره مطابقاً في clean_content و body) بطول مكثف من 35 إلى 65 كلمة.
- الأسلوب: نقد سياسي واجتماعي يمني ساخر وذكي جداً (كوميديا سوداء راقية بروح الشارع التعزي واليمني)، يمزج العربية السلسة بلمسات أصيلة من اللهجة اليمنية/التعزية المفهومة للجميع (مثل: "يا جماعة الخير"، "ما شاء الله تبارك الله"، "المواطن المسكين"، "أبشروا بالخير"، "خبر يثلج الصدر في عز الحر"، "يعني بالعربي...") دون شتائم، بل سخرية لاذعة تضع اليد على الجرح وتجعل القارئ يتفاعل ويبتسم ويشارك المنشور فوراً.
- الخطاف التفاعلي الإلزامي: اختم المنشور الساخر دائماً بسؤال ذكي قصير ومستفز يفتح باب النقاش والتعليقات (مثل: "إيش رأيكم يا خبرة؟"، "تتوقعوا متى بيوصلنا الدور؟"، "شاركونا آرائكم في التعليقات!") مع 3 وسوم فقط في hashtags.

5) قواعد النزاهة الصحفية وحقول الإخراج المطلوبة:
- لا تختلق أي أرقام أو أسماء غير واردة في نص المرشح.
- لكل عنصر في selected_posts تأكد من تعبئة الحقول بدقة: candidate_index (1-based)، scope (`taiz` أو `yemen_trend` مطابقاً للمرشح)، is_relevant_to_taiz (`true` لأخبار تعز)، importance_score (من **1 إلى 100**)، should_publish (`true`)، source_attribution، category، headline، clean_content، و hashtags."""


def _generate_structured_with_fallback(
    active_client: genai.Client,
    prompt: str,
    config: types.GenerateContentConfig,
    response_model: Type[T],
    candidates: List[str],
    context_label: str,
) -> Optional[T]:
    """Send a structured-JSON request to Gemini with quota tracking, retries, and model fallback."""
    for target_model in candidates:
        for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
            try:
                RATE_LIMITER.wait_for_slot(target_model)

                if attempt == 1:
                    print(f"🤖 إرسال {context_label} إلى Gemini عبر النموذج [{target_model}]...")
                else:
                    print(
                        f"🔄 إعادة المحاولة بالنموذج [{target_model}] "
                        f"(محاولة {attempt}/{MAX_RETRIES_PER_MODEL})..."
                    )

                response = active_client.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=config,
                )

                parsed_result: Optional[T] = None
                if hasattr(response, "parsed") and isinstance(response.parsed, response_model):
                    parsed_result = response.parsed
                elif response.text:
                    parsed_result = response_model.model_validate_json(response.text)

                if parsed_result is not None:
                    used = QUOTA_MANAGER.record_success(target_model)
                    limit = QUOTA_MANAGER._load_data().get(target_model, {}).get("daily_limit", 20)
                    LOGGER.info("Successfully processed %s with %s (Usage: %d/%d)", context_label, target_model, used, limit)
                    return parsed_result

                LOGGER.warning("Gemini returned empty response for %s on %s", context_label, target_model)
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
                    print(f"⛔ تم بلوغ الحد الأقصى للنموذج [{target_model}] (429 Quota Exceeded).")
                    print("⏰ تم حفظ توقيت الاستنفاد، وسيتجدد النموذج تلقائياً بعد 24 ساعة.")
                    print("🔄 جاري الانتقال الفوري للنموذج التالي في القائمة...")
                    break

                if is_not_found:
                    QUOTA_MANAGER.mark_exhausted(target_model, reason="404 Model Not Found")
                    print(f"⚠️ النموذج [{target_model}] غير متاح (404 Not Found).")
                    print("🔄 جاري التبديل إلى النموذج البديل...")
                    break

                if is_transient_error:
                    if attempt < MAX_RETRIES_PER_MODEL:
                        backoff = 6 * attempt
                        print(
                            f"⏳ ضغط مؤقت على خوادم Google للنموذج [{target_model}] (503 High Demand / Spikes in demand)."
                        )
                        print(
                            f"🔄 هذا ليس استنفاداً للحصة. انتظار {backoff} ثوانٍ وإعادة المحاولة "
                            f"(محاولة {attempt}/{MAX_RETRIES_PER_MODEL})..."
                        )
                        time.sleep(backoff)
                        continue
                    print(
                        f"⚠️ استمرار ضغط الخوادم على النموذج [{target_model}] بعد {MAX_RETRIES_PER_MODEL} محاولات. "
                        "التبديل للنموذج التالي..."
                    )
                    break

                LOGGER.error("API error with model %s for %s: %s", target_model, context_label, api_err)
                print(f"⚠️ خطأ أثناء الطلب بالنموذج [{target_model}]: {api_err}")
                if attempt < MAX_RETRIES_PER_MODEL:
                    time.sleep(5)
                    continue
                break

    return None


def _resolve_candidate_models(model_name: Optional[str]) -> List[str]:
    """Return ordered list of available Gemini models."""
    available = QUOTA_MANAGER.get_available_models()
    if model_name:
        return [model_name] + [m for m in available if m != model_name]
    return available


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

        candidates = _resolve_candidate_models(model_name)
        if not candidates:
            print("⚠️ جميع نماذج Gemini المعتمدة استنفدت حصتها أو محظورة للـ 24 ساعة الحالية.")
            return None

        result = _generate_structured_with_fallback(
            active_client=active_client,
            prompt=prompt,
            config=config,
            response_model=EditorialPost,
            candidates=candidates,
            context_label="الخبر",
        )
        if result is None:
            LOGGER.error("All candidate models failed for article: '%s'", article.title)
        return result

    except Exception as exc:
        LOGGER.error("AI editorial processing failed for '%s': %s", article.title, exc)
        return None


def _build_batch_prompt(
    indexed_candidates: List[Tuple[int, str, Article]],
    run_plan: Dict[str, Any],
    supported_districts: List[str],
    recent_posts_context: Optional[str],
    max_selections: int,
) -> str:
    """Construct the single-call batch triage & drafting prompt."""
    allowed_formats = run_plan.get("allowed_formats") or ["photo_card"]
    min_importance = run_plan.get("min_importance_score", 6)
    allow_yemen_trend = bool(run_plan.get("allow_yemen_trend", False))
    preferred_voice = run_plan.get("preferred_voice", "")
    directives = run_plan.get("brain_directives_prompt", "")

    candidate_blocks: List[str] = []
    for idx, scope_label, art in indexed_candidates:
        src_name = getattr(art, "source", "") or "غير معروف"
        corr_sources = getattr(art, "corroborating_sources", None) or [src_name]
        corr_count = getattr(art, "corroboration_count", 1)
        has_img = "نعم" if getattr(art, "image_url", None) else "لا"
        local_score = round(score_article_smart(art), 1)
        context_text = art.rich_context_text() if hasattr(art, "rich_context_text") else (art.description or "")
        candidate_blocks.append(
            f"[مرشح #{idx}]\n"
            f"- candidate_index: {idx}\n"
            f"- النطاق (scope): {scope_label}\n"
            f"- المصدر الرئيسي: {src_name} | عدد المصادر المؤكدة (corroboration_count): {corr_count} ({', '.join(corr_sources[:4])})\n"
            f"- النقاط المحلية (local_score): {local_score} | توجد صورة مرفقة: {has_img}\n"
            f"- العنوان: {art.title}\n"
            f"- النص الغني:\n{context_text}"
        )

    history_block = (recent_posts_context or "").strip() or "لا توجد منشورات حديثة مسجلة."
    districts_line = "، ".join(supported_districts)

    selections_rule = (
        f"- العدد المطلوب اختياره (max_selections): {max_selections} (إلزامي عند توفر أكثر من مرشح: منشور 1 مرئي [photo_card أو reel] + منشور 2 ساخر [text_satire] من خبرين مرشحين مختلفين تماماً)."
        if max_selections >= 2 and len(indexed_candidates) >= 2
        else f"- الحد الأقصى لعدد المنشورات المختارة (max_selections): {max_selections}"
    )

    return (
        "═══ خطة العقل المركزي لهذه الجولة ═══\n"
        f"- الصيغ المسموحة حصراً (allowed_formats): {allowed_formats}\n"
        f"- الحد الأدنى للأهمية المطلوب للنشر (min_importance_score): {min_importance}/100\n"
        f"{selections_rule}\n"
        f"- السماح بخبر ترند اليمن (allow_yemen_trend): {'نعم (خبر واحد كحد أقصى)' if allow_yemen_trend else 'لا'}\n"
        f"- الصوت/الزاوية المفضلة (preferred_voice): {preferred_voice or 'غير محدد'}\n"
        f"- التوجيهات الاستراتيجية:\n{directives or 'لا توجد توجيهات إضافية.'}\n\n"
        "═══ مديريات تعز المعتمدة للخريطة (SUPPORTED_TAIZ_DISTRICTS عند اختيار map_reel) ═══\n"
        f"{districts_line}\n\n"
        "═══ سجل المنشورات السابقة (ممنوع تكرار أي موضوع مشابه لها) ═══\n"
        f"{history_block}\n\n"
        "═══ قائمة الأخبار المرشحة للمفاضلة والتحرير ═══\n"
        + "\n\n---\n\n".join(candidate_blocks)
        + "\n\nالمطلوب: أعد كائن JSON متوافقاً مع BatchEditorialResponse يحدد decision (PUBLISH أو SKIP)، "
        "وعند قرار PUBLISH قم بصياغة منشورين متميزين من خبرين مختلفين (المنشور 1 بصيغة مرئية مع أوصاف صور بالإنجليزية، والمنشور 2 بصيغة text_satire مع سؤال تفاعلي)."
    )


def _normalize_batch_response(
    parsed: BatchEditorialResponse,
    candidate_map: Dict[int, Tuple[str, Article]],
    run_plan: Dict[str, Any],
    supported_districts: List[str],
    max_selections: int,
) -> BatchEditorialResponse:
    """Enforce hard CentralBrain format, scope, and quota invariants on Gemini output."""
    decision = (parsed.decision or "SKIP").strip().upper()
    if decision != "PUBLISH":
        return BatchEditorialResponse(
            decision="SKIP",
            skip_reason=(parsed.skip_reason or "Skipped by editorial triage").strip(),
            selected_posts=[],
        )

    allowed_formats = [str(f).strip() for f in (run_plan.get("allowed_formats") or ["photo_card"]) if str(f).strip()]
    if not allowed_formats:
        allowed_formats = ["photo_card"]
    allow_yemen_trend = bool(run_plan.get("allow_yemen_trend", False))

    valid_posts: List[EditorialPost] = []
    seen_indices = set()
    yemen_count = 0

    for post in parsed.selected_posts:
        if len(valid_posts) >= max(1, max_selections):
            break

        idx = post.candidate_index
        if idx not in candidate_map or idx in seen_indices:
            continue

        true_scope, source_article = candidate_map[idx]
        if true_scope == "yemen_trend":
            if not allow_yemen_trend or yemen_count >= 1:
                continue

        # Enforce format within allowed_formats and restrict map_reel to taiz only
        chosen_fmt = (post.chosen_format or "").strip()
        non_map_allowed = [f for f in allowed_formats if f != "map_reel"]

        if chosen_fmt not in allowed_formats:
            if true_scope == "taiz" and allowed_formats:
                chosen_fmt = allowed_formats[0]
            elif non_map_allowed:
                chosen_fmt = non_map_allowed[0]
            else:
                continue

        if chosen_fmt == "map_reel" and true_scope != "taiz":
            if non_map_allowed:
                chosen_fmt = next(
                    (f for f in ("standard_reel", "photo_card", "text_satire") if f in non_map_allowed),
                    non_map_allowed[0],
                )
            else:
                continue

        post.scope = true_scope
        post.is_relevant_to_taiz = true_scope == "taiz"
        post.should_publish = True
        post.chosen_format = chosen_fmt
        if not post.source_attribution:
            src = getattr(source_article, "source", "") or "مصادر محلية"
            post.source_attribution = f"نقلاً عن {src}"

        if chosen_fmt == "map_reel":
            if post.district_name not in supported_districts:
                post.district_name = "مدينة تعز"
            if not post.location_chip_text:
                post.location_chip_text = post.district_name
        elif chosen_fmt == "text_satire":
            sat_text = (post.satirical_text or post.clean_content or post.body or "").strip()
            # Ensure interactive question ending exists for comment engagement
            if not any(q in sat_text for q in ("؟", "?", "رايكم", "رأيكم", "شاركنا", "شاركونا")):
                sat_text = f"{sat_text}\n\n💬 كيف ترون هذه التطورات؟ شاركونا في التعليقات."
            post.satirical_text = sat_text
            post.clean_content = sat_text
            post.body = sat_text
            post.vocalized_content = sat_text
            if not post.headline:
                post.headline = source_article.title
            post.hashtags = list(post.hashtags[:3])

        seen_indices.add(idx)
        if true_scope == "yemen_trend":
            yemen_count += 1
        valid_posts.append(post)

    if not valid_posts:
        return BatchEditorialResponse(
            decision="SKIP",
            skip_reason=parsed.skip_reason or "No selected post satisfied CentralBrain scope/format rules.",
            selected_posts=[],
        )

    # Order posts so the visual post publishes first and text_satire schedules second
    if len(valid_posts) > 1:
        visual_posts = [p for p in valid_posts if p.chosen_format != "text_satire"]
        satire_posts = [p for p in valid_posts if p.chosen_format == "text_satire"]
        ordered_posts: List[EditorialPost] = []
        if visual_posts:
            ordered_posts.append(visual_posts[0])
        if satire_posts:
            ordered_posts.append(satire_posts[0])
        for p in valid_posts:
            if p not in ordered_posts and len(ordered_posts) < max_selections:
                ordered_posts.append(p)
        valid_posts = ordered_posts[:max_selections]

    return BatchEditorialResponse(decision="PUBLISH", skip_reason="", selected_posts=valid_posts)


def evaluate_and_draft_batch(
    taiz_candidates: List[Article],
    yemen_candidates: List[Article],
    run_plan: Dict[str, Any],
    recent_posts_context: Optional[str] = None,
    max_selections: int = 2,
    client: Optional[genai.Client] = None,
    model_name: Optional[str] = None,
) -> Optional[BatchEditorialResponse]:
    """Evaluate top Taiz and Yemen trend candidates in a single Gemini call and draft selected post(s)."""
    try:
        from .map_editorial import SUPPORTED_TAIZ_DISTRICTS  # Lazy import prevents circular import

        if max_selections <= 0:
            return BatchEditorialResponse(decision="SKIP", skip_reason="max_selections <= 0", selected_posts=[])

        indexed_candidates: List[Tuple[int, str, Article]] = []
        candidate_map: Dict[int, Tuple[str, Article]] = {}

        for art in taiz_candidates[:5]:
            idx = len(indexed_candidates) + 1
            indexed_candidates.append((idx, "taiz", art))
            candidate_map[idx] = ("taiz", art)

        if run_plan.get("allow_yemen_trend", False):
            for art in yemen_candidates[:2]:
                idx = len(indexed_candidates) + 1
                indexed_candidates.append((idx, "yemen_trend", art))
                candidate_map[idx] = ("yemen_trend", art)

        if not indexed_candidates:
            return BatchEditorialResponse(decision="SKIP", skip_reason="No candidate articles available.", selected_posts=[])

        active_client = client or get_genai_client()
        prompt = _build_batch_prompt(
            indexed_candidates=indexed_candidates,
            run_plan=run_plan,
            supported_districts=SUPPORTED_TAIZ_DISTRICTS,
            recent_posts_context=recent_posts_context,
            max_selections=max_selections,
        )

        config = types.GenerateContentConfig(
            system_instruction=BATCH_BRAIN_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=BatchEditorialResponse,
            temperature=0.35,
        )

        candidates = _resolve_candidate_models(model_name)
        if not candidates:
            print("⚠️ جميع نماذج Gemini المعتمدة استنفدت حصتها أو محظورة للـ 24 ساعة الحالية.")
            return None

        parsed = _generate_structured_with_fallback(
            active_client=active_client,
            prompt=prompt,
            config=config,
            response_model=BatchEditorialResponse,
            candidates=candidates,
            context_label="الدفعة التحريرية",
        )
        if parsed is None:
            LOGGER.error("All candidate models failed for batch editorial triage (%d candidates).", len(indexed_candidates))
            return None

        return _normalize_batch_response(
            parsed=parsed,
            candidate_map=candidate_map,
            run_plan=run_plan,
            supported_districts=SUPPORTED_TAIZ_DISTRICTS,
            max_selections=max_selections,
        )

    except Exception as exc:
        LOGGER.error("Batch editorial triage failed: %s", exc)
        return None
