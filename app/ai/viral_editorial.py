"""Viral Editorial Engine with Fast Pacing, Multi-Hook Selection & Seamless Loop.

Provides high-retention Facebook Reels / TikTok script generation:
  - 3 Algorithmic-Safe Hooks (Question, Shock, Impact) in the first 3 seconds
  - Fast-pacing body without full tashkeel (prevents voice drag and saves credits)
  - Seamless loop bridge without trailing ellipsis or punctuation
  - First-comment engagement question for maximum Facebook algorithmic reach
  - 5 sequential vertical (9:16) cinematic documentary storyboard prompts
  - Automatic alternation (A/B testing) between Classic and Viral reel engines
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reconfigure stdout/stderr for proper UTF-8 handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.collector.base import Article
from app.ai.editorial import get_genai_client
from app.ai.quota_manager import QUOTA_MANAGER, RATE_LIMITER
from app.ai.models import EditorialPost

load_dotenv()
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
ENGINE_STATE_FILE = BASE_DIR / "data" / "engine_state.json"


# ==============================================================================
# Pydantic Model for Viral Editorial Outputs
# ==============================================================================

class ViralEditorialPost(BaseModel):
    """Structured viral editorial evaluation and script design for Facebook Reels."""

    is_relevant_to_taiz: bool = Field(
        description="هل الخبر يخص محافظة تعز أو يؤثر عليها بشكل مباشر"
    )
    category: str = Field(
        description="تصنيف الخبر: عاجل / أمن / خدمات / مجتمع / محليات / اقتصاد / طقس / صحة"
    )
    importance_score: int = Field(
        description="درجة أهمية الخبر لأهالي محافظة تعز من 1 إلى 100"
    )
    headline: str = Field(
        description="عنوان صحفي مكثف وقوي بدون كليك بايت رخيص"
    )

    # 3 Distinct Viral Hooks for the first 3 seconds (Algorithmic Safe)
    question_hook: str = Field(
        description="خطاف سؤال مثير ومحفز للفضول بدون أي كلمات دموية صريحة (10-12 كلمة كحد أقصى)"
    )
    shock_hook: str = Field(
        description="خطاف حقيقة أو رقم أو واقعة صادمة بدون أي كلمات دموية صريحة (10-12 كلمة كحد أقصى)"
    )
    impact_hook: str = Field(
        description="خطاف يمس حياة الناس وأمنهم أو خدماتهم مباشرة بدون كلمات دموية صريحة (10-12 كلمة كحد أقصى)"
    )
    selected_hook: str = Field(
        description="الخطاف الأقوى المختار من الثلاثة أعلاه ليكون بداية الفيديو فوراً"
    )

    # Fast Pacing Body (Clean Text - No Full Tashkeel)
    fast_body: str = Field(
        description="صلب الخبر مصاغ في جمل قصيرة وسريعة متتالية (8-12 كلمة للجملة) بدون أي تشكيل كامل، مع فواصل عادية فقط"
    )

    # Seamless Loop Bridge
    loop_bridge: str = Field(
        description="جسر ختامي يحفز على المشاركة وينتهي بعبارة مفتوحة بدون أي نقاط حذف (...) أو علامات وقف تلتحم نحوياً ومعنوياً مع أول كلمة في selected_hook"
    )

    # First Comment Engagement Question
    first_comment_question: str = Field(
        description="سؤال تفاعلي مفتوح ومثير للاهتمام يطرح في أول تعليق مثبت على فيسبوك لزيادة التعليقات والنقاش"
    )

    # Sequential Storyboard Prompts
    image_prompts_en: List[str] = Field(
        default_factory=list,
        description="5 photorealistic news photography storyboard prompts, dramatic lighting, AP/Reuters photojournalism style, vertical 9:16, no text, representing event progression"
    )
    hashtags: List[str] = Field(
        default_factory=list,
        description="قائمة وسوم مناسبة تشمل #تعز والموضوع"
    )
    should_publish: bool = Field(
        default=True,
        description="القرار التحريري النهائي (True إذا كان الخبر مؤكداً ومهماً)"
    )
    rejection_reason: str = Field(
        default="",
        description="سبب الاستبعاد إذا كان should_publish = False"
    )


# ==============================================================================
# Viral Editorial System Prompt (Optimized for Fast Pacing & Algorithmic Safety)
# ==============================================================================

VIRAL_EDITORIAL_SYSTEM_PROMPT = """أنت رئيس التحرير الإبداعي وخبير صناعة المحتوى الفيروسي (Viral Video Producer) المتخصص في مقاطع Facebook Reels و TikTok الإخبارية لصفحة "تعز نيوز".

مهمتك: تحويل الخبر الوارد إلى سيناريو فيديو قصير فائق الجاذبية والسرعة، مصمم هندسياً لتحقيق أعلى معدلات المشاهدة والاحتفاظ بالمشاهد (Audience Retention) وإعادة المشاهدة التلقائية (Rewatch Rate) مع الالتزام التام بقواعد الأمان الخوارزمي.

══════════════════════════════════════════════════════════════════════
قواعد التحرير الفيروسي الخمس المحدثة (THE 5 VIRAL REEL RULES):
══════════════════════════════════════════════════════════════════════

1. قاعدة الخطاف الصادم مع الأمان الخوارزمي (Hook in First 3 Seconds & Safety):
   - يمنع منعاً باتاً البدايات الكلاسيكية الباردة ("أعلنت مصادر"، "في خبر عاجل"، "شهدت مدينة تعز"، "أفادت أنباء"، "صرح مسؤول").
   - قاعدة الأمان الخوارزمي الصارمة (Algorithmic Safety): ممنوع منعاً باتاً استخدام كلمات دموية صريحة في أول 3 ثوانٍ (مثل: "قتيلاً"، "قتلى"، "جثث"، "دماء"، "مصرع"، "مجزرة") لتجنب حظر الفيديو أو تقييد انتشاره (Shadowban).
   - استبدل الكلمات الدموية ببدائل صحفية حماسية وآمنة خوارزمياً مثل: ("ضحية"، "سقوط ضحايا"، "خسائر بشرية"، "تصعيد غير مسبوق"، "تطورات دامية"، "مواجهات عنيفة").
   - يجب توليد 3 خطافات بديلة ومبتكرة (طول كل خطاف 10 إلى 12 كلمة كحد أقصى):
     • question_hook (سؤال مثير للفضول): يطرح لغزاً أو تساؤلاً ملحاً يدفع المشاهد للمتابعة فوراً.
     • shock_hook (حقيقة أو رقم صادم وآمن): يبدأ برقم مفاجئ أو واقعة غير متوقعة دون كلمات دموية صريحة.
     • impact_hook (أثر مباشر على حياة الناس): يمس أمن المواطن، رغيف خبزه، أو طريقه المحاصر.
   - selected_hook: اختر الخطاف الأقوى والأكثر جاذبية بين الثلاثة ليكون هو الجملة الافتتاحية الأولى للفيديو.

2. إيقاع الجمل السريع ومنع التشكيل الكامل (Fast Pacing & STRICT NO FULL TASHKEEL):
   - قاعدة منع التشكيل الصارمة: ممنوع منعاً باتاً تشكيل النص بالكامل أو تشكيل أواخر الكلمات (NO FULL TASHKEEL). التشكيل الكامل هو السبب الرئيسي في بطء نطق محرك ElevenLabs وتمطيطه المزعج!
   - اكتب النص العربي بالكامل ناصعاً وخالياً من الحركات، واستخدم التشكيل فقط عند الضرورة القصوى في كلمة أو كلمتين ملتبستين جداً لمنع الخطأ (مثل كتابة: "تَعِز").
   - fast_body: صياغة صلب الخبر في جمل قصيرة ومكثفة ومترابطة (كل جملة تتراوح بين 8 إلى 12 كلمة فقط)، تفصل بينها فواصل عادية "،" فقط لضمان انسيابية الإلقاء.
   - الطول الإجمالي للنص الصوتي بالكامل (selected_hook + fast_body + loop_bridge) يجب أن يتراوح بين 65 إلى 75 كلمة فقط، بحيث يستغرق الإلقاء السريع المشدود بين 23 إلى 27 ثانية.

3. جسر الحلقة اللانهائية الصريح (Seamless Loop Bridge — NO ELLIPSIS):
   - loop_bridge: جملة ختامية تحفز على مشاركة الفيديو، ثم تنتهي بعبارة نحوية مفتوحة تنتهي بكلمة عادية **بدون أي نقاط حذف (...) أو شرطات (—) أو علامات وقف** نهائياً.
   - ممنوع وضع نقاط الحذف الثلاث (...) في نهاية loop_bridge أو نهاية النص الصوتي لأنها تجعل المذيع يبطئ صوته ويمططه في النهاية!
   - هذه النهاية المفتوحة يجب أن تلتحم لغوياً ونحوياً وسياقياً مع أول كلمة في selected_hook ليعيد الفيديو التشغيل تلقائياً بشكل دائري لا نهائي.
   - مثال:
     إذا كان selected_hook: "أكثر من ستين ضحية في أعنف تصعيد يشهده جنوب غرب اليمن"
     الـ loop_bridge: "شارك هذا الفيديو ليعرف الجميع الكواليس الكاملة وراء سقوط"
     (الالتحام: "...الكواليس الكاملة وراء سقوط ➔ أكثر من ستين ضحية في أعنف تصعيد...")

4. سؤال التعليق الأول (First Comment Engagement Question):
   - first_comment_question: سؤال نقاشي مفتوح وقوي حول قضية الخبر يُطرح في أول تعليق مثبت على فيسبوك لزيادة النقاشات والتعليقات.

5. أوامر القصة المصورة (Cinematic Storyboard Image Prompts):
   - image_prompts_en: قائمة من 5 أوصاف دقيقة باللغة الإنجليزية تمثل تسلسل القصة المصورة للخبر (The Hook, Core Action, Ground Level, Human Impact, Aftermath).
   - الأسلوب: Photorealistic documentary news photography, Reuters/AP style, dramatic natural lighting, 35mm lens, strictly vertical 9:16 aspect ratio, realistic Taiz / Yemen environment, no text, no CGI.
"""


# ==============================================================================
# Helper Functions: Fast TTS Preprocessor & Loop Bridge Cleaner
# ==============================================================================

def prepare_fast_tts_text(text: str) -> str:
    """Filter and clean Arabic text before TTS generation:
    - Eliminate ellipsis dots '...', long dashes, multiple exclamation marks
    - Strip full tashkeel/diacritics while preserving essential local pronunciation ('تَعِز')
    - Ensure clean single spaces and remove trailing pauses.
    """
    if not text:
        return ""

    protected_map = {
        "تَعِز": "___PROTECTED_TAIZ___",
        "تَعِزّ": "___PROTECTED_TAIZ___",
    }
    cleaned = text
    for word, placeholder in protected_map.items():
        cleaned = cleaned.replace(word, placeholder)

    cleaned = re.sub(r"[\u0617-\u061A\u064B-\u0652\u06D6-\u06ED]", "", cleaned)
    cleaned = cleaned.replace("___PROTECTED_TAIZ___", "تَعِز")

    cleaned = re.sub(r"\.{2,}", " ", cleaned)
    cleaned = re.sub(r"…+", " ", cleaned)
    cleaned = re.sub(r"[—–_]+", " ", cleaned)
    cleaned = re.sub(r"!+", "!", cleaned)
    cleaned = re.sub(r"\?+", "؟", cleaned)
    cleaned = re.sub(r"([،.!?])\1+", r"\1", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.rstrip(".،… ")
    return cleaned


def clean_loop_bridge(bridge: str) -> str:
    """Ensure the loop bridge has NO trailing punctuation for seamless infinite looping."""
    if not bridge:
        return ""
    cleaned = bridge.strip()
    return cleaned.rstrip(".،!؟:;…- ")


# ==============================================================================
# Alternator / Engine State Switcher (A/B Testing: Classic <-> Viral)
# ==============================================================================

def get_next_reel_engine() -> str:
    """Determine whether the current reel should use 'viral' or 'classic' engine.

    Modes supported via REEL_ENGINE_MODE environment variable:
      - 'alternate' (default): alternating 1-by-1 (Classic -> Viral -> Classic -> Viral)
      - 'viral' or 'viral_only': strictly use Viral engine every time
      - 'classic' or 'classic_only': strictly use Classic engine every time
    """
    mode = os.getenv("REEL_ENGINE_MODE", "alternate").strip().lower()
    if mode in ("viral", "viral_only"):
        return "viral"
    if mode in ("classic", "classic_only"):
        return "classic"

    # Default: Alternate mode
    state = _read_engine_state()
    last_engine = state.get("last_reel_engine", "classic")
    next_engine = "viral" if last_engine == "classic" else "classic"
    return next_engine


def record_reel_engine_used(engine_name: str) -> None:
    """Record which engine was used in data/engine_state.json for persistent state tracking."""
    state = _read_engine_state()
    state["last_reel_engine"] = engine_name
    if engine_name == "viral":
        state["total_viral_reels"] = state.get("total_viral_reels", 0) + 1
    else:
        state["total_classic_reels"] = state.get("total_classic_reels", 0) + 1

    try:
        ENGINE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ENGINE_STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("Could not persist engine state: %s", exc)


def _read_engine_state() -> Dict[str, Any]:
    """Read engine state JSON file."""
    try:
        if ENGINE_STATE_FILE.exists():
            return json.loads(ENGINE_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


# ==============================================================================
# Viral Editorial Engine Class
# ==============================================================================

class ViralEditorialEngine:
    """Executes advanced viral editorial prompt logic with cascading Gemini models."""

    def __init__(self, client: Optional[genai.Client] = None) -> None:
        self.client = client or get_genai_client()

    def build_prompt(self, article: Article, recent_posts_context: Optional[str] = None) -> str:
        """Format raw article details into the viral evaluation prompt."""
        description_text = article.description.strip() if article.description else "لا يوجد وصف إضافي متوفر."
        prompt = f"""يرجى تحليل الخبر التالي وصياغة سيناريو ريلز فيروسي سريع وحماسي وفق القواعد المحددة بدقة:

- المصدر: {article.source}
- العنوان الأصلي: {article.title}
- الرابط: {article.url}
- تاريخ النشر: {article.published_at.strftime('%Y-%m-%d %H:%M UTC')}

تفاصيل وسياق الخبر:
{description_text}
"""
        if recent_posts_context and recent_posts_context != "لا توجد منشورات سابقة مسجلة حتى الآن.":
            prompt += f"""
سياق المنشورات المنشورة مؤخراً لمنع تكرار نفس الزاوية:
{recent_posts_context}
"""
        return prompt

    def process_article(
        self,
        article: Article,
        recent_posts_context: Optional[str] = None,
    ) -> Optional[ViralEditorialPost]:
        """Pass article to Gemini using QuotaManager cascading with viral structured output."""
        prompt = self.build_prompt(article, recent_posts_context=recent_posts_context)
        config = types.GenerateContentConfig(
            system_instruction=VIRAL_EDITORIAL_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ViralEditorialPost,
            temperature=0.25,
        )

        candidates = QUOTA_MANAGER.get_available_models()
        if not candidates:
            candidates = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash"]

        for target_model in candidates:
            try:
                RATE_LIMITER.wait_for_slot(target_model)
                print(f"🤖 [Viral Engine] إرسال الخبر إلى Gemini عبر النموذج [{target_model}]...")

                response = self.client.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=config,
                )

                post: Optional[ViralEditorialPost] = None
                if hasattr(response, "parsed") and isinstance(response.parsed, ViralEditorialPost):
                    post = response.parsed
                elif response.text:
                    post = ViralEditorialPost.model_validate_json(response.text)

                if post:
                    QUOTA_MANAGER.record_success(target_model)
                    post.loop_bridge = clean_loop_bridge(post.loop_bridge)
                    return post

            except Exception as api_err:
                err_str = str(api_err)
                if "429" in err_str or "quota" in err_str.lower() or "resource_exhausted" in err_str.lower():
                    QUOTA_MANAGER.mark_exhausted(target_model, reason="429 Quota Exceeded")
                    print(f"⛔ تم بلوغ حد الحصة للنموذج [{target_model}]. التبديل للنموذج التالي...")
                else:
                    LOGGER.warning("Gemini model %s error: %s", target_model, api_err)
                    print(f"⚠️ خطأ أثناء الطلب بالنموذج [{target_model}]: {err_str[:120]}")
                continue

        LOGGER.error("All Gemini candidate models failed for viral article: %s", article.title)
        return None


# ==============================================================================
# Adapter Layer: ViralEditorialPost -> Production EditorialPost
# ==============================================================================

def adapt_viral_to_editorial_post(
    viral_post: ViralEditorialPost,
    article: Article,
) -> EditorialPost:
    """Map ViralEditorialPost to standard EditorialPost model for seamless pipeline reuse.

    Attaches viral metadata (viral_data, first_comment_question, is_viral=True) to the post.
    """
    hook = prepare_fast_tts_text(viral_post.selected_hook)
    body = prepare_fast_tts_text(viral_post.fast_body)
    bridge = clean_loop_bridge(prepare_fast_tts_text(viral_post.loop_bridge))

    # Fast broadcast flow: separate with natural commas without drag
    full_narration_clean = f"{hook}، {body}، {bridge}"

    prompts = viral_post.image_prompts_en if viral_post.image_prompts_en else []
    primary_prompt = prompts[0] if prompts else ""

    adapted = EditorialPost(
        is_relevant_to_taiz=viral_post.is_relevant_to_taiz,
        category=viral_post.category,
        importance_score=viral_post.importance_score,
        headline=viral_post.headline,
        clean_content=full_narration_clean,
        vocalized_content=full_narration_clean,
        body=full_narration_clean,
        source_attribution=article.source if article else "تعز نيوز",
        hashtags=viral_post.hashtags,
        should_publish=viral_post.should_publish,
        rejection_reason=viral_post.rejection_reason,
        image_prompt_en=primary_prompt,
        image_prompts_en=prompts,
        video_keywords_en=[],
    )

    # Attach dynamic viral attributes for downstream consumers
    adapted.__dict__["is_viral"] = True
    adapted.__dict__["first_comment_question"] = viral_post.first_comment_question
    adapted.__dict__["viral_post"] = viral_post

    return adapted


def process_viral_editorial(
    article: Article,
    recent_posts_context: Optional[str] = None,
) -> Optional[EditorialPost]:
    """Convenience helper: Run viral editorial processing and return adapted EditorialPost."""
    engine = ViralEditorialEngine()
    viral_post = engine.process_article(article, recent_posts_context=recent_posts_context)
    if not viral_post:
        return None
    return adapt_viral_to_editorial_post(viral_post, article)
