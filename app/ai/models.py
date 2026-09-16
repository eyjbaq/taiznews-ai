"""Pydantic models for structured AI editorial outputs."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class EditorialPost(BaseModel):
    """Structured editorial evaluation and publication draft for Taiz News."""

    is_relevant_to_taiz: bool = Field(
        description="هل الخبر يخص محافظة أو مدينة تعز أو يؤثر عليها بشكل مباشر"
    )
    category: str = Field(
        description="تصنيف الخبر: أمن / مجتمع / خدمات / طقس / عاجل / محليات / صحة / اقتصاد"
    )
    importance_score: int = Field(
        description="درجة أهمية الخبر لأهالي محافظة تعز من 1 إلى 100"
    )
    headline: str = Field(
        description="عنوان صحفي جذاب ورصين ومهني بدون أي تهويل أو أسلوب Clickbait"
    )
    clean_content: str = Field(
        default="",
        description="متن المنشور: فقرة أو فقرتان بلغة عربية صحفية دقيقة بدون أي تشكيل أو حركات (مخصص للعرض على الشاشة والمنشور)"
    )
    vocalized_content: str = Field(
        default="",
        description="نفس متن المنشور تماماً مشكولاً بالحركات الإعرابية الكاملة بدقة لغوية ونحوية لضمان النطق الصوتي الدقيق عبر التعليق الصوتي TTS"
    )
    body: str = Field(
        default="",
        description="متن المنشور التقليدي (متوافق مع clean_content)"
    )
    source_attribution: str = Field(
        description="إسناد واضح للمصدر الأصلي (مثال: نقلاً عن الجزيرة، بحسب وكالة سبأ)"
    )
    hashtags: List[str] = Field(
        description="قائمة وسوم مناسبة للمنشور تشمل #تعز واسم المنطقة أو الموضوع"
    )
    should_publish: bool = Field(
        description="القرار التحريري النهائي (True إذا كان الخبر مؤكداً وموثوقاً ومهماً للنشر)"
    )
    rejection_reason: str = Field(
        default="",
        description="سبب استبعاد الخبر إذا كان should_publish = False"
    )
    image_prompt_en: str = Field(
        default="",
        description="A detailed English prompt for generating a photorealistic editorial background image related to the news event."
    )
    image_prompts_en: List[str] = Field(
        default_factory=list,
        description="A sequential list of 4 to 5 detailed English image prompts creating a storyboard matching the progression of the news event in Taiz / Yemen (Hook, Core Action, Perspective/Detail, Human Context, Resolution)."
    )
    video_keywords_en: List[str] = Field(
        default_factory=list,
        description=(
            "3 to 5 strictly impersonal English search keywords for stock vertical B-Roll videos. "
            "STRICT RULE: NEVER include people, soldiers, troops, men, women, faces, officers, or crowds. "
            "Force keywords to be object-based or environmental (e.g. 'military vehicles', 'armored truck', "
            "'desert landscape', 'mountain roads', 'barricade', 'flashing police lights', 'barbed wire', "
            "'destroyed buildings', 'Middle East city aerial', 'Yemen architecture')."
        )
    )

    @model_validator(mode="after")
    def sync_content_fields(self) -> "EditorialPost":
        """Ensure clean_content, vocalized_content, and body stay synchronized."""
        if not self.clean_content and self.body:
            self.clean_content = self.body
        if not self.body and self.clean_content:
            self.body = self.clean_content
        if not self.vocalized_content and self.clean_content:
            self.vocalized_content = self.clean_content
        return self


