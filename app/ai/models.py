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
        description="نفس متن المنشور مشكولاً بتشكيل ذكي مخفف للكلمات الصعبة والملتبسة وأسماء الأعلام فقط لضمان النطق السليم وتوفير الحروف"
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
    chosen_format: str = Field(
        default="photo_card",
        description="Selected post format: text_satire, photo_card, standard_reel, or map_reel."
    )
    scope: str = Field(
        default="taiz",
        description="News scope: taiz or yemen_trend."
    )
    candidate_index: int = Field(
        default=1,
        description="1-based index of the selected candidate article from the batch list."
    )
    satirical_text: str = Field(
        default="",
        description="Short satirical political/social post in Yemeni dialect when chosen_format == 'text_satire'."
    )
    district_name: str = Field(
        default="مدينة تعز",
        description="Target Taiz district name when chosen_format == 'map_reel'."
    )
    map_label: str = Field(
        default="",
        description="Short Arabic pin label displayed on the map when chosen_format == 'map_reel'."
    )
    location_chip_text: str = Field(
        default="",
        description="Location header chip text when chosen_format == 'map_reel'."
    )
    hook_text: str = Field(
        default="",
        description="Concise opening hook line when chosen_format == 'map_reel'."
    )
    teaser: str = Field(
        default="",
        description="Short teaser overlay/caption when chosen_format == 'map_reel'."
    )
    is_military: bool = Field(
        default=False,
        description="True for military or security events requiring tactical map styling."
    )

    @model_validator(mode="after")
    def sync_content_fields(self) -> "EditorialPost":
        """Ensure satirical_text, clean_content, vocalized_content, and body stay synchronized."""
        if self.chosen_format == "text_satire":
            if self.satirical_text and not self.clean_content:
                self.clean_content = self.satirical_text
            elif self.clean_content and not self.satirical_text:
                self.satirical_text = self.clean_content
        if not self.clean_content and self.body:
            self.clean_content = self.body
        if not self.body and self.clean_content:
            self.body = self.clean_content
        if not self.vocalized_content and self.clean_content:
            self.vocalized_content = self.clean_content
        return self


class BatchEditorialResponse(BaseModel):
    """Structured output for the single-call batch editorial triage and drafting engine."""

    decision: str = Field(
        default="SKIP",
        description="PUBLISH if a candidate merits publishing, or SKIP if candidates are weak, duplicate, or below threshold."
    )
    skip_reason: str = Field(
        default="",
        description="Short explanation when decision == 'SKIP'."
    )
    selected_posts: List[EditorialPost] = Field(
        default_factory=list,
        description="Selected and fully drafted posts (typically 1, or up to max_selections during peak hours)."
    )


