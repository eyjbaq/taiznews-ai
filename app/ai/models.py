"""Pydantic models for structured AI editorial outputs."""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


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
    body: str = Field(
        description="متن المنشور: فقرة أو فقرتان بلغة عربية صحفية موضوعية ودقيقة خالية من التضخيم"
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
