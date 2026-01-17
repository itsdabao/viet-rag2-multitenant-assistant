from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, Optional

from app.core.config import (
    DOMAIN_ANCHORS_PATH,
    DOMAIN_ANCHOR_COSINE_THRESHOLD,
    DOMAIN_KEYWORDS,
    ENABLE_DOMAIN_GUARD,
    ENABLE_SMALLTALK,
    OUT_OF_DOMAIN_MESSAGE,
    SMALLTALK_COSINE_THRESHOLD,
    SMALLTALK_PATH,
)
from app.services.guardrails.domain_guard import DomainGuard
from app.services.guardrails.smalltalk import SmalltalkMatcher

from .preprocess import PreprocessResult


logger = logging.getLogger(__name__)

RouteName = Literal[
    "smalltalk",
    "out_of_domain",
    "course_search",
    "tuition_calculator",
    "comparison",
    "create_ticket",
]


@dataclass(frozen=True)
class RouteDecision:
    route: RouteName
    confidence: float
    reason: str
    smalltalk_answer: Optional[str] = None


_SMALLTALK = SmalltalkMatcher(SMALLTALK_PATH)
_DOMAIN_GUARD = DomainGuard(DOMAIN_ANCHORS_PATH, keywords=DOMAIN_KEYWORDS)


def _has_any(q: str, kws: list[str]) -> bool:
    qq = (q or "").lower()
    return any(k in qq for k in kws)


def _norm_ascii(s: str) -> str:
    s = (s or "").lower()
    try:
        s = unicodedata.normalize("NFD", s)
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    except Exception:
        pass
    s = s.replace("đ", "d")
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def route_query(p: PreprocessResult) -> RouteDecision:
    q = p.query

    # 0) Toxic is handled in preprocess workflow (outside router).

    # 1) Cheap smalltalk (avoid retrieval/LLM)
    if ENABLE_SMALLTALK:
        hit = _SMALLTALK.match(q, threshold=SMALLTALK_COSINE_THRESHOLD)
        if hit is not None:
            return RouteDecision("smalltalk", 1.0, f"smalltalk:{hit.id}", smalltalk_answer=hit.answer)

    # 2) Cheap out-of-domain gate (avoid retrieval/LLM)
    if ENABLE_DOMAIN_GUARD:
        try:
            decision = _DOMAIN_GUARD.decide(q, threshold=DOMAIN_ANCHOR_COSINE_THRESHOLD)
            if not decision.in_domain:
                return RouteDecision("out_of_domain", 1.0, f"domain:{decision.reason}:{decision.score:.3f}")
        except Exception as e:
            logger.debug("domain_guard failed: %s", e)

    # 3) Handoff / ticket intent
    if p.phone or _has_any(
        q,
        [
            "tư vấn viên",
            "tu van vien",
            "gặp người",
            "gap nguoi",
            "người thật",
            "nguoi that",
            "liên hệ",
            "lien he",
            # Avoid generic "đăng ký" because it appears in discount contexts ("đăng ký sớm giảm 10%").
            "đăng ký tư vấn",
            "dang ky tu van",
            "đăng ký học",
            "dang ky hoc",
            "ghi danh",
            "đặt lịch",
            "dat lich",
        ],
    ):
        return RouteDecision("create_ticket", 0.9, "handoff_or_phone")

    # 4) Comparison intent
    qn = _norm_ascii(q)
    # Avoid routing "X với Y" as comparison unless there is an explicit comparison cue.
    has_compare_cue = any(k in qn for k in ["so sanh", "khac nhau", "nen hoc", "vs"])
    has_compare_sep = bool(re.search(r"\b(vs|voi)\b", qn)) or ("so sanh" in qn and re.search(r"\bva\b", qn))
    if has_compare_cue and has_compare_sep and len(qn) >= 10:
        return RouteDecision("comparison", 0.7, "comparison_keywords")

    # 5) Tuition calculator intent
    if re.search(r"(?i)\bgiảm\b", q) or "%" in q or _has_any(q, ["sau giảm", "sau giam", "discount", "promo","giảm giá","giảm","khuyến mãi"]):
        return RouteDecision("tuition_calculator", 0.8, "discount_keywords")
    # Policy questions should go to RAG (not calculator), even if they contain "phí".
    if _has_any(q, ["bảo lưu", "bao luu", "hoàn phí", "hoan phi", "học bù", "hoc bu", "chính sách", "chinh sach", "quy định", "quy dinh"]):
        return RouteDecision("course_search", 0.8, "policy_keywords")
    if _has_any(
        q,
        ["học phí", "hoc phi", "bao nhiêu tiền", "bao nhieu tien", "giá", "gia", "phí", "phi", "ưu đãi", "uu dai"],
    ):
        return RouteDecision("tuition_calculator", 0.75, "tuition_keywords")

    # Default: course search (RAG)
    return RouteDecision("course_search", 0.6, "default_rag")


def out_of_domain_answer() -> str:
    return OUT_OF_DOMAIN_MESSAGE
