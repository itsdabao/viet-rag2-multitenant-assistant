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

    # 0.5) Language Guardrail: English queries
    if p.language == "en":
        return RouteDecision(
            "out_of_domain",
            1.0,
            "language_mismatch:en",
        )

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

    # 3) LLM-based Semantic Router (Local Qwen 2.5)
    # Fallback to regex if LLM fails or returns invalid JSON (handled by safety try-except).
    try:
        from llama_index.core import Settings
        
        prompt = (
            "You are a helpful assistant found in a customer service chatbot.\n"
            "Classify the user query into one of the following intents:\n"
            "- tuition_calculator: questions about price, discount, tuition fees, cost.\n"
            "- comparison: comparing two or more courses (e.g. 'A vs B') only. DO NOT classify score conversion/mapping questions here (put them in course_search).\n"
            "- create_ticket: user wants to speak to human, consultant, or register/enroll.\n"
            "- course_search: general questions about course content, schedule, syllabus, policy, exam structure, score mapping/conversion (e.g. 'TOEIC to IELTS').\n"
            "- smalltalk: greeting, thanks, bye.\n"
            "\n"
            f"Query: \"{q}\"\n"
            "Return ONLY the intent name in JSON format: {\"intent\": \"...\"}"
        )
        
        # Use simple complete because we need raw text to parse JSON
        if Settings.llm:
            resp = Settings.llm.complete(prompt)
            raw = resp.text.strip()
            # Simple JSON extraction
            m = re.search(r"\{\s*\"intent\"\s*:\s*\"([^\"]+)\"\s*\}", raw)
            if m:
                intent = m.group(1).lower()
                # Map unknown intents back to course_search
                valid_intents = ["tuition_calculator", "comparison", "create_ticket", "course_search", "smalltalk"]
                if intent in valid_intents:
                    return RouteDecision(intent, 0.95, f"llm_classified:{intent}")
    except Exception as e:
        logger.warning("LLM router failed, falling back to regex: %s", e)

    # 4) Fallback: Regex Heuristic (Legacy)
    qn = _norm_ascii(q)

    # Handoff / ticket intent
    has_consult_phrase = any(
        k in qn
        for k in [
            "can tu van",
            "muon tu van",
            "nho tu van",
            "tu van giup",
            "tu van ho",
            "tu van em",
            "tu van minh",
        ]
    )
    has_question_like = ("?" in q) or any(
        k in qn
        for k in [
            "bao nhieu",
            "khi nao",
            "o dau",
            "dia chi",
            "lich hoc",
            "hoc phi",
            "uu dai",
        ]
    )
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
            "đăng ký tư vấn",
            "dang ky tu van",
            "đăng ký học",
            "dang ky hoc",
            "ghi danh",
            "đặt lịch",
            "dat lich",
        ],
    ) or (has_consult_phrase and not has_question_like):
        return RouteDecision("create_ticket", 0.9, "handoff_or_phone")

    # Comparison intent
    has_compare_cue = any(k in qn for k in ["so sanh", "khac nhau", "nen hoc", "vs"])
    has_compare_sep = bool(re.search(r"\b(vs|voi)\b", qn)) or ("so sanh" in qn and re.search(r"\bva\b", qn))
    if has_compare_cue and has_compare_sep and len(qn) >= 10:
        return RouteDecision("comparison", 0.7, "comparison_keywords")

    # Tuition calculator intent
    if re.search(r"(?i)\bgiảm\b", q) or "%" in q or _has_any(q, ["sau giảm", "sau giam", "discount", "promo","giảm giá","giảm","khuyến mãi"]):
        return RouteDecision("tuition_calculator", 0.8, "discount_keywords")
    if _has_any(q, ["bảo lưu", "bao luu", "hoàn phí", "hoan phi", "học bù", "hoc bu", "chính sách", "chinh sach", "quy định", "quy dinh"]):
        return RouteDecision("course_search", 0.8, "policy_keywords")
    if _has_any(
        q,
        ["học phí", "hoc phi", "bao nhiêu tiền", "bao nhieu tien", "giá", "gia", "phí", "phi", "ưu đãi", "uu dai"],
    ):
        return RouteDecision("tuition_calculator", 0.75, "tuition_keywords")

    # Default: course_search
    return RouteDecision("course_search", 0.6, "default_rag")


def out_of_domain_answer() -> str:
    return OUT_OF_DOMAIN_MESSAGE
