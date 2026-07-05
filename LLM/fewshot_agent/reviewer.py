
from __future__ import annotations

import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any, Dict, List

from fewshot_agent.fewshot_rules import FEWSHOT_RULES


def normalize_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[()（）\[\]【】ㆍ·.,:;]", "", text)
    return text.strip().lower()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def extract_content(value: Any) -> str:
    if isinstance(value, dict):
        return (
            value.get("pep_excerpt")
            or value.get("rfp_excerpt")
            or value.get("rpt_excerpt")
            or value.get("text")
            or value.get("content")
            or ""
        )
    return str(value or "")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_sections(parsed_sections: Any) -> List[Dict[str, str]]:
    sections: List[Dict[str, str]] = []

    if isinstance(parsed_sections, list):
        records = parsed_sections
    elif isinstance(parsed_sections, dict):
        if all(isinstance(v, dict) for v in parsed_sections.values()):
            records = list(parsed_sections.values())
        else:
            records = [
                {"section_id": "", "section_title": k, "content": v}
                for k, v in parsed_sections.items()
            ]
    else:
        return sections

    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue

        title = (
            record.get("section_title")
            or record.get("title")
            or record.get("heading")
            or record.get("name")
            or f"섹션{idx}"
        )
        subtitle = record.get("section_subtitle") or ""
        content = extract_content(record.get("content", ""))

        sections.append({
            "section_id": str(record.get("section_id") or "").strip(),
            "section_title": str(title).strip(),
            "section_subtitle": str(subtitle).strip(),
            "title_for_match": f"{title} {subtitle}".strip(),
            "content": content.strip(),
        })

    return sections


def find_section_by_title(sections: List[Dict[str, str]], title: str) -> tuple[Dict[str, str] | None, float]:
    title_norm = normalize_text(title)

    best = None
    best_score = 0.0

    for section in sections:
        sec_title = section["title_for_match"]
        sec_norm = normalize_text(sec_title)

        if title_norm and (title_norm in sec_norm or sec_norm in title_norm):
            return section, 1.0

        score = similarity(title, sec_title)
        if score > best_score:
            best = section
            best_score = score

    if best_score >= 0.55:
        return best, best_score

    return None, best_score


def contains_any(text: str, keywords: List[str]) -> List[str]:
    text_norm = normalize_text(text)
    matched = []
    for keyword in keywords:
        if normalize_text(keyword) in text_norm:
            matched.append(keyword)
    return matched


def build_expected_text(shot: Dict[str, Any]) -> str:
    """
    Few-shot 예시와 필수 키워드를 합쳐 비교 기준 문장을 만든다.
    이 문장과 실제 content를 비교해서 유사도 점수를 계산한다.
    """
    parts = []
    parts.append(shot.get("title", ""))
    parts.extend(shot.get("must_have", []))
    parts.append(shot.get("good", ""))
    return " ".join(parts)


def review_fewshot(parsed_sections: Any, doc_type: str) -> Dict[str, Any]:
    if doc_type not in FEWSHOT_RULES:
        raise ValueError(f"지원하지 않는 doc_type입니다: {doc_type}")

    rules = FEWSHOT_RULES[doc_type]
    sections = normalize_sections(parsed_sections)

    issues: List[Dict[str, Any]] = []
    section_reviews: List[Dict[str, Any]] = []

    parsed_titles = [normalize_text(s["title_for_match"]) for s in sections]

    for required_title in rules["required"]:
        required_norm = normalize_text(required_title)
        if not any(required_norm in t or t in required_norm for t in parsed_titles):
            issues.append({
                "issue_type": "missing_section",
                "title": required_title,
                "message": f"필수 소제목 '{required_title}'이 파싱 결과에 없습니다.",
                "severity": "blocking",
            })

    for section in sections:
        if not normalize_text(section["content"]):
            issues.append({
                "issue_type": "empty_section",
                "title": section["title_for_match"],
                "message": f"'{section['title_for_match']}' 항목의 내용이 비어 있습니다.",
                "severity": "blocking",
            })

    similarity_scores = []

    for shot in rules["fewshots"]:
        section, title_similarity = find_section_by_title(sections, shot["title"])
        if not section:
            continue

        must_have = shot.get("must_have", [])
        matched = contains_any(section["content"], must_have)
        missing = [k for k in must_have if k not in matched]
        keyword_score = len(matched) / len(must_have) if must_have else 1.0

        expected_text = build_expected_text(shot)
        content_similarity = similarity(section["content"], expected_text)
        similarity_scores.append(content_similarity)

        if keyword_score >= 0.7:
            status = "PASS"
        elif keyword_score >= 0.4:
            status = "WARN"
        else:
            status = "FAIL"

        review = {
            "title": shot["title"],
            "matched_section": section["title_for_match"],
            "status": status,
            "keyword_score": round(keyword_score, 2),
            "title_similarity": round(title_similarity, 4),
            "content_similarity": round(content_similarity, 4),
            "matched_items": matched,
            "missing_items": missing,
            "good_example": shot.get("good", ""),
            "bad_example": shot.get("bad", ""),
        }
        section_reviews.append(review)

        if status == "FAIL":
            issues.append({
                "issue_type": "fewshot_quality_fail",
                "title": shot["title"],
                "message": f"'{shot['title']}' 항목이 few-shot 기준을 충족하지 못했습니다.",
                "missing_items": missing,
                "similarity": round(content_similarity, 4),
                "severity": "blocking",
            })
        elif status == "WARN":
            issues.append({
                "issue_type": "fewshot_quality_warn",
                "title": shot["title"],
                "message": f"'{shot['title']}' 항목은 일부 기준이 부족합니다.",
                "missing_items": missing,
                "similarity": round(content_similarity, 4),
                "severity": "info",
            })

    blocking = [i for i in issues if i.get("severity") == "blocking"]
    info = [i for i in issues if i.get("severity") != "blocking"]

    if blocking:
        final_status = "FAIL"
    elif info:
        final_status = "WARN"
    else:
        final_status = "PASS"

    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0

    return {
        "document_type": doc_type,
        "review_status": final_status,
        "passed": final_status == "PASS",
        "can_auto_proceed": final_status != "FAIL",
        "section_count": len(sections),
        "issue_count": len(issues),
        "blocking_issue_count": len(blocking),
        "info_issue_count": len(info),
        "average_content_similarity": round(avg_similarity, 4),
        "issues": issues,
        "fewshot_section_reviews": section_reviews,
    }
