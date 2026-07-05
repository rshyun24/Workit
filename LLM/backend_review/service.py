from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from backend_review.engine import review_with_rule_first
from backend_review.io_utils import (
    load_json,
    load_text,
    normalize_sections_for_review,
    write_json,
)


SUPPORTED_DOC_TYPES = {"rfp", "pep", "rpt"}


class ReviewServiceError(ValueError):
    """백엔드 검수 서비스 예외"""


def normalize_doc_type(document_type: str) -> str:
    doc_type = (document_type or "").strip().lower()

    if doc_type not in SUPPORTED_DOC_TYPES:
        supported = ", ".join(sorted(SUPPORTED_DOC_TYPES))
        raise ReviewServiceError(
            f"Unsupported document_type: {document_type!r}. "
            f"Use one of: {supported}"
        )

    return doc_type


def detect_doc_type(
    *,
    filename: str = "",
    original_text: str = "",
    parsed_sections: Any = None,
) -> Optional[str]:
    """
    파일명, 원본 텍스트, section_id 등을 이용하여 문서 타입을 자동 추정한다.
    """
    haystacks = [
        filename or "",
        original_text[:2000] if original_text else "",
    ]

    normalized_sections = normalize_sections_for_review(parsed_sections)

    if isinstance(normalized_sections, list):
        for section in normalized_sections[:5]:
            if isinstance(section, dict):
                haystacks.append(str(section.get("section_id") or ""))
                haystacks.append(str(section.get("section_title") or ""))

    elif isinstance(normalized_sections, dict):
        haystacks.extend(
            str(key)
            for key in list(normalized_sections.keys())[:5]
        )

    text = " ".join(haystacks).lower()

    for doc_type in ("rfp", "pep", "rpt"):
        if doc_type in text:
            return doc_type

    return None


def review_parsed_document(
    *,
    original_text: str,
    parsed_sections: Any,
    document_type: str,
    include_fewshot: bool = True,
) -> Dict[str, Any]:
    """
    백엔드에서 호출하는 메인 함수.

    새 key-value 평가셋 구조도 그대로 받을 수 있다.
    content가 dict인 경우 engine.py에서 문자열로 평탄화하여 qa_agent/fewshot_agent에 전달한다.
    """
    document_type = normalize_doc_type(document_type)

    return review_with_rule_first(
        original_text=original_text,
        parsed_sections=parsed_sections,
        document_type=document_type,
        include_fewshot=include_fewshot,
    )


def review_files(
    *,
    original_path: str | Path,
    parsed_path: str | Path,
    document_type: Optional[str] = None,
    include_fewshot: bool = True,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    original_text = load_text(original_path)
    parsed_sections = load_json(parsed_path)

    original_file = Path(original_path)
    parsed_file = Path(parsed_path)

    doc_type = document_type or detect_doc_type(
        filename=f"{original_file.name} {parsed_file.name}",
        original_text=original_text,
        parsed_sections=parsed_sections,
    )

    if not doc_type:
        raise ReviewServiceError(
            "document_type could not be detected. Pass rfp, pep, or rpt explicitly."
        )

    payload = review_parsed_document(
        original_text=original_text,
        parsed_sections=parsed_sections,
        document_type=doc_type,
        include_fewshot=include_fewshot,
    )

    if output_path:
        write_json(output_path, payload)

    return payload


def summarize_for_backend(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    프론트/백엔드 응답용 요약 결과.

    Few-shot 상세 section_reviews는 제외하고,
    최종 판정과 주요 검수 결과만 반환한다.
    """
    return {
        "document_type": payload.get("document_type"),
        "review_status": payload.get("review_status"),
        "passed": payload.get("passed"),
        "can_auto_proceed": payload.get("can_auto_proceed"),
        "content_similarity": payload.get("content_similarity"),
        "expected_section_count": payload.get("expected_section_count"),
        "matched_section_count": payload.get("matched_section_count"),
        "issue_count": payload.get("issue_count"),
        "blocking_issue_count": payload.get("blocking_issue_count"),
        "info_issue_count": payload.get("info_issue_count"),
        "issues": payload.get("issues") or [],
    }


def iter_issue_messages(
    payload: Dict[str, Any],
) -> Iterable[str]:
    for issue in payload.get("issues") or []:
        issue_type = issue.get("issue_type") or "-"
        code = issue.get("code") or "-"
        title = issue.get("title") or "-"
        message = issue.get("message") or ""

        yield (
            f"[{issue_type}] "
            f"{code} / "
            f"{title}: "
            f"{message}"
        )
