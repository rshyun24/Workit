from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from qa_agent.engine import review_section_mapping

from backend_review.io_utils import (
    get_section_content,
    get_standard_structure,
    normalize_sections_for_review,
)
from fewshot_agent.recommendations import build_fewshot_recommendations


PRIMARY_ENGINE = "qa_agent"
OPTIONAL_ENGINE = "fewshot_agent"
COMBINED_ENGINE = "rule_first_fewshot_second"

BORDERLINE_SECTION_SIMILARITY = 0.70


def run_rule_review(
    *,
    original_text: str,
    parsed_sections: Any,
    document_type: str,
) -> Dict[str, Any]:
    """
    Rule 기반 qa_agent 실행.

    key-value 평가셋의 content가 dict일 수 있으므로,
    qa_agent에 넘기기 전에 content를 문자열로 평탄화한다.
    """
    normalized_sections = normalize_sections_for_review(parsed_sections)

    return review_section_mapping(
        original_text=original_text,
        parsed_sections=normalized_sections,
        document_type=document_type,
    )


def run_fewshot_review(
    *,
    parsed_sections: Any,
    document_type: str,
) -> Dict[str, Any]:
    """
    Few-shot 보조 검수 실행.

    fewshot_agent도 content 문자열을 기준으로 보기 때문에,
    content를 문자열로 평탄화한 데이터를 넘긴다.
    """
    normalized_sections = normalize_sections_for_review(parsed_sections)

    llm_dir = Path(__file__).resolve().parents[1]

    if str(llm_dir) not in sys.path:
        sys.path.insert(0, str(llm_dir))

    try:
        from LLM.fewshot_agent.reviewer import review_fewshot
    except ModuleNotFoundError:
        from fewshot_agent.reviewer import review_fewshot

    return review_fewshot(normalized_sections, document_type)


def build_payload_from_rule(rule_report: Dict[str, Any]) -> Dict[str, Any]:
    issues = list(rule_report.get("issues") or [])

    blocking_issues = [
        issue
        for issue in issues
        if issue.get("severity") == "blocking"
    ]

    info_issues = [
        issue
        for issue in issues
        if issue.get("severity") != "blocking"
    ]

    return {
        "engine": COMBINED_ENGINE,
        "selected_model": COMBINED_ENGINE,
        "document_type": rule_report.get("document_type"),
        "review_status": rule_report.get("review_status"),
        "passed": bool(rule_report.get("passed")),
        "can_auto_proceed": bool(rule_report.get("can_auto_proceed")),
        "content_similarity": rule_report.get("content_similarity"),
        "expected_section_count": rule_report.get("expected_section_count"),
        "matched_section_count": rule_report.get("matched_section_count"),
        "issue_count": len(issues),
        "blocking_issue_count": len(blocking_issues),
        "info_issue_count": len(info_issues),
        "issues": issues,
        "recommendations": [],
        "decision_stage": "rule_checked",
        "selection_reason": "Rule 기반 qa_agent를 1차 주요 판단으로 사용합니다.",
        "diagnostics": {
            "primary": {
                "engine": PRIMARY_ENGINE,
                "review_status": rule_report.get("review_status"),
                "content_similarity": rule_report.get("content_similarity"),
            }
        },
    }


def _refresh_issue_counts(payload: Dict[str, Any]) -> None:
    issues = list(payload.get("issues") or [])

    blocking_issues = [
        issue
        for issue in issues
        if issue.get("severity") == "blocking"
    ]

    info_issues = [
        issue
        for issue in issues
        if issue.get("severity") != "blocking"
    ]

    payload["issue_count"] = len(issues)
    payload["blocking_issue_count"] = len(blocking_issues)
    payload["info_issue_count"] = len(info_issues)


def _extract_similarity_from_message(message: str) -> float | None:
    """
    qa_agent 이슈 메시지 안의 '(유사도 0.74)' 값을 추출한다.
    """
    match = re.search(r"유사도\s+([0-9.]+)", message or "")

    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def soften_borderline_content_mismatch(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    section_content_mismatch 중 유사도가 0.70 이상인 애매한 케이스는
    치명 이슈가 아니라 참고 이슈로 낮춘다.

    명확한 누락, 빈 섹션, 내용 오배치는 그대로 FAIL 유지한다.
    """
    changed = False

    for issue in payload.get("issues") or []:
        if issue.get("issue_type") != "section_content_mismatch":
            continue

        similarity = _extract_similarity_from_message(
            issue.get("message", "")
        )

        if similarity is None:
            continue

        if similarity >= BORDERLINE_SECTION_SIMILARITY:
            issue["severity"] = "info"
            changed = True

    if not changed:
        return payload

    _refresh_issue_counts(payload)

    if payload["blocking_issue_count"] == 0:
        payload["review_status"] = "WARN"
        payload["passed"] = False
        payload["can_auto_proceed"] = True
        payload["decision_stage"] = "borderline_content_warn"

    return payload


def _add_issue(
    payload: Dict[str, Any],
    *,
    issue_type: str,
    code: str,
    title: str,
    message: str,
    severity: str,
    source: str | None = None,
) -> None:
    issues = list(payload.get("issues") or [])

    issue = {
        "issue_type": issue_type,
        "code": code,
        "title": title,
        "message": message,
        "severity": severity,
    }

    if source:
        issue["source"] = source

    issues.append(issue)

    payload["issues"] = issues
    _refresh_issue_counts(payload)


def _add_content_review_issue(
    payload: Dict[str, Any],
    *,
    severity: str,
    message: str,
) -> None:
    _add_issue(
        payload,
        issue_type="content_review",
        code="-",
        title="내용 적절성 검토",
        message=message,
        severity=severity,
    )


def _missing_standard_items(section: Dict[str, Any]) -> List[str]:
    """
    standard_structure에 열거된 필수 구성요소가 content에 포함되어 있는지 확인한다.
    """
    content = get_section_content(section)
    standard_items = get_standard_structure(section)

    missing = [
        item
        for item in standard_items
        if item and item not in content
    ]

    return missing


def apply_key_value_criteria_to_payload(
    payload: Dict[str, Any],
    parsed_sections: Any,
) -> Dict[str, Any]:
    """
    key-value 평가셋의 standard_structure를 Rule 검수에 추가 반영한다.

    - 필수 구성요소가 일부 누락되면 info 이슈로 추가한다.
    - 기존 Rule이 PASS라도 필수 구성요소 누락이 있으면 WARN으로 보정한다.
    """
    if not isinstance(parsed_sections, list):
        return payload

    added_warning = False

    for section in parsed_sections:
        if not isinstance(section, dict):
            continue

        missing = _missing_standard_items(section)

        if not missing:
            continue

        section_id = str(section.get("section_id") or "-")
        section_title = str(section.get("section_title") or "-")

        _add_issue(
            payload,
            issue_type="standard_structure_missing",
            code=section_id,
            title=section_title,
            message=(
                "필수 구성요소가 일부 누락되었습니다. "
                + ", ".join(missing)
            ),
            severity="info",
        )

        added_warning = True

    if added_warning and payload.get("review_status") == "PASS":
        payload["review_status"] = "WARN"
        payload["passed"] = False
        payload["can_auto_proceed"] = True
        payload["decision_stage"] = "key_value_criteria_warn"

    return payload


def apply_fewshot_to_payload(
    payload: Dict[str, Any],
    fewshot_report: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Few-shot 결과를 diagnostics에 추가하고, 부족 항목은 참고 추천으로 노출한다.

    Few-shot은 최종 판정, 자동 진행 여부, 최상위 issues를 변경하지 않는다.
    """
    fewshot_status = fewshot_report.get("review_status")
    section_reviews = fewshot_report.get("fewshot_section_reviews") or []

    payload["diagnostics"]["fewshot"] = {
        "engine": OPTIONAL_ENGINE,
        "executed": True,
        "review_status": fewshot_status,
        "average_content_similarity": fewshot_report.get("average_content_similarity"),
        "issue_count": fewshot_report.get("issue_count"),
        "issues": fewshot_report.get("issues") or [],
        "section_reviews": section_reviews,
    }

    payload["recommendations"] = (
        list(payload.get("recommendations") or [])
        + build_fewshot_recommendations(fewshot_report)
    )

    payload["selection_reason"] = (
        "Rule 기반 검수를 1차 주요 판단으로 사용하고, "
        "Few-shot은 소제목별 보완 추천에 참고 진단으로 사용합니다."
    )

    return payload


def review_with_rule_first(
    *,
    original_text: str,
    parsed_sections: Any,
    document_type: str,
    include_fewshot: bool = True,
) -> Dict[str, Any]:
    """
    최종 검수 흐름.

    1. qa_agent Rule 검수
    2. section_content_mismatch 중 경계값 이슈 완화
    3. key-value 평가셋의 standard_structure 보조 검수
    4. Rule이 FAIL이면 Few-shot 생략
    5. Rule이 PASS/WARN이면 Few-shot 보조 진단 실행
    6. 최종 PASS/WARN/FAIL 반환
    """
    rule_report = run_rule_review(
        original_text=original_text,
        parsed_sections=parsed_sections,
        document_type=document_type,
    )

    payload = build_payload_from_rule(rule_report)

    if include_fewshot:
        fewshot_report = run_fewshot_review(
            parsed_sections=parsed_sections,
            document_type=document_type,
        )

        payload = apply_fewshot_to_payload(
            payload=payload,
            fewshot_report=fewshot_report,
        )

        payload["decision_stage"] = "rule_checked_fewshot_referenced"

        if payload.get("recommendations"):
            payload["selection_reason"] = (
                "qa_agent Rule 검수를 최종 판정 기준으로 사용하고, "
                "Few-shot은 섹션별 보완 추천으로만 제공합니다."
            )
        else:
            payload["selection_reason"] = (
                "qa_agent Rule 검수를 최종 판정 기준으로 사용하고, "
                "Few-shot은 섹션별 참고 진단으로만 제공합니다."
            )

    else:
        payload["decision_stage"] = "rule_only"
        payload["diagnostics"]["fewshot"] = {
            "engine": OPTIONAL_ENGINE,
            "executed": False,
            "reason": "include_fewshot=False",
        }

    return payload
