from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fewshot_agent.reviewer import review_fewshot
from qa_agent.engine import review_section_mapping


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def print_rule_report(report: dict) -> None:
    print()
    print("========== 소제목 매핑 QA 검수 ==========")
    print(f"[문서 유형]        {report['document_type']}")
    print(f"[최종 판정]        {report['review_status']}")
    print(
        "[자동 진행 가능]   "
        f"{'예' if report['can_auto_proceed'] else '아니오 (반려 코멘트 확인 필요)'}"
    )
    print(f"[전체 유사도]      {report['content_similarity']}")
    print(f"[기대 소제목 수]   {report['expected_section_count']}")
    print(f"[매칭된 소제목 수] {report['matched_section_count']}")

    issues = report.get("issues") or []
    print()
    print(f"[발견된 이슈: {len(issues)}건]")
    if not issues:
        print("- 없음")
    for issue in issues:
        code = issue.get("code") or "-"
        title = issue.get("title") or "-"
        print(f"- [{issue['issue_type']}] {code} / {title}: {issue['message']}")
        if issue.get("sample"):
            print(f"    예시: {issue['sample']}")


def print_fewshot_report(report: dict) -> None:
    print()
    print("========== Few-shot 참고 진단 ==========")
    print(f"[문서 유형]        {report['document_type']}")
    print(f"[진단 판정]        {report['review_status']}")
    print(
        "[자동 진행 가능]   "
        f"{'예' if report['can_auto_proceed'] else '아니오 (참고 진단 확인 필요)'}"
    )
    print(f"[검토 섹션 수]     {report['section_count']}")
    print(f"[평균 유사도]      {report['average_content_similarity']}")
    print(f"[전체 이슈 수]     {report['issue_count']}")
    print(f"[치명 이슈 수]     {report['blocking_issue_count']}")
    print(f"[참고 이슈 수]     {report['info_issue_count']}")

    issues = report.get("issues") or []
    print()
    print(f"[Few-shot 이슈: {len(issues)}건]")
    if not issues:
        print("- 없음")
    for issue in issues:
        title = issue.get("title") or "-"
        print(f"- [{issue['issue_type']}] {title}: {issue['message']}")
        if issue.get("missing_items"):
            print(f"    부족 항목: {issue['missing_items']}")
        if "similarity" in issue:
            print(f"    유사도: {issue['similarity']}")

    section_reviews = report.get("fewshot_section_reviews") or []
    if section_reviews:
        print()
        print("[Few-shot 섹션별 진단]")
        for review in section_reviews:
            print(
                f"- {review['title']} -> {review['status']} "
                f"(keyword={review['keyword_score']}, "
                f"content_similarity={review['content_similarity']}, "
                f"title_similarity={review['title_similarity']})"
            )
            if review.get("missing_items"):
                print(f"    부족 항목: {review['missing_items']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="기존 qa_agent 검수 결과와 few-shot 참고 진단을 함께 실행합니다."
    )
    parser.add_argument("--original", required=True, help="원본 텍스트 파일 경로")
    parser.add_argument("--parsed", required=True, help="파싱 결과 JSON 파일 경로")
    parser.add_argument("--doc-type", required=True, choices=["rfp", "pep", "rpt"])
    parser.add_argument(
        "--show-fewshot",
        action="store_true",
        help="콘솔에도 few-shot 참고 진단을 함께 출력합니다.",
    )
    args = parser.parse_args()

    original_text = load_text(args.original)
    parsed_sections = load_json(args.parsed)

    rule_report = review_section_mapping(
        original_text=original_text,
        parsed_sections=parsed_sections,
        document_type=args.doc_type,
    )
    fewshot_report = review_fewshot(
        parsed_sections=parsed_sections,
        doc_type=args.doc_type,
    )

    combined_report = {
        "document_type": args.doc_type,
        "final_decision_source": "qa_agent",
        "final_review_status": rule_report["review_status"],
        "final_can_auto_proceed": rule_report["can_auto_proceed"],
        "qa_agent": rule_report,
        "fewshot_reference": fewshot_report,
    }

    print_rule_report(rule_report)

    if args.show_fewshot:
        print_fewshot_report(fewshot_report)

    out_path = Path(args.parsed).with_suffix(".qa_fewshot.review.json")
    out_path.write_text(
        json.dumps(combined_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"[리포트 저장] {out_path}")


if __name__ == "__main__":
    main()
