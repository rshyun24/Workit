
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fewshot_agent.reviewer import load_json, review_fewshot


def print_report(report: dict) -> None:
    print()
    print("========== Few-shot QA 검수 ==========")
    print(f"[문서 유형]        {report['document_type']}")
    print(f"[최종 판정]        {report['review_status']}")
    print(f"[자동 진행 가능]   {'예' if report['can_auto_proceed'] else '아니오 (반려 코멘트 확인 필요)'}")
    print(f"[검수 섹션 수]     {report['section_count']}")
    print(f"[전체 이슈 수]     {report['issue_count']}")
    print(f"[치명 이슈 수]     {report['blocking_issue_count']}")
    print(f"[참고 이슈 수]     {report['info_issue_count']}")
    print(f"[평균 유사도]      {report['average_content_similarity']}")

    if report["issues"]:
        print()
        print("[이슈]")
        for issue in report["issues"]:
            print(f"- [{issue['issue_type']}] {issue.get('title', '-')}: {issue['message']}")
            if "similarity" in issue:
                print(f"  유사도: {issue['similarity']}")
            if issue.get("missing_items"):
                print(f"  부족 항목: {issue['missing_items']}")

    if report["fewshot_section_reviews"]:
        print()
        print("[Few-shot 섹션별 검수]")
        for review in report["fewshot_section_reviews"]:
            print(
                f"- {review['title']} → {review['status']} "
                f"/ keyword_score={review['keyword_score']} "
                f"/ content_similarity={review['content_similarity']} "
                f"/ title_similarity={review['title_similarity']}"
            )
            if review["missing_items"]:
                print(f"  부족: {review['missing_items']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot QA 검수 실행")
    parser.add_argument("--parsed", required=True, help="파싱 결과 JSON 파일")
    parser.add_argument("--doc-type", required=True, choices=["rfp", "pep", "rpt"])
    args = parser.parse_args()

    parsed_sections = load_json(args.parsed)
    report = review_fewshot(parsed_sections, args.doc_type)

    print_report(report)

    out_path = Path(args.parsed).with_suffix(".fewshot.review.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"[리포트 저장] {out_path}")


if __name__ == "__main__":
    main()
