from __future__ import annotations

import argparse
from pathlib import Path

from backend_review.io_utils import (
    load_json,
    load_text,
    write_json,
)

from backend_review.service import (
    detect_doc_type,
    review_parsed_document,
    summarize_for_backend,
)


def print_report(payload: dict) -> None:
    """
    Console output for manual review.

    The backend payload stays unchanged, but the CLI prints the same QA-style
    issue details that reviewers expect from qa_agent/run_with_fewshot.py.
    """
    summary = summarize_for_backend(payload)
    issues = summary.get("issues") or []

    print()
    print("========== 소제목 매핑 QA 검수 ==========")
    print(f"[문서 유형]        {summary['document_type']}")
    print(f"[최종 판정]        {summary['review_status']}")
    print(
        "[자동 진행 가능]   "
        f"{'예' if summary['can_auto_proceed'] else '아니오 (반려 코멘트 확인 필요)'}"
    )
    print(f"[전체 유사도]      {summary['content_similarity']}")
    print(f"[기대 소제목 수]   {summary['expected_section_count']}")
    print(f"[매칭된 소제목 수] {summary['matched_section_count']}")

    print()
    print(f"[발견된 이슈: {len(issues)}건]")

    if not issues:
        print("- 없음")
        print()
        return

    for issue in issues:
        issue_type = issue.get("issue_type") or "-"
        code = issue.get("code") or "-"
        title = issue.get("title") or "-"
        message = issue.get("message") or ""

        print(f"- [{issue_type}] {code} / {title}: {message}")

        if issue.get("sample"):
            print(f"    예시: {issue['sample']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rule + Few-shot 문서 검수"
    )

    parser.add_argument(
        "--original",
        required=True,
        help="원본 txt 파일",
    )

    parser.add_argument(
        "--parsed",
        required=True,
        help="파싱 json 파일",
    )

    parser.add_argument(
        "--doc-type",
        choices=["rfp", "pep", "rpt"],
    )

    parser.add_argument(
        "--no-fewshot",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        help="결과 저장 json",
    )

    args = parser.parse_args()

    original_text = load_text(args.original)
    parsed_sections = load_json(args.parsed)
    document_type = args.doc_type

    if not document_type:
        document_type = detect_doc_type(
            filename=f"{Path(args.original).name} {Path(args.parsed).name}",
            original_text=original_text,
            parsed_sections=parsed_sections,
        )

    if not document_type:
        raise ValueError(
            "문서 유형을 감지하지 못했습니다. "
            "--doc-type을 지정해주세요."
        )

    payload = review_parsed_document(
        original_text=original_text,
        parsed_sections=parsed_sections,
        document_type=document_type,
        include_fewshot=not args.no_fewshot,
    )

    print_report(payload)

    if args.output:
        write_json(
            args.output,
            payload,
        )

        print(f"[리포트 저장] {args.output}")


if __name__ == "__main__":
    main()
