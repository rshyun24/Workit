"""
yoonha_deliver_evaluation.py
-----------------------------
Workit — 산출물 양식 RAG 평가
파싱된 실제 문서 → RAG 검색 → 섹션 매핑 정확도 측정

평가 지표:
  - 직접 매핑률  : section_title 완전/부분 일치로 매핑된 비율
  - fallback률   : 유사도 1위로 fallback된 비율
  - 섹션 커버리지: 실제 문서 섹션 중 양식 KB에서 찾은 비율

실행:
    py rag/evaluation/yoonha_deliver_evaluation.py
    py rag/evaluation/yoonha_deliver_evaluation.py --doc 사업수행계획서
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

# 경로 설정
_THIS_DIR    = Path(__file__).resolve().parent          # evaluation/
_RAG_DIR     = _THIS_DIR.parent                         # rag/
_PROJECT_ROOT = _RAG_DIR.parent                         # Workit/
_DATA_DIR    = _PROJECT_ROOT / "data"

sys.path.insert(0, str(_DATA_DIR))
sys.path.insert(0, str(_RAG_DIR))

from yoonha_deliver_parser import load_parsed_json
from yoonha_deliver_chunking import QDRANT_HOST, QDRANT_PORT
from yoonha_deliver_rag import WorkitRetriever


# ──────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────
PARSED_DIR = _DATA_DIR / "uploads" / "parsed"

# 파일명 키워드 → doc_type 매핑
FILENAME_TO_DOC_TYPE = {
    "사업수행계획서": "사업수행계획서",
    "테스트결과보고서": "테스트결과보고서",
    "테스트설계서": "테스트설계서",
    "최종결과보고서": "최종결과보고서",
}


# ──────────────────────────────────────────
# 1. 평가 결과 데이터클래스
# ──────────────────────────────────────────
@dataclass
class SectionResult:
    section_title: str
    section_depth: int
    matched:       bool    # 직접 매핑 성공 여부
    fallback_used: bool    # fallback 사용 여부
    ref_section:   str     # 매핑된 양식 섹션명


@dataclass
class DocEvalResult:
    filename:       str
    doc_type:       str
    total_sections: int
    matched:        int
    fallback:       int
    no_result:      int
    section_results: list[SectionResult]

    @property
    def match_rate(self) -> float:
        return self.matched / self.total_sections if self.total_sections else 0.0

    @property
    def fallback_rate(self) -> float:
        return self.fallback / self.total_sections if self.total_sections else 0.0

    @property
    def coverage(self) -> float:
        return (self.matched + self.fallback) / self.total_sections if self.total_sections else 0.0


# ──────────────────────────────────────────
# 2. 단일 문서 평가
# ──────────────────────────────────────────
def evaluate_document(
    json_path: Path,
    doc_type: str,
    retriever: WorkitRetriever,
) -> DocEvalResult:
    sections = load_parsed_json(json_path)
    user_sections = [
        {"text": s.text, "section_title": s.section_title}
        for s in sections
    ]

    print(f"  검색 중... ({len(sections)}개 섹션)")
    retrieval_results = retriever.retrieve_document(user_sections, doc_type)

    section_results = []
    matched = fallback = no_result = 0

    for sec, ctx in zip(sections, retrieval_results):
        if ctx.reference_chunk is None:
            no_result += 1
            section_results.append(SectionResult(
                section_title=sec.section_title,
                section_depth=sec.section_depth,
                matched=False,
                fallback_used=False,
                ref_section="[검색 결과 없음]",
            ))
        elif ctx.fallback_used:
            fallback += 1
            section_results.append(SectionResult(
                section_title=sec.section_title,
                section_depth=sec.section_depth,
                matched=False,
                fallback_used=True,
                ref_section=ctx.reference_chunk.section_title,
            ))
        else:
            matched += 1
            section_results.append(SectionResult(
                section_title=sec.section_title,
                section_depth=sec.section_depth,
                matched=True,
                fallback_used=False,
                ref_section=ctx.reference_chunk.section_title,
            ))

    return DocEvalResult(
        filename=json_path.name,
        doc_type=doc_type,
        total_sections=len(sections),
        matched=matched,
        fallback=fallback,
        no_result=no_result,
        section_results=section_results,
    )


# ──────────────────────────────────────────
# 3. 리포트 출력
# ──────────────────────────────────────────
def print_report(results: list[DocEvalResult]) -> None:
    print("\n" + "=" * 65)
    print("Workit 산출물 양식 RAG 평가 결과")
    print("=" * 65)

    for r in results:
        print(f"\n📄 {r.filename} (doc_type={r.doc_type})")
        print(f"   총 섹션: {r.total_sections}개")
        print(f"   직접 매핑: {r.matched}개 ({r.match_rate:.1%})")
        print(f"   fallback:  {r.fallback}개 ({r.fallback_rate:.1%})")
        print(f"   미검색:    {r.no_result}개")
        print(f"   커버리지:  {r.coverage:.1%}")

        print(f"\n   섹션별 결과:")
        for s in r.section_results:
            if s.matched:
                icon = "✅"
            elif s.fallback_used:
                icon = "🟡"
            else:
                icon = "❌"
            print(f"   {icon} [{s.section_title}]")
            print(f"       → {s.ref_section}")

    # 전체 요약
    total_secs    = sum(r.total_sections for r in results)
    total_matched = sum(r.matched for r in results)
    total_fallback = sum(r.fallback for r in results)
    total_no      = sum(r.no_result for r in results)

    print("\n" + "─" * 65)
    print(f"{'전체 요약'}")
    print("─" * 65)
    print(f"{'총 섹션':<20} {total_secs:>5}개")
    print(f"{'직접 매핑':<20} {total_matched:>5}개  ({total_matched/total_secs:.1%})")
    print(f"{'fallback':<20} {total_fallback:>5}개  ({total_fallback/total_secs:.1%})")
    print(f"{'미검색':<20} {total_no:>5}개  ({total_no/total_secs:.1%})")
    print(f"{'커버리지':<20} {(total_matched+total_fallback)/total_secs:>8.1%}")
    print("─" * 65)


# ──────────────────────────────────────────
# 4. JSON 저장
# ──────────────────────────────────────────
def save_results(results: list[DocEvalResult], path: str = "deliver_eval_results.json") -> None:
    output = [
        {
            "filename":       r.filename,
            "doc_type":       r.doc_type,
            "total_sections": r.total_sections,
            "matched":        r.matched,
            "fallback":       r.fallback,
            "no_result":      r.no_result,
            "match_rate":     round(r.match_rate, 4),
            "fallback_rate":  round(r.fallback_rate, 4),
            "coverage":       round(r.coverage, 4),
            "sections": [
                {
                    "section_title": s.section_title,
                    "section_depth": s.section_depth,
                    "matched":       s.matched,
                    "fallback_used": s.fallback_used,
                    "ref_section":   s.ref_section,
                }
                for s in r.section_results
            ],
        }
        for r in results
    ]
    out_path = _THIS_DIR / path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 상세 결과 저장: {out_path}")


# ──────────────────────────────────────────
# 5. 메인
# ──────────────────────────────────────────
def main(doc_filter: str | None = None) -> None:
    print("=" * 65)
    print("Workit 산출물 양식 RAG 평가")
    print("=" * 65)

    # parsed 폴더 확인
    if not PARSED_DIR.exists():
        print(f"[eval] parsed 폴더가 없습니다: {PARSED_DIR}")
        print("  → py data/yoonha_upload_parser.py 를 먼저 실행하세요.")
        return

    json_files = sorted(PARSED_DIR.glob("*.json"))
    if not json_files:
        print(f"[eval] 파싱된 파일이 없습니다: {PARSED_DIR}")
        return

    # Qdrant 연결
    client    = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    retriever = WorkitRetriever(client)

    count = client.count(collection_name="workit_forms")
    print(f"📚 workit_forms KB: {count.count}개 청크")
    print(f"📁 parsed 폴더: {len(json_files)}개 파일\n")

    results = []
    for json_path in json_files:
        # doc_type 추론
        doc_type = None
        for keyword, dtype in FILENAME_TO_DOC_TYPE.items():
            if keyword in json_path.stem:
                doc_type = dtype
                break

        if doc_type is None:
            print(f"⚠️  doc_type 추론 실패, 스킵: {json_path.name}")
            continue

        # doc_filter 적용 (특정 doc_type만 평가)
        if doc_filter and doc_filter not in doc_type:
            continue

        print(f"📄 평가 중: {json_path.name} (doc_type={doc_type})")
        result = evaluate_document(json_path, doc_type, retriever)
        results.append(result)

    if not results:
        print("평가할 파일이 없습니다.")
        return

    print_report(results)
    save_results(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Workit 산출물 양식 RAG 평가")
    ap.add_argument("--doc", default=None, help="특정 doc_type만 평가 (예: 사업수행계획서)")
    args = ap.parse_args()
    main(doc_filter=args.doc)