"""
Workit - 평가셋 후보 구축 스크립트
파일명: yoonha_eval_builder.py
위치:   rag/yoonha_eval_builder.py

Human-in-the-loop 평가셋 구축 방식:
  1. (이 스크립트) JoRAG + HoRAG + HoXrefRAG 세 개 후보를 합쳐서 저장
     → 특정 RAG에 편향되지 않은 중립적 후보 생성
  2. 사람이 eval_candidates.json 열어서 relevant_ids 확정
     - "_candidates" 에서 실제 관련 있는 chunk_id만 "relevant_ids" 로 옮김
     - 각 항목의 "_reviewed" 를 true 로 변경
  3. (yoonha_law_eval_runner.py) 확정된 eval_candidates.json으로
     세 RAG 비교 평가 + alpha sweep + Top-K 민감도 + qtype 분석

실행:
    python yoonha_eval_builder.py
    python yoonha_eval_builder.py --alpha 0.7 --top_k 10
    python yoonha_eval_builder.py --output rag/eval_candidates.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from qdrant_client import QdrantClient

from yoonha_law_rag import (
    LawRef,
    RRF_ALPHA,
    TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
    load_embed_model,
    load_laws_ref,
    load_rerankers,
    search_ho,
    search_ho_xref,
    search_jo,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가용 질문 목록
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTIONS: list[dict] = [
    # ── 위반 ───────────────────────────────
    {
        "query"         : "발주기관의 대금 지급 기한이 법령상 허용 범위를 초과하는가?",
        "source_article": "계약 특수조건 제3조",
        "purpose"       : "대금 지급 기한 적법성 검토",
        "qtype"         : "violation",
    },
    {
        "query"         : "계약상대자의 손해배상 범위를 계약금액의 5%로 제한하는 조항이 적법한가?",
        "source_article": "계약 특수조건 제5조",
        "purpose"       : "손해배상 범위 제한 적법성 검토",
        "qtype"         : "violation",
    },
    {
        "query"         : "발주기관이 사업상 필요에 따라 언제든지 계약을 해지할 수 있도록 한 조항이 적법한가?",
        "source_article": "계약 특수조건 제7조",
        "purpose"       : "일방적 해지권 부여 적법성 검토",
        "qtype"         : "violation",
    },
    # ── 누락 ───────────────────────────────
    {
        "query"         : "추가 과업 요청 시 계약금액 조정 기준이 계약서에 명시되어 있는가?",
        "source_article": "계약 특수조건 제4조",
        "purpose"       : "과업 변경 시 계약금액 조정 기준 누락 검토",
        "qtype"         : "missing",
    },
    {
        "query"         : "대금 지연 지급 시 적용되는 지연이자율이 계약서에 명시되어 있는가?",
        "source_article": "계약 특수조건 제9조",
        "purpose"       : "지연이자율 명시 여부 검토",
        "qtype"         : "missing",
    },
    # ── 정상 ───────────────────────────────
    {
        "query"         : "계약서의 지체상금률 및 상한 기준이 관련 법령을 준수하는가?",
        "source_article": "계약 특수조건 제2조",
        "purpose"       : "지체상금 적법성 검토",
        "qtype"         : "normal",
    },
    {
        "query"         : "계약 종료 후에도 비밀유지 의무가 지속되도록 규정되어 있는가?",
        "source_article": "계약 특수조건 제6조",
        "purpose"       : "비밀유지 의무 범위 검토",
        "qtype"         : "normal",
    },
    {
        "query"         : "하자보수 기간 및 하자보수보증금 비율이 적정하게 설정되어 있는가?",
        "source_article": "계약 특수조건 제8조",
        "purpose"       : "하자보수 조건 적정성 검토",
        "qtype"         : "normal",
    },
    {
        "query"         : "계약 분쟁 발생 시 해결 절차가 명시되어 있는가?",
        "source_article": "계약 특수조건 제10조",
        "purpose"       : "분쟁 해결 절차 검토",
        "qtype"         : "normal",
    },
    {
        "query"         : "계약보증금 비율이 계약금액 대비 적정 수준으로 설정되어 있는가?",
        "source_article": "계약서 표지",
        "purpose"       : "계약보증금 적정성 검토",
        "qtype"         : "normal",
    },
]

DEFAULT_OUTPUT = "rag/eval_candidates.json"
CANDIDATE_K    = 10  # RAG별 검색 후보 수

RAG_CONFIGS = {
    "JoRAG"    : search_jo,
    "HoRAG"    : search_ho,
    "HoXrefRAG": search_ho_xref,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 검색 결과 → chunk_id 리스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_retrieved_ids(law_refs: list[LawRef]) -> list[str]:
    ids, seen = [], set()
    for ref in law_refs:
        cid = ref.parent_id if ref.parent_id else ref.chunk_id
        if cid not in seen:
            ids.append(cid)
            seen.add(cid)
    return ids


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 빌더 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_eval_candidates(
    output_path : str   = DEFAULT_OUTPUT,
    top_k       : int   = CANDIDATE_K,
    alpha       : float = RRF_ALPHA,
    reranker1           = None,
    reranker2           = None,
    verbose     : bool  = True,
) -> list[dict]:

    client   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    model    = load_embed_model()
    laws_ref = load_laws_ref()

    if verbose:
        print(f"\n{'='*55}")
        print(f"평가셋 후보 구축 (3개 RAG 합산)")
        print(f"  질문 수  : {len(QUESTIONS)}개")
        print(f"  RAG별 K  : top-{top_k}  (합산 최대 {top_k * 3}개)")
        print(f"  alpha    : {alpha}")
        print(f"{'='*55}")

    candidates: list[dict] = []

    for i, q in enumerate(QUESTIONS, 1):
        query = q["query"]
        if verbose:
            print(f"\n[{i:2d}/{len(QUESTIONS)}] {query[:55]}...")

        # ── 3개 RAG 각각 검색 후 후보 합산 ──────────
        merged: dict[str, list[str]] = {}   # chunk_id → 어느 RAG에서 나왔는지

        for rag_name, search_fn in RAG_CONFIGS.items():
            refs = search_fn(
                query, client, model, laws_ref,
                reranker1, reranker2, top_k, alpha,
            )
            ids = get_retrieved_ids(refs)
            for cid in ids:
                merged.setdefault(cid, []).append(rag_name)

            if verbose:
                print(f"  {rag_name:<12}: {ids}")

        # 등장 빈도 높은 순 정렬 (여러 RAG에서 공통으로 나온 것 우선)
        sorted_ids = sorted(merged.keys(), key=lambda c: -len(merged[c]))

        if verbose:
            print(f"  → 합산 후보 {len(sorted_ids)}개 (중복 제거)")

        candidates.append({
            "query"         : query,
            "source_article": q.get("source_article", ""),
            "purpose"       : q.get("purpose", ""),
            "qtype"         : q.get("qtype", "unknown"),
            # 사람이 검토 후 관련 있는 것만 남길 최종 필드 (처음엔 비워둠)
            "relevant_ids"  : [],
            # RAG 합산 후보 (검토 참고용)
            "_candidates"   : {
                cid: merged[cid] for cid in sorted_ids
            },
            "_reviewed"     : False,
        })

        time.sleep(0.1)

    # ── 저장 ─────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n{'='*55}")
        print(f"✅ 저장 완료: {output_path}")
        print(f"\n📋 다음 단계 (Human-in-the-loop):")
        print(f"   1. {output_path} 열기")
        print(f"   2. 각 질문의 '_candidates' 확인")
        print(f"      → 실제 관련 있는 chunk_id를 'relevant_ids' 리스트에 옮기기")
        print(f"      → 후보에 없지만 필요한 항목은 직접 추가")
        print(f"   3. 검토 완료한 항목은 '_reviewed': true 로 변경")
        print(f"   4. python rag/yoonha_law_eval_runner.py --dataset {output_path} --all")
        print(f"{'='*55}")

    return candidates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG 중립적 평가셋 후보 구축 (3개 RAG 합산)")
    p.add_argument("--output",  default=DEFAULT_OUTPUT,
                   help=f"출력 JSON 경로 (기본: {DEFAULT_OUTPUT})")
    p.add_argument("--alpha",   type=float, default=RRF_ALPHA,
                   help=f"dense 비중 (기본: {RRF_ALPHA})")
    p.add_argument("--top_k",   type=int,   default=CANDIDATE_K,
                   help=f"RAG별 후보 수 (기본: {CANDIDATE_K})")
    p.add_argument("--rerank",  action="store_true", help="리랭커 사용")
    p.add_argument("--device",  default="cpu",       help="리랭커 디바이스")
    p.add_argument("--quiet",   action="store_true", help="진행 출력 최소화")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    reranker1, reranker2 = None, None
    if args.rerank:
        reranker1, reranker2 = load_rerankers(device=args.device)

    build_eval_candidates(
        output_path = args.output,
        top_k       = args.top_k,
        alpha       = args.alpha,
        reranker1   = reranker1,
        reranker2   = reranker2,
        verbose     = not args.quiet,
    )


if __name__ == "__main__":
    main()