"""
Workit - Eval set 구축용 후보 pool 생성 스크립트
파일명: rag/yoonha_build_eval_candidates.py

목적:
  gold_standard 평가셋을 새로 만들 때, "정답이 뭔지 사람이 직접 고르기 위한"
  후보 청크 목록을 뽑아주는 최소 버전 RAG. yoonha_law_rag.py의 JoRAG 파이프라인을
  그대로 재사용하되, 재현율(recall)을 최우선으로 잡은 구성.

  최종 gold_standard 포맷을 직접 만드는 게 아니라, "후보 + 빈 정답 슬롯"만
  만들어준다 — 사람이 gold_chunk_ids를 채워넣는 게 이 스크립트의 다음 단계.

핵심 설계 결정:
  - JoRAG만 씀 (HoRAG/xref 확장은 여기서 안 함 — 최종 출력 단위가 조라서
    gold_standard도 조 단위 기준이면 JoRAG 후보만으로 충분함)
  - alpha, reranker1은 안 씀 (BEST_JO_CONFIG와 동일한 최소 구성 사상)
  - reranker2는 선택(--with-reranker2):
      끄면 → RRF 점수로만 정렬. 빠르고, reranker 모델 로드도 안 함.
             query가 많을 때(수백 개 조항) 1차로 넓게 후보 뽑을 때 적합.
      켜면 → reranker2로 한 번 더 정렬. 사람이 볼 때 위쪽 후보들이 정답에
             가깝게 배치돼서 라벨링이 빨라짐 (BEST_JO_CONFIG와 동일 사상).
    어느 쪽이든 fetch_k/top_k를 넉넉히 잡아서 정답이 후보 밖으로 밀려나는
    상황을 줄이는 게 목적. 사람이 최종 판단하므로 "정렬 순서"는 참고용일 뿐.

사용법:
  1) queries.json 준비 — 아래 두 형식 중 하나:

     형식 A) 조항 텍스트를 이미 알고 있는 경우
     [
       {"query_id": "제5조", "query_text": "..."},
       {"query_id": "제6조제1항", "query_text": "..."}
     ]

     형식 B) 계약서 원문만 있는 경우 (chunk_contract()로 자동 조 단위 분리)
     {"contract_text": "제1조(목적) ... 제2조(정의) ..."}

  2) python yoonha_build_eval_candidates.py \
         --queries queries.json --out candidates.json \
         --top-k 15 --fetch-k 50 [--with-reranker2]

  3) candidates.json을 열어서 query별 candidates 중 실제 정답 chunk_id를
     "gold_chunk_ids"에 채워넣기. 필요하면 notes에 판단 근거도 남기기.
     이렇게 채운 파일이 gold_standard_v4 같은 다음 버전의 원재료가 됨.

주의:
  - Qdrant 서버(law_kb_jo_fixedid 컬렉션)와 laws_ref.json이 실제 경로에
    준비돼 있어야 동작함 (yoonha_law_rag.py와 동일한 전제).
  - 이 스크립트는 "평가 실행"이 아니라 "평가셋 재료 수집" 용도임에 유의.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yoonha_law_rag import (
    QdrantClient,
    QDRANT_HOST,
    QDRANT_PORT,
    DEFAULT_ALPHA,
    load_embed_model,
    load_laws_ref,
    load_rerankers,
    chunk_contract,
    search_jo,
)


def load_resources(use_reranker2: bool, device: str = "cpu"):
    """
    client/model/laws_ref는 항상 로드. reranker2는 --with-reranker2일 때만
    로드해서, 빠르게 대량 후보만 뽑고 싶을 때는 모델 로딩 비용을 아낀다.
    """
    client   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    model    = load_embed_model()
    laws_ref = load_laws_ref()

    reranker2 = None
    if use_reranker2:
        _, reranker2 = load_rerankers(
            device=device, load_reranker1=False, load_reranker2=True
        )

    return client, model, laws_ref, reranker2


def load_queries(path: str) -> list[dict]:
    """형식 A(query 리스트) / 형식 B(계약서 원문) 둘 다 지원."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(data, list) and data and "query_text" in data[0]:
        return data

    if isinstance(data, dict) and "contract_text" in data:
        clauses = chunk_contract(data["contract_text"])
        return [
            {"query_id": c["clause_number"], "query_text": c["clause_text"]}
            for c in clauses
        ]

    raise ValueError(
        "queries 파일 형식을 인식할 수 없음. "
        '[{"query_id": ..., "query_text": ...}, ...] 또는 '
        '{"contract_text": "..."} 형태여야 함.'
    )


def build_candidate_pool(
    queries  : list[dict],
    client,
    model,
    laws_ref : dict,
    reranker2=None,
    top_k    : int = 15,
    fetch_k  : int = 50,
) -> list[dict]:
    """
    query별로 JoRAG 후보를 top_k개씩 뽑아 반환.
    gold_chunk_ids는 빈 리스트로 초기화 — 사람이 직접 채워넣는 자리.
    """
    use_reranker2 = reranker2 is not None
    pool: list[dict] = []

    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {q['query_id']} 후보 수집 중...", end="\r")

        law_refs = search_jo(
            q["query_text"], client, model, laws_ref,
            reranker1=None, reranker2=reranker2,
            use_reranker1=False, use_reranker2=use_reranker2,
            top_k=top_k, alpha=DEFAULT_ALPHA,
            fetch_k=fetch_k, rerank2_k=top_k,  # rerank2_k=top_k로 잘려나가지 않게
        )

        pool.append({
            "query_id"      : q["query_id"],
            "query_text"    : q["query_text"],
            "candidates"    : [
                {
                    "rank"      : rank,
                    "chunk_id"  : r.chunk_id,
                    "article"   : r.article,
                    "law_name"  : r.law_name,
                    "chunk_text": r.chunk_text,
                    "score"     : r.score,
                }
                for rank, r in enumerate(law_refs, 1)
            ],
            "gold_chunk_ids": [],   # <- 여기 사람이 채워넣기
            "notes"         : "",
        })

    print(f"\n완료: {len(pool)}개 query 처리")
    return pool


def main():
    parser = argparse.ArgumentParser(description="Eval set용 RAG 후보 pool 생성")
    parser.add_argument("--queries", required=True, help="queries json 경로")
    parser.add_argument("--out", required=True, help="후보 pool 저장 경로")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--fetch-k", type=int, default=50)
    parser.add_argument(
        "--with-reranker2", action="store_true",
        help="reranker2로 후보를 미리 정렬 (사람이 보기 편함, 대신 느려짐)"
    )
    args = parser.parse_args()

    client, model, laws_ref, reranker2 = load_resources(args.with_reranker2)
    queries = load_queries(args.queries)
    pool = build_candidate_pool(
        queries, client, model, laws_ref, reranker2,
        top_k=args.top_k, fetch_k=args.fetch_k,
    )

    Path(args.out).write_text(
        json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()