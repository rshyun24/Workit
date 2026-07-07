"""
Workit - RAG 파라미터 sweep (alpha × reranking on/off, 캐싱 적용)
파일명: rag/yoonha_rag_eval.py

목적:
  JoRAG / HoRAG 각각에 대해 alpha(dense/sparse 비중)와 reranker1/reranker2
  사용 여부를 grid로 돌려서, 실버 스탠다드 평가셋 기준 Recall@1/5/10, MRR이
  가장 좋은 조합을 찾는다. (예: 기존에 검증된 "HoRAG alpha=0.6, K=10,
  Reranking OFF, MRR=0.9333"도 이런 sweep으로 나온 결과 — 이번엔 chunk_id
  체계가 fixedid로 바뀌고 ho가 목/세목까지 포함하게 됐으니 재검증 필요)

캐싱 (이번에 추가된 부분):
  - yoonha_law_rag.SweepCache 인스턴스를 sweep 시작 전에 딱 한 번 만들어서
    모든 조합(variant × alpha × rerank on/off)에 걸쳐 재사용한다.
  - alpha는 RRF 가중치 합산에만 쓰이므로, 임베딩과 Qdrant raw 검색 결과는
    (collection, query_text)당 한 번만 계산되고 alpha 5개 전체에서 재사용된다.
  - reranker 점수는 (reranker_name, query_text, chunk_id) 단위로 memoize되므로,
    같은 chunk가 여러 alpha/조합에서 다시 리랭킹 후보로 뽑혀도 한 번만 계산된다.
  - parent fetch / cross-ref 확장도 chunk_id -> payload 조회라 쿼리 간에도
    캐시가 재사용된다 (여러 조항이 같은 법 조항을 참조하는 경우가 많음).
  - 루프 순서(변수 조합이 바깥, 쿼리가 안쪽)는 그대로 둬도 된다 — 캐시가
    "먼저 계산된 값을 나중에 재사용"하는 방식이라 순서에 의존하지 않는다.

입력:
  - eval_set.json : [{"query_id", "query", "relevant_docs_jo": [...],
    "relevant_docs_ho": [...]}]
    (variant별로 정답 필드가 다름 — 아래 GT_FIELD_BY_VARIANT 참고)

출력:
  - eval_results.csv : 모든 조합의 Recall@1/5/10, MRR, 평균 소요시간
  - 콘솔에 최고 조합 + 캐시 히트 통계 출력

사용법:
    pip install qdrant-client FlagEmbedding transformers torch pandas
    python rag/yoonha_rag_eval.py
"""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient

from yoonha_law_rag import (
    load_embed_model,
    load_laws_ref,
    load_rerankers,
    search_jo,
    search_ho,
    SweepCache,
    QDRANT_HOST,
    QDRANT_PORT,
    DEFAULT_FETCH_K,
    DEFAULT_RERANK1_K,
    DEFAULT_RERANK2_K,
)

# Colab 등 로컬 서버(localhost:6333)에 못 붙는 환경에서는
# QDRANT_LOCAL_PATH 환경변수를 잡아주면 파일 기반 로컬 Qdrant를 쓴다
# (yoonha_colab_upsert.py가 만들어둔 QDRANT_LOCAL_PATH와 같은 경로를 넣으면 됨).
# GPU 리랭커를 쓰려면 RERANKER_DEVICE=cuda로 잡아준다 (기본값 cpu).
QDRANT_LOCAL_PATH_ENV = os.environ.get("QDRANT_LOCAL_PATH")
RERANKER_DEVICE = os.environ.get("RERANKER_DEVICE", "cpu")


def get_qdrant_client() -> QdrantClient:
    if QDRANT_LOCAL_PATH_ENV:
        print(f"🗄️  로컬(파일 기반) Qdrant 사용: {QDRANT_LOCAL_PATH_ENV}")
        return QdrantClient(path=QDRANT_LOCAL_PATH_ENV)
    print(f"🗄️  서버 Qdrant 사용: {QDRANT_HOST}:{QDRANT_PORT}")
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

_THIS_DIR = Path(__file__).resolve().parent
GOLD_STANDARD_PATH = _THIS_DIR.parent / "rag" / "evaluation" / "eval_set.json"
RESULTS_CSV = _THIS_DIR / "eval_results.csv"

# sweep 그리드 — 필요하면 여기만 수정
ALPHA_GRID = [0.3, 0.5, 0.6, 0.7, 1.0]           # 1.0 = dense only
RERANK_GRID = [
    # (use_reranker1, use_reranker2)
    (False, False),   # reranking 전체 OFF
    (True,  False),   # 1단계만
    (False, True),    # 2단계만
    (True,  True),    # 둘 다 ON
]
VARIANTS = ["jo", "ho"]  # JoRAG / HoRAG 둘 다 sweep

# gold_standard_v3.json은 variant별로 정답 필드가 다르다.
#   - JoRAG는 항상 조 단위로 직접 검색하므로 relevant_docs_jo와 비교
#   - HoRAG는 chunk_id를 호/목/세목 단위 그대로 반환하므로(parent fetch는
#     text만 교체하고 chunk_id는 안 바꿈) relevant_docs_ho와 비교
GT_FIELD_BY_VARIANT = {
    "jo": "relevant_docs_jo",
    "ho": "relevant_docs_ho",
}

# 주의: fetch_k/rerank1_k/rerank2_k를 sweep 그리드에 추가하고 싶다면
# SweepCache의 raw_search 캐시 키에 fetch_k가 이미 포함돼 있어 안전하지만,
# rerank_score 캐시는 rerank1_k/rerank2_k와 무관하게 chunk 단위로 캐시되므로
# 그 자체로는 문제 없다 (top_k 절단만 나중에 다시 적용됨).


def load_gold_standard(path: Path = GOLD_STANDARD_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_combo(
    variant       : str,
    client        : QdrantClient,
    model,
    laws_ref      : dict,
    reranker1,
    reranker2,
    use_reranker1 : bool,
    use_reranker2 : bool,
    alpha         : float,
    gold_standard : list[dict],
    cache         : SweepCache,
    top_k_eval    : int = 10,
) -> dict:
    """단일 조합(variant, alpha, reranker on/off)에 대해 Recall@1/5/10, MRR 계산."""
    recall1 = recall5 = recall10 = mrr = 0
    n = len(gold_standard)
    t0 = time.time()

    search_fn = search_jo if variant == "jo" else search_ho
    # variant(jo/ho)에 맞는 정답 필드를 미리 골라둔다.
    gt_field = GT_FIELD_BY_VARIANT[variant]

    for item in gold_standard:
        gt_docs = set(item[gt_field])

        kwargs = dict(
            clause_text=item["query"],
            client=client,
            model=model,
            laws_ref=laws_ref,
            reranker1=reranker1,
            reranker2=reranker2,
            use_reranker1=use_reranker1,
            use_reranker2=use_reranker2,
            top_k=top_k_eval,
            alpha=alpha,
            fetch_k=DEFAULT_FETCH_K,
            rerank1_k=DEFAULT_RERANK1_K,
            rerank2_k=DEFAULT_RERANK2_K,
            cache=cache,
        )
        law_refs = search_fn(**kwargs)
        ranked_ids = [r.chunk_id for r in law_refs]

        if any(d in gt_docs for d in ranked_ids[:1]):  recall1  += 1
        if any(d in gt_docs for d in ranked_ids[:5]):  recall5  += 1
        if any(d in gt_docs for d in ranked_ids[:10]): recall10 += 1
        for rank, d in enumerate(ranked_ids, 1):
            if d in gt_docs:
                mrr += 1 / rank
                break

    elapsed = time.time() - t0

    return {
        "variant":        variant,
        "alpha":          alpha,
        "use_reranker1":  use_reranker1,
        "use_reranker2":  use_reranker2,
        "Recall@1":       round(recall1  / n, 4),
        "Recall@5":       round(recall5  / n, 4),
        "Recall@10":      round(recall10 / n, 4),
        "MRR":            round(mrr      / n, 4),
        "avg_sec_per_query": round(elapsed / n, 3),
    }


def main():
    client   = get_qdrant_client()
    model    = load_embed_model()
    laws_ref = load_laws_ref()
    gold_standard = load_gold_standard()

    print(f"실버 스탠다드 {len(gold_standard)}개 로드 완료")

    # reranker는 무거우니 한 번만 로드 (grid 안에서 껐다 켰다는 플래그로만 토글)
    reranker1, reranker2 = load_rerankers(device=RERANKER_DEVICE)

    # sweep 전체(40조합)에서 재사용할 캐시 — combo/쿼리 루프 밖에서 딱 한 번만 생성
    cache = SweepCache()

    all_results = []
    combos = list(itertools.product(VARIANTS, ALPHA_GRID, RERANK_GRID))
    print(f"총 {len(combos)}개 조합 sweep 시작...\n")

    t_sweep_start = time.time()

    for i, (variant, alpha, (use_r1, use_r2)) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] variant={variant} alpha={alpha} "
              f"reranker1={use_r1} reranker2={use_r2} ...")

        result = evaluate_combo(
            variant, client, model, laws_ref, reranker1, reranker2,
            use_r1, use_r2, alpha, gold_standard, cache,
        )
        all_results.append(result)
        print(f"    -> Recall@10={result['Recall@10']} MRR={result['MRR']} "
              f"({result['avg_sec_per_query']}초/쿼리) | 캐시: {cache.stats()}\n")

    t_sweep_total = time.time() - t_sweep_start

    df = pd.DataFrame(all_results)
    df = df.sort_values(["variant", "MRR"], ascending=[True, False])
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n✅ 전체 결과 저장: {RESULTS_CSV}")
    print(f"⏱️  전체 sweep 소요 시간: {t_sweep_total:.1f}초 "
          f"({t_sweep_total/60:.1f}분)")
    print(f"📦 최종 캐시 통계: {cache.stats()}")

    print("\n=== variant별 최고 조합 (MRR 기준) ===")
    for variant in VARIANTS:
        best = df[df["variant"] == variant].iloc[0]
        print(f"[{variant}] alpha={best['alpha']}, reranker1={best['use_reranker1']}, "
              f"reranker2={best['use_reranker2']} "
              f"-> Recall@10={best['Recall@10']}, MRR={best['MRR']}")


if __name__ == "__main__":
    main()