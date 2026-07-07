"""
Workit - 계층형(Cascaded) RAG 파라미터 sweep
파일명: rag/yoonha_rag_eval_cascaded.py

목적:
  yoonha_rag_eval.py(병렬 JoRAG/HoRAG 독립 검색)와는 별개로, 계층형
  (1단계 JoRAG top-k → 2단계 그 조로 제한된 HoRAG) 구조의 최적 조합을 찾는다.

  병렬 버전과 다른 점:
    - variant 축이 없다 (jo 단독은 계층형 개념이 성립하지 않음 — 계층형은
      항상 "jo top-k → ho 제한 검색" 전체 파이프라인 단위로만 평가한다).
    - jo_top_k가 새 sweep 축으로 추가된다. 1단계 recall이 2단계 recall의
      상한선이 되므로, jo_top_k가 너무 작으면 아무리 alpha/reranker를
      잘 조합해도 최종 성능이 낮게 나온다.
    - 각 조합마다 stage1_recall(1단계에서 정답 조가 top-k 안에 살아남은
      비율)을 별도로 기록한다. 최종 MRR이 낮게 나왔을 때 "1단계 필터링
      자체가 문제였는지" "2단계 검색/리랭킹이 문제였는지"를 구분하기 위함.

  주의 — grid 크기:
    alpha 5개 × rerank 조합 4개 × jo_top_k 3개 = 60조합. 병렬 버전(40조합)
    보다 많다. jo_top_k 값이 실제로 몇 개 필요할지는 상황 봐가며 줄이거나
    늘리면 된다 (예: 1차로 성능이 jo_top_k에 얼마나 민감한지부터 보고,
    민감하지 않으면 그리드를 줄여도 됨).

캐싱:
  yoonha_law_rag.SweepCache를 sweep 시작 전에 한 번만 생성해서 전체 조합에서
  재사용한다. 다만 계층형은 jo_top_k가 바뀌면 2단계 검색의 Qdrant 필터
  자체가 달라지므로, raw_search 캐시 재사용률은 병렬 버전만큼 높지 않다
  (필터 지문이 캐시 키에 포함되어 있어서 서로 다른 필터끼리 섞이진 않음 —
  yoonha_law_rag_cascaded._hybrid_search_ho_filtered 참고). 반면 임베딩
  캐시와 parent/cross-ref scroll 캐시는 jo_top_k와 무관하게 그대로
  재사용된다.

입력/출력은 yoonha_rag_eval.py와 동일한 eval_set.json,
eval_results_cascaded.csv (파일명만 다름, 기존 결과를 덮어쓰지 않음).
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
    SweepCache,
    QDRANT_HOST,
    QDRANT_PORT,
    DEFAULT_FETCH_K,
    DEFAULT_RERANK1_K,
    DEFAULT_RERANK2_K,
    derive_jo_id,
)
from yoonha_law_rag_cascaded import (
    search_cascaded,
    get_stage1_jo_candidates,
    DEFAULT_JO_TOP_K,
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
RESULTS_CSV = _THIS_DIR / "eval_results_cascaded.csv"

# sweep 그리드 — 필요하면 여기만 수정
ALPHA_GRID   = [0.3, 0.5, 0.6, 0.7, 1.0]           # 1.0 = dense only
JO_TOP_K_GRID = [10, 20, 30, 40, 50]                        # 1단계(조 단위)에서 남길 후보 수
RERANK_GRID = [
    # (use_reranker1, use_reranker2)
    (False, False),   # reranking 전체 OFF
    (True,  False),   # 1단계만
    (False, True),    # 2단계만
    (True,  True),    # 둘 다 ON
]

# 계층형은 항상 relevant_docs_ho와 비교 (최종 출력은 조 텍스트로 통일되지만
# chunk_id 자체는 ho-level 그대로 반환 — 병렬 버전의 HoRAG와 동일한 규칙)
GT_FIELD = "relevant_docs_ho"


def load_gold_standard(path: Path = GOLD_STANDARD_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _get_ho_parent_id(chunk_id: str) -> str:
    """
    ho-level chunk_id에서 그 조에 해당하는 jo_id를 역산한다 (stage1_recall 진단용).

    이전 버전은 payload의 parent_chunk_id 필드를 Qdrant scroll로 조회했었는데,
    실제 데이터 검증 결과 그 필드는 JO 컬렉션이 아니라 HO 컬렉션 자기 자신을
    가리키고 있어서 JO id와 절대 일치하지 않았다 (그래서 stage1_recall이 항상
    0.0으로 나왔음). derive_jo_id()는 chunk_id 문자열 자체에서 직접 역산하므로
    Qdrant 조회 자체가 필요 없다 — 훨씬 빠르고 정확하다.
    """
    return derive_jo_id(chunk_id)


def evaluate_combo_cascaded(
    client        : QdrantClient,
    model,
    laws_ref      : dict,
    reranker1,
    reranker2,
    use_reranker1 : bool,
    use_reranker2 : bool,
    alpha         : float,
    jo_top_k      : int,
    gold_standard : list[dict],
    cache         : SweepCache,
    top_k_eval    : int = 10,
) -> dict:
    """단일 조합(alpha, jo_top_k, reranker on/off)에 대해 Recall@1/5/10, MRR, stage1_recall 계산."""
    recall1 = recall5 = recall10 = mrr = 0
    stage1_hits = stage1_total = 0
    n = len(gold_standard)
    t0 = time.time()

    for item in gold_standard:
        gt_docs = set(item[GT_FIELD])
        query   = item["query"]

        # --- stage1 진단: 정답 ho chunk의 parent 조가 1단계 top-k 안에 살아남았는지 ---
        stage1_candidates = get_stage1_jo_candidates(
            query, client, model, alpha=alpha, fetch_k=DEFAULT_FETCH_K,
            jo_top_k=jo_top_k, cache=cache,
        )
        gt_parents = set()
        for d in gt_docs:
            pid = _get_ho_parent_id(d)
            if pid:
                gt_parents.add(pid)
        if gt_parents:
            stage1_total += 1
            if gt_parents & set(stage1_candidates):
                stage1_hits += 1

        # --- 실제 최종 검색 (1단계+2단계 전체) ---
        law_refs = search_cascaded(
            clause_text=query,
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
            jo_top_k=jo_top_k,
            cache=cache,
        )
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
        "alpha":            alpha,
        "jo_top_k":         jo_top_k,
        "use_reranker1":    use_reranker1,
        "use_reranker2":    use_reranker2,
        "Recall@1":         round(recall1  / n, 4),
        "Recall@5":         round(recall5  / n, 4),
        "Recall@10":        round(recall10 / n, 4),
        "MRR":              round(mrr      / n, 4),
        "stage1_recall":    round(stage1_hits / stage1_total, 4) if stage1_total else None,
        "avg_sec_per_query": round(elapsed / n, 3),
    }


def main():
    client   = get_qdrant_client()
    model    = load_embed_model()
    laws_ref = load_laws_ref()
    gold_standard = load_gold_standard()

    print(f"실버 스탠다드 {len(gold_standard)}개 로드 완료")

    reranker1, reranker2 = load_rerankers(device=RERANKER_DEVICE)

    # sweep 전체에서 재사용할 캐시 — combo/쿼리 루프 밖에서 딱 한 번만 생성.
    # yoonha_rag_eval.py(병렬 버전)의 캐시와는 별개 인스턴스 — 두 sweep을
    # 동시에 돌리더라도 서로 캐시가 섞이지 않는다.
    cache = SweepCache()

    all_results = []
    combos = list(itertools.product(ALPHA_GRID, JO_TOP_K_GRID, RERANK_GRID))
    print(f"총 {len(combos)}개 조합(계층형) sweep 시작...\n")

    t_sweep_start = time.time()

    for i, (alpha, jo_top_k, (use_r1, use_r2)) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] alpha={alpha} jo_top_k={jo_top_k} "
              f"reranker1={use_r1} reranker2={use_r2} ...")

        result = evaluate_combo_cascaded(
            client, model, laws_ref, reranker1, reranker2,
            use_r1, use_r2, alpha, jo_top_k, gold_standard, cache,
        )
        all_results.append(result)
        print(f"    -> Recall@10={result['Recall@10']} MRR={result['MRR']} "
              f"stage1_recall={result['stage1_recall']} "
              f"({result['avg_sec_per_query']}초/쿼리) | 캐시: {cache.stats()}\n")

    t_sweep_total = time.time() - t_sweep_start

    df = pd.DataFrame(all_results)
    df = df.sort_values("MRR", ascending=False)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n✅ 전체 결과 저장: {RESULTS_CSV}")
    print(f"⏱️  전체 sweep 소요 시간: {t_sweep_total:.1f}초 "
          f"({t_sweep_total/60:.1f}분)")
    print(f"📦 최종 캐시 통계: {cache.stats()}")

    print("\n=== 계층형 최고 조합 (MRR 기준) ===")
    best = df.iloc[0]
    print(f"alpha={best['alpha']}, jo_top_k={best['jo_top_k']}, "
          f"reranker1={best['use_reranker1']}, reranker2={best['use_reranker2']} "
          f"-> Recall@10={best['Recall@10']}, MRR={best['MRR']}, "
          f"stage1_recall={best['stage1_recall']}")

    print("\n※ 병렬 버전(eval_results.csv)의 HoRAG 최고 MRR과 이 결과를 나란히")
    print("   비교해서, 계층형이 실제로 이득인지 판단하시면 됩니다.")


if __name__ == "__main__":
    main()