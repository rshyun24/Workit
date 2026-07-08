"""
Workit - min_score 임계값 잡기용 간단 체크 스크립트
파일명: rag/yoonha_score_threshold_check.py

전체 하이퍼파라미터 sweep이 아니라, 이미 확정된 파라미터(yoonha_law_rag.py의
DEFAULT_*)로 gold standard 평가셋을 딱 한 번 검색해서, 정답 chunk와 오답
chunk의 rerank_score(또는 rrf_score) 분포만 뽑아본다. 이 분포를 보고
min_score threshold를 감이 아니라 데이터 기반으로 잡기 위한 용도.

평가셋 형식 (list of dict), 각 항목:
    {"query": "...", "relevant_docs_jo": ["chunk_id1", "chunk_id2", ...]}

사용법:
    python rag/yoonha_score_threshold_check.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from yoonha_law_rag import (
    DEFAULT_ALPHA,
    DEFAULT_FETCH_K,
    DEFAULT_RERANK_K,
    DEFAULT_RRF_K,
    get_qdrant_client,
    load_embed_model,
    load_reranker,
    search_jo,
)

# 실제 경로에 맞게 수정하세요.
GOLD_STANDARD_PATH = Path(__file__).resolve().parent / "evaluation" / "eval_set.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "score_threshold_check.json"

# top_k를 넉넉히 잡아야 오답 후보도 충분히 걸려서 분포를 볼 수 있음
TOP_K_FOR_CHECK = 10


def main() -> None:
    with open(GOLD_STANDARD_PATH, encoding="utf-8") as f:
        gold_standard = json.load(f)

    print(f"평가셋 {len(gold_standard)}개 로드 완료: {GOLD_STANDARD_PATH}")

    client = get_qdrant_client()
    model = load_embed_model()
    reranker = load_reranker()

    correct_scores: list[float] = []
    incorrect_scores: list[float] = []

    for i, item in enumerate(gold_standard, 1):
        query = item["query"]
        gt_docs = set(item["relevant_docs_jo"])

        law_refs = search_jo(query, client, model, reranker, top_k=TOP_K_FOR_CHECK)

        for ref in law_refs:
            if ref.chunk_id in gt_docs:
                correct_scores.append(ref.score)
            else:
                incorrect_scores.append(ref.score)

        print(f"  [{i}/{len(gold_standard)}] 처리 완료", end="\r")

    print(f"\n\n정답으로 걸린 score 개수: {len(correct_scores)}")
    print(f"오답으로 걸린 score 개수: {len(incorrect_scores)}")

    correct_pctl = {
        "p10": round(float(np.percentile(correct_scores, 10)), 4),
        "p25": round(float(np.percentile(correct_scores, 25)), 4),
        "p50": round(float(np.percentile(correct_scores, 50)), 4),
    }
    incorrect_pctl = {
        "p50": round(float(np.percentile(incorrect_scores, 50)), 4),
        "p75": round(float(np.percentile(incorrect_scores, 75)), 4),
        "p90": round(float(np.percentile(incorrect_scores, 90)), 4),
    }
    recommended_min_score = round((correct_pctl["p10"] + incorrect_pctl["p90"]) / 2, 4)

    print("\n=== 정답(True Positive) score 분포 ===")
    print(f"  10th percentile: {correct_pctl['p10']}")
    print(f"  25th percentile: {correct_pctl['p25']}")
    print(f"  50th (median):   {correct_pctl['p50']}")

    print("\n=== 오답(False Positive) score 분포 ===")
    print(f"  50th (median):   {incorrect_pctl['p50']}")
    print(f"  75th percentile: {incorrect_pctl['p75']}")
    print(f"  90th percentile: {incorrect_pctl['p90']}")

    print(f"\n권장 min_score (오답 90th ~ 정답 10th 중간값): {recommended_min_score}")

    result = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "gold_standard_path": str(GOLD_STANDARD_PATH),
        "num_queries": len(gold_standard),
        "top_k_for_check": TOP_K_FOR_CHECK,
        "search_params": {
            "alpha": DEFAULT_ALPHA,
            "rrf_k": DEFAULT_RRF_K,
            "fetch_k": DEFAULT_FETCH_K,
            "rerank_k": DEFAULT_RERANK_K,
            "use_reranker": True,
        },
        "counts": {
            "correct": len(correct_scores),
            "incorrect": len(incorrect_scores),
        },
        "correct_score_percentiles": correct_pctl,
        "incorrect_score_percentiles": incorrect_pctl,
        "recommended_min_score": recommended_min_score,
        "raw_scores": {
            "correct": [round(s, 4) for s in correct_scores],
            "incorrect": [round(s, 4) for s in incorrect_scores],
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 결과 저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()