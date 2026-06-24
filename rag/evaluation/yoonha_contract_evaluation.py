"""
yoonha_hybrid_evaluation.py
----------------------------
Workit law_kb — Dense 단독 vs Hybrid(Dense+Sparse RRF) 성능 비교 평가

평가셋 구성:
  - chunks.json의 is_risk_ref=True 청크 (148개)
  - 각 청크 text 앞부분으로 쿼리 생성
  - 해당 chunk_id를 ground truth로 사용

측정 지표:
  - Recall@1, Recall@5, Recall@10
  - MRR (Mean Reciprocal Rank)
  - 카테고리별 breakdown

실행:
    python rag/evaluation/yoonha_contract_evaluation.py

요구사항:
    pip install qdrant-client FlagEmbedding
    Qdrant law_kb 컬렉션에 dense + sparse 벡터 모두 업로드된 상태
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Prefetch,
    SparseVector,
)

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
CHUNKS_PATH = Path("data/export_old/chunks.json")   # chunks.json 경로
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION  = "law_kb"
EMBED_MODEL = "BAAI/bge-m3"

TOP_K_LIST  = [1, 5, 10]   # Recall@k 측정 기준
FETCH_K     = 20            # RRF용 각 소스에서 가져오는 후보 수
QUERY_CHARS = 150           # 쿼리로 쓸 텍스트 앞부분 길이


# ──────────────────────────────────────────
# 평가셋 생성
# ──────────────────────────────────────────
def build_eval_dataset(chunks_path: Path) -> list[dict]:
    """
    is_risk_ref=True 청크만 추려서 평가셋 구성.
    query: chunk text 앞 QUERY_CHARS자
    ground_truth: chunk_id
    """
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    eval_data = []
    for chunk in chunks:
        if not chunk.get("is_risk_ref"):
            continue
        text = chunk.get("text", "").strip()
        if len(text) < 30:
            continue
        eval_data.append({
            "query_id":     chunk["chunk_id"],
            "query":        text[:QUERY_CHARS],
            "ground_truth": chunk["chunk_id"],
            "category":     chunk.get("category", ""),
            "article":      chunk.get("article", ""),
        })

    print(f"✅ 평가셋 구성: {len(eval_data)}개")
    cat_counts = defaultdict(int)
    for item in eval_data:
        cat_counts[item["category"]] += 1
    print("   카테고리별:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"     {cat}: {cnt}개")
    return eval_data


# ──────────────────────────────────────────
# 벡터 추출
# ──────────────────────────────────────────
def get_vectors(
    text: str,
    model: BGEM3FlagModel,
) -> tuple[list[float], dict[int, float]]:
    output = model.encode(
        [text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = output["dense_vecs"][0].tolist()

    sparse: dict[int, float] = {}
    for token_str, weight in output["lexical_weights"][0].items():
        token_id = model.tokenizer.convert_tokens_to_ids(token_str)
        if isinstance(token_id, int):
            sparse[token_id] = sparse.get(token_id, 0.0) + float(weight)

    return dense, sparse


# ──────────────────────────────────────────
# Dense 단독 검색
# ──────────────────────────────────────────
def search_dense(
    dense_vector: list[float],
    client: QdrantClient,
    top_k: int = max(TOP_K_LIST),
) -> list[str]:
    response = client.query_points(
        collection_name=COLLECTION,
        query=dense_vector,
        using="dense",
        limit=top_k,
    )
    return [p.payload.get("chunk_id", "") for p in response.points]


# ──────────────────────────────────────────
# Hybrid 검색 (Dense + Sparse RRF)
# ──────────────────────────────────────────
def search_hybrid(
    dense_vector: list[float],
    sparse_vector: dict[int, float],
    client: QdrantClient,
    top_k: int = max(TOP_K_LIST),
) -> list[str]:
    indices = list(sparse_vector.keys())
    values  = list(sparse_vector.values())

    try:
        response = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                Prefetch(query=dense_vector,                                 limit=FETCH_K, using="dense"),
                Prefetch(query=SparseVector(indices=indices, values=values), limit=FETCH_K, using="sparse"),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        )
        return [p.payload.get("chunk_id", "") for p in response.points]
    except Exception as e:
        print(f"  ⚠️  Hybrid 검색 실패 ({e}), dense 폴백")
        return search_dense(dense_vector, client, top_k)


# ──────────────────────────────────────────
# 지표 계산
# ──────────────────────────────────────────
def compute_metrics(
    eval_data: list[dict],
    results: list[list[str]],   # 각 쿼리의 ranked chunk_id 리스트
) -> dict:
    n = len(eval_data)
    recall_counts = defaultdict(int)
    mrr_sum = 0.0
    miss_list = []

    cat_recall5 = defaultdict(lambda: {"hit": 0, "total": 0})

    for i, (item, ranked) in enumerate(zip(eval_data, results)):
        gt = item["ground_truth"]
        cat = item["category"]

        # Recall@k
        for k in TOP_K_LIST:
            if gt in ranked[:k]:
                recall_counts[k] += 1

        # MRR
        if gt in ranked:
            rank = ranked.index(gt) + 1
            mrr_sum += 1.0 / rank
        else:
            miss_list.append(item)

        # 카테고리별 Recall@5
        cat_recall5[cat]["total"] += 1
        if gt in ranked[:5]:
            cat_recall5[cat]["hit"] += 1

    metrics = {
        "n": n,
        "MRR": round(mrr_sum / n, 4),
        "miss_count": len(miss_list),
        "miss_items": miss_list,
        "category_recall5": {
            cat: {
                "recall@5": round(v["hit"] / v["total"], 4),
                "hit": v["hit"],
                "total": v["total"],
            }
            for cat, v in cat_recall5.items()
        },
    }
    for k in TOP_K_LIST:
        metrics[f"Recall@{k}"] = round(recall_counts[k] / n, 4)

    return metrics


# ──────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────
def print_metrics(label: str, m: dict):
    print(f"\n{'='*50}")
    print(f"  {label}  (n={m['n']})")
    print(f"{'='*50}")
    for k in TOP_K_LIST:
        print(f"  Recall@{k:2d} : {m[f'Recall@{k}']:.4f}")
    print(f"  MRR      : {m['MRR']:.4f}")
    print(f"  미검색   : {m['miss_count']}개")

    print(f"\n  카테고리별 Recall@5:")
    for cat, v in sorted(m["category_recall5"].items(),
                         key=lambda x: -x[1]["recall@5"]):
        bar = "█" * int(v["recall@5"] * 20)
        print(f"    {cat:12s} {v['recall@5']:.3f}  {bar}  ({v['hit']}/{v['total']})")

    if m["miss_items"]:
        print(f"\n  미검색 항목:")
        for item in m["miss_items"]:
            print(f"    [{item['category']}] {item['article']} ({item['query_id']})")


def print_comparison(dense_m: dict, hybrid_m: dict):
    print(f"\n{'='*50}")
    print("  Dense vs Hybrid 비교")
    print(f"{'='*50}")
    print(f"  {'지표':<12} {'Dense':>8} {'Hybrid':>8} {'개선':>8}")
    print(f"  {'-'*40}")
    for k in TOP_K_LIST:
        key = f"Recall@{k}"
        d = dense_m[key]
        h = hybrid_m[key]
        diff = h - d
        sign = "+" if diff >= 0 else ""
        print(f"  {key:<12} {d:>8.4f} {h:>8.4f} {sign+f'{diff:.4f}':>8}")
    d_mrr = dense_m["MRR"]
    h_mrr = hybrid_m["MRR"]
    diff_mrr = h_mrr - d_mrr
    sign = "+" if diff_mrr >= 0 else ""
    print(f"  {'MRR':<12} {d_mrr:>8.4f} {h_mrr:>8.4f} {sign+f'{diff_mrr:.4f}':>8}")
    print()


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Workit law_kb — Dense vs Hybrid 평가")
    print("=" * 60)

    # 평가셋
    eval_data = build_eval_dataset(CHUNKS_PATH)

    # 클라이언트 / 모델
    print(f"\n📡 Qdrant 연결: {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print(f"📦 모델 로드: {EMBED_MODEL}")
    model = BGEM3FlagModel(EMBED_MODEL, use_fp16=True)

    # 벡터 추출 (쿼리별 1회만)
    print(f"\n🔢 {len(eval_data)}개 쿼리 벡터 추출 중...")
    t0 = time.time()
    dense_vecs = []
    sparse_vecs = []
    for i, item in enumerate(eval_data, 1):
        d, s = get_vectors(item["query"], model)
        dense_vecs.append(d)
        sparse_vecs.append(s)
        if i % 20 == 0:
            print(f"  {i}/{len(eval_data)} 완료")
    print(f"  완료 ({time.time()-t0:.1f}초)")

    # Dense 검색
    print(f"\n🔍 Dense 단독 검색 중...")
    t0 = time.time()
    dense_results = []
    for d in dense_vecs:
        ranked = search_dense(d, client, top_k=max(TOP_K_LIST))
        dense_results.append(ranked)
    print(f"  완료 ({time.time()-t0:.1f}초)")

    # Hybrid 검색
    print(f"\n🔍 Hybrid (Dense+Sparse RRF) 검색 중...")
    t0 = time.time()
    hybrid_results = []
    for d, s in zip(dense_vecs, sparse_vecs):
        ranked = search_hybrid(d, s, client, top_k=max(TOP_K_LIST))
        hybrid_results.append(ranked)
    print(f"  완료 ({time.time()-t0:.1f}초)")

    # 지표 계산
    dense_metrics  = compute_metrics(eval_data, dense_results)
    hybrid_metrics = compute_metrics(eval_data, hybrid_results)

    # 출력
    print_metrics("Dense 단독", dense_metrics)
    print_metrics("Hybrid (Dense + Sparse RRF)", hybrid_metrics)
    print_comparison(dense_metrics, hybrid_metrics)

    # JSON 저장
    out = {
        "eval_set_size": len(eval_data),
        "dense":  {k: v for k, v in dense_metrics.items()  if k != "miss_items"},
        "hybrid": {k: v for k, v in hybrid_metrics.items() if k != "miss_items"},
        "dense_miss":  [f"{i['category']} | {i['article']}" for i in dense_metrics["miss_items"]],
        "hybrid_miss": [f"{i['category']} | {i['article']}" for i in hybrid_metrics["miss_items"]],
    }
    with open("hybrid_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("💾 결과 저장: hybrid_eval_results.json")


if __name__ == "__main__":
    main()