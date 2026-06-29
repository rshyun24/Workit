"""
yoonha_contract_evaluation_jo.py
==================================
Workit law_kb — 조(條) 단위 컬렉션(law_kb_jo) 기반 RAG 평가

기존 yoonha_contract_evaluation.py와 차이점:
  1. 검색 컬렉션: law_kb_ho → law_kb_jo
  2. ground_truth chunk_id를 조 단위로 변환
       예) LCAE_67_1   → LCAE_67
           LCAE_73_1_1 → LCAE_73
           LCAR_75     → LCAR_75  (이미 조 단위)
           LCA_30_의2_1_5 → LCA_30_의2
  3. hard_negative chunk_id도 동일하게 조 단위로 변환
  4. --alpha 옵션으로 RRF alpha 값 지정 가능
  5. 결과 파일명에 alpha 값 자동 포함 (eval_results_jo_alpha{N}.json)

[ 실행 ]
  py rag/yoonha_contract_evaluation_jo.py --skip-rerank
  py rag/yoonha_contract_evaluation_jo.py --skip-rerank --alpha 0.7
  py rag/yoonha_contract_evaluation_jo.py --skip-rerank --alpha 0.9
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

_THIS_DIR     = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from yoonha_law_rag import (
    load_laws_ref,
    CrossEncoderReranker,
    get_vectors,
    LawRef,
    QDRANT_HOST,
    QDRANT_PORT,
    EMBED_MODEL,
    RERANKER1_MODEL,
    RERANKER2_MODEL,
    RRF_ALPHA,
    FETCH_K,
    RERANK1_K,
    RERANK2_K,
)
from qdrant_client.models import SparseVector


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

COLLECTION_JO = "law_kb_jo"
HN_PATH       = _PROJECT_ROOT / "data" / "hn_seed" / "hard_negatives_output.json"
TOP_K         = 10
TOP_K_LIST    = [1, 5, 10]


def make_output_path(alpha: float) -> Path:
    """alpha 값을 파일명에 포함한 출력 경로 생성."""
    alpha_str = str(alpha).replace(".", "")   # 0.7 → 07, 1.0 → 10
    return _PROJECT_ROOT / "data" / "hn_seed" / f"eval_results_jo_alpha{alpha_str}.json"


# ═══════════════════════════════════════════════════════════════
# 1. chunk_id → 조 단위 변환
# ═══════════════════════════════════════════════════════════════

def to_jo_id(chunk_id: str) -> str:
    m = re.match(r'^([A-Za-z_]+?)_(\d.*)$', chunk_id)
    if not m:
        return chunk_id
    prefix = m.group(1)
    rest   = m.group(2)
    jo_match = re.match(r'^(\d+(?:_의\d+)?)', rest)
    if not jo_match:
        return chunk_id
    jo_num = jo_match.group(1)
    return f"{prefix}_{jo_num}"


# ═══════════════════════════════════════════════════════════════
# 2. 데이터 로드
# ═══════════════════════════════════════════════════════════════

def load_hn_seed(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    valid = [it for it in items if it.get("ground_truth") and it.get("clause_text")]
    skipped = len(items) - len(valid)
    if skipped:
        print(f"  ⚠️  ground_truth/clause_text 없는 항목 {skipped}개 제외")
    print(f"  평가 쿼리: {len(valid)}개 로드")
    return valid


# ═══════════════════════════════════════════════════════════════
# 3. 조 단위 하이브리드 검색
# ═══════════════════════════════════════════════════════════════

def _qdrant_hybrid_search_jo(
    clause_text : str,
    client      : QdrantClient,
    model       : BGEM3FlagModel,
    fetch_k     : int   = FETCH_K,
    alpha       : float = RRF_ALPHA,
) -> list[dict]:
    dense_vec, sparse_vec = get_vectors(clause_text, model)
    indices = list(sparse_vec.keys())
    values  = list(sparse_vec.values())
    RRF_K   = 60

    try:
        dense_results = client.query_points(
            collection_name=COLLECTION_JO,
            query=dense_vec,
            using="dense",
            limit=fetch_k,
            with_payload=True,
        ).points
        sparse_results = client.query_points(
            collection_name=COLLECTION_JO,
            query=SparseVector(indices=indices, values=values),
            using="sparse",
            limit=fetch_k,
            with_payload=True,
        ).points
    except Exception as e:
        print(f"  ⚠️  sparse 검색 실패, dense만 사용: {e}")
        dense_results = client.query_points(
            collection_name=COLLECTION_JO,
            query=dense_vec,
            using="dense",
            limit=fetch_k,
            with_payload=True,
        ).points
        sparse_results = []

    scores: dict[str, dict] = {}
    for rank, point in enumerate(dense_results, 1):
        cid = point.payload.get("chunk_id", str(point.id))
        scores[cid] = {"payload": point.payload, "dense_rank": rank, "sparse_rank": len(dense_results) + 1}
    for rank, point in enumerate(sparse_results, 1):
        cid = point.payload.get("chunk_id", str(point.id))
        if cid in scores:
            scores[cid]["sparse_rank"] = rank
        else:
            scores[cid] = {"payload": point.payload, "dense_rank": len(sparse_results) + 1, "sparse_rank": rank}

    results = []
    for cid, info in scores.items():
        rrf_score = (
            alpha         * (1 / (RRF_K + info["dense_rank"]))
            + (1 - alpha) * (1 / (RRF_K + info["sparse_rank"]))
        )
        results.append({"chunk_id": cid, "payload": info["payload"], "rrf_score": rrf_score})

    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results


def _rerank(query, candidates, reranker, top_k):
    if not candidates:
        return []
    texts = [c["payload"].get("text", c["payload"].get("chunk_text", "")) for c in candidates]
    pairs = [[query, t] for t in texts]
    scores = reranker.compute_score(pairs, normalize=True)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [item for _, item in ranked[:top_k]]


def search_law_jo(
    clause_text : str,
    client      : QdrantClient,
    model       : BGEM3FlagModel,
    laws_ref    : dict,
    reranker1   : CrossEncoderReranker | None = None,
    reranker2   : CrossEncoderReranker | None = None,
    top_k       : int   = TOP_K,
    alpha       : float = RRF_ALPHA,
) -> list[LawRef]:
    candidates = _qdrant_hybrid_search_jo(clause_text=clause_text, client=client, model=model, fetch_k=FETCH_K, alpha=alpha)
    if reranker1 is not None and candidates:
        candidates = _rerank(clause_text, candidates, reranker1, RERANK1_K)
    if reranker2 is not None and candidates:
        candidates = _rerank(clause_text, candidates, reranker2, RERANK2_K)

    law_refs: list[LawRef] = []
    for c in candidates[:top_k]:
        payload  = c["payload"]
        chunk_id = payload.get("chunk_id", "")
        ref_meta = laws_ref.get(chunk_id, {})
        law_refs.append(LawRef(
            chunk_id    = chunk_id,
            article     = ref_meta.get("article",  payload.get("article_number", "")),
            category    = ref_meta.get("category", payload.get("category", "")),
            law_name    = payload.get("law_name", ""),
            chunk_text  = payload.get("text", ""),
            score       = round(float(c.get("rrf_score", 0.0)), 4),
            is_risk_ref = bool(payload.get("is_risk_ref", False)),
            parent_id   = payload.get("parent_id", ""),
        ))
    return law_refs


# ═══════════════════════════════════════════════════════════════
# 4. 단일 쿼리 평가
# ═══════════════════════════════════════════════════════════════

def evaluate_single(item, client, model, laws_ref, reranker1, reranker2, top_k=TOP_K, alpha=RRF_ALPHA):
    clause_text = item["clause_text"]
    ground_truth_jo = {to_jo_id(cid) for cid in item.get("ground_truth", [])}
    hard_neg_jo = {
        to_jo_id(hn["chunk_id"])
        for hn in item.get("hard_negatives", [])
        if hn.get("chunk_id")
    }
    hard_neg_jo -= ground_truth_jo

    law_refs   = search_law_jo(clause_text=clause_text, client=client, model=model, laws_ref=laws_ref,
                               reranker1=reranker1, reranker2=reranker2, top_k=top_k, alpha=alpha)
    ranked_ids = [ref.chunk_id for ref in law_refs]

    gt_rank = None
    for rank, cid in enumerate(ranked_ids, 1):
        if cid in ground_truth_jo:
            gt_rank = rank
            break

    hn_ranks = [rank for rank, cid in enumerate(ranked_ids, 1) if cid in hard_neg_jo]

    return {
        "query_id":          item.get("query_id", ""),
        "category":          item.get("category", ""),
        "ground_truth":      list(ground_truth_jo),
        "ground_truth_orig": item.get("ground_truth", []),
        "hard_neg_ids":      list(hard_neg_jo),
        "ranked_ids":        ranked_ids,
        "gt_rank":           gt_rank,
        "hn_ranks":          hn_ranks,
    }


# ═══════════════════════════════════════════════════════════════
# 5. 지표 계산
# ═══════════════════════════════════════════════════════════════

def compute_metrics(eval_results):
    n = len(eval_results)
    if n == 0:
        return {}

    recall_counts = defaultdict(int)
    hn_hit_counts = defaultdict(int)
    mrr_sum       = 0.0
    miss_list     = []
    cat_stats = defaultdict(lambda: {"total": 0, "recall5_hit": 0, "mrr_sum": 0.0, "hn_hit5": 0})

    for r in eval_results:
        cat      = r["category"]
        gt_rank  = r["gt_rank"]
        hn_ranks = r["hn_ranks"]
        cat_stats[cat]["total"] += 1
        for k in TOP_K_LIST:
            if gt_rank is not None and gt_rank <= k:
                recall_counts[k] += 1
        if gt_rank is not None and gt_rank <= 5:
            cat_stats[cat]["recall5_hit"] += 1
        if gt_rank is not None:
            mrr_sum += 1.0 / gt_rank
            cat_stats[cat]["mrr_sum"] += 1.0 / gt_rank
        else:
            miss_list.append(r)
        for k in TOP_K_LIST:
            if any(rank <= k for rank in hn_ranks):
                hn_hit_counts[k] += 1
        if any(rank <= 5 for rank in hn_ranks):
            cat_stats[cat]["hn_hit5"] += 1

    metrics = {"n": n, "MRR": round(mrr_sum / n, 4), "miss_count": len(miss_list), "miss_items": miss_list}
    for k in TOP_K_LIST:
        metrics[f"Recall@{k}"] = round(recall_counts[k] / n, 4)
        metrics[f"HN_Hit@{k}"] = round(hn_hit_counts[k] / n, 4)
    metrics["category_breakdown"] = {
        cat: {
            "total":    v["total"],
            "Recall@5": round(v["recall5_hit"] / v["total"], 4),
            "MRR":      round(v["mrr_sum"]     / v["total"], 4),
            "HN_Hit@5": round(v["hn_hit5"]     / v["total"], 4),
        }
        for cat, v in cat_stats.items()
    }
    return metrics


# ═══════════════════════════════════════════════════════════════
# 6. 결과 출력
# ═══════════════════════════════════════════════════════════════

def print_metrics(metrics, label="평가 결과"):
    print(f"\n{'='*55}")
    print(f"  {label}  (n={metrics['n']})")
    print(f"{'='*55}")
    print(f"\n  [ Retrieval 지표 ]")
    for k in TOP_K_LIST:
        print(f"  Recall@{k:<3}  : {metrics[f'Recall@{k}']:.4f}")
    print(f"  MRR        : {metrics['MRR']:.4f}")
    print(f"\n  [ Hard Negative 억제율 (낮을수록 좋음) ]")
    for k in TOP_K_LIST:
        val = metrics[f'HN_Hit@{k}']
        bar = "█" * int(val * 20)
        print(f"  HN_Hit@{k:<3}  : {val:.4f}  {bar}")
    print(f"\n  [ 카테고리별 Recall@5 / MRR / HN_Hit@5 ]")
    breakdown = metrics.get("category_breakdown", {})
    print(f"  {'카테고리':<18} {'Recall@5':>9} {'MRR':>8} {'HN_Hit@5':>10} {'n':>4}")
    print(f"  {'-'*52}")
    for cat, v in sorted(breakdown.items(), key=lambda x: -x[1]["Recall@5"]):
        print(f"  {cat:<18} {v['Recall@5']:>9.4f} {v['MRR']:>8.4f} {v['HN_Hit@5']:>10.4f} {v['total']:>4}")
    if metrics.get("miss_items"):
        print(f"\n  [ 미검색 항목 (GT가 top{max(TOP_K_LIST)} 밖) ]")
        for r in metrics["miss_items"]:
            gt_orig = ", ".join(r.get("ground_truth_orig", r["ground_truth"]))
            gt_jo   = ", ".join(r["ground_truth"])
            print(f"    [{r['category']}] {r['query_id']} — 원본: {gt_orig} → 조단위: {gt_jo}")
    print(f"\n  미검색 총계: {metrics['miss_count']}개")
    print(f"{'='*55}")


# ═══════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════

def main(hn_path=HN_PATH, output_path=None, skip_rerank=False, alpha=RRF_ALPHA):
    if output_path is None:
        output_path = make_output_path(alpha)

    print("=" * 60)
    print(f"  Workit law_kb_jo — 조 단위 RAG 평가  (alpha={alpha})")
    print("=" * 60)

    print(f"\n[load] {hn_path}")
    if not hn_path.exists():
        print(f"  ❌ 파일 없음: {hn_path}")
        return

    seed_items = load_hn_seed(hn_path)

    sample_ids = [it["ground_truth"][0] for it in seed_items[:5]]
    print(f"\n  [변환 샘플] 항·호 단위 → 조 단위")
    for sid in sample_ids:
        print(f"    {sid} → {to_jo_id(sid)}")

    print(f"\n[init] Qdrant 연결: {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    laws_ref = load_laws_ref()

    print(f"\n[init] BGE-M3 임베딩 모델 로드: {EMBED_MODEL}")
    print("  ※ CPU 모드 — use_fp16=False")
    model = BGEM3FlagModel(EMBED_MODEL, use_fp16=False)

    reranker1, reranker2 = None, None
    if not skip_rerank:
        print(f"\n[init] Re-ranker 1단계 로드: {RERANKER1_MODEL}")
        reranker1 = CrossEncoderReranker(RERANKER1_MODEL, device='cpu')
        print(f"[init] Re-ranker 2단계 로드: {RERANKER2_MODEL}")
        reranker2 = CrossEncoderReranker(RERANKER2_MODEL, device='cpu')
    else:
        print("\n[skip] Reranker 생략 (--skip-rerank 옵션)")

    print(f"\n[eval] {len(seed_items)}개 쿼리 평가 시작 (컬렉션: {COLLECTION_JO}, alpha={alpha})")
    print("  ※ ground_truth를 조 단위로 변환하여 평가\n")

    t_total = time.time()
    eval_results = []

    for i, item in enumerate(seed_items, 1):
        t0 = time.time()
        result = evaluate_single(item=item, client=client, model=model, laws_ref=laws_ref,
                                 reranker1=reranker1, reranker2=reranker2, top_k=TOP_K, alpha=alpha)
        elapsed = time.time() - t0
        gt_rank  = result["gt_rank"]
        rank_str = f"rank={gt_rank}" if gt_rank else "미검색"
        hn_str   = f"HN상위={result['hn_ranks']}" if result["hn_ranks"] else "HN없음"
        print(f"  [{i:>3}/{len(seed_items)}] {item.get('query_id',''):<6} [{item.get('category',''):<10}] {rank_str:<10} {hn_str:<20} ({elapsed:.1f}초)")
        eval_results.append(result)

    elapsed_total = time.time() - t_total
    print(f"\n  완료 (총 {elapsed_total/60:.1f}분)")

    metrics = compute_metrics(eval_results)
    label   = f"조 단위 Hybrid (alpha={alpha})" + ("" if skip_rerank else " + 2-stage Reranker")
    print_metrics(metrics, label=label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {
            "collection":  COLLECTION_JO,
            "granularity": "조(條) 단위",
            "reranker":    not skip_rerank,
            "rrf_alpha":   alpha,
            "top_k_list":  TOP_K_LIST,
            "embed_model": EMBED_MODEL,
            "reranker1":   RERANKER1_MODEL if not skip_rerank else None,
            "reranker2":   RERANKER2_MODEL if not skip_rerank else None,
        },
        "metrics": {k: v for k, v in metrics.items() if k != "miss_items"},
        "miss_items": [
            {"query_id": r["query_id"], "category": r["category"],
             "ground_truth_orig": r.get("ground_truth_orig", []), "ground_truth_jo": r["ground_truth"]}
            for r in metrics.get("miss_items", [])
        ],
        "per_query": [
            {"query_id": r["query_id"], "category": r["category"],
             "ground_truth_orig": r.get("ground_truth_orig", []), "ground_truth_jo": r["ground_truth"],
             "gt_rank": r["gt_rank"], "hn_ranks": r["hn_ranks"], "ranked_ids": r["ranked_ids"]}
            for r in eval_results
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Workit 조 단위 RAG 평가")
    parser.add_argument("--hn-path",     type=Path,  default=HN_PATH)
    parser.add_argument("--output",      type=Path,  default=None, help="결과 저장 경로 (기본값: alpha값 자동 포함)")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--alpha",       type=float, default=RRF_ALPHA, help=f"RRF alpha 값 (기본값: {RRF_ALPHA})")
    args = parser.parse_args()
    main(hn_path=args.hn_path, output_path=args.output, skip_rerank=args.skip_rerank, alpha=args.alpha)