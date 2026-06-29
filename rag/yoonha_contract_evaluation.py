"""
yoonha_contract_evaluation.py
==============================
Workit law_kb — Hard Negative 포함 RAG 평가 파이프라인

[ 실행 ]
  py rag/yoonha_contract_evaluation.py --skip-rerank
  py rag/yoonha_contract_evaluation.py --skip-rerank --alpha 0.7
  py rag/yoonha_contract_evaluation.py --skip-rerank --alpha 1.0
  py rag/yoonha_contract_evaluation.py --hn-path data/hn_seed/hard_negatives_output.json

결과 파일명: eval_results_ho_alpha{N}.json (alpha값 자동 포함)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_THIS_DIR     = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from yoonha_law_rag import (
    load_laws_ref,
    search_law_for_clause,
    CrossEncoderReranker,
    QDRANT_HOST,
    QDRANT_PORT,
    EMBED_MODEL,
    RERANKER1_MODEL,
    RERANKER2_MODEL,
    TOP_K,
    RRF_ALPHA,
)


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

HN_PATH    = _PROJECT_ROOT / "data" / "hn_seed" / "hard_negatives_output.json"
TOP_K_LIST = [1, 5, 10]


def make_output_path(alpha: float) -> Path:
    """alpha 값을 파일명에 포함한 출력 경로 생성."""
    alpha_str = str(alpha).replace(".", "")   # 0.7 → 07, 1.0 → 10
    return _PROJECT_ROOT / "data" / "hn_seed" / f"eval_results_ho_alpha{alpha_str}.json"


# ═══════════════════════════════════════════════════════════════
# 1. 데이터 로드
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
# 2. 단일 쿼리 평가
# ═══════════════════════════════════════════════════════════════

def evaluate_single(item, client, model, laws_ref, reranker1, reranker2, top_k=max([1,5,10]), alpha=RRF_ALPHA):
    clause_text  = item["clause_text"]
    ground_truth = set(item.get("ground_truth", []))
    hard_neg_ids = {hn["chunk_id"] for hn in item.get("hard_negatives", []) if hn.get("chunk_id")}

    law_refs = search_law_for_clause(
        clause_text=clause_text, client=client, model=model, laws_ref=laws_ref,
        reranker1=reranker1, reranker2=reranker2, top_k=top_k, alpha=alpha,
    )
    ranked_ids = [ref.chunk_id for ref in law_refs]

    gt_rank = None
    for rank, cid in enumerate(ranked_ids, 1):
        if cid in ground_truth:
            gt_rank = rank
            break

    hn_ranks = [rank for rank, cid in enumerate(ranked_ids, 1) if cid in hard_neg_ids]

    return {
        "query_id":     item.get("query_id", ""),
        "category":     item.get("category", ""),
        "ground_truth": list(ground_truth),
        "hard_neg_ids": list(hard_neg_ids),
        "ranked_ids":   ranked_ids,
        "gt_rank":      gt_rank,
        "hn_ranks":     hn_ranks,
    }


# ═══════════════════════════════════════════════════════════════
# 3. 지표 계산
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
# 4. 결과 출력
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
            gt_str = ", ".join(r["ground_truth"])
            print(f"    [{r['category']}] {r['query_id']} — GT: {gt_str}")
    print(f"\n  미검색 총계: {metrics['miss_count']}개")
    print(f"{'='*55}")


# ═══════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════

def main(hn_path=HN_PATH, output_path=None, skip_rerank=False, alpha=RRF_ALPHA):
    if output_path is None:
        output_path = make_output_path(alpha)

    print("=" * 60)
    print(f"  Workit law_kb_ho — 항·호 단위 RAG 평가  (alpha={alpha})")
    print("=" * 60)

    print(f"\n[load] {hn_path}")
    if not hn_path.exists():
        print(f"  ❌ 파일 없음: {hn_path}")
        return

    seed_items = load_hn_seed(hn_path)

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

    print(f"\n[eval] {len(seed_items)}개 쿼리 평가 시작 (컬렉션: law_kb_ho, alpha={alpha})")
    print("  ※ CPU 모드는 시간이 걸립니다\n")

    t_total = time.time()
    eval_results = []

    for i, item in enumerate(seed_items, 1):
        t0 = time.time()
        result = evaluate_single(item=item, client=client, model=model, laws_ref=laws_ref,
                                 reranker1=reranker1, reranker2=reranker2,
                                 top_k=max(TOP_K_LIST), alpha=alpha)
        elapsed = time.time() - t0
        gt_rank  = result["gt_rank"]
        rank_str = f"rank={gt_rank}" if gt_rank else "미검색"
        hn_str   = f"HN상위={result['hn_ranks']}" if result["hn_ranks"] else "HN없음"
        print(f"  [{i:>3}/{len(seed_items)}] {item.get('query_id',''):<6} [{item.get('category',''):<10}] {rank_str:<10} {hn_str:<20} ({elapsed:.1f}초)")
        eval_results.append(result)

    elapsed_total = time.time() - t_total
    print(f"\n  완료 (총 {elapsed_total/60:.1f}분)")

    metrics = compute_metrics(eval_results)
    label   = f"항·호 단위 Hybrid (alpha={alpha})" + ("" if skip_rerank else " + 2-stage Reranker")
    print_metrics(metrics, label=label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {
            "collection":  "law_kb_ho",
            "granularity": "항·호 단위",
            "reranker":    not skip_rerank,
            "rrf_alpha":   alpha,
            "top_k_list":  TOP_K_LIST,
            "embed_model": EMBED_MODEL,
            "reranker1":   RERANKER1_MODEL if not skip_rerank else None,
            "reranker2":   RERANKER2_MODEL if not skip_rerank else None,
        },
        "metrics": {k: v for k, v in metrics.items() if k != "miss_items"},
        "miss_items": [
            {"query_id": r["query_id"], "category": r["category"], "ground_truth": r["ground_truth"]}
            for r in metrics.get("miss_items", [])
        ],
        "per_query": [
            {"query_id": r["query_id"], "category": r["category"],
             "gt_rank": r["gt_rank"], "hn_ranks": r["hn_ranks"], "ranked_ids": r["ranked_ids"]}
            for r in eval_results
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Workit 항·호 단위 RAG 평가")
    parser.add_argument("--hn-path",     type=Path,  default=HN_PATH)
    parser.add_argument("--output",      type=Path,  default=None, help="결과 저장 경로 (기본값: alpha값 자동 포함)")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--alpha",       type=float, default=RRF_ALPHA, help=f"RRF alpha 값 (기본값: {RRF_ALPHA})")
    args = parser.parse_args()
    main(hn_path=args.hn_path, output_path=args.output, skip_rerank=args.skip_rerank, alpha=args.alpha)