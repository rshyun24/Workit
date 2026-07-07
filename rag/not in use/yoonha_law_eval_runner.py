"""
Workit - RAG 성능 평가 실행 스크립트
파일명: yoonha_law_eval_runner.py
위치:   rag/yoonha_law_eval_runner.py

확정된 평가셋(eval_dataset.json 또는 eval_candidates.json)으로
3개 RAG (JoRAG / HoRAG / HoXrefRAG) 를 평가.

실험 모드:
  기본     : 단일 alpha + 단일 K로 빠른 평가
  --sweep  : alpha 0.1~0.9 구간 탐색 → 최적 alpha 탐색
  --topk   : K=5,10,20,30 변화에 따른 Recall 포화점 탐색
  --chunk  : law_kb_jo vs law_kb_ho 청크 단위 비교
  --qtype  : 위반/누락/정상 질문 유형별 성능 분석
  --all    : 위 4가지 모두 실행

실행 예시:
    python yoonha_law_eval_runner.py --dataset rag/eval_candidates.json
    python yoonha_law_eval_runner.py --dataset rag/eval_dataset.json --sweep
    python yoonha_law_eval_runner.py --dataset rag/eval_dataset.json --all
    python yoonha_law_eval_runner.py --dataset rag/eval_dataset.json --alpha 0.7 --k 10
    python yoonha_law_eval_runner.py --dataset rag/eval_dataset.json --rerank --all

결과 저장:
    rag/eval_results/result_{timestamp}.json
    rag/eval_results/result_{timestamp}_summary.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
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
# 상수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTPUT_DIR    = Path("rag/eval_results")
ALPHA_SWEEP   = [round(a, 1) for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
TOPK_SWEEP    = [5, 10, 20, 30]
K_DEFAULT     = [1, 3, 5, 10]

# 질문 유형 태그 (eval_dataset.json의 "qtype" 필드 값)
QTYPE_LABELS = {
    "violation" : "위반",
    "missing"   : "누락",
    "normal"    : "정상",
}

RAG_CONFIGS = {
    "jo"  : ("JoRAG",     "search_jo"),
    "ho"  : ("HoRAG",     "search_ho"),
    "xref": ("HoXrefRAG", "search_ho_xref"),
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 데이터셋 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_dataset(path: str) -> list[dict]:
    """
    eval_candidates.json 또는 eval_dataset.json 로드.
    - _reviewed=false 항목은 경고 후 포함 (--strict 옵션으로 제외 가능)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    unreviewed = [d for d in data if not d.get("_reviewed", True)]
    if unreviewed:
        print(f"⚠️  미검토 항목 {len(unreviewed)}개 포함됨 (--strict 옵션으로 제외 가능)")

    # 평가에 필요한 필드만 추출
    dataset = []
    for item in data:
        if not item.get("relevant_ids"):
            print(f"  ⚠️  relevant_ids 없음, 건너뜀: {item.get('query', '')[:40]}")
            continue
        dataset.append({
            "query"       : item["query"],
            "relevant_ids": item["relevant_ids"],
            "qtype"       : item.get("qtype", "unknown"),   # 질문 유형 (없으면 unknown)
            "purpose"     : item.get("purpose", ""),
        })

    print(f"평가셋 로드: {len(dataset)}개 쿼리 (원본 {len(data)}개 중)")
    return dataset


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 지표 계산
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    top_k = set(retrieved[:k])
    hits  = sum(1 for r in relevant if r in top_k)
    return hits / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    for rank, rid in enumerate(retrieved, 1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    relevant_set = set(relevant)

    def dcg(ids: list[str]) -> float:
        return sum(
            1.0 / math.log2(rank + 1)
            for rank, rid in enumerate(ids[:k], 1)
            if rid in relevant_set
        )

    ideal = [r for r in relevant if r in relevant_set][:k]
    idcg  = dcg(ideal + ["__pad__"] * k)
    return dcg(retrieved) / idcg if idcg > 0 else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 검색 결과 → chunk_id 추출
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
# 단일 조건 평가
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class EvalResult:
    rag_name   : str
    alpha      : float
    top_k      : int
    k_values   : list[int]
    recall     : dict[int, float] = field(default_factory=dict)
    mrr        : float            = 0.0
    ndcg       : dict[int, float] = field(default_factory=dict)
    latency_ms : float            = 0.0
    per_sample : list[dict]       = field(default_factory=list)   # 샘플별 상세 결과


def evaluate_single(
    rag_name  : str,
    search_fn : Callable,
    dataset   : list[dict],
    client    : QdrantClient,
    model,
    laws_ref  : dict,
    reranker1 = None,
    reranker2 = None,
    k_values  : list[int] = K_DEFAULT,
    top_k     : int       = TOP_K,
    alpha     : float     = RRF_ALPHA,
    verbose   : bool      = True,
) -> EvalResult:
    result    = EvalResult(rag_name=rag_name, alpha=alpha, top_k=top_k, k_values=list(k_values))
    rr_list   : list[float]            = []
    recall_acc: dict[int, list[float]] = {k: [] for k in k_values}
    ndcg_acc  : dict[int, list[float]] = {k: [] for k in k_values}
    latencies : list[float]            = []

    for i, sample in enumerate(dataset, 1):
        query    = sample["query"]
        relevant = sample["relevant_ids"]

        t0       = time.perf_counter()
        refs     = search_fn(query, client, model, laws_ref, reranker1, reranker2, top_k, alpha)
        elapsed  = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        retrieved = get_retrieved_ids(refs)
        rr        = reciprocal_rank(retrieved, relevant)
        rr_list.append(rr)

        sample_result = {
            "query"        : query,
            "relevant_ids" : relevant,
            "retrieved_ids": retrieved[:top_k],
            "qtype"        : sample.get("qtype", "unknown"),
            "rr"           : rr,
            "recall"       : {},
            "ndcg"         : {},
        }

        for k in k_values:
            r = recall_at_k(retrieved, relevant, k)
            n = ndcg_at_k(retrieved, relevant, k)
            recall_acc[k].append(r)
            ndcg_acc[k].append(n)
            sample_result["recall"][k] = r
            sample_result["ndcg"][k]   = n

        result.per_sample.append(sample_result)

        if verbose:
            hit = "✅" if any(r in retrieved[:5] for r in relevant) else "❌"
            print(f"  [{i:2d}/{len(dataset)}] {hit} RR={rr:.3f} | {query[:35]}...")

    result.mrr        = sum(rr_list) / len(rr_list) if rr_list else 0.0
    result.recall     = {k: sum(v) / len(v) for k, v in recall_acc.items()}
    result.ndcg       = {k: sum(v) / len(v) for k, v in ndcg_acc.items()}
    result.latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 1: 기본 평가
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_baseline(args, dataset, client, model, laws_ref, reranker1, reranker2, search_fns) -> dict:
    print(f"\n{'━'*55}")
    print("📊 기본 평가 (alpha={:.1f}, K={})".format(args.alpha, max(args.k_values)))
    print(f"{'━'*55}")

    results = {}
    for key in args.rag_targets:
        name, fn = RAG_CONFIGS[key][0], search_fns[key]
        print(f"\n▶ {name}")
        er = evaluate_single(
            rag_name=name, search_fn=fn, dataset=dataset,
            client=client, model=model, laws_ref=laws_ref,
            reranker1=reranker1, reranker2=reranker2,
            k_values=args.k_values, top_k=max(args.k_values),
            alpha=args.alpha,
        )
        results[key] = er

    _print_table(list(results.values()), args.k_values, title="기본 평가 결과")
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 2: Alpha Sweep
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_alpha_sweep(args, dataset, client, model, laws_ref, reranker1, reranker2, search_fns) -> dict:
    """
    alpha를 0.1~0.9 구간으로 변화시키며 Recall@10, MRR 추적.
    alpha = dense 벡터 가중치 (1-alpha = sparse 가중치)
    """
    print(f"\n{'━'*55}")
    print("🔍 Alpha Sweep 실험 (alpha: dense 비중)")
    print(f"  dense↑  = 의미 기반 검색 강화")
    print(f"  sparse↑ = 키워드 정확 매칭 강화 (1-alpha)")
    print(f"{'━'*55}")

    sweep_results: dict[str, list[dict]] = {k: [] for k in args.rag_targets}

    for alpha in ALPHA_SWEEP:
        print(f"\n  alpha={alpha:.1f} ...", end="", flush=True)
        for key in args.rag_targets:
            name, fn = RAG_CONFIGS[key][0], search_fns[key]
            er = evaluate_single(
                rag_name=name, search_fn=fn, dataset=dataset,
                client=client, model=model, laws_ref=laws_ref,
                reranker1=reranker1, reranker2=reranker2,
                k_values=[10], top_k=10,
                alpha=alpha, verbose=False,
            )
            sweep_results[key].append({
                "alpha"    : alpha,
                "mrr"      : er.mrr,
                "recall@10": er.recall.get(10, 0.0),
                "ndcg@10"  : er.ndcg.get(10, 0.0),
            })
        print(" done")

    # 결과 출력
    print(f"\n{'='*70}")
    print(f"{'Alpha Sweep 결과':^70}")
    print(f"{'='*70}")
    for key in args.rag_targets:
        name = RAG_CONFIGS[key][0]
        rows = sweep_results[key]
        print(f"\n  [{name}]")
        print(f"  {'alpha':>6} | {'MRR':>6} | {'R@10':>6} | {'N@10':>6}")
        print(f"  {'─'*35}")
        best_r10   = max(rows, key=lambda x: x["recall@10"])
        best_mrr   = max(rows, key=lambda x: x["mrr"])
        for row in rows:
            mark = ""
            if row["alpha"] == best_r10["alpha"]:
                mark += " ← best R@10"
            if row["alpha"] == best_mrr["alpha"] and row["alpha"] != best_r10["alpha"]:
                mark += " ← best MRR"
            print(f"  {row['alpha']:>6.1f} | {row['mrr']:>6.4f} | {row['recall@10']:>6.4f} | {row['ndcg@10']:>6.4f}{mark}")
        print(f"\n  ✅ 최적 alpha (R@10 기준): {best_r10['alpha']:.1f} → R@10={best_r10['recall@10']:.4f}")
        print(f"  ✅ 최적 alpha (MRR  기준): {best_mrr['alpha']:.1f} → MRR={best_mrr['mrr']:.4f}")

    return sweep_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 3: Top-K 민감도
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_topk_sensitivity(args, dataset, client, model, laws_ref, reranker1, reranker2, search_fns) -> dict:
    """
    K를 늘릴수록 Recall이 어느 지점에서 포화되는지 확인.
    Recall이 거의 오르지 않는 K가 실용적인 top_k 기준.
    """
    print(f"\n{'━'*55}")
    print("📈 Top-K 민감도 실험")
    print(f"  K 값: {TOPK_SWEEP}")
    print(f"{'━'*55}")

    topk_results: dict[str, list[dict]] = {k: [] for k in args.rag_targets}

    for k in TOPK_SWEEP:
        print(f"\n  K={k} ...", end="", flush=True)
        for key in args.rag_targets:
            name, fn = RAG_CONFIGS[key][0], search_fns[key]
            er = evaluate_single(
                rag_name=name, search_fn=fn, dataset=dataset,
                client=client, model=model, laws_ref=laws_ref,
                reranker1=reranker1, reranker2=reranker2,
                k_values=[k], top_k=k,
                alpha=args.alpha, verbose=False,
            )
            topk_results[key].append({
                "k"        : k,
                "recall"   : er.recall.get(k, 0.0),
                "ndcg"     : er.ndcg.get(k, 0.0),
                "latency"  : er.latency_ms,
            })
        print(" done")

    # 결과 출력
    print(f"\n{'='*65}")
    print(f"{'Top-K 민감도 결과':^65}")
    print(f"{'='*65}")
    for key in args.rag_targets:
        name = RAG_CONFIGS[key][0]
        rows = topk_results[key]
        print(f"\n  [{name}]")
        print(f"  {'K':>4} | {'Recall':>7} | {'NDCG':>7} | {'Δ Recall':>9} | {'Lat(ms)':>8}")
        print(f"  {'─'*48}")
        prev_r = 0.0
        for row in rows:
            delta = row["recall"] - prev_r
            mark  = " ← 포화" if delta < 0.02 and prev_r > 0 else ""
            print(f"  {row['k']:>4} | {row['recall']:>7.4f} | {row['ndcg']:>7.4f} | {delta:>+9.4f} | {row['latency']:>8.1f}{mark}")
            prev_r = row["recall"]

    return topk_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 4: 청크 단위 비교 (JoRAG vs HoRAG)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_chunk_comparison(args, dataset, client, model, laws_ref, reranker1, reranker2, search_fns) -> dict:
    """
    law_kb_jo (조 단위) vs law_kb_ho (항·호 단위) 청크 단위 비교.
    동일 쿼리에서 어느 단위가 더 정확히 관련 조항을 찾아오는지 확인.
    """
    print(f"\n{'━'*55}")
    print("🔬 청크 단위 비교 (조 단위 vs 항·호 단위)")
    print(f"{'━'*55}")

    targets = ["jo", "ho"] if "jo" in args.rag_targets and "ho" in args.rag_targets else args.rag_targets

    results = {}
    for key in targets:
        name, fn = RAG_CONFIGS[key][0], search_fns[key]
        print(f"\n▶ {name}")
        er = evaluate_single(
            rag_name=name, search_fn=fn, dataset=dataset,
            client=client, model=model, laws_ref=laws_ref,
            reranker1=reranker1, reranker2=reranker2,
            k_values=args.k_values, top_k=max(args.k_values),
            alpha=args.alpha,
        )
        results[key] = er

    _print_table(list(results.values()), args.k_values, title="청크 단위 비교 결과")

    # 쿼리별 승패 분석
    if "jo" in results and "ho" in results:
        jo_samples = {s["query"]: s for s in results["jo"].per_sample}
        ho_samples = {s["query"]: s for s in results["ho"].per_sample}
        print(f"\n  쿼리별 R@10 비교 (JoRAG vs HoRAG):")
        print(f"  {'query':<40} | {'JoRAG':>6} | {'HoRAG':>6} | 우위")
        print(f"  {'─'*65}")
        for q, js in jo_samples.items():
            hs    = ho_samples.get(q, {})
            jr    = js["recall"].get(10, 0.0)
            hr    = hs.get("recall", {}).get(10, 0.0) if hs else 0.0
            winner = "Jo↑" if jr > hr else ("Ho↑" if hr > jr else "동일")
            print(f"  {q[:40]:<40} | {jr:>6.3f} | {hr:>6.3f} | {winner}")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 5: 질문 유형별 성능 분석
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_qtype_analysis(args, dataset, client, model, laws_ref, reranker1, reranker2, search_fns) -> dict:
    """
    위반/누락/정상 질문 유형별 성능 비교.
    어떤 유형의 계약 문제를 RAG가 잘 잡아내는지 확인.
    eval_dataset의 각 항목에 "qtype": "violation" | "missing" | "normal" 필드 필요.
    """
    print(f"\n{'━'*55}")
    print("🏷️  질문 유형별 성능 분석 (위반/누락/정상)")
    print(f"{'━'*55}")

    qtypes = list(set(s["qtype"] for s in dataset if s["qtype"] != "unknown"))
    if not qtypes:
        print("  ⚠️  qtype 필드가 없습니다. eval_dataset.json에 'qtype' 추가 후 재실행하세요.")
        print("      예: 'qtype': 'violation' | 'missing' | 'normal'")
        return {}

    qtype_results: dict[str, dict] = {}

    for key in args.rag_targets:
        name, fn = RAG_CONFIGS[key][0], search_fns[key]
        print(f"\n▶ {name}")

        # 전체 평가 1회 실행
        er = evaluate_single(
            rag_name=name, search_fn=fn, dataset=dataset,
            client=client, model=model, laws_ref=laws_ref,
            reranker1=reranker1, reranker2=reranker2,
            k_values=args.k_values, top_k=max(args.k_values),
            alpha=args.alpha, verbose=False,
        )

        # qtype별 집계
        type_metrics: dict[str, dict] = {}
        for qt in qtypes:
            samples = [s for s in er.per_sample if s["qtype"] == qt]
            if not samples:
                continue
            type_metrics[qt] = {
                "n"     : len(samples),
                "mrr"   : sum(s["rr"] for s in samples) / len(samples),
                "recall": {k: sum(s["recall"][k] for s in samples) / len(samples) for k in args.k_values},
                "ndcg"  : {k: sum(s["ndcg"][k]   for s in samples) / len(samples) for k in args.k_values},
            }

        qtype_results[key] = type_metrics

        # 출력
        print(f"\n  {'유형':<8} | {'N':>3} | {'MRR':>6} | " +
              " | ".join(f"R@{k:>2}" for k in args.k_values))
        print(f"  {'─'*55}")
        for qt, m in type_metrics.items():
            label = QTYPE_LABELS.get(qt, qt)
            row   = f"  {label:<8} | {m['n']:>3} | {m['mrr']:>6.4f} | "
            row  += " | ".join(f"{m['recall'].get(k, 0):>4.3f}" for k in args.k_values)
            print(row)

    return qtype_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 결과 출력 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _print_table(results: list[EvalResult], k_values: list[int], title: str = "") -> None:
    K_SHOW = [k for k in k_values if k in (1, 3, 5, 10)]
    header = f"{'RAG':<14} | {'MRR':>6} | "
    header += " | ".join(f"R@{k:>2}" for k in K_SHOW)
    header += " | "
    header += " | ".join(f"N@{k:>2}" for k in K_SHOW)
    header += f" | {'Lat(ms)':>8}"

    sep = "=" * len(header)
    if title:
        print(f"\n{sep}")
        print(f"{title:^{len(header)}}")
    print(f"{sep}")
    print(header)
    print(sep)

    best: dict[str, tuple[str, float]] = {}
    for r in results:
        row  = f"{r.rag_name:<14} | {r.mrr:>6.4f} | "
        row += " | ".join(f"{r.recall.get(k, 0):>4.3f}" for k in K_SHOW)
        row += " | "
        row += " | ".join(f"{r.ndcg.get(k, 0):>4.3f}" for k in K_SHOW)
        row += f" | {r.latency_ms:>8.1f}"
        print(row)

        for k in K_SHOW:
            if f"R@{k}" not in best or r.recall.get(k, 0) > best[f"R@{k}"][1]:
                best[f"R@{k}"] = (r.rag_name, r.recall.get(k, 0))
            if f"N@{k}" not in best or r.ndcg.get(k, 0) > best[f"N@{k}"][1]:
                best[f"N@{k}"] = (r.rag_name, r.ndcg.get(k, 0))
        if "MRR" not in best or r.mrr > best["MRR"][1]:
            best["MRR"] = (r.rag_name, r.mrr)

    print(sep)
    print("\n🏆 지표별 최고:")
    for metric, (name, val) in sorted(best.items()):
        print(f"   {metric:<6}: {name}  ({val:.4f})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 결과 저장
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_results(all_results: dict, args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"result_{ts}.json"
    txt_path  = OUTPUT_DIR / f"result_{ts}_summary.txt"

    # JSON 직렬화 (EvalResult → dict)
    def _serialize(obj):
        if isinstance(obj, EvalResult):
            return {
                "rag_name"   : obj.rag_name,
                "alpha"      : obj.alpha,
                "top_k"      : obj.top_k,
                "mrr"        : obj.mrr,
                "recall"     : {str(k): v for k, v in obj.recall.items()},
                "ndcg"       : {str(k): v for k, v in obj.ndcg.items()},
                "latency_ms" : obj.latency_ms,
                "per_sample" : obj.per_sample,
            }
        return obj

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=_serialize)

    # 요약 텍스트
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Workit RAG 평가 결과\n")
        f.write(f"실행 시각: {ts}\n")
        f.write(f"데이터셋 : {args.dataset}\n")
        f.write(f"Alpha    : {args.alpha}\n")
        f.write(f"K 값     : {args.k_values}\n")
        f.write(f"리랭커   : {'ON' if args.rerank else 'OFF'}\n")
        f.write(f"실험 모드: {', '.join(args.experiments)}\n\n")
        f.write("상세 결과는 JSON 파일을 참고하세요.\n")

    print(f"\n💾 결과 저장:")
    print(f"   JSON   : {json_path}")
    print(f"   Summary: {txt_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Workit RAG 성능 평가")
    p.add_argument("--dataset",  required=True, help="평가셋 JSON 파일 경로")
    p.add_argument("--alpha",    type=float, default=RRF_ALPHA,
                   help=f"dense 비중 (기본: {RRF_ALPHA})")
    p.add_argument("--k",        type=int,   default=None,
                   help="top-K 고정 (기본: 1,3,5,10 모두)")
    p.add_argument("--rag",      choices=["jo", "ho", "xref", "all"], default="all",
                   help="평가할 RAG (기본: all)")
    p.add_argument("--rerank",   action="store_true", help="리랭커 사용")
    p.add_argument("--device",   default="cpu", help="리랭커 디바이스")
    p.add_argument("--strict",   action="store_true", help="미검토(_reviewed=false) 항목 제외")

    # 실험 모드
    exp = p.add_argument_group("실험 모드 (복수 선택 가능)")
    exp.add_argument("--sweep",  action="store_true", help="Alpha Sweep")
    exp.add_argument("--topk",   action="store_true", help="Top-K 민감도")
    exp.add_argument("--chunk",  action="store_true", help="청크 단위 비교")
    exp.add_argument("--qtype",  action="store_true", help="질문 유형별 분석")
    exp.add_argument("--all",    action="store_true", help="모든 실험 실행")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    args.k_values    = [args.k] if args.k else K_DEFAULT
    args.rag_targets = list(RAG_CONFIGS.keys()) if args.rag == "all" else [args.rag]
    args.experiments = []
    if args.all or not any([args.sweep, args.topk, args.chunk, args.qtype]):
        args.experiments = ["baseline"]
    if args.sweep or args.all: args.experiments.append("sweep")
    if args.topk  or args.all: args.experiments.append("topk")
    if args.chunk or args.all: args.experiments.append("chunk")
    if args.qtype or args.all: args.experiments.append("qtype")
    if "baseline" not in args.experiments:
        args.experiments.insert(0, "baseline")

    print("=" * 55)
    print("Workit RAG 성능 평가")
    print(f"  데이터셋 : {args.dataset}")
    print(f"  Alpha    : {args.alpha}")
    print(f"  K 값     : {args.k_values}")
    print(f"  리랭커   : {'ON' if args.rerank else 'OFF'}")
    print(f"  실험     : {', '.join(args.experiments)}")
    print(f"  대상 RAG : {', '.join(args.rag_targets)}")
    print("=" * 55)

    # 데이터셋 로드
    dataset = load_dataset(args.dataset)
    if args.strict:
        dataset = [d for d in dataset if d.get("_reviewed", True)]
        print(f"  (strict 모드) 검토 완료 항목만: {len(dataset)}개")

    # 모델 / 클라이언트 로드
    client   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    model    = load_embed_model()
    laws_ref = load_laws_ref()

    reranker1, reranker2 = None, None
    if args.rerank:
        reranker1, reranker2 = load_rerankers(device=args.device)

    search_fns = {
        "jo"  : search_jo,
        "ho"  : search_ho,
        "xref": search_ho_xref,
    }

    all_results = {}

    if "baseline" in args.experiments:
        all_results["baseline"] = run_baseline(
            args, dataset, client, model, laws_ref, reranker1, reranker2, search_fns)

    if "sweep" in args.experiments:
        all_results["alpha_sweep"] = run_alpha_sweep(
            args, dataset, client, model, laws_ref, reranker1, reranker2, search_fns)

    if "topk" in args.experiments:
        all_results["topk_sensitivity"] = run_topk_sensitivity(
            args, dataset, client, model, laws_ref, reranker1, reranker2, search_fns)

    if "chunk" in args.experiments:
        all_results["chunk_comparison"] = run_chunk_comparison(
            args, dataset, client, model, laws_ref, reranker1, reranker2, search_fns)

    if "qtype" in args.experiments:
        all_results["qtype_analysis"] = run_qtype_analysis(
            args, dataset, client, model, laws_ref, reranker1, reranker2, search_fns)

    save_results(all_results, args)


if __name__ == "__main__":
    main()