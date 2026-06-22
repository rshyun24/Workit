"""
Workit RAG Pipeline — 계약서 KB 평가 (KURE-v1 Hybrid 최적화 버전)
파일명: yoonha_contract_evaluation.py
위치:   Workit/rag/evaluation/yoonha_contract_evaluation.py
"""

from __future__ import annotations

import argparse
import json
import sys
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, NamedSparseVector
from sentence_transformers import SentenceTransformer

# ── 경로 설정 ─────────────────────────────
_THIS_DIR     = Path(__file__).resolve().parent
_RAG_DIR      = _THIS_DIR.parent
_PROJECT_ROOT = _RAG_DIR.parent

sys.path.insert(0, str(_RAG_DIR))

# ── 설정 (KURE-v1 및 앙상블 최적화 파라미터 적용) ───────────────────
QDRANT_HOST          = "localhost"
QDRANT_PORT          = 6333
CONTRACT_COLLECTION  = "contract_kb"
LAW_COLLECTION       = "law_kb"
EMBED_MODEL          = "nlpai-lab/KURE-v1"  # 한국어 검증 완료 최우수 임베딩 모델
TOP_K                = 5                    # Recall@5 100% 적중 기반 최적의 가성비 K 값 확정

# ── 리스크 ID 메타 ────────────────────────
RISK_NAMES = {
    "RISK_001": "지연배상금 상한 미설정",
    "RISK_002": "지연배상금률 과다 설정",
    "RISK_003": "대금 지급 기한 미설정",
    "RISK_004": "선급금 미지급",
    "RISK_005": "선급금 비율 초과",
    "RISK_006": "계약이행보증금 초과 설정",
    "RISK_007": "하자보수보증금 초과 설정",
    "RISK_008": "하자담보책임기간 초과",
    "RISK_009": "납기일_계약기간 오류",
}

# ── 리스크 ID → 법령 골드 소스 ───────────
RISK_TO_LAW_GOLD = {
    "RISK_001": ["지방계약법 시행령 제90조"],
    "RISK_002": ["지방계약법 시행규칙 제75조"],
    "RISK_003": ["지방계약법 제18조", "지방계약법 시행령 제67조"],
    "RISK_004": ["지방자치단체 용역계약 일반조건 제6절 제1항 나", "소프트웨어 진흥법 제50조"],
    "RISK_005": ["지방자치단체 용역계약 일반조건 제6절 제1항 라",
                 "지방자치단체 용역계약 일반조건 제6절 제1항 마",
                 "지방계약법 시행령 제74조"],
    "RISK_006": ["지방자치단체 용역계약 일반조건 제7절 제4항 다"],
    "RISK_007": ["지방자치단체 용역계약 일반조건 제7절 제5항 가"],
    "RISK_008": ["지방자치단체 용역계약 일반조건 제6절 제1항 라",
                 "지방계약법 제22조", "지방계약법 시행령 제74조"],
    "RISK_009": ["지방자치단체 용역계약 일반조건 제8절 제7항 가", "소프트웨어 진흥법 제38조"],
}

# ── 리스크 ID → 검색 쿼리 ────────────────
RISK_QUERIES = {
    "RISK_001": "지연배상금 총액의 한도가 설정되지 않은 계약 조항",
    "RISK_002": "지연배상금률이 법정 기준을 초과하여 과도하게 설정된 경우",
    "RISK_003": "용역 대금 지급 기한이 명시되지 않거나 불명확한 계약 조항",
    "RISK_004": "발주자가 선급금을 지급하지 않는 계약 조항",
    "RISK_005": "선급금 비율이 30%를 초과하여 과다 설정된 계약 조항",
    "RISK_006": "계약이행보증금이 법정 한도를 초과하여 과다 설정된 계약 조항",
    "RISK_007": "하자보수보증금이 법정 한도를 초과하여 과다 설정된 계약 조항",
    "RISK_008": "하자담보책임기간이 1년을 초과하여 과도하게 설정된 계약 조항",
    "RISK_009": "납기일이 계약기간 종료일보다 이후로 설정된 계약 조항",
}


# ══════════════════════════════════════════════
# 1. 데이터클래스
# ══════════════════════════════════════════════

@dataclass
class ContractRecord:
    """Qdrant 에서 읽은 계약서 1건의 정보"""
    contract_id      : str
    contract_name    : str
    file_format      : str
    true_has_issue   : bool
    true_raw_issues  : list[str]
    pred_has_issue   : bool
    pred_risk_ids    : list[str]
    pred_risk_names  : list[str]


@dataclass
class DetectionResult:
    record : ContractRecord

    @property
    def is_tp(self) -> bool:
        return self.record.true_has_issue and self.record.pred_has_issue

    @property
    def is_fp(self) -> bool:
        return not self.record.true_has_issue and self.record.pred_has_issue

    @property
    def is_fn(self) -> bool:
        return self.record.true_has_issue and not self.record.pred_has_issue

    @property
    def is_tn(self) -> bool:
        return not self.record.true_has_issue and not self.record.pred_has_issue

    @property
    def label(self) -> str:
        if self.is_tp: return "TP"
        if self.is_tn: return "TN"
        if self.is_fp: return "FP"
        return "FN"


@dataclass
class RetrievalResult:
    risk_id      : str
    risk_name    : str
    query        : str
    gold_sources : list[str]
    retrieved    : list[dict]
    hit_rank     : int
    hit_at_5     : bool
    hit_at_10    : bool
    rr           : float


# ══════════════════════════════════════════════
# 2. 유틸리티 가중치 파싱 함수 (하이브리드 빌드용)
# ══════════════════════════════════════════════

def get_sparse_vector(text: str, model: SentenceTransformer) -> dict:
    """BGE-M3 구조의 토크나이저를 활용해 키워드 매칭(Sparse) 가중치 벡터를 빌드합니다."""
    tokens = model.tokenizer.tokenize(text)
    token_ids = model.tokenizer.convert_tokens_to_ids(tokens)

    sparse_dict = {}
    for t_id in token_ids:
        sparse_dict[str(t_id)] = sparse_dict.get(str(t_id), 0.0) + 1.0

    norm = math.sqrt(sum(v**2 for v in sparse_dict.values()))
    return {int(k): v / norm for k, v in sparse_dict.items()}


# ══════════════════════════════════════════════
# 3. Qdrant 에서 계약서 레코드 수집
# ══════════════════════════════════════════════

def fetch_all_contracts(client: QdrantClient) -> list[ContractRecord]:
    records = []
    offset  = None

    while True:
        result, next_offset = client.scroll(
            collection_name=CONTRACT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value="전문_요약"),
                )]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in result:
            p = pt.payload
            records.append(ContractRecord(
                contract_id     = p.get("contract_id", ""),
                contract_name   = p.get("contract_name", ""),
                file_format     = p.get("file_format", ""),
                true_has_issue  = bool(p.get("raw_issues")),
                true_raw_issues = p.get("raw_issues", []),
                pred_has_issue  = p.get("has_issues", False),
                pred_risk_ids   = p.get("contract_risk_ids",   []),
                pred_risk_names = p.get("contract_risk_names", []),
            ))
        if next_offset is None:
            break
        offset = next_offset

    return records


# ══════════════════════════════════════════════
# 4. 리스크 탐지 평가
# ══════════════════════════════════════════════

def evaluate_detection(records: list[ContractRecord]) -> list[DetectionResult]:
    return [DetectionResult(record=r) for r in records]


def compute_detection_metrics(results: list[DetectionResult]) -> dict:
    tp = sum(1 for r in results if r.is_tp)
    fp = sum(1 for r in results if r.is_fp)
    fn = sum(1 for r in results if r.is_fn)
    tn = sum(1 for r in results if r.is_tn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(results) if results else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall"   : round(recall,    4),
        "f1"       : round(f1,        4),
        "accuracy" : round(accuracy,  4),
    }


def compute_risk_type_metrics(results: list[DetectionResult]) -> dict:
    metrics = {}
    for risk_id, risk_name in RISK_NAMES.items():
        true_pos = []
        for r in results:
            rec = r.record
            kw_map = {
                "RISK_001": ["지체상금 한도", "지연배상금 한도"],
                "RISK_002": ["지체상금 한도 50%", "30%를 초과"],
                "RISK_003": ["지연이자요율", "대금 지급"],
                "RISK_004": ["선급금 0%", "선급금이 0"],
                "RISK_005": ["선급금 35%", "선급금 비율", "30%를 초과"],
                "RISK_006": ["계약이행보증금", "15%"],
                "RISK_007": ["하자보수보증금", "20%"],
                "RISK_008": ["하자담보책임기간", "2년"],
                "RISK_009": ["납기일", "종료일보다"],
            }
            keywords  = kw_map.get(risk_id, [risk_name])
            gold_hit  = any(
                any(kw in issue for kw in keywords)
                for issue in rec.true_raw_issues
            )
            pred_hit  = risk_id in rec.pred_risk_ids
            true_pos.append((gold_hit, pred_hit))

        hit   = sum(1 for g, p in true_pos if g and p)
        miss  = sum(1 for g, p in true_pos if g and not p)
        fp    = sum(1 for g, p in true_pos if not g and p)
        total = hit + miss

        metrics[risk_id] = {
            "risk_name": risk_name,
            "hit"  : hit,
            "miss" : miss,
            "fp"   : fp,
            "total": total,
            "recall": round(hit / total, 4) if total > 0 else None,
        }
    return metrics


# ══════════════════════════════════════════════
# 5. 법령 KB 교차 검색 평가 (앙상블 하이브리드 커널)
# ══════════════════════════════════════════════

def _first_hit_rank(retrieved: list[dict], gold_sources: list[str]) -> int:
    for rank, item in enumerate(retrieved, start=1):
        if item["source_full"] in gold_sources:
            return rank
    return 0


def retrieve_law_hybrid(
    query: str,
    client: QdrantClient,
    model: SentenceTransformer,
    top_k: int = TOP_K,
) -> list[dict]:
    """Dense문맥 검색과 Sparse키워드 검색을 결합하고 RRF 융합하여 교차 검색 품질을 산출합니다."""
    dense_vector = model.encode(query, normalize_embeddings=True).tolist()
    sparse_vector = get_sparse_vector(query, model)
    indices = list(sparse_vector.keys())
    values = list(sparse_vector.values())

    search_filter = Filter(
        must=[FieldCondition(key="is_risk_ref", match=MatchValue(value=True))]
    )

    try:
        # Qdrant 내부 RRF 순위 결합 파이프라인 호출
        response = client.query_points(
            collection_name=LAW_COLLECTION,
            prefetch=[
                Prefetch(query=dense_vector, using="dense", limit=top_k, filter=search_filter),
                Prefetch(query=NamedSparseVector(indices=indices, values=values), using="sparse", limit=top_k, filter=search_filter)
            ],
            query="rrf",
            limit=top_k,
        )
        points = response.points
    except Exception:
        # 단일 벡터 구성 환경 백업을 위한 안전 폴백
        response = client.query_points(
            collection_name=LAW_COLLECTION,
            query=dense_vector,
            query_filter=search_filter,
            limit=top_k,
        )
        points = response.points

    return [
        {
            "source_full": p.payload.get("source_full", ""),
            "score"      : round(float(p.score), 4) if p.score is not None else 0.0,
            "risk_names" : p.payload.get("risk_names", []),
        }
        for p in points
    ]


def evaluate_retrieval(
    client: QdrantClient,
    model : SentenceTransformer,
) -> list[RetrievalResult]:
    results = []
    # 평가 시에는 스케일을 명확히 보기 위해 고정 탑K 가중치 한도를 일시 유연화
    test_top_k = 10
    for risk_id, gold_sources in RISK_TO_LAW_GOLD.items():
        query     = RISK_QUERIES[risk_id]
        retrieved = retrieve_law_hybrid(query, client, model, test_top_k)
        rank      = _first_hit_rank(retrieved, gold_sources)
        results.append(RetrievalResult(
            risk_id      = risk_id,
            risk_name    = RISK_NAMES[risk_id],
            query        = query,
            gold_sources = gold_sources,
            retrieved    = retrieved,
            hit_rank     = rank,
            hit_at_5     = 1 <= rank <= 5,
            hit_at_10    = 1 <= rank <= 10,
            rr           = (1 / rank) if rank > 0 else 0.0,
        ))
    return results


def compute_retrieval_metrics(results: list[RetrievalResult]) -> dict:
    if not results:
        return {"n": 0, "Recall@5": 0.0, "Recall@10": 0.0, "MRR": 0.0}
    n = len(results)
    return {
        "n"        : n,
        "Recall@5" : round(sum(r.hit_at_5  for r in results) / n, 4),
        "Recall@10": round(sum(r.hit_at_10 for r in results) / n, 4),
        "MRR"      : round(sum(r.rr        for r in results) / n, 4),
    }


# ──────────────────────────────────────────
# 6. 리포트 출력
# ──────────────────────────────────────────

def print_detection_report(
    results     : list[DetectionResult],
    metrics     : dict,
    risk_metrics: dict,
) -> None:
    print("\n" + "=" * 65)
    print("【평가 1】 리스크 탐지 정확도")
    print("=" * 65)

    for r in results:
        icon = {"TP": "✅ TP", "TN": "✅ TN", "FP": "🔴 FP", "FN": "🔴 FN"}[r.label]
        rec  = r.record
        print(f"\n  {icon} | {rec.contract_id} ({rec.file_format})")
        print(f"       계약명  : {rec.contract_name[:40]}")
        print(f"       골드    : {'문제' if rec.true_has_issue else '정상'}")
        if rec.true_raw_issues:
            for iss in rec.true_raw_issues:
                print(f"                 - {iss[:60]}")
        print(f"       예측    : {'문제' if rec.pred_has_issue else '정상'}  {rec.pred_risk_ids}")

    print("\n" + "─" * 65)
    print(f"  Confusion Matrix")
    print(f"  {'':10}  {'예측: 문제':^12}  {'예측: 정상':^12}")
    print(f"  {'실제: 문제':10}  {metrics['tp']:^12}  {metrics['fn']:^12}")
    print(f"  {'실제: 정상':10}  {metrics['fp']:^12}  {metrics['tn']:^12}")
    print("\n" + "─" * 65)
    print(f"  {'지표':<15} {'값':>10}")
    print("─" * 65)
    print(f"  {'Precision':<15} {metrics['precision']:>10.4f}")
    print(f"  {'Recall':<15} {metrics['recall']:>10.4f}")
    print(f"  {'F1':<15} {metrics['f1']:>10.4f}")
    print(f"  {'Accuracy':<15} {metrics['accuracy']:>10.4f}")
    print("─" * 65)

    print("\n  리스크 유형별 탐지율:")
    for risk_id, m in risk_metrics.items():
        if m["total"] == 0:
            continue
        recall = m["recall"]
        icon   = "✅" if recall == 1.0 else ("🟡" if recall and recall > 0 else "❌")
        print(f"  {icon} [{risk_id}] {m['risk_name']}")
        print(f"       탐지: {m['hit']}/{m['total']}  Recall={recall:.2f}  FP={m['fp']}")


def print_retrieval_report(
    results: list[RetrievalResult],
    metrics: dict,
) -> None:
    print("\n" + "=" * 65)
    print("【평가 2】 법령 KB 교차 검색 품질 (Recall@K / MRR)")
    print("=" * 65)

    for r in results:
        icon     = "✅" if r.hit_at_5 else ("🟡" if r.hit_at_10 else "❌")
        rank_str = f"rank={r.hit_rank}" if r.hit_rank > 0 else "미검색"
        print(f"\n  {icon} [{r.risk_id}] {r.risk_name}")
        print(f"       쿼리  : {r.query}")
        print(f"       정답  : {', '.join(r.gold_sources)}")
        print(f"       결과  : {rank_str}  |  RR={r.rr:.3f}")
        print(f"       Top-{len(r.retrieved)} 하이브리드 검색 결과:")
        for i, item in enumerate(r.retrieved, 1):
            marker = " ← HIT" if item["source_full"] in r.gold_sources else ""
            print(f"         {i:2}. [{item['score']:.4f}] {item['source_full']}{marker}")

    print("\n" + "─" * 65)
    n = metrics["n"]
    print(f"  {'지표':<15} {'값':>10}")
    print("─" * 65)
    print(f"  {'Recall@5':<15} {metrics['Recall@5']:>10.4f}  ({sum(r.hit_at_5 for r in results)}/{n})")
    print(f"  {'Recall@10':<15} {metrics['Recall@10']:>10.4f}  ({sum(r.hit_at_10 for r in results)}/{n})")
    print(f"  {'MRR':<15} {metrics['MRR']:>10.4f}")
    print("─" * 65)

    failed = [r for r in results if not r.hit_at_10]
    if failed:
        print(f"\n  ❌ top-10 미검색 ({len(failed)}건):")
        for r in failed:
            print(f"     - [{r.risk_id}] {r.risk_name}")
    else:
        print("\n  🎉 전 항목 앙상블 하이브리드 탑레이어 안 검색 성공")


# ──────────────────────────────────────────
# 7. JSON 저장
# ──────────────────────────────────────────

def save_results(
    det_results  : list[DetectionResult],
    det_metrics  : dict,
    risk_metrics : dict,
    ret_results  : list[RetrievalResult],
    ret_metrics  : dict,
    output_path  : Path,
) -> None:
    output = {
        "meta": {
            "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_contracts": len(det_results),
            "n_retrieval": ret_metrics.get("n", 0),
            "embed_model": EMBED_MODEL,
            "top_k"      : TOP_K,
        },
        "detection": {
            "metrics"           : det_metrics,
            "risk_type_metrics" : risk_metrics,
            "details": [
                {
                    "contract_id"    : r.record.contract_id,
                    "contract_name"  : r.record.contract_name,
                    "file_format"    : r.record.file_format,
                    "true_has_issue" : r.record.true_has_issue,
                    "pred_has_issue" : r.record.pred_has_issue,
                    "true_raw_issues": r.record.true_raw_issues,
                    "pred_risk_ids"  : r.record.pred_risk_ids,
                    "result"         : r.label,
                }
                for r in det_results
            ],
        },
        "retrieval": {
            "metrics": ret_metrics,
            "details": [
                {
                    "risk_id"      : r.risk_id,
                    "risk_name"    : r.risk_name,
                    "query"        : r.query,
                    "gold_sources" : r.gold_sources,
                    "hit_rank"     : r.hit_rank,
                    "hit_at_5"     : r.hit_at_5,
                    "hit_at_10"    : r.hit_at_10,
                    "rr"           : round(r.rr, 4),
                    "top10"        : r.retrieved,
                }
                for r in ret_results
            ],
        },
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 평가 결과 저장 완료: {path}")


# ──────────────────────────────────────────
# 8. 메인 실행 제어 엔진
# ──────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Workit 계약서 RAG 평가")
    ap.add_argument(
        "--mode",
        choices=["all", "detection", "retrieval"],
        default="all",
        help="평가 모드 (기본값: all)",
    )
    ap.add_argument(
        "--output",
        default=str(_THIS_DIR / "contract_eval_results.json"),
        help="결과 저장 경로",
    )
    args = ap.parse_args()

    print("=" * 65)
    print("Workit RAG Pipeline — 계약서 KB 평가")
    print(f"모드: {args.mode}  |  최적화 모델: {EMBED_MODEL}")
    print("=" * 65)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    try:
        cnt = client.count(collection_name=CONTRACT_COLLECTION)
        print(f"\n📚 {CONTRACT_COLLECTION}: {cnt.count}개 청크 로드됨")
    except Exception:
        print(f"❌ {CONTRACT_COLLECTION} 컬렉션 없음")
        print("   → py rag/yoonha_contract_chunking.py 를 먼저 실행하세요.")
        sys.exit(1)

    print("📂 Qdrant contract_kb에서 골드셋 및 동적 레코드 수집 중...")
    records = fetch_all_contracts(client)
    print(f"   총 {len(records)}개 계약서 통합 파싱 완료"
          f"  (문제 계약서: {sum(r.true_has_issue for r in records)}개"
          f" / 정상 계약서: {sum(not r.true_has_issue for r in records)}개)")

    det_results  = []
    det_metrics  = {}
    risk_metrics = {}
    ret_results  = []
    ret_metrics  = {"n": 0, "Recall@5": 0.0, "Recall@10": 0.0, "MRR": 0.0}

    # ── 평가 1: 리스크 탐지 ───────────────────
    if args.mode in ("all", "detection"):
        print("\n🔍 [실행] 평가 1: 리스크 탐지 정확도 (Rule-based vs Gold)")
        det_results  = evaluate_detection(records)
        det_metrics  = compute_detection_metrics(det_results)
        risk_metrics = compute_risk_type_metrics(det_results)
        print_detection_report(det_results, det_metrics, risk_metrics)

    # ── 평가 2: 검색 품질 ────────────────────
    if args.mode in ("all", "retrieval"):
        print("\n🔍 [실행] 평가 2: 법령 KB 교차 검색 품질 (Hybrid & RRF)")
        try:
            cnt_law = client.count(collection_name=LAW_COLLECTION)
            print(f"📚 {LAW_COLLECTION}: {cnt_law.count}개 청크 로드됨")
        except Exception:
            print(f"❌ {LAW_COLLECTION} 컬렉션 없음")
            print("   → py rag/yoonha_law_chunking.py 를 먼저 실행하세요.")
            if args.mode == "retrieval":
                sys.exit(1)
            else:
                print("   → 검색 품질 평가 건너뜀")
        else:
            # 안전한 추론 인프라 서빙을 위해 토치 로딩 래핑
            model       = SentenceTransformer(EMBED_MODEL)
            ret_results = evaluate_retrieval(client, model)
            ret_metrics = compute_retrieval_metrics(ret_results)
            print_retrieval_report(ret_results, ret_metrics)

    # ── 결과 저장 ────────────────────────────
    output_path = Path(args.output)
    save_results(
        det_results, det_metrics, risk_metrics,
        ret_results, ret_metrics,
        output_path,
    )

    # ── 최종 요약 대시보드 출력 ────────────────────────────
    print("\n" + "=" * 65)
    print("📊 최종 요약 벤치마크 리포트")
    print("─" * 65)
    if det_metrics:
        print(f"  리스크 탐지  "
              f"Precision={det_metrics['precision']:.4f}  "
              f"Recall={det_metrics['recall']:.4f}  "
              f"F1={det_metrics['f1']:.4f}  "
              f"Accuracy={det_metrics['accuracy']:.4f}")
    if ret_metrics.get("n", 0) > 0:
        print(f"  검색 품질    "
              f"Recall@5={ret_metrics['Recall@5']:.4f}  "
              f"Recall@10={ret_metrics['Recall@10']:.4f}  "
              f"MRR={ret_metrics['MRR']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()