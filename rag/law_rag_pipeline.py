"""
Workit - JoRAG 검색 모듈 (최종 확정판)
파일명: rag/law_rag_pipeline.py

법령 지식베이스(law_kb_jo)에서 텍스트 질의 하나에 대해 관련 조문을
찾아주는 순수 검색 모듈이다. 계약서 조항은 호출하는 쪽에서 이미 하나씩
분리해서 넘겨준다고 가정한다 — 이 모듈은 그 텍스트를 어디서 어떻게
쪼갰는지 몰라도 되고, 질의 텍스트 하나를 받아 검색만 한다.

확정된 하이퍼파라미터:
    alpha=0.7, rrf_k=20, fetch_k=50, rerank_k=10, reranker=bge-reranker-v2-m3 (ON), top_k=2

스코어 threshold: min_score=0.8 (score_threshold_check 근거, alpha=0.7 조건 — 재검증 필요)
    필터링 후 후보가 0개면 fallback으로 top_k개를 그대로 반환한다(1등·2등이 근소하게
    갈려 threshold를 걸치는 경계 케이스에서 상위 후보 전체가 사라지는 것을 방지).

필요 패키지: FlagEmbedding==1.3.2, transformers==4.44.2 (버전 어긋나면 임베더/리랭커 중 하나 깨짐)
"""

from __future__ import annotations

from dataclasses import dataclass

import os
import torch
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

# ── 상수 ──────────────────────────────────────────────────────────

QDRANT_HOST = os.environ.get('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.environ.get('QDRANT_PORT', '6333'))
COLLECTION_JO = "law_kb_jo"

EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# fp16은 GPU 전용 — CPU 환경에서는 자동으로 꺼진다.
USE_FP16 = torch.cuda.is_available()

DEFAULT_ALPHA = 0.7
DEFAULT_RRF_K = 20
DEFAULT_FETCH_K = 50
DEFAULT_RERANK_K = 10
DEFAULT_TOP_K = 2
DEFAULT_MIN_SCORE = 0.8  # rerank_score 기준. reranker OFF(rrf_score)에는 적용 안 함


# ── 데이터 모델 ────────────────────────────────────────────────────


@dataclass
class LawRef:
    """검색된 법령 조문 하나."""

    chunk_id: str
    law_name: str
    article_id: str
    text: str
    score: float


# ── 모델 로더 ─────────────────────────────────────────────────────


def get_qdrant_client(host: str = QDRANT_HOST, port: int = QDRANT_PORT) -> QdrantClient:
    return QdrantClient(host=host, port=port)


def load_embed_model() -> BGEM3FlagModel:
    return BGEM3FlagModel(EMBED_MODEL_NAME, use_fp16=USE_FP16)


def load_reranker() -> FlagReranker:
    return FlagReranker(RERANKER_MODEL_NAME, use_fp16=USE_FP16)


# ── 검색 파이프라인 ────────────────────────────────────────────────


def _hybrid_search(
    query_text: str,
    client: QdrantClient,
    model: BGEM3FlagModel,
    fetch_k: int,
    alpha: float,
    rrf_k: int,
) -> list[dict]:
    """dense(의미 유사도) + sparse(키워드 매칭)를 RRF로 결합해 상위 fetch_k개를 가져온다."""
    vectors = model.encode([query_text], return_dense=True, return_sparse=True)
    dense_vec = vectors["dense_vecs"][0]
    sparse_weights = vectors["lexical_weights"][0]
    sparse_vec = SparseVector(
        indices=[int(k) for k in sparse_weights.keys()],
        values=[float(v) for v in sparse_weights.values()],
    )

    results = client.query_points(
        collection_name=COLLECTION_JO,
        prefetch=[
            Prefetch(query=dense_vec.tolist(), using="dense", limit=fetch_k),
            Prefetch(query=sparse_vec, using="sparse", limit=fetch_k),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=fetch_k,
        with_payload=True,
    )

    return [
        {
            "chunk_id": point.payload["chunk_id"],
            "law_name": point.payload["law_name"],
            "article_id": point.payload["article_id"],
            "text": point.payload["text"],
            "rrf_score": point.score,
        }
        for point in results.points
    ]


def _rerank(
    query_text: str,
    candidates: list[dict],
    reranker: FlagReranker,
    rerank_k: int,
) -> list[dict]:
    """상위 rerank_k개만 쿼리와 1:1 비교해 재채점하고 점수 내림차순으로 정렬한다."""
    top_candidates = candidates[:rerank_k]
    pairs = [[query_text, c["text"]] for c in top_candidates]
    scores = reranker.compute_score(pairs, normalize=True)

    for candidate, score in zip(top_candidates, scores):
        candidate["rerank_score"] = score

    return sorted(top_candidates, key=lambda c: c["rerank_score"], reverse=True)


def _apply_min_score(
    candidates: list[dict],
    score_key: str,
    min_score: float | None,
) -> list[dict]:
    """min_score 미만 후보 제거 (None이면 미적용)."""
    if min_score is None:
        return candidates
    return [c for c in candidates if c[score_key] >= min_score]


def _build_law_refs(
    candidates: list[dict], top_k: int, use_reranker: bool
) -> list[LawRef]:
    score_key = "rerank_score" if use_reranker else "rrf_score"
    return [
        LawRef(
            chunk_id=c["chunk_id"],
            law_name=c["law_name"],
            article_id=c["article_id"],
            text=c["text"],
            score=c[score_key],
        )
        for c in candidates[:top_k]
    ]


# ── 공개 API ──────────────────────────────────────────────────────


def search_jo(
    query_text: str,
    client: QdrantClient,
    model: BGEM3FlagModel,
    reranker: FlagReranker | None = None,
    use_reranker: bool = True,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = DEFAULT_ALPHA,
    rrf_k: int = DEFAULT_RRF_K,
    fetch_k: int = DEFAULT_FETCH_K,
    rerank_k: int = DEFAULT_RERANK_K,
    min_score: float | None = DEFAULT_MIN_SCORE,
) -> list[LawRef]:
    """
    임의의 텍스트 질의에 대해 관련 법령 조문을 검색한다. 이 모듈의 유일한 진입점.
    흐름: 하이브리드 검색 → [옵션] 리랭크 → min_score 미만 제거 → 상위 top_k개 반환.

    min_score 미만인 후보는 top_k 안에 들어도 제외되므로 반환 개수가 top_k보다
    적을 수 있다. 단, 필터링 후 0개가 되면 fallback으로 top_k개를 그대로 반환한다
    (min_score 미달이어도 "관련 조문 없음"보다는 최선의 후보를 보여주는 쪽을 택함
    — 1등·2등이 근소하게 갈린 경계 케이스에서 상위 후보가 통째로 사라지는 것도 방지.
    호출 쪽에서 score로 신뢰도를 다시 판단할 것).
    끄려면 min_score=None. use_reranker=False일 때는 rrf_score 스케일이 달라
    적용하지 않는다.
    """
    candidates = _hybrid_search(query_text, client, model, fetch_k, alpha, rrf_k)

    if use_reranker:
        if reranker is None:
            raise ValueError("use_reranker=True인데 reranker가 전달되지 않았습니다.")
        candidates = _rerank(query_text, candidates, reranker, rerank_k)
        filtered = _apply_min_score(candidates, "rerank_score", min_score)
        candidates = filtered if filtered else candidates[:top_k]

    return _build_law_refs(candidates, top_k, use_reranker)


# ── 실행 진입점 ────────────────────────────────────────────────────
# 호출 예시: 조항 텍스트 하나를 그대로 넘기면 관련 조문을 찾아준다.

SAMPLE_CLAUSES = [
    "제15조(지체상금) 을은 본 계약에서 정한 준공기한까지 용역을 완료하지 못한 경우, "
    "지체일수 1일당 계약금액의 1천분의 3에 해당하는 금액을 지체상금으로 갑에게 지급하여야 하며, "
    "지체상금이 계약금액의 100분의 30을 초과하는 경우 갑은 계약을 해제할 수 있다.",
    "제22조(계약보증금 및 하자보수보증금) 을은 계약체결일로부터 7일 이내에 계약금액의 100분의 15에 해당하는 "
    "계약보증금을 현금 또는 보증서로 납부하여야 하며, 검사 완료 후 하자담보책임기간 동안 "
    "계약금액의 100분의 5에 해당하는 하자보수보증금을 별도로 예치하여야 한다.",
    "제31조(재하도급 및 기술자료 제공) 을은 갑의 서면 승인 없이 본 용역의 주요 부분을 제3자에게 재하도급할 수 없으며, "
    "부득이하게 하도급이 필요한 경우에도 갑이 요구하는 즉시 하도급 계약서 및 관련 기술자료 일체를 제출하여야 한다.",
    # LCA(공공계약법) 매칭 예상 — 입찰담합
    "제8조(입찰담합의 금지) 을은 본 입찰에 참가함에 있어 다른 입찰참가자와 사전에 협의하여 "
    "입찰가격, 낙찰자 또는 낙찰순위를 미리 정하는 등의 담합행위를 하여서는 아니 되며, "
    "이를 위반한 사실이 확인되는 경우 갑은 낙찰자 결정을 취소하고 관계 법령에 따라 부정당업자로 제재할 수 있다.",
    # PIPA(개인정보보호법) 매칭 예상 — 개인정보 처리
    "제18조(개인정보의 처리) 을은 본 용역 수행 과정에서 수집한 개인정보를 계약 목적 범위 내에서만 처리하여야 하며, "
    "갑의 사전 서면 동의 없이 제3자에게 제공하거나 목적 외로 이용할 수 없다. "
    "을은 계약 종료 시 보유 중인 개인정보를 지체 없이 파기하여야 한다.",
]


def main() -> None:
    client = get_qdrant_client()
    model = load_embed_model()
    reranker = load_reranker()

    for clause in SAMPLE_CLAUSES:
        law_refs = search_jo(clause, client, model, reranker)

        print(f"\n질의(계약서 조항): {clause}")
        if not law_refs:
            print("  (결과 없음)")
        for ref in law_refs:
            flag = (
                " [주의: min_score 미달, fallback]"
                if ref.score < DEFAULT_MIN_SCORE
                else ""
            )
            print(
                f"  - {ref.chunk_id} ({ref.law_name} {ref.article_id}) score={ref.score:.4f}{flag}"
            )
            preview = ref.text if len(ref.text) <= 150 else ref.text[:150] + "..."
            print(f"    {preview}")


if __name__ == "__main__":
    main()