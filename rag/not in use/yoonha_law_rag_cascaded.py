"""
Workit - 계층형(Cascaded) RAG 파이프라인
파일명: rag/yoonha_law_rag_cascaded.py

병렬 버전(yoonha_law_rag.py의 JoRAG/HoRAG 독립 검색)과는 완전히 별개로
관리되는 계층형(coarse-to-fine) variant.

흐름:
  1단계 (coarse) : law_kb_jo_fixedid에서 조 단위로 넓게 검색해 상위
                    jo_top_k개 조만 후보로 추린다 (JoRAG와 동일한 하이브리드
                    검색 로직 재사용 — _hybrid_search를 그대로 import).
  2단계 (fine)   : law_kb_ho_fixedid에서 호/목/세목 단위로 검색한 뒤,
                    각 후보의 chunk_id를 derive_jo_id()로 조 단위 id로 역산해서
                    1단계 후보 조 안에 있는 것만 Python 사이드에서 남긴다
                    (payload의 parent_chunk_id 필드는 HO 컬렉션 자기 자신을
                    가리키는 값이라 이 용도로 쓸 수 없음 — derive_jo_id 참고).
  3단계          : (옵션) cross_refs 확장 — 명시적 인용은 1단계 필터 밖이어도
                    살려서 확장한다 (조 단위 후보에 안 걸렸다고 인용 관계까지
                    끊어버리면 너무 공격적인 필터링이라고 판단).
  4단계          : parent fetch로 최종 출력을 조 텍스트로 통일 (병렬 버전과 동일).

주의:
  - 1단계 recall이 2단계 recall의 상한선이다. 1단계에서 정답 조가 top-k
    밖으로 밀리면 2단계에서 아무리 잘 검색해도 복구 불가능하다. 그래서
    jo_top_k는 반드시 sweep 대상에 넣어야 하고, 이 모듈은 "1단계에서
    정답 조가 살아남았는가"를 별도로 진단할 수 있는 get_stage1_jo_candidates도
    노출한다 (yoonha_rag_eval_cascaded.py에서 stage1_recall 지표로 사용).
  - PYG(예규)는 조가 없어 jo-level이 항 단위로 anchor된다. derive_jo_id()가
    PYG는 앞 5토큰({prefix}_{장}_{절}_{조=0}_{항})을, 일반 법령은 앞 4토큰을
    jo_id로 역산하도록 이미 분기 처리돼 있어서 필터링 자체는 정상 동작하지만,
    "항 단위로 넓게 검색 → 그 항에 속하는 호/목/세목만 좁혀서 검색"이라
    병렬 버전의 "조 단위 필터"와는 의미가 다르므로 평가 시 PYG만 따로
    recall을 봐두는 걸 권장한다.
  - 이 모듈은 yoonha_law_rag.py의 함수 이름을 재사용하지 않는다. 공용 유틸
    (get_vectors, CrossEncoderReranker, SweepCache, LawRef, ClauseResult,
    chunk_contract, _hybrid_search, _rerank, _fetch_parent_texts,
    _expand_with_cross_refs, _build_law_refs)은 그대로 import해서 쓰고,
    새로 정의하는 함수는 전부 *_cascaded 접미사를 붙였다.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from FlagEmbedding import BGEM3FlagModel

from yoonha_law_rag import (
    COLLECTION_JO,
    COLLECTION_HO,
    DEFAULT_FETCH_K,
    DEFAULT_RERANK1_K,
    DEFAULT_RERANK2_K,
    DEFAULT_TOP_K,
    DEFAULT_ALPHA,
    QDRANT_HOST,
    QDRANT_PORT,
    CrossEncoderReranker,
    LawRef,
    ClauseResult,
    SweepCache,
    chunk_contract,
    get_vectors,
    derive_jo_id,
    load_laws_ref,
    load_embed_model,
    load_rerankers,
    _hybrid_search,
    _rerank,
    _fetch_parent_texts,
    _expand_with_cross_refs,
    _build_law_refs,
)

# 계층형 전용 기본값 — 1단계(조 단위)에서 몇 개 조를 후보로 남길지.
# 너무 작으면 1단계에서 정답 조가 걸러져서 2단계가 복구 불가능해지고,
# 너무 크면 사실상 병렬 버전과 차이가 없어진다. sweep으로 찾아야 한다.
DEFAULT_JO_TOP_K = 20


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1단계: 조 단위 후보 추출
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_stage1_jo_candidates(
    clause_text: str,
    client     : QdrantClient,
    model      : BGEM3FlagModel,
    alpha      : float = DEFAULT_ALPHA,
    fetch_k    : int   = DEFAULT_FETCH_K,
    jo_top_k   : int   = DEFAULT_JO_TOP_K,
    cache      : SweepCache | None = None,
) -> list[str]:
    """
    1단계: law_kb_jo_fixedid에서 조 단위로 넓게 검색해 상위 jo_top_k개
    조의 chunk_id만 반환한다. _hybrid_search를 그대로 재사용하므로
    JoRAG(병렬 버전)의 캐시(raw_search)와도 공유된다.

    sweep 스크립트에서 진단용으로 따로 호출해도, 이미 search_cascaded 내부에서
    같은 캐시 키로 호출된 적이 있으면 cache hit이라 추가 비용이 거의 없다.
    """
    candidates = _hybrid_search(clause_text, client, model, COLLECTION_JO, fetch_k, alpha, cache)
    return [c["chunk_id"] for c in candidates[:jo_top_k]]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2단계: jo_id로 제한된 ho 후보 필터링 (Python 사이드)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _hybrid_search_ho_filtered(
    clause_text       : str,
    client            : QdrantClient,
    model             : BGEM3FlagModel,
    fetch_k           : int,
    alpha             : float,
    allowed_parent_ids: list[str],
    cache             : SweepCache | None = None,
) -> list[dict]:
    """
    2단계: law_kb_ho_fixedid에서 검색한 뒤, 각 후보의 chunk_id를 derive_jo_id()로
    조 단위 id로 역산해서 allowed_parent_ids(1단계 후보 조) 안에 있는 것만 남긴다.

    처음에는 Qdrant Filter(parent_chunk_id MatchAny)로 서버 사이드 필터링을
    시도했었는데, 실제 데이터를 보면 payload의 parent_chunk_id 필드가 JO
    컬렉션이 아니라 HO 컬렉션 자기 자신(조 단위로 롤업된 다른 ho chunk 또는
    중간 단계인 호/목)을 가리키고 있어서 JO id와 절대 일치하지 않았다
    (검증 결과 0% 매치 — 그래서 이전 버전은 항상 빈 결과만 반환했음).
    derive_jo_id()가 chunk_id 문자열 자체에서 직접 역산하는 방식이라 이 필드에
    의존하지 않고, ho 7640개 전체에서 100% 정확하게 검증됐다.

    부가 효과: _hybrid_search를 필터 없이 그대로 재사용하므로, jo_top_k가
    달라져도 raw_search 캐시가 (collection, clause_text, fetch_k) 키 하나로
    공유된다 — 이전 버전(필터 지문을 캐시 키에 포함하던 방식)보다 캐시 재사용률이
    오히려 더 좋아졌다.
    """
    candidates = _hybrid_search(clause_text, client, model, COLLECTION_HO, fetch_k, alpha, cache)
    allowed_set = set(allowed_parent_ids)
    return [c for c in candidates if derive_jo_id(c["chunk_id"]) in allowed_set]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 전체 계층형 검색
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _search_cascaded(
    clause_text  : str,
    client       : QdrantClient,
    model        : BGEM3FlagModel,
    laws_ref     : dict[str, dict],
    reranker1    : CrossEncoderReranker | None = None,
    reranker2    : CrossEncoderReranker | None = None,
    use_reranker1: bool  = False,
    use_reranker2: bool  = False,
    use_cross_refs: bool = True,
    top_k        : int   = DEFAULT_TOP_K,
    alpha        : float = DEFAULT_ALPHA,
    fetch_k      : int   = DEFAULT_FETCH_K,
    rerank1_k    : int   = DEFAULT_RERANK1_K,
    rerank2_k    : int   = DEFAULT_RERANK2_K,
    jo_top_k     : int   = DEFAULT_JO_TOP_K,
    cache        : SweepCache | None = None,
) -> list[LawRef]:
    """
    1단계(JoRAG top-k) → 2단계(그 조들로 제한된 HoRAG) → cross-ref 확장
    → parent fetch(조 텍스트 통일) 순서로 진행하는 계층형 검색.
    """
    jo_candidates = _hybrid_search(clause_text, client, model, COLLECTION_JO, fetch_k, alpha, cache)
    allowed_parent_ids = [c["chunk_id"] for c in jo_candidates[:jo_top_k]]

    if not allowed_parent_ids:
        return []

    candidates = _hybrid_search_ho_filtered(
        clause_text, client, model, fetch_k, alpha, allowed_parent_ids, cache,
    )

    if use_reranker1 and reranker1 and candidates:
        candidates = _rerank(clause_text, candidates, reranker1, rerank1_k, "cascaded_r1", cache)

    if use_cross_refs:
        candidates = _expand_with_cross_refs(candidates, client, cache)

    # parent fetch: 호/목/세목 → 조 텍스트로 교체 (최종 출력 단위 통일, 병렬 버전과 동일)
    candidates = _fetch_parent_texts(candidates, client, parent_collection=COLLECTION_JO, cache=cache)

    if use_reranker2 and reranker2 and candidates:
        candidates = _rerank(clause_text, candidates, reranker2, rerank2_k, "cascaded_r2", cache)

    return _build_law_refs(candidates, laws_ref, top_k, with_xref=use_cross_refs)


def review_contract_cascaded(
    contract_text: str,
    client       : QdrantClient,
    model        : BGEM3FlagModel,
    laws_ref     : dict[str, dict] | None = None,
    reranker1    : CrossEncoderReranker | None = None,
    reranker2    : CrossEncoderReranker | None = None,
    use_reranker1: bool  = False,
    use_reranker2: bool  = False,
    use_cross_refs: bool = True,
    top_k        : int   = DEFAULT_TOP_K,
    alpha        : float = DEFAULT_ALPHA,
    fetch_k      : int   = DEFAULT_FETCH_K,
    rerank1_k    : int   = DEFAULT_RERANK1_K,
    rerank2_k    : int   = DEFAULT_RERANK2_K,
    jo_top_k     : int   = DEFAULT_JO_TOP_K,
    cache        : SweepCache | None = None,
) -> list[ClauseResult]:
    """계층형 RAG 메인 인터페이스 (review_contract_jo/ho와 동일한 사용 패턴)."""
    if laws_ref is None:
        laws_ref = load_laws_ref()

    clauses = chunk_contract(contract_text)
    results : list[ClauseResult] = []
    print(f"[Cascaded] 총 {len(clauses)}개 청크 검색 중...")

    for i, clause in enumerate(clauses, 1):
        print(f"  [{i}/{len(clauses)}] {clause['clause_number']} ...", end="\r")

        law_refs = _search_cascaded(
            clause["clause_text"], client, model, laws_ref,
            reranker1, reranker2, use_reranker1, use_reranker2, use_cross_refs,
            top_k, alpha, fetch_k, rerank1_k, rerank2_k, jo_top_k, cache,
        )
        categories = list(dict.fromkeys(r.category for r in law_refs if r.category))

        results.append(ClauseResult(
            clause_number=clause["clause_number"],
            clause_text  =clause["clause_text"],
            law_refs     =law_refs,
            categories   =categories,
        ))

    print(f"\n[Cascaded] ✅ 완료")
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 편의: 단일 조항 검색 (sweep 스크립트에서 직접 호출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def search_cascaded(clause_text: str, client: QdrantClient, model: BGEM3FlagModel,
                     laws_ref: dict, reranker1=None, reranker2=None,
                     use_reranker1=False, use_reranker2=False, use_cross_refs=True,
                     top_k=DEFAULT_TOP_K, alpha=DEFAULT_ALPHA, fetch_k=DEFAULT_FETCH_K,
                     rerank1_k=DEFAULT_RERANK1_K, rerank2_k=DEFAULT_RERANK2_K,
                     jo_top_k=DEFAULT_JO_TOP_K,
                     cache: SweepCache | None = None) -> list[LawRef]:
    return _search_cascaded(clause_text, client, model, laws_ref, reranker1, reranker2,
                             use_reranker1, use_reranker2, use_cross_refs,
                             top_k, alpha, fetch_k, rerank1_k, rerank2_k, jo_top_k, cache)