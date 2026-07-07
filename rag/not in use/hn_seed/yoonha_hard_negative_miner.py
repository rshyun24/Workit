"""
yoonha_hard_negative_miner.py
==============================
Hard Negative Mining for Workit law_kb evaluation set

[ 목적 ]
현재 Recall@5/10 = 1.0인데, 이게 평가셋이 너무 쉬운 건지 확인하기 위해
"헷갈리기 쉬운 오답(hard negative)"을 평가셋에 추가합니다.
모델 가중치는 건드리지 않고, 평가 기준만 더 엄격하게 만드는 작업입니다.

[ 전체 흐름 ]
씨드 JSON (clause_text + ground_truth)
    │
    ├─ Step 1: BM25로 어휘적으로 유사한 청크 자동 추출
    │           → "단어가 비슷해서 헷갈릴 수 있는" 후보
    │
    ├─ Step 2: BGE-M3 + Qdrant로 의미적으로 유사한 청크 추출
    │           → "모델이 실제로 헷갈린" 후보 (가장 가치 있음)
    │
    └─ Step 3: 수동 검수 (자동으로 못 잡는 준용·역방향 케이스)

[ hard negative 4가지 타입 ]
타입 1 동일주제_다른법령 : 주제는 같은데 적용 법령이 다름 (지방계약법 vs 소프트웨어진흥법)
타입 2 인접조문          : 같은 법에서 바로 앞뒤 조항 (±2조 이내)
타입 3 준용_역방향_후보  : A가 B를 준용할 때 B를 쿼리하면 A가 올라오는 케이스 (수동 검수 필요)
타입 4 숫자_변형         : 같은 조문인데 항·호만 다름 (1천분의 3 vs 1천분의 5)

[ chunk_id 형식 ]
{LAW_ABBR}_{조}[_의N][_{항}][_{호}]   예: LCAR_75_1_2
항 없이 바로 호인 경우: {LAW_ABBR}_{조}_0_{호}   예: LCA_23_0_1

법령 약어 (2026-06 기준):
    LCA    지방계약법
    LCAE   지방계약법 시행령
    LCAR   지방계약법 시행규칙
    SWPA   소프트웨어 진흥법
    SWPAE  소프트웨어 진흥법 시행령
    LARA   지방회계법
    LARAE  지방회계법 시행령
    PYG    지방자치단체 용역계약 일반조건 (예규367호)
    PPMA   공유재산법
    PPMAE  공유재산법 시행령
    PIPA   개인정보보호법
    PIPAE  개인정보보호법 시행령
"""

import json
import re
import os
from pathlib import Path
from typing import Optional
from collections import defaultdict

from rank_bm25 import BM25Okapi          # pip install rank-bm25
from FlagEmbedding import BGEM3FlagModel  # pip install FlagEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
)


# ═══════════════════════════════════════════════════════════════
# CONFIG — 경로·파라미터 한 곳에서 관리
# ═══════════════════════════════════════════════════════════════

# Qdrant Docker 연결 설정
# Docker가 실행 중이어야 함: docker run -p 6333:6333 qdrant/qdrant
QDRANT_HOST   = "localhost"
QDRANT_PORT   = 6333

# ★ law_kb_ho: 호 단위 컬렉션 (BM25 코퍼스 + Qdrant 검색 대상)
#   hard negative는 호 단위 검색 기준으로 생성하므로 law_kb_ho 사용
COLLECTION    = "law_kb_ho"

BGE_MODEL_ID  = "BAAI/bge-m3"              # HuggingFace 모델 ID
LAW_REFS_PATH = "./data/laws_ref.json"      # chunk_id → 법령 텍스트 매핑
SEED_PATH     = "./data/hn_seed/hard_negative_seed.json"    # 입력: 씨드 쿼리 목록
OUTPUT_PATH   = "./data/hn_seed/hard_negatives_output.json" # 출력: mining 결과

# BM25 후보 풀: 상위 20개 뽑아서 그 중 ground truth 제외
TOP_K_BM25    = 20
# Qdrant 후보 풀: 상위 20개 뽑아서 상위 10개 중 false positive 추출
TOP_K_QDRANT  = 20
# 쿼리당 최대 hard negative 수
MAX_HN_BM25   = 5
MAX_HN_MODEL  = 5

# chunk_id 앞 약어 → 한국어 법령명
# 타입 분류 시 reason 메시지에 활용
# ★ 2026-06 기준 전체 법령 약어 목록으로 업데이트
LAW_ABBR_KR = {
    "LCA":   "지방계약법",
    "LCAE":  "지방계약법_시행령",
    "LCAR":  "지방계약법_시행규칙",
    "SWPA":  "소프트웨어진흥법",
    "SWPAE": "소프트웨어진흥법_시행령",
    "LARA":  "지방회계법",
    "LARAE": "지방회계법_시행령",
    "PYG":   "지방자치단체용역계약일반조건",
    "PPMA":  "공유재산법",
    "PPMAE": "공유재산법_시행령",
    "PIPA":  "개인정보보호법",
    "PIPAE": "개인정보보호법_시행령",
}


# ═══════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════════════════════════════

def load_law_refs(path: str) -> dict:
    """
    law_refs.json 로드.
    구조: { "LCAR_75": { "chunk_id": ..., "text": ..., "category": ... }, ... }
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_seed(path: str) -> list[dict]:
    """
    씨드 쿼리 JSON 로드.
    각 항목은 최소한 아래 필드를 가져야 합니다:
      - query_id     : "Q001"
      - clause_text  : 검토 대상 계약 조항 텍스트
      - ground_truth : 정답 chunk_id 리스트 (예: ["LCAR_75", "LCAE_90_3"])
      - category     : 카테고리 문자열 (예: "지체상금")
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_chunks_from_qdrant(client: QdrantClient, collection: str) -> tuple[list[str], list[str]]:
    """
    Qdrant law_kb_ho 컬렉션에서 전체 청크를 페이지네이션으로 가져옵니다.
    BM25 코퍼스 구성에 사용됩니다.

    Qdrant scroll API: offset=None에서 시작해서 반환된 offset이 None이면 끝.
    벡터는 BM25에 불필요하므로 with_vectors=False로 생략해 속도를 높입니다.

    Returns:
        chunk_ids : Qdrant 페이로드의 chunk_id 값 목록
        texts     : 각 청크의 본문 텍스트 목록 (chunk_ids와 인덱스 대응)
    """
    chunk_ids, texts = [], []
    offset = None

    while True:
        results, offset = client.scroll(
            collection_name=collection,
            limit=500,          # 한 번에 500개씩 가져옴
            offset=offset,
            with_payload=True,  # chunk_id, text 등 메타데이터 필요
            with_vectors=False, # 벡터는 불필요 → 생략으로 속도↑
        )
        for r in results:
            # 페이로드에 chunk_id 없으면 Qdrant 내부 UUID를 fallback으로 사용
            chunk_ids.append(r.payload.get("chunk_id", r.id))
            texts.append(r.payload.get("text", r.payload.get("chunk_text", "")))

        if offset is None:  # 마지막 페이지
            break

    print(f"[load] Qdrant에서 {len(chunk_ids)}개 청크 로드 완료")
    return chunk_ids, texts


# ═══════════════════════════════════════════════════════════════
# 2. Hard Negative 타입 분류 헬퍼
# ═══════════════════════════════════════════════════════════════

def parse_chunk_id(chunk_id: str) -> dict:
    """
    chunk_id를 구성 요소로 분해합니다.

    예시:
        "LCAR_75"       → {"abbr": "LCAR", "jo": "75",  "hang": "",  "ho": ""}
        "LCAR_75_1_2"   → {"abbr": "LCAR", "jo": "75",  "hang": "1", "ho": "2"}
        "LCA_30_의2_1"  → {"abbr": "LCA",  "jo": "30",  "hang": "의2","ho": "1"}
        "LCA_23_0_1"    → {"abbr": "LCA",  "jo": "23",  "hang": "0", "ho": "1"}
                           (항 없는 호: hang=0)

    PYG(용역계약일반조건)는 "PYG_9_8_1_다" 형태로 숫자+한글이 혼합되므로
    abbr 외 나머지는 그대로 문자열로 유지합니다.
    """
    parts = chunk_id.split("_")
    return {
        "abbr": parts[0] if len(parts) > 0 else "",
        "jo":   parts[1] if len(parts) > 1 else "",
        "hang": parts[2] if len(parts) > 2 else "",
        "ho":   parts[3] if len(parts) > 3 else "",
    }


def classify_hn_type(gt_id: str, hn_id: str) -> tuple[str, str]:
    """
    ground truth(gt_id)와 hard negative 후보(hn_id)를 비교해
    어떤 타입의 hard negative인지 자동으로 분류합니다.

    분류 우선순위:
      1순위 — 법령 약어(abbr)가 다르면 → 동일주제_다른법령
      2순위 — 같은 법령, 조 번호 차이 ≤ 2 → 인접조문
      3순위 — 같은 법령, 같은 조, 항/호만 다름 → 숫자_변형
      4순위 — 위 3가지에 해당 안 됨 → 준용_역방향_후보 (수동 검수 필요)

    주의:
      타입 3(준용 역방향)은 청크 텍스트 분석 없이는 자동 판별 불가능하므로
      fallback 레이블로 처리하고 needs_manual_review에 분리합니다.

    Returns:
        (type_label, reason_string)
    """
    gt = parse_chunk_id(gt_id)
    hn = parse_chunk_id(hn_id)

    # ── 타입 1: 다른 법령 ──────────────────────────────────────
    if gt["abbr"] != hn["abbr"]:
        gt_law = LAW_ABBR_KR.get(gt["abbr"], gt["abbr"])
        hn_law = LAW_ABBR_KR.get(hn["abbr"], hn["abbr"])
        # 조 번호까지 같으면 더 강력한 hard negative
        if gt["jo"] == hn["jo"]:
            return ("동일주제_다른법령", f"{hn_law} {hn['jo']}조 — 동일 조번호, 다른 법령")
        return ("동일주제_다른법령", f"{gt_law} → {hn_law}, 유사 주제 이종 법령")

    # ── 이하 같은 법령 내 ──────────────────────────────────────

    # 조 번호를 정수로 변환해서 거리 계산
    # 변환 실패(예: "30_의2" 같은 특수 형태)는 999로 처리 → 타입 2에서 제외
    try:
        jo_diff = abs(int(hn["jo"]) - int(gt["jo"]))
    except ValueError:
        jo_diff = 999

    # ── 타입 2: 인접 조문 ──────────────────────────────────────
    # 같은 조는 타입 4에서 처리하므로 jo_diff > 0 조건 추가
    if jo_diff <= 2 and gt["jo"] != hn["jo"]:
        return ("인접조문", f"같은 법 {hn['jo']}조 — 정답 {gt['jo']}조와 인접")

    # ── 타입 4: 항·호만 다름 (숫자 표현 변형) ──────────────────
    # 조는 같은데 항이나 호가 다른 경우
    # Dense 임베딩이 가장 못 잡는 유형 (조문 본문이 매우 유사)
    if gt["jo"] == hn["jo"] and (gt["hang"] != hn["hang"] or gt["ho"] != hn["ho"]):
        return ("숫자_변형", f"동일 조({gt['jo']}조) 내 항/호 다름 — 숫자 표현 변형")

    # ── 타입 3 fallback: 준용 역방향 후보 ─────────────────────
    return ("준용_역방향_후보", f"동일 법령 내 {hn_id} — 수동 검수 필요")


# ═══════════════════════════════════════════════════════════════
# 3. Step 1 — BM25 Hard Negative 생성
# ═══════════════════════════════════════════════════════════════

def build_bm25_hard_negatives(
    seed_items : list[dict],
    chunk_ids  : list[str],
    texts      : list[str],
    top_k      : int = TOP_K_BM25,
    max_hn     : int = MAX_HN_BM25,
) -> list[dict]:
    """
    BM25(어휘 기반 TF-IDF 계열)로 각 씨드 쿼리와 어휘가 유사한 청크를 찾습니다.

    BM25를 쓰는 이유:
      - BGE-M3(dense)는 의미 유사도 기반이라 "1천분의 3"과 "1천분의 5"를
        거의 구분 못 함 (숫자 변형 타입 hard negative를 못 잡음)
      - BM25는 정확한 단어 매칭 기반이라 이런 숫자·용어 변형을 잘 잡음
      - 두 방법의 보완적 사용 → hard negative 다양성 확보

    처리 흐름:
      1. Qdrant 전체 청크 텍스트를 공백 기준으로 토크나이즈해서 BM25 코퍼스 구성
      2. 각 씨드 clause_text를 쿼리로 BM25 점수 계산
      3. 상위 top_k개 중 ground_truth에 없는 것을 hard negative 후보로 선택
      4. 타입 분류 후 최대 max_hn개 저장
    """
    print("\n[Step 1] BM25 Hard Negative 생성 중...")

    def tokenize(text: str) -> list[str]:
        """
        한국어 간단 토크나이저.
        특수문자 제거 후 공백 분리.
        형태소 분석기(KoNLPy) 없이도 동작하지만
        형태소 분석기 쓰면 더 정확해짐.
        """
        text = re.sub(r"[^\w\s]", " ", text)  # 특수문자 → 공백
        return text.split()

    # 전체 청크 텍스트를 토크나이즈해서 BM25 코퍼스 구성
    corpus_tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus_tokenized)

    results = []
    for item in seed_items:
        query_tokens = tokenize(item["clause_text"])
        scores = bm25.get_scores(query_tokens)  # numpy array

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        ground_truth_set = set(item.get("ground_truth", []))
        hard_negs = []

        for idx in top_indices:
            cid = chunk_ids[idx]

            if cid in ground_truth_set:
                continue  # 정답은 hard negative 아님

            hn_type, reason = classify_hn_type(
                list(ground_truth_set)[0] if ground_truth_set else "",
                cid,
            )
            hard_negs.append({
                "chunk_id":   cid,
                "type":       hn_type,
                "reason":     reason,
                "source":     "bm25",
                "bm25_score": round(float(scores[idx]), 4),
            })

            if len(hard_negs) >= max_hn:
                break

        result_item = dict(item)
        result_item["hard_negatives_bm25"] = hard_negs
        results.append(result_item)
        print(f"  [{item['query_id']}] BM25 hard neg {len(hard_negs)}개 생성")

    return results


# ═══════════════════════════════════════════════════════════════
# 4. Step 2 — BGE-M3 Qdrant False Positive 추출
# ═══════════════════════════════════════════════════════════════

def encode_sparse(model: BGEM3FlagModel, text: str) -> SparseVector:
    """
    BGE-M3의 sparse(SPLADE) 인코딩.
    동일 token_id 중복 시 가중치 합산 (Qdrant unique indices 제약 준수).
    """
    out = model.encode(
        [text],
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    lexical = out["lexical_weights"][0]

    id_weight: dict[int, float] = defaultdict(float)
    for token_str, weight in lexical.items():
        token_id = model.tokenizer.convert_tokens_to_ids(token_str)
        id_weight[token_id] += float(weight)

    indices = list(id_weight.keys())
    values  = [id_weight[i] for i in indices]
    return SparseVector(indices=indices, values=values)


def encode_dense(model: BGEM3FlagModel, text: str) -> list[float]:
    """
    BGE-M3의 dense 인코딩.
    출력: 1024차원 float 리스트.
    """
    out = model.encode(
        [text],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return out["dense_vecs"][0].tolist()


def qdrant_hybrid_search(
    client     : QdrantClient,
    model      : BGEM3FlagModel,
    query_text : str,
    top_k      : int = TOP_K_QDRANT,
) -> list[dict]:
    """
    BGE-M3 hybrid RRF 검색 (yoonha_contract_rag.py와 동일한 방식).

    dense + sparse prefetch → RRF fusion.

    Returns:
        [{"chunk_id": ..., "score": ...}]  RRF 점수 순 정렬
    """
    dense_vec  = encode_dense(model, query_text)
    sparse_vec = encode_sparse(model, query_text)

    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using="dense",
                limit=top_k * 2,
            ),
            Prefetch(
                query=SparseVector(
                    indices=list(sparse_vec.indices),
                    values=list(sparse_vec.values),
                ),
                using="sparse",
                limit=top_k * 2,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "chunk_id": r.payload.get("chunk_id", str(r.id)),
            "score":    round(r.score, 6),
        }
        for r in results.points
    ]


def build_model_hard_negatives(
    items_with_bm25 : list[dict],
    client          : QdrantClient,
    model           : BGEM3FlagModel,
    top_k           : int = TOP_K_QDRANT,
    max_hn          : int = MAX_HN_MODEL,
) -> list[dict]:
    """
    BGE-M3 + Qdrant hybrid 검색으로 false positive를 추출합니다.

    "False positive"란?
      모델이 상위 10개 안에 올려보낸 결과 중 ground truth가 아닌 것.
      = 모델이 "정답"이라고 착각한 오답
      = 가장 가치 있는 hard negative (실제 파이프라인 약점 반영)

    통합 전략:
      model-based (priority=high)를 먼저 채우고,
      BM25(priority=medium)로 나머지 채움.
      중복 chunk_id는 model-based 유지.

    추가 필드:
      hard_negatives_model : Step 2 결과만
      hard_negatives       : Step 1+2 통합 (최종 사용)
      needs_manual_review  : 준용_역방향_후보 chunk_id 목록 (Step 3 체크리스트)
    """
    print("\n[Step 2] BGE-M3 Qdrant False Positive 추출 중...")
    results = []

    for item in items_with_bm25:
        ground_truth_set = set(item.get("ground_truth", []))

        search_results = qdrant_hybrid_search(client, model, item["clause_text"], top_k=top_k)

        # 상위 10개만 확인 — 10위 밖은 "헷갈린 것"이 아니라 "못 찾은 것"
        hard_negs_model = []
        for rank, r in enumerate(search_results[:10]):
            cid = r["chunk_id"]

            if cid in ground_truth_set:
                continue  # 정답이 상위에 올라온 건 좋은 것

            hn_type, reason = classify_hn_type(
                list(ground_truth_set)[0] if ground_truth_set else "",
                cid,
            )
            hard_negs_model.append({
                "chunk_id":       cid,
                "type":           hn_type,
                "reason":         reason,
                "source":         "bge_m3_false_positive",
                "retrieval_rank": rank + 1,   # 낮을수록 강한 hard negative
                "rrf_score":      r["score"],
            })

            if len(hard_negs_model) >= max_hn:
                break

        result_item = dict(item)
        result_item["hard_negatives_model"] = hard_negs_model

        # ── Step 1 + Step 2 통합 (중복 제거, model-based 우선) ──
        all_hn_ids = set()
        merged = []

        for hn in hard_negs_model:
            if hn["chunk_id"] not in all_hn_ids:
                merged.append({**hn, "priority": "high"})
                all_hn_ids.add(hn["chunk_id"])

        for hn in item.get("hard_negatives_bm25", []):
            if hn["chunk_id"] not in all_hn_ids:
                merged.append({**hn, "priority": "medium"})
                all_hn_ids.add(hn["chunk_id"])

        result_item["hard_negatives"] = merged

        # 준용_역방향_후보만 따로 추려서 수동 검수 체크리스트 생성
        result_item["needs_manual_review"] = [
            hn["chunk_id"] for hn in merged if "준용" in hn["type"]
        ]

        results.append(result_item)

        gt_str = ", ".join(ground_truth_set)
        print(
            f"  [{item['query_id']}] GT={gt_str} | "
            f"model HN={len(hard_negs_model)}개 | merged={len(merged)}개"
        )

    return results


# ═══════════════════════════════════════════════════════════════
# 5. 통계 요약
# ═══════════════════════════════════════════════════════════════

def print_summary(final_items: list[dict]):
    """
    mining 완료 후 결과 통계를 출력합니다.

    이상적인 분포: 동일주제_다른법령 비중이 가장 높아야 변별력 있는 평가셋.
    """
    total_hn = sum(len(it.get("hard_negatives", [])) for it in final_items)
    type_counter: dict[str, int] = defaultdict(int)
    for item in final_items:
        for hn in item.get("hard_negatives", []):
            type_counter[hn["type"]] += 1

    needs_review = sum(len(it.get("needs_manual_review", [])) for it in final_items)

    print("\n" + "=" * 55)
    print("  Hard Negative Mining 결과 요약")
    print("=" * 55)
    print(f"  총 쿼리 수         : {len(final_items)}")
    print(f"  총 hard negative   : {total_hn}")
    print(f"  평균 HN/쿼리       : {total_hn / max(len(final_items), 1):.1f}")
    print(f"  수동 검수 필요     : {needs_review}개 (준용_역방향_후보)")
    print()
    print("  타입별 분포:")
    for t, cnt in sorted(type_counter.items(), key=lambda x: -x[1]):
        print(f"    {t:<25} {cnt}개")
    print("=" * 55)


# ═══════════════════════════════════════════════════════════════
# 6. 씨드 템플릿 생성 (처음 시작할 때 사용)
# ═══════════════════════════════════════════════════════════════

# 카테고리별 대표 예시 — 실제 운영 시 yoonha_build_hn_seed.py로 자동 생성
SEED_TEMPLATE = [
    {
        "query_id":     "Q001",
        "category":     "지체상금",
        "clause_text":  "계약상대자의 귀책사유로 인한 계약이행 지연 시 1일당 계약금액의 1천분의 3에 해당하는 금액을 지체상금으로 납부하여야 한다.",
        "ground_truth": ["LCAR_75"],
        "hard_negatives": []
    },
    {
        "query_id":     "Q002",
        "category":     "대금지급",
        "clause_text":  "발주기관은 검사에 합격한 날부터 5일 이내에 대가를 지급하여야 한다.",
        "ground_truth": ["LCA_22"],
        "hard_negatives": []
    },
    {
        "query_id":     "Q003",
        "category":     "하자담보",
        "clause_text":  "소프트웨어 사업의 하자담보 책임기간은 인수일로부터 1년으로 한다.",
        "ground_truth": ["SWPA_51"],
        "hard_negatives": []
    },
    {
        "query_id":     "Q004",
        "category":     "계약보증금",
        "clause_text":  "계약보증금은 계약금액의 100분의 10 이상으로 한다.",
        "ground_truth": ["LCAE_52"],
        "hard_negatives": []
    },
    {
        "query_id":     "Q005",
        "category":     "지체상금",
        "clause_text":  "지체상금의 총액은 계약금액의 100분의 30을 초과할 수 없다.",
        "ground_truth": ["LCAR_75"],
        "hard_negatives": []
    },
    # ── Q006 이후는 yoonha_build_hn_seed.py로 자동 생성 ──
    # 권장: 카테고리별 5~6개씩 → 총 50개 이상
]


def create_seed_template(path: str):
    """
    씨드 파일이 없을 때 템플릿을 자동 생성합니다.
    이미 있으면 덮어쓰지 않고 스킵합니다.
    """
    if os.path.exists(path):
        print(f"[skip] {path} 이미 존재함")
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(SEED_TEMPLATE, f, ensure_ascii=False, indent=2)
    print(f"[생성] 씨드 템플릿 저장: {path}")
    print("  → ground_truth chunk_id를 채운 후 main()을 실행하세요.")
    print("  → 또는 yoonha_build_hn_seed.py로 train_all.jsonl에서 자동 생성하세요.")


# ═══════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════

def main(
    skip_step2  : bool = False,
    seed_path   : str  = SEED_PATH,
    output_path : str  = OUTPUT_PATH,
):
    """
    전체 파이프라인 실행.

    Args:
        skip_step2  : True면 BGE-M3 로딩 없이 BM25(Step 1)만 실행
                      → 빠른 테스트나 GPU 없는 환경에서 사용
        seed_path   : 씨드 쿼리 JSON 경로
        output_path : 결과 저장 경로

    실행 전 체크리스트:
      □ Docker 실행: docker run -p 6333:6333 qdrant/qdrant
      □ law_kb_ho 컬렉션 업로드 완료 여부 확인
      □ hard_negative_seed.json 준비
        (없으면 템플릿 자동 생성 → 또는 yoonha_build_hn_seed.py 실행)
      □ GPU 메모리 확인 (BGE-M3 use_fp16=True → 약 2.5GB VRAM 필요)
    """
    # 씨드 파일 없으면 템플릿 먼저 만들고 종료
    if not os.path.exists(seed_path):
        print(f"[안내] 씨드 파일({seed_path})이 없어 템플릿을 생성합니다.")
        create_seed_template(seed_path)
        return

    # Qdrant 연결
    print(f"[init] Qdrant 연결: {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # BM25용 전체 청크 로드 (law_kb_ho)
    chunk_ids, texts = load_chunks_from_qdrant(client, COLLECTION)

    # 씨드 쿼리 로드
    seed_items = load_seed(seed_path)
    print(f"[load] 씨드 쿼리 {len(seed_items)}개 로드")

    # ── Step 1: BM25 ───────────────────────────────────────────
    items_with_bm25 = build_bm25_hard_negatives(seed_items, chunk_ids, texts)

    if skip_step2:
        # BM25만으로 최종 결과 구성 (Step 2 생략)
        final_items = items_with_bm25
        for it in final_items:
            it["hard_negatives"] = it.get("hard_negatives_bm25", [])
            it["needs_manual_review"] = [
                hn["chunk_id"] for hn in it["hard_negatives"] if "준용" in hn["type"]
            ]
    else:
        # ── Step 2: BGE-M3 ─────────────────────────────────────
        print(f"\n[init] BGE-M3 모델 로딩: {BGE_MODEL_ID}")
        model = BGEM3FlagModel(BGE_MODEL_ID, use_fp16=True)
        final_items = build_model_hard_negatives(items_with_bm25, client, model)

    # 결과 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_items, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {output_path}")

    # 통계 출력
    print_summary(final_items)

    # Step 3 수동 검수 안내
    print("\n⚠️  Step 3 수동 검수 체크리스트:")
    print("  1. needs_manual_review 목록 chunk_id가 실제로 오답인지 확인")
    print("     (준용·역방향 관계여서 실제로는 정답일 수 있음 → 발견 시 ground_truth에 추가)")
    print("  2. ground_truth에 빠진 정답 있으면 추가")
    print("  3. 타입 분류 오류 발견 시 수동 수정 후 재저장")
    print("  4. 최소 목표: 쿼리 50개 이상, 쿼리당 HN 3개 이상, 전체 카테고리 커버")
    print("\n다음 단계:")
    print(f"  py rag/evaluation/yoonha_contract_evaluation.py  ← 평가 실행")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Workit Hard Negative Miner")
    parser.add_argument(
        "--bm25-only", action="store_true",
        help="BGE-M3 로딩 없이 BM25 Step 1만 실행 (빠른 테스트용)"
    )
    parser.add_argument("--seed",   default=SEED_PATH,   help="씨드 JSON 경로")
    parser.add_argument("--output", default=OUTPUT_PATH, help="출력 JSON 경로")
    args = parser.parse_args()

    main(
        skip_step2  = args.bm25_only,
        seed_path   = args.seed,
        output_path = args.output,
    )