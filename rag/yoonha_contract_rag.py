"""
Workit - 계약서 검토 RAG 파이프라인
파일명: yoonha_contract_rag.py
위치:   Workit/rag/yoonha_contract_rag.py

흐름:
  계약서 텍스트 입력
      ↓
  조항+항 단위 청킹 (제N조 → ①②③ 항 분리)
      ↓
  각 청크 → Qdrant law_kb 하이브리드 검색
    (Dense BGE-M3 + Sparse BGE-M3 SPLADE, Qdrant 내부 RRF 융합)
      ↓
  chunk_id → laws_ref.json 에서 article + category 조회
      ↓
  ClauseResult 반환
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    NamedSparseVector,
    Prefetch,
)

# ──────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────
_THIS_DIR     = Path(__file__).resolve().parent
_DATA_DIR     = _THIS_DIR.parent / "data"
LAWS_REF_PATH = _DATA_DIR / "laws_ref.json"

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION  = "law_kb"
EMBED_MODEL = "BAAI/bge-m3"

# TOP_K: 조항 하나가 여러 리스크 카테고리에 걸칠 수 있으므로 넉넉하게 설정
# 추후 실험 기반으로 조정 필요
TOP_K = 10
FETCH_K = 20

# MIN_SCORE: RRF 점수는 dense cosine similarity와 스케일이 다르므로
# 고정 threshold를 걸면 오히려 결과를 잘라낼 수 있음
# → top_k로만 제어하고 threshold는 사용하지 않음
MIN_SCORE = None


# ──────────────────────────────────────────
# 1. 데이터 클래스
# ──────────────────────────────────────────
@dataclass
class LawRef:
    """검색된 법령 조문 1건"""
    chunk_id    : str
    article     : str   # 예: "지방계약법 제18조제1항"
    category    : str   # 예: "대금지급"
    law_name    : str
    chunk_text  : str
    score       : float
    is_risk_ref : bool


@dataclass
class ClauseResult:
    """계약서 조항 1건의 검색 결과"""
    clause_number : str
    clause_text   : str
    law_refs      : list[LawRef] = field(default_factory=list)
    categories    : list[str]   = field(default_factory=list)


# ──────────────────────────────────────────
# 2. laws_ref.json 로드
# ──────────────────────────────────────────
def load_laws_ref(path: Path = LAWS_REF_PATH) -> dict[str, dict]:
    if not path.exists():
        print(f"  ⚠️  laws_ref.json 없음: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────
# 3. BGE-M3 모델 로드
# ──────────────────────────────────────────
def load_model(model_name: str = EMBED_MODEL) -> BGEM3FlagModel:
    """BGE-M3 모델 로드 — Dense + Sparse 동시 추출 지원."""
    print(f"📦 임베딩 모델 로드: {model_name}")
    return BGEM3FlagModel(model_name, use_fp16=True)


# ──────────────────────────────────────────
# 4. BGE-M3 Dense + Sparse 벡터 추출
# ──────────────────────────────────────────
def get_vectors(text: str, model: BGEM3FlagModel) -> tuple[list[float], dict[int, float]]:
    """
    BGE-M3로 Dense + Sparse(SPLADE) 벡터를 동시 추출.
    KURE-v1의 토크나이저 TF 근사 대신 모델 자체 lexical weights 사용.

    Returns:
        dense_vector  : list[float] (1024차원)
        sparse_vector : dict[int, float] {token_id: weight}
    """
    output = model.encode(
        [text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense_vector    = output["dense_vecs"][0].tolist()
    lexical_weights = output["lexical_weights"][0]  # {token_str: weight}

    # token_str → token_id 변환
    sparse_vector: dict[int, float] = {}
    for token_str, weight in lexical_weights.items():
        token_id = model.tokenizer.convert_tokens_to_ids(token_str)
        if isinstance(token_id, int):
            sparse_vector[token_id] = float(weight)

    return dense_vector, sparse_vector


# ──────────────────────────────────────────
# 5. 계약서 조항+항 단위 청킹
# ──────────────────────────────────────────
def chunk_contract(text: str) -> list[dict]:
    """
    계약서 텍스트를 조항(제N조) 단위로 1차 분할 후,
    내부 ①②③ 항 단위로 2차 분할.

    법령 KB가 항/호 단위로 청킹되어 있으므로
    계약서도 항 단위로 맞춰야 임베딩 매칭 품질이 올라감.
    조항 단위로만 쪼개면 쿼리가 너무 길어져 임베딩이 희석됨.

    항이 없는 조항은 조 단위 청크로 유지.
    조항 패턴이 없으면 단락 단위로 fallback.
    """
    HANG_MAP = {c: i + 1 for i, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮")}

    text    = text.strip()
    pattern = r"(제\d+조(?:의\d+)?(?:\s*\([^)]*\))?)"
    parts   = re.split(pattern, text)

    clauses = []
    i = 1
    while i < len(parts) - 1:
        raw_header    = parts[i].strip()
        body          = parts[i + 1].strip()
        match         = re.match(r"(제\d+조(?:의\d+)?)", raw_header)
        clause_number = match.group(1) if match else raw_header

        if not body:
            i += 2
            continue

        # 항 분리 시도 (①②③ 원문자 기준)
        hang_splits = re.split(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])", body)

        if len(hang_splits) <= 1:
            # 항 없음 → 조 단위 청크
            clauses.append({
                "clause_number": clause_number,
                "clause_text":   f"{raw_header} {body}".strip(),
            })
        else:
            # 항 있음 → 항 단위 청크
            j = 1
            while j < len(hang_splits) - 1:
                hang_char = hang_splits[j]
                hang_body = hang_splits[j + 1].strip() if j + 1 < len(hang_splits) else ""
                hang_num  = HANG_MAP.get(hang_char, j)
                if hang_body:
                    clauses.append({
                        "clause_number": f"{clause_number}제{hang_num}항",
                        "clause_text":   f"{raw_header} {hang_char}{hang_body}".strip(),
                    })
                j += 2

        i += 2

    if not clauses:
        # fallback: 단락 단위
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        clauses = [
            {"clause_number": f"단락{i + 1}", "clause_text": para}
            for i, para in enumerate(paragraphs)
        ]

    return clauses


# ──────────────────────────────────────────
# 6. 단일 청크 → 법령 하이브리드 검색
# ──────────────────────────────────────────
def search_law_for_clause(
    clause_text : str,
    client      : QdrantClient,
    model       : BGEM3FlagModel,
    laws_ref    : dict[str, dict],
    top_k       : int = TOP_K,
) -> list[LawRef]:
    dense_vector, sparse_vector = get_vectors(clause_text, model)
    indices = list(sparse_vector.keys())
    values  = list(sparse_vector.values())

    try:
        # Dense + Sparse SPLADE → Qdrant 내부 RRF 융합
        # RRF 점수는 cosine similarity와 스케일이 다르므로 threshold 없이 top_k로만 제어
        response = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                Prefetch(query=dense_vector,                                      using="dense",  limit=FETCH_K),
                Prefetch(query=NamedSparseVector(indices=indices, values=values), using="sparse", limit=FETCH_K),
            ],
            query="rrf",
            limit=top_k,
        )
    except Exception:
        # sparse 컬렉션 미구성 환경 폴백 — dense 단독, threshold 없이 top_k만 사용
        response = client.query_points(
            collection_name=COLLECTION,
            query=dense_vector,
            limit=top_k,
        )

    law_refs: list[LawRef] = []
    for point in response.points:
        payload  = point.payload or {}
        chunk_id = payload.get("chunk_id", "")
        ref_meta = laws_ref.get(chunk_id, {})

        law_refs.append(LawRef(
            chunk_id    = chunk_id,
            article     = ref_meta.get("article",  payload.get("article", "")),
            category    = ref_meta.get("category", payload.get("category", "")),
            law_name    = payload.get("law_name",  ""),
            chunk_text  = payload.get("chunk_text", payload.get("text", "")),
            score       = round(float(point.score or 0.0), 4),
            is_risk_ref = bool(payload.get("is_risk_ref", False)),
        ))

    return law_refs


# ──────────────────────────────────────────
# 7. 전체 계약서 검토 (메인 인터페이스)
# ──────────────────────────────────────────
def review_contract(
    contract_text : str,
    client        : QdrantClient,
    model         : BGEM3FlagModel,
    laws_ref      : dict[str, dict],
    top_k         : int = TOP_K,
) -> list[ClauseResult]:
    """
    계약서 전체 텍스트 → 조항/항별 관련 법령 검색 결과 반환.
    """
    clauses = chunk_contract(contract_text)
    results : list[ClauseResult] = []

    print(f"  총 {len(clauses)}개 청크 검색 중...")

    for i, clause in enumerate(clauses, 1):
        print(f"  [{i}/{len(clauses)}] {clause['clause_number']} 검색 중...", end="\r")

        law_refs = search_law_for_clause(
            clause_text = clause["clause_text"],
            client      = client,
            model       = model,
            laws_ref    = laws_ref,
            top_k       = top_k,
        )

        categories = list(dict.fromkeys(
            ref.category for ref in law_refs if ref.category
        ))

        results.append(ClauseResult(
            clause_number = clause["clause_number"],
            clause_text   = clause["clause_text"],
            law_refs      = law_refs,
            categories    = categories,
        ))

    print("\n  ✅ 검색 완료")
    return results