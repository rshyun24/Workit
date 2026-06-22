"""
Workit - 계약서 검토 RAG 파이프라인
파일명: yoonha_contract_rag.py
위치:   Workit/rag/yoonha_contract_rag.py

흐름:
  계약서 텍스트 입력
      ↓
  조항 단위 청킹 (제N조 기준, 법령 인용·번호 역행 필터링 적용)
      ↓
  각 조항 → Qdrant law_kb 하이브리드 검색
    (Dense KURE-v1 + Sparse TF, Qdrant 내부 RRF 융합)
    (is_risk_ref=True 필터 적용)
      ↓
  chunk_id → laws_ref.json 에서 article + category 조회
      ↓
  ClauseResult 반환 (PDF 좌표 page/bbox 포함 — tasks.py에서 병합)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    NamedSparseVector,
    Prefetch,
)
from sentence_transformers import SentenceTransformer

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
EMBED_MODEL = "nlpai-lab/KURE-v1"

TOP_K     = 5
MIN_SCORE = 0.40


# ──────────────────────────────────────────
# 1. 데이터 클래스
# ──────────────────────────────────────────
@dataclass
class LawRef:
    """검색된 법령 조문 1건"""
    chunk_id    : str
    article     : str   # 예: "지방계약법 제18조"
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
    page          : int = 0
    bbox          : dict | None = None
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
# 3. 계약서 조항 단위 청킹
# ──────────────────────────────────────────
def chunk_contract(text: str) -> list[dict]:
    """
    계약서 텍스트를 조항(제N조) 단위로 분할.

    "제N조 (제목)" 형태만 헤더로 인식하되,
    - 바로 앞 5글자가 "법"으로 끝나면 타 법령 인용으로 보고 제외
      (예: "하도급법 제14조(하도급대금의 직접 지급)" → 제외)
    - 조항 번호가 직전 번호보다 작거나 +5를 초과해 튀면 인용으로 보고 제외
      (예: 계약서 본문 중 "소프트웨어진흥법 제38조(공정계약의 원칙)에 따라" → 제외)

    조항 패턴이 전혀 없으면 단락 단위로 fallback.
    """
    text = text.strip()
    header_pattern = re.compile(r"제(\d+)조(?:의(\d+))?\s*\(([^)]*)\)")

    raw_matches = list(header_pattern.finditer(text))

    candidates = []
    for m in raw_matches:
        prefix = text[max(0, m.start() - 5):m.start()]
        if re.search(r"법\s*$", prefix):
            continue  # 타 법령 인용 제외

        num = int(m.group(1))
        sub = m.group(2)
        clause_number = f"제{m.group(1)}조" + (f"의{sub}" if sub else "")
        candidates.append((num, clause_number, m.start()))

    # 번호 순서 검사 (직전 번호와 같거나 +1~+5 이내만 허용)
    filtered = []
    last_num = 0
    for num, clause_number, start in candidates:
        if num >= last_num and num <= last_num + 5:
            filtered.append((clause_number, start))
            last_num = num

    clauses = []
    for idx, (clause_number, start) in enumerate(filtered):
        end = filtered[idx + 1][1] if idx + 1 < len(filtered) else len(text)
        clause_text = text[start:end].strip()
        clause_text = re.sub(r"\s+", " ", clause_text)
        if clause_text:
            clauses.append({"clause_number": clause_number, "clause_text": clause_text})

    if not clauses:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        clauses = [
            {"clause_number": f"단락{i + 1}", "clause_text": para}
            for i, para in enumerate(paragraphs)
        ]

    return clauses


# ──────────────────────────────────────────
# 4. Sparse 벡터 생성
# ──────────────────────────────────────────
def get_sparse_vector(text: str, model: SentenceTransformer) -> dict[int, float]:
    tokens    = model.tokenizer.tokenize(text)
    token_ids = model.tokenizer.convert_tokens_to_ids(tokens)

    freq: dict[int, float] = {}
    for tid in token_ids:
        freq[tid] = freq.get(tid, 0.0) + 1.0

    norm = math.sqrt(sum(v ** 2 for v in freq.values()))
    return {k: v / norm for k, v in freq.items()} if norm > 0 else freq


# ──────────────────────────────────────────
# 5. 단일 조항 → 법령 하이브리드 검색
# ──────────────────────────────────────────
def search_law_for_clause(
    clause_text : str,
    client      : QdrantClient,
    model       : SentenceTransformer,
    laws_ref    : dict[str, dict],
    top_k       : int   = TOP_K,
    min_score   : float = MIN_SCORE,
) -> list[LawRef]:
    dense_vector = model.encode(clause_text, normalize_embeddings=True).tolist()
    sparse_dict  = get_sparse_vector(clause_text, model)
    indices      = list(sparse_dict.keys())
    values       = list(sparse_dict.values())

    risk_filter = Filter(
        must=[FieldCondition(key="is_risk_ref", match=MatchValue(value=True))]
    )

    try:
        response = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                Prefetch(query=dense_vector,                                       using="dense",  limit=top_k, filter=risk_filter),
                Prefetch(query=NamedSparseVector(indices=indices, values=values),  using="sparse", limit=top_k, filter=risk_filter),
            ],
            query="rrf",
            limit=top_k,
        )
    except Exception:
        # sparse 미지원 환경 폴백
        response = client.query_points(
            collection_name=COLLECTION,
            query=dense_vector,
            query_filter=risk_filter,
            score_threshold=min_score,
            limit=top_k,
        )

    law_refs: list[LawRef] = []
    for point in response.points:
        if point.score is not None and point.score < min_score:
            continue

        payload  = point.payload or {}
        chunk_id = payload.get("chunk_id", "")
        ref_meta = laws_ref.get(chunk_id, {})

        law_refs.append(LawRef(
            chunk_id    = chunk_id,
            article     = ref_meta.get("article",  payload.get("source_full", "")),
            category    = ref_meta.get("category", ""),
            law_name    = payload.get("law_name",  ""),
            chunk_text  = payload.get("chunk_text", payload.get("text", "")),
            score       = round(float(point.score or 0.0), 4),
            is_risk_ref = bool(payload.get("is_risk_ref", False)),
        ))

    return law_refs


# ──────────────────────────────────────────
# 6. 전체 계약서 검토 (메인 인터페이스)
# ──────────────────────────────────────────
def review_contract(
    contract_text : str,
    client        : QdrantClient,
    model         : SentenceTransformer,
    laws_ref      : dict[str, dict] | None = None,
    top_k         : int   = TOP_K,
    min_score     : float = MIN_SCORE,
) -> list[ClauseResult]:
    """
    계약서 전체 텍스트 → 조항별 관련 법령 검색 결과 반환.
    laws_ref를 안 넘기면 LAWS_REF_PATH에서 자동 로드한다.
    """
    if laws_ref is None:
        laws_ref = load_laws_ref()

    clauses = chunk_contract(contract_text)
    results : list[ClauseResult] = []

    print(f"  총 {len(clauses)}개 조항 검색 중...")

    for i, clause in enumerate(clauses, 1):
        print(f"  [{i}/{len(clauses)}] {clause['clause_number']} 검색 중...", end="\r")

        law_refs = search_law_for_clause(
            clause_text = clause["clause_text"],
            client      = client,
            model       = model,
            laws_ref    = laws_ref,
            top_k       = top_k,
            min_score   = min_score,
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


# ──────────────────────────────────────────
# 7. JSON 변환 (tasks.py에서 사용)
# ──────────────────────────────────────────
def results_to_json(results: list[ClauseResult]) -> list[dict]:
    """
    ClauseResult 리스트를 dict 리스트로 변환.
    sLLM(jihye_inference.predict) 및 좌표 병합(tasks.py)에서 그대로 사용한다.
    """
    return [asdict(result) for result in results]