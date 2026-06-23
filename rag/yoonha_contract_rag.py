"""
Workit - 계약서 검토 RAG 파이프라인
파일명: yoonha_contract_rag.py
위치:   Workit/rag/yoonha_contract_rag.py

흐름:
  계약서 텍스트 입력
      ↓
  조항 단위 청킹 (법령 인용·번호 역행 필터링) → 항 단위 2차 분할 (①②③)
      ↓
  각 청크 → Qdrant law_kb 하이브리드 검색
    (Dense BGE-M3 + Sparse BGE-M3 SPLADE, Qdrant 내부 RRF 융합)
      ↓
  chunk_id → laws_ref.json 에서 article + category 조회
      ↓
  ClauseResult 반환 (PDF 좌표 page/bbox는 tasks.py에서 병합)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
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

TOP_K     = 10
FETCH_K   = 20
MIN_SCORE = None  # RRF 점수는 스케일이 달라 threshold 대신 top_k로만 제어


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
    """계약서 조항(또는 항) 1건의 검색 결과"""
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
    lexical_weights = output["lexical_weights"][0]

    sparse_vector: dict[int, float] = {}
    for token_str, weight in lexical_weights.items():
        token_id = model.tokenizer.convert_tokens_to_ids(token_str)
        if isinstance(token_id, int):
            sparse_vector[token_id] = sparse_vector.get(token_id, 0.0) + float(weight)

    return dense_vector, sparse_vector


# ──────────────────────────────────────────
# 5. 계약서 조항+항 단위 청킹
# ──────────────────────────────────────────
def chunk_contract(text: str) -> list[dict]:
    """
    계약서 텍스트를 조항(제N조) 단위로 1차 분할 후,
    내부 ①②③ 항 단위로 2차 분할.

    1차 분할(조 단위) 헤더 인식 시 다음 두 가지 필터를 적용해 본문 인용을 제외한다.
    - "제N조 (제목)" 바로 앞 5글자가 "법"으로 끝나면 타 법령 인용으로 제외
      (예: "하도급법 제14조(하도급대금의 직접 지급)")
    - 조항 번호가 직전 번호보다 작거나 +5를 초과해 튀면 인용으로 제외
      (예: "소프트웨어진흥법 제38조(공정계약의 원칙)에 따라")

    법령 KB가 항/호 단위로 청킹되어 있으므로
    계약서도 항 단위로 맞춰야 임베딩 매칭 품질이 올라감.
    조항 단위로만 쪼개면 쿼리가 너무 길어져 임베딩이 희석됨.

    항이 없는 조항은 조 단위 청크로 유지.
    조항 패턴이 전혀 없으면 단락 단위로 fallback.
    """
    HANG_MAP = {c: i + 1 for i, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮")}
    # 호 패턴: "1. " "2. " 형태. 본문 중 우연한 "1." (예: 금액 표기)과 구분하기 위해
    # 반드시 공백/문장 시작 뒤에 오고, 뒤에 한글/괄호 등 항목 설명이 이어지는 경우만 인정한다.
    HO_SPLIT_PATTERN = r"(?:^|\s)(\d{1,2}\.\s)"

    text = text.strip()
    header_pattern = re.compile(r"제(\d+)조(?:의(\d+))?\s*\(([^)]*)\)")

    raw_matches = list(header_pattern.finditer(text))

    # ── 1차 필터: 법령 인용 제외 ──
    candidates = []
    for m in raw_matches:
        prefix = text[max(0, m.start() - 5):m.start()]
        if re.search(r"법\s*$", prefix):
            continue  # "하도급법 제14조" 같은 타 법령 인용 제외

        num = int(m.group(1))
        sub = m.group(2)
        clause_number = f"제{m.group(1)}조" + (f"의{sub}" if sub else "")
        candidates.append((num, clause_number, m.start()))

    # ── 2차 필터: 번호 순서 검사 (직전 번호와 같거나 +1~+5 이내만 허용) ──
    header_spans = []  # [(clause_number, start), ...]
    last_num = 0
    for num, clause_number, start in candidates:
        if num >= last_num and num <= last_num + 5:
            header_spans.append((clause_number, start))
            last_num = num

    def split_into_ho(parent_number: str, unit_text: str) -> list[dict]:
        """
        조 또는 항 영역(unit_text) 안에서 "1. 2. 3." 형태의 호를 찾아 분할한다.
        호가 없으면 [{"clause_number": parent_number, "clause_text": unit_text}] 하나만 반환.
        """
        ho_splits = re.split(HO_SPLIT_PATTERN, unit_text)
        # ho_splits 형태: [머리말, "1. ", 본문1, "2. ", 본문2, ...]

        if len(ho_splits) <= 1:
            return [{"clause_number": parent_number, "clause_text": unit_text}]

        head = ho_splits[0].strip()
        chunks = []
        if head:
            chunks.append({"clause_number": parent_number, "clause_text": head})

        k = 1
        last_ho_num = 0
        while k < len(ho_splits) - 1:
            marker = ho_splits[k].strip()  # "1." 같은 형태 (끝 공백 strip됨)
            ho_num_match = re.match(r"(\d{1,2})\.", marker)
            ho_num = int(ho_num_match.group(1)) if ho_num_match else (k // 2 + 1)
            ho_body = ho_splits[k + 1].strip() if k + 1 < len(ho_splits) else ""

            # 번호 순서 검사: 1,2,3... 순서로 증가해야 진짜 호 목록으로 인정
            # (순서가 어긋나면 본문 중 우연한 "N." 표현으로 보고 이전 청크에 이어붙인다)
            if ho_num == last_ho_num + 1 and ho_body:
                chunks.append({
                    "clause_number": f"{parent_number}제{ho_num}호",
                    "clause_text": re.sub(r"\s+", " ", f"{marker} {ho_body}").strip(),
                })
                last_ho_num = ho_num
            elif ho_body:
                # 순서가 안 맞으면 직전 청크에 이어붙임(분리하지 않음)
                if chunks:
                    chunks[-1]["clause_text"] += f" {marker} {ho_body}"
                else:
                    chunks.append({"clause_number": parent_number, "clause_text": f"{marker} {ho_body}"})

            k += 2

        return chunks if chunks else [{"clause_number": parent_number, "clause_text": unit_text}]

    clauses = []
    for idx, (clause_number, start) in enumerate(header_spans):
        end = header_spans[idx + 1][1] if idx + 1 < len(header_spans) else len(text)
        raw_block = text[start:end].strip()

        # raw_block = "제N조 (제목) 본문..." → 헤더와 본문(body) 분리
        m = header_pattern.match(raw_block)
        raw_header = m.group(0) if m else clause_number
        body = raw_block[m.end():].strip() if m else raw_block

        if not body:
            continue

        # 항 분리 시도 (①②③ 원문자 기준)
        hang_splits = re.split(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])", body)

        if len(hang_splits) <= 1:
            # 항 없음 → 조 영역 안에서 바로 호 분리 시도
            clause_text = re.sub(r"\s+", " ", f"{raw_header} {body}").strip()
            clauses.extend(split_into_ho(clause_number, clause_text))
        else:
            # 항 있음 → 항 단위로 먼저 나누고, 각 항 안에서 호 분리 시도
            j = 1
            while j < len(hang_splits) - 1:
                hang_char = hang_splits[j]
                hang_body = hang_splits[j + 1].strip() if j + 1 < len(hang_splits) else ""
                hang_num = HANG_MAP.get(hang_char, j)
                if hang_body:
                    hang_number = f"{clause_number}제{hang_num}항"
                    hang_text = re.sub(r"\s+", " ", f"{raw_header} {hang_char}{hang_body}").strip()
                    clauses.extend(split_into_ho(hang_number, hang_text))
                j += 2

    if not clauses:
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
        response = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                Prefetch(query=dense_vector,                                 limit=FETCH_K, using="dense"),
                Prefetch(query=SparseVector(indices=indices, values=values), limit=FETCH_K, using="sparse"),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        )
    except Exception:
        response = client.query_points(
            collection_name=COLLECTION,
            query=dense_vector,
            using="dense",
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
    laws_ref      : dict[str, dict] | None = None,
    top_k         : int = TOP_K,
) -> list[ClauseResult]:
    """
    계약서 전체 텍스트 → 조항/항별 관련 법령 검색 결과 반환.
    laws_ref를 안 넘기면 LAWS_REF_PATH에서 자동 로드.
    """
    if laws_ref is None:
        laws_ref = load_laws_ref()

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


# ──────────────────────────────────────────
# 8. JSON 변환 (tasks.py에서 사용)
# ──────────────────────────────────────────
def results_to_json(results: list[ClauseResult]) -> list[dict]:
    """
    ClauseResult 리스트를 dict 리스트로 변환.
    sLLM(jihye_inference.predict) 및 좌표 병합(tasks.py)에서 사용.
    """
    return [asdict(result) for result in results]