"""
Workit - 계약서 검토 RAG 파이프라인 (3-variant)
파일명: yoonha_law_rag.py

컬렉션별 독립 RAG (성능 비교용):
  - JoRAG      : law_kb_jo      (조 단위, parent 없음)
  - HoRAG      : law_kb_ho      (호 단위, parent fetch → 조 텍스트)
  - HoXrefRAG  : law_kb_ho_xref (호 단위 + cross_refs, parent fetch → 조 텍스트)

공통 출력: list[ClauseResult]  ← 항상 조 단위로 반환
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 경로 / 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_THIS_DIR     = Path(__file__).resolve().parent
_DATA_DIR     = _THIS_DIR.parent / "data"
LAWS_REF_PATH = _DATA_DIR / "hn_seed" / "law_refs.json"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_JO      = "law_kb_jo"
COLLECTION_HO      = "law_kb_ho"
COLLECTION_HO_XREF = "law_kb_ho_xref"

EMBED_MODEL = "BAAI/bge-m3"

FETCH_K   = 50
RERANK1_K = 30   # 1단계에서 너무 좁게 자르지 않도록 완화 (12 → 30)
RERANK2_K = 10   # 최종 TOP_K와 동일하게 맞춤 (7 → 10)
TOP_K     = 10
RRF_ALPHA = 0.5  # 1.0 = dense only, 0.0 = sparse only, 0.5 = 균등

# RERANKER1: 한국어 특화 cross-encoder (ms-marco 영어 모델 → ko 모델로 교체)
RERANKER1_MODEL = "Dongjin-kr/ko-reranker"
RERANKER2_MODEL = "BAAI/bge-reranker-v2-m3"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cross-encoder Reranker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CrossEncoderReranker:
    """
    transformers AutoModel 기반 Cross-encoder reranker.
    FlagReranker 대체용 — 최신 transformers 호환.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def compute_score(
        self,
        pairs     : list[list[str]],
        batch_size: int  = 32,
        normalize : bool = True,
    ) -> list[float]:
        all_scores: list[float] = []

        for i in range(0, len(pairs), batch_size):
            batch   = pairs[i : i + batch_size]
            encoded = self.tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                logits = self.model(**encoded).logits

            scores = logits.squeeze(-1) if logits.shape[-1] == 1 else logits[:, 1]
            if normalize:
                scores = torch.sigmoid(scores)

            all_scores.extend(scores.cpu().tolist())

        return all_scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터 클래스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class LawRef:
    """검색된 법령 조문 1건."""
    chunk_id   : str
    article    : str
    category   : str
    law_name   : str
    chunk_text : str
    score      : float
    is_risk_ref: bool
    parent_id  : str = ""
    cross_refs : list[str] = field(default_factory=list)  # ho_xref 전용


@dataclass
class ClauseResult:
    """계약서 조항 1건의 검색 결과 — 항상 조 단위."""
    clause_number: str
    clause_text  : str
    page         : int            = 0
    bbox         : dict | None    = None
    law_refs     : list[LawRef]   = field(default_factory=list)
    categories   : list[str]      = field(default_factory=list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 공통 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_laws_ref(path: Path = LAWS_REF_PATH) -> dict[str, dict]:
    if not path.exists():
        print(f"  ⚠️  laws_ref.json 없음: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_embed_model(model_name: str = EMBED_MODEL, use_fp16: bool = True) -> BGEM3FlagModel:
    print(f"📦 임베딩 모델 로드: {model_name}")
    return BGEM3FlagModel(model_name, use_fp16=use_fp16)


def load_rerankers(device: str = "cpu") -> tuple[CrossEncoderReranker, CrossEncoderReranker]:
    print(f"📦 Re-ranker 1단계 로드: {RERANKER1_MODEL}")
    r1 = CrossEncoderReranker(RERANKER1_MODEL, device=device)
    print(f"📦 Re-ranker 2단계 로드: {RERANKER2_MODEL}")
    r2 = CrossEncoderReranker(RERANKER2_MODEL, device=device)
    return r1, r2


def get_vectors(
    text : str,
    model: BGEM3FlagModel,
) -> tuple[list[float], dict[int, float]]:
    output = model.encode(
        [text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_vec       = output["dense_vecs"][0].tolist()
    lexical_weights = output["lexical_weights"][0]

    sparse_vec: dict[int, float] = {}
    for token_str, weight in lexical_weights.items():
        token_id = model.tokenizer.convert_tokens_to_ids(token_str)
        if isinstance(token_id, int):
            sparse_vec[token_id] = sparse_vec.get(token_id, 0.0) + float(weight)

    return dense_vec, sparse_vec


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 계약서 청킹 (조 단위 출력)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def chunk_contract(text: str) -> list[dict]:
    """계약서를 조 단위로 청킹."""
    HANG_MAP = {c: i + 1 for i, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮")}
    HO_SPLIT_PATTERN = r"(?:^|\s)(\d{1,2}\.\s)"

    text = text.strip()
    header_pattern = re.compile(r"제(\d+)조(?:의(\d+))?\s*\(([^)]*)\)")
    raw_matches    = list(header_pattern.finditer(text))

    candidates = []
    for m in raw_matches:
        prefix = text[max(0, m.start() - 5):m.start()]
        if re.search(r"법\s*$", prefix):
            continue
        num           = int(m.group(1))
        sub           = m.group(2)
        clause_number = f"제{m.group(1)}조" + (f"의{sub}" if sub else "")
        candidates.append((num, clause_number, m.start()))

    header_spans = []
    last_num = 0
    for num, clause_number, start in candidates:
        if num >= last_num and num <= last_num + 5:
            header_spans.append((clause_number, start))
            last_num = num

    def split_into_ho(parent_number: str, unit_text: str) -> list[dict]:
        ho_splits = re.split(HO_SPLIT_PATTERN, unit_text)
        if len(ho_splits) <= 1:
            return [{"clause_number": parent_number, "clause_text": unit_text}]

        head   = ho_splits[0].strip()
        chunks = []
        if head:
            chunks.append({"clause_number": parent_number, "clause_text": head})

        k, last_ho_num = 1, 0
        while k < len(ho_splits) - 1:
            marker       = ho_splits[k].strip()
            ho_num_match = re.match(r"(\d{1,2})\.", marker)
            ho_num       = int(ho_num_match.group(1)) if ho_num_match else (k // 2 + 1)
            ho_body      = ho_splits[k + 1].strip() if k + 1 < len(ho_splits) else ""

            if ho_num == last_ho_num + 1 and ho_body:
                chunks.append({
                    "clause_number": f"{parent_number}제{ho_num}호",
                    "clause_text":   re.sub(r"\s+", " ", f"{marker} {ho_body}").strip(),
                })
                last_ho_num = ho_num
            elif ho_body:
                if chunks:
                    chunks[-1]["clause_text"] += f" {marker} {ho_body}"
                else:
                    chunks.append({"clause_number": parent_number, "clause_text": f"{marker} {ho_body}"})
            k += 2

        return chunks if chunks else [{"clause_number": parent_number, "clause_text": unit_text}]

    clauses = []
    for idx, (clause_number, start) in enumerate(header_spans):
        end       = header_spans[idx + 1][1] if idx + 1 < len(header_spans) else len(text)
        raw_block = text[start:end].strip()

        m          = header_pattern.match(raw_block)
        raw_header = m.group(0) if m else clause_number
        body       = raw_block[m.end():].strip() if m else raw_block

        if not body:
            continue

        hang_splits = re.split(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])", body)

        if len(hang_splits) <= 1:
            clause_text = re.sub(r"\s+", " ", f"{raw_header} {body}").strip()
            clauses.extend(split_into_ho(clause_number, clause_text))
        else:
            j = 1
            while j < len(hang_splits) - 1:
                hang_char   = hang_splits[j]
                hang_body   = hang_splits[j + 1].strip() if j + 1 < len(hang_splits) else ""
                hang_num    = HANG_MAP.get(hang_char, j)
                if hang_body:
                    hang_text = re.sub(r"\s+", " ", f"{raw_header} {hang_char}{hang_body}").strip()
                    clauses.extend(split_into_ho(f"{clause_number}제{hang_num}항", hang_text))
                j += 2

    if not clauses:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        clauses = [
            {"clause_number": f"단락{i + 1}", "clause_text": para}
            for i, para in enumerate(paragraphs)
        ]

    return clauses


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 공통 검색 / 리랭크 / parent fetch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _hybrid_search(
    clause_text: str,
    client     : QdrantClient,
    model      : BGEM3FlagModel,
    collection : str,
    fetch_k    : int   = FETCH_K,
    alpha      : float = RRF_ALPHA,
) -> list[dict]:
    """Dense + Sparse 하이브리드 검색 (수동 RRF)."""
    dense_vec, sparse_vec = get_vectors(clause_text, model)
    indices = list(sparse_vec.keys())
    values  = list(sparse_vec.values())
    RRF_K   = 60

    try:
        dense_results = client.query_points(
            collection_name=collection,
            query=dense_vec,
            using="dense",
            limit=fetch_k,
            with_payload=True,
        ).points

        sparse_results = client.query_points(
            collection_name=collection,
            query=SparseVector(indices=indices, values=values),
            using="sparse",
            limit=fetch_k,
            with_payload=True,
        ).points

    except Exception as e:
        print(f"  ⚠️  sparse 검색 실패, dense만 사용: {e}")
        dense_results = client.query_points(
            collection_name=collection,
            query=dense_vec,
            using="dense",
            limit=fetch_k,
            with_payload=True,
        ).points
        sparse_results = []

    scores: dict[str, dict] = {}

    for rank, point in enumerate(dense_results, 1):
        cid = point.payload.get("chunk_id", str(point.id))
        scores[cid] = {
            "payload":     point.payload,
            "dense_rank":  rank,
            "sparse_rank": len(dense_results) + 1,
        }

    for rank, point in enumerate(sparse_results, 1):
        cid = point.payload.get("chunk_id", str(point.id))
        if cid in scores:
            scores[cid]["sparse_rank"] = rank
        else:
            scores[cid] = {
                "payload":     point.payload,
                "dense_rank":  len(sparse_results) + 1,
                "sparse_rank": rank,
            }

    results = []
    for cid, info in scores.items():
        rrf_score = (
            alpha         * (1 / (RRF_K + info["dense_rank"]))
            + (1 - alpha) * (1 / (RRF_K + info["sparse_rank"]))
        )
        results.append({
            "chunk_id" : cid,
            "payload"  : info["payload"],
            "rrf_score": rrf_score,
        })

    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results


def _rerank(
    query     : str,
    candidates: list[dict],
    reranker  : CrossEncoderReranker,
    top_k     : int,
) -> list[dict]:
    if not candidates:
        return []

    texts  = [c["payload"].get("text", c["payload"].get("chunk_text", "")) for c in candidates]
    pairs  = [[query, t] for t in texts]
    scores = reranker.compute_score(pairs, normalize=True)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [item for _, item in ranked[:top_k]]


def _fetch_parent_texts(
    candidates: list[dict],
    client    : QdrantClient,
    parent_collection: str = COLLECTION_JO,
) -> list[dict]:
    """parent_id로 조 단위 텍스트를 조회해 payload["text"]를 교체."""
    parent_ids = list({
        c["payload"].get("parent_id")
        for c in candidates
        if c["payload"].get("parent_id")
    })

    if not parent_ids:
        return candidates

    parent_texts: dict[str, str] = {}
    try:
        for parent_id in parent_ids:
            results = client.scroll(
                collection_name=parent_collection,
                scroll_filter={
                    "must": [{"key": "chunk_id", "match": {"value": parent_id}}]
                },
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            points = results[0]
            if points:
                p = points[0].payload
                parent_texts[parent_id] = p.get("text", p.get("chunk_text", ""))
    except Exception as e:
        print(f"  ⚠️  parent fetch 실패: {e}")
        return candidates

    updated = []
    for c in candidates:
        pid = c["payload"].get("parent_id")
        if pid and pid in parent_texts:
            updated_payload         = dict(c["payload"])
            updated_payload["text"] = parent_texts[pid]
            updated.append({**c, "payload": updated_payload})
        else:
            updated.append(c)

    return updated


def _build_law_refs(
    candidates : list[dict],
    laws_ref   : dict[str, dict],
    top_k      : int,
    with_xref  : bool = False,
) -> list[LawRef]:
    law_refs: list[LawRef] = []
    for c in candidates[:top_k]:
        payload  = c["payload"]
        chunk_id = payload.get("chunk_id", "")
        ref_meta = laws_ref.get(chunk_id, {})

        law_refs.append(LawRef(
            chunk_id    = chunk_id,
            article     = ref_meta.get("article",  payload.get("article_number", "")),
            category    = ref_meta.get("category", payload.get("category", "")),
            law_name    = payload.get("law_name",  ""),
            chunk_text  = payload.get("text", payload.get("chunk_text", "")),
            score       = round(float(c.get("rrf_score", 0.0)), 4),
            is_risk_ref = bool(payload.get("is_risk_ref", False)),
            parent_id   = payload.get("parent_id", "") or "",
            cross_refs  = payload.get("cross_refs", []) if with_xref else [],
        ))

    return law_refs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAG 1: JoRAG — 조 단위 검색
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _search_jo(
    clause_text: str,
    client     : QdrantClient,
    model      : BGEM3FlagModel,
    laws_ref   : dict[str, dict],
    reranker1  : CrossEncoderReranker | None = None,
    reranker2  : CrossEncoderReranker | None = None,
    top_k      : int   = TOP_K,
    alpha      : float = RRF_ALPHA,
) -> list[LawRef]:
    """
    JoRAG: law_kb_jo에서 조 단위로 직접 검색.
    parent fetch 없음 — 이미 조 단위가 최상위.
    """
    candidates = _hybrid_search(clause_text, client, model, COLLECTION_JO, FETCH_K, alpha)

    if reranker1 and candidates:
        candidates = _rerank(clause_text, candidates, reranker1, RERANK1_K)
    if reranker2 and candidates:
        candidates = _rerank(clause_text, candidates, reranker2, RERANK2_K)

    return _build_law_refs(candidates, laws_ref, top_k, with_xref=False)


def review_contract_jo(
    contract_text: str,
    client       : QdrantClient,
    model        : BGEM3FlagModel,
    laws_ref     : dict[str, dict] | None = None,
    reranker1    : CrossEncoderReranker | None = None,
    reranker2    : CrossEncoderReranker | None = None,
    top_k        : int   = TOP_K,
    alpha        : float = RRF_ALPHA,
) -> list[ClauseResult]:
    """JoRAG 메인 인터페이스."""
    if laws_ref is None:
        laws_ref = load_laws_ref()

    clauses = chunk_contract(contract_text)
    results : list[ClauseResult] = []
    print(f"[JoRAG] 총 {len(clauses)}개 청크 검색 중...")

    for i, clause in enumerate(clauses, 1):
        print(f"  [{i}/{len(clauses)}] {clause['clause_number']} ...", end="\r")

        law_refs   = _search_jo(
            clause["clause_text"], client, model, laws_ref,
            reranker1, reranker2, top_k, alpha,
        )
        categories = list(dict.fromkeys(r.category for r in law_refs if r.category))

        results.append(ClauseResult(
            clause_number=clause["clause_number"],
            clause_text  =clause["clause_text"],
            law_refs     =law_refs,
            categories   =categories,
        ))

    print(f"\n[JoRAG] ✅ 완료")
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAG 2: HoRAG — 호 단위 검색 + parent fetch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _search_ho(
    clause_text: str,
    client     : QdrantClient,
    model      : BGEM3FlagModel,
    laws_ref   : dict[str, dict],
    reranker1  : CrossEncoderReranker | None = None,
    reranker2  : CrossEncoderReranker | None = None,
    top_k      : int   = TOP_K,
    alpha      : float = RRF_ALPHA,
) -> list[LawRef]:
    """
    HoRAG: law_kb_ho에서 호 단위 검색 후
    parent_id로 law_kb_jo에서 조 전체 텍스트 fetch.
    """
    candidates = _hybrid_search(clause_text, client, model, COLLECTION_HO, FETCH_K, alpha)

    if reranker1 and candidates:
        candidates = _rerank(clause_text, candidates, reranker1, RERANK1_K)

    # parent fetch: 호 → 조 텍스트로 교체
    candidates = _fetch_parent_texts(candidates, client, parent_collection=COLLECTION_JO)

    if reranker2 and candidates:
        candidates = _rerank(clause_text, candidates, reranker2, RERANK2_K)

    return _build_law_refs(candidates, laws_ref, top_k, with_xref=False)


def review_contract_ho(
    contract_text: str,
    client       : QdrantClient,
    model        : BGEM3FlagModel,
    laws_ref     : dict[str, dict] | None = None,
    reranker1    : CrossEncoderReranker | None = None,
    reranker2    : CrossEncoderReranker | None = None,
    top_k        : int   = TOP_K,
    alpha        : float = RRF_ALPHA,
) -> list[ClauseResult]:
    """HoRAG 메인 인터페이스."""
    if laws_ref is None:
        laws_ref = load_laws_ref()

    clauses = chunk_contract(contract_text)
    results : list[ClauseResult] = []
    print(f"[HoRAG] 총 {len(clauses)}개 청크 검색 중...")

    for i, clause in enumerate(clauses, 1):
        print(f"  [{i}/{len(clauses)}] {clause['clause_number']} ...", end="\r")

        law_refs   = _search_ho(
            clause["clause_text"], client, model, laws_ref,
            reranker1, reranker2, top_k, alpha,
        )
        categories = list(dict.fromkeys(r.category for r in law_refs if r.category))

        results.append(ClauseResult(
            clause_number=clause["clause_number"],
            clause_text  =clause["clause_text"],
            law_refs     =law_refs,
            categories   =categories,
        ))

    print(f"\n[HoRAG] ✅ 완료")
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAG 3: HoXrefRAG — 호 단위 + cross_refs + parent fetch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _expand_with_cross_refs(
    candidates: list[dict],
    client    : QdrantClient,
) -> list[dict]:
    """
    각 후보의 cross_refs에 있는 chunk_id를 law_kb_ho_xref에서 추가 조회.
    이미 후보에 있는 chunk_id는 중복 추가하지 않음.
    추가된 항목의 rrf_score는 원본의 절반 (참조 가중치 낮춤).
    """
    existing_ids  = {c["chunk_id"] for c in candidates}
    extra_chunks  : list[dict] = []
    ref_ids_total : list[str]  = []

    for c in candidates:
        cross_refs = c["payload"].get("cross_refs", [])
        for ref_id in cross_refs:
            if ref_id not in existing_ids and ref_id not in ref_ids_total:
                ref_ids_total.append(ref_id)

    if not ref_ids_total:
        return candidates

    try:
        for ref_id in ref_ids_total:
            results = client.scroll(
                collection_name=COLLECTION_HO_XREF,
                scroll_filter={
                    "must": [{"key": "chunk_id", "match": {"value": ref_id}}]
                },
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            points = results[0]
            if points:
                p = points[0].payload
                extra_chunks.append({
                    "chunk_id" : ref_id,
                    "payload"  : p,
                    "rrf_score": 0.0,   # 리랭크에서 점수 재계산됨
                })
                existing_ids.add(ref_id)
    except Exception as e:
        print(f"  ⚠️  cross_ref fetch 실패: {e}")

    return candidates + extra_chunks


def _search_ho_xref(
    clause_text: str,
    client     : QdrantClient,
    model      : BGEM3FlagModel,
    laws_ref   : dict[str, dict],
    reranker1  : CrossEncoderReranker | None = None,
    reranker2  : CrossEncoderReranker | None = None,
    top_k      : int   = TOP_K,
    alpha      : float = RRF_ALPHA,
) -> list[LawRef]:
    """
    HoXrefRAG: law_kb_ho_xref에서 호 단위 검색
    → cross_refs 확장 → parent fetch (조 텍스트) → 2단계 리랭크.
    """
    candidates = _hybrid_search(clause_text, client, model, COLLECTION_HO_XREF, FETCH_K, alpha)

    if reranker1 and candidates:
        candidates = _rerank(clause_text, candidates, reranker1, RERANK1_K)

    # cross_refs 확장 (참조 조문 추가 수집)
    candidates = _expand_with_cross_refs(candidates, client)

    # parent fetch: 호 → 조 텍스트로 교체
    candidates = _fetch_parent_texts(candidates, client, parent_collection=COLLECTION_JO)

    if reranker2 and candidates:
        candidates = _rerank(clause_text, candidates, reranker2, RERANK2_K)

    return _build_law_refs(candidates, laws_ref, top_k, with_xref=True)


def review_contract_ho_xref(
    contract_text: str,
    client       : QdrantClient,
    model        : BGEM3FlagModel,
    laws_ref     : dict[str, dict] | None = None,
    reranker1    : CrossEncoderReranker | None = None,
    reranker2    : CrossEncoderReranker | None = None,
    top_k        : int   = TOP_K,
    alpha        : float = RRF_ALPHA,
) -> list[ClauseResult]:
    """HoXrefRAG 메인 인터페이스."""
    if laws_ref is None:
        laws_ref = load_laws_ref()

    clauses = chunk_contract(contract_text)
    results : list[ClauseResult] = []
    print(f"[HoXrefRAG] 총 {len(clauses)}개 청크 검색 중...")

    for i, clause in enumerate(clauses, 1):
        print(f"  [{i}/{len(clauses)}] {clause['clause_number']} ...", end="\r")

        law_refs   = _search_ho_xref(
            clause["clause_text"], client, model, laws_ref,
            reranker1, reranker2, top_k, alpha,
        )
        categories = list(dict.fromkeys(r.category for r in law_refs if r.category))

        results.append(ClauseResult(
            clause_number=clause["clause_number"],
            clause_text  =clause["clause_text"],
            law_refs     =law_refs,
            categories   =categories,
        ))

    print(f"\n[HoXrefRAG] ✅ 완료")
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JSON 변환
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def results_to_json(results: list[ClauseResult]) -> list[dict]:
    return [asdict(r) for r in results]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 편의: 단일 조항 검색 (개별 RAG)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def search_jo(clause_text: str, client: QdrantClient, model: BGEM3FlagModel,
              laws_ref: dict, reranker1=None, reranker2=None,
              top_k=TOP_K, alpha=RRF_ALPHA) -> list[LawRef]:
    return _search_jo(clause_text, client, model, laws_ref, reranker1, reranker2, top_k, alpha)


def search_ho(clause_text: str, client: QdrantClient, model: BGEM3FlagModel,
              laws_ref: dict, reranker1=None, reranker2=None,
              top_k=TOP_K, alpha=RRF_ALPHA) -> list[LawRef]:
    return _search_ho(clause_text, client, model, laws_ref, reranker1, reranker2, top_k, alpha)


def search_ho_xref(clause_text: str, client: QdrantClient, model: BGEM3FlagModel,
                   laws_ref: dict, reranker1=None, reranker2=None,
                   top_k=TOP_K, alpha=RRF_ALPHA) -> list[LawRef]:
    return _search_ho_xref(clause_text, client, model, laws_ref, reranker1, reranker2, top_k, alpha)