"""
Workit - 계약서 검토 RAG 파이프라인 (2-flow, sweep 가능한 버전 + 캐싱)
파일명: rag/yoonha_law_rag.py

전체 흐름 (다이어그램 기준):
  1) JoRAG   : law_kb_jo_fixedid 에서 "조" 단위로 직접 검색 (넓은 단위, 청킹 크기 큼)
  2) HoRAG   : law_kb_ho_fixedid 에서 "호/목/세목"까지 쪼갠 세부 단위로 검색
               → 후보들의 cross_refs로 관련 조항 추가 확장 (별도 xref 컬렉션 없음,
                 cross_refs가 이미 ho payload 안에 들어있음)
               → 항상 parent_chunk_id로 "조" 텍스트를 fetch해서 최종 출력 단위를
                 조로 통일 (하위 단위는 검색에만 쓰고 최종 결과로는 노출 안 함)
  두 flow 모두 계약서 조항 1개당 결과를 병합해서 list[ClauseResult]로 반환.

이번 개정 (2026-07-04, reranker 단일화 + sweep 축 확장):
  - reranker를 bge-reranker-v2-m3 하나로 고정. 기존 reranker1(ko-reranker)/
    reranker2 이원 구조를 전부 걷어내고 reranker 파라미터 하나로 통합했다.
    JoRAG/HoRAG 둘 다 이 reranker 하나만 쓴다.
  - RRF_K가 _hybrid_search 안에 하드코딩(60)돼 있던 걸 함수 인자로 뺐다.
    alpha와 마찬가지로 RRF 합산 단계에서만 쓰이므로, raw_search 캐시를
    전혀 건드리지 않고 sweep 가능하다 (사실상 공짜 축).
  - HoRAG의 cross-ref 확장에 토큰 예산(max_cross_ref_tokens) 개념을 추가했다.
    "몇 개까지 추가할지"가 아니라 "reranker가 실제로 받는 총 토큰이 예산을
    넘지 않는 선까지 추가"하는 방식이다. 원본 후보(hybrid search로 직접 찾은
    것)는 무조건 유지하고, cross-ref로 딸려온 추가 후보만 트리밍 대상이다.
    우선순위는 "어떤 원본 후보가 인용한 것인지"의 rrf_score(source_score)가
    높은 순.
  - 이 트리밍은 parent fetch *이후*에 수행한다. cross-ref 확장 시점(parent
    fetch 전)에는 아직 호/목 단위의 짧은 텍스트라서 토큰 수를 재봤자 의미가
    없다 — parent fetch로 조 전체 텍스트로 바뀐 뒤에야 실제로 reranker가
    보게 될 길이를 알 수 있다.
  - top_k는 기존과 동일하게 "이미 순위 매겨진 후보 리스트를 자르기만" 하는
    구조라 별도 캐싱/재검색이 필요 없다. sweep 스크립트에서는 top_k=1/3/5/10을
    동일한 한 번의 검색 결과에서 전부 계산하면 된다 (재검색 불필요).

sweep 가능하게 바뀐 점 (이전 버전과 공통):
  - alpha, use_reranker, rerank_k, fetch_k, top_k, rrf_k, max_cross_ref_tokens을
    전부 모듈 상수가 아니라 함수 인자로 뺐다.
  - 모듈 상수는 "sweep 안 할 때 쓰는 기본값" 역할만 한다 (DEFAULT_* 접두사).

캐싱 구조:
  - alpha, rrf_k는 RRF 가중치 합산에만 쓰이고, 임베딩/Qdrant raw 검색 결과와는
    무관하다. 그래서 이 두 값을 sweep할 때 (collection, query_text, fetch_k)별
    raw 검색 결과를 한 번만 구하고, 조합마다는 캐시된 raw 결과로 RRF 점수만
    다시 계산한다.
  - reranker 점수는 (reranker_name, query_text, chunk_id) 쌍에만 의존한다.
    alpha/rrf_k/max_cross_ref_tokens이 바뀌면 어떤 chunk_id가 rerank_k 안에
    들어오는지는 달라질 수 있지만, 같은 chunk_id에 대한 점수 자체는 항상 같다.
    그래서 미스(cache miss)만 골라서 계산하고 나머지는 캐시에서 꺼내 쓴다.
  - parent fetch(조 텍스트)와 cross-ref 확장(scroll 조회)도 chunk_id -> payload
    조회라 alpha/rrf_k/쿼리와 무관하게 전역 캐시가 가능하다. 여러 조항이 같은
    법 조항을 참조하는 경우가 많아서 쿼리 간에도 재사용된다.
  - fetch_k는 예외다 — raw_search 캐시 키에 fetch_k가 포함돼 있어서, fetch_k
    값이 바뀌면 Qdrant 검색 자체가 다시 돈다 (alpha/rrf_k처럼 공짜가 아님).
  - cache=None으로 호출하면 캐싱 없이 기존과 완전히 동일하게 동작한다.

컬렉션:
  - JoRAG : law_kb_jo_fixedid (조 단위, parent 없음)
  - HoRAG : law_kb_ho_fixedid (호/목/세목 단위 + cross_refs, parent fetch → 조 텍스트)

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
from qdrant_client.models import SparseVector, Filter, FieldCondition, MatchValue

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 경로 / 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_THIS_DIR     = Path(__file__).resolve().parent
_DATA_DIR     = _THIS_DIR.parent / "data"
LAWS_REF_PATH = _DATA_DIR / "hn_seed" / "law_refs.json"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_JO = "law_kb_jo_fixedid"
COLLECTION_HO = "law_kb_ho_fixedid"
# HoXrefRAG는 별도 컬렉션 없이 COLLECTION_HO를 그대로 씀 (cross_refs가 payload에 이미 있음)

EMBED_MODEL = "BAAI/bge-m3"

# sweep 안 할 때 쓰는 기본값. 실제 최적값은 yoonha_rag_eval.py로 찾는다.
DEFAULT_FETCH_K              = 50
DEFAULT_RERANK_K             = 10
DEFAULT_TOP_K                = 10
DEFAULT_ALPHA                = 0.5    # 1.0 = dense only, 0.0 = sparse only
DEFAULT_RRF_K                = 60     # RRF 공식의 스무딩 상수 (기존 하드코딩값)
DEFAULT_MAX_CROSS_REF_TOKENS = None   # None = 트리밍 없음 (기존과 동일 동작)

# reranker는 이제 이거 하나만 쓴다 (ko-reranker/reranker1 계열 제거).
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 확정된 최적 조합
# reranker가 하나로 통합됐으므로, use_reranker on/off만 남는다.
# fetch_k / rerank_k / rrf_k / max_cross_ref_tokens의 최종값은
# yoonha_rag_eval.py 재sweep 결과로 다시 채워 넣을 것.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BEST_JO_CONFIG = dict(
    use_reranker=True,
    alpha=DEFAULT_ALPHA,
)

BEST_HO_CONFIG = dict(
    use_reranker=True,
    alpha=DEFAULT_ALPHA,
    use_cross_refs=True,
    max_cross_ref_tokens=DEFAULT_MAX_CROSS_REF_TOKENS,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sweep 캐시
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SweepCache:
    """
    alpha × rrf_k × reranker on/off × fetch_k × ... sweep에서 재사용 가능한
    중간 계산 결과를 담는 캐시.

    - embed        : query_text -> (dense_vec, sparse_vec)
                      (임베딩은 collection/alpha/rrf_k와 무관 — jo/ho variant 간에도 공유됨)
    - raw_search   : (collection, query_text, fetch_k) -> (dense_points, sparse_points)
                      (alpha/rrf_k와 무관한 Qdrant 원본 검색 결과. RRF 합산만 alpha/rrf_k
                       조합마다 다시 함. fetch_k가 바뀌면 캐시 키 자체가 달라짐 — 재검색 필요.)
    - scroll       : (collection, chunk_id) -> payload
                      (parent fetch / cross-ref 확장에서 쓰는 단건 조회. 여러 쿼리에서
                       같은 법 조항을 참조하면 자동으로 재사용됨)
    - rerank_score : (reranker_name, query_text, chunk_id) -> score
                      (reranker cross-encoder 점수. alpha/rrf_k/max_cross_ref_tokens이
                       달라져도 같은 chunk_id에 대한 점수는 항상 같으므로 미스만 계산)

    cache=None으로 함수를 호출하면 캐싱 없이 기존과 동일하게 동작한다.
    """
    embed        : dict = field(default_factory=dict)
    raw_search   : dict = field(default_factory=dict)
    scroll       : dict = field(default_factory=dict)
    rerank_score : dict = field(default_factory=dict)

    def stats(self) -> dict:
        return {
            "embed_cached":      len(self.embed),
            "raw_search_cached": len(self.raw_search),
            "scroll_cached":     len(self.scroll),
            "rerank_cached":     len(self.rerank_score),
        }


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

    def count_tokens(self, text: str) -> int:
        """토큰 예산 계산용. truncation 없이 실제 토큰 수를 그대로 센다."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))


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
    cross_refs : list[str] = field(default_factory=list)  # HoRAG(xref 확장) 전용


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


def load_reranker(device: str = "cpu") -> CrossEncoderReranker:
    """
    reranker는 로드 비용이 커서 sweep 중에는 한 번만 로드해두고,
    실제로 쓸지 말지는 각 검색 함수의 use_reranker 플래그로 토글한다.
    """
    print(f"📦 Re-ranker 로드: {RERANKER_MODEL}")
    return CrossEncoderReranker(RERANKER_MODEL, device=device)


def derive_jo_id(chunk_id: str) -> str:
    """
    ho-level chunk_id 문자열에서 그 조에 해당하는 jo-level chunk_id를 직접 역산한다.

    ho id 형식: {prefix}_{장}_{절}_{조}_{항}_{호}_{목}_{세목} (8토큰 고정)
    jo id 형식: 일반 법령은 {prefix}_{장}_{절}_{조} (앞 4토큰),
                PYG(예규)는 조가 없어 항이 anchor이므로 {prefix}_{장}_{절}_{조=0}_{항} (앞 5토큰).

    주의: payload의 parent_chunk_id 필드는 이 용도로 쓰면 안 된다. 실제 데이터를
    검증해보면 parent_chunk_id는 JO 컬렉션이 아니라 HO 컬렉션 자기 자신 안의 다른
    chunk(조 단위로 롤업된 chunk, 혹은 중간 단계인 호/목)를 가리키고 있어서 JO
    컬렉션 chunk_id와 절대 일치하지 않는다 (검증: ho 7640개 중 parent_chunk_id가
    JO chunk_id와 일치한 건 0개). 반면 이 함수처럼 chunk_id 자신의 앞쪽 토큰만
    잘라내는 방식은 ho 7640개 전부 100% 올바른 jo_id로 매핑된다 (leaf 청크든
    조 단위 롤업 청크든 동일하게 성립).
    """
    tokens = chunk_id.split("_")
    if tokens[0] == "PYG":
        return "_".join(tokens[:5])
    return "_".join(tokens[:4])


def get_vectors(
    text : str,
    model: BGEM3FlagModel,
    cache: SweepCache | None = None,
) -> tuple[list[float], dict[int, float]]:
    """
    BGE-M3 dense/sparse 임베딩. collection이나 alpha/rrf_k와 무관하므로
    query_text만으로 캐시 가능 (jo/ho variant 간에도 재사용됨).
    """
    if cache is not None and text in cache.embed:
        return cache.embed[text]

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

    if cache is not None:
        cache.embed[text] = (dense_vec, sparse_vec)

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
    fetch_k    : int,
    alpha      : float,
    rrf_k      : int = DEFAULT_RRF_K,
    cache      : SweepCache | None = None,
) -> list[dict]:
    """
    Dense + Sparse 하이브리드 검색 (수동 RRF). alpha=1.0이면 dense만, 0.0이면 sparse만.

    alpha와 rrf_k는 아래 RRF 합산 단계에서만 쓰이므로, Qdrant raw 검색 결과
    (dense_results, sparse_results)는 (collection, clause_text, fetch_k)가
    같으면 alpha/rrf_k와 무관하게 재사용 가능 — cache가 있으면 캐시에서 꺼내온다.
    (fetch_k가 바뀌면 캐시 키 자체가 달라지므로 이 값은 sweep 시 재검색이 발생한다.)
    """
    cache_key = (collection, clause_text, fetch_k)

    if cache is not None and cache_key in cache.raw_search:
        dense_results, sparse_results = cache.raw_search[cache_key]
    else:
        dense_vec, sparse_vec = get_vectors(clause_text, model, cache)
        indices = list(sparse_vec.keys())
        values  = list(sparse_vec.values())

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

        if cache is not None:
            cache.raw_search[cache_key] = (dense_results, sparse_results)

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
            alpha         * (1 / (rrf_k + info["dense_rank"]))
            + (1 - alpha) * (1 / (rrf_k + info["sparse_rank"]))
        )
        results.append({
            "chunk_id"    : cid,
            "payload"     : info["payload"],
            "rrf_score"   : rrf_score,
            "is_cross_ref": False,   # 원본 후보 표시 (cross-ref 트리밍에서 구분용)
        })

    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results


def _rerank(
    query        : str,
    candidates   : list[dict],
    reranker     : CrossEncoderReranker,
    top_k        : int,
    reranker_name: str = "reranker",
    cache        : SweepCache | None = None,
) -> list[dict]:
    """
    reranker_name은 캐시 키 네임스페이스 구분용 (예: "jo", "ho").
    같은 (reranker_name, query, chunk_id) 조합은 alpha/rrf_k가 달라져도 점수가
    동일하므로, cache가 있으면 미스(cache miss)만 계산하고 나머지는 재사용한다.
    """
    if not candidates:
        return []

    if cache is None:
        texts  = [c["payload"].get("text", c["payload"].get("chunk_text", "")) for c in candidates]
        pairs  = [[query, t] for t in texts]
        scores = reranker.compute_score(pairs, normalize=True)
    else:
        scores: list[float | None] = [None] * len(candidates)
        miss_idx: list[int] = []

        for i, c in enumerate(candidates):
            key = (reranker_name, query, c["chunk_id"])
            if key in cache.rerank_score:
                scores[i] = cache.rerank_score[key]
            else:
                miss_idx.append(i)

        if miss_idx:
            miss_texts = [
                candidates[i]["payload"].get("text", candidates[i]["payload"].get("chunk_text", ""))
                for i in miss_idx
            ]
            miss_pairs  = [[query, t] for t in miss_texts]
            miss_scores = reranker.compute_score(miss_pairs, normalize=True)

            for i, s in zip(miss_idx, miss_scores):
                scores[i] = s
                cache.rerank_score[(reranker_name, query, candidates[i]["chunk_id"])] = s

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [item for _, item in ranked[:top_k]]


def _fetch_parent_texts(
    candidates: list[dict],
    client    : QdrantClient,
    parent_collection: str = COLLECTION_JO,
    cache     : SweepCache | None = None,
) -> list[dict]:
    """
    각 후보 ho chunk_id에서 derive_jo_id()로 조 단위 chunk_id를 역산해,
    그 조 텍스트를 law_kb_jo_fixedid에서 조회해 payload["text"]를 교체한다.

    payload의 parent_chunk_id 필드는 쓰지 않는다 (derive_jo_id 함수 docstring 참고
    — 그 필드는 JO 컬렉션이 아니라 HO 컬렉션 자기 자신을 가리키고 있어서 이 용도로
    쓰면 매번 조회가 실패한다). chunk_id 자신의 앞쪽 토큰만으로 역산하는 방식은
    leaf 청크든 조 단위 롤업 청크든 상관없이 항상 올바른 jo_id를 준다.
    """
    jo_ids = list({
        derive_jo_id(c["payload"].get("chunk_id", c["chunk_id"]))
        for c in candidates
    })

    if not jo_ids:
        return candidates

    parent_texts: dict[str, str] = {}
    to_fetch: list[str] = []

    for jid in jo_ids:
        cache_key = (parent_collection, jid)
        if cache is not None and cache_key in cache.scroll:
            payload = cache.scroll[cache_key]
            parent_texts[jid] = payload.get("text", payload.get("chunk_text", ""))
        else:
            to_fetch.append(jid)

    try:
        for jid in to_fetch:
            results = client.scroll(
                collection_name=parent_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="chunk_id", match=MatchValue(value=jid))]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            points = results[0]
            if points:
                p = points[0].payload
                parent_texts[jid] = p.get("text", p.get("chunk_text", ""))
                if cache is not None:
                    cache.scroll[(parent_collection, jid)] = p
    except Exception as e:
        print(f"  ⚠️  parent fetch 실패: {e}")
        return candidates

    updated = []
    for c in candidates:
        jid = derive_jo_id(c["payload"].get("chunk_id", c["chunk_id"]))
        if jid in parent_texts:
            updated_payload         = dict(c["payload"])
            updated_payload["text"] = parent_texts[jid]
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
    """
    top_k는 여기서 "이미 순위 매겨진 candidates를 자르기만" 한다 — 재검색이나
    재계산이 전혀 없으므로, 같은 candidates에 대해 top_k=1/3/5/10을 전부
    별도 비용 없이 뽑아낼 수 있다 (sweep 스크립트에서 활용).
    """
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
            parent_id   = payload.get("parent_chunk_id", "") or "",
            cross_refs  = payload.get("cross_refs", []) if with_xref else [],
        ))

    return law_refs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAG 1: JoRAG — 조 단위 검색
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _search_jo(
    clause_text  : str,
    client       : QdrantClient,
    model        : BGEM3FlagModel,
    laws_ref     : dict[str, dict],
    reranker     : CrossEncoderReranker | None = None,
    use_reranker : bool  = False,
    top_k        : int   = DEFAULT_TOP_K,
    alpha        : float = DEFAULT_ALPHA,
    fetch_k      : int   = DEFAULT_FETCH_K,
    rerank_k     : int   = DEFAULT_RERANK_K,
    rrf_k        : int   = DEFAULT_RRF_K,
    cache        : SweepCache | None = None,
) -> list[LawRef]:
    """
    JoRAG: law_kb_jo_fixedid에서 조 단위로 직접 검색.
    parent fetch 없음 — 이미 조 단위가 최상위.
    """
    candidates = _hybrid_search(clause_text, client, model, COLLECTION_JO, fetch_k, alpha, rrf_k, cache)

    if use_reranker and reranker and candidates:
        candidates = _rerank(clause_text, candidates, reranker, rerank_k, "jo", cache)

    return _build_law_refs(candidates, laws_ref, top_k, with_xref=False)


def review_contract_jo(
    contract_text: str,
    client       : QdrantClient,
    model        : BGEM3FlagModel,
    laws_ref     : dict[str, dict] | None = None,
    reranker     : CrossEncoderReranker | None = None,
    use_reranker : bool  = False,
    top_k        : int   = DEFAULT_TOP_K,
    alpha        : float = DEFAULT_ALPHA,
    fetch_k      : int   = DEFAULT_FETCH_K,
    rerank_k     : int   = DEFAULT_RERANK_K,
    rrf_k        : int   = DEFAULT_RRF_K,
    cache        : SweepCache | None = None,
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
            reranker, use_reranker,
            top_k, alpha, fetch_k, rerank_k, rrf_k, cache,
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
# RAG 2: HoRAG — 호/목/세목 단위 검색 + cross_refs 확장 + parent fetch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _expand_with_cross_refs(
    candidates: list[dict],
    client    : QdrantClient,
    cache     : SweepCache | None = None,
) -> list[dict]:
    """
    각 후보의 cross_refs(같은 law_kb_ho_fixedid payload 안의 필드)에 있는
    chunk_id를 추가 조회. 이미 후보에 있는 chunk_id는 중복 추가하지 않음.

    여기서는 개수/토큰 제한을 걸지 않고 전부 모은다 — 토큰 예산 트리밍은
    parent fetch 이후(_trim_cross_refs_by_token_budget)에서 한다. 이 시점의
    텍스트는 아직 호/목 단위라 parent fetch 후 조 전체 텍스트로 바뀌면
    길이가 크게 달라지기 때문에, 여기서 트리밍하면 실제 reranker가 보게 될
    길이 기준과 안 맞다.

    추가된 항목에는 "어떤 원본 후보가 이 조항을 인용했는지"의 rrf_score를
    source_score로 같이 기록해둔다 — 트리밍 시 우선순위(원 후보 점수가 높을수록
    먼저 살림)로 쓰기 위함. 여러 원본 후보가 같은 chunk_id를 인용하면
    source_score가 가장 높은 것을 기록한다.

    chunk_id -> payload 조회라 쿼리/alpha와 무관 — 여러 조항이 같은 참조를
    가지면 캐시에서 재사용된다.
    """
    existing_ids = {c["chunk_id"] for c in candidates}
    best_source_score: dict[str, float] = {}

    for c in candidates:
        cross_refs = c["payload"].get("cross_refs", [])
        for ref_id in cross_refs:
            if ref_id in existing_ids:
                continue
            prev = best_source_score.get(ref_id, float("-inf"))
            if c["rrf_score"] > prev:
                best_source_score[ref_id] = c["rrf_score"]

    ref_ids_total = list(best_source_score.keys())
    if not ref_ids_total:
        return candidates

    payload_by_ref: dict[str, dict] = {}
    to_fetch: list[str] = []

    for ref_id in ref_ids_total:
        cache_key = (COLLECTION_HO, ref_id)
        if cache is not None and cache_key in cache.scroll:
            payload_by_ref[ref_id] = cache.scroll[cache_key]
        else:
            to_fetch.append(ref_id)

    try:
        for ref_id in to_fetch:
            results = client.scroll(
                collection_name=COLLECTION_HO,
                scroll_filter=Filter(
                    must=[FieldCondition(key="chunk_id", match=MatchValue(value=ref_id))]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            points = results[0]
            if points:
                p = points[0].payload
                payload_by_ref[ref_id] = p
                if cache is not None:
                    cache.scroll[(COLLECTION_HO, ref_id)] = p
    except Exception as e:
        print(f"  ⚠️  cross_ref fetch 실패: {e}")

    extra_chunks = []
    for ref_id, p in payload_by_ref.items():
        extra_chunks.append({
            "chunk_id"    : ref_id,
            "payload"     : p,
            "rrf_score"   : 0.0,                       # 리랭크에서 점수 재계산됨
            "is_cross_ref": True,                       # 트리밍 대상 표시
            "source_score": best_source_score[ref_id],   # 우선순위 기준
        })

    return candidates + extra_chunks


def _trim_cross_refs_by_token_budget(
    candidates: list[dict],
    reranker  : CrossEncoderReranker,
    max_tokens: int | None,
) -> list[dict]:
    """
    parent fetch 이후(조 전체 텍스트로 교체된 뒤)에 호출한다.

    원본 후보(is_cross_ref=False, hybrid search로 직접 찾은 것)는 무조건
    유지하고, cross-ref로 추가된 후보만 source_score(이 조항을 인용한 원본
    후보의 rrf_score) 내림차순으로 정렬해서, 누적 토큰 수가 max_tokens를
    넘기 전까지만 살린다.

    max_tokens=None이면 트리밍 없이 그대로 반환 (기존 동작과 동일).
    """
    if max_tokens is None:
        return candidates

    originals = [c for c in candidates if not c.get("is_cross_ref")]
    extras    = [c for c in candidates if c.get("is_cross_ref")]

    if not extras:
        return candidates

    extras.sort(key=lambda c: c.get("source_score", 0.0), reverse=True)

    used_tokens = sum(
        reranker.count_tokens(c["payload"].get("text", c["payload"].get("chunk_text", "")))
        for c in originals
    )

    kept_extras = []
    for c in extras:
        text = c["payload"].get("text", c["payload"].get("chunk_text", ""))
        n_tokens = reranker.count_tokens(text)
        if used_tokens + n_tokens > max_tokens:
            continue
        used_tokens += n_tokens
        kept_extras.append(c)

    return originals + kept_extras


def _search_ho(
    clause_text  : str,
    client       : QdrantClient,
    model        : BGEM3FlagModel,
    laws_ref     : dict[str, dict],
    reranker     : CrossEncoderReranker | None = None,
    use_reranker : bool  = False,
    use_cross_refs: bool = True,
    max_cross_ref_tokens: int | None = DEFAULT_MAX_CROSS_REF_TOKENS,
    top_k        : int   = DEFAULT_TOP_K,
    alpha        : float = DEFAULT_ALPHA,
    fetch_k      : int   = DEFAULT_FETCH_K,
    rerank_k     : int   = DEFAULT_RERANK_K,
    rrf_k        : int   = DEFAULT_RRF_K,
    cache        : SweepCache | None = None,
) -> list[LawRef]:
    """
    HoRAG: law_kb_ho_fixedid에서 호/목/세목 단위 검색
    → (옵션) cross_refs로 관련 조항 확장 (무제한으로 일단 모음)
    → parent_chunk_id로 law_kb_jo_fixedid에서 조 전체 텍스트 fetch
      (최종 출력 단위는 항상 조 — 하위 단위는 검색 후보로만 쓰고 결과로는 안 남김)
    → (옵션) cross-ref로 추가된 후보만 토큰 예산 기준으로 트리밍
      (원본 후보는 항상 유지, parent fetch 이후 실제 조 텍스트 길이 기준으로 자름)
    → reranker
    """
    candidates = _hybrid_search(clause_text, client, model, COLLECTION_HO, fetch_k, alpha, rrf_k, cache)

    if use_cross_refs:
        candidates = _expand_with_cross_refs(candidates, client, cache)

    # parent fetch: 호/목/세목 → 조 텍스트로 교체 (최종 출력 단위 통일)
    candidates = _fetch_parent_texts(candidates, client, parent_collection=COLLECTION_JO, cache=cache)

    if use_cross_refs and max_cross_ref_tokens is not None:
        if reranker is not None:
            candidates = _trim_cross_refs_by_token_budget(candidates, reranker, max_cross_ref_tokens)
        else:
            print("  ⚠️  max_cross_ref_tokens가 설정됐지만 reranker가 없어 토큰 트리밍을 건너뜁니다.")

    if use_reranker and reranker and candidates:
        candidates = _rerank(clause_text, candidates, reranker, rerank_k, "ho", cache)

    return _build_law_refs(candidates, laws_ref, top_k, with_xref=use_cross_refs)


def review_contract_ho(
    contract_text: str,
    client       : QdrantClient,
    model        : BGEM3FlagModel,
    laws_ref     : dict[str, dict] | None = None,
    reranker     : CrossEncoderReranker | None = None,
    use_reranker : bool  = False,
    use_cross_refs: bool = True,
    max_cross_ref_tokens: int | None = DEFAULT_MAX_CROSS_REF_TOKENS,
    top_k        : int   = DEFAULT_TOP_K,
    alpha        : float = DEFAULT_ALPHA,
    fetch_k      : int   = DEFAULT_FETCH_K,
    rerank_k     : int   = DEFAULT_RERANK_K,
    rrf_k        : int   = DEFAULT_RRF_K,
    cache        : SweepCache | None = None,
) -> list[ClauseResult]:
    """HoRAG 메인 인터페이스 (cross_refs 확장 포함, 별도 xref variant 없음)."""
    if laws_ref is None:
        laws_ref = load_laws_ref()

    clauses = chunk_contract(contract_text)
    results : list[ClauseResult] = []
    print(f"[HoRAG] 총 {len(clauses)}개 청크 검색 중...")

    for i, clause in enumerate(clauses, 1):
        print(f"  [{i}/{len(clauses)}] {clause['clause_number']} ...", end="\r")

        law_refs   = _search_ho(
            clause["clause_text"], client, model, laws_ref,
            reranker, use_reranker, use_cross_refs, max_cross_ref_tokens,
            top_k, alpha, fetch_k, rerank_k, rrf_k, cache,
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
# JSON 변환
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def results_to_json(results: list[ClauseResult]) -> list[dict]:
    return [asdict(r) for r in results]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 편의: 단일 조항 검색 (sweep 스크립트에서 이 함수들을 직접 호출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def search_jo(clause_text: str, client: QdrantClient, model: BGEM3FlagModel,
              laws_ref: dict, reranker=None, use_reranker=False,
              top_k=DEFAULT_TOP_K, alpha=DEFAULT_ALPHA, fetch_k=DEFAULT_FETCH_K,
              rerank_k=DEFAULT_RERANK_K, rrf_k=DEFAULT_RRF_K,
              cache: SweepCache | None = None) -> list[LawRef]:
    return _search_jo(clause_text, client, model, laws_ref, reranker,
                       use_reranker, top_k, alpha, fetch_k, rerank_k, rrf_k,
                       cache)


def search_ho(clause_text: str, client: QdrantClient, model: BGEM3FlagModel,
              laws_ref: dict, reranker=None, use_reranker=False, use_cross_refs=True,
              max_cross_ref_tokens=DEFAULT_MAX_CROSS_REF_TOKENS,
              top_k=DEFAULT_TOP_K, alpha=DEFAULT_ALPHA, fetch_k=DEFAULT_FETCH_K,
              rerank_k=DEFAULT_RERANK_K, rrf_k=DEFAULT_RRF_K,
              cache: SweepCache | None = None) -> list[LawRef]:
    return _search_ho(clause_text, client, model, laws_ref, reranker,
                       use_reranker, use_cross_refs, max_cross_ref_tokens,
                       top_k, alpha, fetch_k, rerank_k, rrf_k, cache)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 프로덕션 인터페이스 — 확정된 조합(BEST_JO_CONFIG / BEST_HO_CONFIG)을 그대로
# 적용한 고수준 함수. 호출부에서 alpha/reranker on-off 조합을 매번 기억할
# 필요 없이 이 두 함수만 쓰면 된다.
#
#   review_contract()         : 메인 경로 — 계약서 전체를 조 단위로 검토.
#   get_detailed_citations()  : 보조 경로 — 계약서 조항 1개에 대해 호/목 단위
#                                 세부 근거가 필요할 때만 호출 (사용자가 "세부
#                                 근거 보기" 등을 요청했을 때 온디맨드로 사용).
#
# 주의: reranker가 하나로 통합됐으므로, 재sweep 후 BEST_JO_CONFIG/BEST_HO_CONFIG의
# fetch_k/rerank_k/rrf_k/max_cross_ref_tokens 값을 확정해서 채워 넣을 것.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def review_contract(
    contract_text: str,
    client       : QdrantClient,
    model        : BGEM3FlagModel,
    reranker     : CrossEncoderReranker,
    laws_ref     : dict[str, dict] | None = None,
    top_k        : int = DEFAULT_TOP_K,
    cache        : SweepCache | None = None,
) -> list[ClauseResult]:
    """메인 검토 경로. BEST_JO_CONFIG를 그대로 적용한 JoRAG 호출이다."""
    return review_contract_jo(
        contract_text, client, model, laws_ref,
        reranker=reranker,
        top_k=top_k, cache=cache,
        **BEST_JO_CONFIG,
    )


def get_detailed_citations(
    clause_text  : str,
    client       : QdrantClient,
    model        : BGEM3FlagModel,
    reranker     : CrossEncoderReranker,
    laws_ref     : dict[str, dict] | None = None,
    top_k        : int = DEFAULT_TOP_K,
    cache        : SweepCache | None = None,
) -> list[LawRef]:
    """
    보조 경로. 계약서 조항 1개(review_contract가 반환한 ClauseResult.clause_text)에
    대해 호/목 단위 세부 근거가 필요할 때만 호출한다. BEST_HO_CONFIG를 그대로
    적용한 HoRAG 호출이다.

    review_contract()와 매 요청마다 같이 부르지 말 것 — 세부 근거는 사용자가
    명시적으로 요청했을 때만 온디맨드로 호출하는 게 설계 의도다 (하이브리드
    서비스 구조 — Notion 문서 참고).
    """
    if laws_ref is None:
        laws_ref = load_laws_ref()

    return search_ho(
        clause_text, client, model, laws_ref,
        reranker=reranker,
        top_k=top_k, cache=cache,
        **BEST_HO_CONFIG,
    )