"""
Workit - law_kb Qdrant upsert 스크립트 (Hybrid: Dense + Sparse)
파일명: yoonha_law_upsert.py
위치:   Workit/rag/yoonha_law_upsert.py

데이터:
  data/export/chunks_ho.json      → 호 단위 payload (child 청크만)
  data/export/vectors_ho.npz      → dense 벡터 (child 청크만, N × 1024)
  data/export/sparse_weights_ho.json → BGE-M3 sparse lexical weights (child 청크만)

  data/export/chunks_jo.json      → 조 단위 payload (parent 청크만)
  data/export/vectors_jo.npz      → dense 벡터 (parent 청크, M × 1024)
  data/export/sparse_weights_jo.json → BGE-M3 sparse lexical weights (parent 청크)

컬렉션 구조:
  law_kb_ho — 호 단위 child 청크 (dense + sparse 벡터, 실제 검색 대상)
  law_kb_jo — 조 단위 parent 청크 (dense + sparse 벡터, Hierarchical RAG fetch용)

Hierarchical RAG 흐름:
  검색: law_kb_ho에서 호 단위로 hit
      → child payload의 parent_id 확인
      → law_kb_jo에서 조 단위 전체 텍스트 fetch
      → LLM에 조 단위 맥락 전달

실행:
  python rag/yoonha_law_upsert.py
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# 컬렉션 이름
#   COLLECTION_HO — 호 단위 child 청크 (검색 대상)
#   COLLECTION_JO — 조 단위 parent 청크 (Hierarchical RAG fetch용)
COLLECTION_HO = "law_kb_ho"
COLLECTION_JO = "law_kb_jo"

VECTOR_DIM  = 1024
BATCH_SIZE  = 64
EMBED_MODEL = "BAAI/bge-m3"  # sparse 변환용 토크나이저

_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR.parent / "data" / "export"

# 호 단위 파일
CHUNKS_HO_PATH  = _DATA_DIR / "chunks_ho.json"
VECTORS_HO_PATH = _DATA_DIR / "vectors_ho.npz"
SPARSE_HO_PATH  = _DATA_DIR / "sparse_weights_ho.json"

# 조 단위 파일
CHUNKS_JO_PATH  = _DATA_DIR / "chunks_jo.json"
VECTORS_JO_PATH = _DATA_DIR / "vectors_jo.npz"
SPARSE_JO_PATH  = _DATA_DIR / "sparse_weights_jo.json"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def to_sparse_vector(lexical_weights: dict, tokenizer) -> SparseVector:
    """
    BGE-M3 lexical_weights {token_str: weight} →
    Qdrant SparseVector {indices: [...], values: [...]}

    동일 token_id가 여러 token_str에서 나올 수 있으므로
    가중치를 합산해 Qdrant의 unique indices 제약을 준수.
    """
    id_to_weight: dict[int, float] = {}
    for token_str, weight in lexical_weights.items():
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if isinstance(token_id, int):
            id_to_weight[token_id] = id_to_weight.get(token_id, 0.0) + float(weight)

    return SparseVector(
        indices=list(id_to_weight.keys()),
        values=list(id_to_weight.values()),
    )


def load_chunks(path: Path) -> dict[str, dict]:
    """
    chunks JSON 로드 후 chunk_id 기준 중복 제거.
    Returns: {chunk_id: chunk_dict}
    """
    with open(path, encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)
    chunk_map = {}
    for c in chunks:
        chunk_map[c["chunk_id"]] = c
    return chunk_map


def load_vectors(npz_path: Path) -> tuple[dict[str, list[float]], list[str]]:
    """
    vectors.npz 로드.
    Returns:
        id_to_dense : {chunk_id: dense_vector}
        vector_ids  : chunk_id 순서 리스트 (sparse 매핑용)
    """
    npz        = np.load(npz_path)
    vectors    = npz["vectors"].astype(np.float32)
    vector_ids = npz["chunk_ids"].tolist()
    id_to_dense = {cid: vec.tolist() for cid, vec in zip(vector_ids, vectors)}
    return id_to_dense, vector_ids


def load_sparse(sparse_path: Path, vector_ids: list[str]) -> dict[str, dict]:
    """
    sparse_weights.json 로드.
    Returns: {chunk_id: lexical_weights_dict}
    """
    with open(sparse_path, encoding="utf-8") as f:
        sparse_list: list[dict] = json.load(f)
    return dict(zip(vector_ids, sparse_list))


def ensure_collection(
    client     : QdrantClient,
    collection : str,
    use_sparse : bool,
):
    """
    컬렉션이 이미 있으면 삭제 후 재생성.
    use_sparse=True면 dense + sparse 벡터 설정,
    False면 dense 단독 설정.
    """
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        print(f"  ⚠️  컬렉션 '{collection}' 이미 존재 → 재생성합니다.")
        client.delete_collection(collection)

    if use_sparse:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
        print(f"  ✅ 컬렉션 '{collection}' 생성 완료 (Dense + Sparse)")
    else:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"  ✅ 컬렉션 '{collection}' 생성 완료 (Dense only)")


def upsert_chunks(
    client      : QdrantClient,
    collection  : str,
    chunk_map   : dict[str, dict],
    id_to_dense : dict[str, list[float]],
    id_to_sparse: dict[str, dict] | None,
    tokenizer,
    start_id    : int = 0,
) -> int:
    """
    청크 리스트를 Qdrant에 batch upsert.

    벡터가 없는 chunk_id는 스킵.
    sparse가 None이면 dense만 upsert.

    Returns:
        다음 point_id 시작값 (ho + jo 연속 ID 부여용)
    """
    use_sparse = id_to_sparse is not None
    points: list[PointStruct] = []
    point_id = start_id
    count = 0

    for cid, chunk in chunk_map.items():
        if cid not in id_to_dense:
            # 임베딩에서 제외된 청크 (예: 텍스트가 없는 경우) 스킵
            continue

        payload = {
            "chunk_id":       cid,
            "law_name":       chunk.get("law_name",       ""),
            "article_id":     chunk.get("article_id",     ""),
            "article_number": chunk.get("article_number", ""),
            # payload 필드명을 "text"로 통일 (yoonha_contract_rag.py에서 "text"로 읽음)
            "text":           chunk.get("text",           ""),
            "is_parent":      bool(chunk.get("is_parent", False)),
            "parent_id":      chunk.get("parent_id"),       # Hierarchical RAG용
            "is_ref_article": bool(chunk.get("is_ref_article", False)),
            "is_upper_law":   bool(chunk.get("is_upper_law",   False)),
            "hierarchy":      chunk.get("hierarchy",      {}),
        }

        if use_sparse and cid in id_to_sparse:
            sparse_vec = to_sparse_vector(id_to_sparse[cid], tokenizer)
            point = PointStruct(
                id=point_id,
                vector={"dense": id_to_dense[cid], "sparse": sparse_vec},
                payload=payload,
            )
        else:
            point = PointStruct(
                id=point_id,
                vector={"dense": id_to_dense[cid]},
                payload=payload,
            )

        points.append(point)
        point_id += 1
        count += 1

        if len(points) == BATCH_SIZE:
            client.upsert(collection_name=collection, points=points)
            print(f"    [{count}개 처리 중...]", end="\r")
            points = []

    if points:
        client.upsert(collection_name=collection, points=points)

    print(f"    upsert 완료: {count}개")
    return point_id  # 다음 컬렉션 upsert 시 ID 연속성 유지


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    print("=" * 55)
    print("Workit law_kb — Qdrant Hybrid Upsert")
    print("  law_kb_ho : 호 단위 (검색 대상)")
    print("  law_kb_jo : 조 단위 (Hierarchical RAG fetch용)")
    print("=" * 55)

    # ── 토크나이저 로드 (sparse 변환용) ──
    use_sparse_ho = SPARSE_HO_PATH.exists()
    use_sparse_jo = SPARSE_JO_PATH.exists()

    tokenizer = None
    if use_sparse_ho or use_sparse_jo:
        print(f"\n📦 토크나이저 로드: {EMBED_MODEL}")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        print(f"   vocab 크기: {tokenizer.vocab_size}")

    # ── Qdrant 연결 ───────────────────────
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # law_kb_ho — 호 단위 child 청크
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n{'─'*55}")
    print(f"[1/2] law_kb_ho 업로드")
    print(f"{'─'*55}")

    print(f"  📂 chunks_ho.json 로드: {CHUNKS_HO_PATH}")
    chunk_map_ho = load_chunks(CHUNKS_HO_PATH)
    print(f"     {len(chunk_map_ho)}개 청크")

    print(f"  📂 vectors_ho.npz 로드: {VECTORS_HO_PATH}")
    id_to_dense_ho, vector_ids_ho = load_vectors(VECTORS_HO_PATH)
    print(f"     벡터 수: {len(id_to_dense_ho)}")

    id_to_sparse_ho = None
    if use_sparse_ho:
        print(f"  📂 sparse_weights_ho.json 로드: {SPARSE_HO_PATH}")
        id_to_sparse_ho = load_sparse(SPARSE_HO_PATH, vector_ids_ho)
        print(f"     sparse 벡터 수: {len(id_to_sparse_ho)}")
    else:
        print(f"  ⚠️  sparse_weights_ho.json 없음 → dense 단독 upsert")

    ensure_collection(client, COLLECTION_HO, use_sparse=use_sparse_ho)

    print(f"\n  ⬆️  upsert 시작...")
    next_id = upsert_chunks(
        client=client,
        collection=COLLECTION_HO,
        chunk_map=chunk_map_ho,
        id_to_dense=id_to_dense_ho,
        id_to_sparse=id_to_sparse_ho,
        tokenizer=tokenizer,
        start_id=0,
    )

    ho_total = client.count(collection_name=COLLECTION_HO).count
    print(f"  ✅ law_kb_ho 완료: {ho_total}개 포인트")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # law_kb_jo — 조 단위 parent 청크
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print(f"\n{'─'*55}")
    print(f"[2/2] law_kb_jo 업로드")
    print(f"{'─'*55}")

    print(f"  📂 chunks_jo.json 로드: {CHUNKS_JO_PATH}")
    chunk_map_jo = load_chunks(CHUNKS_JO_PATH)
    print(f"     {len(chunk_map_jo)}개 청크")

    print(f"  📂 vectors_jo.npz 로드: {VECTORS_JO_PATH}")
    id_to_dense_jo, vector_ids_jo = load_vectors(VECTORS_JO_PATH)
    print(f"     벡터 수: {len(id_to_dense_jo)}")

    id_to_sparse_jo = None
    if use_sparse_jo:
        print(f"  📂 sparse_weights_jo.json 로드: {SPARSE_JO_PATH}")
        id_to_sparse_jo = load_sparse(SPARSE_JO_PATH, vector_ids_jo)
        print(f"     sparse 벡터 수: {len(id_to_sparse_jo)}")
    else:
        print(f"  ⚠️  sparse_weights_jo.json 없음 → dense 단독 upsert")

    ensure_collection(client, COLLECTION_JO, use_sparse=use_sparse_jo)

    print(f"\n  ⬆️  upsert 시작...")
    upsert_chunks(
        client=client,
        collection=COLLECTION_JO,
        chunk_map=chunk_map_jo,
        id_to_dense=id_to_dense_jo,
        id_to_sparse=id_to_sparse_jo,
        tokenizer=tokenizer,
        start_id=next_id,  # ho와 ID 연속성 유지
    )

    jo_total = client.count(collection_name=COLLECTION_JO).count
    print(f"  ✅ law_kb_jo 완료: {jo_total}개 포인트")

    # ── 최종 요약 ─────────────────────────
    print(f"\n{'='*55}")
    print(f"✅ 전체 완료")
    print(f"   law_kb_ho (검색 대상)     : {ho_total}개")
    print(f"   law_kb_jo (fetch용)       : {jo_total}개")
    print(f"{'='*55}")
    print(f"\n다음 단계:")
    print(f"  py rag/yoonha_evaluator.py  ← 컬렉션 성능 평가")


if __name__ == "__main__":
    main()