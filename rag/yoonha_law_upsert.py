"""
Workit - law_kb Qdrant upsert 스크립트 (Hybrid: Dense + Sparse)
파일명: yoonha_law_upsert.py
위치:   Workit/rag/yoonha_law_upsert.py

데이터:
  data/export/chunks.json         → payload (parent + child 모두 포함)
  data/export/vectors.npz         → dense 벡터 (child 청크만, N × 1024)
  data/export/sparse_weights.json → BGE-M3 sparse lexical weights (child 청크만)

Hierarchical RAG 구조:
  - child 청크: dense + sparse 벡터로 upsert → 실제 검색 대상
  - parent 청크: 벡터 없이 payload만 upsert → child hit 후 fetch용

실행:
  python rag/yoonha_law_upsert.py
"""

from __future__ import annotations

import json
import re
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

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION  = "law_kb"
VECTOR_DIM  = 1024
BATCH_SIZE  = 64
EMBED_MODEL = "BAAI/bge-m3"

_THIS_DIR   = Path(__file__).resolve().parent
_DATA_DIR   = _THIS_DIR.parent / "data" / "export"

CHUNKS_PATH  = _DATA_DIR / "chunks.json"
VECTORS_PATH = _DATA_DIR / "vectors.npz"
SPARSE_PATH  = _DATA_DIR / "sparse_weights.json"


# ──────────────────────────────────────────
# sparse weight 변환
# ──────────────────────────────────────────
def to_sparse_vector(lexical_weights: dict, tokenizer) -> SparseVector:
    """
    BGE-M3 lexical_weights {token_str: weight} →
    Qdrant SparseVector {indices: [...], values: [...]}

    - token_str → token_id: 모델 vocab 기반 변환 (retrieval과 동일 방식)
    - 동일 token_id 중복 시 weight 합산 (Qdrant indices unique 조건 충족)
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


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────
def main() -> None:
    print("=" * 55)
    print("Workit law_kb — Qdrant Hybrid Upsert")
    print("=" * 55)

    # ── 청크 로드 ─────────────────────────
    print(f"\n📂 chunks.json 로드: {CHUNKS_PATH}")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)

    # chunk_id 기준 중복 제거
    chunk_map: dict[str, dict] = {}
    for c in chunks:
        chunk_map[c["chunk_id"]] = c
    print(f"   원본 {len(chunks)}개 → chunk_id 중복 제거 후 {len(chunk_map)}개")

    # parent / child 분리
    parent_chunks = {cid: c for cid, c in chunk_map.items() if c.get("is_parent")}
    child_chunks  = {cid: c for cid, c in chunk_map.items() if not c.get("is_parent")}
    print(f"   parent: {len(parent_chunks)}개 | child: {len(child_chunks)}개")

    # ── 벡터 로드 (child 청크 기준) ───────
    print(f"\n📂 vectors.npz 로드: {VECTORS_PATH}")
    npz            = np.load(VECTORS_PATH)
    vectors        = npz["vectors"].astype(np.float32)
    vector_ids     = npz["chunk_ids"].tolist()
    print(f"   벡터 shape: {vectors.shape}")

    id_to_dense: dict[str, list[float]] = {
        cid: vec.tolist() for cid, vec in zip(vector_ids, vectors)
    }

    # ── sparse 로드 (child 청크 기준) ─────
    use_sparse    = SPARSE_PATH.exists()
    id_to_sparse: dict[str, dict] = {}
    tokenizer     = None

    if use_sparse:
        print(f"\n📂 sparse_weights.json 로드: {SPARSE_PATH}")
        with open(SPARSE_PATH, encoding="utf-8") as f:
            sparse_list: list[dict] = json.load(f)
        id_to_sparse = dict(zip(vector_ids, sparse_list))
        print(f"   sparse 벡터: {len(id_to_sparse)}개")

        print(f"\n📦 토크나이저 로드: {EMBED_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        print(f"   토크나이저 vocab 크기: {tokenizer.vocab_size}")
    else:
        print(f"\n⚠️  sparse_weights.json 없음 → dense 단독 upsert")

    # ── Qdrant 컬렉션 준비 ────────────────
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print(f"\n⚠️  컬렉션 '{COLLECTION}' 이미 존재 → 재생성합니다.")
        client.delete_collection(COLLECTION)

    if use_sparse:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
        print(f"✅ 컬렉션 '{COLLECTION}' 생성 완료 (Dense + Sparse)")
    else:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"✅ 컬렉션 '{COLLECTION}' 생성 완료 (Dense only)")

    # ── child 청크 upsert (벡터 + payload) ─
    print(f"\n⬆️  child 청크 upsert 시작 (batch_size={BATCH_SIZE})...")
    points: list[PointStruct] = []
    point_id = 0
    child_upsert_count = 0

    for cid, chunk in child_chunks.items():
        if cid not in id_to_dense:
            continue

        payload = {
            "chunk_id":       cid,
            "law_name":       chunk.get("law_name",       ""),
            "article_id":     chunk.get("article_id",     ""),
            "article_number": chunk.get("article_number", ""),
            "chunk_text":     chunk.get("text",           ""),
            "is_parent":      False,
            "parent_id":      chunk.get("parent_id"),
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
                vector=id_to_dense[cid],
                payload=payload,
            )

        points.append(point)
        point_id += 1
        child_upsert_count += 1

        if len(points) == BATCH_SIZE:
            client.upsert(collection_name=COLLECTION, points=points)
            print(f"   [{child_upsert_count}/{len(child_chunks)}] upsert...", end="\r")
            points = []

    if points:
        client.upsert(collection_name=COLLECTION, points=points)

    print(f"\n   child 청크 upsert 완료: {child_upsert_count}개")

    # ── parent 청크 upsert (payload만, 벡터 없음) ─
    # 검색 대상은 아니지만 Hierarchical fetch 시
    # parent_id로 조회할 수 있도록 payload만 저장
    print(f"\n⬆️  parent 청크 upsert 시작 (payload only)...")
    points = []
    parent_upsert_count = 0

    for cid, chunk in parent_chunks.items():
        payload = {
            "chunk_id":       cid,
            "law_name":       chunk.get("law_name",       ""),
            "article_id":     chunk.get("article_id",     ""),
            "article_number": chunk.get("article_number", ""),
            "chunk_text":     chunk.get("text",           ""),
            "is_parent":      True,
            "parent_id":      None,
            "is_ref_article": bool(chunk.get("is_ref_article", False)),
            "is_upper_law":   bool(chunk.get("is_upper_law",   False)),
            "hierarchy":      chunk.get("hierarchy",      {}),
        }

        # parent는 벡터 없이 0벡터로 upsert — payload 조회 전용
        zero_vec = [0.0] * VECTOR_DIM
        if use_sparse:
            point = PointStruct(
                id=point_id,
                vector={"dense": zero_vec, "sparse": SparseVector(indices=[], values=[])},
                payload=payload,
            )
        else:
            point = PointStruct(
                id=point_id,
                vector=zero_vec,
                payload=payload,
            )

        points.append(point)
        point_id += 1
        parent_upsert_count += 1

        if len(points) == BATCH_SIZE:
            client.upsert(collection_name=COLLECTION, points=points)
            points = []

    if points:
        client.upsert(collection_name=COLLECTION, points=points)

    print(f"   parent 청크 upsert 완료: {parent_upsert_count}개")

    # ── 확인 ─────────────────────────────
    total = client.count(collection_name=COLLECTION)
    print(f"\n✅ 완료: 총 {total.count}개 포인트 저장됨")
    print(f"   child(검색 대상): {child_upsert_count}개")
    print(f"   parent(fetch 전용): {parent_upsert_count}개")


if __name__ == "__main__":
    main()