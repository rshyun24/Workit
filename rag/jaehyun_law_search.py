"""
Qdrant 없이 로컬 벡터로 top-k 법령 검색
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel

_THIS_DIR   = Path(__file__).resolve().parent
_DATA_DIR   = _THIS_DIR.parent / "data" / "export"

CHUNKS_PATH  = _DATA_DIR / "chunks.json"
VECTORS_PATH = _DATA_DIR / "vectors.npz"

TOP_K   = 10
FETCH_K = 20


def load_db():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    seen = {}
    for c in chunks:
        seen[c["chunk_id"]] = c
    chunks = list(seen.values())

    npz = np.load(VECTORS_PATH, allow_pickle=True)
    vectors   = npz["vectors"].astype(np.float32)
    chunk_ids = npz["chunk_ids"].tolist()

    # chunk_id 순서 맞추기
    chunk_map = {c["chunk_id"]: c for c in chunks}

    return vectors, chunk_ids, chunk_map


def search(query: str, model: BGEM3FlagModel, vectors: np.ndarray,
           chunk_ids: list, chunk_map: dict,
           top_k: int = TOP_K, fetch_k: int = FETCH_K) -> list[dict]:
    # fetch_k: Qdrant 하이브리드와 인터페이스 통일용 (로컬에선 top_k로 동작)
    k = max(top_k, fetch_k)

    out = model.encode([query], return_dense=True, return_sparse=False)
    qvec = out["dense_vecs"][0]  # (1024,)

    # 코사인 유사도 (벡터가 이미 정규화돼 있으면 내적 = 코사인)
    scores = vectors @ qvec  # (N,)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        cid   = chunk_ids[idx]
        chunk = chunk_map.get(cid, {})
        results.append({
            "chunk_id"  : cid,
            "law_name"  : chunk.get("law_name", ""),
            "article_id": chunk.get("article_id", ""),
            "text"      : chunk.get("text", ""),
            "score"     : float(scores[idx]),
        })

    return results


if __name__ == "__main__":
    print("모델 로드 중...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    print("DB 로드 중...")
    vectors, chunk_ids, chunk_map = load_db()
    print(f"벡터 수: {len(chunk_ids)}")

    while True:
        query = input("\n검색어 입력 (종료: q): ").strip()
        if query == "q":
            break

        results = search(query, model, vectors, chunk_ids, chunk_map)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r['chunk_id']} | {r['law_name']} | score={r['score']:.4f}")
            print(f"     {r['text'][:100]}...")
