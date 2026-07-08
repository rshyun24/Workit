"""
Workit - fixedid 임베딩 결과 Qdrant 업로드 (upsert)
파일명: rag/yoonha_law_upsert.py

임베딩(vectors_*.npz, sparse_weights_*.json) + 파싱 결과(chunks_*_fixedid.json)를
합쳐 Qdrant에 upsert. 검색은 yoonha_law_rag.py 담당 (이 스크립트가 먼저 실행돼
컬렉션이 존재해야 함).

입력: project_root/data/merged/ 안의 chunks_*_fixedid.json, vectors_*_fixedid.npz,
      sparse_weights_*_fixedid.json (jo/ho 각각)
출력: Qdrant collection "law_kb_jo", "law_kb_ho"

RECREATE_COLLECTIONS=True(기본값): 매번 컬렉션 삭제 후 재생성 → 소스 JSON에서
빠진 chunk가 Qdrant에 고아로 안 남고 항상 동기화됨. False면 있으면 skip(빠른
반복 실험용).

Point id는 chunk_id 해시값(결정론적) — 배열 순서 아님, 매칭은 항상 chunk_id로.

사용법:
    pip install qdrant-client
    python rag/yoonha_law_upsert.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "merged"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# True: 매번 컬렉션 삭제 후 재생성 (Qdrant를 소스 JSON과 100% 동기화, 고아 데이터 방지)
# False: 있으면 건너뛰고 upsert만 (빠른 반복 실험용)
RECREATE_COLLECTIONS = True

DATASETS = {
    "jo_fixedid": {
        "chunks_file":  DATA_DIR / "chunks_jo_fixedid.json",
        "vectors_file": DATA_DIR / "vectors_jo_fixedid.npz",
        "sparse_file":  DATA_DIR / "sparse_weights_jo_fixedid.json",
        "collection":   "law_kb_jo",
    },
    "ho_fixedid": {
        "chunks_file":  DATA_DIR / "chunks_ho_fixedid.json",
        "vectors_file": DATA_DIR / "vectors_ho_fixedid.npz",
        "sparse_file":  DATA_DIR / "sparse_weights_ho_fixedid.json",
        "collection":   "law_kb_ho",
    },
}

# 임베딩에 쓴 payload 필드 (text는 인덱싱만 하고 payload에도 남겨서 리랭크/표시에 사용)
PAYLOAD_FIELDS = [
    "chunk_id", "law_name", "article_id", "article_number", "title",
    "text", "hierarchy", "parent_chunk_id", "cross_refs",
    "is_ref_article", "is_upper_law",
]


def chunk_id_to_point_id(chunk_id: str) -> int:
    """chunk_id -> 결정론적 정수 id. Qdrant point id는 int/UUID만 허용."""
    return abs(hash(chunk_id)) % (2 ** 63)


def load_chunks_by_id(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    return {c["chunk_id"]: c for c in chunks}


def load_sparse_by_id(path: Path, chunk_ids_order: list[str]) -> dict[str, dict[int, float]]:
    """sparse_weights_*.json은 chunk_ids_order(npz의 chunk_ids)와 같은 순서라고 가정하고 매칭."""
    with open(path, encoding="utf-8") as f:
        sparse_list = json.load(f)

    if len(sparse_list) != len(chunk_ids_order):
        raise ValueError(
            f"sparse_weights 길이({len(sparse_list)})와 chunk_ids 길이({len(chunk_ids_order)})가 다릅니다. "
            "임베딩 노트북에서 저장한 순서가 어긋났을 가능성이 있습니다."
        )

    return {
        cid: {int(tok): float(w) for tok, w in sw.items()}
        for cid, sw in zip(chunk_ids_order, sparse_list)
    }


def ensure_collection(client: QdrantClient, name: str, dense_dim: int, recreate: bool) -> None:
    exists = client.collection_exists(name)

    if exists and not recreate:
        print(f"  [SKIP] 컬렉션 이미 존재: {name}")
        return

    if exists:
        client.delete_collection(name)
        print(f"  [DELETE] 기존 컬렉션 삭제: {name}")

    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    print(f"  [CREATE] 컬렉션 생성: {name} (dense_dim={dense_dim})")


def build_points(
    chunk_ids: list[str],
    dense_vectors: np.ndarray,
    sparse_by_id: dict[str, dict[int, float]],
    chunks_by_id: dict[str, dict],
) -> list[PointStruct]:
    points = []
    missing_payload = 0

    for idx, cid in enumerate(chunk_ids):
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            missing_payload += 1
            continue

        payload = {k: chunk.get(k) for k in PAYLOAD_FIELDS if k in chunk}
        sparse = sparse_by_id.get(cid, {})

        points.append(PointStruct(
            id=chunk_id_to_point_id(cid),
            vector={
                "dense": dense_vectors[idx].tolist(),
                "sparse": SparseVector(indices=list(sparse.keys()), values=list(sparse.values())),
            },
            payload=payload,
        ))

    if missing_payload:
        print(f"  ⚠️  chunks json에 없는 chunk_id {missing_payload}개는 스킵됨 (dense/sparse는 있는데 payload가 없는 경우)")

    return points


def upload_dataset(client: QdrantClient, name: str, cfg: dict, batch_size: int = 256) -> int:
    print(f"\n=== {name} ===")

    npz = np.load(cfg["vectors_file"], allow_pickle=True)
    dense_vectors = npz["vectors"]
    chunk_ids = [str(c) for c in npz["chunk_ids"]]

    chunks_by_id = load_chunks_by_id(cfg["chunks_file"])
    sparse_by_id = load_sparse_by_id(cfg["sparse_file"], chunk_ids)

    ensure_collection(client, cfg["collection"], dense_vectors.shape[1], recreate=RECREATE_COLLECTIONS)

    points = build_points(chunk_ids, dense_vectors, sparse_by_id, chunks_by_id)
    print(f"  업로드할 point 수: {len(points)}")

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=cfg["collection"], points=batch)
        print(f"    upsert {i + len(batch)}/{len(points)}", end="\r")

    count = client.count(cfg["collection"]).count
    print(f"\n  ✅ 완료: {cfg['collection']} (업로드 {len(points)}개 / 컬렉션 총 {count}개)")
    return count


def missing_input_files(cfg: dict) -> list[Path]:
    return [cfg[key] for key in ("chunks_file", "vectors_file", "sparse_file") if not cfg[key].exists()]


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    for name, cfg in DATASETS.items():
        missing = missing_input_files(cfg)
        if missing:
            print(f"[ERROR] {name}: 다음 파일이 없어 건너뜁니다 -> {[str(p) for p in missing]}")
            continue
        upload_dataset(client, name, cfg)

    print("\nDone!")


if __name__ == "__main__":
    main()