"""
Workit - fixedid 임베딩 결과 Qdrant 업로드 (upsert)
파일명: rag/yoonha_law_upsert.py

이 파일은 임베딩(Colab에서 만든 vectors_*.npz, sparse_weights_*.json)과
파싱 결과(chunks_*_fixedid.json)를 합쳐서 Qdrant에 upsert하는 "업로드 전용"
스크립트다. 검색은 별도 파일인 rag/yoonha_law_rag.py에서 담당하며, 그 스크립트는
여기서 만든 컬렉션이 이미 존재한다고 가정하고 조회만 한다. 즉 순서는 항상:
  1) 이 스크립트(rag/yoonha_law_upsert.py) 실행 → 컬렉션 생성/upsert
  2) rag/yoonha_law_rag.py에서 검색

입력 (project_root/data/merged/ 에 위치, 스크립트 실행 위치와 무관하게 자동 탐색):
  - chunks_jo_fixedid.json / chunks_ho_fixedid.json   (merge_chunks.py 출력, payload 원본)
  - vectors_jo_fixedid.npz / vectors_ho_fixedid.npz   (dense 벡터 + chunk_ids)
  - sparse_weights_jo_fixedid.json / sparse_weights_ho_fixedid.json (sparse 가중치, chunk_ids와 순서 동일)

출력:
  - Qdrant collection "law_kb_jo_fixedid" (dense + sparse named vectors)
  - Qdrant collection "law_kb_ho_fixedid" (dense + sparse named vectors)

주의:
  - chunk_id를 key로 세 입력을 정렬/매칭한다 (npz/sparse의 순서가 chunks json과
    다를 수 있으므로, 순서를 믿지 말고 항상 chunk_id로 매칭).
  - payload에는 파서가 만든 필드를 그대로 싣는다: chunk_id, law_name, article_id,
    article_number, text, hierarchy, parent_chunk_id, cross_refs, is_ref_article,
    is_upper_law. (category / is_risk_ref 등 계약서 리스크 태깅 필드는 별도 태깅
    단계가 아직 없어서 이 시점엔 채워지지 않음 — yoonha_law_rag.py 쪽에서
    laws_ref.json으로 보강하는 구조를 그대로 유지)

사용법:
    pip install qdrant-client
    python rag/yoonha_law_upsert.py
"""

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

DATASETS = {
    "jo_fixedid": {
        "chunks_file":  DATA_DIR / "chunks_jo_fixedid.json",
        "vectors_file": DATA_DIR / "vectors_jo_fixedid.npz",
        "sparse_file":  DATA_DIR / "sparse_weights_jo_fixedid.json",
        "collection":   "law_kb_jo_fixedid",
    },
    "ho_fixedid": {
        "chunks_file":  DATA_DIR / "chunks_ho_fixedid.json",
        "vectors_file": DATA_DIR / "vectors_ho_fixedid.npz",
        "sparse_file":  DATA_DIR / "sparse_weights_ho_fixedid.json",
        "collection":   "law_kb_ho_fixedid",
    },
}

# 임베딩에 쓴 payload 필드 (text는 인덱싱만 하고 payload에도 남겨서 리랭크/표시에 사용)
PAYLOAD_FIELDS = [
    "chunk_id", "law_name", "article_id", "article_number", "title",
    "text", "hierarchy", "parent_chunk_id", "cross_refs",
    "is_ref_article", "is_upper_law",
]


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


def ensure_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    if client.collection_exists(name):
        print(f"  [SKIP] 컬렉션 이미 존재: {name}")
        return
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
        sparse_vec = SparseVector(
            indices=list(sparse.keys()),
            values=list(sparse.values()),
        )

        points.append(PointStruct(
            id=idx,  # Qdrant point id는 정수/UUID만 허용 → chunk_id는 payload에 별도 저장
            vector={
                "dense": dense_vectors[idx].tolist(),
                "sparse": sparse_vec,
            },
            payload=payload,
        ))

    if missing_payload:
        print(f"  ⚠️  chunks json에 없는 chunk_id {missing_payload}개는 스킵됨 (dense/sparse는 있는데 payload가 없는 경우)")

    return points


def upload_dataset(client: QdrantClient, name: str, cfg: dict, batch_size: int = 256) -> None:
    print(f"\n=== {name} ===")

    npz = np.load(cfg["vectors_file"], allow_pickle=True)
    dense_vectors = npz["vectors"]
    chunk_ids = [str(c) for c in npz["chunk_ids"]]

    chunks_by_id = load_chunks_by_id(cfg["chunks_file"])
    sparse_by_id = load_sparse_by_id(cfg["sparse_file"], chunk_ids)

    ensure_collection(client, cfg["collection"], dense_vectors.shape[1])

    points = build_points(chunk_ids, dense_vectors, sparse_by_id, chunks_by_id)
    print(f"  업로드할 point 수: {len(points)}")

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=cfg["collection"], points=batch)
        print(f"    upsert {i + len(batch)}/{len(points)}", end="\r")

    print(f"\n  ✅ 완료: {cfg['collection']} ({len(points)} points)")


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    for name, cfg in DATASETS.items():
        for key in ("chunks_file", "vectors_file", "sparse_file"):
            if not cfg[key].exists():
                print(f"[ERROR] {name}: {cfg[key]} 파일이 없습니다. 건너뜁니다.")
                break
        else:
            upload_dataset(client, name, cfg)

    print("\nDone!")


if __name__ == "__main__":
    main()