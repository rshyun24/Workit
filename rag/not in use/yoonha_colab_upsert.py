"""
Workit - Colab용 로컬 Qdrant 셋업 + upsert 스크립트
파일명: rag/yoonha_colab_upsert.py

목적:
  Colab은 로컬 머신의 Qdrant(localhost:6333)에 접근할 수 없다. 대신 이미
  임베딩이 끝난 4개 파일(vectors_*.npz, sparse_weights_*.json)과 payload
  메타데이터(chunks_*_fixedid.json)를 Colab에 직접 업로드해서, Colab
  안에서 "로컬 파일 기반" Qdrant를 새로 띄우고 그 안에 upsert한다.
  서버 프로세스가 따로 필요 없다 (QdrantClient(path=...) 모드).

  이렇게 하면:
    - ngrok/터널링 없이 Colab 세션 안에서 완결됨
    - 로컬 머신을 계속 켜둘 필요 없음
    - GPU는 reranker/embedding 계산에만 쓰이고, Qdrant 자체는 CPU로 충분

전제:
  - vectors_jo_fixedid.npz / vectors_ho_fixedid.npz : {"vectors": (N,1024) float16,
    "chunk_ids": (N,) 문자열 배열}
  - sparse_weights_jo_fixedid.json / sparse_weights_ho_fixedid.json : 길이 N인
    리스트, vectors의 chunk_ids와 "같은 순서"로 정렬돼 있다고 이미 확인함
    (jo/ho 둘 다 순서 일치, 중복 0 — 실제 업로드 파일로 검증 완료).
  - chunks_jo_fixedid.json / chunks_ho_fixedid.json : chunk_id별 payload
    메타데이터(text, parent_chunk_id, cross_refs, law_name, article_number,
    title, hierarchy, is_ref_article, is_upper_law 등). vectors의 chunk_ids와
    "순서까지" 동일하다고 확인했지만, 이 스크립트는 순서에 의존하지 않고
    chunk_id로 다시 매핑해서 병합한다 (더 안전한 방식).

  주의 — 지금 chunks_*_fixedid.json에는 category, is_risk_ref 필드가 없다.
  yoonha_law_rag.py의 _build_law_refs가 이 필드들을 payload.get(key, 기본값)
  으로 읽기 때문에 없어도 에러는 안 나고 그냥 빈 문자열/False로 채워진다.
  나중에 laws_ref.json(load_laws_ref)에서 category를 따로 채워주는 구조라
  RAG 파이프라인 동작 자체엔 문제 없음.

사용법 (Colab 셀에서):
    !pip install qdrant-client --quiet
    !python yoonha_colab_upsert.py
  또는 노트북 안에서 build_local_qdrant() 함수를 직접 호출해도 된다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 경로 — Colab에서 업로드한 파일 위치에 맞게 수정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA_DIR = Path("/content")  # 업로드한 4+2개 파일이 있는 디렉터리

VECTORS_JO      = DATA_DIR / "vectors_jo_fixedid.npz"
VECTORS_HO      = DATA_DIR / "vectors_ho_fixedid.npz"
SPARSE_JO       = DATA_DIR / "sparse_weights_jo_fixedid.json"
SPARSE_HO       = DATA_DIR / "sparse_weights_ho_fixedid.json"
CHUNKS_JO       = DATA_DIR / "chunks_jo_fixedid.json"
CHUNKS_HO       = DATA_DIR / "chunks_ho_fixedid.json"

# Colab 세션 안에 로컬로 만들 Qdrant 저장 경로 (서버 프로세스 없이 파일 기반)
QDRANT_LOCAL_PATH = "/content/qdrant_local"

COLLECTION_JO = "law_kb_jo_fixedid"
COLLECTION_HO = "law_kb_ho_fixedid"

DENSE_DIM = 1024  # BGE-M3 dense 차원 (npz vectors.shape[1]로 실제 검증도 함께 함)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 로드 / 병합
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_merged(vectors_path: Path, sparse_path: Path, chunks_path: Path) -> list[dict]:
    """
    vectors(.npz) + sparse_weights(.json) + chunks(.json, payload)를
    chunk_id 기준으로 병합해서 [{chunk_id, dense_vec, sparse_vec, payload}, ...] 반환.
    """
    npz = np.load(vectors_path, allow_pickle=True)
    dense_vectors = npz["vectors"]          # (N, 1024) float16
    chunk_ids     = npz["chunk_ids"].tolist()

    assert dense_vectors.shape[1] == DENSE_DIM, (
        f"예상한 dense 차원({DENSE_DIM})과 실제({dense_vectors.shape[1]})가 다릅니다 — "
        f"임베딩 모델이 바뀐 건 아닌지 확인하세요."
    )

    with open(sparse_path, encoding="utf-8") as f:
        sparse_weights = json.load(f)  # chunk_ids와 같은 순서의 리스트

    assert len(sparse_weights) == len(chunk_ids), (
        f"sparse_weights 개수({len(sparse_weights)})와 chunk_ids 개수({len(chunk_ids)})가 다릅니다."
    )

    with open(chunks_path, encoding="utf-8") as f:
        chunks_list = json.load(f)

    payload_by_id = {c["chunk_id"]: c for c in chunks_list}

    missing = [cid for cid in chunk_ids if cid not in payload_by_id]
    if missing:
        print(f"  ⚠️  payload를 못 찾은 chunk_id {len(missing)}개 (예: {missing[:5]}) "
              f"— 해당 chunk는 text 없이 upsert됩니다.")

    merged = []
    for i, cid in enumerate(chunk_ids):
        payload = dict(payload_by_id.get(cid, {}))
        payload["chunk_id"] = cid  # payload 안에도 chunk_id를 넣어야 검색 결과에서 식별 가능
        merged.append({
            "chunk_id":   cid,
            "dense_vec":  dense_vectors[i].astype(np.float32).tolist(),
            "sparse_vec": sparse_weights[i],  # {"token_id": weight, ...}
            "payload":    payload,
        })

    return merged


def _to_qdrant_points(merged: list[dict]) -> list[PointStruct]:
    points = []
    for item in merged:
        sparse = item["sparse_vec"]
        indices = [int(k) for k in sparse.keys()]
        values  = [float(v) for v in sparse.values()]

        points.append(PointStruct(
            id=abs(hash(item["chunk_id"])) % (2**63),  # Qdrant point id는 int/UUID만 허용, chunk_id는 payload에 문자열로 그대로 보존
            vector={
                "dense":  item["dense_vec"],
                "sparse": SparseVector(indices=indices, values=values),
            },
            payload=item["payload"],
        ))
    return points


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 컬렉션 생성 + upsert
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _create_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(name):
        print(f"  ℹ️  {name} 이미 존재 — 삭제 후 재생성")
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )


def _upsert_batched(client: QdrantClient, name: str, points: list[PointStruct], batch_size: int = 256) -> None:
    total = len(points)
    for i in range(0, total, batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=name, points=batch)
        print(f"  [{name}] {min(i + batch_size, total)}/{total} upsert 완료", end="\r")
    print()


def build_local_qdrant(recreate: bool = True) -> QdrantClient:
    """
    Colab 세션 안에서 로컬(파일 기반) Qdrant를 만들고, 업로드된 4+2개 파일로
    law_kb_jo_fixedid / law_kb_ho_fixedid 컬렉션을 채운다.
    반환된 client를 그대로 yoonha_rag_eval*.py의 QdrantClient 자리에 넣으면 된다.
    """
    client = QdrantClient(path=QDRANT_LOCAL_PATH)

    print("📦 jo 데이터 로드/병합 중...")
    merged_jo = _load_merged(VECTORS_JO, SPARSE_JO, CHUNKS_JO)
    print(f"  -> {len(merged_jo)}개 chunk 병합 완료")

    print("📦 ho 데이터 로드/병합 중...")
    merged_ho = _load_merged(VECTORS_HO, SPARSE_HO, CHUNKS_HO)
    print(f"  -> {len(merged_ho)}개 chunk 병합 완료")

    print(f"🗂️  {COLLECTION_JO} 컬렉션 생성...")
    _create_collection(client, COLLECTION_JO)
    _upsert_batched(client, COLLECTION_JO, _to_qdrant_points(merged_jo))

    print(f"🗂️  {COLLECTION_HO} 컬렉션 생성...")
    _create_collection(client, COLLECTION_HO)
    _upsert_batched(client, COLLECTION_HO, _to_qdrant_points(merged_ho))

    jo_count = client.count(COLLECTION_JO).count
    ho_count = client.count(COLLECTION_HO).count
    print(f"\n✅ 완료 — {COLLECTION_JO}: {jo_count}개, {COLLECTION_HO}: {ho_count}개")

    return client


if __name__ == "__main__":
    build_local_qdrant()