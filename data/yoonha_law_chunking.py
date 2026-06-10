"""
Workit - 법령 KB 구축
법령 JSON → 청킹 → taxonomy 매핑 → Qdrant 저장

실행 전 설치:
    pip install qdrant-client sentence-transformers transformers numpy
"""

import json
import uuid
from pathlib import Path
from collections import defaultdict

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────
DATA_DIR      = Path("data/structured")
QDRANT_PATH   = "./qdrant_storage"
COLLECTION    = "law_kb"
EMBED_MODEL   = "BAAI/bge-m3"
MAX_TOKENS    = 1024
MIN_TOKENS    = 30

# 법령 파일명 → (law_name, law_type) 매핑
LAW_META = {
    "소프트웨어_진흥법.json":                              {"law_name": "소프트웨어 진흥법",              "law_type": "법률"},
    "지방계약법.json":                                    {"law_name": "지방계약법",                    "law_type": "법률"},
    "지방계약법_시행규칙.json":                            {"law_name": "지방계약법 시행규칙",             "law_type": "시행규칙"},
    "지방계약법_시행령.json":                              {"law_name": "지방계약법 시행령",               "law_type": "시행령"},
    "지방자치단체 용역계약 일반조건 (행안부 예규).json":       {"law_name": "지방자치단체 용역계약 일반조건",   "law_type": "예규"},
}

# taxonomy: 9개 리스크 유형 + 법령 조문 매핑
TAXONOMY = [
    {
        "risk_id":   "RISK_001",
        "risk_name": "지연배상금 상한 미설정",
        "law_refs":  [
            {"filename": "지방계약법_시행령.json", "article_id": "제90조"},
        ],
    },
    {
        "risk_id":   "RISK_002",
        "risk_name": "지연배상금률 과다 설정",
        "law_refs":  [
            {"filename": "지방계약법_시행규칙.json", "article_id": "제75조"},
        ],
    },
    {
        "risk_id":   "RISK_003",
        "risk_name": "대금 지급 기한 미설정",
        "law_refs":  [
            {"filename": "지방계약법.json",       "article_id": "제18조"},
            {"filename": "지방계약법_시행령.json", "article_id": "제67조"},
        ],
    },
    {
        "risk_id":   "RISK_004",
        "risk_name": "합의 없는 일방적 과업 추가",
        "law_refs":  [
            {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규).json", "article_id": "제6절_제1항_나"},
            {"filename": "소프트웨어_진흥법.json",                          "article_id": "제50조"},
        ],
    },
    {
        "risk_id":   "RISK_005",
        "risk_name": "추가 과업 비용 미지급",
        "law_refs":  [
            {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규).json", "article_id": "제6절_제1항_라"},
            {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규).json", "article_id": "제6절_제1항_마"},
            {"filename": "지방계약법_시행령.json",                          "article_id": "제74조"},
        ],
    },
    {
        "risk_id":   "RISK_006",
        "risk_name": "갑의 일방적 해지권",
        "law_refs":  [
            {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규).json", "article_id": "제7절_제4항_다"},
        ],
    },
    {
        "risk_id":   "RISK_007",
        "risk_name": "을의 해제권 배제/제한",
        "law_refs":  [
            {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규).json", "article_id": "제7절_제5항_가"},
        ],
    },
    {
        "risk_id":   "RISK_008",
        "risk_name": "계약금액 조정 없는 과업변경",
        "law_refs":  [
            {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규).json", "article_id": "제6절_제1항_라"},
            {"filename": "지방계약법.json",       "article_id": "제22조"},
            {"filename": "지방계약법_시행령.json", "article_id": "제74조"},
        ],
    },
    {
        "risk_id":   "RISK_009",
        "risk_name": "손해배상 범위 일방적 제한",
        "law_refs":  [
            {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규).json", "article_id": "제8절_제7항_가"},
            {"filename": "소프트웨어_진흥법.json",                          "article_id": "제38조"},
        ],
    },
]


# ──────────────────────────────────────────
# 1. taxonomy 역인덱스 빌드
#    (filename, article_id) → [risk_id, risk_name, ...]
# ──────────────────────────────────────────
def build_risk_index(taxonomy: list[dict]) -> dict:
    """
    반환: { (filename, article_id): [{"risk_id": ..., "risk_name": ...}, ...] }
    하나의 조문이 여러 리스크에 매핑될 수 있으므로 list로 관리
    """
    index = defaultdict(list)
    for risk in taxonomy:
        for ref in risk["law_refs"]:
            key = (ref["filename"], ref["article_id"])
            index[key].append({
                "risk_id":   risk["risk_id"],
                "risk_name": risk["risk_name"],
            })
    return dict(index)


# ──────────────────────────────────────────
# 2. JSON 로드
# ──────────────────────────────────────────
def load_law_json(filepath: Path) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("articles", [])


# ──────────────────────────────────────────
# 3. 토큰 분포 측정
# ──────────────────────────────────────────
def measure_token_distribution(articles: list[dict], tokenizer) -> dict:
    counts = [len(tokenizer.encode(a["text"])) for a in articles if a.get("text")]
    return {
        "count":  len(counts),
        "mean":   round(float(np.mean(counts)), 1),
        "median": round(float(np.median(counts)), 1),
        "p75":    round(float(np.percentile(counts, 75)), 1),
        "p95":    round(float(np.percentile(counts, 95)), 1),
        "max":    int(np.max(counts)),
    }


# ──────────────────────────────────────────
# 4. RSC 청킹
# ──────────────────────────────────────────
SEPARATORS = ["\n\n", "\n", ". ", " "]

def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))

def recursive_split(text: str, tokenizer, sep_index: int = 0) -> list[str]:
    if count_tokens(text, tokenizer) <= MAX_TOKENS:
        return [text.strip()] if text.strip() else []
    if sep_index >= len(SEPARATORS):
        return [text.strip()]
    sep = SEPARATORS[sep_index]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) <= 1:
        return recursive_split(text, tokenizer, sep_index + 1)
    result = []
    for part in parts:
        if count_tokens(part, tokenizer) > MAX_TOKENS:
            result.extend(recursive_split(part, tokenizer, sep_index + 1))
        else:
            result.append(part.strip())
    return result


# ──────────────────────────────────────────
# 5. 청크 생성 + 메타데이터 태깅
# ──────────────────────────────────────────
def chunk_articles(
    articles:   list[dict],
    tokenizer,
    law_meta:   dict,
    filename:   str,
    risk_index: dict,
) -> list[dict]:
    chunks = []

    for article in articles:
        text = article.get("text", "").strip()
        if not text:
            continue

        article_id     = article.get("article_id", "")
        article_number = article.get("article_number", "")
        title          = article.get("title", "")

        # taxonomy 매핑 조회
        key        = (filename, article_id)
        risk_hits  = risk_index.get(key, [])           # 없으면 빈 리스트
        risk_ids   = [r["risk_id"]   for r in risk_hits]
        risk_names = [r["risk_name"] for r in risk_hits]

        # 청킹
        sub_chunks = (
            [text] if count_tokens(text, tokenizer) <= MAX_TOKENS
            else recursive_split(text, tokenizer)
        )

        for idx, chunk_text in enumerate(sub_chunks):
            if not chunk_text:
                continue

            chunk = {
                # ── 식별 ──────────────────────────────
                "chunk_id":       str(uuid.uuid4()),
                "source_type":    "law",
                # ── 법령 정보 ─────────────────────────
                "law_name":       law_meta["law_name"],
                "law_type":       law_meta["law_type"],
                "article_id":     article_id,
                "article_number": article_number,
                "article_title":  title,
                "source_full":    f"{law_meta['law_name']} {article_number}",
                # ── 리스크 매핑 ───────────────────────
                # 매핑 없으면 빈 리스트 → Qdrant 필터에서 제외 가능
                "risk_ids":       risk_ids,
                "risk_names":     risk_names,
                "is_risk_ref":    len(risk_hits) > 0,   # 빠른 필터용 bool
                # ── 청크 정보 ─────────────────────────
                "chunk_text":     chunk_text,
                "chunk_tokens":   count_tokens(chunk_text, tokenizer),
                "sub_index":      idx,
            }
            chunks.append(chunk)

    return chunks


# ──────────────────────────────────────────
# 6. Qdrant 저장
# ──────────────────────────────────────────
def init_qdrant(embed_dim: int) -> QdrantClient:
    client = QdrantClient(path=QDRANT_PATH)

    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION in existing:
        client.delete_collection(collection_name=COLLECTION)
        print(f"🗑️ 기존 컬렉션 삭제: {COLLECTION}")

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
    )

    print(f"✅ 컬렉션 생성: {COLLECTION} (dim={embed_dim})")
    return client


def store_chunks(client: QdrantClient, chunks: list[dict], model) -> None:
    texts = [c["chunk_text"] for c in chunks]
    print(f"  임베딩 생성 중... ({len(texts)}개)")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    points = []
    for chunk, vector in zip(chunks, embeddings):
        payload = {k: v for k, v in chunk.items() if k != "chunk_id"}
        points.append(PointStruct(
            id=chunk["chunk_id"],
            vector=vector.tolist(),
            payload=payload,
        ))

    for i in range(0, len(points), 100):
        client.upsert(collection_name=COLLECTION, points=points[i:i+100])
    print(f"  ✅ {len(points)}개 청크 저장")


# ──────────────────────────────────────────
# 7. 메인
# ──────────────────────────────────────────
def main():
    print("=" * 60)
    print("Workit 법령 KB 구축")
    print("=" * 60)

    # 모델 로드
    print(f"\n📦 모델 로드: {EMBED_MODEL}")
    print("  (첫 실행 시 약 2GB 다운로드)")
    tokenizer   = AutoTokenizer.from_pretrained(EMBED_MODEL)
    embed_model = SentenceTransformer(EMBED_MODEL)
    embed_dim   = embed_model.get_sentence_embedding_dimension()
    print(f"  임베딩 차원: {embed_dim}")

    # taxonomy 역인덱스 빌드
    risk_index = build_risk_index(TAXONOMY)
    print(f"\n📋 taxonomy 로드: {len(TAXONOMY)}개 리스크 유형")
    print(f"   매핑된 조문 수: {len(risk_index)}개")

    # Qdrant 초기화
    client      = init_qdrant(embed_dim)
    total       = 0
    risk_tagged = 0

    # 파일별 처리
    for filename, law_meta in LAW_META.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"\n⚠️  파일 없음: {filepath}")
            continue

        print(f"\n{'─' * 50}")
        print(f"📄 {law_meta['law_name']} ({law_meta['law_type']})")

        articles = load_law_json(filepath)
        print(f"  조문 수: {len(articles)}개")

        dist = measure_token_distribution(articles, tokenizer)
        print(f"  토큰 분포: 평균={dist['mean']}, p95={dist['p95']}, 최대={dist['max']}")

        chunks = chunk_articles(articles, tokenizer, law_meta, filename, risk_index)
        tagged = sum(1 for c in chunks if c["is_risk_ref"])
        print(f"  청크 수: {len(chunks)}개  |  리스크 태깅: {tagged}개")

        store_chunks(client, chunks, embed_model)
        total       += len(chunks)
        risk_tagged += tagged

    # 결과 요약
    print(f"\n{'=' * 60}")
    print(f"✅ 완료!")
    print(f"   총 청크: {total}개")
    print(f"   리스크 태깅된 청크: {risk_tagged}개")
    count = client.count(collection_name=COLLECTION)
    print(f"   Qdrant 저장: {count.count}개")
    print(f"   저장 위치: {QDRANT_PATH}")

    # 검색 테스트 — 리스크 태깅된 것만 필터링
    print(f"\n🔍 검색 테스트: '지연배상금 한도' (리스크 조문만)")

    from qdrant_client.models import Filter, FieldCondition, MatchValue

    query_vec = embed_model.encode("지연배상금 한도")

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vec.tolist(),
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="is_risk_ref",
                    match=MatchValue(value=True)
                )
            ]
        ),
        limit=3,
    ).points

    for i, r in enumerate(results, 1):
        risk_str = ", ".join(r.payload.get("risk_names", [])) or "—"
        print(f"  [{i}] score={r.score:.4f} | {r.payload['source_full']}")
        print(f"       리스크: {risk_str}")
        print(f"       {r.payload['chunk_text'][:80]}...")


if __name__ == "__main__":
    main()