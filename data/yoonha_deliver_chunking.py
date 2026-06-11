"""
Workit - 산출물 양식 KB 구축
산출물 양식 JSON → 청킹 → 메타데이터 태깅 → Qdrant 저장

실행 전 설치:
    pip install qdrant-client sentence-transformers
"""

import argparse
import hashlib
import uuid
import sys
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from yoonha_deliver_parser import ParsedSection, parse_file, load_parsed_json


# ──────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────
DATA_DIR       = Path("data/structured")
QDRANT_HOST    = "localhost"
QDRANT_PORT    = 6333
COLLECTION     = "workit_forms"
EMBED_MODEL    = "BAAI/bge-m3"
MAX_CHARS      = 800
OVERLAP        = 100
VECTOR_DIM     = 1024
STRUCTURED_DIR = DATA_DIR

SUPPORTED_DOC_TYPES = {
    "테스트결과보고서",
    "사업수행계획서",
    "테스트설계서",
    "최종결과보고서",
}

FILENAME_TO_DOC_TYPE: dict[str, str] = {
    "테스트 결과보고서 양식": "테스트결과보고서",
    "테스트결과보고서 양식":  "테스트결과보고서",
    "테스트_결과보고서_양식": "테스트결과보고서",
    "테스트 설계서 양식":     "테스트설계서",
    "테스트설계서 양식":      "테스트설계서",
    "테스트_설계서_양식":     "테스트설계서",
    "사업수행계획서 양식":    "사업수행계획서",
    "사업수행계획서_양식":    "사업수행계획서",
    "최종결과보고서 양식":    "최종결과보고서",
    "최종결과보고서_양식":    "최종결과보고서",
}

_SECTION_TYPE_MAP = {
    "결과표":     ["결과 요약", "결과표", "TC", "PASS", "FAIL", "결함"],
    "수치목표":   ["목표", "성능 지표", "성과 지표", "KPI", "달성"],
    "체크리스트": ["체크", "확인 항목", "점검", "준수 여부"],
    "서술":       [],
}


# ──────────────────────────────────────────
# 1. 유틸
# ──────────────────────────────────────────
def _infer_section_type(title: str, text: str) -> str:
    combined = title + " " + text
    for stype, keywords in _SECTION_TYPE_MAP.items():
        if any(kw in combined for kw in keywords):
            return stype
    return "서술"


def _infer_doc_type(stem: str) -> Optional[str]:
    if stem in FILENAME_TO_DOC_TYPE:
        return FILENAME_TO_DOC_TYPE[stem]
    for keyword, doc_type in FILENAME_TO_DOC_TYPE.items():
        if keyword in stem:
            return doc_type
    return None


# ──────────────────────────────────────────
# 2. 청크 분할
# ──────────────────────────────────────────
def _split_into_chunks(section: ParsedSection) -> list[dict]:
    text = section.text
    if len(text) <= MAX_CHARS:
        return [{"text": text, "chunk_index": 0, "chunk_total": 1}]

    chunks, start = [], 0
    step = MAX_CHARS - OVERLAP
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step

    total = len(chunks)
    return [{"text": c, "chunk_index": i, "chunk_total": total}
            for i, c in enumerate(chunks)]


# ──────────────────────────────────────────
# 3. chunk_id / doc_id 생성 (해시 기반)
# ──────────────────────────────────────────
def _make_doc_id(source_file: str, doc_type: str, doc_version: str) -> str:
    raw = f"{source_file}::{doc_type}::{doc_version}"
    return "workit-doc-" + hashlib.md5(raw.encode()).hexdigest()[:8]


def _make_chunk_id(doc_id: str, section_title: str, chunk_index: int) -> str:
    raw = f"{doc_id}::{section_title}::{chunk_index}"
    return "workit-chunk-" + hashlib.md5(raw.encode()).hexdigest()[:12]


# ──────────────────────────────────────────
# 4. 메타데이터 생성
# ──────────────────────────────────────────
def build_metadata(
    section:             ParsedSection,
    chunk_index:         int,
    chunk_total:         int,
    doc_type:            str,
    source_file:         str,
    doc_version:         str = "v1.0",
    is_required:         bool = True,
    guideline_reference: Optional[str] = None,
    guideline_merged:    bool = False,
) -> dict:
    doc_id   = _make_doc_id(source_file, doc_type, doc_version)
    chunk_id = _make_chunk_id(doc_id, section.section_title, chunk_index)

    return {
        "chunk_id":            chunk_id,
        "source_type":         "form",                  # ← law와 구분
        "doc_id":              doc_id,
        "doc_type":            doc_type,
        "doc_version":         doc_version,
        "source_file":         source_file,
        "section_path":        section.section_path,
        "section_title":       section.section_title,
        "section_depth":       section.section_depth,
        "chunk_index":         chunk_index,
        "chunk_total":         chunk_total,
        "is_required":         is_required,
        "section_type":        _infer_section_type(section.section_title, section.text),
        "guideline_reference": guideline_reference,
        "guideline_merged":    guideline_merged,
    }


# ──────────────────────────────────────────
# 5. 임베딩 모델
# ──────────────────────────────────────────
_model_cache: dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    if model_name not in _model_cache:
        print(f"[indexer] 임베딩 모델 로딩: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_texts(texts: list[str], model_name: str = EMBED_MODEL) -> list[list[float]]:
    model = get_embedding_model(model_name)
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()


# ──────────────────────────────────────────
# 6. Qdrant — ensure 방식 초기화
# ──────────────────────────────────────────
def get_qdrant_client(in_memory: bool = False) -> QdrantClient:
    if in_memory:
        return QdrantClient(":memory:")
    print(f"[indexer] Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(
    client:    QdrantClient,
    reset:     bool = False,
) -> None:
    existing = [c.name for c in client.get_collections().collections]

    if reset and COLLECTION in existing:
        client.delete_collection(collection_name=COLLECTION)
        print(f"[indexer] 🗑️  컬렉션 초기화: {COLLECTION}")
        existing = []

    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"[indexer] ✅ 컬렉션 생성: {COLLECTION} (dim={VECTOR_DIM})")
    else:
        print(f"[indexer] ✅ 컬렉션 기존 사용: {COLLECTION}")


# ──────────────────────────────────────────
# 7. 적재 (upsert)
# ──────────────────────────────────────────
def upsert_sections(
    client:      QdrantClient,
    sections:    list[ParsedSection],
    doc_type:    str,
    source_file: str,
    doc_version: str = "v1.0",
    batch_size:  int = 32,
) -> int:
    all_points: list[PointStruct] = []

    for section in sections:
        for cd in _split_into_chunks(section):
            metadata = build_metadata(
                section=section,
                chunk_index=cd["chunk_index"],
                chunk_total=cd["chunk_total"],
                doc_type=doc_type,
                source_file=source_file,
                doc_version=doc_version,
            )
            all_points.append(PointStruct(
                id=str(uuid.UUID(hashlib.md5(metadata["chunk_id"].encode()).hexdigest())),
                vector=[0.0] * VECTOR_DIM,
                payload={**metadata, "text": cd["text"]},   # ← 통일된 키
            ))

    texts = [p.payload["text"] for p in all_points]
    print(f"[indexer] 임베딩 중... {len(texts)}개 청크")

    for i in range(0, len(texts), batch_size):
        batch_vecs = embed_texts(texts[i:i + batch_size])
        for j, vec in enumerate(batch_vecs):
            all_points[i + j].vector = vec

    client.upsert(collection_name=COLLECTION, points=all_points)
    print(f"[indexer] ✅ {len(all_points)}개 포인트 적재 완료 (doc_type={doc_type})")
    return len(all_points)


def delete_by_doc_id(client: QdrantClient, doc_id: str) -> None:
    """특정 doc_id 포인트 전체 삭제 (양식 교체 시)."""
    client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
    )
    print(f"[indexer] doc_id={doc_id} 삭제 완료")


def merge_guideline(
    client:              QdrantClient,
    chunk_id:            str,
    guideline_text:      str,
    guideline_reference: str,
) -> None:
    """행안부 기준 텍스트를 청크에 병합하고 재임베딩."""
    scroll_result, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id))]
        ),
        limit=1, with_payload=True, with_vectors=True,
    )
    if not scroll_result:
        raise ValueError(f"chunk_id={chunk_id} 미존재")

    point = scroll_result[0]
    merged_text = (
        f"[양식 내용]\n{point.payload['text']}\n\n"
        f"[작성 기준]\n{guideline_text}\n근거: {guideline_reference}"
    )
    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(
            id=point.id,
            vector=embed_texts([merged_text])[0],
            payload={
                **point.payload,
                "text":                merged_text,
                "guideline_reference": guideline_reference,
                "guideline_merged":    True,
            },
        )],
    )
    print(f"[indexer] chunk_id={chunk_id} 행안부 기준 병합 완료")


# ──────────────────────────────────────────
# 8. structured/ 전체 일괄 적재
# ──────────────────────────────────────────
def index_all_structured(doc_version: str = "v1.0", reset: bool = False) -> None:
    json_files = sorted(Path(STRUCTURED_DIR).glob("*.json"))

    if not json_files:
        print(f"[indexer] structured 폴더에 JSON 파일이 없습니다: {STRUCTURED_DIR}")
        print("  → 먼저 parser.py 를 실행해 JSON을 생성하세요.")
        return

    client = get_qdrant_client()
    ensure_collection(client, reset=reset)
    total_points = 0

    for json_path in json_files:
        doc_type = _infer_doc_type(json_path.stem)
        if doc_type is None:
            continue  # 법령 등 산출물 외 파일 스킵

        print(f"\n[indexer] {json_path.name} → doc_type={doc_type}")
        sections = load_parsed_json(json_path)
        n = upsert_sections(
            client=client,
            sections=sections,
            doc_type=doc_type,
            source_file=json_path.stem,
            doc_version=doc_version,
        )
        total_points += n

    print(f"\n[indexer] 전체 완료: 총 {total_points}개 포인트 적재됨")
    print(f"[indexer] Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")


# ──────────────────────────────────────────
# 9. CLI
# ──────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Workit 산출물 인덱서")
    ap.add_argument("--json",        default=None, help="structured/ JSON 경로 (단일 파일)")
    ap.add_argument("--file",        default=None, help="원본 hwpx/pdf 경로 (JSON 없을 때)")
    ap.add_argument("--doc_type",    default=None, choices=list(SUPPORTED_DOC_TYPES))
    ap.add_argument("--doc_version", default="v1.0")
    ap.add_argument(
        "--reset",
        action="store_true",
        help="컬렉션 삭제 후 재구축 (기본값: 기존 컬렉션 유지하며 upsert)",
    )
    args = ap.parse_args()

    # 인자 없음 → structured/ 전체 자동 적재
    if args.json is None and args.file is None:
        index_all_structured(doc_version=args.doc_version, reset=args.reset)
        return

    # JSON 지정
    if args.json:
        json_path = Path(args.json)
        doc_type  = args.doc_type or _infer_doc_type(json_path.stem)
        if not doc_type:
            print("[ERROR] doc_type 을 --doc_type 으로 지정해주세요.")
            return
        sections = load_parsed_json(json_path)
        source   = json_path.stem

    # 원본 파일 지정
    elif args.file:
        if not args.doc_type:
            print("[ERROR] --file 사용 시 --doc_type 필수")
            return
        file_path = Path(args.file)
        sections  = parse_file(file_path)
        doc_type  = args.doc_type
        source    = file_path.name

    client = get_qdrant_client()
    ensure_collection(client, reset=args.reset)
    total  = upsert_sections(
        client=client,
        sections=sections,
        doc_type=doc_type,
        source_file=source,
        doc_version=args.doc_version,
    )
    print(f"\n완료: {total}개 포인트 적재됨 → {QDRANT_HOST}:{QDRANT_PORT}")


if __name__ == "__main__":
    main()