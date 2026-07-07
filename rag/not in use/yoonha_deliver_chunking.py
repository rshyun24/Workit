"""
═══════════════════════════════════════════════════════════════════
Workit RAG Pipeline — 산출물 양식 KB 구축
파일명: yoonha_deliver_chunking.py
작성자: 윤하
위치:   Workit/rag/yoonha_deliver_chunking.py
═══════════════════════════════════════════════════════════════════

■ 이 파일의 역할
──────────────────────────────────────────────────────────────────
data/structured/ 의 산출물 양식 JSON 파일을 직접 읽어
청킹 → 메타데이터 태깅 → Qdrant 저장까지의 전체 파이프라인을 담당합니다.

parser(hwpx/pdf 변환)는 이미 완료된 상태이므로
이 파일은 JSON → Qdrant 적재만 담당합니다.

처리 흐름:
    1. yoonha_qdrant_manager  → Docker 컨테이너 자동 기동
    2. load_parsed_json()     → structured/ JSON 직접 파싱
    3. _split_into_chunks()   → 섹션을 MAX_CHARS 기준으로 분할
    4. build_metadata()       → 각 청크에 메타데이터 부착
    5. embed_texts()          → BGE-M3 임베딩
    6. upsert_sections()      → Qdrant workit_forms 컬렉션에 저장
    (+) yoonha_token_stats_logger → 청킹 전 토큰 분포 측정 및 로그 저장

■ Qdrant 컬렉션
──────────────────────────────────────────────────────────────────
  컬렉션명 : workit_forms
  벡터 차원 : 1024 (BGE-M3 고정)
  거리 방식 : Cosine Similarity

■ 실행 방법
──────────────────────────────────────────────────────────────────
  # structured/ 전체 일괄 적재
  python rag/yoonha_deliver_chunking.py

  # 단일 JSON 파일 지정
  python rag/yoonha_deliver_chunking.py --json data/structured/테스트결과보고서_양식.json

  # 컬렉션 초기화 후 재적재
  python rag/yoonha_deliver_chunking.py --reset

■ 실행 전 설치
──────────────────────────────────────────────────────────────────
  pip install qdrant-client sentence-transformers
"""

import argparse
import hashlib
import json
import uuid
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from yoonha_qdrant_manager import ensure_qdrant_running
from yoonha_token_stats_logger import log_token_stats_from_texts


# ──────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────
STRUCTURED_DIR = Path("data/structured")   # 파싱된 JSON 파일 위치
QDRANT_HOST    = "localhost"               # Qdrant Docker 호스트
QDRANT_PORT    = 6333                      # Qdrant REST API 포트
COLLECTION     = "workit_forms"            # Qdrant 컬렉션명
EMBED_MODEL    = "BAAI/bge-m3"            # 임베딩 모델 (한국어 특화)
MAX_CHARS      = 800                       # 청크 최대 글자 수
OVERLAP        = 100                       # 청크 간 겹치는 글자 수 (문맥 유지용)
VECTOR_DIM     = 1024                      # BGE-M3 임베딩 차원 (고정값)

# 지원하는 산출물 doc_type 목록
SUPPORTED_DOC_TYPES = {
    "테스트결과보고서",
    "사업수행계획서",
    "테스트설계서",
    "최종결과보고서",
}

# 파일명 stem → doc_type 매핑
# 파일명에 공백/언더스코어 혼용 가능하도록 여러 패턴 등록
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

# 섹션 제목/내용 키워드 → section_type 매핑
# 메타데이터 필터링 시 사용 (예: 결과표만 검색)
_SECTION_TYPE_MAP = {
    "결과표":     ["결과 요약", "결과표", "TC", "PASS", "FAIL", "결함"],
    "수치목표":   ["목표", "성능 지표", "성과 지표", "KPI", "달성"],
    "체크리스트": ["체크", "확인 항목", "점검", "준수 여부"],
    "서술":       [],   # 위 키워드 미해당 시 기본값
}


# ──────────────────────────────────────────
# 1. ParsedSection 데이터 클래스
# ──────────────────────────────────────────
@dataclass
class ParsedSection:
    """
    JSON에서 읽어온 산출물 섹션 하나를 담는 데이터 클래스입니다.

    parser(hwpx/pdf → JSON 변환)가 생성한 JSON 구조와 동일하게
    이 파일 안에서 직접 정의하여 parser 파일 의존성을 제거했습니다.

    Attributes:
        section_path  : 섹션 계층 경로 (예: ["1. 개요", "1.1 목적"])
        section_title : 현재 섹션 제목 (예: "1.1 목적")
        section_depth : 계층 깊이 (1=장, 2=절, 3=항)
        text          : 섹션 본문 텍스트 (청킹 대상)
        raw_lines     : 원본 라인 리스트
    """
    section_path:  list[str]
    section_title: str
    section_depth: int
    text:          str
    raw_lines:     list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ParsedSection":
        """JSON 딕셔너리에서 ParsedSection 객체를 생성합니다."""
        return cls(
            section_path  = d["section_path"],
            section_title = d["section_title"],
            section_depth = d["section_depth"],
            text          = d["text"],
            raw_lines     = d.get("raw_lines", []),
        )


# ──────────────────────────────────────────
# 2. JSON 로드
# ──────────────────────────────────────────
def load_parsed_json(json_path: str | Path) -> list[ParsedSection]:
    """
    structured/ 폴더의 JSON 파일을 읽어 ParsedSection 리스트로 반환합니다.

    parser(hwpx/pdf → JSON)가 이미 생성한 파일을 직접 읽으므로
    별도 parser import 없이 동작합니다.

    Args:
        json_path : JSON 파일 경로

    Returns:
        list[ParsedSection]: 섹션 객체 리스트
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [ParsedSection.from_dict(d) for d in data]


# ──────────────────────────────────────────
# 3. 유틸
# ──────────────────────────────────────────
def _infer_section_type(title: str, text: str) -> str:
    """
    섹션 제목과 본문 키워드로 section_type을 추론합니다.

    _SECTION_TYPE_MAP 의 키워드를 순서대로 검사하며
    첫 번째로 매칭되는 타입을 반환합니다.
    어떤 키워드도 매칭되지 않으면 "서술" 을 반환합니다.

    Args:
        title : 섹션 제목 (예: "3.2 테스트 결과표")
        text  : 섹션 본문 텍스트

    Returns:
        str: "결과표" / "수치목표" / "체크리스트" / "서술" 중 하나
    """
    combined = title + " " + text
    for stype, keywords in _SECTION_TYPE_MAP.items():
        if any(kw in combined for kw in keywords):
            return stype
    return "서술"


def _infer_doc_type(stem: str) -> Optional[str]:
    """
    JSON 파일명(확장자 제외)에서 doc_type을 추론합니다.

    1) 정확히 일치하는 키 탐색
    2) 부분 문자열로 포함 여부 탐색
    3) 매칭 실패 시 None 반환 (법령 파일 등은 스킵 처리됨)

    Args:
        stem : Path.stem 값 (예: "테스트 결과보고서 양식")

    Returns:
        str | None: doc_type 문자열 또는 None
    """
    if stem in FILENAME_TO_DOC_TYPE:
        return FILENAME_TO_DOC_TYPE[stem]
    for keyword, doc_type in FILENAME_TO_DOC_TYPE.items():
        if keyword in stem:
            return doc_type
    return None


# ──────────────────────────────────────────
# 4. 청크 분할
# ──────────────────────────────────────────
def _split_into_chunks(section: ParsedSection) -> list[dict]:
    """
    하나의 섹션 텍스트를 MAX_CHARS 기준으로 청크 단위로 분할합니다.

    텍스트가 MAX_CHARS 이하면 분할 없이 단일 청크로 반환합니다.
    초과하면 OVERLAP 글자씩 겹치도록 슬라이딩 윈도우 방식으로 분할합니다.
    겹침(OVERLAP)은 청크 경계에서 문맥이 끊기는 것을 방지합니다.

    Args:
        section : ParsedSection 객체 (text 속성 사용)

    Returns:
        list[dict]: {"text": str, "chunk_index": int, "chunk_total": int} 리스트
    """
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
# 5. chunk_id / doc_id 생성 (해시 기반)
# ──────────────────────────────────────────
def _make_doc_id(source_file: str, doc_type: str, doc_version: str) -> str:
    """
    문서 단위 고유 ID를 생성합니다.

    source_file + doc_type + doc_version 조합을 MD5 해싱하여
    동일 문서는 항상 동일한 ID를 갖도록 보장합니다. (멱등성)

    Args:
        source_file : JSON 파일명 stem (예: "테스트 결과보고서 양식")
        doc_type    : 산출물 유형 (예: "테스트결과보고서")
        doc_version : 버전 문자열 (예: "v1.0")

    Returns:
        str: "workit-doc-{8자리 해시}" 형태의 ID
    """
    raw = f"{source_file}::{doc_type}::{doc_version}"
    return "workit-doc-" + hashlib.md5(raw.encode()).hexdigest()[:8]


def _make_chunk_id(doc_id: str, section_title: str, chunk_index: int) -> str:
    """
    청크 단위 고유 ID를 생성합니다.

    doc_id + section_title + chunk_index 조합을 MD5 해싱합니다.
    동일 청크는 재실행 시에도 항상 동일한 ID → Qdrant upsert 시 중복 방지.

    Args:
        doc_id        : _make_doc_id() 반환값
        section_title : 섹션 제목 (예: "3.2 테스트 결과표")
        chunk_index   : 해당 섹션 내 청크 순번 (0부터 시작)

    Returns:
        str: "workit-chunk-{12자리 해시}" 형태의 ID
    """
    raw = f"{doc_id}::{section_title}::{chunk_index}"
    return "workit-chunk-" + hashlib.md5(raw.encode()).hexdigest()[:12]


# ──────────────────────────────────────────
# 6. 메타데이터 생성
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
    """
    청크에 부착할 메타데이터 딕셔너리를 생성합니다.

    Qdrant payload로 저장되며 검색 필터링에 활용됩니다.
    예) doc_type="테스트결과보고서" 인 청크만 검색
        section_type="결과표" 인 청크만 검색

    Args:
        section             : ParsedSection 객체
        chunk_index         : 청크 순번
        chunk_total         : 해당 섹션의 전체 청크 수
        doc_type            : 산출물 유형
        source_file         : 원본 파일명
        doc_version         : 버전 (기본값 "v1.0")
        is_required         : 필수 섹션 여부 (기본값 True)
        guideline_reference : 행안부 가이드라인 출처 (병합 전엔 None)
        guideline_merged    : 행안부 기준 텍스트 병합 여부

    Returns:
        dict: Qdrant payload에 저장될 메타데이터
    """
    doc_id   = _make_doc_id(source_file, doc_type, doc_version)
    chunk_id = _make_chunk_id(doc_id, section.section_title, chunk_index)

    return {
        "chunk_id":            chunk_id,
        "source_type":         "form",         # law 컬렉션과 구분하는 식별자
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
# 7. 임베딩 모델
# ──────────────────────────────────────────
_model_cache: dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    """
    임베딩 모델을 로드하고 캐싱합니다.

    동일 모델을 여러 번 호출해도 최초 1회만 로드하고
    이후에는 캐시에서 반환합니다. (메모리 절약 + 속도 향상)

    Args:
        model_name : HuggingFace 모델명 (기본값: BAAI/bge-m3)

    Returns:
        SentenceTransformer: 로드된 임베딩 모델
    """
    if model_name not in _model_cache:
        print(f"[indexer] 임베딩 모델 로딩: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_texts(texts: list[str], model_name: str = EMBED_MODEL) -> list[list[float]]:
    """
    텍스트 리스트를 임베딩 벡터로 변환합니다.

    normalize_embeddings=True 로 L2 정규화하여
    코사인 유사도 계산에 최적화된 벡터를 반환합니다.

    Args:
        texts      : 임베딩할 텍스트 리스트
        model_name : 사용할 모델명

    Returns:
        list[list[float]]: 각 텍스트의 임베딩 벡터 (1024차원)
    """
    model = get_embedding_model(model_name)
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()


# ──────────────────────────────────────────
# 8. Qdrant 초기화
# ──────────────────────────────────────────
def get_qdrant_client() -> QdrantClient:
    """
    Qdrant 클라이언트를 생성하여 반환합니다.

    yoonha_qdrant_manager.ensure_qdrant_running() 이 먼저 호출된
    이후에 사용되므로 컨테이너 실행 여부는 이미 보장된 상태입니다.

    Returns:
        QdrantClient: 연결된 Qdrant 클라이언트
    """
    print(f"[indexer] Qdrant 연결: {QDRANT_HOST}:{QDRANT_PORT}")
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client: QdrantClient, reset: bool = False) -> None:
    """
    workit_forms 컬렉션이 없으면 생성하고, 있으면 그대로 사용합니다.

    reset=True 이면 기존 컬렉션을 삭제 후 재생성합니다.
    데이터를 완전히 초기화하고 처음부터 재적재할 때 사용합니다.

    Args:
        client : QdrantClient 인스턴스
        reset  : True이면 기존 컬렉션 삭제 후 재생성
    """
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
# 9. 적재 (upsert)
# ──────────────────────────────────────────
def upsert_sections(
    client:      QdrantClient,
    sections:    list[ParsedSection],
    doc_type:    str,
    source_file: str,
    doc_version: str = "v1.0",
    batch_size:  int = 32,
) -> int:
    """
    ParsedSection 리스트를 청킹 → 임베딩 → Qdrant upsert 합니다.

    처리 순서:
        1. 토큰 통계 측정 및 로그 저장 (yoonha_token_stats_logger)
        2. 각 섹션을 MAX_CHARS 기준으로 청크 분할
        3. 메타데이터 생성 및 PointStruct 구성
        4. batch_size 단위로 임베딩 후 벡터 할당
        5. Qdrant upsert (동일 chunk_id면 덮어쓰기)

    chunk_id는 해시 기반이므로 동일 데이터를 재실행해도
    중복 포인트 없이 업데이트됩니다. (멱등성 보장)

    Args:
        client      : QdrantClient 인스턴스
        sections    : ParsedSection 리스트
        doc_type    : 산출물 유형
        source_file : 원본 파일명
        doc_version : 버전 문자열
        batch_size  : 한 번에 임베딩할 청크 수 (메모리 조절용)

    Returns:
        int: 실제 적재된 포인트(청크) 수
    """
    # 토큰 통계 측정 및 로그 저장 (임베딩 전 원본 섹션 기준)
    model     = get_embedding_model()
    tokenizer = model.tokenizer
    texts_for_stats = [s.text for s in sections if s.text.strip()]
    log_token_stats_from_texts(f"deliver:{doc_type}", texts_for_stats, tokenizer)

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
                payload={**metadata, "text": cd["text"]},
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
    """
    특정 doc_id에 해당하는 포인트를 전체 삭제합니다.

    양식이 개정되어 특정 문서만 교체할 때 사용합니다.
    컬렉션 전체를 초기화하지 않고 해당 문서 청크만 삭제 후
    새 버전을 upsert 하는 방식으로 부분 업데이트가 가능합니다.

    Args:
        client : QdrantClient 인스턴스
        doc_id : 삭제할 문서의 doc_id (_make_doc_id() 반환값)
    """
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
    """
    행안부 가이드라인 텍스트를 기존 청크에 병합하고 재임베딩합니다.

    산출물 양식 청크에 행안부 작성 기준을 추가하면
    "이 섹션을 어떻게 작성해야 하는가" 에 대한 검색 정확도가 높아집니다.

    병합 후 payload 변경:
        text              : "[양식 내용]\n...\n[작성 기준]\n..." 형태로 확장
        guideline_merged  : False → True
        guideline_reference : 근거 문서명으로 업데이트

    Args:
        client               : QdrantClient 인스턴스
        chunk_id             : 병합 대상 청크의 chunk_id
        guideline_text       : 행안부 가이드라인 본문
        guideline_reference  : 가이드라인 출처 (예: "행안부 SW사업 산출물 관리 가이드 v2.0")
    """
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
# 10. structured/ 전체 일괄 적재
# ──────────────────────────────────────────
def index_all_structured(doc_version: str = "v1.0", reset: bool = False) -> None:
    """
    data/structured/ 폴더의 모든 JSON 파일을 순서대로 적재합니다.

    FILENAME_TO_DOC_TYPE 에 매핑되지 않는 파일(법령 JSON 등)은
    자동으로 스킵됩니다.

    Args:
        doc_version : 전체 적재 시 적용할 버전 문자열
        reset       : True이면 컬렉션 초기화 후 재적재
    """
    json_files = sorted(STRUCTURED_DIR.glob("*.json"))

    if not json_files:
        print(f"[indexer] structured 폴더에 JSON 파일이 없습니다: {STRUCTURED_DIR}")
        return

    client = get_qdrant_client()
    ensure_collection(client, reset=reset)
    total_points = 0

    for json_path in json_files:
        doc_type = _infer_doc_type(json_path.stem)
        if doc_type is None:
            print(f"[indexer] 스킵: {json_path.name} (doc_type 매핑 없음)")
            continue

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
# 11. CLI
# ──────────────────────────────────────────
def main():
    # Qdrant Docker 컨테이너 자동 기동
    ensure_qdrant_running()

    ap = argparse.ArgumentParser(description="Workit 산출물 인덱서")
    ap.add_argument("--json",        default=None, help="structured/ JSON 경로 (단일 파일)")
    ap.add_argument("--doc_type",    default=None, choices=list(SUPPORTED_DOC_TYPES))
    ap.add_argument("--doc_version", default="v1.0")
    ap.add_argument(
        "--reset",
        action="store_true",
        help="컬렉션 삭제 후 재구축 (기본값: 기존 컬렉션 유지하며 upsert)",
    )
    args = ap.parse_args()

    if args.json is None:
        index_all_structured(doc_version=args.doc_version, reset=args.reset)
        return

    json_path = Path(args.json)
    doc_type  = args.doc_type or _infer_doc_type(json_path.stem)
    if not doc_type:
        print("[ERROR] doc_type 을 --doc_type 으로 지정해주세요.")
        return

    sections = load_parsed_json(json_path)
    client   = get_qdrant_client()
    ensure_collection(client, reset=args.reset)
    total = upsert_sections(
        client=client,
        sections=sections,
        doc_type=doc_type,
        source_file=json_path.stem,
        doc_version=args.doc_version,
    )
    print(f"\n완료: {total}개 포인트 적재됨 → {QDRANT_HOST}:{QDRANT_PORT}")


if __name__ == "__main__":
    main()