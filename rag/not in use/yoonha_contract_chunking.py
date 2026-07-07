"""
═══════════════════════════════════════════════════════════════════
Workit RAG Pipeline — 계약서 KB 구축
파일명: yoonha_contract_chunking.py
위치:   Workit/rag/yoonha_contract_chunking.py
═══════════════════════════════════════════════════════════════════

■ 이 파일의 역할
──────────────────────────────────────────────────────────────────
계약서 파싱 JSON (yoonha_contract_parser.py 출력) 을 읽어
청킹 → 리스크 태그 부착 → Qdrant 저장까지의 전체 파이프라인을
담당합니다.

처리 흐름:
    1. yoonha_qdrant_manager    → Docker 컨테이너 자동 기동
    2. load_contract_json()     → 파싱 JSON 로드 (전문 + 본문)
    3. chunk_contract()         → 전문/본문 영역별 청킹
                                  + RISK taxonomy 태그 부착
    4. store_chunks()           → BGE-M3 임베딩 후 Qdrant 저장
    (+) yoonha_token_stats_logger → 청킹 전 토큰 분포 측정 및 로그 저장

■ 계약서 JSON 구조 (yoonha_contract_parser.py 출력)
──────────────────────────────────────────────────────────────────
  {
    "contract_id"  : str,
    "doc_type"     : "정보시스템_개발구축_표준계약서",
    "file_format"  : "pdf" | "docx",
    "전문": {
        "contract_name", "start_date", "end_date", "due_date",
        "total_amount", "supply_amount", "vat",
        "payment_schedule": [{"type", "rate", "amount", "date", "method"}, ...],
        "performance_bond", "performance_bond_rate",
        "task_confirm_date", "delay_rate",
        "penalty_limit", "penalty_limit_rate",
        "warranty_bond", "warranty_bond_rate", "warranty_period",
        "client": {"name", "ceo", "addr", "tel", "biz_no"},
        "vendor": {"name", "ceo", "addr", "tel", "biz_no"},
    },
    "본문": [{"article_id", "article_number", "title", "text"}, ...],
    "raw_issues": [str, ...]
  }

■ 청크 유형 (chunk_type)
──────────────────────────────────────────────────────────────────
  "전문_요약"   : 계약 기본 정보 (금액, 기간, 납기일 등) 요약 텍스트 1개
  "전문_대금"   : 선급금/중도금/잔금 각 행 → 행별 1개씩
  "전문_보증"   : 보증금/지체상금/하자 등 조건값 → 항목별 묶음
  "전문_당사자" : 발주자/공급자 정보 요약 1개
  "본문_조항"   : 제N조 단위 청크 (RSC로 MAX_TOKENS 초과 시 분할)

■ 리스크 taxonomy
──────────────────────────────────────────────────────────────────
  law_chunking.py 의 9개 리스크 유형과 동일한 RISK_ID를 사용합니다.
  계약서 전문의 수치 조건이 리스크 기준을 위반하는지 여부를
  chunk 생성 시점에 태깅합니다.

  계약서 청크에서 is_risk_ref=True 인 경우:
    → 해당 청크가 직접 리스크 조건을 포함함을 의미
    → 법령 KB 검색 시 동일 risk_id 로 관련 법령 조문 교차 검색 가능

■ Qdrant 컬렉션
──────────────────────────────────────────────────────────────────
  컬렉션명 : contract_kb
  벡터 차원 : BGE-M3 자동 감지
  거리 방식 : Cosine Similarity

■ 실행 방법
──────────────────────────────────────────────────────────────────
  # 전체 계약서 적재
  python yoonha_contract_chunking.py

  # 특정 폴더 지정
  python yoonha_contract_chunking.py --input data/contract_parsed

  # 컬렉션 초기화 후 재적재
  python yoonha_contract_chunking.py --reset

■ 실행 전 설치
──────────────────────────────────────────────────────────────────
  pip install qdrant-client sentence-transformers numpy
"""

import argparse
import hashlib
import json
import uuid
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from yoonha_qdrant_manager import ensure_qdrant_running
from yoonha_token_stats_logger import log_token_stats_from_texts


# ──────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────
DATA_DIR    = Path("data/contract_parsed")  # 계약서 파싱 JSON 위치
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION  = "contract_kb"
EMBED_MODEL = "BAAI/bge-m3"
MAX_TOKENS  = 1024
MIN_TOKENS  = 10


# ──────────────────────────────────────────
# 1. 리스크 taxonomy
#    (law_chunking.py 의 RISK_ID와 동일하게 맞춤)
# ──────────────────────────────────────────
# 계약서 전문의 수치 조건을 체크하는 rule 기반 리스크
# check_fn(전문_dict) → bool (True면 리스크 있음)
CONTRACT_RISK_RULES = [
    {
        "risk_id":   "RISK_001",
        "risk_name": "지연배상금 상한 미설정",
        "field":     "penalty_limit",
        "check_fn":  lambda f: f.get("penalty_limit") in (None, 0),
        "desc":      "지체상금 한도 금액이 미설정(0 또는 공란)입니다.",
    },
    {
        "risk_id":   "RISK_002",
        "risk_name": "지연배상금률 과다 설정",
        "field":     "penalty_limit_rate",
        "check_fn":  lambda f: (f.get("penalty_limit_rate") or 0) > 30,
        "desc":      "지체상금 한도 비율이 계약금액의 30%를 초과합니다.",
    },
    {
        "risk_id":   "RISK_003",
        "risk_name": "대금 지급 기한 미설정",
        "field":     "delay_rate",
        "check_fn":  lambda f: f.get("delay_rate") in (None, 0),
        "desc":      "지연이자요율이 0% 또는 미기재입니다.",
    },
    {
        "risk_id":   "RISK_004",
        "risk_name": "선급금 미지급",
        "field":     "payment_schedule",
        "check_fn":  lambda f: _check_prepay_missing(f),
        "desc":      "계약금액 2,000만 원 이상임에도 선급금이 0%로 설정되어 있습니다.",
    },
    {
        "risk_id":   "RISK_005",
        "risk_name": "선급금 비율 초과",
        "field":     "payment_schedule",
        "check_fn":  lambda f: _check_prepay_excess(f),
        "desc":      "선급금 비율이 30%를 초과합니다.",
    },
    {
        "risk_id":   "RISK_006",
        "risk_name": "계약이행보증금 초과 설정",
        "field":     "performance_bond_rate",
        "check_fn":  lambda f: (f.get("performance_bond_rate") or 0) > 10,
        "desc":      "계약이행보증금이 계약금액의 10%를 초과합니다.",
    },
    {
        "risk_id":   "RISK_007",
        "risk_name": "하자보수보증금 초과 설정",
        "field":     "warranty_bond_rate",
        "check_fn":  lambda f: (f.get("warranty_bond_rate") or 0) > 10,
        "desc":      "하자보수보증금이 계약금액의 10%를 초과합니다.",
    },
    {
        "risk_id":   "RISK_008",
        "risk_name": "하자담보책임기간 초과",
        "field":     "warranty_period",
        "check_fn":  lambda f: _check_warranty_period(f),
        "desc":      "하자담보책임기간이 1년을 초과합니다.",
    },
    {
        "risk_id":   "RISK_009",
        "risk_name": "납기일_계약기간 오류",
        "field":     "due_date",
        "check_fn":  lambda f: _check_due_date(f),
        "desc":      "납기일이 계약 종료일보다 늦게 설정되어 있습니다.",
    },
]


# ── 리스크 체크 헬퍼 함수 ─────────────────
def _check_prepay_missing(전문: dict) -> bool:
    """선급금 0% 여부 (계약금액 2천만 이상 조건)"""
    total = 전문.get("total_amount") or 0
    if total < 20000000:
        return False
    for row in 전문.get("payment_schedule", []):
        if row.get("type") == "선급금":
            rate_str = row.get("rate", "").replace("%", "").strip()
            try:
                return float(rate_str) == 0
            except ValueError:
                return False
    return False


def _check_prepay_excess(전문: dict) -> bool:
    """선급금 30% 초과 여부"""
    for row in 전문.get("payment_schedule", []):
        if row.get("type") == "선급금":
            rate_str = row.get("rate", "").replace("%", "").strip()
            try:
                return float(rate_str) > 30
            except ValueError:
                return False
    return False


def _check_warranty_period(전문: dict) -> bool:
    """하자담보책임기간 '2년' 포함 여부"""
    period = 전문.get("warranty_period", "")
    import re
    m = re.search(r"(\d+)\s*년", period)
    if m:
        return int(m.group(1)) > 1
    return False


def _check_due_date(전문: dict) -> bool:
    """납기일 > 계약 종료일 여부"""
    due = 전문.get("due_date", "")
    end = 전문.get("end_date", "")
    if due and end and len(due) == 10 and len(end) == 10:
        return due > end
    return False


# ──────────────────────────────────────────
# 2. 계약서 전문 → 텍스트 변환 함수들
# ──────────────────────────────────────────

def _전문_to_summary_text(전문: dict) -> str:
    """계약 기본 정보를 자연어 요약 텍스트로 변환"""
    lines = [
        f"계약명: {전문.get('contract_name', '')}",
        f"계약기간: {전문.get('start_date', '')} ~ {전문.get('end_date', '')}",
        f"납기일: {전문.get('due_date', '')}",
        f"계약 체결일: {전문.get('contract_date', '')}",
    ]
    if 전문.get("total_amount"):
        lines.append(f"계약금액: {전문['total_amount']:,}원 "
                     f"(공급가액 {전문.get('supply_amount', 0):,}원 + "
                     f"부가가치세 {전문.get('vat', 0):,}원)")
    return "\n".join(lines)


def _전문_to_payment_texts(전문: dict) -> list[tuple[str, str]]:
    """대금 지급 계획 각 행 → (행 유형, 텍스트) 리스트"""
    results = []
    for row in 전문.get("payment_schedule", []):
        ptype = row.get("type", "")
        rate  = row.get("rate", "")
        amt   = row.get("amount")
        date  = row.get("date", "")
        meth  = row.get("method", "")
        amt_str = f"{amt:,}원" if amt is not None else "미기재"
        text = (
            f"대금 지급 - {ptype}: "
            f"비율 {rate}, 금액 {amt_str}, "
            f"지급기일 {date}, 지급방법 {meth}"
        )
        results.append((ptype, text))
    return results


def _전문_to_guarantee_text(전문: dict) -> str:
    """보증금/지체상금/하자 등 수치 조건을 하나의 텍스트로 묶음"""
    lines = []

    pb      = 전문.get("performance_bond")
    pb_rate = 전문.get("performance_bond_rate")
    if pb is not None:
        lines.append(f"계약이행보증금: {pb:,}원 (계약금액의 {pb_rate}%)")

    tc = 전문.get("task_confirm_date", "")
    if tc:
        lines.append(f"과업내용서 발급 예정기일: {tc}")

    dr = 전문.get("delay_rate")
    lines.append(f"지연이자요율: {dr}%" if dr else "지연이자요율: 미기재")

    pl      = 전문.get("penalty_limit")
    pl_rate = 전문.get("penalty_limit_rate")
    if pl is not None:
        lines.append(f"지체상금 한도: {pl:,}원 (계약금액의 {pl_rate}%)")
    else:
        lines.append("지체상금 한도: 미기재")

    wb      = 전문.get("warranty_bond")
    wb_rate = 전문.get("warranty_bond_rate")
    if wb is not None:
        lines.append(f"하자보수보증금: {wb:,}원 (계약금액의 {wb_rate}%)")

    wp = 전문.get("warranty_period", "")
    if wp:
        lines.append(f"하자담보책임기간: {wp}")

    return "\n".join(lines)


def _전문_to_party_text(전문: dict) -> str:
    """발주자/공급자 정보 요약 텍스트"""
    c = 전문.get("client", {})
    v = 전문.get("vendor", {})
    lines = [
        f"발주자: {c.get('name', '')} | 대표자: {c.get('ceo', '')} | "
        f"주소: {c.get('addr', '')} | 전화: {c.get('tel', '')} | "
        f"사업자번호: {c.get('biz_no', '')}",
        f"공급자: {v.get('name', '')} | 대표자: {v.get('ceo', '')} | "
        f"주소: {v.get('addr', '')} | 전화: {v.get('tel', '')} | "
        f"사업자번호: {v.get('biz_no', '')}",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────
# 3. RSC 청킹 (law_chunking.py 와 동일)
# ──────────────────────────────────────────
SEPARATORS = ["\n\n", "\n", ". ", " "]


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))


def recursive_split(text: str, tokenizer, sep_index: int = 0) -> list[str]:
    if count_tokens(text, tokenizer) <= MAX_TOKENS:
        return [text.strip()] if text.strip() else []
    if sep_index >= len(SEPARATORS):
        return [text.strip()]
    sep   = SEPARATORS[sep_index]
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
# 4. chunk_id 생성 (해시 기반 — 멱등성 보장)
# ──────────────────────────────────────────
def make_chunk_id(contract_id: str, chunk_type: str, sub_key: str) -> str:
    raw = f"{contract_id}::{chunk_type}::{sub_key}"
    return str(uuid.UUID(hashlib.md5(raw.encode()).hexdigest()))


# ──────────────────────────────────────────
# 5. 리스크 태깅
# ──────────────────────────────────────────
def detect_risks(전문: dict) -> list[dict]:
    """
    계약서 전문의 수치 조건을 CONTRACT_RISK_RULES 로 체크하여
    위반된 리스크 목록을 반환합니다.

    Returns:
        [{"risk_id": ..., "risk_name": ..., "desc": ...}, ...]
    """
    detected = []
    for rule in CONTRACT_RISK_RULES:
        try:
            if rule["check_fn"](전문):
                detected.append({
                    "risk_id":   rule["risk_id"],
                    "risk_name": rule["risk_name"],
                    "desc":      rule["desc"],
                })
        except Exception:
            pass
    return detected


# ──────────────────────────────────────────
# 6. 계약서 JSON → 청크 리스트 생성
# ──────────────────────────────────────────
def chunk_contract(parsed: dict, tokenizer) -> list[dict]:
    """
    파싱된 계약서 dict를 청크 리스트로 변환합니다.

    청크 유형별 처리:
        전문_요약   : 계약 기본 정보 1개
        전문_대금   : 대금 지급 행별 1개씩
        전문_보증   : 보증/지체상금/하자 조건 묶음 1개
        전문_당사자 : 발주자/공급자 정보 1개
        본문_조항   : 제N조 단위 (RSC 분할 적용)

    각 청크에는 계약서 메타데이터 + 리스크 태그가 부착됩니다.

    Args:
        parsed    : parse_contract() 반환값
        tokenizer : BGE-M3 내장 tokenizer

    Returns:
        list[dict]: 메타데이터가 부착된 청크 리스트
    """
    contract_id   = parsed.get("contract_id", "")
    doc_type      = parsed.get("doc_type", "")
    file_format   = parsed.get("file_format", "")
    전문           = parsed.get("전문", {})
    본문           = parsed.get("본문", [])
    raw_issues    = parsed.get("raw_issues", [])

    # 계약서 전체에 적용되는 리스크 탐지
    detected_risks = detect_risks(전문)
    all_risk_ids   = [r["risk_id"]   for r in detected_risks]
    all_risk_names = [r["risk_name"] for r in detected_risks]

    # 계약서 공통 메타데이터 (모든 청크에 포함)
    base_meta = {
        "source_type"   : "contract",
        "contract_id"   : contract_id,
        "doc_type"      : doc_type,
        "file_format"   : file_format,
        "contract_name" : 전문.get("contract_name", ""),
        "start_date"    : 전문.get("start_date", ""),
        "end_date"      : 전문.get("end_date", ""),
        "due_date"      : 전문.get("due_date", ""),
        "total_amount"  : 전문.get("total_amount"),
        "client_name"   : 전문.get("client", {}).get("name", ""),
        "vendor_name"   : 전문.get("vendor", {}).get("name", ""),
        # 계약서 전체 리스크 요약 (검색 필터용)
        "contract_risk_ids"  : all_risk_ids,
        "contract_risk_names": all_risk_names,
        "has_issues"         : len(detected_risks) > 0,
        "raw_issues"         : raw_issues,
    }

    chunks = []

    def _make_chunk(chunk_type: str, sub_key: str, text: str,
                    risk_ids=None, risk_names=None, extra_meta=None) -> dict:
        """청크 딕셔너리 생성 헬퍼"""
        if not text or not text.strip():
            return None
        c = {
            "chunk_id"   : make_chunk_id(contract_id, chunk_type, sub_key),
            "chunk_type" : chunk_type,
            "text"       : text.strip(),
            "chunk_tokens": count_tokens(text, tokenizer),
            "risk_ids"   : risk_ids or [],
            "risk_names" : risk_names or [],
            "is_risk_ref": bool(risk_ids),
            **base_meta,
        }
        if extra_meta:
            c.update(extra_meta)
        return c

    # ── 전문_요약 ────────────────────────────
    summary_text = _전문_to_summary_text(전문)
    c = _make_chunk("전문_요약", "summary", summary_text,
                    risk_ids=all_risk_ids, risk_names=all_risk_names)
    if c:
        chunks.append(c)

    # ── 전문_대금 ────────────────────────────
    for ptype, text in _전문_to_payment_texts(전문):
        # 선급금 행에 RISK_004(미지급) / RISK_005(초과) 태그
        row_risks = []
        if ptype == "선급금":
            for r in detected_risks:
                if r["risk_id"] in ("RISK_004", "RISK_005"):
                    row_risks.append(r)
        rids   = [r["risk_id"]   for r in row_risks]
        rnames = [r["risk_name"] for r in row_risks]
        c = _make_chunk("전문_대금", f"payment_{ptype}", text,
                        risk_ids=rids, risk_names=rnames,
                        extra_meta={"payment_type": ptype})
        if c:
            chunks.append(c)

    # ── 전문_보증 ────────────────────────────
    guarantee_text = _전문_to_guarantee_text(전문)
    # 보증 관련 리스크 태그 (RISK_001~003, 006~009)
    guarantee_risk_ids = [
        "RISK_001","RISK_002","RISK_003",
        "RISK_006","RISK_007","RISK_008","RISK_009"
    ]
    g_risks = [r for r in detected_risks if r["risk_id"] in guarantee_risk_ids]
    c = _make_chunk("전문_보증", "guarantee", guarantee_text,
                    risk_ids=[r["risk_id"]   for r in g_risks],
                    risk_names=[r["risk_name"] for r in g_risks])
    if c:
        chunks.append(c)

    # ── 전문_당사자 ──────────────────────────
    party_text = _전문_to_party_text(전문)
    c = _make_chunk("전문_당사자", "party", party_text)
    if c:
        chunks.append(c)

    # ── 본문_조항 ────────────────────────────
    for article in 본문:
        article_id = article.get("article_id", "")
        title      = article.get("title", "")
        text       = article.get("text", "").strip()
        if not text:
            continue

        sub_chunks = (
            [text] if count_tokens(text, tokenizer) <= MAX_TOKENS
            else recursive_split(text, tokenizer)
        )

        for idx, chunk_text in enumerate(sub_chunks):
            c = _make_chunk(
                "본문_조항", f"{article_id}_{idx}", chunk_text,
                extra_meta={
                    "article_id"    : article_id,
                    "article_number": article.get("article_number", ""),
                    "article_title" : title,
                    "source_full"   : f"{전문.get('contract_name','')} {article_id}",
                    "sub_index"     : idx,
                }
            )
            if c:
                chunks.append(c)

    return chunks


# ──────────────────────────────────────────
# 7. JSON 로드
# ──────────────────────────────────────────
def load_contract_json(filepath: Path) -> dict:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────
# 8. Qdrant 컬렉션 초기화
# ──────────────────────────────────────────
def ensure_collection(client: QdrantClient, embed_dim: int, reset: bool = False) -> None:
    existing = [c.name for c in client.get_collections().collections]

    if reset and COLLECTION in existing:
        client.delete_collection(collection_name=COLLECTION)
        print(f"🗑️  컬렉션 초기화: {COLLECTION}")
        existing = []

    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )
        print(f"✅ 컬렉션 생성: {COLLECTION} (dim={embed_dim})")
    else:
        print(f"✅ 컬렉션 기존 사용: {COLLECTION}")


# ──────────────────────────────────────────
# 9. 청크 저장 — 배치 임베딩
# ──────────────────────────────────────────
def store_chunks(
    client:     QdrantClient,
    chunks:     list[dict],
    model:      SentenceTransformer,
    batch_size: int = 32,
) -> None:
    texts      = [c["text"] for c in chunks]
    all_points = []

    print(f"  임베딩 생성 중... ({len(texts)}개)")
    for i in range(0, len(texts), batch_size):
        batch_vecs = model.encode(
            texts[i : i + batch_size],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for j, vec in enumerate(batch_vecs):
            chunk   = chunks[i + j]
            payload = {k: v for k, v in chunk.items() if k != "chunk_id"}
            all_points.append(PointStruct(
                id      = chunk["chunk_id"],
                vector  = vec.tolist(),
                payload = payload,
            ))

    for i in range(0, len(all_points), 100):
        client.upsert(collection_name=COLLECTION, points=all_points[i : i + 100])
    print(f"  ✅ {len(all_points)}개 청크 저장")


# ──────────────────────────────────────────
# 10. 메인
# ──────────────────────────────────────────
def main():
    ensure_qdrant_running()

    ap = argparse.ArgumentParser(description="Workit 계약서 KB 구축")
    ap.add_argument(
        "--input",
        default=str(DATA_DIR),
        help=f"파싱 JSON 폴더 경로 (기본값: {DATA_DIR})",
    )
    ap.add_argument(
        "--reset",
        action="store_true",
        help="컬렉션 삭제 후 재구축",
    )
    args = ap.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"❌ 폴더 없음: {input_dir}")
        sys.exit(1)

    json_files = sorted(input_dir.glob("*.json"))
    # labels.json 제외
    json_files = [f for f in json_files if f.name != "labels.json"]

    if not json_files:
        print(f"❌ JSON 파일 없음: {input_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Workit 계약서 KB 구축")
    print("=" * 60)

    print(f"\n📦 모델 로드: {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL)
    tokenizer   = embed_model.tokenizer
    embed_dim   = embed_model.get_sentence_embedding_dimension()
    print(f"  임베딩 차원: {embed_dim}")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(client, embed_dim, reset=args.reset)

    total        = 0
    risk_tagged  = 0
    normal_count = 0
    issue_count  = 0

    print(f"\n📂 입력 폴더: {input_dir}")
    print(f"   파일 수: {len(json_files)}개")

    for filepath in json_files:
        parsed = load_contract_json(filepath)
        cid    = parsed.get("contract_id", filepath.stem)
        cname  = parsed.get("전문", {}).get("contract_name", "")

        print(f"\n{'─' * 50}")
        print(f"📄 {cid}")
        print(f"   {cname}")

        # 토큰 통계 (전문 요약 + 본문 텍스트 기준)
        stat_texts = [_전문_to_summary_text(parsed.get("전문", {}))]
        stat_texts += [a["text"] for a in parsed.get("본문", []) if a.get("text")]
        log_token_stats_from_texts(f"contract:{cid}", stat_texts, tokenizer)

        chunks = chunk_contract(parsed, tokenizer)
        tagged = sum(1 for c in chunks if c["is_risk_ref"])
        is_issue = parsed.get("전문", {}).get("has_issues", False) or \
                   len(parsed.get("raw_issues", [])) > 0

        print(f"  청크 수: {len(chunks)}개  |  리스크 태깅: {tagged}개  |  "
              f"{'⚠️  문제 계약서' if is_issue else '✅ 정상 계약서'}")

        store_chunks(client, chunks, embed_model)
        total       += len(chunks)
        risk_tagged += tagged
        if is_issue:
            issue_count  += 1
        else:
            normal_count += 1

    # 결과 요약
    print(f"\n{'=' * 60}")
    print(f"✅ 완료!")
    print(f"   처리 계약서: {len(json_files)}개 "
          f"(정상 {normal_count}개 / 문제 {issue_count}개)")
    print(f"   총 청크: {total}개")
    print(f"   리스크 태깅된 청크: {risk_tagged}개")
    count = client.count(collection_name=COLLECTION)
    print(f"   Qdrant 저장: {count.count}개")

    # 검색 테스트
    print(f"\n🔍 검색 테스트 1: '선급금 미지급 리스크'")
    query_vec = embed_model.encode("선급금 미지급 리스크", normalize_embeddings=True)
    results   = client.query_points(
        collection_name=COLLECTION,
        query=query_vec.tolist(),
        query_filter=Filter(
            must=[FieldCondition(key="is_risk_ref", match=MatchValue(value=True))]
        ),
        limit=3,
    ).points
    for i, r in enumerate(results, 1):
        rnames = ", ".join(r.payload.get("risk_names", [])) or "—"
        print(f"  [{i}] score={r.score:.4f} | {r.payload.get('contract_name','')[:30]}")
        print(f"       chunk_type: {r.payload.get('chunk_type','')}")
        print(f"       리스크: {rnames}")
        print(f"       {r.payload['text'][:80]}...")

    print(f"\n🔍 검색 테스트 2: '지체상금 한도 초과' (문제 계약서만)")
    query_vec2 = embed_model.encode("지체상금 한도 초과", normalize_embeddings=True)
    results2   = client.query_points(
        collection_name=COLLECTION,
        query=query_vec2.tolist(),
        query_filter=Filter(
            must=[FieldCondition(key="has_issues", match=MatchValue(value=True))]
        ),
        limit=3,
    ).points
    for i, r in enumerate(results2, 1):
        print(f"  [{i}] score={r.score:.4f} | {r.payload.get('contract_name','')[:30]}")
        print(f"       {r.payload['text'][:80]}...")


if __name__ == "__main__":
    main()