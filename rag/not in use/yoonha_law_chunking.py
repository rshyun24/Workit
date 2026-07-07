"""
═══════════════════════════════════════════════════════════════════
STEP 1 — PyCharm에서 실행
역할: 법령 JSON (structured/) → 청크 생성 → chunks_jo.json / chunks_ho.json 저장
═══════════════════════════════════════════════════════════════════

실행:
    python rag/yoonha_law_chunking.py

출력:
    data/export/chunks_jo.json   ← 조 단위 청크 (Hierarchical RAG fetch용)
    data/export/chunks_ho.json   ← 호 단위 청크 (임베딩 & 벡터 검색 대상)

    두 파일 모두 Google Colab에 업로드할 파일.

지원 법령 (2026-06 기준):
    LCA    지방계약법
    LCAE   지방계약법 시행령
    LCAR   지방계약법 시행규칙
    SWPA   소프트웨어 진흥법
    SWPAE  소프트웨어 진흥법 시행령
    LARA   지방회계법
    LARAE  지방회계법 시행령
    PYG    지방자치단체 용역계약 일반조건 (예규367호)
    PPMA   공유재산법
    PPMAE  공유재산법 시행령
    PIPA   개인정보보호법
    PIPAE  개인정보보호법 시행령

chunk_id는 yoonha_law_parser.py에서 생성한 값을 그대로 사용.
청킹 스크립트에서 chunk_id를 재생성하지 않음 —
생성 로직은 파서 한 곳에서만 관리.
"""

import json
import time
from datetime import datetime
from pathlib import Path


# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
# 파서가 jo/ho 폴더로 분리해서 저장한 결과물을 각각 읽음
STRUCTURED_DIR_JO  = Path("data/structured/jo")   # 조 단위 JSON 위치
STRUCTURED_DIR_HO  = Path("data/structured/ho")   # 호 단위 JSON 위치
EXPORT_DIR         = Path("data/export")           # chunks_jo.json / chunks_ho.json 저장 위치

# 법령 파일 목록
# filename: 파서가 생성한 파일명 (_jo.json / _ho.json suffix 제외한 stem)
# law_name: 법령명 (청크 메타데이터로 저장)
# prefix  : 법령 약어 (참고용)
LAW_FILES = [
    {"filename": "지방계약법",                                  "law_name": "지방계약법",                    "prefix": "LCA"},
    {"filename": "지방계약법_시행령",                           "law_name": "지방계약법 시행령",              "prefix": "LCAE"},
    {"filename": "지방계약법_시행규칙",                         "law_name": "지방계약법 시행규칙",            "prefix": "LCAR"},
    {"filename": "소프트웨어_진흥법",                           "law_name": "소프트웨어 진흥법",              "prefix": "SWPA"},
    {"filename": "소프트웨어 진흥법 시행령",                    "law_name": "소프트웨어 진흥법 시행령",       "prefix": "SWPAE"},
    {"filename": "지방회계법",                                  "law_name": "지방회계법",                    "prefix": "LARA"},
    {"filename": "지방회계법_시행령",                           "law_name": "지방회계법 시행령",              "prefix": "LARAE"},
    {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규)", "law_name": "지방자치단체 용역계약 일반조건", "prefix": "PYG"},
    {"filename": "공유재산법",                                  "law_name": "공유재산법",                    "prefix": "PPMA"},
    {"filename": "공유재산 및 물품 관리법 시행령",              "law_name": "공유재산법 시행령",              "prefix": "PPMAE"},
    {"filename": "개인정보 보호법",                             "law_name": "개인정보보호법",                 "prefix": "PIPA"},
    {"filename": "개인정보 보호법 시행령",                      "law_name": "개인정보보호법 시행령",          "prefix": "PIPAE"},
]


# ──────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────

def now() -> str:
    return datetime.now().strftime("%H:%M:%S")

def elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}초" if s < 60 else f"{int(s//60)}분 {int(s%60)}초"


# ──────────────────────────────────────────
# 법령 JSON 로드
# ──────────────────────────────────────────

def load_law_json(filepath: Path) -> list[dict]:
    """
    yoonha_law_parser.py 가 생성한 JSON 파일을 읽어
    텍스트가 있는 articles 리스트만 반환.
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return [a for a in data.get("articles", []) if a.get("text", "").strip()]


# ──────────────────────────────────────────
# 청크 생성
# ──────────────────────────────────────────

def build_chunks(structured_dir: Path, suffix: str) -> list[dict]:
    """
    LAW_FILES에 정의된 모든 법령을 순회하며 청크 리스트 생성.

    인자:
        structured_dir — 읽을 JSON 폴더 (jo/ 또는 ho/)
        suffix         — 파일명 suffix (_jo 또는 _ho)

    각 청크 구조:
        chunk_id      : 파서가 생성한 고유 식별자 (예: LCA_30_4)
        law_name      : 법령명 (예: 지방계약법)
        article_id    : 조문 ID (예: 제30조제4항)
        article_number: 조문 번호 전문
        text          : 조문 본문
        is_parent     : 조 단위 parent 청크 여부 (Hierarchical RAG)
        parent_id     : 소속 parent chunk_id (단항/parent이면 None)
        is_ref_article: 용역계약 일반조건 핵심 조항 여부
        is_upper_law  : PYG가 직접 인용하는 상위법 조문 여부
        hierarchy     : 조/항/호 계층 정보
    """
    chunks = []
    t0 = time.time()

    print(f"\n[{now()}] 📂 {structured_dir} 로드 + 청크 생성")

    for law in LAW_FILES:
        filepath = structured_dir / f"{law['filename']}{suffix}.json"
        if not filepath.exists():
            print(f"  ⚠️  파일 없음 (파서 미실행?): {filepath.name}")
            continue

        t_file   = time.time()
        articles = load_law_json(filepath)

        for article in articles:
            text = article.get("text", "").strip()
            if not text:
                continue

            chunks.append({
                "chunk_id":       article["chunk_id"],
                "law_name":       law["law_name"],
                "article_id":     article.get("article_id", ""),
                "article_number": article.get("article_number", ""),
                "text":           text,
                "is_parent":      article.get("is_parent", False),
                "parent_id":      article.get("parent_id"),
                "is_ref_article": article.get("is_ref_article", False),
                "is_upper_law":   article.get("is_upper_law", False),
                "hierarchy":      article.get("hierarchy", {}),
            })

        print(f"  ✅ [{now()}] {law['law_name']} — {len(articles)}개 | {elapsed(t_file)}")

    print(f"\n  → 전체 {len(chunks)}개 청크 | {elapsed(t0)}")
    return chunks


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────

def main():
    t_total = time.time()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  STEP 1 — 청크 생성 & 내보내기  [{now()}]")
    print("=" * 60)

    # 조 단위 청크 (Hierarchical RAG fetch용)
    chunks_jo = build_chunks(STRUCTURED_DIR_JO, suffix="_jo")
    out_jo = EXPORT_DIR / "chunks_jo.json"
    with open(out_jo, "w", encoding="utf-8") as f:
        json.dump(chunks_jo, f, ensure_ascii=False, indent=2)
    print(f"\n[{now()}] 💾 저장 완료: {out_jo}  ({out_jo.stat().st_size / 1024 / 1024:.1f} MB)")

    # 호 단위 청크 (임베딩 & 벡터 검색 대상)
    chunks_ho = build_chunks(STRUCTURED_DIR_HO, suffix="_ho")
    out_ho = EXPORT_DIR / "chunks_ho.json"
    with open(out_ho, "w", encoding="utf-8") as f:
        json.dump(chunks_ho, f, ensure_ascii=False, indent=2)
    print(f"\n[{now()}] 💾 저장 완료: {out_ho}  ({out_ho.stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"\n  → 두 파일을 Google Colab에 업로드하세요")
    print(f"\n총 소요: {elapsed(t_total)}")
    print("=" * 60)


if __name__ == "__main__":
    main()