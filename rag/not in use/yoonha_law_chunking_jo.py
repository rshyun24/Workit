"""
Workit - 청킹 스크립트 (조 단위)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
    yoonha_law_parser_jo.py 가 생성한 _jo.json 파일들을
    하나로 합쳐 chunks_jo.json 으로 내보낸다.

실행:
    python yoonha_law_chunking_jo.py

입력:
    data/structured/jo/{법령명}_jo.json

출력:
    data/export/chunks_jo.json
        → Hierarchical RAG에서 child hit 후 parent 텍스트 fetch용.
        → chunk_id = {PREFIX}_{조} 형태로 parent_chunk_id와 1:1 매핑.
"""

import json
import time
from datetime import datetime
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRUCTURED_DIR = Path("data/structured/jo")
EXPORT_DIR     = Path("data/export")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def now() -> str:
    return datetime.now().strftime("%H:%M:%S")

def elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}초" if s < 60 else f"{int(s//60)}분 {int(s%60)}초"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 청크 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_chunks() -> list[dict]:
    chunks = []
    t0     = time.time()

    print(f"\n[{now()}] 📂 {STRUCTURED_DIR} 로드 + 청크 생성")

    for law in LAW_FILES:
        filepath = STRUCTURED_DIR / f"{law['filename']}_jo.json"
        if not filepath.exists():
            print(f"  ⚠️  파일 없음 (파서 미실행?): {filepath.name}")
            continue

        t_file = time.time()
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        articles = [a for a in data.get("articles", []) if a.get("text", "").strip()]

        for article in articles:
            chunks.append({
                "chunk_id":       article["chunk_id"],
                "law_name":       law["law_name"],
                "article_id":     article.get("article_id", ""),
                "article_number": article.get("article_number", ""),
                "text":           article["text"].strip(),
                # 조 단위는 parent 없음
                "parent_chunk_id": None,
                # 메타 필드
                "is_ref_article": article.get("is_ref_article", False),
                "is_upper_law":   article.get("is_upper_law", False),
                "hierarchy":      article.get("hierarchy", {}),
            })

        print(f"  ✅ [{now()}] {law['law_name']} — {len(articles)}개 | {elapsed(t_file)}")

    print(f"\n  → 전체 {len(chunks)}개 청크 | {elapsed(t0)}")
    return chunks


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    t_total = time.time()

    print("=" * 60)
    print(f"  조 단위 청킹  [{now()}]")
    print("=" * 60)

    chunks = build_chunks()

    out_path = EXPORT_DIR / "chunks_jo.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n[{now()}] 💾 저장 완료: {out_path}  ({size_mb:.1f} MB)")
    print(f"  → Hierarchical RAG: child hit 후 parent_chunk_id로 이 파일에서 조 전체 텍스트 fetch.")
    print(f"\n총 소요: {elapsed(t_total)}")
    print("=" * 60)


if __name__ == "__main__":
    main()