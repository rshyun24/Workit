"""
═══════════════════════════════════════════════════════════════════
STEP 1 — PyCharm에서 실행
역할: 법령 JSON → 청크 생성 → chunks.json 저장
═══════════════════════════════════════════════════════════════════

실행:
    python rag/jaehyun_law_chunking.py

출력:
    data/export/chunks.json   ← Colab에 업로드할 파일
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path


# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
STRUCTURED_DIR = Path("structured")
EXPORT_DIR     = Path("export")

LAW_FILES = [
    {"filename": "지방계약법.json",                                "law_name": "지방계약법",                    "prefix": "LCA"},
    {"filename": "지방계약법_시행령.json",                          "law_name": "지방계약법 시행령",              "prefix": "LCAE"},
    {"filename": "지방계약법_시행규칙.json",                        "law_name": "지방계약법 시행규칙",            "prefix": "LCAR"},
    {"filename": "소프트웨어_진흥법.json",                          "law_name": "소프트웨어 진흥법",              "prefix": "SWPA"},
    {"filename": "지방회계법.json",                                 "law_name": "지방회계법",                    "prefix": "LARA"},
    {"filename": "지방자치단체 용역계약 일반조건 (행안부 예규).json", "law_name": "지방자치단체 용역계약 일반조건",  "prefix": "PYG"},
    {"filename": "공유재산법.json",                                  "law_name": "공유재산법",                    "prefix": "PPMA"},
    {"filename": "지방회계법_시행령.json",                           "law_name": "지방회계법 시행령",              "prefix": "LARAE"},
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
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return [a for a in data.get("articles", []) if a.get("text", "").strip()]


# ──────────────────────────────────────────
# [버그1 수정] 주석 제거 헬퍼
# <개정 2013. 3. 23.> / [전문개정 ...] 같은 패턴이
# article_id에 섞이면 날짜 숫자를 호 번호로 오인함
# ──────────────────────────────────────────
def _strip_comments(s: str) -> str:
    s = re.sub(r'<[^>]+>', '', s)   # <개정 ...>, <신설 ...> 등
    s = re.sub(r'\[[^\]]+\]', '', s)  # [전문개정 ...] 등
    return s.strip()


# ──────────────────────────────────────────
# chunk_id 생성
# ──────────────────────────────────────────
def _make_chunk_id(prefix: str, article_id: str, hierarchy: dict | None = None) -> str:
    # [버그1 수정] 주석 제거 후 정규식 적용
    article_id = _strip_comments(article_id)

    def extract_num(s: str) -> str:
        """제N조/제N절 → N, 제N조의M → N_의M"""
        m = re.search(r"제\s*(\d+)(?:[조절]의(\d+))?", s)
        if m:
            return f"{m.group(1)}_의{m.group(2)}" if m.group(2) else m.group(1)
        return s

    if hierarchy:
        if prefix == "PYG":
            parts = []
            if "장" in hierarchy:
                pyg_chapter = re.search(r"제\s*(\d+)\s*장", article_id)
                parts.append(pyg_chapter.group(1) if pyg_chapter else extract_num(hierarchy["장"]))
            if "절" in hierarchy:
                pyg_section = re.search(r"제\s*(\d+)\s*절", article_id)
                parts.append(pyg_section.group(1) if pyg_section else extract_num(hierarchy["절"]))
            if "항" in hierarchy:
                parts.append(extract_num(hierarchy["항"]))
            if "호" in hierarchy and "장" in hierarchy:
                # 9장 구조는 호까지 chunk_id에 포함해 구분
                parts.append(hierarchy["호"])
            return f"{prefix}_{'_'.join(parts)}" if parts else f"{prefix}_{article_id}"
        else:
            parts = []
            for key in ("조", "항", "호"):
                if key in hierarchy:
                    val = hierarchy[key]
                    if key == "호":
                        num = re.search(r"제\s*(\d+)", val)
                        parts.append(num.group(1) if num else val.strip())
                    else:
                        parts.append(extract_num(val))
            return f"{prefix}_{'_'.join(parts)}" if parts else f"{prefix}_{article_id}"

    if prefix == "PYG":
        # [버그2 수정] 절 번호를 article_id 텍스트에서 직접 추출
        section = re.search(r"제\s*(\d+)\s*절", article_id)
        clause  = re.search(r"제\s*(\d+)\s*항", article_id)
        if section:
            parts = [section.group(1)]
            if clause:
                parts.append(clause.group(1))
            return f"{prefix}_{'_'.join(parts)}"
        # fallback: 제N 숫자 두 개
        nums = re.findall(r"제(\d+)", article_id)
        return f"{prefix}_{'_'.join(nums[:2])}" if nums else f"{prefix}_{article_id}"
    else:
        # [버그3 수정] 조의N → _의N suffix로 항 번호와 구분
        m = re.search(r"제(\d+)조(?:의(\d+))?", article_id)
        if m:
            base   = m.group(1) + (f"_의{m.group(2)}" if m.group(2) else "")
            sub    = re.findall(r"제(\d+)[항호]", article_id)
            sub_str = f"_{'_'.join(sub)}" if sub else ""
            return f"{prefix}_{base}{sub_str}"

        # fallback
        nums = re.findall(r"\d+", article_id)
        return f"{prefix}_{'_'.join(nums)}" if nums else f"{prefix}_{article_id}"


# ──────────────────────────────────────────
# 청크 생성
# ──────────────────────────────────────────
def build_chunks() -> list[dict]:
    chunks = []
    t0 = time.time()

    print(f"\n[{now()}] 📂 법령 JSON 로드 + 청크 생성")
    for law in LAW_FILES:
        filepath = STRUCTURED_DIR / law["filename"]
        if not filepath.exists():
            print(f"  ⚠️  파일 없음: {law['filename']}")
            continue

        t_file = time.time()
        articles = load_law_json(filepath)

        for article in articles:
            article_id = article.get("article_id", "")
            text       = article.get("text", "").strip()
            if not text:
                continue

            hierarchy = article.get("hierarchy")
            chunk_id  = _make_chunk_id(law["prefix"], article_id, hierarchy)

            chunks.append({
                "chunk_id":   chunk_id,
                "law_name":   law["law_name"],
                "article_id": article_id,
                "text":       text,
            })

        print(f"  ✅ [{now()}] {law['law_name']} — {len(articles)}개 조문 | {elapsed(t_file)}")

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

    chunks = build_chunks()

    out_path = EXPORT_DIR / "chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n[{now()}] 💾 저장 완료: {out_path}  ({size_mb:.1f} MB)")
    print(f"  → 이 파일을 Google Colab에 업로드하세요")
    print(f"\n총 소요: {elapsed(t_total)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
