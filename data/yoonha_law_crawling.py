"""
Workit - 법령 문서 파싱 스크립트
input : C:/project/Workit/data/law/ 내 doc/docx 파일
output: C:/project/Workit/data/structured/ 내 JSON 파일

사용법:
    pip install python-docx pywin32
    python parse_law_to_json.py
"""

import os
import re
import json
from pathlib import Path
from docx import Document
import win32com.client

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
LAW_DIR = Path("C:/project/Workit/data/law")
OUTPUT_DIR = Path("C:/project/Workit/data/structured")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# REF_ARTICLE & UPPER_LAW (필터링 기준)
# ─────────────────────────────────────────
REF_ARTICLE = [
    "제7절 제1항 가",
    "제8절 제4항 나",
    "제6절 제1항 가",
    "제6절 제1항 라",
    "제6절 제1항 마",
    "제7절 제4항 다",
    "제7절 제5항 가",
    "제8절 제7항 가",
    "제59조",   # 소프트웨어 진흥법
    "제75조",   # 지방계약법 시행규칙
]

UPPER_LAW = [
    "제90조",   # 지방계약법 시행령
    "제75조",   # 지방계약법 시행규칙
    "제27조",   # 지방계약법
    "제50조",   # 소프트웨어 진흥법
    "제59조",   # 소프트웨어 진흥법
    "제22조",   # 지방계약법
]

# ─────────────────────────────────────────
# 파일명 → document_type 매핑
# ─────────────────────────────────────────
FILE_META = {
    "지방자치단체 용역계약 일반조건 (행안부 예규)": {
        "document_type": "지방자치단체 용역계약 일반조건",
        "source": "행정안전부 예규",
        "is_ref_article_doc": True,
    },
    "지방계약법": {
        "document_type": "지방계약법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
    "지방계약법_시행령": {
        "document_type": "지방계약법 시행령",
        "source": "대통령령",
        "is_ref_article_doc": False,
    },
    "지방계약법_시행규칙": {
        "document_type": "지방계약법 시행규칙",
        "source": "행정안전부령",
        "is_ref_article_doc": False,
    },
    "소프트웨어_진흥법": {
        "document_type": "소프트웨어 진흥법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
}

# ─────────────────────────────────────────
# 조문 번호 추출 패턴
# ─────────────────────────────────────────
# 용역계약 일반조건: 제N절, 제N항, 가/나/다...
YONG_ARTICLE_PATTERN = re.compile(
    r"(제\d+절)\s*(제\d+항)?\s*([가-힣](?=\.)|\d+(?=\.))?",
    re.UNICODE
)

# 일반 법령: 제N조, 제N항, 제N호
LAW_ARTICLE_PATTERN = re.compile(
    r"제\s*\d+\s*조(\s*의\s*\d+)?",
    re.UNICODE
)

# ─────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────
def read_docx(path: Path) -> list[tuple[str, str]]:
    """
    doc/docx에서 (타입, 텍스트) 튜플 리스트 반환
    타입: 'p' (단락) | 'tbl' (표 셀)
    용역계약 일반조건처럼 절 헤더가 표 안에 있는 문서를 위해 구분
    """
    suffix = path.suffix.lower()

    if suffix == ".docx":
        from docx.oxml.ns import qn
        from docx.table import Table as DocxTable
        doc = Document(str(path))
        lines = []
        for block in doc.element.body:
            tag = block.tag.split('}')[-1]
            if tag == 'p':
                text = ''.join(r.text for r in block.iter(qn('w:t'))).strip()
                if text:
                    lines.append(('p', text))
            elif tag == 'tbl':
                tbl = DocxTable(block, doc)
                for row in tbl.rows:
                    for cell in row.cells:
                        t = cell.text.strip()
                        if t:
                            lines.append(('tbl', t))
        return lines

    elif suffix == ".doc":
        # win32com으로 .doc 읽기 (Word 필요) — 표 구분 없이 단락만
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        try:
            doc = word.Documents.Open(str(path.resolve()))
            lines = []
            for i in range(1, doc.Paragraphs.Count + 1):
                text = doc.Paragraphs(i).Range.Text.strip()
                if text:
                    lines.append(('p', text))
            doc.Close(False)
            return lines
        finally:
            word.Quit()

    else:
        raise ValueError(f"지원하지 않는 파일 형식: {suffix}")


def find_meta(filename: str) -> dict | None:
    """파일명에서 메타 정보 찾기"""
    for key, meta in FILE_META.items():
        if key in filename:
            return meta
    return None


def make_article_id(article_number: str) -> str:
    """article_number → article_id (공백/특수문자 → 언더스코어)"""
    return re.sub(r"[\s/·]", "_", article_number).strip("_")


# ─────────────────────────────────────────
# 파서 1: 용역계약 일반조건 (절/항/호 구조)
# ─────────────────────────────────────────
def parse_yongye(lines: list[tuple[str, str]]) -> list[dict]:
    """
    용역계약 일반조건 파싱
    - 절 헤더: 표(tbl) 셀 안에 "제N절 제목" 형태
    - 항: 단락(p) "1.", "2." 숫자 목록
    - 호: 단락(p) "가.", "나." 한글 목록
    - 주의: 이 문서는 제7절 내부에 제8절 내용이 섹션 구분 없이 이어지므로
            동일한 항 번호가 재등장하면 절 번호를 하나 올려서 처리
    """
    articles = []
    cur_section = None
    cur_clause  = None
    cur_item    = None
    buf = []
    seen_section_nums = []   # 절 번호 등장 순서 (중복 감지용)

    section_pat = re.compile(r"^제\s*(\d+)\s*절")
    clause_pat  = re.compile(r"^\s*(\d+)\s*\.")
    item_pat    = re.compile(r"^\s*([가나다라마바사아자차카타파하])\s*\.")

    def flush():
        if cur_section and cur_clause and cur_item and buf:
            an = f"제{cur_section}절 제{cur_clause}항 {cur_item}"
            articles.append({
                "article_id": make_article_id(an),
                "article_number": an,
                "text": " ".join(buf),
                "hierarchy": {
                    "절": f"제{cur_section}절",
                    "항": f"제{cur_clause}항",
                    "호": cur_item,
                }
            })

    for typ, text in lines:
        sm = section_pat.match(text) if typ == 'tbl' else None
        cm = clause_pat.match(text) if typ == 'p' else None
        im = item_pat.match(text) if typ == 'p' else None

        if sm:
            flush(); buf = []
            raw_num = sm.group(1)
            # 동일 절 번호 재등장 → 다음 절 번호로 올림 (7절 → 8절)
            count = seen_section_nums.count(raw_num)
            cur_section = str(int(raw_num) + count)
            seen_section_nums.append(raw_num)
            cur_clause = None; cur_item = None
        elif cm and cur_section:
            flush(); buf = []
            cur_clause = cm.group(1)
            cur_item = None
        elif im and cur_clause:
            flush()
            buf = [text]
            cur_item = im.group(1)
        elif cur_item:
            buf.append(text)

    flush()

    # 같은 절 안에서 항+호 조합이 중복될 경우 절 번호 +1 처리
    seen_ids: dict[str, bool] = {}
    for a in articles:
        an = a["article_number"]
        if an in seen_ids:
            old_sec = int(re.search(r"제(\d+)절", an).group(1))
            new_an = an.replace(f"제{old_sec}절", f"제{old_sec + 1}절")
            a["article_number"] = new_an
            a["article_id"] = make_article_id(new_an)
            a["hierarchy"]["절"] = f"제{old_sec + 1}절"
        else:
            seen_ids[an] = True

    return articles


# ─────────────────────────────────────────
# 파서 2: 일반 법령 (제N조 구조)
# ─────────────────────────────────────────
def parse_law(lines: list[tuple[str, str]]) -> list[dict]:
    """
    일반 법령 파싱
    - 제N조(제목) 구조
    """
    articles = []
    current_article = None
    current_title = None
    buffer = []

    article_pat = re.compile(r"^(제\s*\d+\s*조(?:의\s*\d+)?)\s*[(\[〔]?([^)\]\)〕]*)[)\]\)〕]?")

    def flush():
        if current_article and buffer:
            article_number = current_article
            articles.append({
                "article_id": make_article_id(article_number),
                "article_number": article_number,
                "title": current_title or "",
                "text": " ".join(buffer),
                "hierarchy": {
                    "조": current_article,
                }
            })

    for typ, text in lines:
        m = article_pat.match(text)
        if m:
            flush()
            buffer = [text]
            current_article = re.sub(r"\s+", "", m.group(1))
            current_title = m.group(2).strip() if m.group(2) else ""
        else:
            buffer.append(text)

    flush()
    return articles


# ─────────────────────────────────────────
# 필터링: REF_ARTICLE / UPPER_LAW 해당 여부
# ─────────────────────────────────────────
def tag_article(article: dict, is_ref_doc: bool) -> dict:
    an = article["article_number"]
    article["is_ref_article"] = is_ref_doc and any(ref in an for ref in REF_ARTICLE)
    article["is_upper_law"]   = any(ref in an for ref in UPPER_LAW)
    return article


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
def process_file(path: Path):
    filename = path.stem
    meta = find_meta(filename)
    if meta is None:
        print(f"[SKIP] 메타 없음: {filename}")
        return

    print(f"[PARSE] {filename}")
    paragraphs = read_docx(path)

    if meta["is_ref_article_doc"]:
        articles = parse_yongye(paragraphs)
    else:
        articles = parse_law(paragraphs)

    # 태깅
    articles = [tag_article(a, meta["is_ref_article_doc"]) for a in articles]

    result = {
        "document_type": meta["document_type"],
        "source": meta["source"],
        "filename": path.name,
        "total_articles": len(articles),
        "ref_article_count": sum(1 for a in articles if a["is_ref_article"]),
        "upper_law_count": sum(1 for a in articles if a["is_upper_law"]),
        "articles": articles,
    }

    out_path = OUTPUT_DIR / f"{filename}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  → 저장: {out_path}")
    print(f"  → 전체 조문: {len(articles)}개 | REF: {result['ref_article_count']}개 | UPPER: {result['upper_law_count']}개")


def main():
    files = list(LAW_DIR.glob("*.doc")) + list(LAW_DIR.glob("*.docx"))
    if not files:
        print(f"[ERROR] {LAW_DIR} 에 파일이 없습니다.")
        return

    for f in sorted(files):
        process_file(f)

    print("\n✅ 완료! 결과물:", OUTPUT_DIR)


if __name__ == "__main__":
    main()