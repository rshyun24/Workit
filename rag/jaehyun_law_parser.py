"""
Workit - 법령 문서 파싱 스크립트
input : C:/project/Workit/data/law/ 내 doc/docx 파일
output: C:/project/Workit/data/structured/ 내 JSON 파일

사용법:
    pip install python-docx pywin32
    python yoonha_law_parser.py
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
LAW_DIR    = Path("C:/lecture/Workit/data/law")
OUTPUT_DIR = Path("C:/lecture/Workit/data/structured")
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
    "제59조",
    "제75조",
]

UPPER_LAW = [
    "제90조",
    "제75조",
    "제27조",
    "제50조",
    "제59조",
    "제22조",
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
    "지방회계법": {
        "document_type": "지방회계법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
    "공유재산법": {
        "document_type": "공유재산법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
    "지방회계법_시행령": {
    "document_type": "지방회계법 시행령",
    "source": "대통령령",
    "is_ref_article_doc": False,
},
}

# ─────────────────────────────────────────
# 법령약자 매핑 (chunk_id 생성용)
# ─────────────────────────────────────────
DOC_TYPE_TO_PREFIX = {
    "지방계약법":                    "LCA",
    "지방계약법 시행령":              "LCAE",
    "지방계약법 시행규칙":            "LCAR",
    "소프트웨어 진흥법":              "SWPA",
    "지방회계법":                    "LARA",
    "지방자치단체 용역계약 일반조건":   "PYG",
    "공유재산법":                    "PPMA",
    "지방회계법 시행령":              "LARAE",
}

# ─────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────
def read_docx(path: Path) -> list[tuple[str, str]]:
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
        # .doc → 임시 .docx로 변환 후 python-docx로 읽기
        import tempfile, shutil
        from docx.oxml.ns import qn
        from docx.table import Table as DocxTable

        tmp_dir  = Path(tempfile.mkdtemp())
        tmp_docx = tmp_dir / (path.stem + ".docx")
        try:
            word = win32com.client.DispatchEx("Word.Application")
            try:
                word.Visible = False
            except Exception:
                pass
            wdoc = None
            try:
                wdoc = word.Documents.Open(str(path.resolve()))
                wdoc.SaveAs2(str(tmp_docx), FileFormat=16)  # 16 = docx
            finally:
                try:
                    if wdoc is not None:
                        wdoc.Close(False)
                except Exception:
                    pass
                try:
                    word.Quit()
                except Exception:
                    pass

            # 변환된 docx 읽기
            doc = Document(str(tmp_docx))
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
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {suffix}")


def find_meta(filename: str) -> dict | None:
    for key in sorted(FILE_META.keys(), key=len, reverse=True):
        if key in filename:
            return FILE_META[key]
    return None


def make_article_id(article_number: str) -> str:
    return re.sub(r"[\s/·]", "_", article_number).strip("_")


# ─────────────────────────────────────────
# [버그3 수정] chunk_id 생성
# jo_ui(조의N)는 "의N"으로 표기 — 항 번호와 구분
# 예: LCAE_64_의2 (64조의2) vs LCAE_64_2 (64조 제2항)
# ─────────────────────────────────────────
def make_chunk_id(prefix: str, jo: int, hang: int | None = None, ho: int | None = None, jo_ui: int | None = None) -> str:
    # 조의N은 조 번호에 직접 붙임: 43의10 → 언더스코어 없이 항 번호와 구분
    jo_part = str(jo) + (f"_의{jo_ui}" if jo_ui is not None else "")
    parts = [jo_part]
    if hang is not None:
        parts.append(str(hang))
    if ho is not None:
        parts.append(str(ho))
    return f"{prefix}_{'_'.join(parts)}"


# ─────────────────────────────────────────
# [버그1 수정] 주석 제거 헬퍼
# 호 분리 정규식 적용 전에 <개정 ...>, [전문개정 ...] 제거
# ─────────────────────────────────────────
def _strip_comments(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[[^\]]+\]', '', text)
    return text


# ─────────────────────────────────────────
# 파서 1: 용역계약 일반조건 (절/항/호 구조)
# [버그2 수정] seen_section_nums 증가 로직 + seen_ids 후처리 제거
#             → 문서에 적힌 절 번호를 그대로 사용
# ─────────────────────────────────────────
def parse_yongye(lines: list[tuple[str, str]], prefix: str = "PYG") -> list[dict]:
    articles = []
    cur_chapter = None   # 장 (없으면 None)
    cur_section = None
    cur_clause  = None
    cur_item    = None
    buf = []

    chapter_pat = re.compile(r"^제\s*(\d+)\s*장")
    section_pat = re.compile(r"^제\s*(\d+)\s*절")
    clause_pat  = re.compile(r"^\s*(\d+)\s*\.")
    item_pat    = re.compile(r"^\s*([가나다라마바사아자차카타파하])\s*\.")

    def flush():
        if cur_section and cur_clause and cur_item and buf:
            if cur_chapter:
                an = f"제{cur_chapter}장 제{cur_section}절 제{cur_clause}항 {cur_item}"
                chunk_id = f"{prefix}_{cur_chapter}_{cur_section}_{cur_clause}"
            else:
                an = f"제{cur_section}절 제{cur_clause}항 {cur_item}"
                chunk_id = f"{prefix}_{cur_section}_{cur_clause}"

            hierarchy = {
                "절": f"제{cur_section}절",
                "항": f"제{cur_clause}항",
                "호": cur_item,
            }
            if cur_chapter:
                hierarchy["장"] = f"제{cur_chapter}장"

            articles.append({
                "chunk_id":       chunk_id,
                "article_id":     make_article_id(an),
                "article_number": an,
                "text":           " ".join(buf),
                "hierarchy":      hierarchy,
            })

    for typ, text in lines:
        chm = chapter_pat.match(text)                        # 장은 p/tbl 모두 허용
        sm  = section_pat.match(text) if not chm else None  # 절도 p/tbl 모두 허용, 장이면 제외
        cm  = clause_pat.match(text)  if typ == 'p' and not chm and not sm else None
        im  = item_pat.match(text)    if typ == 'p' and not chm and not sm else None

        if chm:
            flush(); buf = []
            cur_chapter = chm.group(1)
            cur_section = None; cur_clause = None; cur_item = None
        elif sm:
            flush(); buf = []
            cur_section = sm.group(1)
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

    return articles


# ─────────────────────────────────────────
# 파서 1-B: PYG 전용 — 9장 포함 구조 처리
# 9장 이전: 기존 tbl(절)/p(항·호) 구조
# 9장 이후: 모두 p 타입 — 장/절/항/호 전부 p로 들어옴
# ─────────────────────────────────────────
def parse_pyg(lines: list[tuple[str, str]], prefix: str = "PYG") -> list[dict]:
    articles = []
    cur_chapter = None
    cur_section = None
    cur_clause  = None
    cur_item    = None
    buf: list[str] = []

    chapter_pat = re.compile(r"^제\s*(\d+)\s*장")
    section_pat = re.compile(r"^제\s*(\d+)\s*절")
    clause_pat  = re.compile(r"^\s*(\d+)\s*\.")
    item_pat    = re.compile(r"^\s*([가나다라마바사아자차카타파하])\s*\.")

    def flush():
        nonlocal buf, cur_item
        if not buf or not cur_section or not cur_clause:
            buf = []; cur_item = None
            return
        id_parts = ([cur_chapter] if cur_chapter else []) + [cur_section, cur_clause]
        prefix_str = (f"제{cur_chapter}장 " if cur_chapter else "")

        if cur_item:
            an = prefix_str + f"제{cur_section}절 제{cur_clause}항 {cur_item}"
            hierarchy = {"절": f"제{cur_section}절", "항": f"제{cur_clause}항", "호": cur_item}
        else:
            an = prefix_str + f"제{cur_section}절 제{cur_clause}항"
            hierarchy = {"절": f"제{cur_section}절", "항": f"제{cur_clause}항"}

        if cur_chapter:
            hierarchy["장"] = f"제{cur_chapter}장"

        # 호가 있으면 chunk_id 마지막에 붙여 구분
        cid_parts = id_parts + ([cur_item] if cur_item else [])
        articles.append({
            "chunk_id":       f"{prefix}_{'_'.join(cid_parts)}",
            "article_id":     make_article_id(an),
            "article_number": an,
            "text":           " ".join(buf),
            "hierarchy":      hierarchy,
        })
        buf = []; cur_item = None

    for typ, text in lines:
        chm = chapter_pat.match(text)
        # 절: 9장 이전은 tbl만, 9장 이후(cur_chapter 있음)는 p도 허용
        sm  = section_pat.match(text) if not chm and (typ == 'tbl' or cur_chapter) else None
        cm  = clause_pat.match(text)  if typ == 'p' and not chm and not sm else None
        im  = item_pat.match(text)    if typ == 'p' and not chm and not sm and not cm else None

        if chm:
            flush()
            cur_chapter = chm.group(1)
            cur_section = None; cur_clause = None; cur_item = None
        elif sm:
            flush()
            cur_section = sm.group(1)
            cur_clause = None; cur_item = None
        elif cm and cur_section:
            flush()
            cur_clause = cm.group(1)
            buf = [text]          # clause 줄 자체도 내용에 포함
        elif im and cur_clause:
            flush()
            buf = [text]
            cur_item = im.group(1)
        elif cur_clause:          # 항 내 본문 누적 (호 유무 관계없이)
            buf.append(text)

    flush()
    return articles


# ─────────────────────────────────────────
# 파서 2: 일반 법령 — 항/호 단위 분리
# ─────────────────────────────────────────
def parse_law(lines: list[tuple[str, str]], prefix: str) -> list[dict]:
    article_pat = re.compile(r"^(제\s*\d+\s*조(?:의\s*\d+)?)\s*[(\[〔]?([^)\]\)〕\n]*)[)\]\)〕]?")

    raw_articles: list[dict] = []
    cur_jo    = None
    cur_jo_ui = None
    cur_title = ""
    buf: list[str] = []

    def flush_jo():
        if cur_jo is not None and buf:
            raw_articles.append({
                "jo":    cur_jo,
                "jo_ui": cur_jo_ui,
                "title": cur_title,
                "text":  " ".join(buf),
            })

    bujik_pat = re.compile(r"^부\s*칙")
    in_bujik = False

    for _, text in lines:
        if bujik_pat.match(text):
            in_bujik = True
            flush_jo()
            cur_jo = None; buf = []
            continue
        if in_bujik:
            continue

        m = article_pat.match(text)
        if m:
            flush_jo()
            buf = [text]
            raw_jo_str = re.sub(r"\s+", "", m.group(1))
            jo_m = re.match(r"제(\d+)조(?:의(\d+))?", raw_jo_str)
            cur_jo    = int(jo_m.group(1)) if jo_m else None
            cur_jo_ui = int(jo_m.group(2)) if jo_m and jo_m.group(2) else None
            cur_title = m.group(2).strip() if m.group(2) else ""
        else:
            buf.append(text)

    flush_jo()

    HANG_MAP = {c: i+1 for i, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮")}

    articles: list[dict] = []

    for raw in raw_articles:
        jo    = raw["jo"]
        jo_ui = raw["jo_ui"]
        text  = raw["text"]

        hang_splits = re.split(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])", text)

        if len(hang_splits) <= 1:
            chunk_id = make_chunk_id(prefix, jo, jo_ui=jo_ui)
            articles.append({
                "chunk_id":       chunk_id,
                "article_id":     f"제{jo}조" + (f"의{jo_ui}" if jo_ui else ""),
                "article_number": f"제{jo}조" + (f"의{jo_ui}" if jo_ui else ""),
                "title":          raw["title"],
                "text":           text,
                "hierarchy":      {"조": f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "")},
            })
            continue

        i = 1
        while i < len(hang_splits) - 1:
            hang_char = hang_splits[i]
            hang_text = hang_splits[i + 1].strip() if i + 1 < len(hang_splits) else ""
            hang_num  = HANG_MAP.get(hang_char, i)

            # [버그1 수정] 호 분리 전에 <개정 ...>, [전문개정 ...] 주석 제거
            hang_text_clean = _strip_comments(hang_text)

            ho_splits = re.split(r"\s(\d{1,2})\.\s", hang_text_clean)

            if len(ho_splits) <= 1:
                chunk_id = make_chunk_id(prefix, jo, hang=hang_num, jo_ui=jo_ui)
                articles.append({
                    "chunk_id":       chunk_id,
                    "article_id":     f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "") + f"제{hang_num}항",
                    "article_number": f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "") + f"제{hang_num}항",
                    "title":          raw["title"],
                    "text":           hang_char + hang_text,   # 원본 텍스트(주석 포함) 저장
                    "hierarchy": {
                        "조": f"제{jo}조" + (f"의{jo_ui}" if jo_ui else ""),
                        "항": f"제{hang_num}항",
                    },
                })
            else:
                j = 1
                while j < len(ho_splits) - 1:
                    ho_num  = int(ho_splits[j])
                    ho_text = ho_splits[j + 1].strip() if j + 1 < len(ho_splits) else ""
                    chunk_id = make_chunk_id(prefix, jo, hang=hang_num, ho=ho_num, jo_ui=jo_ui)
                    articles.append({
                        "chunk_id":       chunk_id,
                        "article_id":     f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "") + f"제{hang_num}항제{ho_num}호",
                        "article_number": f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "") + f"제{hang_num}항제{ho_num}호",
                        "title":          raw["title"],
                        "text":           f"{hang_char} {ho_splits[0].strip()} {ho_num}. {ho_text}",
                        "hierarchy": {
                            "조": f"제{jo}조" + (f"의{jo_ui}" if jo_ui else ""),
                            "항": f"제{hang_num}항",
                            "호": f"제{ho_num}호",
                        },
                    })
                    j += 2

            i += 2

    return articles


# ─────────────────────────────────────────
# 필터링
# ─────────────────────────────────────────
def tag_article(article: dict, is_ref_doc: bool) -> dict:
    an = article.get("article_number", "")
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
    prefix     = DOC_TYPE_TO_PREFIX.get(meta["document_type"], "UNK")

    if prefix == "PYG":
        # 디버그: 장 패턴 주변 텍스트 출력
        chap_pat = re.compile(r"제\s*\d+\s*장")
        for i, (typ, text) in enumerate(paragraphs):
            if chap_pat.search(text):
                print(f"  [DEBUG] 장 감지 typ={typ!r} text={text!r}")
        articles = parse_pyg(paragraphs, prefix=prefix)
    elif meta["is_ref_article_doc"]:
        articles = parse_yongye(paragraphs, prefix=prefix)
    else:
        articles = parse_law(paragraphs, prefix=prefix)

    articles = [tag_article(a, meta["is_ref_article_doc"]) for a in articles]

    result = {
        "document_type":      meta["document_type"],
        "source":             meta["source"],
        "filename":           path.name,
        "total_articles":     len(articles),
        "ref_article_count":  sum(1 for a in articles if a.get("is_ref_article")),
        "upper_law_count":    sum(1 for a in articles if a.get("is_upper_law")),
        "articles":           articles,
    }

    out_path = OUTPUT_DIR / f"{filename}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  → 저장: {out_path}")
    print(f"  → 전체 조문: {len(articles)}개 | REF: {result['ref_article_count']}개 | UPPER: {result['upper_law_count']}개")

    chunk_ids = {a["chunk_id"] for a in articles}
    print(f"  → 생성된 chunk_id 샘플: {list(chunk_ids)[:5]}")


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
