"""
Workit - 법령 문서 파싱 + cross_refs 추출 스크립트 (단일 파일, 독립 실행)
input : C:/lecture/Workit/data/law/ 내 docx 파일
output: C:/lecture/Workit/data/structured/ 내 JSON 파일

처리 단계:
  1. docx 파싱 → 조/항/호/목(또는 PYG 절/항/호) chunk 분리
  2. 항 없는 조의 직접 호 분리, 호의N(예: 7의2.) 처리
  3. 같은 (조,조의N) 중복(원본 복붙/시행일 예고 병기) dedupe
  4. 2패스로 chunk_id 레지스트리 구축 후 cross_refs 추출

cross_refs 처리 패턴:
  - 단일형:  「개인정보 보호법 시행령」 제19조제1호          → PIPAE_19_1
  - 열거형:  제1호 또는 제2호 / 제1항·제2항 / 제1항 및 제2항 → 전부 추출
  - 범위형:  제1항부터 제7항까지                            → 1~7 전부 추출
  - 동일법:  이 법(영/규칙) 제N조...                        → 현재 법령 prefix 사용

사용법:
    pip install python-docx
    python jaehyun_psref.py
"""

import re
import json
from pathlib import Path
from docx import Document

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
# LAW_DIR    = Path("C:/lecture/Workit/data/law")
# OUTPUT_DIR = Path("C:/lecture/Workit/data/structured")

LAW_DIR    = Path("C:/project/Workit/data/law")
OUTPUT_DIR = Path("C:/project/Workit/data/structured")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# REF_ARTICLE & UPPER_LAW
# ─────────────────────────────────────────
REF_ARTICLE = [
    "제7절 제1항 가", "제8절 제4항 나", "제6절 제1항 가",
    "제6절 제1항 라", "제6절 제1항 마", "제7절 제4항 다",
    "제7절 제5항 가", "제8절 제7항 가", "제59조", "제75조",
]

UPPER_LAW = ["제90조", "제75조", "제27조", "제50조", "제59조", "제22조"]

# ─────────────────────────────────────────
# 파일명(stem) → 메타 매핑
# ─────────────────────────────────────────
FILE_META = {
    # 파일명이 "지방자치단체 용역계약.docx"(일반조건 생략)으로 바뀐 적이 있어
    # 짧은 쪽도 키로 등록해 find_meta()가 substring 매칭에 실패하지 않도록 함
    "지방자치단체 용역계약": {
        "document_type": "지방자치단체 용역계약 일반조건",
        "law_name":      "지방자치단체 용역계약 일반조건",
        "source":        "행정안전부 예규",
        "is_ref_article_doc": True,
    },
    "지방계약법_시행규칙": {
        "document_type": "지방계약법 시행규칙",
        "law_name":      "지방계약법 시행규칙",
        "source":        "행정안전부령",
        "is_ref_article_doc": False,
    },
    "지방계약법_시행령": {
        "document_type": "지방계약법 시행령",
        "law_name":      "지방계약법 시행령",
        "source":        "대통령령",
        "is_ref_article_doc": False,
    },
    "지방계약법": {
        "document_type": "지방계약법",
        "law_name":      "지방계약법",
        "source":        "법률",
        "is_ref_article_doc": False,
    },
    "소프트웨어 진흥법 시행령": {
        "document_type": "소프트웨어 진흥법 시행령",
        "law_name":      "소프트웨어 진흥법 시행령",
        "source":        "대통령령",
        "is_ref_article_doc": False,
    },
    "소프트웨어_진흥법": {
        "document_type": "소프트웨어 진흥법",
        "law_name":      "소프트웨어 진흥법",
        "source":        "법률",
        "is_ref_article_doc": False,
    },
    "지방회계법_시행령": {
        "document_type": "지방회계법 시행령",
        "law_name":      "지방회계법 시행령",
        "source":        "대통령령",
        "is_ref_article_doc": False,
    },
    "지방회계법": {
        "document_type": "지방회계법",
        "law_name":      "지방회계법",
        "source":        "법률",
        "is_ref_article_doc": False,
    },
    "공유재산 및 물품 관리법 시행령": {
        "document_type": "공유재산법 시행령",
        "law_name":      "공유재산법 시행령",
        "source":        "대통령령",
        "is_ref_article_doc": False,
    },
    "공유재산법": {
        "document_type": "공유재산법",
        "law_name":      "공유재산법",
        "source":        "법률",
        "is_ref_article_doc": False,
    },
    "개인정보 보호법 시행령": {
        "document_type": "개인정보보호법 시행령",
        "law_name":      "개인정보보호법 시행령",
        "source":        "대통령령",
        "is_ref_article_doc": False,
    },
    "개인정보 보호법": {
        "document_type": "개인정보보호법",
        "law_name":      "개인정보보호법",
        "source":        "법률",
        "is_ref_article_doc": False,
    },
}

DOC_TYPE_TO_PREFIX = {
    "지방계약법":                    "LCA",
    "지방계약법 시행령":              "LCAE",
    "지방계약법 시행규칙":            "LCAR",
    "소프트웨어 진흥법":              "SWPA",
    "소프트웨어 진흥법 시행령":        "SWPAE",
    "지방회계법":                    "LARA",
    "지방회계법 시행령":              "LARAE",
    "지방자치단체 용역계약 일반조건":   "PYG",
    "공유재산법":                    "PPMA",
    "공유재산법 시행령":              "PPMAE",
    "개인정보보호법":                 "PIPA",
    "개인정보보호법 시행령":           "PIPAE",
}

# ─────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────
def read_docx(path: Path) -> list[tuple[str, str]]:
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


def find_meta(filename: str) -> dict | None:
    for key in sorted(FILE_META.keys(), key=len, reverse=True):
        if key in filename:
            return FILE_META[key]
    return None


def make_chunk_id(prefix: str, jo: int, hang: int | None = None,
                  ho: int | None = None, mok: str | None = None,
                  jo_ui: int | None = None, jang: int | None = None,
                  ho_ui: int | None = None, jeol: int | None = None) -> str:
    parts = []
    if jang is not None:
        parts.append(str(jang))
    if jeol is not None:
        parts.append(str(jeol))
    jo_part = str(jo) + (f"_의{jo_ui}" if jo_ui else "")
    parts.append(jo_part)
    if hang is not None:
        parts.append(str(hang))
    if ho is not None:
        parts.append(str(ho) + (f"_의{ho_ui}" if ho_ui else ""))
    if mok is not None:
        parts.append(mok)
    return f"{prefix}_{'_'.join(parts)}"


def _strip_comments(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[[^\]]+\]', '', text)
    return text.strip()


def _make_hierarchy(jang: int | None, jo: int, jo_ui: int | None,
                    hang: int | None = None, ho: int | None = None,
                    mok: str | None = None, ho_ui: int | None = None,
                    jeol: int | None = None) -> dict:
    jang_str = f"제{jang}장" if jang else None
    jeol_str = f"제{jeol}절" if jeol else None
    jo_str   = f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "")
    h = {}
    if jang_str:
        h["장"] = jang_str
    if jeol_str:
        h["절"] = jeol_str
    h["조"] = jo_str
    if hang is not None:
        h["항"] = f"제{hang}항"
    if ho is not None:
        h["호"] = f"제{ho}호" + (f"의{ho_ui}" if ho_ui else "")
    if mok is not None:
        h["목"] = mok
    return h


def _parse_ho_raw(ho_raw: str) -> tuple[int, int | None]:
    """'7', '7의2' 같은 호 번호 문자열 → (ho_num, ho_ui)"""
    m = re.match(r'(\d+)(?:의(\d+))?', ho_raw)
    ho_num = int(m.group(1)) if m else 0
    ho_ui  = int(m.group(2)) if m and m.group(2) else None
    return ho_num, ho_ui


# ─────────────────────────────────────────
# 파서 1: PYG 전용 — 9장 포함 구조 처리
# ─────────────────────────────────────────
def parse_pyg(lines: list[tuple[str, str]], prefix: str = "PYG",
              law_name: str = "지방자치단체 용역계약 일반조건") -> list[dict]:
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

    def get_chunk_id():
        id_parts = ([cur_chapter] if cur_chapter else []) + [cur_section, cur_clause]
        cid_parts = id_parts + ([cur_item] if cur_item else [])
        return f"{prefix}_{'_'.join(cid_parts)}"

    def get_parent_chunk_id():
        if cur_item:
            id_parts = ([cur_chapter] if cur_chapter else []) + [cur_section, cur_clause]
            return f"{prefix}_{'_'.join(id_parts)}"
        elif cur_clause:
            id_parts = ([cur_chapter] if cur_chapter else []) + [cur_section]
            return f"{prefix}_{'_'.join(id_parts)}"
        return None

    def flush():
        nonlocal buf, cur_item
        if not buf or not cur_section or not cur_clause:
            buf = []; cur_item = None
            return

        chapter_str = (f"제{cur_chapter}장" if cur_chapter else "")
        if cur_item:
            an = chapter_str + f"제{cur_section}절제{cur_clause}항{cur_item}"
            hierarchy = {"절": f"제{cur_section}절", "항": f"제{cur_clause}항", "호": cur_item}
        else:
            an = chapter_str + f"제{cur_section}절제{cur_clause}항"
            hierarchy = {"절": f"제{cur_section}절", "항": f"제{cur_clause}항"}

        if cur_chapter:
            hierarchy["장"] = f"제{cur_chapter}장"

        full_text = _strip_comments(" ".join(buf))
        ho_chunk_id = get_chunk_id()
        ho_parent   = get_parent_chunk_id()

        mok_splits = re.split(r'\s+(\d+)\)\s+', full_text) if cur_item else []

        if cur_item and len(mok_splits) > 1:
            articles.append({
                "chunk_id":        ho_chunk_id,
                "law_name":        law_name,
                "article_id":      an,
                "article_number":  an,
                "text":            mok_splits[0].strip(),
                "parent_chunk_id": ho_parent,
                "is_ref_article":  False,
                "is_upper_law":    False,
                "hierarchy":       hierarchy,
            })
            # 콤마/인용부호 뒤에 오는 "다), 라)" 등은 진짜 세목이 아니라 인용된 참조 표현
            # (예: "가-1)-가), 다) , 라) "의 경우) 이므로 분리 대상에서 제외
            semok_pat = re.compile(r'(?<![,“"”])\s+([가나다라마바사아자차카타파하])\)\s*')
            j = 1
            while j < len(mok_splits) - 1:
                mok_num  = mok_splits[j]
                mok_text = mok_splits[j + 1].strip() if j + 1 < len(mok_splits) else ""
                mok_an   = an + f"제{mok_num}호"
                mok_hier = {**hierarchy, "목": f"제{mok_num}호"}
                id_parts = ([cur_chapter] if cur_chapter else []) + [cur_section, cur_clause, cur_item, mok_num]
                mok_chunk_id = f"{prefix}_{'_'.join(id_parts)}"

                # 세목(가)/나)/다)) 분리
                semok_splits = semok_pat.split(mok_text)
                if len(semok_splits) > 1:
                    articles.append({
                        "chunk_id":        mok_chunk_id,
                        "law_name":        law_name,
                        "article_id":      mok_an,
                        "article_number":  mok_an,
                        "text":            f"{mok_num}) {semok_splits[0].strip()}",
                        "parent_chunk_id": ho_chunk_id,
                        "is_ref_article":  False,
                        "is_upper_law":    False,
                        "hierarchy":       mok_hier,
                    })
                    k = 1
                    while k < len(semok_splits) - 1:
                        semok_char = semok_splits[k]
                        semok_text = semok_splits[k + 1].strip() if k + 1 < len(semok_splits) else ""
                        semok_an   = mok_an + semok_char
                        articles.append({
                            "chunk_id":        f"{mok_chunk_id}_{semok_char}",
                            "law_name":        law_name,
                            "article_id":      semok_an,
                            "article_number":  semok_an,
                            "text":            f"{semok_char}) {semok_text}",
                            "parent_chunk_id": mok_chunk_id,
                            "is_ref_article":  False,
                            "is_upper_law":    False,
                            "hierarchy":       {**mok_hier, "세목": f"{semok_char})"},
                        })
                        k += 2
                else:
                    articles.append({
                        "chunk_id":        mok_chunk_id,
                        "law_name":        law_name,
                        "article_id":      mok_an,
                        "article_number":  mok_an,
                        "text":            f"{mok_num}) {mok_text}",
                        "parent_chunk_id": ho_chunk_id,
                        "is_ref_article":  False,
                        "is_upper_law":    False,
                        "hierarchy":       mok_hier,
                    })
                j += 2
        else:
            articles.append({
                "chunk_id":        ho_chunk_id,
                "law_name":        law_name,
                "article_id":      an,
                "article_number":  an,
                "text":            full_text,
                "parent_chunk_id": ho_parent,
                "is_ref_article":  False,
                "is_upper_law":    False,
                "hierarchy":       hierarchy,
            })
        buf = []; cur_item = None

    for typ, text in lines:
        chm = chapter_pat.match(text)
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
            buf = [text]
        elif im and cur_clause:
            flush()
            buf = [text]
            cur_item = im.group(1)
        elif cur_clause:
            buf.append(text)

    flush()
    return articles


# ─────────────────────────────────────────
# 파서 2: 일반 법령 — 조/항/호/목
# - 장 헤더 감지 → cur_jang 추적
# - 호 아래 가/나/다 목 분리
# - 항에 호가 있으면 항 intro chunk + 호 chunk 분리
# - 항 없이 바로 호로 가는 조도 직접 분리
# - "7의2." 같은 호의N 번호 지원
# - 같은 (조,조의N) 중복(원본 복붙/시행일 예고 병기) 발생 시 첫 번째(현행)만 유지
# ─────────────────────────────────────────
def parse_law(lines: list[tuple[str, str]], prefix: str, law_name: str) -> list[dict]:
    article_pat = re.compile(r"^(제\s*\d+\s*조(?:의\s*\d+)?)\s*[(\[〔]?([^)\]\)〕\n]*)[)\]\)〕]?")
    jang_pat    = re.compile(r"^제\s*(\d+)\s*장")
    jeol_pat    = re.compile(r"^제\s*(\d+)\s*절")
    HANG_MAP    = {c: i + 1 for i, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮")}

    # ── 1단계: 조 단위로 텍스트 수집 ──
    raw_articles: list[dict] = []
    cur_jo    = None
    cur_jo_ui = None
    cur_jang  = None
    cur_jeol  = None
    cur_title = ""
    buf: list[str] = []
    in_bujik  = False
    bujik_pat = re.compile(r"^부\s*칙")

    def flush_jo():
        if cur_jo is not None and buf:
            raw_articles.append({
                "jo":    cur_jo,
                "jo_ui": cur_jo_ui,
                "jang":  cur_jang,
                "jeol":  cur_jeol,
                "title": cur_title,
                "text":  " ".join(buf),
            })

    for _, text in lines:
        if bujik_pat.match(text):
            in_bujik = True
            flush_jo()
            cur_jo = None; buf = []
            continue
        if in_bujik:
            continue

        jm = jang_pat.match(text)
        if jm and not article_pat.match(text):
            flush_jo()
            cur_jang = int(jm.group(1))
            cur_jeol = None  # 장이 바뀌면 절도 초기화
            cur_jo = None; buf = []
            continue

        # 절 헤더 추적 (장처럼 추적, 본문에는 포함시키지 않음)
        jlm = jeol_pat.match(text)
        if jlm and not article_pat.match(text):
            flush_jo()
            cur_jeol = int(jlm.group(1))
            cur_jo = None; buf = []
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

    # 같은 (조, 조의N) 중복 시 첫 번째(현행 시행 버전)만 유지
    # - 원본 문서 복붙 중복(예: 지방계약법 시행규칙 제25조)
    # - 법제처 "[시행일: YYYY.M.D] 제N조" 표기로 향후 개정본이 같이 실린 경우(예: 개인정보보호법)
    seen_jo: set[tuple[int, int | None]] = set()
    deduped_raw_articles: list[dict] = []
    for raw in raw_articles:
        key = (raw["jo"], raw["jo_ui"])
        if key in seen_jo:
            continue
        seen_jo.add(key)
        deduped_raw_articles.append(raw)
    raw_articles = deduped_raw_articles

    # ── 2단계: 조 → 항 → 호 → 목(가/나/다) 분리 ──
    articles: list[dict] = []
    mok_pat = re.compile(r'\s([가나다라마바사아자차카타파하])\.\s')
    # 마침표 뒤가 공백이거나(보통의 경우), 공백 없이 바로 「『《 인용부호로 이어지는 경우도 호 구분자로 인식
    ho_split_pat = re.compile(r"\s(\d{1,2}(?:의\d+)?)\.(?:\s|(?=[「『《]))")

    for raw in raw_articles:
        jo     = raw["jo"]
        jo_ui  = raw["jo_ui"]
        jang   = raw["jang"]
        jeol   = raw["jeol"]
        text   = raw["text"]
        title  = raw["title"]

        jang_str = f"제{jang}장" if jang else ""
        jeol_str = f"제{jeol}절" if jeol else ""
        jo_str   = jang_str + jeol_str + f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "")
        jo_chunk_id = make_chunk_id(prefix, jo, jo_ui=jo_ui, jang=jang, jeol=jeol)

        hang_splits = re.split(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])", text)

        # 항 없는 조 → 직접 호 분리 시도
        if len(hang_splits) <= 1:
            text_clean = _strip_comments(text)
            ho_splits  = ho_split_pat.split(text_clean)

            if len(ho_splits) <= 1:
                articles.append({
                    "chunk_id":        jo_chunk_id,
                    "law_name":        law_name,
                    "article_id":      jo_str,
                    "article_number":  jo_str,
                    "title":           title,
                    "text":            text_clean,
                    "parent_chunk_id": None,
                    "is_ref_article":  False,
                    "is_upper_law":    False,
                    "hierarchy":       _make_hierarchy(jang, jo, jo_ui, jeol=jeol),
                })
            else:
                articles.append({
                    "chunk_id":        jo_chunk_id,
                    "law_name":        law_name,
                    "article_id":      jo_str,
                    "article_number":  jo_str,
                    "title":           title,
                    "text":            ho_splits[0].strip(),
                    "parent_chunk_id": None,
                    "is_ref_article":  False,
                    "is_upper_law":    False,
                    "hierarchy":       _make_hierarchy(jang, jo, jo_ui, jeol=jeol),
                })
                j = 1
                while j < len(ho_splits) - 1:
                    ho_raw  = ho_splits[j]
                    ho_text = ho_splits[j + 1].strip() if j + 1 < len(ho_splits) else ""
                    ho_num, ho_ui = _parse_ho_raw(ho_raw)
                    ho_str  = jo_str + f"제{ho_raw}호"
                    ho_chunk_id = make_chunk_id(prefix, jo, ho=ho_num, ho_ui=ho_ui, jo_ui=jo_ui, jang=jang, jeol=jeol)

                    mok_splits = re.split(mok_pat, ho_text)
                    if len(mok_splits) > 1:
                        articles.append({
                            "chunk_id":        ho_chunk_id,
                            "law_name":        law_name,
                            "article_id":      ho_str,
                            "article_number":  ho_str,
                            "title":           title,
                            "text":            mok_splits[0].strip(),
                            "parent_chunk_id": jo_chunk_id,
                            "is_ref_article":  False,
                            "is_upper_law":    False,
                            "hierarchy":       _make_hierarchy(jang, jo, jo_ui, ho=ho_num, ho_ui=ho_ui, jeol=jeol),
                        })
                        k = 1
                        while k < len(mok_splits) - 1:
                            mok_char = mok_splits[k]
                            mok_text = mok_splits[k + 1].strip() if k + 1 < len(mok_splits) else ""
                            articles.append({
                                "chunk_id":        make_chunk_id(prefix, jo, ho=ho_num, ho_ui=ho_ui, mok=mok_char, jo_ui=jo_ui, jang=jang, jeol=jeol),
                                "law_name":        law_name,
                                "article_id":      ho_str + mok_char,
                                "article_number":  ho_str + mok_char,
                                "title":           title,
                                "text":            f"{mok_char}. {mok_text}",
                                "parent_chunk_id": ho_chunk_id,
                                "is_ref_article":  False,
                                "is_upper_law":    False,
                                "hierarchy":       _make_hierarchy(jang, jo, jo_ui, ho=ho_num, mok=mok_char, ho_ui=ho_ui, jeol=jeol),
                            })
                            k += 2
                    else:
                        articles.append({
                            "chunk_id":        ho_chunk_id,
                            "law_name":        law_name,
                            "article_id":      ho_str,
                            "article_number":  ho_str,
                            "title":           title,
                            "text":            f"{ho_raw}. {ho_text}",
                            "parent_chunk_id": jo_chunk_id,
                            "is_ref_article":  False,
                            "is_upper_law":    False,
                            "hierarchy":       _make_hierarchy(jang, jo, jo_ui, ho=ho_num, ho_ui=ho_ui, jeol=jeol),
                        })
                    j += 2
            continue

        # 조 chunk (① 이전 intro)
        jo_intro = _strip_comments(hang_splits[0].strip())
        articles.append({
            "chunk_id":        jo_chunk_id,
            "law_name":        law_name,
            "article_id":      jo_str,
            "article_number":  jo_str,
            "title":           title,
            "text":            jo_intro,
            "parent_chunk_id": None,
            "is_ref_article":  False,
            "is_upper_law":    False,
            "hierarchy":       _make_hierarchy(jang, jo, jo_ui, jeol=jeol),
        })

        i = 1
        while i < len(hang_splits) - 1:
            hang_char = hang_splits[i]
            hang_text = hang_splits[i + 1].strip() if i + 1 < len(hang_splits) else ""
            hang_num  = HANG_MAP.get(hang_char, i)

            hang_str      = jo_str + f"제{hang_num}항"
            hang_chunk_id = make_chunk_id(prefix, jo, hang=hang_num, jo_ui=jo_ui, jang=jang, jeol=jeol)

            hang_text_clean = _strip_comments(hang_text)
            ho_splits = ho_split_pat.split(hang_text_clean)

            if len(ho_splits) <= 1:
                articles.append({
                    "chunk_id":        hang_chunk_id,
                    "law_name":        law_name,
                    "article_id":      hang_str,
                    "article_number":  hang_str,
                    "title":           title,
                    "text":            hang_char + " " + hang_text_clean,
                    "parent_chunk_id": jo_chunk_id,
                    "is_ref_article":  False,
                    "is_upper_law":    False,
                    "hierarchy":       _make_hierarchy(jang, jo, jo_ui, hang_num, jeol=jeol),
                })
            else:
                hang_intro = ho_splits[0].strip()
                articles.append({
                    "chunk_id":        hang_chunk_id,
                    "law_name":        law_name,
                    "article_id":      hang_str,
                    "article_number":  hang_str,
                    "title":           title,
                    "text":            hang_char + " " + hang_intro,
                    "parent_chunk_id": jo_chunk_id,
                    "is_ref_article":  False,
                    "is_upper_law":    False,
                    "hierarchy":       _make_hierarchy(jang, jo, jo_ui, hang_num, jeol=jeol),
                })

                j = 1
                while j < len(ho_splits) - 1:
                    ho_raw   = ho_splits[j]
                    ho_text  = ho_splits[j + 1].strip() if j + 1 < len(ho_splits) else ""
                    ho_num, ho_ui = _parse_ho_raw(ho_raw)
                    ho_str   = hang_str + f"제{ho_raw}호"
                    ho_chunk_id = make_chunk_id(prefix, jo, hang=hang_num, ho=ho_num, ho_ui=ho_ui, jo_ui=jo_ui, jang=jang, jeol=jeol)

                    mok_splits = re.split(mok_pat, ho_text)

                    if len(mok_splits) > 1:
                        articles.append({
                            "chunk_id":        ho_chunk_id,
                            "law_name":        law_name,
                            "article_id":      ho_str,
                            "article_number":  ho_str,
                            "title":           title,
                            "text":            mok_splits[0].strip(),
                            "parent_chunk_id": hang_chunk_id,
                            "is_ref_article":  False,
                            "is_upper_law":    False,
                            "hierarchy":       _make_hierarchy(jang, jo, jo_ui, hang_num, ho_num, ho_ui=ho_ui, jeol=jeol),
                        })
                        k = 1
                        while k < len(mok_splits) - 1:
                            mok_char = mok_splits[k]
                            mok_text = mok_splits[k + 1].strip() if k + 1 < len(mok_splits) else ""
                            articles.append({
                                "chunk_id":        make_chunk_id(prefix, jo, hang=hang_num, ho=ho_num, ho_ui=ho_ui, mok=mok_char, jo_ui=jo_ui, jang=jang, jeol=jeol),
                                "law_name":        law_name,
                                "article_id":      ho_str + mok_char,
                                "article_number":  ho_str + mok_char,
                                "title":           title,
                                "text":            f"{mok_char}. {mok_text}",
                                "parent_chunk_id": ho_chunk_id,
                                "is_ref_article":  False,
                                "is_upper_law":    False,
                                "hierarchy":       _make_hierarchy(jang, jo, jo_ui, hang_num, ho_num, mok_char, ho_ui=ho_ui, jeol=jeol),
                            })
                            k += 2
                    else:
                        articles.append({
                            "chunk_id":        ho_chunk_id,
                            "law_name":        law_name,
                            "article_id":      ho_str,
                            "article_number":  ho_str,
                            "title":           title,
                            "text":            f"{ho_raw}. {ho_text}",
                            "parent_chunk_id": hang_chunk_id,
                            "is_ref_article":  False,
                            "is_upper_law":    False,
                            "hierarchy":       _make_hierarchy(jang, jo, jo_ui, hang_num, ho_num, ho_ui=ho_ui, jeol=jeol),
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
# cross_refs: 법령명 → prefix 매핑
# ─────────────────────────────────────────
LAW_TO_PREFIX = {
    "지방계약법 시행규칙":           "LCAR",
    "지방계약법 시행령":             "LCAE",
    "지방계약법":                   "LCA",
    "소프트웨어 진흥법 시행령":      "SWPAE",
    "소프트웨어진흥법 시행령":       "SWPAE",
    "소프트웨어 진흥법":             "SWPA",
    "소프트웨어진흥법":              "SWPA",
    "지방회계법 시행령":             "LARAE",
    "지방회계법":                   "LARA",
    "지방자치단체 용역계약 일반조건": "PYG",
    "공유재산 및 물품 관리법 시행령": "PPMAE",
    "공유재산법 시행령":             "PPMAE",
    "공유재산 및 물품 관리법":       "PPMA",
    "공유재산법":                   "PPMA",
    "개인정보 보호법 시행령":        "PIPAE",
    "개인정보보호법 시행령":         "PIPAE",
    "개인정보 보호법":               "PIPA",
    "개인정보보호법":                "PIPA",
}


def resolve_prefix(law_name: str) -> str | None:
    for key in sorted(LAW_TO_PREFIX.keys(), key=len, reverse=True):
        if key in law_name:
            return LAW_TO_PREFIX[key]
    return None


# "법"/"영"/"규칙" 단독 언급 시 현재 prefix 기준으로 대상 prefix 결정
RELATIVE_LAW_MAP: dict[str, dict[str, str]] = {
    "LCA":   {"법": "LCA"},
    "LCAE":  {"법": "LCA",  "영": "LCAE"},
    "LCAR":  {"법": "LCA",  "영": "LCAE", "규칙": "LCAR"},
    "SWPA":  {"법": "SWPA"},
    "SWPAE": {"법": "SWPA", "영": "SWPAE"},
    "LARA":  {"법": "LARA"},
    "LARAE": {"법": "LARA", "영": "LARAE"},
    "PPMA":  {"법": "PPMA"},
    "PPMAE": {"법": "PPMA", "영": "PPMAE"},
    "PIPA":  {"법": "PIPA"},
    "PIPAE": {"법": "PIPA", "영": "PIPAE"},
    # PYG는 모법이 없으므로 지방계약법 계열을 직접 가리킴
    "PYG":   {"법": "LCA", "영": "LCAE", "규칙": "LCAR"},
}

# "시행령"/"시행규칙" 같은 긴 표기를 "영"/"규칙"으로 정규화
KW_NORMALIZE = {"시행령": "영", "시행규칙": "규칙"}


def resolve_relative(current_prefix: str, keyword: str) -> str | None:
    return RELATIVE_LAW_MAP.get(current_prefix, {}).get(keyword)


# ─────────────────────────────────────────
# 항 없는 조 chunk에서 직접 호 분리 (후처리, PYG 등 잔여 케이스 대비)
# ─────────────────────────────────────────
def split_direct_ho(articles: list[dict], prefix: str, law_name: str) -> list[dict]:
    ho_pat  = re.compile(r'\s(\d{1,2}(?:의\d+)?)\.(?:\s|(?=[「『《]))')
    mok_pat = re.compile(r'\s([가나다라마바사아자차카타파하])\.\s')
    result  = []

    for article in articles:
        h = article.get("hierarchy", {})
        if "항" in h or "호" in h or "목" in h:
            result.append(article)
            continue

        text      = article.get("text", "")
        ho_splits = ho_pat.split(text)

        if len(ho_splits) <= 1:
            result.append(article)
            continue

        jo_m = re.match(r"제(\d+)조(?:의(\d+))?", h.get("조", ""))
        if not jo_m:
            result.append(article)
            continue
        jo    = int(jo_m.group(1))
        jo_ui = int(jo_m.group(2)) if jo_m.group(2) else None

        jang_m = re.match(r"제(\d+)장", h.get("장", ""))
        jang   = int(jang_m.group(1)) if jang_m else None

        jo_chunk_id = article["chunk_id"]
        jang_str    = f"제{jang}장" if jang else ""
        jo_str      = jang_str + f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "")

        result.append({**article, "text": ho_splits[0].strip()})

        j = 1
        while j < len(ho_splits) - 1:
            ho_raw  = ho_splits[j]
            ho_text = ho_splits[j + 1].strip() if j + 1 < len(ho_splits) else ""
            ho_num, ho_ui = _parse_ho_raw(ho_raw)

            ho_chunk_id = make_chunk_id(prefix, jo, ho=ho_num, ho_ui=ho_ui, jo_ui=jo_ui, jang=jang)
            ho_str      = jo_str + f"제{ho_raw}호"
            ho_hier     = {**h, "호": f"제{ho_raw}호"}
            common      = {
                "law_name":        law_name,
                "title":           article.get("title", ""),
                "parent_chunk_id": jo_chunk_id,
                "is_ref_article":  article.get("is_ref_article", False),
                "is_upper_law":    article.get("is_upper_law", False),
            }

            mok_splits = mok_pat.split(ho_text)
            if len(mok_splits) > 1:
                result.append({
                    "chunk_id": ho_chunk_id, "article_id": ho_str,
                    "article_number": ho_str, "text": mok_splits[0].strip(),
                    "hierarchy": ho_hier, **common,
                })
                k = 1
                while k < len(mok_splits) - 1:
                    mok_char = mok_splits[k]
                    mok_text = mok_splits[k + 1].strip() if k + 1 < len(mok_splits) else ""
                    mok_str  = ho_str + mok_char
                    result.append({
                        "chunk_id": ho_chunk_id + f"_{mok_char}",
                        "article_id": mok_str, "article_number": mok_str,
                        "text": f"{mok_char}. {mok_text}",
                        "hierarchy": {**ho_hier, "목": mok_char},
                        "parent_chunk_id": ho_chunk_id,
                        "law_name": law_name,
                        "title": article.get("title", ""),
                        "is_ref_article": article.get("is_ref_article", False),
                        "is_upper_law": article.get("is_upper_law", False),
                    })
                    k += 2
            else:
                result.append({
                    "chunk_id": ho_chunk_id, "article_id": ho_str,
                    "article_number": ho_str, "text": f"{ho_raw}. {ho_text}",
                    "hierarchy": ho_hier, **common,
                })
            j += 2

    return result


def build_registry(articles: list[dict]) -> dict:
    """
    조/항/호/목 구조 법령 → (jo, jo_ui, hang, ho, mok, semok) 키
    PYG(절/항/호/목/세목 구조) → 절 번호를 jo 자리에 사용. 호는 가/나/다 글자 그대로 저장
    (숫자 "제N호" 형식이 아니므로 build_registry가 원본 문자열을 그대로 키로 사용해야
    동일 항 안의 가/나/다 항목들이 서로 다른 키로 구분됨 — 안 그러면 키 충돌로 덮어써짐)
    """
    reg = {}
    for a in articles:
        chunk_id = a["chunk_id"]
        h = a.get("hierarchy", {})

        jo_m = re.match(r"제(\d+)조(?:의(\d+))?", h.get("조", ""))
        if jo_m:
            jo    = int(jo_m.group(1))
            jo_ui = int(jo_m.group(2)) if jo_m.group(2) else None
        else:
            jeol_m = re.match(r"제(\d+)절", h.get("절", ""))
            if not jeol_m:
                continue
            jo    = int(jeol_m.group(1))
            jo_ui = None

        hang_m = re.match(r"제(\d+)항", h.get("항", ""))
        hang   = int(hang_m.group(1)) if hang_m else None

        ho_field = h.get("호", "")
        ho_m = re.match(r"제(\d+)호", ho_field)
        if ho_m:
            ho = int(ho_m.group(1))
        elif ho_field:
            ho = ho_field  # PYG: 가/나/다 글자 그대로
        else:
            ho = None

        mok = h.get("목") or None
        semok = h.get("세목") or None

        reg[(jo, jo_ui, hang, ho, mok, semok)] = chunk_id
    return reg


def lookup(registry: dict, target_prefix: str,
           jo: int, jo_ui: int | None,
           hang: int | None, ho: int | str | None,
           mok: str | None = None, semok: str | None = None) -> str:
    """레지스트리에서 실제 chunk_id 조회, 없으면 fallback 생성"""
    key = (jo, jo_ui, hang, ho, mok, semok)
    reg = registry.get(target_prefix, {})
    if key in reg:
        return reg[key]
    parts = [str(jo) + (f"_의{jo_ui}" if jo_ui else "")]
    if hang is not None:
        parts.append(str(hang))
    if ho is not None:
        parts.append(str(ho))
    if mok is not None:
        parts.append(mok)
    if semok is not None:
        parts.append(semok)
    return f"{target_prefix}_{'_'.join(parts)}"


def lookup_all_ho(registry: dict, target_prefix: str,
                  jo: int, jo_ui: int | None, hang: int | None) -> list[str]:
    """특정 조(항) 아래의 모든 호 chunk_id 조회 ('각 호' 패턴 처리용)"""
    reg = registry.get(target_prefix, {})
    results = []
    for (rjo, rjo_ui, rhang, rho, rmok, rsemok) in reg:
        if rjo == jo and rjo_ui == jo_ui and rhang == hang and rho is not None and rmok is None:
            results.append(reg[(rjo, rjo_ui, rhang, rho, rmok, rsemok)])
    return results


# ─────────────────────────────────────────
# 조항호 번호 추출 (범위형·열거형)
# ─────────────────────────────────────────
def extract_nums(text: str, level: str) -> list[int]:
    nums: set[int] = set()
    # "까지"가 생략된 형태("제1항부터 제7항 및...")도 있으므로 "까지"는 선택사항으로 처리
    range_pat = re.compile(rf'제(\d+){level}부터\s*제(\d+){level}(?:까지)?')
    for m in range_pat.finditer(text):
        nums.update(range(int(m.group(1)), int(m.group(2)) + 1))
    single_pat = re.compile(rf'제(\d+){level}')
    for m in single_pat.finditer(text):
        nums.add(int(m.group(1)))
    return sorted(nums)


def parse_ref_segment(seg: str, target_prefix: str, registry: dict) -> list[str]:
    """
    조항호 참조 텍스트 → chunk_id 목록
    레지스트리로 실제 chunk_id 조회 (jang 포함 여부 자동 처리)
    """
    results: list[str] = []
    jo_pat = re.compile(r'제(\d+)조(?:의(\d+))?')
    jo_matches = list(jo_pat.finditer(seg))

    for idx, jo_m in enumerate(jo_matches):
        jo    = int(jo_m.group(1))
        jo_ui = int(jo_m.group(2)) if jo_m.group(2) else None
        end   = jo_matches[idx + 1].start() if idx + 1 < len(jo_matches) else len(seg)
        chunk = seg[jo_m.start():end]

        hang_refs = extract_nums(chunk, '항')
        ho_refs   = extract_nums(chunk, '호')

        if not hang_refs and not ho_refs:
            results.append(lookup(registry, target_prefix, jo, jo_ui, None, None))
        elif hang_refs and not ho_refs:
            for h in hang_refs:
                results.append(lookup(registry, target_prefix, jo, jo_ui, h, None))
        elif ho_refs and not hang_refs:
            for h in ho_refs:
                results.append(lookup(registry, target_prefix, jo, jo_ui, None, h))
        else:
            for hang in hang_refs:
                for ho in ho_refs:
                    results.append(lookup(registry, target_prefix, jo, jo_ui, hang, ho))

    return results


def extract_cross_refs(text: str, current_prefix: str, registry: dict,
                       article: dict | None = None) -> list[str]:
    refs: list[str] = []

    cur_jo, cur_jo_ui, cur_hang, cur_ho = None, None, None, None
    cur_jeol, cur_pyg_ho = None, None  # PYG(절/항/호 구조) 전용
    if article:
        h = article.get("hierarchy", {})
        jo_m = re.match(r"제(\d+)조(?:의(\d+))?", h.get("조", ""))
        if jo_m:
            cur_jo    = int(jo_m.group(1))
            cur_jo_ui = int(jo_m.group(2)) if jo_m.group(2) else None
        hang_m = re.match(r"제(\d+)항", h.get("항", ""))
        cur_hang = int(hang_m.group(1)) if hang_m else None
        ho_m = re.match(r"제(\d+)호", h.get("호", ""))
        cur_ho = int(ho_m.group(1)) if ho_m else None

        jeol_m = re.match(r"제(\d+)절", h.get("절", ""))
        if jeol_m:
            cur_jeol = int(jeol_m.group(1))
            if h.get("호") and re.match(r"^[가나다라마바사아자차카타파하]$", h.get("호")):
                cur_pyg_ho = h.get("호")

    consumed: list[tuple[int, int]] = []

    # 1. 「법령명」 뒤 조항호 참조
    law_pat = re.compile(r'[「『《]([^」』》]+)[」』》]')
    for law_m in law_pat.finditer(text):
        target_prefix = resolve_prefix(law_m.group(1))
        after = text[law_m.end():].lstrip()
        stop  = re.search(r'[。.\n]|[「『《]', after)
        seg   = after[:stop.start()].strip() if stop else after[:150].strip()
        seg_start = text.index(seg, law_m.end()) if seg else law_m.end()
        consumed.append((law_m.start(), seg_start + len(seg)))
        if not target_prefix:
            continue
        refs.extend(parse_ref_segment(seg, target_prefix, registry))

    # 2. 법/영/규칙 제N조... (이 법, 이 영, 이 규칙, 시행령, 시행규칙, 또는 단독 법/영/규칙)
    same_pat = re.compile(r'(?:이\s*)?(?P<kw>시행규칙|시행령|법|영|규칙)\s+(제\d+조[^。\n「」]{0,80})')
    for m in same_pat.finditer(text):
        kw     = KW_NORMALIZE.get(m.group("kw"), m.group("kw"))
        target = resolve_relative(current_prefix, kw)
        if not target:
            continue
        consumed.append(m.span())
        refs.extend(parse_ref_segment(m.group(2), target, registry))

    # 2-4. 조 단위 범위 참조: "제20조부터 제22조까지" (법명 없음)
    jo_range_pat = re.compile(r'제(\d+)조부터\s*제(\d+)조(?:까지)?')
    for m in jo_range_pat.finditer(text):
        consumed.append(m.span())
        for j in range(int(m.group(1)), int(m.group(2)) + 1):
            refs.append(lookup(registry, current_prefix, j, None, None, None))

    # 2-5. 「」/법영규칙 없이 단독으로 나오는 "제N조(의M)제K항제L호" — 동일 법령 참조
    bare_jo_pat = re.compile(r'제(\d+)조(?:의(\d+))?(?:제(\d+)항)?(?:제(\d+)호)?')
    for m in bare_jo_pat.finditer(text):
        if any(s <= m.start() < e for s, e in consumed):
            continue
        ref_jo    = int(m.group(1))
        ref_jo_ui = int(m.group(2)) if m.group(2) else None
        ref_hang  = int(m.group(3)) if m.group(3) else None
        ref_ho    = int(m.group(4)) if m.group(4) else None
        if (cur_jo is not None and ref_jo == cur_jo and ref_jo_ui == cur_jo_ui
                and (ref_hang is None or ref_hang == cur_hang)
                and (ref_ho is None or ref_ho == cur_ho)):
            continue
        refs.append(lookup(registry, current_prefix, ref_jo, ref_jo_ui, ref_hang, ref_ho))

    # 3. 같은 조 내 항 참조: 제N조 없이 "제N항"/"제N항제M호" 또는 범위형이 나오는 경우
    # (단, 「」/법영규칙/이미 소비된 구간에 속한 항은 그쪽 조에 이미 정확히 귀속됐으므로 제외 —
    #  예: "법 제55조제1항... 같은 조 제4항" 에서 "제4항"이 현재 조(49조)로 잘못 잡히는 것 방지)
    if cur_jo is not None:
        hang_range_spans: list[tuple[int, int]] = []
        hang_range_pat = re.compile(r'(?<!조)(?<!\d)제(\d+)항부터\s*제(\d+)항(?:까지)?')
        for m in hang_range_pat.finditer(text):
            if any(s <= m.start() < e for s, e in consumed):
                continue
            hang_range_spans.append(m.span())
            for h in range(int(m.group(1)), int(m.group(2)) + 1):
                refs.append(lookup(registry, current_prefix, cur_jo, cur_jo_ui, h, None))

        bare_pat = re.compile(r'(?<!조)(?<!\d)제(\d+)항(?:제(\d+)호)?')
        for m in bare_pat.finditer(text):
            if any(s <= m.start() < e for s, e in hang_range_spans):
                continue
            if any(s <= m.start() < e for s, e in consumed):
                continue
            hang = int(m.group(1))
            ho   = int(m.group(2)) if m.group(2) else None
            refs.append(lookup(registry, current_prefix, cur_jo, cur_jo_ui, hang, ho))

    # 3-4. 같은 조(항) 내 호 참조: 조/항 없이 "제N호" 또는 범위형("제N호부터 제M호까지")만 나오는 경우
    # (호 레벨 chunk 안에서 형제 호를 가리키는 경우. 예: "제1호부터 제3호까지의 지수와 유사한")
    # cur_hang이 None인 경우(항 없이 조 바로 밑에 호가 있는 구조)도 포함
    if cur_jo is not None and cur_ho is not None:
        ho_range_spans: list[tuple[int, int]] = []
        ho_range_pat = re.compile(r'(?<!호)(?<!\d)제(\d+)호부터\s*제(\d+)호(?:까지)?')
        for m in ho_range_pat.finditer(text):
            if any(s <= m.start() < e for s, e in consumed):
                continue
            ho_range_spans.append(m.span())
            for h in range(int(m.group(1)), int(m.group(2)) + 1):
                refs.append(lookup(registry, current_prefix, cur_jo, cur_jo_ui, cur_hang, h))

        bare_ho_pat = re.compile(r'(?<!조)(?<!항)(?<!\d)제(\d+)호')
        for m in bare_ho_pat.finditer(text):
            if any(s <= m.start() < e for s, e in ho_range_spans):
                continue
            if any(s <= m.start() < e for s, e in consumed):
                continue
            ho = int(m.group(1))
            if ho == cur_ho:
                continue  # 자기 자신 제외
            refs.append(lookup(registry, current_prefix, cur_jo, cur_jo_ui, cur_hang, ho))

    # 3-5. PYG 전용: 같은 문서 내 "제N절 "M[-호)?][-목)?][-세목)?]"" 자기참조
    # 예: 제7절 "1"        → 절-항
    #     제6절 "1-다"      → 절-항-호
    #     제5절 "1-마)-1)-나)" → 절-항-호-목-세목
    pyg_deep_pat = re.compile(
        r'제(\d+)절\s*[“"](\d+)(?:-([가나다라마바사아자차카타파하])\)?)?'
        r'(?:-(\d+)\)?)?(?:-([가나다라마바사아자차카타파하])\)?)?[”"]'
    )
    # 따옴표 안 내용("M-X)-Y)-Z)" 또는 줄임형 "다"/"5"/"5)"/"다-1)")을 파싱.
    # 줄임형은 이전 참조의 항/호를 이어받음(예: "8-가"와"다" → 둘 다 제8항, 호만 가/다로 다름)
    def _parse_pyg_token(token: str, prev_clause: int, prev_ho: str | None):
        token = token.strip()
        m = re.match(r'^(\d+)(?:-([가나다라마바사아자차카타파하])\)?)?'
                     r'(?:-(\d+)\)?)?(?:-([가나다라마바사아자차카타파하])\)?)?$', token)
        if m:
            clause = int(m.group(1))
            ho     = m.group(2)
            mok    = f"제{m.group(3)}호" if m.group(3) else None
            semok  = f"{m.group(4)})" if m.group(4) else None
            return clause, ho, mok, semok
        m = re.match(r'^([가나다라마바사아자차카타파하])$', token)  # 글자만(호)
        if m:
            return prev_clause, m.group(1), None, None
        m = re.match(r'^(\d+)\)$', token)  # 숫자) 만(목)
        if m:
            return prev_clause, prev_ho, f"제{m.group(1)}호", None
        return None, None, None, None

    for m in pyg_deep_pat.finditer(text):
        sec    = int(m.group(1))
        clause = int(m.group(2))
        ho     = m.group(3)
        mok    = f"제{m.group(4)}호" if m.group(4) else None
        semok  = f"{m.group(5)})" if m.group(5) else None
        refs.append(lookup(registry, current_prefix, sec, None, clause, ho, mok, semok))

        pos = m.end()
        consumed.append((m.start(), pos))

        # 같은 절을 가리키는 연속 인용("X"와"Y", "X","Y", "X"및"Y") 처리
        cont_pat = re.compile(r'^\s*(?:와|및|,|·)\s*[“"]([^”"]{1,20})[”"]')
        loop_clause, loop_ho = clause, ho
        while True:
            cm = cont_pat.match(text[pos:])
            if not cm:
                break
            c_clause, c_ho, c_mok, c_semok = _parse_pyg_token(cm.group(1), loop_clause, loop_ho)
            if c_clause is None:
                break
            refs.append(lookup(registry, current_prefix, sec, None, c_clause, c_ho, c_mok, c_semok))
            loop_clause, loop_ho = c_clause, c_ho
            consumed.append((pos, pos + cm.end()))
            pos += cm.end()

    # 3-6. PYG 전용: "제N절" 없이 맨몸 따옴표("가"/"나")로 같은 절·항 안의 형제 호를 가리키는 경우
    # 예: ""가"와 "나"에 따라 계약금액을 조정하는 경우"
    if cur_jeol is not None and cur_hang is not None:
        bare_letter_pat = re.compile(r'[“"]([가나다라마바사아자차카타파하])[”"]')
        for m in bare_letter_pat.finditer(text):
            if any(s <= m.start() < e for s, e in consumed):
                continue
            letter = m.group(1)
            if letter == cur_pyg_ho:
                continue  # 자기 자신 제외
            refs.append(lookup(registry, current_prefix, cur_jeol, None, cur_hang, letter))

    # 4. "제N조(의M)제K항 각 호" / "제N조 각 호" — 명시적으로 다른 조항의 모든 호 참조
    each_ho_pat = re.compile(r'제(\d+)조(?:의(\d+))?(?:제(\d+)항)?\s*각\s*호')
    for m in each_ho_pat.finditer(text):
        ref_jo    = int(m.group(1))
        ref_jo_ui = int(m.group(2)) if m.group(2) else None
        ref_hang  = int(m.group(3)) if m.group(3) else None
        refs.extend(lookup_all_ho(registry, current_prefix, ref_jo, ref_jo_ui, ref_hang))

    seen: set[str] = set()
    return [r for r in refs if not (r in seen or seen.add(r))]  # type: ignore


# ─────────────────────────────────────────
# 메인 (2패스)
# ─────────────────────────────────────────
def main():
    files = [f for f in LAW_DIR.glob("*.docx") if not f.name.startswith("~$")]
    if not files:
        print(f"[ERROR] {LAW_DIR} 에 파일이 없습니다.")
        return

    # ── Pass 1: 파싱 + 레지스트리 구축 ──
    print("=== Pass 1: 파싱 ===")
    parsed: list[tuple[Path, list[dict], str, dict]] = []
    registry: dict[str, dict] = {}

    for path in sorted(files):
        filename = path.stem
        meta = find_meta(filename)
        if meta is None:
            print(f"[SKIP] 메타 없음: {filename}")
            continue

        print(f"[PARSE] {filename}")
        paragraphs = read_docx(path)
        prefix     = DOC_TYPE_TO_PREFIX.get(meta["document_type"], "UNK")
        law_name   = meta["law_name"]

        if prefix == "PYG":
            articles = parse_pyg(paragraphs, prefix=prefix, law_name=law_name)
        else:
            articles = parse_law(paragraphs, prefix=prefix, law_name=law_name)

        articles = [tag_article(a, meta["is_ref_article_doc"]) for a in articles]
        articles = split_direct_ho(articles, prefix, law_name)
        registry[prefix] = build_registry(articles)
        parsed.append((path, articles, prefix, meta))
        print(f"  -> {len(articles)} chunk, registry {len(registry[prefix])}")

    # ── Pass 2: cross_refs 추출 + 저장 ──
    print("\n=== Pass 2: cross_refs ===")
    for path, articles, prefix, meta in parsed:
        print(f"[REFS] {path.stem}")

        parent_to_children: dict[str, list[str]] = {}
        for a in articles:
            p = a.get("parent_chunk_id")
            if p and ("호" in a.get("hierarchy", {}) or "항" in a.get("hierarchy", {})):
                parent_to_children.setdefault(p, []).append(a["chunk_id"])

        for article in articles:
            refs = extract_cross_refs(
                article.get("text", ""), prefix, registry, article
            )
            text = article.get("text", "")
            if re.search(r'다음\s*각\s*(?:호|항)', text):
                children = parent_to_children.get(article["chunk_id"], [])
                for c in children:
                    if c not in refs:
                        refs.append(c)
            article["cross_refs"] = refs

        result = {
            "document_type":     meta["document_type"],
            "law_name":          meta["law_name"],
            "source":            meta["source"],
            "filename":          path.name,
            "total_articles":    len(articles),
            "ref_article_count": sum(1 for a in articles if a.get("is_ref_article")),
            "upper_law_count":   sum(1 for a in articles if a.get("is_upper_law")),
            "articles":          articles,
        }

        out_path = OUTPUT_DIR / f"{path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        ref_count = sum(len(a.get("cross_refs", [])) for a in articles)
        print(f"  -> saved: {out_path} | cross_refs total {ref_count}")

    print("\nDone! output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
