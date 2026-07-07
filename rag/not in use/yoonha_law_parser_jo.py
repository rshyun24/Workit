"""
Workit - 법령 문서 파싱 스크립트 (조 단위)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
    법령 docx 파일을 읽어 조(條) 단위 청크 JSON으로 변환한다.
    Hierarchical RAG에서 child hit 후 parent 텍스트 fetch용.

출력:
    data/structured/jo/{법령명}_jo.json

chunk_id 규칙:
    {PREFIX}_{조}[_의N]
    예: LCA_30, LCAE_64_의2

    PYG(용역계약 일반조건)는 조 구조가 없으므로
    절/항 단위를 조 단위 대용으로 사용.
"""

import re
import json
from pathlib import Path
from docx import Document

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 경로 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAW_DIR    = Path("C:/project/Workit/data/law")
OUTPUT_DIR = Path("C:/project/Workit/data/structured/jo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REF_ARTICLE / UPPER_LAW_IDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REF_ARTICLE = [
    "제7절 제1항 가", "제8절 제4항 나", "제6절 제1항 가",
    "제6절 제1항 라", "제6절 제1항 마", "제7절 제4항 다",
    "제7절 제5항 가", "제8절 제7항 가", "제59조", "제75조",
]

UPPER_LAW_IDS = {
    "LCA_6", "LCA_6_1", "LCA_6_의2", "LCA_7_1", "LCA_8",
    "LCA_15_3", "LCA_16", "LCA_17", "LCA_25", "LCA_28",
    "LCA_30", "LCA_31", "LCA_31_1_3", "LCA_34", "LCA_34_의2", "LCA_43",
    "LCAE_3", "LCAE_5", "LCAE_15", "LCAE_15_1", "LCAE_15_6",
    "LCAE_15_7_1", "LCAE_15_7_2", "LCAE_19_1", "LCAE_26_1",
    "LCAE_30", "LCAE_35", "LCAE_37", "LCAE_37_2_1", "LCAE_37_2_2",
    "LCAE_42", "LCAE_50", "LCAE_51", "LCAE_51_1_2", "LCAE_53",
    "LCAE_53_2", "LCAE_54", "LCAE_56_1_2", "LCAE_59",
    "LCAE_64_1", "LCAE_64_의2", "LCAE_69", "LCAE_71", "LCAE_71_의3",
    "LCAE_73", "LCAE_73_6", "LCAE_73_8", "LCAE_74", "LCAE_74_1",
    "LCAE_74_7", "LCAE_75", "LCAE_75_2", "LCAE_75_의2",
    "LCAE_78", "LCAE_78_의2", "LCAE_88_1", "LCAE_92_2_1",
    "LCAE_94", "LCAE_96", "LCAE_98", "LCAE_103", "LCAE_126",
    "LCAE_127", "LCAE_132",
    "LCAR_2", "LCAR_23_의2", "LCAR_65", "LCAR_68", "LCAR_70",
    "LCAR_72", "LCAR_72_7",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILE_META / DOC_TYPE_TO_PREFIX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE_META = {
    "지방자치단체 용역계약 일반조건 (행안부 예규)": {
        "document_type": "지방자치단체 용역계약 일반조건", "source": "행정안전부 예규", "is_ref_article_doc": True,
    },
    "지방자치단체_용역계약_일반조건__행안부_예규_": {
        "document_type": "지방자치단체 용역계약 일반조건", "source": "행정안전부 예규", "is_ref_article_doc": True,
    },
    "지방계약법_시행규칙": {"document_type": "지방계약법 시행규칙", "source": "행정안전부령", "is_ref_article_doc": False},
    "지방계약법_시행령":   {"document_type": "지방계약법 시행령",   "source": "대통령령",    "is_ref_article_doc": False},
    "지방계약법":          {"document_type": "지방계약법",          "source": "법률",        "is_ref_article_doc": False},
    "소프트웨어 진흥법 시행령":  {"document_type": "소프트웨어 진흥법 시행령", "source": "대통령령", "is_ref_article_doc": False},
    "소프트웨어_진흥법_시행령":  {"document_type": "소프트웨어 진흥법 시행령", "source": "대통령령", "is_ref_article_doc": False},
    "소프트웨어 진흥법":   {"document_type": "소프트웨어 진흥법",   "source": "법률",        "is_ref_article_doc": False},
    "소프트웨어_진흥법":   {"document_type": "소프트웨어 진흥법",   "source": "법률",        "is_ref_article_doc": False},
    "지방회계법_시행령":   {"document_type": "지방회계법 시행령",   "source": "대통령령",    "is_ref_article_doc": False},
    "지방회계법":          {"document_type": "지방회계법",          "source": "법률",        "is_ref_article_doc": False},
    "공유재산 및 물품 관리법 시행령": {"document_type": "공유재산법 시행령", "source": "대통령령", "is_ref_article_doc": False},
    "공유재산_및_물품_관리법_시행령": {"document_type": "공유재산법 시행령", "source": "대통령령", "is_ref_article_doc": False},
    "공유재산법":          {"document_type": "공유재산법",          "source": "법률",        "is_ref_article_doc": False},
    "개인정보 보호법 시행령":  {"document_type": "개인정보보호법 시행령", "source": "대통령령", "is_ref_article_doc": False},
    "개인정보_보호법_시행령":  {"document_type": "개인정보보호법 시행령", "source": "대통령령", "is_ref_article_doc": False},
    "개인정보 보호법":     {"document_type": "개인정보보호법",      "source": "법률",        "is_ref_article_doc": False},
    "개인정보_보호법":     {"document_type": "개인정보보호법",      "source": "법률",        "is_ref_article_doc": False},
}

DOC_TYPE_TO_PREFIX = {
    "지방계약법":                    "LCA",
    "지방계약법 시행령":              "LCAE",
    "지방계약법 시행규칙":            "LCAR",
    "소프트웨어 진흥법":              "SWPA",
    "소프트웨어 진흥법 시행령":       "SWPAE",
    "지방회계법":                    "LARA",
    "지방회계법 시행령":              "LARAE",
    "지방자치단체 용역계약 일반조건":  "PYG",
    "공유재산법":                    "PPMA",
    "공유재산법 시행령":              "PPMAE",
    "개인정보보호법":                 "PIPA",
    "개인정보보호법 시행령":          "PIPAE",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def read_docx(path: Path) -> list[tuple[str, str]]:
    from docx.oxml.ns import qn
    from docx.table import Table as DocxTable
    doc   = Document(str(path))
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
    clean = re.sub(r"^\d+_", "", filename)
    for key in sorted(FILE_META.keys(), key=len, reverse=True):
        if key in clean or key in filename:
            return FILE_META[key]
    return None


def make_article_id(article_number: str) -> str:
    return re.sub(r"[\s/·]", "_", article_number).strip("_")


def _strip_comments(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[[^\]]+\]', '', text)
    return text


def tag_article(article: dict, is_ref_doc: bool) -> dict:
    an = article.get("article_number", "")
    article["is_ref_article"] = is_ref_doc and any(ref in an for ref in REF_ARTICLE)
    article["is_upper_law"]   = article.get("chunk_id", "") in UPPER_LAW_IDS
    return article


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파서: 용역계약 일반조건 (PYG) — 절/항 단위 조 대용
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_pyg_jo(lines: list[tuple[str, str]], prefix: str = "PYG") -> list[dict]:
    """
    PYG는 조 구조가 없으므로 절(節) 단위를 조 단위 대용으로 사용.
    각 절의 전체 텍스트를 하나의 청크로 묶는다.
    """
    articles    = []
    cur_chapter = None
    cur_section = None
    buf: list[str] = []

    chapter_pat = re.compile(r"^제\s*(\d+)\s*장")
    section_pat = re.compile(r"^제\s*(\d+)\s*절")

    def flush():
        nonlocal buf
        if not buf or not cur_section:
            buf = []; return
        prefix_str = f"제{cur_chapter}장 " if cur_chapter else ""
        an         = prefix_str + f"제{cur_section}절"
        id_parts   = ([cur_chapter] if cur_chapter else []) + [cur_section]
        hierarchy  = {"절": f"제{cur_section}절"}
        if cur_chapter:
            hierarchy["장"] = f"제{cur_chapter}장"

        articles.append({
            "chunk_id":        f"{prefix}_{'_'.join(id_parts)}",
            "article_id":      make_article_id(an),
            "article_number":  an,
            "text":            " ".join(buf),
            "hierarchy":       hierarchy,
            "parent_chunk_id": None,
        })
        buf = []

    for typ, text in lines:
        chm = chapter_pat.match(text)
        sm  = section_pat.match(text) if not chm and (typ == "tbl" or cur_chapter) else None

        if chm:
            flush(); cur_chapter = chm.group(1); cur_section = None
        elif sm:
            flush(); cur_section = sm.group(1)
        elif cur_section:
            buf.append(text)

    flush()
    return articles


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파서: 일반 법령 — 조 단위 전체 텍스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_law_jo(lines: list[tuple[str, str]], prefix: str) -> list[dict]:
    """조(條) 전체 원문을 하나의 청크로 묶는다."""
    article_pat = re.compile(
        r"^(제\s*\d+\s*조(?:의\s*\d+)?)\s*[(\[〔]?([^)\]\)〕\n]*)[)\]\)〕]?"
    )

    raw_articles: list[dict] = []
    seen_jo: set[tuple] = set()
    cur_jo = cur_jo_ui = None
    cur_title = ""
    buf: list[str] = []

    def flush_jo():
        if cur_jo is not None and buf:
            key = (cur_jo, cur_jo_ui)
            if key not in seen_jo:
                seen_jo.add(key)
                raw_articles.append({
                    "jo": cur_jo, "jo_ui": cur_jo_ui,
                    "title": cur_title, "text": " ".join(buf),
                })

    bujik_pat = re.compile(r"^부\s*칙")
    in_bujik  = False

    for _, text in lines:
        if bujik_pat.match(text):
            in_bujik = True; flush_jo(); cur_jo = None; buf = []; continue
        if in_bujik:
            continue
        m = article_pat.match(text)
        if m:
            flush_jo(); buf = [text]
            raw_jo_str = re.sub(r"\s+", "", m.group(1))
            jo_m = re.match(r"제(\d+)조(?:의(\d+))?", raw_jo_str)
            cur_jo    = int(jo_m.group(1)) if jo_m else None
            cur_jo_ui = int(jo_m.group(2)) if jo_m and jo_m.group(2) else None
            cur_title = m.group(2).strip() if m.group(2) else ""
        else:
            buf.append(text)

    flush_jo()

    articles: list[dict] = []
    for raw in raw_articles:
        jo    = raw["jo"]
        jo_ui = raw["jo_ui"]
        jo_str = f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "")
        jo_part = str(jo) + (f"_의{jo_ui}" if jo_ui else "")
        chunk_id = f"{prefix}_{jo_part}"

        articles.append({
            "chunk_id":        chunk_id,
            "article_id":      jo_str,
            "article_number":  jo_str,
            "title":           raw["title"],
            "text":            raw["text"],
            "parent_chunk_id": None,   # 조 단위는 parent 없음
            "hierarchy":       {"조": jo_str},
        })

    return articles


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파일 처리 & 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_file(path: Path):
    filename = path.stem
    if filename.startswith("~$"):
        return

    meta = find_meta(filename)
    if meta is None:
        print(f"[SKIP] 메타 없음: {filename}")
        return

    print(f"[PARSE] {filename}")
    paragraphs = read_docx(path)
    prefix     = DOC_TYPE_TO_PREFIX.get(meta["document_type"], "UNK")

    if prefix == "PYG":
        articles = parse_pyg_jo(paragraphs, prefix=prefix)
    else:
        articles = parse_law_jo(paragraphs, prefix=prefix)

    articles = [tag_article(a, meta["is_ref_article_doc"]) for a in articles]

    result = {
        "document_type":     meta["document_type"],
        "source":            meta["source"],
        "filename":          path.name,
        "total_articles":    len(articles),
        "ref_article_count": sum(1 for a in articles if a.get("is_ref_article")),
        "upper_law_count":   sum(1 for a in articles if a.get("is_upper_law")),
        "articles":          articles,
    }

    out_path = OUTPUT_DIR / f"{filename}_jo.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  → 저장: {out_path} ({len(articles)}개)")
    print(f"  → chunk_id 샘플: {[a['chunk_id'] for a in articles[:5]]}")


def main():
    files = list(LAW_DIR.glob("*.docx"))
    if not files:
        print(f"[ERROR] {LAW_DIR} 에 .docx 파일이 없습니다.")
        return
    for f in sorted(files):
        process_file(f)
    print("\n✅ 완료! 결과물:", OUTPUT_DIR)


if __name__ == "__main__":
    main()