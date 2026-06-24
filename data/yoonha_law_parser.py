"""
Workit - 법령 문서 파싱 스크립트
input : C:/project/Workit/data/law/ 내 docx 파일
output: C:/project/Workit/data/structured/ 내 JSON 파일

사용법:
    pip install python-docx
    python yoonha_law_parser.py

지원 법령 (2026-06 기준):
    LCA    지방계약법
    LCAE   지방계약법 시행령
    LCAR   지방계약법 시행규칙
    SWPA   소프트웨어 진흥법
    SWPAE  소프트웨어 진흥법 시행령        ← 신규
    LARA   지방회계법
    LARAE  지방회계법 시행령
    PYG    지방자치단체 용역계약 일반조건 (예규367호)
    PPMA   공유재산법
    PPMAE  공유재산법 시행령               ← 신규
    PIPA   개인정보보호법                  ← 신규
    PIPAE  개인정보보호법 시행령           ← 신규
"""

import re
import json

from pathlib import Path
from docx import Document

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
LAW_DIR    = Path("C:/project/Workit/data/law")
OUTPUT_DIR = Path("C:/project/Workit/data/structured")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# REF_ARTICLE & UPPER_LAW (필터링 기준)
# 계약서 조항과 매핑되는 법령 조문 번호 목록
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

# 상위법 참조 chunk_id 세트 — PYG(용역계약 일반조건) 텍스트에서
# "법/시행령/시행규칙 제N조" 패턴을 자동 추출하여 생성한 목록.
# 조문 번호 문자열 부분 일치 대신 chunk_id 직접 비교로 태깅 정확도 향상.
UPPER_LAW_IDS = {
    # LCA (지방계약법)
    "LCA_6",        "LCA_6_1",
    "LCA_6_의2",
    "LCA_7_1",
    "LCA_8",
    "LCA_15_3",
    "LCA_16",       "LCA_17",
    "LCA_25",
    "LCA_28",
    "LCA_30",       "LCA_31",       "LCA_31_1_3",
    "LCA_34",       "LCA_34_의2",
    "LCA_43",
    # LCAE (지방계약법 시행령)
    "LCAE_3",       "LCAE_5",
    "LCAE_15",      "LCAE_15_1",    "LCAE_15_6",
    "LCAE_15_7_1",  "LCAE_15_7_2",
    "LCAE_19_1",
    "LCAE_26_1",
    "LCAE_30",
    "LCAE_35",
    "LCAE_37",      "LCAE_37_2_1",  "LCAE_37_2_2",
    "LCAE_42",
    "LCAE_50",
    "LCAE_51",      "LCAE_51_1_2",
    "LCAE_53",      "LCAE_53_2",
    "LCAE_54",
    "LCAE_56_1_2",
    "LCAE_59",
    "LCAE_64_1",    "LCAE_64_의2",
    "LCAE_69",
    "LCAE_71",      "LCAE_71_의3",
    "LCAE_73",      "LCAE_73_6",    "LCAE_73_8",
    "LCAE_74",      "LCAE_74_1",    "LCAE_74_7",
    "LCAE_75",      "LCAE_75_2",    "LCAE_75_의2",
    "LCAE_78",      "LCAE_78_의2",
    "LCAE_88_1",
    "LCAE_92_2_1",
    "LCAE_94",
    "LCAE_96",
    "LCAE_98",
    "LCAE_103",
    "LCAE_126",     "LCAE_127",
    "LCAE_132",
    # LCAR (지방계약법 시행규칙)
    "LCAR_2",
    "LCAR_23_의2",
    "LCAR_65",
    "LCAR_68",
    "LCAR_70",
    "LCAR_72",      "LCAR_72_7",
}

# ─────────────────────────────────────────
# 파일명 → document_type 매핑
# key: 파일명에 포함된 문자열 (부분 일치)
# is_ref_article_doc: 용역계약 일반조건처럼 절/항/호 구조인 경우 True
# ─────────────────────────────────────────
FILE_META = {
    # 용역계약 일반조건 — 띄어쓰기/언더스코어 두 가지 파일명 모두 대응
    "지방자치단체 용역계약 일반조건 (행안부 예규)": {
        "document_type": "지방자치단체 용역계약 일반조건",
        "source": "행정안전부 예규",
        "is_ref_article_doc": True,
    },
    "지방자치단체_용역계약_일반조건__행안부_예규_": {
        "document_type": "지방자치단체 용역계약 일반조건",
        "source": "행정안전부 예규",
        "is_ref_article_doc": True,
    },
    # 지방계약법 — 시행규칙/시행령을 본법보다 먼저 매핑
    "지방계약법_시행규칙": {
        "document_type": "지방계약법 시행규칙",
        "source": "행정안전부령",
        "is_ref_article_doc": False,
    },
    "지방계약법_시행령": {
        "document_type": "지방계약법 시행령",
        "source": "대통령령",
        "is_ref_article_doc": False,
    },
    "지방계약법": {
        "document_type": "지방계약법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
    # 소프트웨어 진흥법 — 띄어쓰기/언더스코어 두 가지 파일명 모두 대응
    "소프트웨어 진흥법 시행령": {
        "document_type": "소프트웨어 진흥법 시행령",
        "source": "대통령령",
        "is_ref_article_doc": False,
    },
    "소프트웨어_진흥법_시행령": {
        "document_type": "소프트웨어 진흥법 시행령",
        "source": "대통령령",
        "is_ref_article_doc": False,
    },
    "소프트웨어 진흥법": {
        "document_type": "소프트웨어 진흥법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
    "소프트웨어_진흥법": {
        "document_type": "소프트웨어 진흥법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
    # 지방회계법
    "지방회계법_시행령": {
        "document_type": "지방회계법 시행령",
        "source": "대통령령",
        "is_ref_article_doc": False,
    },
    "지방회계법": {
        "document_type": "지방회계법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
    # 공유재산법 — 띄어쓰기/언더스코어 두 가지 파일명 모두 대응
    "공유재산 및 물품 관리법 시행령": {
        "document_type": "공유재산법 시행령",
        "source": "대통령령",
        "is_ref_article_doc": False,
    },
    "공유재산_및_물품_관리법_시행령": {
        "document_type": "공유재산법 시행령",
        "source": "대통령령",
        "is_ref_article_doc": False,
    },
    "공유재산법": {
        "document_type": "공유재산법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
    # 개인정보보호법 — 시행령을 본법보다 먼저 매핑 (부분 일치 오류 방지)
    # 띄어쓰기/언더스코어 두 가지 파일명 모두 대응
    "개인정보 보호법 시행령": {
        "document_type": "개인정보보호법 시행령",
        "source": "대통령령",
        "is_ref_article_doc": False,
    },
    "개인정보_보호법_시행령": {
        "document_type": "개인정보보호법 시행령",
        "source": "대통령령",
        "is_ref_article_doc": False,
    },
    "개인정보 보호법": {
        "document_type": "개인정보보호법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
    "개인정보_보호법": {
        "document_type": "개인정보보호법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
}

# ─────────────────────────────────────────
# 법령약자 매핑 (chunk_id 생성용)
# document_type → prefix
# ─────────────────────────────────────────
DOC_TYPE_TO_PREFIX = {
    "지방계약법":                    "LCA",
    "지방계약법 시행령":              "LCAE",
    "지방계약법 시행규칙":            "LCAR",
    "소프트웨어 진흥법":              "SWPA",
    "소프트웨어 진흥법 시행령":       "SWPAE",   # 신규
    "지방회계법":                    "LARA",
    "지방회계법 시행령":              "LARAE",
    "지방자치단체 용역계약 일반조건":  "PYG",
    "공유재산법":                    "PPMA",
    "공유재산법 시행령":              "PPMAE",   # 신규
    "개인정보보호법":                 "PIPA",    # 신규
    "개인정보보호법 시행령":          "PIPAE",   # 신규
}

# ─────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────

def read_docx(path: Path) -> list[tuple[str, str]]:
    """
    docx 파일을 읽어 (타입, 텍스트) 튜플 리스트로 반환.
    타입: 'p' = 일반 단락, 'tbl' = 표 셀
    """
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
    """
    파일명에 부분 일치하는 FILE_META 키를 찾아 메타 반환.
    키 길이 내림차순 정렬로 더 구체적인 키가 먼저 매핑되도록 함.
    (예: '지방계약법_시행령'이 '지방계약법'보다 먼저 매핑)

    파일명 앞에 타임스탬프(숫자_) 형태의 prefix가 붙어있는 경우
    제거 후 매핑 시도.
    예: '1782288864349_개인정보_보호법_시행령' → '개인정보_보호법_시행령'
    """
    # 타임스탬프 prefix 제거 (숫자로 시작하면)
    clean = re.sub(r"^\d+_", "", filename)

    for key in sorted(FILE_META.keys(), key=len, reverse=True):
        if key in clean or key in filename:
            return FILE_META[key]
    return None


def make_article_id(article_number: str) -> str:
    """조문 번호에서 특수문자를 언더스코어로 치환."""
    return re.sub(r"[\s/·]", "_", article_number).strip("_")


def _strip_comments(text: str) -> str:
    """
    <개정 2013. 3. 23.>, [전문개정 ...] 같은 주석 제거.
    호 분리 정규식 적용 전에 날짜 숫자가 호 번호로 오인되는 버그 방지.
    """
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[[^\]]+\]', '', text)
    return text


def make_chunk_id(
    prefix: str,
    jo: int,
    hang: int | None = None,
    ho: int | None = None,
    jo_ui: int | None = None,
) -> str:
    """
    chunk_id 생성 규칙:
        {PREFIX}_{조}[_의{조의N}][_{항}][_{호}]

    예시:
        LCA_30          → 지방계약법 제30조
        LCA_30_4        → 지방계약법 제30조 제4항
        LCAE_64_의2     → 지방계약법 시행령 제64조의2
        LCAE_64_의2_1   → 지방계약법 시행령 제64조의2 제1항
        LCA_30_4_1      → 지방계약법 제30조 제4항 제1호
    """
    jo_part = str(jo) + (f"_의{jo_ui}" if jo_ui is not None else "")
    parts = [jo_part]
    if hang is not None:
        parts.append(str(hang))
    if ho is not None:
        parts.append(str(ho))
    return f"{prefix}_{'_'.join(parts)}"


# ─────────────────────────────────────────
# 파서 A: 용역계약 일반조건 (절/항/호 구조)
# 9장 이전: tbl=절, p=항·호
# 9장 이후: p에 장·절·항·호 모두 포함
# ─────────────────────────────────────────

def parse_pyg(lines: list[tuple[str, str]], prefix: str = "PYG") -> list[dict]:
    """
    용역계약 일반조건 전용 파서.

    chunk_id 형식: PYG_{장}_{절}_{항}[_{호}]
        예: PYG_9_7_5_가  (9장 7절 5항 가목)
            PYG_3_1       (3절 1항, 장 없음)

    구조:
        - 장(章): '제N장' — p 또는 tbl
        - 절(節): '제N절' — 9장 이전은 tbl만, 9장 이후는 p도 허용
        - 항(項): 'N.'    — p만
        - 호(號): '가.'   — p만
    """
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

        prefix_str = f"제{cur_chapter}장 " if cur_chapter else ""

        if cur_item:
            an        = prefix_str + f"제{cur_section}절 제{cur_clause}항 {cur_item}"
            hierarchy = {"절": f"제{cur_section}절", "항": f"제{cur_clause}항", "호": cur_item}
        else:
            an        = prefix_str + f"제{cur_section}절 제{cur_clause}항"
            hierarchy = {"절": f"제{cur_section}절", "항": f"제{cur_clause}항"}

        if cur_chapter:
            hierarchy["장"] = f"제{cur_chapter}장"

        id_parts = ([cur_chapter] if cur_chapter else []) + [cur_section, cur_clause]
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
        sm  = section_pat.match(text) if not chm and (typ == "tbl" or cur_chapter) else None
        cm  = clause_pat.match(text)  if typ == "p" and not chm and not sm else None
        im  = item_pat.match(text)    if typ == "p" and not chm and not sm and not cm else None

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
# 파서 B: 일반 법령 (조/항/호 구조)
# 지방계약법, 소프트웨어 진흥법, 공유재산법,
# 개인정보보호법 등 표준 법령 구조에 사용
# ─────────────────────────────────────────

def parse_law(lines: list[tuple[str, str]], prefix: str) -> list[dict]:
    """
    표준 법령 파서. 조(條) → 항(項) → 호(號) 단위로 청크 분리.

    chunk_id 형식: {PREFIX}_{조}[_의N][_{항}][_{호}]
        예: LCA_30        → 지방계약법 제30조 (단항)
            LCA_30_4      → 지방계약법 제30조 제4항
            LCA_30_4_1    → 지방계약법 제30조 제4항 제1호
            LCAE_64_의2_1 → 지방계약법 시행령 제64조의2 제1항

    항 구분자: ① ② ③ ... (원문자)
    호 구분자: '1. 2. 3. ...' (숫자+점, 정규식으로 분리)
    부칙은 파싱 제외.
    """
    article_pat = re.compile(r"^(제\s*\d+\s*조(?:의\s*\d+)?)\s*[(\[〔]?([^)\]\)〕\n]*)[)\]\)〕]?")

    HANG_MAP = {c: i + 1 for i, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮")}

    # ── 1단계: 조 단위로 묶기 ──────────────────
    raw_articles: list[dict] = []
    # seen_jo: 동일 조문이 문서 내에 두 번 이상 등장하는 경우(예: 개정 예정 조문이
    # 현행 조문과 함께 수록된 경우) 첫 번째 등장(현행)만 파싱에 사용하고 이후는 스킵.
    seen_jo: set[tuple] = set()
    cur_jo    = None
    cur_jo_ui = None
    cur_title = ""
    buf: list[str] = []

    def flush_jo():
        if cur_jo is not None and buf:
            key = (cur_jo, cur_jo_ui)
            if key not in seen_jo:
                seen_jo.add(key)
                raw_articles.append({
                    "jo":    cur_jo,
                    "jo_ui": cur_jo_ui,
                    "title": cur_title,
                    "text":  " ".join(buf),
                })

    bujik_pat = re.compile(r"^부\s*칙")
    in_bujik  = False

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

    # ── 2단계: 조 → 항 → 호 단위로 분리 + parent 청크 생성 ──────
    # Hierarchical RAG 구조:
    #   parent 청크 (is_parent=True) : 조 전체 텍스트 합본, 조 단위 인용 매핑용
    #   child  청크 (is_parent=False): 항/호 단위 실제 검색 대상
    # retrieval 시 child hit → parent_id로 형제 청크 함께 fetch
    articles: list[dict] = []

    for raw in raw_articles:
        jo    = raw["jo"]
        jo_ui = raw["jo_ui"]
        text  = raw["text"]
        jo_str    = f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "")
        parent_id = make_chunk_id(prefix, jo, jo_ui=jo_ui)

        hang_splits = re.split(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])", text)

        if len(hang_splits) <= 1:
            # 항 구분자 없는 단항 조문 — parent/child 구분 없이 단일 청크
            articles.append({
                "chunk_id":       parent_id,
                "article_id":     jo_str,
                "article_number": jo_str,
                "title":          raw["title"],
                "text":           text,
                "is_parent":      False,   # 단항이므로 parent 별도 불필요
                "parent_id":      None,
                "hierarchy":      {"조": jo_str},
            })
            continue

        # 다항 조문 — parent 청크(조 전체) 먼저 추가
        articles.append({
            "chunk_id":       parent_id,
            "article_id":     jo_str,
            "article_number": jo_str,
            "title":          raw["title"],
            "text":           text,        # 조 전체 원문
            "is_parent":      True,
            "parent_id":      None,
            "hierarchy":      {"조": jo_str},
        })

        # child 청크(항/호 단위) 추가
        i = 1
        while i < len(hang_splits) - 1:
            hang_char = hang_splits[i]
            hang_text = hang_splits[i + 1].strip() if i + 1 < len(hang_splits) else ""
            hang_num  = HANG_MAP.get(hang_char, i)

            hang_text_clean = _strip_comments(hang_text)
            ho_splits = re.split(r"\s(\d{1,2})\.\s", hang_text_clean)

            if len(ho_splits) <= 1:
                chunk_id = make_chunk_id(prefix, jo, hang=hang_num, jo_ui=jo_ui)
                articles.append({
                    "chunk_id":       chunk_id,
                    "article_id":     jo_str + f"제{hang_num}항",
                    "article_number": jo_str + f"제{hang_num}항",
                    "title":          raw["title"],
                    "text":           hang_char + hang_text,
                    "is_parent":      False,
                    "parent_id":      parent_id,
                    "hierarchy": {
                        "조": jo_str,
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
                        "article_id":     jo_str + f"제{hang_num}항제{ho_num}호",
                        "article_number": jo_str + f"제{hang_num}항제{ho_num}호",
                        "title":          raw["title"],
                        "text":           f"{hang_char} {ho_splits[0].strip()} {ho_num}. {ho_text}",
                        "is_parent":      False,
                        "parent_id":      parent_id,
                        "hierarchy": {
                            "조": jo_str,
                            "항": f"제{hang_num}항",
                            "호": f"제{ho_num}호",
                        },
                    })
                    j += 2

            i += 2

    return articles


# ─────────────────────────────────────────
# 필터링 — is_ref_article / is_upper_law 태깅
# ─────────────────────────────────────────

def tag_article(article: dict, is_ref_doc: bool) -> dict:
    """
    각 청크에 두 가지 플래그 추가:
        is_ref_article: 용역계약 일반조건의 핵심 조항 여부
        is_upper_law:   상위법 직접 참조 조문 여부
    """
    an = article.get("article_number", "")
    article["is_ref_article"] = is_ref_doc and any(ref in an for ref in REF_ARTICLE)
    article["is_upper_law"]   = article.get("chunk_id", "") in UPPER_LAW_IDS
    return article


# ─────────────────────────────────────────
# 파일 단위 처리
# ─────────────────────────────────────────

def process_file(path: Path):
    filename = path.stem
    # Word 임시 잠금 파일 (~$...) 스킵
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
        chap_pat = re.compile(r"제\s*\d+\s*장")
        for i, (typ, text) in enumerate(paragraphs):
            if chap_pat.search(text):
                print(f"  [DEBUG] 장 감지 typ={typ!r} text={text!r}")
        articles = parse_pyg(paragraphs, prefix=prefix)
    else:
        articles = parse_law(paragraphs, prefix=prefix)

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

    out_path = OUTPUT_DIR / f"{filename}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  → 저장: {out_path}")
    print(f"  → 전체 조문: {len(articles)}개 | REF: {result['ref_article_count']}개 | UPPER: {result['upper_law_count']}개")
    print(f"  → chunk_id 샘플: {[a['chunk_id'] for a in articles[:5]]}")


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

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