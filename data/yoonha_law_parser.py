"""
Workit - 법령 문서 파싱 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
    공공 소프트웨어 조달 계약서 검토 시스템(Workit)의 RAG 파이프라인 첫 번째 단계.
    법령 docx 파일을 읽어 조·항·호 단위 JSON 청크로 변환한다.

실행 흐름:
    docx 파일 (data/law/)
        ↓  read_docx()       — 단락·표 셀을 (타입, 텍스트) 리스트로 추출
        ↓  parse_law()       — 조 → 항 → 호 단위 청크 분리
           parse_pyg()       — 용역계약 일반조건 전용 (절 → 항 → 목 구조)
        ↓  tag_article()     — is_ref_article / is_upper_law 플래그 태깅
        ↓  process_file()    — _jo.json / _ho.json 두 파일로 분리 저장

출력 파일 두 종류:
    {법령명}_jo.json    — 조(條) 단위 parent 청크 (Hierarchical RAG에서 맥락 제공용)
    {법령명}_ho.json  — 호 단위 child 청크 (실제 벡터 검색 대상)

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

chunk_id 규칙:
    {PREFIX}_{조}[_의N][_{항}][_{호}]

    일반 예시:
        LCA_30          → 지방계약법 제30조 (단항, parent 없음)
        LCA_30_4        → 지방계약법 제30조 제4항
        LCA_30_4_1      → 지방계약법 제30조 제4항 제1호
        LCAE_64_의2_1   → 지방계약법 시행령 제64조의2 제1항

    ★ 항 없이 바로 호가 나오는 경우 → 항을 0으로 고정:
        LCA_30_0_1      → 지방계약법 제30조 제1호 (항 구분자 없음)
        (항=1인 LCA_30_1_1 과 구별하기 위해 0을 명시)
"""

import re
import json

from pathlib import Path
from docx import Document

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 경로 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAW_DIR        = Path("C:/project/Workit/data/law")              # 원본 docx 위치
OUTPUT_DIR_JO  = Path("C:/project/Workit/data/structured/jo")    # 조 단위 JSON 저장 위치
OUTPUT_DIR_HO= Path("C:/project/Workit/data/structured/ho")  # 호 단위 JSON 저장 위치
OUTPUT_DIR_JO.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_HO.mkdir(parents=True, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REF_ARTICLE — 용역계약 일반조건 핵심 조항 목록
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 계약서 검토 시드(seed)의 ref_answer와 매핑되는 PYG 조항 번호.
# tag_article()에서 is_ref_article=True 로 태깅하는 기준이 된다.
# → Qdrant 필터링 시 핵심 조항만 우선 검색하는 데 사용.
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UPPER_LAW_IDS — PYG가 직접 인용하는 상위법 chunk_id 세트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 용역계약 일반조건(PYG) 본문에서 "법/시행령/시행규칙 제N조" 형태로
# 명시적으로 참조되는 상위법 조문의 chunk_id를 수동으로 정리한 목록.
#
# 사용 목적:
#   - tag_article()에서 is_upper_law=True 로 태깅
#   - Qdrant 검색 시 상위법 조문을 우선 fetch하는 필터로 활용
#   - 조문 번호 문자열 부분 일치 대신 chunk_id 직접 비교 → 태깅 정확도 향상
#
# 관리 방법:
#   PYG 문서 개정 시 참조 조문이 바뀌면 이 세트를 함께 업데이트할 것.
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILE_META — 파일명 → 법령 메타데이터 매핑
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# key   : 파일명(stem)에 포함된 문자열 (부분 일치)
# value :
#   document_type      — 법령 공식 명칭 (DOC_TYPE_TO_PREFIX 키와 1:1 매핑)
#   source             — 법령 형식 (법률 / 대통령령 / 행정안전부령 / 예규)
#   is_ref_article_doc — True면 REF_ARTICLE 목록으로 is_ref_article 태깅 활성화
#                        (용역계약 일반조건만 True, 나머지 법령은 False)
#
# 주의:
#   키 길이 내림차순으로 정렬해 매핑하므로, 더 구체적인 키가 먼저 매핑됨.
#   예) '지방계약법_시행령'이 '지방계약법'보다 먼저 매핑되어 오탐 방지.
#   같은 법령을 띄어쓰기/언더스코어 두 가지 파일명으로 모두 등록한 이유도 동일.
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
    # 지방계약법 — 시행규칙/시행령을 본법보다 먼저 등록 (부분 일치 오탐 방지)
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
    # 개인정보보호법 — 시행령을 본법보다 먼저 등록 (부분 일치 오탐 방지)
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DOC_TYPE_TO_PREFIX — 법령명 → chunk_id 접두어 매핑
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# document_type(FILE_META의 값)을 키로 받아 chunk_id 접두어(PREFIX)를 반환.
# chunk_id = f"{PREFIX}_{조}[_의N][_{항}][_{호}]"
# 예) "지방계약법 시행령" → "LCAE" → LCAE_30_4
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
# 유틸 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def read_docx(path: Path) -> list[tuple[str, str]]:
    """
    docx 파일을 읽어 (타입, 텍스트) 튜플 리스트로 반환.

    반환 타입:
        'p'   — 일반 단락 (paragraph)
        'tbl' — 표(table) 셀

    표와 단락을 구분하는 이유:
        용역계약 일반조건(PYG)의 경우 장·절 헤더가 표 셀에 들어 있음.
        parse_pyg()에서 typ=='tbl'일 때만 절 헤더로 인식하도록 분기함.
    """
    from docx.oxml.ns import qn
    from docx.table import Table as DocxTable

    doc   = Document(str(path))
    lines = []
    for block in doc.element.body:
        tag = block.tag.split('}')[-1]
        if tag == 'p':
            # w:t 태그에서 텍스트만 추출 (서식 태그 제외)
            text = ''.join(r.text for r in block.iter(qn('w:t'))).strip()
            if text:
                lines.append(('p', text))
        elif tag == 'tbl':
            # 표의 모든 셀을 순서대로 읽음
            tbl = DocxTable(block, doc)
            for row in tbl.rows:
                for cell in row.cells:
                    t = cell.text.strip()
                    if t:
                        lines.append(('tbl', t))
    return lines


def find_meta(filename: str) -> dict | None:
    """
    파일명(stem)에 부분 일치하는 FILE_META 키를 찾아 메타 반환.

    매핑 전략:
        키 길이 내림차순 정렬 → 더 구체적인 키(시행령, 시행규칙)가 본법보다 먼저 매핑됨.
        예) '지방계약법_시행령.docx' → '지방계약법_시행령' 키 먼저 매칭,
            '지방계약법' 키에 오탐되지 않음.

    타임스탬프 prefix 처리:
        Google Drive 다운로드 파일명 앞에 숫자_가 붙는 경우 제거 후 재매핑.
        예) '1782288864349_개인정보_보호법_시행령' → '개인정보_보호법_시행령'
    """
    # 타임스탬프 prefix 제거 (숫자로 시작하면)
    clean = re.sub(r"^\d+_", "", filename)

    for key in sorted(FILE_META.keys(), key=len, reverse=True):
        if key in clean or key in filename:
            return FILE_META[key]
    return None


def make_article_id(article_number: str) -> str:
    """
    조문 번호 문자열에서 공백·슬래시·가운뎃점을 언더스코어로 치환.
    파일명이나 딕셔너리 키로 안전하게 쓸 수 있는 식별자를 만든다.
    예) '제9장 제7절 제5항 가' → '제9장_제7절_제5항_가'
    """
    return re.sub(r"[\s/·]", "_", article_number).strip("_")


def _strip_comments(text: str) -> str:
    """
    법령 텍스트에서 개정 주석을 제거한다.
        <개정 2013. 3. 23.>  →  제거
        [전문개정 2020. 6. 9.] →  제거

    이유:
        호 분리 정규식 r"\\s(\\d{1,2})\\.\\s" 적용 전에
        날짜 안의 숫자(예: '3. 23.')가 호 번호로 오인될 수 있음.
        주석을 먼저 제거하면 이 버그를 방지할 수 있다.
    """
    text = re.sub(r'<[^>]+>', '', text)   # <...> 형태 주석 제거
    text = re.sub(r'\[[^\]]+\]', '', text) # [...] 형태 주석 제거
    return text


def make_chunk_id(
    prefix: str,
    jo: int,
    hang: int | None = None,
    ho: int | None = None,
    jo_ui: int | None = None,
) -> str:
    """
    chunk_id 생성 함수.

    규칙: {PREFIX}_{조}[_의{조의N}][_{항}][_{호}]

    인자:
        prefix  — 법령 약어 (예: LCA, LCAE)
        jo      — 조 번호 (정수)
        hang    — 항 번호 (정수, 없으면 None)
                  ★ 항 없이 바로 호인 경우 hang=0 으로 호출
        ho      — 호 번호 (정수, 없으면 None)
        jo_ui   — 조의N (예: 제64조의2 → jo=64, jo_ui=2)

    예시:
        make_chunk_id("LCA", 30)          → "LCA_30"
        make_chunk_id("LCA", 30, hang=4)  → "LCA_30_4"
        make_chunk_id("LCA", 30, hang=0, ho=1) → "LCA_30_0_1"  ← 항 없는 호
        make_chunk_id("LCAE", 64, jo_ui=2, hang=1) → "LCAE_64_의2_1"
    """
    jo_part = str(jo) + (f"_의{jo_ui}" if jo_ui is not None else "")
    parts = [jo_part]
    if hang is not None:   # hang=0도 포함 — 항 없는 호를 나타냄
        parts.append(str(hang))
    if ho is not None:
        parts.append(str(ho))
    return f"{prefix}_{'_'.join(parts)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파서 A: 용역계약 일반조건 (PYG 전용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 지방자치단체 용역계약 일반조건(행안부 예규 제367호)은 일반 법령과
# 구조가 다르다. 조(條) 대신 장(章)·절(節)·항(項)·목(目) 체계를 쓴다.
#
# 청크 단위:  절·항·목 (예: 제9장 제7절 제5항 가목)
# chunk_id:  PYG_{장}_{절}_{항}[_{목}]  (예: PYG_9_7_5_가)
#
# 구조 특이사항:
#   - 9장 이전: 절 헤더가 표(tbl) 셀에 있음 → typ=='tbl'일 때만 절로 인식
#   - 9장 이후: 절 헤더가 일반 단락(p)에도 나타남 → cur_chapter가 있으면 p도 허용

def parse_pyg(lines: list[tuple[str, str]], prefix: str = "PYG") -> list[dict]:
    articles    = []
    cur_chapter = None  # 현재 장 번호 (문자열, 없으면 None)
    cur_section = None  # 현재 절 번호 (문자열)
    cur_clause  = None  # 현재 항 번호 (문자열)
    cur_item    = None  # 현재 목 (가·나·다 등, 없으면 None)
    buf: list[str] = [] # 현재 청크에 누적 중인 텍스트 줄들

    # 구조 인식 정규식
    chapter_pat = re.compile(r"^제\s*(\d+)\s*장")          # 장 헤더
    section_pat = re.compile(r"^제\s*(\d+)\s*절")          # 절 헤더
    clause_pat  = re.compile(r"^\s*(\d+)\s*\.")            # 항 (1. 2. 3. ...)
    item_pat    = re.compile(r"^\s*([가나다라마바사아자차카타파하])\s*\.") # 목 (가. 나. ...)

    def flush():
        """
        buf에 누적된 텍스트를 하나의 청크로 articles에 추가하고 버퍼를 비운다.
        절·항이 확정되지 않은 상태에서는 아무것도 하지 않는다.
        """
        nonlocal buf, cur_item
        if not buf or not cur_section or not cur_clause:
            buf = []; cur_item = None
            return

        # 장이 있으면 article_number 앞에 "제N장 " 붙임
        prefix_str = f"제{cur_chapter}장 " if cur_chapter else ""

        if cur_item:
            an        = prefix_str + f"제{cur_section}절 제{cur_clause}항 {cur_item}"
            hierarchy = {"절": f"제{cur_section}절", "항": f"제{cur_clause}항", "호": cur_item}
        else:
            an        = prefix_str + f"제{cur_section}절 제{cur_clause}항"
            hierarchy = {"절": f"제{cur_section}절", "항": f"제{cur_clause}항"}

        if cur_chapter:
            hierarchy["장"] = f"제{cur_chapter}장"

        # chunk_id 조립: PYG_{장}_{절}_{항}[_{목}]
        id_parts  = ([cur_chapter] if cur_chapter else []) + [cur_section, cur_clause]
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
        # 절 헤더: 9장 이전은 tbl만, 9장 이후(cur_chapter 있음)는 p도 허용
        sm  = section_pat.match(text) if not chm and (typ == "tbl" or cur_chapter) else None
        # 항·목은 일반 단락(p)에만 나타남
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
            # 목(가·나·다) 시작 — 이전 항 또는 목 flush 후 새 목 시작
            flush()
            buf = [text]
            cur_item = im.group(1)
        elif cur_clause:
            # 항·목 본문 계속 — buf에 누적
            buf.append(text)

    flush()  # 마지막 청크 저장
    return articles


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파서 B: 일반 법령 (조/항/호 구조)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 지방계약법·소프트웨어 진흥법·공유재산법·개인정보보호법 등
# 표준 법령 구조(조 → 항 → 호)에 사용하는 파서.
#
# Hierarchical RAG 구조:
#   parent 청크 (is_parent=True):
#       조 전체 원문을 하나로 합친 청크.
#       검색 대상이 아니라 맥락 제공용.
#       child 청크가 hit됐을 때 parent_id로 조 전체를 함께 fetch.
#   child 청크 (is_parent=False):
#       항 또는 호 단위 실제 검색 대상.
#       parent_id로 소속 조의 parent 청크를 참조.
#   단항 조문 (항 구분자 없고 호도 없음):
#       parent/child 구분 없이 단일 청크. parent_id=None.
#
# ★ 항 없이 바로 호인 경우 (신규 처리):
#   원문에 ①②... 항 구분자가 없지만 '1. 2. 3.' 호 구분자가 있는 조문.
#   → parent 청크(조 전체) + child 청크들(hang=0, ho=N) 생성.
#   → chunk_id: {PREFIX}_{조}_0_{호}  예) LCA_30_0_1
#   → hierarchy: {"조": ..., "호": ...}  # 항 키 없음

def parse_law(lines: list[tuple[str, str]], prefix: str) -> list[dict]:
    # 조문 헤더 패턴: "제N조" 또는 "제N조의N" + 조문 제목(선택)
    article_pat = re.compile(r"^(제\s*\d+\s*조(?:의\s*\d+)?)\s*[(\[〔]?([^)\]\)〕\n]*)[)\]\)〕]?")

    # 원문자 → 항 번호 매핑 (①=1, ②=2, ...)
    HANG_MAP = {c: i + 1 for i, c in enumerate("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮")}

    # ── 1단계: 조 단위로 원문 묶기 ────────────────────────────────
    # 조문 헤더가 나올 때마다 이전 조를 raw_articles에 저장.
    # seen_jo: 개정 예정 조문이 현행 조문과 함께 수록된 경우
    #          동일 (조, 조의N) 키가 두 번 등장할 수 있음.
    #          첫 번째(현행)만 파싱하고 이후는 스킵.
    raw_articles: list[dict] = []
    seen_jo: set[tuple] = set()
    cur_jo    = None
    cur_jo_ui = None
    cur_title = ""
    buf: list[str] = []

    def flush_jo():
        """현재 조문 버퍼를 raw_articles에 저장."""
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
    in_bujik  = False  # 부칙 이후 텍스트는 파싱 제외

    for _, text in lines:
        if bujik_pat.match(text):
            # 부칙 시작 — 이후 모든 줄 스킵
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
            raw_jo_str = re.sub(r"\s+", "", m.group(1))  # 공백 제거 후 파싱
            jo_m = re.match(r"제(\d+)조(?:의(\d+))?", raw_jo_str)
            cur_jo    = int(jo_m.group(1)) if jo_m else None
            cur_jo_ui = int(jo_m.group(2)) if jo_m and jo_m.group(2) else None
            cur_title = m.group(2).strip() if m.group(2) else ""
        else:
            buf.append(text)

    flush_jo()  # 마지막 조문 저장

    # ── 2단계: 조 → 항 → 호 단위로 분리하여 청크 생성 ────────────
    articles: list[dict] = []

    for raw in raw_articles:
        jo    = raw["jo"]
        jo_ui = raw["jo_ui"]
        text  = raw["text"]
        jo_str    = f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "")
        parent_id = make_chunk_id(prefix, jo, jo_ui=jo_ui)  # 조 단위 ID

        # ① 항 구분자(원문자)로 먼저 분리 시도
        hang_splits = re.split(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])", text)

        if len(hang_splits) <= 1:
            # 항 구분자 없음 → 호 구분자가 있는지 추가 확인
            text_clean = _strip_comments(text)  # 날짜 주석 제거 후 호 분리
            ho_splits  = re.split(r"\s(\d{1,2})\.\s", text_clean)

            if len(ho_splits) <= 1:
                # ─ 케이스 1: 단항 단호 — parent 없이 단일 청크
                articles.append({
                    "chunk_id":       parent_id,
                    "article_id":     jo_str,
                    "article_number": jo_str,
                    "title":          raw["title"],
                    "text":           text,
                    "is_parent":      False,  # 단일 청크이므로 parent 구분 불필요
                    "parent_id":      None,
                    "hierarchy":      {"조": jo_str},
                })
            else:
                # ─ 케이스 2: ★ 항 없이 바로 호
                #   parent(조 전체) + child들(hang=0) 생성
                articles.append({
                    "chunk_id":       parent_id,
                    "article_id":     jo_str,
                    "article_number": jo_str,
                    "title":          raw["title"],
                    "text":           text,       # 조 전체 원문
                    "is_parent":      True,
                    "parent_id":      None,
                    "hierarchy":      {"조": jo_str},
                })
                # 각 호를 hang=0 child 청크로 생성
                j = 1
                while j < len(ho_splits) - 1:
                    ho_num  = int(ho_splits[j])
                    ho_text = ho_splits[j + 1].strip() if j + 1 < len(ho_splits) else ""
                    chunk_id = make_chunk_id(prefix, jo, hang=0, ho=ho_num, jo_ui=jo_ui)
                    articles.append({
                        "chunk_id":       chunk_id,
                        "article_id":     jo_str + f"제{ho_num}호",
                        "article_number": jo_str + f"제{ho_num}호",
                        "title":          raw["title"],
                        "text":           f"{ho_splits[0].strip()} {ho_num}. {ho_text}",
                        "is_parent":      False,
                        "parent_id":      parent_id,
                        "hierarchy": {
                            "조": jo_str,
                            "호": f"제{ho_num}호",  # 항이 없으므로 항 키 생략
                        },
                    })
                    j += 2

            continue  # 다음 조문으로

        # ─ 케이스 3: 다항 조문 — parent(조 전체) + child(항/호) 생성
        articles.append({
            "chunk_id":       parent_id,
            "article_id":     jo_str,
            "article_number": jo_str,
            "title":          raw["title"],
            "text":           text,       # 조 전체 원문 (항 포함)
            "is_parent":      True,
            "parent_id":      None,
            "hierarchy":      {"조": jo_str},
        })

        # 각 항을 순서대로 처리
        i = 1
        while i < len(hang_splits) - 1:
            hang_char = hang_splits[i]                                        # ① ② ...
            hang_text = hang_splits[i + 1].strip() if i + 1 < len(hang_splits) else ""
            hang_num  = HANG_MAP.get(hang_char, i)                            # 원문자 → 숫자

            # 항 내부에 호 구분자가 있는지 확인 (날짜 주석 먼저 제거)
            hang_text_clean = _strip_comments(hang_text)
            ho_splits = re.split(r"\s(\d{1,2})\.\s", hang_text_clean)

            if len(ho_splits) <= 1:
                # ─ 케이스 3a: 호 없는 항 — 항 단위 단일 child 청크
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
                # ─ 케이스 3b: 호 있는 항 — 호 단위 child 청크들
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
                        # 항 원문자 + 항 앞부분 + 호 번호 + 호 본문
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

            i += 2  # hang_splits는 [앞텍스트, ①, 본문, ②, 본문, ...] 구조

    return articles


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 태깅 — is_ref_article / is_upper_law
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def tag_article(article: dict, is_ref_doc: bool) -> dict:
    """
    각 청크에 두 가지 불리언 플래그를 추가한다.

    is_ref_article (bool):
        용역계약 일반조건(PYG) 문서의 핵심 조항 여부.
        is_ref_doc=True(PYG 문서)이고, article_number가 REF_ARTICLE 목록에
        포함될 때만 True. 나머지 법령은 항상 False.

    is_upper_law (bool):
        PYG 본문이 직접 인용하는 상위법 조문 여부.
        chunk_id가 UPPER_LAW_IDS 세트에 있으면 True.
        Qdrant 검색 시 상위법 우선 fetch 필터로 활용.
    """
    an = article.get("article_number", "")
    article["is_ref_article"] = is_ref_doc and any(ref in an for ref in REF_ARTICLE)
    article["is_upper_law"]   = article.get("chunk_id", "") in UPPER_LAW_IDS
    return article


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 파일 단위 처리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_file(path: Path):
    """
    docx 파일 1개를 파싱하고 결과를 두 개의 JSON 파일로 저장한다.

    출력 파일:
        {stem}_jo.json
            조(條) 단위 청크만 모은 파일.
            is_parent=True 인 parent 청크 + 단항 단호 청크(parent_id=None) 포함.
            Hierarchical RAG에서 child hit 후 맥락 fetch용.

        {stem}_ho.json
            호 단위 child 청크만 모은 파일.
            is_parent=False 인 청크 전체.
            BGE-M3로 임베딩해서 Qdrant law_kb_ho 컬렉션에 업로드하는 실제 검색 대상.

    디버그 출력:
        - chunk_id 샘플 5개 (_hang 기준)
        - 항 없는 호 청크(chunk_id에 '_0_' 포함) 개수 및 예시
          → cross_ref 파일과 대조해 hang=0 처리가 올바른지 확인할 때 사용
    """
    filename = path.stem

    # Word 임시 잠금 파일 (~$로 시작) 스킵
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
        # 용역계약 일반조건 전용 파서 사용
        # 장 감지 디버그 출력으로 구조 파악 가능
        chap_pat = re.compile(r"제\s*\d+\s*장")
        for i, (typ, text) in enumerate(paragraphs):
            if chap_pat.search(text):
                print(f"  [DEBUG] 장 감지 typ={typ!r} text={text!r}")
        articles = parse_pyg(paragraphs, prefix=prefix)
    else:
        # 표준 법령 파서 사용
        articles = parse_law(paragraphs, prefix=prefix)

    # 태깅: is_ref_article / is_upper_law 플래그 추가
    articles = [tag_article(a, meta["is_ref_article_doc"]) for a in articles]

    # 조 단위 / 항·호 단위 분리
    #   jo   : parent 청크(is_parent=True) + 단항 단호(parent_id=None)
    #   ho   : 전체 청크 (parent + child) — Hierarchical RAG 구조 그대로 유지
    jo_articles = [a for a in articles if a.get("is_parent") or not a.get("parent_id")]
    ho_articles = articles  # 전체 (parent + child) — Hierarchical RAG 구조 유지

    # JSON 저장 — jo/ho 폴더 분리
    for out_dir, subset, label in [
        (OUTPUT_DIR_JO,   jo_articles,   "jo"),
        (OUTPUT_DIR_HO, ho_articles, "ho"),
    ]:
        result = {
            "document_type":     meta["document_type"],
            "source":            meta["source"],
            "filename":          path.name,
            "total_articles":    len(subset),
            "ref_article_count": sum(1 for a in subset if a.get("is_ref_article")),
            "upper_law_count":   sum(1 for a in subset if a.get("is_upper_law")),
            "articles":          subset,
        }
        out_path = out_dir / f"{filename}_{label}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  → 저장({label}): {out_path} ({len(subset)}개)")

    # 디버그 출력
    print(f"  → chunk_id 샘플(_ho): {[a['chunk_id'] for a in ho_articles[:5]]}")

    # 항 없는 호 청크 확인 (chunk_id에 '_0_' 포함 여부로 판별)
    no_hang_ho = [a for a in ho_articles if "_0_" in a["chunk_id"]]
    if no_hang_ho:
        print(f"  → 항 없는 호 청크: {len(no_hang_ho)}개 (예: {no_hang_ho[0]['chunk_id']})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    """
    LAW_DIR 내 모든 .docx 파일을 순서대로 파싱한다.
    파일이 없으면 에러 메시지 출력 후 종료.
    """
    files = list(LAW_DIR.glob("*.docx"))
    if not files:
        print(f"[ERROR] {LAW_DIR} 에 .docx 파일이 없습니다.")
        return

    for f in sorted(files):
        process_file(f)

    print("\n✅ 완료! 결과물:", OUTPUT_DIR_JO.parent)


if __name__ == "__main__":
    main()