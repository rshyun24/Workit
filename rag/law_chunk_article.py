"""
Workit - 법령 문서 파싱 + cross_refs 추출 스크립트 (조 단위 청크, chunk_id 고정 슬롯)

"조" 하나를 통째로 chunk 1개로 묶어서 법령 docx를 파싱한다 (호/항/목까지는
쪼개지 않음. 호/항/목 단위는 law_chunk_reference.py 담당).

chunk_id 규칙:
  - 일반 법령: "{prefix}_{장}_{절}_{조}" 고정 4토큰.
    장/절이 없으면 위치 상관없이 '0'으로 채움 (가변 길이 id 금지).
    예) LCA_0_0_7   (장/절 없는 법, 제7조)
        PIPA_1_0_7  (장1, 절 없음, 제7조)
        PIPA_3_1_15 (장3, 절1, 제15조)
  - PYG는 조가 없는 구조라서 "조 단위"에 대응하는 최상위 fetch 단위가 항이다.
    그래서 PYG만 "절" 대신 "항" 단위로 chunk를 쪼갠다.
    chunk_id는 "{prefix}_{장}_{절}_0_{항}" 5토큰 (조 자리에 '0'을 명시적으로 넣음).
    예) PYG_9_1_0_3  (9장 1절 3항)
    -> 이 '0'은 ho-level id "{prefix}_{장}_{절}_0_{항}_{호}_{목}_{세목}"의
       앞부분과 그대로 겹치게 하기 위한 것 (ho-level도 PYG는 조 슬롯이 항상
       0으로 고정). 앞부분 문자열이 겹쳐야 ho-chunk가 자기 부모(jo-chunk)를
       "startswith" 방식으로 찾을 수 있다. 일반 법령은 조가 곧 최상위 단위라
       이런 보정이 필요 없어서 4토큰 그대로 둔다.
  - 제N조의M 가지번호는 '조번호-M' 형태로 보존 (예: 제7조의2 -> '7-2')
    -> 이거 안 하면 제7조/제7조의2/제7조의3...이 전부 같은 id로 뭉개짐 (실측 검증 완료)
  - parent 매칭 규칙: ho_chunk_id가 "jo_chunk_id + '_'" 로 시작하는지로 판단한다
    (길이를 억지로 맞추는 게 아니라 접두사 일치로 비교 — 일반 법령은 4토큰, PYG는
    5토큰으로 서로 길이가 달라도 이 방식으로 문제없이 매칭된다).

input : C:/project/Workit/data/law/ 내 docx 파일
output: C:/project/Workit/data/structured_jo_fixedid/ 내 JSON 파일

사용법:
    pip install python-docx
    python law_chunk_article.py
"""

import json
import re
from pathlib import Path

from docx import Document

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
LAW_DIR = Path("C:/project/Workit/data/law")
OUTPUT_DIR = Path("C:/project/Workit/data/structured_jo_fixedid")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# REF_ARTICLE & UPPER_LAW
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

UPPER_LAW = ["제90조", "제75조", "제27조", "제50조", "제59조", "제22조"]

# ─────────────────────────────────────────
# 파일명(stem) → 메타 매핑
# ─────────────────────────────────────────
FILE_META = {
    "지방자치단체 용역계약": {
        "document_type": "지방자치단체 용역계약 일반조건",
        "law_name": "지방자치단체 용역계약 일반조건",
        "source": "행정안전부 예규",
        "is_ref_article_doc": True,
    },
    "지방계약법_시행규칙": {
        "document_type": "지방계약법 시행규칙",
        "law_name": "지방계약법 시행규칙",
        "source": "행정안전부령",
        "is_ref_article_doc": False,
    },
    "지방계약법_시행령": {
        "document_type": "지방계약법 시행령",
        "law_name": "지방계약법 시행령",
        "source": "대통령령",
        "is_ref_article_doc": False,
    },
    "지방계약법": {
        "document_type": "지방계약법",
        "law_name": "지방계약법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
    "소프트웨어 진흥법 시행령": {
        "document_type": "소프트웨어 진흥법 시행령",
        "law_name": "소프트웨어 진흥법 시행령",
        "source": "대통령령",
        "is_ref_article_doc": False,
    },
    "소프트웨어_진흥법": {
        "document_type": "소프트웨어 진흥법",
        "law_name": "소프트웨어 진흥법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
    "지방회계법_시행령": {
        "document_type": "지방회계법 시행령",
        "law_name": "지방회계법 시행령",
        "source": "대통령령",
        "is_ref_article_doc": False,
    },
    "지방회계법": {
        "document_type": "지방회계법",
        "law_name": "지방회계법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
    "공유재산 및 물품 관리법 시행령": {
        "document_type": "공유재산법 시행령",
        "law_name": "공유재산법 시행령",
        "source": "대통령령",
        "is_ref_article_doc": False,
    },
    "공유재산법": {
        "document_type": "공유재산법",
        "law_name": "공유재산법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
    "개인정보 보호법 시행령": {
        "document_type": "개인정보보호법 시행령",
        "law_name": "개인정보보호법 시행령",
        "source": "대통령령",
        "is_ref_article_doc": False,
    },
    "개인정보 보호법": {
        "document_type": "개인정보보호법",
        "law_name": "개인정보보호법",
        "source": "법률",
        "is_ref_article_doc": False,
    },
}

DOC_TYPE_TO_PREFIX = {
    "지방계약법": "LCA",
    "지방계약법 시행령": "LCAE",
    "지방계약법 시행규칙": "LCAR",
    "소프트웨어 진흥법": "SWPA",
    "소프트웨어 진흥법 시행령": "SWPAE",
    "지방회계법": "LARA",
    "지방회계법 시행령": "LARAE",
    "지방자치단체 용역계약 일반조건": "PYG",
    "공유재산법": "PPMA",
    "공유재산법 시행령": "PPMAE",
    "개인정보보호법": "PIPA",
    "개인정보보호법 시행령": "PIPAE",
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
        tag = block.tag.split("}")[-1]
        if tag == "p":
            text = "".join(r.text for r in block.iter(qn("w:t"))).strip()
            if text:
                lines.append(("p", text))
        elif tag == "tbl":
            tbl = DocxTable(block, doc)
            for row in tbl.rows:
                for cell in row.cells:
                    t = cell.text.strip()
                    if t:
                        lines.append(("tbl", t))
    return lines


def find_meta(filename: str) -> dict | None:
    for key in sorted(FILE_META.keys(), key=len, reverse=True):
        if key in filename:
            return FILE_META[key]
    return None


def make_jo_chunk_id(
    prefix: str,
    jo: int | None = None,
    jo_ui: int | None = None,
    jang: int | None = None,
    jeol: int | None = None,
    hang: int | None = None,
) -> str:
    """
    일반 법령: {prefix}_{장}_{절}_{조}            (4토큰, 조까지)
    PYG(조 없음): {prefix}_{장}_{절}_0_{항}        (5토큰, 조 자리에 0 + 항까지)
    - 장/절이 없으면 위치 상관없이 '0'
    - 조의N 가지번호는 '조번호-N' 형태로 보존 (예: 제7조의2 -> '7-2')
    """
    jang_part = str(jang) if jang is not None else "0"
    jeol_part = str(jeol) if jeol is not None else "0"

    if jo is not None:
        jo_part = f"{jo}-{jo_ui}" if jo_ui else str(jo)
        return f"{prefix}_{jang_part}_{jeol_part}_{jo_part}"

    if hang is not None:
        # PYG 전용: 조가 없으므로 조 자리에 '0'을 명시적으로 넣어
        # ho-level id 앞부분과 문자열이 겹치도록 함 (parent 매칭용)
        return f"{prefix}_{jang_part}_{jeol_part}_0_{hang}"

    return f"{prefix}_{jang_part}_{jeol_part}_0"


def _strip_comments(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[[^\]]+\]", "", text)
    return text.strip()


def _make_jo_hierarchy(
    jang: int | None, jo: int, jo_ui: int | None, jeol: int | None = None
) -> dict:
    jang_str = f"제{jang}장" if jang else None
    jeol_str = f"제{jeol}절" if jeol else None
    jo_str = f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "")
    h = {}
    if jang_str:
        h["장"] = jang_str
    if jeol_str:
        h["절"] = jeol_str
    h["조"] = jo_str
    return h


# ─────────────────────────────────────────
# 파서 1: PYG 전용 — 항 단위 (조가 없으므로 항이 최상위 fetch 단위)
# 절 밑의 "1." "2." 같은 항 번호마다 chunk 하나, 그 안의 가/나/다(호)는
# 안 쪼개고 텍스트에 통째로 포함 (jo-level이라 항 이하는 안 나눔).
# ─────────────────────────────────────────
def parse_pyg_hang(
    lines: list[tuple[str, str]],
    prefix: str = "PYG",
    law_name: str = "지방자치단체 용역계약 일반조건",
) -> list[dict]:
    articles = []
    cur_chapter = None
    cur_section = None
    cur_clause = None
    buf: list[str] = []

    chapter_pat = re.compile(r"^제\s*(\d+)\s*장")
    section_pat = re.compile(r"^제\s*(\d+)\s*절")
    clause_pat = re.compile(r"^\s*(\d+)\s*\.")

    def flush():
        nonlocal buf
        if not buf or not cur_section or not cur_clause:
            buf = []
            return

        chapter_str = f"제{cur_chapter}장" if cur_chapter else ""
        an = chapter_str + f"제{cur_section}절제{cur_clause}항"
        hierarchy = {"절": f"제{cur_section}절", "항": f"제{cur_clause}항"}
        if cur_chapter:
            hierarchy["장"] = f"제{cur_chapter}장"

        full_text = _strip_comments(" ".join(buf))
        chunk_id = make_jo_chunk_id(
            prefix,
            jang=int(cur_chapter) if cur_chapter else None,
            jeol=int(cur_section) if cur_section else None,
            hang=int(cur_clause),
        )

        articles.append(
            {
                "chunk_id": chunk_id,
                "law_name": law_name,
                "article_id": an,
                "article_number": an,
                "text": full_text,
                "hierarchy": hierarchy,
                "is_ref_article": False,
                "is_upper_law": False,
            }
        )
        buf = []

    for typ, text in lines:
        chm = chapter_pat.match(text)
        sm = (
            section_pat.match(text)
            if not chm and (typ == "tbl" or cur_chapter)
            else None
        )
        cm = clause_pat.match(text) if typ == "p" and not chm and not sm else None

        if chm:
            flush()
            cur_chapter = chm.group(1)
            cur_section = None
            cur_clause = None
        elif sm:
            flush()
            cur_section = sm.group(1)
            cur_clause = None
        elif cm and cur_section:
            flush()
            cur_clause = cm.group(1)
            buf = [text]
        elif cur_clause:
            buf.append(text)

    flush()
    return articles


# ─────────────────────────────────────────
# 파서 2: 일반 법령 — 조 단위 (항/호/목 분리 없이 본문 전체를 한 chunk로)
# ─────────────────────────────────────────
def parse_law_jo(
    lines: list[tuple[str, str]], prefix: str, law_name: str
) -> list[dict]:
    article_pat = re.compile(
        r"^(제\s*\d+\s*조(?:의\s*\d+)?)\s*[(\[〔]?([^)\]\)〕\n]*)[)\]\)〕]?"
    )
    jang_pat = re.compile(r"^제\s*(\d+)\s*장")
    jeol_pat = re.compile(r"^제\s*(\d+)\s*절")

    raw_articles: list[dict] = []
    cur_jo = None
    cur_jo_ui = None
    cur_jang = None
    cur_jeol = None
    cur_title = ""
    buf: list[str] = []
    in_bujik = False
    bujik_pat = re.compile(r"^부\s*칙")

    def flush_jo():
        if cur_jo is not None and buf:
            raw_articles.append(
                {
                    "jo": cur_jo,
                    "jo_ui": cur_jo_ui,
                    "jang": cur_jang,
                    "jeol": cur_jeol,
                    "title": cur_title,
                    "text": " ".join(buf),
                }
            )

    for _, text in lines:
        if bujik_pat.match(text):
            in_bujik = True
            flush_jo()
            cur_jo = None
            buf = []
            continue
        if in_bujik:
            continue

        jm = jang_pat.match(text)
        if jm and not article_pat.match(text):
            flush_jo()
            cur_jang = int(jm.group(1))
            cur_jeol = None
            cur_jo = None
            buf = []
            continue

        jlm = jeol_pat.match(text)
        if jlm and not article_pat.match(text):
            flush_jo()
            cur_jeol = int(jlm.group(1))
            cur_jo = None
            buf = []
            continue

        m = article_pat.match(text)
        if m:
            flush_jo()
            buf = [text]
            raw_jo_str = re.sub(r"\s+", "", m.group(1))
            jo_m = re.match(r"제(\d+)조(?:의(\d+))?", raw_jo_str)
            cur_jo = int(jo_m.group(1)) if jo_m else None
            cur_jo_ui = int(jo_m.group(2)) if jo_m and jo_m.group(2) else None
            cur_title = m.group(2).strip() if m.group(2) else ""
        else:
            buf.append(text)

    flush_jo()

    # 같은 (조, 조의N) 중복(원본 복붙 / 시행일 예고 병기) 시 첫 번째(현행)만 유지
    seen_jo: set[tuple[int, int | None]] = set()
    deduped_raw_articles: list[dict] = []
    for raw in raw_articles:
        key = (raw["jo"], raw["jo_ui"])
        if key in seen_jo:
            continue
        seen_jo.add(key)
        deduped_raw_articles.append(raw)
    raw_articles = deduped_raw_articles

    # 조 단위로 하나의 chunk 생성 (항/호/목 안 쪼갬, 텍스트 전체 보존)
    articles: list[dict] = []
    for raw in raw_articles:
        jo, jo_ui = raw["jo"], raw["jo_ui"]
        jang, jeol = raw["jang"], raw["jeol"]
        title = raw["title"]

        jang_str = f"제{jang}장" if jang else ""
        jeol_str = f"제{jeol}절" if jeol else ""
        jo_str = jang_str + jeol_str + f"제{jo}조" + (f"의{jo_ui}" if jo_ui else "")
        jo_chunk_id = make_jo_chunk_id(prefix, jo, jo_ui=jo_ui, jang=jang, jeol=jeol)

        full_text = _strip_comments(raw["text"])

        articles.append(
            {
                "chunk_id": jo_chunk_id,
                "law_name": law_name,
                "article_id": jo_str,
                "article_number": jo_str,
                "title": title,
                "text": full_text,
                "hierarchy": _make_jo_hierarchy(jang, jo, jo_ui, jeol=jeol),
                "is_ref_article": False,
                "is_upper_law": False,
            }
        )

    return articles


# ─────────────────────────────────────────
# 필터링
# ─────────────────────────────────────────
def tag_article(article: dict, is_ref_doc: bool) -> dict:
    an = article.get("article_number", "")
    article["is_ref_article"] = is_ref_doc and any(ref in an for ref in REF_ARTICLE)
    article["is_upper_law"] = any(ref in an for ref in UPPER_LAW)
    return article


# ─────────────────────────────────────────
# cross_refs: 법령명 → prefix 매핑
# ─────────────────────────────────────────
LAW_TO_PREFIX = {
    "지방계약법 시행규칙": "LCAR",
    "지방계약법 시행령": "LCAE",
    "지방계약법": "LCA",
    "소프트웨어 진흥법 시행령": "SWPAE",
    "소프트웨어진흥법 시행령": "SWPAE",
    "소프트웨어 진흥법": "SWPA",
    "소프트웨어진흥법": "SWPA",
    "지방회계법 시행령": "LARAE",
    "지방회계법": "LARA",
    "지방자치단체 용역계약 일반조건": "PYG",
    "공유재산 및 물품 관리법 시행령": "PPMAE",
    "공유재산법 시행령": "PPMAE",
    "공유재산 및 물품 관리법": "PPMA",
    "공유재산법": "PPMA",
    "개인정보 보호법 시행령": "PIPAE",
    "개인정보보호법 시행령": "PIPAE",
    "개인정보 보호법": "PIPA",
    "개인정보보호법": "PIPA",
}


def resolve_prefix(law_name: str) -> str | None:
    for key in sorted(LAW_TO_PREFIX.keys(), key=len, reverse=True):
        if key in law_name:
            return LAW_TO_PREFIX[key]
    return None


RELATIVE_LAW_MAP: dict[str, dict[str, str]] = {
    "LCA": {"법": "LCA"},
    "LCAE": {"법": "LCA", "영": "LCAE"},
    "LCAR": {"법": "LCA", "영": "LCAE", "규칙": "LCAR"},
    "SWPA": {"법": "SWPA"},
    "SWPAE": {"법": "SWPA", "영": "SWPAE"},
    "LARA": {"법": "LARA"},
    "LARAE": {"법": "LARA", "영": "LARAE"},
    "PPMA": {"법": "PPMA"},
    "PPMAE": {"법": "PPMA", "영": "PPMAE"},
    "PIPA": {"법": "PIPA"},
    "PIPAE": {"법": "PIPA", "영": "PIPAE"},
    "PYG": {"법": "LCA", "영": "LCAE", "규칙": "LCAR"},
}

KW_NORMALIZE = {"시행령": "영", "시행규칙": "규칙"}


def resolve_relative(current_prefix: str, keyword: str) -> str | None:
    return RELATIVE_LAW_MAP.get(current_prefix, {}).get(keyword)


# ─────────────────────────────────────────
# registry: 일반 법령 (jo, jo_ui) -> chunk_id / PYG (jeol, hang) -> chunk_id
# ─────────────────────────────────────────
def build_registry_jo(articles: list[dict]) -> dict:
    reg = {}
    for a in articles:
        h = a.get("hierarchy", {})
        jo_m = re.match(r"제(\d+)조(?:의(\d+))?", h.get("조", ""))
        if jo_m:
            jo = int(jo_m.group(1))
            jo_ui = int(jo_m.group(2)) if jo_m.group(2) else None
            reg[("jo", jo, jo_ui)] = a["chunk_id"]
        else:
            # PYG: 조가 없으므로 (절, 항) 키로 등록
            jeol_m = re.match(r"제(\d+)절", h.get("절", ""))
            hang_m = re.match(r"제(\d+)항", h.get("항", ""))
            if not jeol_m or not hang_m:
                continue
            jeol = int(jeol_m.group(1))
            hang = int(hang_m.group(1))
            reg[("pyg", jeol, hang)] = a["chunk_id"]
    return reg


def lookup_jo(registry: dict, target_prefix: str, jo: int, jo_ui: int | None) -> str:
    """일반 법령 조 단위 registry에서 chunk_id 조회, 없으면 fallback 생성"""
    reg = registry.get(target_prefix, {})
    key = ("jo", jo, jo_ui)
    if key in reg:
        return reg[key]
    return make_jo_chunk_id(target_prefix, jo, jo_ui=jo_ui)


def lookup_pyg_jo(registry: dict, target_prefix: str, jeol: int, hang: int) -> str:
    """PYG 절/항 단위 registry에서 chunk_id 조회, 없으면 fallback 생성"""
    reg = registry.get(target_prefix, {})
    key = ("pyg", jeol, hang)
    if key in reg:
        return reg[key]
    return make_jo_chunk_id(target_prefix, jeol=jeol, hang=hang)


# ─────────────────────────────────────────
# 조항호 번호 추출 (범위형) — "제N조부터 제M조까지" 처리용
# ─────────────────────────────────────────
def extract_jo_nums(text: str) -> list[tuple[int, int | None]]:
    """본문에서 등장하는 모든 (조, 조의N) 를 set으로 반환"""
    nums: set[tuple[int, int | None]] = set()
    range_pat = re.compile(r"제(\d+)조부터\s*제(\d+)조(?:까지)?")
    range_spans = []
    for m in range_pat.finditer(text):
        range_spans.append(m.span())
        for j in range(int(m.group(1)), int(m.group(2)) + 1):
            nums.add((j, None))
    single_pat = re.compile(r"제(\d+)조(?:의(\d+))?")
    for m in single_pat.finditer(text):
        if any(s <= m.start() < e for s, e in range_spans):
            continue
        jo = int(m.group(1))
        jo_ui = int(m.group(2)) if m.group(2) else None
        nums.add((jo, jo_ui))
    return sorted(nums, key=lambda x: (x[0], x[1] if x[1] is not None else -1))


def extract_cross_refs_jo(
    text: str,
    current_prefix: str,
    registry: dict,
    cur_jo: int | None,
    cur_jo_ui: int | None,
) -> list[str]:
    """조 단위 cross_refs: 본문에서 언급되는 다른 조(자기 자신 제외)를 모두 chunk_id로 변환"""
    refs: list[str] = []
    consumed: list[tuple[int, int]] = []

    # 1. 「법령명」 뒤에 나오는 조 참조
    law_pat = re.compile(r"[「『《]([^」』》]+)[」』》]")
    for law_m in law_pat.finditer(text):
        target_prefix = resolve_prefix(law_m.group(1))
        after = text[law_m.end() :].lstrip()
        stop = re.search(r"[。.\n]|[「『《]", after)
        seg = after[: stop.start()].strip() if stop else after[:150].strip()
        seg_start = text.index(seg, law_m.end()) if seg else law_m.end()
        consumed.append((law_m.start(), seg_start + len(seg)))
        if not target_prefix:
            continue
        for jo, jo_ui in extract_jo_nums(seg):
            refs.append(lookup_jo(registry, target_prefix, jo, jo_ui))

    # 2. 법/영/규칙 제N조... (이 법, 이 영, 이 규칙, 시행령, 시행규칙, 단독 법/영/규칙)
    same_pat = re.compile(
        r"(?:이\s*)?(?P<kw>시행규칙|시행령|법|영|규칙)\s+(제\d+조[^。\n「」]{0,80})"
    )
    for m in same_pat.finditer(text):
        kw = KW_NORMALIZE.get(m.group("kw"), m.group("kw"))
        target = resolve_relative(current_prefix, kw)
        if not target:
            continue
        consumed.append(m.span())
        for jo, jo_ui in extract_jo_nums(m.group(2)):
            refs.append(lookup_jo(registry, target, jo, jo_ui))

    # 3. 「」/법영규칙 없이 단독으로 나오는 "제N조(의M)" — 동일 법령 참조
    bare_jo_pat = re.compile(r"제(\d+)조(?:의(\d+))?")
    for m in bare_jo_pat.finditer(text):
        if any(s <= m.start() < e for s, e in consumed):
            continue
        ref_jo = int(m.group(1))
        ref_jo_ui = int(m.group(2)) if m.group(2) else None
        if cur_jo is not None and ref_jo == cur_jo and ref_jo_ui == cur_jo_ui:
            continue  # 자기 자신 참조 제외
        refs.append(lookup_jo(registry, current_prefix, ref_jo, ref_jo_ui))

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

    print("=== Pass 1: 파싱 (일반 법령=조 단위 / PYG=항 단위, 고정 슬롯 chunk_id) ===")
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
        prefix = DOC_TYPE_TO_PREFIX.get(meta["document_type"], "UNK")
        law_name = meta["law_name"]

        if prefix == "PYG":
            articles = parse_pyg_hang(paragraphs, prefix=prefix, law_name=law_name)
        else:
            articles = parse_law_jo(paragraphs, prefix=prefix, law_name=law_name)

        articles = [tag_article(a, meta["is_ref_article_doc"]) for a in articles]
        registry[prefix] = build_registry_jo(articles)
        parsed.append((path, articles, prefix, meta))
        print(f"  -> {len(articles)} chunk")

    print("\n=== 저장 ===")
    for path, articles, prefix, meta in parsed:
        result = {
            "document_type": meta["document_type"],
            "law_name": meta["law_name"],
            "source": meta["source"],
            "filename": path.name,
            "total_articles": len(articles),
            "ref_article_count": sum(1 for a in articles if a.get("is_ref_article")),
            "upper_law_count": sum(1 for a in articles if a.get("is_upper_law")),
            "articles": articles,
        }

        out_path = OUTPUT_DIR / f"{path.stem}_jo.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"  -> saved: {out_path}")

    print("\nDone! output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()