"""
양식이 고정된 RFP(제안요청서) 문서를 정규식·키워드 기반으로 파싱한다.
LLM 없이 동작하며, extract_text()로 추출한 텍스트를 입력받는다.

출력 JSON 구조는 노션 파싱 코드 체계(RFP-01-01 ~ RFP-04-04-11)를 따른다.

── 설계 메모 (2번째 버전) ──────────────────────────────────────────────
이전 버전은 "키워드가 문서 어디에 등장하든 첫/마지막 위치를 찾는다"는 방식이었는데,
아래 세 가지 문제가 반복적으로 발생했다.
  1) 문서 맨 앞 목차에 모든 제목이 나열돼 있어 전부 거기서 잡힘
  2) "요구사항 총괄표" 같은 요약 표에도 같은 단어들이 다시 나열돼 있어 또 잡힘
  3) 상위 챕터 제목이나, 다른 곳에서 그 제목을 인용하는 각주 문장에도 같은
     문구가 등장해 실제 헤더보다 먼저 걸림
     (예: "※ 세부내용은 'Ⅳ. 제안요청내용' 참조" 안의 "제안요청내용")

세 문제 모두 본질은 같다 — "문자열이 어디에 있든 매치되면 그만"이라는 접근이
표/각주/목차/챕터제목을 구분하지 못하는 것. 그래서 이번 버전은 반대로 간다:
문서 전체에서 문자열을 찾는 대신, 먼저 "실제 제목처럼 보이는 줄"만 구조적으로
추출한 다음, 그 후보들 안에서만 키워드를 매칭한다.

정부 RFP는 항상 3단 번호 체계를 쓴다:
  - 챕터: Ⅰ. Ⅱ. Ⅲ. Ⅳ.  (로마 숫자, 줄 맨 앞)
  - 중분류: 1. 2. 3. 4.   (아라비아 숫자, 줄 맨 앞)
  - 요구사항 상세: ① ② ③ ... ⑪  (원문자)
표 셀 안의 라벨(예: "성능 요구사항\n(PER)")이나 각주 인용문은 이 번호 체계를
갖추지 않으므로, 번호가 붙은 줄만 후보로 삼으면 자동으로 걸러진다.

목차에도 똑같은 번호+제목 형태가 나오지만(예: "1. 제안개요"), 이건 실제 본문
보다 항상 앞서 등장하므로 같은 키워드에 여러 후보가 걸리면 "가장 마지막(=본문)
위치"를 사용해 해결한다. 로마 숫자(챕터) 후보는 RFP 코드 매칭에는 아예 쓰지
않는다 — 그래야 "Ⅰ. 사업개요"(챕터 제목)가 "RFP-01-01 제안개요"(그 아래 1번
소제목)를 잘못 잡아채는 일이 없다.
────────────────────────────────────────────────────────────────────────
"""

import re
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 코드 → 탐지 키워드 매핑
# 이제 "실제 제목처럼 보이는 줄" 후보 안에서만 검사하므로, 예전처럼 키워드를
# 억지로 구체화/축소할 필요 없이 자연스러운 동의어를 넉넉히 넣어도 안전하다.
# ─────────────────────────────────────────────────────────────────────────────

_RFP_SECTION_KEYWORDS: list[tuple[str, str, list[str]]] = [
    # (코드, 부제목, 탐지 키워드 목록)

    # ── 챕터 I: 사업개요 ─────────────────────────────────────────────────────
    ('RFP-01-01', '제안개요',              ['제안 개요', '사업 개요', '제안개요', '사업개요']),
    ('RFP-01-02', '추진배경 및 필요성',    ['추진 목적', '추진배경', '추진 배경', '배경 및 필요성', '추진목적']),
    ('RFP-01-03', '서비스 내용',           ['주요 사업범위 및 과업', '서비스 내용', '주요 사업범위', '사업 내용']),
    ('RFP-01-04', '사업 범위',             ['사업 범위', '사업범위', '추진 범위']),

    # ── 챕터 II: 시스템현황 ──────────────────────────────────────────────────
    ('RFP-02-01', '현행 시스템 개요',      ['현황 및 문제점', '현행 시스템 개요', '현황및문제점']),
    ('RFP-02-02', '현행 시스템 현황',      ['시스템 현황', '현행 시스템 현황', '시스템현황']),

    # ── 챕터 III: 사업추진방안 (있는 경우) ──────────────────────────────────
    ('RFP-03-01', '추진목표',              ['추진 목표', '추진목표']),
    ('RFP-03-02', '추진 체계',             ['추진 체계', '추진체계', '추진체계도']),
    ('RFP-03-03', '추진일정',              ['추진 일정', '추진일정']),
    ('RFP-03-04', '추진방안',              ['추진 방안', '추진방안']),

    # ── 챕터 IV: 제안요청내용 ────────────────────────────────────────────────
    ('RFP-04-01', '제안요청 개요',         ['제안 요청 개요', '제안요청 개요']),
    ('RFP-04-02', '목표 시스템 개념도',    ['목표시스템', '목표 시스템', '메뉴 구성도', '메뉴구성도']),
    ('RFP-04-03', '요구사항 총괄표',       ['요구사항 총괄표', '요구사항총괄', '요구사항 목록', '제안 요구사항 목록']),

    # ── 요구사항 상세 (가나다 순) ──────────────────────────────────────────
    ('RFP-04-04-01', '기능 요구사항',          ['기능 요구사항', '기능요구사항']),
    ('RFP-04-04-02', '시스템장비구성 요구사항', ['시스템장비구성', '장비구성 요구사항', '시스템 장비']),
    ('RFP-04-04-03', '성능 요구사항',          ['성능 요구사항', '성능요구사항', '성능일반', '응답 시간']),
    ('RFP-04-04-04', '인터페이스 요구사항',    ['인터페이스 요구사항', '인터페이스요구사항']),
    ('RFP-04-04-05', '데이터 요구사항',        ['데이터 요구사항', '데이터요구사항', '데이터 표준']),
    ('RFP-04-04-06', '테스트 요구사항',        ['테스트 요구사항', '테스트요구사항', '테스트 일반']),
    ('RFP-04-04-07', '보안 요구사항',          ['보안 요구사항', '보안요구사항', '시스템 보안']),
    ('RFP-04-04-08', '품질 요구사항',          ['품질 요구사항', '품질요구사항', '품질 관리']),
    ('RFP-04-04-09', '제약사항',               ['제약 사항', '제약사항']),
    ('RFP-04-04-10', '프로젝트 관리',          ['프로젝트 관리', '프로젝트관리 요구사항', '사업 관리']),
    ('RFP-04-04-11', '프로젝트 지원',          ['프로젝트 지원', '프로젝트지원 요구사항', '사업 지원']),
]

# 요구사항 ID 패턴 (SFR-01, SER-001 등)
# 주의: 예전 목록(TER|QUR|COR|INR|SIR)은 실제 11개 요구사항 카테고리
# (SFR/SER/PER/UIR/DAR/TSR/SCR/QAR/CTR/PMR/PSR)와 맞지 않아 UIR·TSR·SCR·QAR·CTR
# 다섯 카테고리는 애초에 매칭될 수 없었다. 또한 문서마다 번호 자릿수가
# 2자리(SFR-01)/3자리(SFR-001)로 다를 수 있어 자릿수도 유연하게 허용한다.
_REQ_ID_PATTERN = re.compile(
    r'(SFR|SER|PER|UIR|DAR|TSR|SCR|QAR|CTR|PMR|PSR)-(\d{2,3})',
    re.IGNORECASE,
)

# 요구사항 세부 블록 안의 필드 라벨. PDF 표 추출 과정에서 "요구사항 고유번호"가
# "요구사항 고유번" / "호" 처럼 줄바꿈으로 쪼개지거나, ID와 "호"의 등장 순서가
# 문서마다 바뀌는 등 레이아웃이 불안정하므로, "몇 번째 줄에 뭐가 있다"는 위치
# 가정 대신 라벨 자체를 정규식으로 찾아 그 뒤 텍스트를 값으로 취한다.
_REQ_NAME_PATTERN = re.compile(r'요구사항\s*명칭\s*(.+?)(?:\n|$)')
_REQ_DEF_PATTERN = re.compile(r'\[정의\]\s*(.+?)(?:\n|$)')
_REQ_OUTPUT_PATTERN = re.compile(r'산출정보\s*(.+?)(?:\n|$)')

# 메타 정보 추출 패턴
_META_PATTERNS = {
    'project_name': [
        re.compile(r'사\s*업\s*명\s*[:\:]\s*(.+?)(?:\n|$)'),
        re.compile(r'사업명\s*[:\:]\s*(.+?)(?:\n|$)'),
    ],
    'client': [
        re.compile(r'발\s*주\s*기\s*관\s*[:\:]\s*(.+?)(?:\n|$)'),
        re.compile(r'수\s*요\s*기\s*관\s*[:\:]\s*(.+?)(?:\n|$)'),
    ],
    'budget': [
        re.compile(r'사업\s*예산\s*[:\:]?\s*금?\s*([\d,]+)\s*원'),
        re.compile(r'예산\s*[:\:]?\s*금?\s*([\d,]+)\s*원'),
    ],
    'duration': [
        re.compile(r'사업\s*기간\s*[:\:]\s*(.+?)(?:\n|$)'),
        re.compile(r'계약\s*기간\s*[:\:]\s*(.+?)(?:\n|$)'),
    ],
    'contract_method': [
        re.compile(r'계약\s*방법\s*[:\:]\s*(.+?)(?:\n|$)'),
        re.compile(r'계약방법\s*[:\:]\s*(.+?)(?:\n|$)'),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# 제목줄(heading) 구조 탐지
#
# - 로마 숫자 / 아라비아 숫자 제목은 반드시 "줄 맨 앞"에서 시작해야 후보로 인정한다.
#   (각주 인용문 등은 문장 중간에 있으므로 이 조건만으로 자동 제외된다)
# - 원문자(①~⑫)는 그 자체로 이미 특이한 기호라 줄 시작 여부를 따지지 않는다.
#   (실제 PDF 추출 시 앞 섹션과 줄바꿈 없이 붙어버리는 경우가 있어서, 줄 시작을
#   요구하면 그런 헤더를 놓치게 된다)
# ─────────────────────────────────────────────────────────────────────────────
_LINE_HEADING_PATTERN = re.compile(
    r'(?m)^[ \t]*(?:'
    r'(?P<roman>Ⅰ|Ⅱ|Ⅲ|Ⅳ|Ⅴ|Ⅵ|Ⅶ)[ \t]*\.[ \t]*(?P<roman_title>[^\n]*)'
    r'|(?P<num>[1-9])[ \t]*\.[ \t]*(?P<num_title>[^\n]*)'
    r')'
)
_CIRCLED_HEADING_PATTERN = re.compile(
    r'(?P<circled>[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])[ \t]*(?P<circled_title>[^\n]*)'
)

# 요구사항 상세 항목(4.의 하위 11개 분류)을 원문자(①~⑪) 대신 "□ 제목 (SER)"처럼
# 박스 기호 + 코드 접미사로 표기하는 RFP 서식도 있다. 이 경우 본문에는 그 표기가
# 딱 한 번만 등장하고(목차에만 원문자가 나온다), 개별 요구사항의 세부내용도
# 흔히 "□"로 시작하는 불릿을 쓰기 때문에 아무 "□" 줄이나 후보로 잡으면 안 되고,
# 반드시 "(SER)"처럼 코드 접미사가 붙은 줄만 후보로 인정한다.
_BOXED_CATEGORY_HEADING_PATTERN = re.compile(
    r'(?m)^[ \t]*□[ \t]*(?P<boxed_title>[^\n(]*?)[ \t]*'
    r'\((?:SER|SFR|PER|UIR|DAR|TSR|SCR|QAR|CTR|PMR|PSR)\)'
)


def _find_heading_candidates(text: str) -> list[tuple[int, str, str]]:
    """
    문서에서 "실제 제목처럼 보이는 줄"의 후보를 (위치, 종류, 제목텍스트) 형태로
    모두 추출한다. 종류는 'roman'(챕터) / 'num'(중분류) / 'circled'/'boxed'(요구사항
    상세 항목, 표기 방식에 따라 둘 중 하나) 중 하나.
    """
    candidates: list[tuple[int, str, str]] = []

    for m in _LINE_HEADING_PATTERN.finditer(text):
        if m.group('roman') is not None:
            candidates.append((m.start(), 'roman', m.group('roman_title').strip()))
        else:
            candidates.append((m.start(), 'num', m.group('num_title').strip()))

    for m in _CIRCLED_HEADING_PATTERN.finditer(text):
        candidates.append((m.start(), 'circled', m.group('circled_title').strip()))

    for m in _BOXED_CATEGORY_HEADING_PATTERN.finditer(text):
        candidates.append((m.start(), 'boxed', m.group('boxed_title').strip()))

    return candidates


def _resolve_anchor(
    keywords: list[str],
    candidates: list[tuple[int, str, str]],
) -> int | None:
    """
    주어진 키워드 중 하나라도 제목텍스트에 포함되는 후보들의 위치 중
    가장 마지막(=목차가 아니라 실제 본문일 가능성이 높은) 위치를 반환한다.
    'roman'(챕터 제목) 후보는 검사 대상에서 제외한다 — 안 그러면
    "Ⅰ. 사업개요" 같은 챕터 제목이 그 아래 소제목("1. 제안개요" 등)을
    잘못 잡아채는 문제가 재발한다.
    """
    matched = [
        pos for pos, kind, title in candidates
        if kind != 'roman' and any(kw in title for kw in keywords)
    ]
    return max(matched) if matched else None


def _first_match(text: str, patterns: list[re.Pattern]) -> str | None:
    for p in patterns:
        m = p.search(text)
        if m:
            return m.group(1).strip()
    return None


def _extract_requirements(text: str) -> list[dict]:
    """
    SFR-01, SER-01 등 요구사항 ID를 기준으로 항목을 추출한다.

    ID 뒤에 "요구사항 명칭", "[정의]", "산출정보" 같은 필드 라벨이 어떤 순서로
    등장하든(PDF 표 추출 과정에서 줄바꿈 위치가 문서마다 달라질 수 있음)
    라벨 자체를 정규식으로 찾아 값을 취하므로, 줄 순서에 의존하지 않는다.
    """
    requirements = []
    seen_ids: set[str] = set()
    matches = list(_REQ_ID_PATTERN.finditer(text))

    for i, m in enumerate(matches):
        req_id = m.group(0).upper()
        if req_id in seen_ids:
            continue
        seen_ids.add(req_id)

        # 이 항목의 범위: 다음 ID 등장 전까지, 없으면 최대 600자
        end = matches[i + 1].start() if i + 1 < len(matches) else min(len(text), m.end() + 600)
        block = text[m.start():end]

        name_m = _REQ_NAME_PATTERN.search(block)
        name = name_m.group(1).strip()[:100] if name_m else ''

        def_m = _REQ_DEF_PATTERN.search(block)
        definition = def_m.group(1).strip() if def_m else ''

        out_m = _REQ_OUTPUT_PATTERN.search(block)
        output_info = out_m.group(1).strip() if out_m else ''
        # 줄바꿈이 없어 다음 블록의 '요구사항 분류' 라벨까지 값에 섞여드는
        # 경우가 있어(PDF 표 추출 특성상 셀 사이 줄바꿈이 생략되기도 함),
        # 그 지점에서 잘라낸다.
        output_info = re.split(r'요구사항\s*분류', output_info)[0].strip()

        requirements.append({
            'id': req_id,
            'type': m.group(1).upper(),
            'name': name,
            'definition': definition,
            'output': output_info,
        })

    return requirements


def _extract_meta(text: str) -> dict:
    meta: dict[str, Any] = {}

    for key, patterns in _META_PATTERNS.items():
        value = _first_match(text, patterns)
        if value:
            if key == 'budget':
                try:
                    meta[key] = int(value.replace(',', ''))
                except ValueError:
                    meta[key] = value
            else:
                meta[key] = value

    return meta


def parse_rfp(text: str) -> dict:
    """
    RFP 텍스트를 파싱해 코드 체계(RFP-01-01 ~ RFP-04-04-11) 기반 JSON 반환.

    Args:
        text: extract_text()로 추출된 원문 텍스트

    Returns:
        {
            "meta": {"project_name": "...", "client": "...", ...},
            "sections": {
                "RFP-01-01": {"subtitle": "...", "content": "...", "found": True},
                "RFP-04-04-01": {"subtitle": "...", "content": "...", "requirements": [...]},
                ...
            }
        }
    """
    candidates = _find_heading_candidates(text)

    # 코드별 anchor 위치 탐색 (목차/각주/표는 구조적으로 후보에서 제외되거나,
    # 후보가 되더라도 실제 본문 위치가 항상 더 뒤에 있으므로 마지막 위치를 사용)
    anchor_positions: list[tuple[str, str, int]] = []
    for code, subtitle, keywords in _RFP_SECTION_KEYWORDS:
        pos = _resolve_anchor(keywords, candidates)
        if pos is not None:
            anchor_positions.append((code, subtitle, pos))

    anchor_positions.sort(key=lambda x: x[2])

    # 코드별 텍스트 슬라이싱
    sections: dict[str, Any] = {}

    for i, (code, subtitle, start) in enumerate(anchor_positions):
        end = anchor_positions[i + 1][2] if i + 1 < len(anchor_positions) else len(text)
        content = text[start:end].strip()

        # 최대 2,000자로 제한 (단, 요구사항 섹션은 5,000자)
        max_len = 5000 if code.startswith('RFP-04-04') else 2000
        if len(content) > max_len:
            content = content[:max_len] + ' [이하 생략]'

        sections[code] = {
            'subtitle': subtitle,
            'content': content,
            'found': True,
        }

        # 요구사항 섹션이면 상세 항목 파싱
        if code in ('RFP-04-04-01', 'RFP-04-04-03', 'RFP-04-04-04',
                    'RFP-04-04-05', 'RFP-04-04-06', 'RFP-04-04-07',
                    'RFP-04-04-08', 'RFP-04-04-09', 'RFP-04-04-10',
                    'RFP-04-04-11'):
            reqs = _extract_requirements(content)
            if reqs:
                sections[code]['requirements'] = reqs

    # 찾지 못한 섹션은 found=False 로 채움
    found_codes = {item[0] for item in anchor_positions}
    for code, subtitle, _ in _RFP_SECTION_KEYWORDS:
        if code not in found_codes:
            sections[code] = {'subtitle': subtitle, 'content': '', 'found': False}

    return {
        'meta': _extract_meta(text),
        'sections': sections,
    }