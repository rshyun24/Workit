# -*- coding: utf-8 -*-
"""
기술적용결과표(tech_apply) PDF의 적용/부분적용/미적용/해당없음 체크 표를
룰베이스로 검증한다. (행정기관 및 공공기관 정보시스템 구축·운영 지침 별지 서식)

검증 규칙:
1. 4개 체크 칸(적용/부분적용/미적용/해당없음) 중 아무것도 체크되지 않음 → 오류
2. 체크 칸이 2개 이상 체크됨 → 오류
3. '부분적용' 또는 '미적용'이 체크됐는데 '부분적용/미적용시 사유 및 대체기술'란이
   비어있음 → 오류

파싱 방식 (실제 PDF 여러 페이지로 검증하며 재설계됨):
- 표 헤더가 2개 물리 행에 걸쳐 나뉜다("구분/항목/적용계획·결과(4열 병합)/사유"
  → "적용/부분적용/미적용/해당없음"). 한 행 안에서 6개 컬럼을 모두 찾는 방식으로는
  헤더를 못 찾으므로, 여러 행에 걸쳐 누적 탐색한다.
- 헤더 행과 본문 행의 실제 열 경계(x좌표)가 PDF 렌더링상 미묘하게 어긋나는
  페이지가 있다(예: 헤더에서는 '항목'이 3번째 컬럼 인덱스인데 본문 행에서는
  2번째 컬럼에 병합되어 나타남). 그래서 컬럼 "인덱스"가 아니라 헤더 셀의 x좌표
  범위와 가장 많이 겹치는 본문 셀을 찾는 방식으로 컬럼을 재해석한다.
- 같은 표의 한 물리적 행 안에 여러 세부 항목이 셀 내부 줄바꿈으로 뭉쳐 들어간다
  (예: "웹브라우저 관련" 그룹 아래 HTML/XHTML/XML 등 5개 항목이 '항목' 셀 하나에
  줄바꿈으로 들어가고, 체크 표시도 각 체크박스 열에 줄 단위로 쌓인다). 이때
  pdfplumber가 반환하는 셀 텍스트는 빈 줄(체크 안 된 항목)을 생략하므로, 줄
  "개수"만으로는 항목과 체크 표시를 1:1로 대응시킬 수 없다.
  → 그래서 셀 텍스트가 아니라 **글자 좌표(top)** 를 이용한다: 항목 셀 안의 각
    논리 줄의 top 좌표와, 체크박스/사유 셀 안의 각 텍스트 줄의 top 좌표를 뽑아
    가장 가까운 top을 가진 항목에 체크/사유를 매칭한다.
- 항목 셀 안에서 여러 줄이 "워드랩(한 항목이 길어서 줄바꿈된 것)"인지 "서로 다른
  항목이 나열된 것"인지는 불릿("-"/"o"/"ㅇ") 유무만으로는 구분 안 된다(예: "IPv4",
  "IPv6"는 둘 다 불릿 없이 각자 별도 항목). 대신 줄 사이 세로 간격을 본다 —
  워드랩은 간격이 거의 없고(제목/본문 폰트 한 줄 높이 미만), 별도 항목은 간격이
  뚜렷하다.
- "o 웹서비스"/"o DBMS"처럼 하위 항목을 묶기만 하는 그룹 인트로 줄은 그 자체로는
  체크 대상이 아니다. 이런 줄은 이름이 특정 접미어로 끝나는 게 아니라, "바로 다음
  논리 줄이 '-'로 시작하는 항목인가"로 판별된다(예: "o 화상회의... "는 다음 줄이
  "o 부가통신..."이라 그룹 인트로가 아닌 실제 체크 대상 항목).
"""

import re
from dataclasses import dataclass
from typing import Optional

import pdfplumber

CHECKBOX_COLUMNS = ("적용", "부분적용", "미적용", "해당없음")
REASON_REQUIRED_FOR = ("부분적용", "미적용")
REQUIRED_ZONES = ("item", "적용", "부분적용", "미적용", "해당없음", "사유")

# 체크 대상이 아닌 '구분선' 행 판별용 (표 안에서 통째로 한 행을 차지하는 정확한 라벨)
SKIP_LABELS_EXACT = {"기본 지침", "세부 기술 지침"}

TABLE_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}

BULLET_DASH_RE = re.compile(r"^[\-–—]")
BULLET_O_RE = re.compile(r"^[ㅇoO○ｏ]")

# 워드랩(한 항목이 길어 줄바꿈된 것) vs 별개 항목을 가르는 줄간 세로 간격 기준(pt).
# 실측: 워드랩 줄간 간격 ~2pt, 별개 항목(IPv4/IPv6 등) 간격 ~30pt.
LINE_GAP_THRESHOLD = 6.0


def _normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[\s\.\,\·\-\(\)\[\]]", "", text)


def _clean(cell) -> str:
    return (cell or "").strip()


@dataclass
class TechApplyCheckItem:
    page: int
    item_text: str
    checked: dict            # {"적용": bool, "부분적용": bool, "미적용": bool, "해당없음": bool}
    checked_count: int
    reason_filled: bool
    error: Optional[str] = None  # None이면 정상
    bbox: Optional[dict] = None  # 프런트에서 해당 위치로 스크롤/하이라이트하기 위한 페이지 내 상대 좌표(%)


def _find_header_zones(table, grid) -> Optional[tuple]:
    """
    표의 텍스트 그리드에서 항목/체크박스 4개/사유 헤더 셀을 fuzzy 매칭으로 찾고,
    그 셀들의 x좌표 범위(zone)를 반환한다. 헤더가 1~2개 물리 행에 걸쳐 나뉘어
    있어도 누적 탐색으로 처리한다.

    컬럼 "인덱스"가 아니라 "x좌표 범위"를 반환하는 이유: 헤더 행과 본문 행의
    열 경계가 PDF 렌더링상 어긋나는 경우가 있어, 인덱스 매칭은 신뢰할 수 없다.

    Returns: (zones: {key: (x0, x1)}, header_row_count) 또는 못 찾으면 None
    """
    found_at = {}  # key -> (row_i, col_i)
    header_row_count = 0
    for row_i, row in enumerate(grid):
        for i, cell in enumerate(row):
            norm = _normalize(cell)
            if not norm:
                continue
            if norm in ("항목", "항목명") and "item" not in found_at:
                found_at["item"] = (row_i, i)
            elif norm == "구분" and "category" not in found_at:
                found_at["category"] = (row_i, i)
            elif norm == "적용" and "적용" not in found_at:
                found_at["적용"] = (row_i, i)
            elif norm == "부분적용" and "부분적용" not in found_at:
                found_at["부분적용"] = (row_i, i)
            elif norm == "미적용" and "미적용" not in found_at:
                found_at["미적용"] = (row_i, i)
            elif norm == "해당없음" and "해당없음" not in found_at:
                found_at["해당없음"] = (row_i, i)
            elif "사유" in norm and "대체기술" in norm and "사유" not in found_at:
                found_at["사유"] = (row_i, i)

        header_row_count = row_i + 1
        if all(k in found_at for k in REQUIRED_ZONES):
            break
        if row_i >= 2:
            # 안전장치: 헤더가 3행을 넘어가면 이 표는 대상 표가 아니라고 판단
            return None
    else:
        return None

    if not all(k in found_at for k in REQUIRED_ZONES):
        return None

    zones = {}
    for key, (r, c) in found_at.items():
        bbox = table.rows[r].cells[c] if r < len(table.rows) and c < len(table.rows[r].cells) else None
        if bbox:
            zones[key] = (bbox[0], bbox[2])
    if not all(k in zones for k in REQUIRED_ZONES):
        return None

    return zones, header_row_count


def _resolve_cell(row, grid_row, zones: dict, key: str):
    """헤더 zone과 x좌표가 가장 많이 겹치는 본문 행의 셀을 찾아 (bbox, text)를 반환."""
    zone = zones.get(key)
    if not zone:
        return None, ""
    zx0, zx1 = zone
    best_i, best_overlap = None, 0.0
    for i, bbox in enumerate(row.cells):
        if not bbox:
            continue
        bx0, _, bx1, _ = bbox
        overlap = min(bx1, zx1) - max(bx0, zx0)
        if overlap > best_overlap:
            best_overlap = overlap
            best_i = i
    if best_i is None:
        return None, ""
    text = grid_row[best_i] if best_i < len(grid_row) else ""
    return row.cells[best_i], _clean(text)


def _text_lines(page, bbox) -> list:
    """bbox 안의 텍스트를 줄 단위 (top, bottom, text) 리스트로 반환 (빈 줄 제외)."""
    if not bbox:
        return []
    x0, top, x1, bottom = bbox
    x0, x1 = max(x0, 0), min(x1, page.width)
    top, bottom = max(top, 0), min(bottom, page.height)
    if x1 <= x0 or bottom <= top:
        return []
    lines = page.within_bbox((x0, top, x1, bottom)).extract_text_lines() or []
    return [(ln["top"], ln["bottom"], ln["text"].strip()) for ln in lines if ln["text"].strip()]


def _group_logical_lines(raw_lines: list) -> list:
    """
    여러 물리 줄을 논리적 항목 단위로 묶는다. 새 항목은 다음 중 하나일 때 시작된다:
      - '-' 또는 'o/ㅇ'로 시작하는 줄
      - 직전 줄과의 세로 간격이 LINE_GAP_THRESHOLD를 넘는 줄(불릿 없는 별개 항목)
    그 외(불릿 없고 간격도 좁은 줄)는 워드랩 연속 줄로 보고 직전 항목에 이어붙인다.

    Returns: [(top, bottom, text, is_group_intro), ...]
        is_group_intro: 'o/ㅇ'로 시작하면서 바로 다음 논리 줄이 '-'로 시작하는
        하위 항목 묶음용 소제목인 경우 True (그 자체는 체크 대상이 아님).
    """
    groups: list = []
    for top, bottom, text in raw_lines:
        gap = (top - groups[-1][1]) if groups else None
        starts_new = (
            not groups
            or BULLET_DASH_RE.match(text)
            or BULLET_O_RE.match(text)
            or gap > LINE_GAP_THRESHOLD
        )
        if starts_new:
            groups.append([top, bottom, text])
        else:
            groups[-1][1] = bottom
            groups[-1][2] = (groups[-1][2] + ' ' + text).strip()

    result = []
    for i, (top, bottom, text) in enumerate(groups):
        is_o = bool(BULLET_O_RE.match(text))
        next_is_dash = (i + 1 < len(groups)) and bool(BULLET_DASH_RE.match(groups[i + 1][2]))
        result.append((top, bottom, text, is_o and next_is_dash))
    return result


def _nearest_index(top: float, tops: list) -> int:
    diffs = [abs(top - t) for t in tops]
    return diffs.index(min(diffs))


def _validate(checked: dict, reason_filled: bool) -> Optional[str]:
    checked_count = sum(1 for v in checked.values() if v)
    if checked_count == 0:
        return "체크된 항목이 없습니다."
    if checked_count >= 2:
        return "체크박스가 2개 이상 선택되어 있습니다."
    for col in REASON_REQUIRED_FOR:
        if checked.get(col) and not reason_filled:
            return f"'{col}' 선택 시 사유 및 대체기술을 작성해야 합니다."
    return None


def check_tech_apply(pdf_path: str) -> dict:
    """
    기술적용결과표 PDF를 페이지별로 순회하며 체크 표를 검증한다.

    Returns:
        {
          "total": int,
          "error_count": int,
          "items": [
            {"page": int, "item_text": str, "checked": dict,
             "checked_count": int, "reason_filled": bool, "error": str|None},
            ...
          ],
        }
    """
    items: list = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for table in page.find_tables(TABLE_SETTINGS):
                grid = table.extract()
                header = _find_header_zones(table, grid)
                if header is None:
                    continue
                zones, n_header_rows = header

                for row_i in range(n_header_rows, len(table.rows)):
                    row = table.rows[row_i]
                    grid_row = grid[row_i] if row_i < len(grid) else []

                    item_bbox, item_text_flat = _resolve_cell(row, grid_row, zones, "item")
                    category_bbox, category_text = _resolve_cell(row, grid_row, zones, "category")

                    # '기본 지침'/'세부 기술 지침' 구분선 행은 건너뜀
                    if (category_text or item_text_flat) in SKIP_LABELS_EXACT:
                        continue

                    # '항목' 칸이 비어있으면(구분/항목 셀이 병합된 행 — 예: 기본
                    # 지침 설명문) '구분' 칸을 대신 사용한다
                    if not item_bbox:
                        item_bbox = category_bbox
                    if not item_bbox:
                        continue

                    logical_lines = _group_logical_lines(_text_lines(page, item_bbox))
                    item_lines = [(top, bottom, text)
                                  for top, bottom, text, is_intro in logical_lines
                                  if not is_intro]
                    if not item_lines:
                        continue

                    # 매칭 기준점은 줄의 top이 아니라 세로 중심(top~bottom 중간값)을
                    # 쓴다 — 사유란처럼 여러 줄에 걸친 블록은 첫 줄의 top만 보면
                    # 실제로는 다른 항목에 더 가까운데도 엉뚱한 항목에 붙는다.
                    item_centers = [(top + bottom) / 2 for top, bottom, _ in item_lines]
                    per_item_checked = [dict.fromkeys(CHECKBOX_COLUMNS, False) for _ in item_lines]
                    per_item_reason = ["" for _ in item_lines]

                    for colname in CHECKBOX_COLUMNS:
                        mark_bbox, _ = _resolve_cell(row, grid_row, zones, colname)
                        for mtop, mbottom, _ in _text_lines(page, mark_bbox):
                            idx = _nearest_index((mtop + mbottom) / 2, item_centers)
                            per_item_checked[idx][colname] = True

                    reason_bbox, _ = _resolve_cell(row, grid_row, zones, "사유")
                    reason_blobs = _group_logical_lines(_text_lines(page, reason_bbox))
                    for rtop, rbottom, rtext, _ in reason_blobs:
                        idx = _nearest_index((rtop + rbottom) / 2, item_centers)
                        per_item_reason[idx] = (per_item_reason[idx] + ' ' + rtext).strip()

                    row_x0, row_top, row_x1, row_bottom = row.bbox

                    for i, (item_top, item_bottom, item_text) in enumerate(item_lines):
                        checked = per_item_checked[i]
                        reason_filled = bool(per_item_reason[i])
                        error = _validate(checked, reason_filled)

                        # 하이라이트용 좌표: 가로는 행 전체 폭, 세로는 이 항목 줄의
                        # top~bottom만 — 페이지 크기 대비 비율(%)로 저장해서
                        # 프런트가 렌더링된 이미지 크기와 무관하게 위치를 잡을 수 있게 한다.
                        bbox = {
                            'left': round(row_x0 / page.width * 100, 3),
                            'top': round(max(item_top, row_top) / page.height * 100, 3),
                            'width': round((row_x1 - row_x0) / page.width * 100, 3),
                            'height': round((min(item_bottom, row_bottom) - max(item_top, row_top)) / page.height * 100, 3),
                        }

                        items.append(TechApplyCheckItem(
                            page=page_no,
                            item_text=item_text,
                            checked=checked,
                            checked_count=sum(1 for v in checked.values() if v),
                            bbox=bbox,
                            reason_filled=reason_filled,
                            error=error,
                        ))

    error_items = [it for it in items if it.error]

    return {
        "total": len(items),
        "error_count": len(error_items),
        "items": [
            {
                "page": it.page,
                "item_text": it.item_text,
                "checked": it.checked,
                "checked_count": it.checked_count,
                "reason_filled": it.reason_filled,
                "error": it.error,
                "bbox": it.bbox,
            }
            for it in items
        ],
    }
