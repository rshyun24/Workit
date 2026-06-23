import fitz
import re

HANG_CHARS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"
HANG_MAP = {c: i + 1 for i, c in enumerate(HANG_CHARS)}

# 호(號) 패턴: 줄 맨 앞이 "1." "2." 형태로 시작.
# 표지 목록(예: "1. 계약명 :")과 구분하기 위해, 이 패턴은 항상 어떤 조/항의
# "영역 범위 안"에서만 호로 인정한다(extract_clause_positions 내부에서 범위 검사).
HO_PATTERN = re.compile(r"^(\d{1,2})\.\s*")


def reconstruct_line_text(line):
    """
    글자(char) 단위로 x좌표 기준 재정렬해서 텍스트 순서를 복원한다.
    """
    chars = []
    for span in line["spans"]:
        if "chars" in span:
            for c in span["chars"]:
                chars.append((c["c"], c["bbox"][0]))
        else:
            chars.append((span["text"], span["bbox"][0]))
    chars.sort(key=lambda x: x[1])
    return "".join(c for c, _ in chars)


def _split_bbox_by_page(page_num: int, x0: float, y0: float, width: float,
                         height: float, doc: "fitz.Document") -> list[dict]:
    """
    하나의 논리적 영역(조/항/호)이 페이지 경계를 넘는 경우,
    페이지별로 bbox를 잘라서 여러 조각으로 반환한다.
    """
    fragments = []
    remaining_height = height
    cur_page = page_num
    cur_y = y0

    max_pages = len(doc)

    while remaining_height > 0 and cur_page <= max_pages:
        page_height = doc[cur_page - 1].rect.height
        available = page_height - cur_y

        if available <= 0:
            cur_page += 1
            cur_y = 0
            continue

        take = min(available, remaining_height)

        fragments.append({
            "page": cur_page,
            "x": x0,
            "y": cur_y,
            "width": width,
            "height": take,
        })

        remaining_height -= take
        cur_page += 1
        cur_y = 0

    return fragments


def _logical_height(doc, start_page, start_y, end_page, end_y) -> float:
    """
    start_page/start_y 부터 end_page/end_y 까지의 "논리적 높이"를 계산한다.
    페이지가 다르면 중간 페이지들의 전체 높이를 합산한다.
    """
    if start_page == end_page:
        return max(0.0, end_y - start_y)

    height = doc[start_page - 1].rect.height - start_y
    for p in range(start_page + 1, end_page):
        height += doc[p - 1].rect.height
    height += end_y
    return max(0.0, height)


def _in_range(page, y, start_page, start_y, end_page, end_y) -> bool:
    """page/y 좌표가 (start_page,start_y) ~ (end_page,end_y) 범위 안에 있는지 검사."""
    if page < start_page or page > end_page:
        return False
    if page == start_page and y < start_y:
        return False
    if page == end_page and y >= end_y:
        return False
    return True


def _register_with_fragments(positions, key, page_num, x0, y0, width, height, doc):
    fragments = _split_bbox_by_page(page_num, x0, y0, width, height, doc)
    positions[key] = {"fragments": fragments}


def extract_clause_positions(pdf_path: str) -> dict:
    """
    조항(조/항/호) 번호 → 위치 정보 매핑 생성.

    반환 형식:
      {
        "제2조": {"fragments": [ {page, x, y, width, height}, ... ]},
        "제2조제1항": {"fragments": [ ... ]},
        "제2조제1항제1호": {"fragments": [ ... ]},
        ...
      }

    - "제N조(제목)" 패턴을 줄 안 어디에서든 찾되, 바로 앞에 "법"이 붙어있으면
      (타 법령 인용) 제외하고, 번호 순서가 크게 벗어나면 제외한다.
    - 각 조 영역 안에서 ①②③ 원문자를 찾아 항 단위로도 좌표를 등록한다.
    - 각 항(없으면 조) 영역 안에서 "1. 2. 3." 형태의 호를 찾아 호 단위로도 등록한다.
      표지의 목록(예: "1. 계약명 :")은 조/항 영역 밖이라 자동으로 제외된다.
    - 항/호가 없으면 상위 단위(조 또는 항)까지만 등록한다.
    - 하나의 단위가 페이지 경계를 넘으면 fragments가 여러 개로 분할된다.
    """
    doc = fitz.open(pdf_path)

    header_pattern = re.compile(r"제(\d+)조(?:의(\d+))?\s*\(([^)]*)\)")
    hang_pattern = re.compile(r"^[" + HANG_CHARS + r"]")

    raw_candidates = []   # 조 후보: [(num, clause_number, page_num, x0, y0), ...]
    hang_candidates = []  # 항 후보: [(page_num, x0, y0, hang_char), ...]
    ho_candidates = []    # 호 후보: [(page_num, x0, y0, ho_num), ...]

    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("rawdict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = reconstruct_line_text(line)
                stripped = line_text.strip()

                # 조 헤더 탐지
                for m in header_pattern.finditer(line_text):
                    prefix = line_text[max(0, m.start() - 5):m.start()]
                    if re.search(r"법\s*$", prefix):
                        continue  # 타 법령 인용 제외

                    num = int(m.group(1))
                    sub = m.group(2)
                    clause_number = f"제{m.group(1)}조" + (f"의{sub}" if sub else "")

                    x0 = min(s["bbox"][0] for s in line["spans"])
                    y0 = min(s["bbox"][1] for s in line["spans"])
                    raw_candidates.append((num, clause_number, page_num, x0, y0))

                # 항 헤더 탐지 (줄이 ①②③ 으로 시작)
                hm = hang_pattern.match(stripped)
                if hm:
                    hang_char = hm.group(0)
                    x0 = min(s["bbox"][0] for s in line["spans"])
                    y0 = min(s["bbox"][1] for s in line["spans"])
                    hang_candidates.append((page_num, x0, y0, hang_char))
                    continue  # 항 헤더 줄은 호 패턴 검사 생략(원문자가 숫자.패턴에 안 걸리므로 사실 무관)

                # 호 탐지 (줄이 "1." "2." 형태로 시작) — 범위 검증은 뒤에서 한다
                hom = HO_PATTERN.match(stripped)
                if hom:
                    ho_num = int(hom.group(1))
                    x0 = min(s["bbox"][0] for s in line["spans"])
                    y0 = min(s["bbox"][1] for s in line["spans"])
                    ho_candidates.append((page_num, x0, y0, ho_num))

    # ── 조 헤더 정렬 + 필터링 ──
    raw_candidates.sort(key=lambda x: (x[2], x[4]))

    filtered = []  # [(clause_number, page_num, x0, y0), ...]
    last_num = 0
    for num, clause_number, page_num, x0, y0 in raw_candidates:
        if num >= last_num and num <= last_num + 5:
            filtered.append((clause_number, page_num, x0, y0))
            last_num = num

    if len(filtered) < 3:
        return {}

    hang_candidates.sort(key=lambda x: (x[0], x[2]))
    ho_candidates.sort(key=lambda x: (x[0], x[2]))

    positions: dict[str, dict] = {}

    for idx, (clause_number, page_num, x0, y0) in enumerate(filtered):
        next_clause = filtered[idx + 1] if idx + 1 < len(filtered) else None

        if next_clause and next_clause[1] == page_num:
            clause_end_page, clause_end_y = page_num, next_clause[3]
        elif next_clause:
            clause_end_page, clause_end_y = next_clause[1], next_clause[3]
        else:
            clause_end_page = len(doc)
            clause_end_y = doc[clause_end_page - 1].rect.height

        page_width = doc[page_num - 1].rect.width

        hangs_in_clause = [
            h for h in hang_candidates
            if _in_range(h[0], h[2], page_num, y0, clause_end_page, clause_end_y)
        ]

        clause_total_height = _logical_height(doc, page_num, y0, clause_end_page, clause_end_y)
        _register_with_fragments(
            positions, clause_number, page_num, x0, y0,
            page_width - x0 - 20, clause_total_height, doc
        )

        if not hangs_in_clause:
            # 항이 없는 조 → 조 영역 안에서 바로 호를 찾는다
            hos_in_unit = [
                h for h in ho_candidates
                if _in_range(h[0], h[2], page_num, y0, clause_end_page, clause_end_y)
            ]
            _register_ho_list(positions, clause_number, hos_in_unit, page_width,
                               page_num, y0, clause_end_page, clause_end_y, doc)
            continue

        # 항 단위 좌표 등록 + 항 안에서 호 등록
        for h_idx, (h_page, h_x, h_y, h_char) in enumerate(hangs_in_clause):
            next_h = hangs_in_clause[h_idx + 1] if h_idx + 1 < len(hangs_in_clause) else None

            if next_h:
                h_end_page, h_end_y = next_h[0], next_h[2]
            else:
                h_end_page, h_end_y = clause_end_page, clause_end_y

            h_height = _logical_height(doc, h_page, h_y, h_end_page, h_end_y)
            hang_num = HANG_MAP.get(h_char, h_idx + 1)
            hang_number = f"{clause_number}제{hang_num}항"

            _register_with_fragments(
                positions, hang_number, h_page, h_x, h_y,
                page_width - h_x - 20, h_height, doc
            )

            hos_in_hang = [
                h for h in ho_candidates
                if _in_range(h[0], h[2], h_page, h_y, h_end_page, h_end_y)
            ]
            _register_ho_list(positions, hang_number, hos_in_hang, page_width,
                               h_page, h_y, h_end_page, h_end_y, doc)

    return positions


def _register_ho_list(positions, parent_number, hos_in_unit, page_width,
                       unit_start_page, unit_start_y, unit_end_page, unit_end_y, doc):
    """
    parent_number(조 또는 항) 영역 안에서 발견된 호(號) 후보 리스트를
    "{parent_number}제N호" 키로 등록한다.
    """
    for ho_idx, (ho_page, ho_x, ho_y, ho_num) in enumerate(hos_in_unit):
        next_ho = hos_in_unit[ho_idx + 1] if ho_idx + 1 < len(hos_in_unit) else None

        if next_ho:
            ho_end_page, ho_end_y = next_ho[0], next_ho[2]
        else:
            ho_end_page, ho_end_y = unit_end_page, unit_end_y

        ho_height = _logical_height(doc, ho_page, ho_y, ho_end_page, ho_end_y)
        ho_number = f"{parent_number}제{ho_num}호"

        _register_with_fragments(
            positions, ho_number, ho_page, ho_x, ho_y,
            page_width - ho_x - 20, ho_height, doc
        )


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    result = extract_clause_positions(path)
    for k, v in result.items():
        print(k, v)