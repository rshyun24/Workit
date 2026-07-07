"""
yoonha_build_hn_seed.py
========================
train_all.jsonl → hard_negative_seed.json 자동 생성 스크립트

[ 왜 이 스크립트가 필요한가 ]
yoonha_hard_negative_miner.py는 아래 형태의 씨드 JSON이 필요합니다:
  {
    "query_id":    "Q001",
    "category":    "지체상금",
    "clause_text": "지연배상금은 계약금액의 1000분의 1.3...",  ← 계약서 조항
    "ground_truth": ["LCAR_75", "LCAE_90_3"]                  ← 정답 chunk_id
  }

train_all.jsonl에는 이미 아래 구조로 이 정보가 있습니다:
  user   → "카테고리: 지체상금\n검토조항: ..."  (clause_text, category)
  asst   → "판정: 일치\n근거: 지방계약법 시행규칙 제75조"  (ground_truth)

즉 train_all.jsonl을 파싱하면 씨드를 수동 작업 없이 만들 수 있습니다.

[ 필터링 기준 ]
- "검토조항" 없는 샘플 제외 (결과보고서 검토 태스크 등 다른 태스크)
- 판정 "판단보류" 제외 (ground_truth 특정 불가)
- 근거 "해당 근거 없음" 제외 (chunk_id 매핑 불가)
- 근거에서 chunk_id 변환 실패한 경우 제외

[ 실행 결과 ]
train_all.jsonl 278개 중 약 116개 → hard_negative_seed.json
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────

INPUT_PATH  = "./data/hn_seed/train_all.jsonl"          # train_all 경로
OUTPUT_PATH = "./data/hn_seed/hard_negative_seed.json"  # 출력 경로

# 법령 한국어명 → chunk_id 약어 매핑
# 긴 이름이 짧은 이름보다 먼저 매칭되도록 sorted_laws에서 길이 역순 정렬
LAW_NAME_TO_ABBR = {
    # 지방계약법 계열
    "지방자치단체를 당사자로 하는 계약에 관한 법률 시행규칙": "LCAR",
    "지방계약법 시행규칙": "LCAR",
    "지방자치단체를 당사자로 하는 계약에 관한 법률 시행령": "LCAE",
    "지방계약법 시행령": "LCAE",
    "지방자치단체를 당사자로 하는 계약에 관한 법률": "LCA",
    "지방계약법": "LCA",
    # 소프트웨어 진흥법 계열
    "소프트웨어 진흥법 시행령": "SWPAE",
    "소프트웨어진흥법 시행령": "SWPAE",
    "소프트웨어 진흥법": "SWPA",
    "소프트웨어진흥법": "SWPA",
    # 지방회계법 계열
    "지방회계법 시행령": "LARAE",
    "지방회계법": "LARA",
    # 용역계약 일반조건
    "지방자치단체 용역계약 일반조건": "PYG",
    # 공유재산법 계열
    "공유재산 및 물품관리법 시행령": "PPMAE",
    "공유재산 및 물품관리법": "PPMA",
    "공유재산법 시행령": "PPMAE",
    "공유재산법": "PPMA",
    "공공데이터의 제공 및 이용 활성화에 관한 법률": "PPMA",
    # 개인정보보호법 계열
    "개인정보보호법 시행령": "PIPAE",
    "개인정보 보호법 시행령": "PIPAE",
    "개인정보보호법": "PIPA",
    "개인정보 보호법": "PIPA",
}
# 긴 이름 먼저 매칭해야 "지방계약법 시행령"이 "지방계약법"으로 잘못 매칭되는 걸 방지
SORTED_LAWS = sorted(LAW_NAME_TO_ABBR.items(), key=lambda x: -len(x[0]))


# ───────────────────────────────────────────────
# 1. 법령명 → chunk_id 변환
# ───────────────────────────────────────────────

def normalize_pyg_id(law_name_jo: str) -> str:
    """
    PYG(지방자치단체 용역계약 일반조건) chunk_id 정규화.

    train_all의 근거 표현:
      "지방자치단체 용역계약 일반조건 제9장_제8절_제1항_다"
    목표 chunk_id:
      "PYG_9_8_1_다"

    변환 규칙:
      "제N장" / "제N절" / "제N항" → 숫자 N만 추출
      "가", "나", "다" 등 한글 목차 → 그대로 유지
    """
    # "지방자치단체 용역계약 일반조건 " 앞부분 제거
    jo_part = re.sub(r"^.*?일반조건\s*", "", law_name_jo).strip()
    # "제9장_제8절_제1항_다" → ["제9장", "제8절", "제1항", "다"]
    parts = jo_part.split("_")
    normalized = []
    for p in parts:
        p = p.strip()
        # "제9장", "제8절", "제1항" → "9", "8", "1"
        m = re.match(r"^제(\d+)[장절항]$", p)
        if m:
            normalized.append(m.group(1))
        else:
            # "가", "나", "다", "바" 등 → 그대로
            normalized.append(p)
    return "PYG_" + "_".join(normalized)


def law_name_to_chunk_id(law_name_jo: str) -> str | None:
    """
    근거 조항 표현 문자열을 chunk_id로 변환합니다.

    입력 예시:
      "지방계약법 시행규칙 제75조"           → "LCAR_75"
      "지방계약법 시행령 제67조제1항"         → "LCAE_67_1"
      "지방계약법 제30조의2제1항제1호"        → "LCA_30_의2_1_1"
      "지방자치단체 용역계약 일반조건 제9장_제8절_제1항_다" → "PYG_9_8_1_다"

    Returns:
        chunk_id 문자열, 또는 변환 실패 시 None
    """
    # 법령 약어 결정 (긴 이름 우선 매칭)
    abbr = None
    for name, a in SORTED_LAWS:
        if name in law_name_jo:
            abbr = a
            break
    if not abbr:
        return None  # 알 수 없는 법령

    # PYG는 별도 처리
    if abbr == "PYG":
        return normalize_pyg_id(law_name_jo)

    # 일반 법령: "제30조의2제1항제1호" 패턴 파싱
    # 조번호(필수) + 의X(선택) + 항(선택) + 호(선택)
    jo_match = re.search(
        r"제(\d+)조(?:의(\d+))?(?:제(\d+)항)?(?:제(\d+)호)?",
        law_name_jo
    )
    if not jo_match:
        return None

    jo   = jo_match.group(1)   # 조번호 (필수)
    ui   = jo_match.group(2)   # 의X (선택, 예: 30조의2 → "2")
    hang = jo_match.group(3)   # 항 (선택)
    ho   = jo_match.group(4)   # 호 (선택)

    # 조번호 문자열 구성 (의X 있으면 포함)
    jo_str = f"{jo}_의{ui}" if ui else jo

    parts = [abbr, jo_str]
    if hang: parts.append(hang)
    if ho:   parts.append(ho)
    return "_".join(parts)


# ───────────────────────────────────────────────
# 2. 단일 샘플 파싱
# ───────────────────────────────────────────────

def parse_sample(item: dict, query_id: str) -> dict | None:
    """
    train_all.jsonl의 단일 샘플을 씨드 형식으로 변환합니다.

    필터링 조건 (None 반환):
      - user 메시지에 "검토조항" 없음 → 다른 태스크 (결과보고서 검토 등)
      - 판정이 "판단보류" → ground_truth 특정 불가
      - 근거가 "해당 근거 없음" → chunk_id 변환 불가
      - 근거에서 chunk_id 변환 실패 → 알 수 없는 법령

    근거가 여러 개인 경우:
      train_all에서 근거는 단일 조항만 나오지만,
      참고조항 블록에 여러 조항이 있으면 모두 ground_truth에 포함합니다.
      (실제 계약 검토 시 여러 조항이 동시 근거가 될 수 있음)

    Args:
        item     : jsonl 한 줄을 파싱한 dict
        query_id : "Q001" 형식 ID

    Returns:
        씨드 dict 또는 필터링 시 None
    """
    msgs = item.get("messages", [])
    if len(msgs) < 3:
        return None

    user_content = msgs[1]["content"]
    asst_content = msgs[2]["content"]

    # ── 필터 1: 검토조항 없는 샘플 제외 ──────────────────────
    if "검토조항" not in user_content:
        return None

    # ── 필터 2: 판단보류 제외 ────────────────────────────────
    판정_match = re.search(r"판정:\s*(\S+)", asst_content)
    if not 판정_match:
        return None
    if "판단보류" in 판정_match.group(1):
        return None

    # ── 필터 3: 근거 없음 제외 ───────────────────────────────
    근거_match = re.search(r"근거:\s*(.+)", asst_content)
    if not 근거_match:
        return None
    근거_text = 근거_match.group(1).strip()
    if "없음" in 근거_text:
        return None

    # ── clause_text 추출 ──────────────────────────────────────
    # "검토조항: ...텍스트...\n\n참고조항:" 사이의 텍스트
    clause_match = re.search(r"검토조항:\s*(.+?)(?:\n\n참고조항|\Z)", user_content, re.DOTALL)
    if not clause_match:
        return None
    clause_text = clause_match.group(1).strip()

    # ── category 추출 ─────────────────────────────────────────
    category_match = re.search(r"카테고리:\s*(\S+)", user_content)
    category = category_match.group(1) if category_match else ""

    # ── ground_truth 추출 ─────────────────────────────────────
    # 방법 1: assistant 근거 필드 (가장 신뢰도 높음)
    chunk_id = law_name_to_chunk_id(근거_text)

    ground_truth = []
    if chunk_id:
        ground_truth.append(chunk_id)

    # 방법 2: user 참고조항 블록에서 추가 chunk_id 수집
    # "[1] 법령명 — 텍스트" 형태로 여러 참고조항이 있을 수 있음
    ref_blocks = re.findall(
        r"\[\d+\]\s*(.*?)\s*[—–-]+",
        user_content
    )
    for law_name_jo in ref_blocks:
        cid = law_name_to_chunk_id(law_name_jo.strip())
        # assistant 근거와 일치하는 것만 ground_truth에 포함
        # (참고조항 전부가 정답은 아니고, 근거와 같은 법령·조항만 포함)
        if cid and cid == chunk_id and cid not in ground_truth:
            ground_truth.append(cid)

    # ground_truth가 비어있으면 chunk_id 변환 실패 → 제외
    if not ground_truth:
        return None

    return {
        "query_id":       query_id,
        "category":       category,
        "clause_text":    clause_text,
        "ground_truth":   ground_truth,
        "hard_negatives": [],   # mining 후 채워질 필드
        # 디버깅용: 원본 근거 표현 보존 (mining 후 삭제해도 됨)
        "_source_근거":   근거_text,
        "_source_판정":   판정_match.group(1),
    }


# ───────────────────────────────────────────────
# 3. MAIN
# ───────────────────────────────────────────────

def main():
    input_path  = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)

    # 출력 디렉토리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # train_all.jsonl 로드
    print(f"[load] {input_path}")
    with open(input_path, encoding="utf-8") as f:
        raw_items = [json.loads(line) for line in f if line.strip()]
    print(f"  총 {len(raw_items)}개 샘플 로드")

    # 파싱 + 필터링
    seed_items = []
    skip_stats = defaultdict(int)

    for i, item in enumerate(raw_items):
        msgs = item.get("messages", [])
        user = msgs[1]["content"] if len(msgs) > 1 else ""
        asst = msgs[2]["content"] if len(msgs) > 2 else ""

        # 필터링 사유별 카운트 (디버깅용)
        if "검토조항" not in user:
            skip_stats["검토조항_없음"] += 1
            continue
        판정 = re.search(r"판정:\s*(\S+)", asst)
        if not 판정:
            skip_stats["판정_없음"] += 1
            continue
        if "판단보류" in 판정.group(1):
            skip_stats["판단보류"] += 1
            continue
        근거 = re.search(r"근거:\s*(.+)", asst)
        if not 근거 or "없음" in 근거.group(1):
            skip_stats["근거_없음"] += 1
            continue

        query_id = f"Q{len(seed_items) + 1:03d}"
        result = parse_sample(item, query_id)

        if result is None:
            skip_stats["chunk_id_변환_실패"] += 1
            continue

        seed_items.append(result)

    # 결과 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(seed_items, f, ensure_ascii=False, indent=2)

    # 통계 출력
    print(f"\n[결과]")
    print(f"  유효 씨드       : {len(seed_items)}개")
    print(f"  필터링 제외     : {sum(skip_stats.values())}개")
    for reason, cnt in sorted(skip_stats.items(), key=lambda x: -x[1]):
        print(f"    {reason:<22} {cnt}개")

    # 카테고리 분포
    from collections import Counter
    cat_dist = Counter(it["category"] for it in seed_items)
    print(f"\n  카테고리 분포:")
    for cat, cnt in sorted(cat_dist.items(), key=lambda x: -x[1]):
        print(f"    {cat:<20} {cnt}개")

    print(f"\n[저장] {output_path}")
    print("\n다음 단계:")
    print("  py data/hn_seed/yoonha_hard_negative_miner.py \\")
    print(f"     --seed {output_path} \\")
    print("     --output data/hn_seed/hard_negatives_output.json")


if __name__ == "__main__":
    main()