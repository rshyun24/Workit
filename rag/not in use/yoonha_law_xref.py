"""
Workit - 법령 간 cross_ref 병합 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
    chunks_ho.json 의 각 청크에 cross_refs 필드를 추가한다.
    cross_refs: 해당 청크가 참조하는 상위/연관 법령 chunk_id 목록.

실행:
    python yoonha_law_xref.py

입력:
    data/export/chunks_ho.json         ← yoonha_law_chunking_hierarchical.py 출력
    data/law_cross_refs.json  ← 법령 간 참조 관계 (수동 관리)

출력:
    data/export/chunks_ho_xref.json
        → cross_refs 필드가 추가된 버전.
        → Quantitative RAG에서 chunk_id 직접 lookup 시 사용.

cross_refs 필드:
    - 값: 참조 대상 chunk_id 문자열 리스트
    - 예: ["PIPA_33", "LCA_30_4"]
    - cross_ref가 없는 청크: 빈 리스트 []
    - chunk_id 목록만 저장 (텍스트 인라인 X — 업데이트 관리 및 payload 크기 고려)

법령 개정 시:
    law_cross_refs.json 업데이트 후 이 스크립트만 재실행하면 된다.
    (파서/청킹 재실행 불필요)
"""

import json
import time
from datetime import datetime
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 경로 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHUNKS_HO_PATH  = Path("data/export/chunks_ho.json")
CROSS_REF_PATH  = Path("data/law_cross_refs.json")
OUTPUT_PATH     = Path("data/export/chunks_ho_xref.json")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def now() -> str:
    return datetime.now().strftime("%H:%M:%S")

def elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}초" if s < 60 else f"{int(s//60)}분 {int(s%60)}초"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    t_total = time.time()

    print("=" * 60)
    print(f"  Cross-ref 병합  [{now()}]")
    print("=" * 60)

    # ── 1. 입력 파일 로드 ────────────────────────────────────
    if not CHUNKS_HO_PATH.exists():
        print(f"[ERROR] chunks_ho.json 없음: {CHUNKS_HO_PATH}")
        print("  → yoonha_law_chunking_hierarchical.py 먼저 실행하세요.")
        return

    if not CROSS_REF_PATH.exists():
        print(f"[ERROR] cross_ref 파일 없음: {CROSS_REF_PATH}")
        return

    print(f"\n[{now()}] 📂 chunks_ho.json 로드 중...")
    with open(CHUNKS_HO_PATH, encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)
    print(f"  → {len(chunks)}개 청크 로드")

    print(f"[{now()}] 📂 law_cross_refs.json 로드 중...")
    with open(CROSS_REF_PATH, encoding="utf-8") as f:
        cross_ref_map: dict[str, list[str]] = json.load(f)
    print(f"  → {len(cross_ref_map)}개 cross_ref 항목 로드")

    # ── 2. cross_refs 필드 병합 ──────────────────────────────
    print(f"\n[{now()}] 🔗 cross_refs 병합 중...")

    matched = 0
    fallback = 0
    unmatched = 0

    result: list[dict] = []
    for chunk in chunks:
        cid = chunk["chunk_id"]
        cross_refs = cross_ref_map.get(cid, [])

        if cross_refs:
            matched += 1
        else:
            # Fallback: map에 조 레벨 키가 있고 이 청크가 그 하위인 경우
            # 예: PIPAE_4_의2_0_1 → PIPAE_4_의2_ 로 시작하는 키 검색
            inherited = []
            for key, refs in cross_ref_map.items():
                if cid.startswith(key + "_"):
                    inherited.extend(refs)
            if inherited:
                cross_refs = list(dict.fromkeys(inherited))  # 중복 제거, 순서 유지
                fallback += 1
            else:
                unmatched += 1

        result.append({**chunk, "cross_refs": cross_refs})

    # ── 3. 통계 출력 ─────────────────────────────────────────
    print(f"  → 정확 매칭:      {matched}개")
    print(f"  → 조문 상속:      {fallback}개")
    print(f"  → cross_ref 없음: {unmatched}개")

    # ── 4. 저장 ──────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"\n[{now()}] 💾 저장 완료: {OUTPUT_PATH}  ({size_mb:.1f} MB)")
    print(f"  → Quantitative RAG: chunk_id lookup 시 cross_refs 필드로 연관 조문 직접 fetch.")
    print(f"\n총 소요: {elapsed(t_total)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

