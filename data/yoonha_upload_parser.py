"""
yoonha_upload_parser.py
-----------------------
Workit — 사용자 업로드 파일 파싱
data/uploads/raw/ 의 pdf/hwpx 파일을
data/uploads/parsed/ 에 JSON으로 저장한다.

실행:
    # 전체 일괄 처리
    py data/uploads/raw/yoonha_upload_parser.py

    # 특정 파일 하나
    py data/uploads/raw/yoonha_upload_parser.py data/uploads/raw/사업수행계획서.pdf
"""

import sys
from pathlib import Path

# data/uploads/raw/ 에 있으므로 세 단계 위가 프로젝트 루트
_PROJECT_ROOT      = Path(__file__).resolve().parent.parent
_DATA_DIR          = _PROJECT_ROOT / "data"
sys.path.insert(0, str(_DATA_DIR))

from yoonha_deliver_parser import parse_file, save_parsed_json


# ──────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────
UPLOADS_RAW_DIR    = _DATA_DIR / "uploads" / "raw"
UPLOADS_PARSED_DIR = _DATA_DIR / "uploads" / "parsed"


# ──────────────────────────────────────────
# 단일 파일 파싱
# ──────────────────────────────────────────
def parse_upload(file_path: str | Path) -> Path:
    """
    단일 파일을 파싱해 uploads/parsed/ 에 JSON으로 저장한다.

    Returns
    -------
    Path : 저장된 JSON 파일 경로
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    print(f"[upload_parser] 파싱 중: {file_path.name} ...", end=" ")
    sections = parse_file(file_path)

    UPLOADS_PARSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = UPLOADS_PARSED_DIR / (file_path.stem + ".json")
    save_parsed_json(sections, out_path)

    print(f"{len(sections)}개 섹션 → {out_path.name}")
    return out_path


# ──────────────────────────────────────────
# 전체 일괄 파싱
# ──────────────────────────────────────────
def parse_all_uploads(
    uploads_raw_dir: str | Path = UPLOADS_RAW_DIR,
) -> dict[str, Path]:
    """
    uploads/raw/ 폴더의 모든 pdf/hwpx 파일을 일괄 파싱한다.

    Returns
    -------
    dict[파일명 → 저장된 JSON 경로]
    """
    uploads_raw_dir = Path(uploads_raw_dir)

    if not uploads_raw_dir.exists():
        print(f"[upload_parser] 폴더가 없습니다: {uploads_raw_dir}")
        return {}

    target_files = sorted([
        f for f in uploads_raw_dir.iterdir()
        if f.suffix.lower() in (".pdf", ".hwpx")
    ])

    if not target_files:
        print(f"[upload_parser] 처리할 파일이 없습니다: {uploads_raw_dir}")
        return {}

    print(f"[upload_parser] 총 {len(target_files)}개 파일 파싱 시작")
    print("=" * 60)

    results = {}
    for file_path in target_files:
        try:
            out_path = parse_upload(file_path)
            results[file_path.name] = out_path
        except Exception as e:
            print(f"실패 ({e})")

    print(f"\n[upload_parser] 완료: {len(results)}/{len(target_files)}개 파일 처리됨")
    print(f"[upload_parser] 저장 위치: {UPLOADS_PARSED_DIR}")
    return results


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 인자 없음 → 전체 일괄 처리
        parse_all_uploads()
    else:
        # 특정 파일 하나
        parse_upload(sys.argv[1])