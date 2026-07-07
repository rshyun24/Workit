"""
Workit - 법령 chunk JSON 폴더 통합 스크립트

각 폴더 안에 법령별로 흩어진 JSON 파일들(예: 개인정보보호법.json, 지방계약법.json ...)의
"articles" 리스트를 뽑아서, 폴더 하나당 flat한 chunk 리스트 파일 하나로 합친다.
(chunks_ho.json / chunks_jo.json 처럼 Colab 노트북에 바로 업로드할 수 있는 형태)

input : 아래 FOLDERS에 지정된 4개 폴더
output: OUTPUT_DIR 안에 폴더당 파일 1개씩, 총 4개

사용법:
    python merge_chunks.py
"""
import json
from pathlib import Path

# 폴더 경로 → 출력 파일명
FOLDERS: dict[Path, str] = {
    Path("C:/project/Workit/data/structured_jo"):         "chunks_jo.json",
    Path("C:/project/Workit/data/structured_jo_fixedid"): "chunks_jo_fixedid.json",
    Path("C:/project/Workit/data/structured"):             "chunks_ho.json",
    Path("C:/project/Workit/data/structured_fixedid"):     "chunks_ho_fixedid.json",
}

OUTPUT_DIR = Path("C:/project/Workit/data/merged")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def merge_folder(folder: Path) -> list[dict]:
    """폴더 안의 모든 *.json 파일에서 'articles'를 뽑아 하나의 리스트로 합친다."""
    all_chunks: list[dict] = []
    seen_ids: set[str] = set()
    dup_count = 0

    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        print(f"  [WARN] {folder} 안에 json 파일이 없습니다.")
        return all_chunks

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        articles = data.get("articles", [])
        added = 0
        for a in articles:
            cid = a.get("chunk_id")
            if cid in seen_ids:
                dup_count += 1
                continue
            seen_ids.add(cid)
            all_chunks.append(a)
            added += 1

        print(f"  [READ] {jf.name}: {added}개 (누적 {len(all_chunks)}개)")

    if dup_count:
        print(f"  [DEDUPE] chunk_id 중복 {dup_count}건 제거 (먼저 나온 것 유지)")

    return all_chunks


def main():
    summary = []
    for folder, out_name in FOLDERS.items():
        print(f"\n=== {folder} ===")
        if not folder.exists():
            print(f"  [SKIP] 폴더 없음: {folder}")
            continue

        chunks = merge_folder(folder)
        out_path = OUTPUT_DIR / out_name

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"  -> saved: {out_path} ({len(chunks)}개 chunk)")
        summary.append((out_name, len(chunks)))

    print("\n=== 요약 ===")
    for name, count in summary:
        print(f"  {name:28s} {count:>6,}개")
    print("\nDone! output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()