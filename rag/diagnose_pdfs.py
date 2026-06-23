"""
여러 PDF에서 '제N조' 헤더가 정상 추출되는지 일괄 진단.
- 정상: '제1조(목적)' 처럼 숫자가 온전히 붙어서 나옴
- 비정상: '제조 목적' 처럼 숫자가 빠짐
"""
import fitz
import re
import sys
import glob


def reconstruct_line_text(line):
    chars = []
    for span in line["spans"]:
        if "chars" in span:
            for c in span["chars"]:
                chars.append((c["c"], c["bbox"][0]))
        else:
            chars.append((span["text"], span["bbox"][0]))
    chars.sort(key=lambda x: x[1])
    return "".join(c for c, _ in chars)


def diagnose(pdf_path: str):
    doc = fitz.open(pdf_path)

    # "제N조" 형태(숫자 포함) vs "제조"(숫자 빠짐) 비교
    pattern_ok = re.compile(r"제\d+조")
    pattern_broken = re.compile(r"제조(?!\d)")  # "제조" 뒤에 숫자가 안 붙은 경우

    ok_count = 0
    broken_count = 0
    broken_samples = []

    for page in doc:
        blocks = page.get_text("rawdict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                text = reconstruct_line_text(line)
                ok_count += len(pattern_ok.findall(text))
                broken_matches = pattern_broken.findall(text)
                if broken_matches:
                    broken_count += len(broken_matches)
                    if len(broken_samples) < 3:
                        broken_samples.append(text.strip()[:40])

    total = ok_count + broken_count
    ratio = (broken_count / total * 100) if total else 0

    return {
        "file": pdf_path,
        "pages": len(doc),
        "ok": ok_count,
        "broken": broken_count,
        "broken_ratio": round(ratio, 1),
        "samples": broken_samples,
    }


if __name__ == "__main__":
    pdf_files = sys.argv[1:] if len(sys.argv) > 1 else glob.glob("media/contracts/docs/*표준계약서*.pdf")

    print(f"총 {len(pdf_files)}개 파일 진단\n")
    print(f"{'파일명':<60} {'정상':>6} {'깨짐':>6} {'깨짐비율':>8}")
    print("-" * 90)

    for path in pdf_files:
        try:
            result = diagnose(path)
            fname = result["file"].split("/")[-1][:55]
            print(f"{fname:<60} {result['ok']:>6} {result['broken']:>6} {result['broken_ratio']:>7}%")
            if result["broken"] > 0:
                for s in result["samples"]:
                    print(f"    예시: {s!r}")
        except Exception as e:
            print(f"{path}: 에러 발생 - {e}")