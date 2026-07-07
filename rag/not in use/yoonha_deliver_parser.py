"""
parser.py
---------
Workit — SW/SI 산출물 피드백 플랫폼
hwpx (한글) 및 PDF 파일에서 텍스트를 추출하고,
섹션 계층 구조(장 > 절 > 항)를 파싱한다.

지원 포맷:
  - .hwpx  : ZIP 내 Contents/section*.xml 의 hp:t 태그 파싱
  - .pdf   : pdfplumber 기반 페이지별 텍스트 추출

반환 구조 (ParsedSection):
  {
    "section_path": ["1. 문서 개요", "1.1 문서 목적"],
    "section_title": "1.1 문서 목적",
    "section_depth": 2,          # 1=장, 2=절, 3=항
    "text": "본 문서는 ...",
    "raw_lines": ["본 문서는 ...", ...]
  }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CLI 실행 방법
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # deliverable 폴더 전체 일괄 파싱 → data/structured/ 에 JSON 저장
  py data/parser.py

  # 특정 파일 하나만 파싱 (결과 터미널 출력)
  py data/parser.py "data/deliverable/테스트 결과보고서 양식.hwpx"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 저장 위치
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  data/structured/테스트 결과보고서 양식.json
  data/structured/테스트 설계서 양식.json
  data/structured/사업수행계획서 양식.json
  data/structured/최종결과보고서 양식.json
  data/structured/SK대학교_LMS구축_테스트결과보고서.json
  ... (deliverable 폴더 안의 파일명 그대로)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 다른 파일에서 import
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  from data.parser import parse_file, load_parsed_json

  # 파일 직접 파싱
  sections = parse_file("data/deliverable/테스트 결과보고서 양식.hwpx")

  # 저장된 JSON 불러오기 (이미 파싱한 경우)
  sections = load_parsed_json("data/structured/테스트 결과보고서 양식.json")
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore

# ─────────────────────────────────────────────
# 경로 상수
# ─────────────────────────────────────────────

# parser.py 가 data/ 안에 있다고 가정
# 프로젝트 루트 = parser.py 의 부모(data/)의 부모
_THIS_FILE   = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent          # C:\project\Workit
DELIVERABLE_DIR = _PROJECT_ROOT / "data" / "deliverable"
STRUCTURED_DIR  = _PROJECT_ROOT / "data" / "structured"


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class ParsedSection:
    section_path: list[str]
    section_title: str
    section_depth: int               # 1=장 / 2=절 / 3=항
    text: str
    raw_lines: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"[depth={self.section_depth}] {self.section_title!r} | {preview}..."

    def to_dict(self) -> dict:
        return {
            "section_path":  self.section_path,
            "section_title": self.section_title,
            "section_depth": self.section_depth,
            "text":          self.text,
            "raw_lines":     self.raw_lines,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ParsedSection":
        return cls(
            section_path  = d["section_path"],
            section_title = d["section_title"],
            section_depth = d["section_depth"],
            text          = d["text"],
            raw_lines     = d.get("raw_lines", []),
        )


# ─────────────────────────────────────────────
# 섹션 헤더 감지 정규식
# ─────────────────────────────────────────────

_RE_CHAP = re.compile(
    r"^(?:제\s*\d+\s*[장절]|"
    r"\d+\.\s+[^\d]|"
    r"\d+장\b)",
    re.UNICODE,
)
_RE_SEC  = re.compile(r"^\d+\.\d+\s+\S",       re.UNICODE)
_RE_SUB  = re.compile(r"^\d+\.\d+\.\d+\s+\S",  re.UNICODE)
_RE_TOC  = re.compile(r"^.{1,60}[\s·.]{3,}\d+\s*$")


def _detect_depth(line: str) -> Optional[int]:
    s = line.strip()
    if not s:
        return None
    if _RE_SUB.match(s): return 3
    if _RE_SEC.match(s): return 2
    if _RE_CHAP.match(s): return 1
    return None


# ─────────────────────────────────────────────
# hwpx 파서
# ─────────────────────────────────────────────

_HWPX_NS_P   = "http://www.hancom.co.kr/hwpml/2011/paragraph"
_HWPX_NS_P10 = "http://www.hancom.co.kr/hwpml/2016/paragraph"

def _extract_hwpx_lines(hwpx_path: str | Path) -> list[str]:
    from xml.etree import ElementTree as ET
    lines: list[str] = []
    with zipfile.ZipFile(hwpx_path, "r") as zf:
        section_files = sorted(
            [n for n in zf.namelist() if re.match(r"Contents/section\d+\.xml", n)]
        )
        for sec_file in section_files:
            xml_bytes = zf.read(sec_file)
            root = ET.fromstring(xml_bytes.decode("utf-8", errors="ignore"))
            for para in root.iter():
                if para.tag not in (
                    f"{{{_HWPX_NS_P}}}p",
                    f"{{{_HWPX_NS_P10}}}p",
                ):
                    continue
                parts = []
                for t_tag in para.iter():
                    tag_local = t_tag.tag.split("}")[-1] if "}" in t_tag.tag else t_tag.tag
                    if tag_local == "t" and t_tag.text:
                        parts.append(t_tag.text)
                line = "".join(parts).strip()
                if line:
                    lines.append(line)
    return lines


# ─────────────────────────────────────────────
# PDF 파서
# ─────────────────────────────────────────────

def _extract_pdf_lines(pdf_path: str | Path) -> list[str]:
    if pdfplumber is None:
        raise ImportError("pdfplumber 미설치. `pip install pdfplumber` 실행 필요.")
    lines: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if not page_text:
                continue
            for raw_line in page_text.splitlines():
                stripped = raw_line.strip()
                if stripped:
                    lines.append(stripped)
    return lines


# ─────────────────────────────────────────────
# 라인 → 섹션 분할
# ─────────────────────────────────────────────

def _lines_to_sections(lines: list[str]) -> list[ParsedSection]:
    filtered = [l for l in lines if not _RE_TOC.match(l)]

    sections: list[ParsedSection] = []
    path_stack: dict[int, str] = {}
    current_depth: Optional[int] = None
    current_lines: list[str] = []
    current_title: str = "서문"

    def _flush():
        nonlocal current_lines, current_title, current_depth
        if not current_lines:
            return
        depth = current_depth or 1
        path_stack[depth] = current_title
        for d in list(path_stack.keys()):
            if d > depth:
                del path_stack[d]
        ordered = [path_stack[d] for d in sorted(path_stack.keys())]
        sections.append(ParsedSection(
            section_path  = ordered[:],
            section_title = current_title,
            section_depth = depth,
            text          = "\n".join(current_lines),
            raw_lines     = current_lines[:],
        ))
        current_lines = []

    for line in filtered:
        depth = _detect_depth(line)
        if depth is not None:
            _flush()
            current_depth = depth
            current_title = line.strip()
            current_lines = [line.strip()]
        else:
            current_lines.append(line)

    _flush()
    return sections


# ─────────────────────────────────────────────
# JSON 저장 / 불러오기
# ─────────────────────────────────────────────

def save_parsed_json(sections: list[ParsedSection], out_path: str | Path) -> None:
    """파싱 결과를 JSON 파일로 저장."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in sections], f, ensure_ascii=False, indent=2)


def load_parsed_json(json_path: str | Path) -> list[ParsedSection]:
    """저장된 JSON 파일에서 ParsedSection 리스트 복원."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [ParsedSection.from_dict(d) for d in data]


# ─────────────────────────────────────────────
# 공개 인터페이스
# ─────────────────────────────────────────────

def parse_file(file_path: str | Path) -> list[ParsedSection]:
    """hwpx 또는 PDF 파일을 파싱해 ParsedSection 리스트 반환."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    suffix = path.suffix.lower()
    if suffix == ".hwpx":
        lines = _extract_hwpx_lines(path)
    elif suffix == ".pdf":
        lines = _extract_pdf_lines(path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {suffix} (hwpx / pdf 만 지원)")
    return _lines_to_sections(lines)


def parse_all_deliverables(
    deliverable_dir: str | Path = DELIVERABLE_DIR,
    structured_dir:  str | Path = STRUCTURED_DIR,
) -> dict[str, list[ParsedSection]]:
    """
    deliverable 폴더 안의 hwpx / pdf 파일을 전부 파싱하고
    structured 폴더에 JSON으로 저장한다.

    Returns
    -------
    dict[파일명 → List[ParsedSection]]
    """
    deliverable_dir = Path(deliverable_dir)
    structured_dir  = Path(structured_dir)
    structured_dir.mkdir(parents=True, exist_ok=True)

    target_files = sorted(
        [f for f in deliverable_dir.iterdir()
         if f.suffix.lower() in (".hwpx", ".pdf")]
    )

    if not target_files:
        print(f"[parser] deliverable 폴더에 파싱할 파일이 없습니다: {deliverable_dir}")
        return {}

    results: dict[str, list[ParsedSection]] = {}

    for file_path in target_files:
        print(f"[parser] 파싱 중: {file_path.name} ...", end=" ")
        try:
            sections = parse_file(file_path)
            out_path = structured_dir / (file_path.stem + ".json")
            save_parsed_json(sections, out_path)
            results[file_path.name] = sections
            print(f"{len(sections)}개 섹션 → {out_path.name}")
        except Exception as e:
            print(f"실패 ({e})")

    print(f"\n[parser] 완료: {len(results)}/{len(target_files)}개 파일 처리됨")
    print(f"[parser] 저장 위치: {structured_dir}")
    return results


# ─────────────────────────────────────────────
# CLI 엔트리포인트
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # 인자 없음 → deliverable 폴더 전체 일괄 처리
        parse_all_deliverables()

    else:
        # 특정 파일 하나 → 터미널 출력
        target = sys.argv[1]
        result = parse_file(target)
        print(f"\n총 {len(result)}개 섹션 파싱 완료\n{'='*60}")
        for i, sec in enumerate(result, 1):
            print(f"\n[{i}] depth={sec.section_depth} | {' > '.join(sec.section_path)}")
            print(f"     {sec.text[:120].replace(chr(10), ' ')}...")