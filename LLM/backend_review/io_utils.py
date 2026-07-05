from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_text(path: str | Path) -> str:
    """
    txt 파일 읽기
    """
    return Path(path).read_text(
        encoding="utf-8"
    )


def load_json(path: str | Path) -> Any:
    """
    json 파일 읽기
    """
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def write_json(
    path: str | Path,
    payload: Dict[str, Any],
) -> None:
    """
    결과 json 저장
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def get_section_content(section: Dict[str, Any]) -> str:
    """
    key-value 평가셋의 content 값을 문자열로 변환한다.

    지원 형식 1:
    {
      "content": "본문 내용"
    }

    지원 형식 2:
    {
      "content": {
        "pep_excerpt": "본문 내용"
      }
    }

    지원 형식 3:
    {
      "content": {
        "rfp_excerpt": "...",
        "extra": "..."
      }
    }
    """
    content = section.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        values = [
            str(value)
            for value in content.values()
            if value is not None
        ]
        return "\n".join(values).strip()

    return str(content or "")


def normalize_section_for_review(section: Dict[str, Any]) -> Dict[str, Any]:
    """
    새 key-value 평가셋을 기존 qa_agent/fewshot_agent가 읽을 수 있는 형태로 변환한다.

    기존 검수 코드는 content가 문자열이라고 가정하므로,
    content 내부의 pep_excerpt/rfp_excerpt/rpt_excerpt 값을 문자열 content로 평탄화한다.

    description, standard_structure, quality, mapping 정보는 보존한다.
    """
    normalized = dict(section)
    normalized["content"] = get_section_content(section)
    return normalized


def normalize_sections_for_review(parsed_sections: Any) -> Any:
    """
    parsed_sections 전체를 검수 가능한 형태로 변환한다.
    """
    if isinstance(parsed_sections, list):
        return [
            normalize_section_for_review(section)
            if isinstance(section, dict)
            else section
            for section in parsed_sections
        ]

    if isinstance(parsed_sections, dict):
        normalized: Dict[str, Any] = {}
        for key, value in parsed_sections.items():
            if isinstance(value, dict):
                normalized[key] = normalize_section_for_review(value)
            else:
                normalized[key] = value
        return normalized

    return parsed_sections


def get_standard_structure(section: Dict[str, Any]) -> List[str]:
    """
    key-value 평가셋의 필수 구성요소 목록을 반환한다.
    """
    values = section.get("standard_structure") or []

    if isinstance(values, list):
        return [str(value).strip() for value in values if str(value).strip()]

    if isinstance(values, str):
        return [values.strip()] if values.strip() else []

    return []


def get_quality_criteria(section: Dict[str, Any]) -> List[str]:
    """
    key-value 평가셋의 quality 목록을 반환한다.
    """
    values = section.get("quality") or []

    if isinstance(values, list):
        return [str(value).strip() for value in values if str(value).strip()]

    if isinstance(values, str):
        return [values.strip()] if values.strip() else []

    return []


def get_mapping_values(section: Dict[str, Any]) -> Dict[str, Any]:
    """
    RFP_mapping, PEP_mapping, RPT_mapping 같이 mapping으로 끝나는 필드를 모아서 반환한다.
    """
    return {
        key: value
        for key, value in section.items()
        if key.lower().endswith("_mapping")
    }
