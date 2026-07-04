"""
performance/parsers.py

과업수행계획서(PEP)를 정규식·키워드 기반으로 파싱하고,
RFP 파싱 결과와 구조적으로 비교한다. LLM 없이 동작한다.

PEP 코드 체계: PEP-01 ~ PEP-16 (노션 파싱 코드 기준)
"""

import re
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# PEP 섹션 코드 → 탐지 키워드 매핑
# ─────────────────────────────────────────────────────────────────────────────

_PEP_SECTION_KEYWORDS: list[tuple[str, str, list[str]]] = [
    ('PEP-01',    '사업명',                   ['사업명']),
    ('PEP-02',    '사업기간',                  ['사업기간', '사업 기간']),
    ('PEP-03-01', '추진배경',                  ['추진배경', '추진 배경', '사업 목적']),
    ('PEP-03-02', '목적',                      ['사업 목적', '추진목적', '목적']),
    ('PEP-04-01', '개발대상업무',               ['개발대상업무', '개발 대상 업무', '개발대상']),
    ('PEP-04-02', '개발 및 운영환경',           ['개발 및 운영환경', '개발환경', '운영환경']),
    ('PEP-04-03', '기타 사업범위',              ['기타 사업범위', '사업 범위 기타']),
    ('PEP-05-01', '총괄추진체계',               ['총괄추진체계', '총괄 추진체계', '추진체계']),
    ('PEP-05-02', '사업자 추진체계',            ['사업자 추진체계', '수행체계']),
    ('PEP-06',    '사업추진절차',               ['사업추진절차', '사업 추진 절차', '추진절차']),
    ('PEP-07',    '산출물계획',                 ['산출물계획', '산출물 계획', '산출물관리']),
    ('PEP-08',    '일정계획',                   ['일정계획', '일정 계획', '추진일정', '개발일정']),
    ('PEP-09',    '공정별 투입인력계획',         ['투입인력', '투입 인력', '인력계획', '인력 계획']),
    ('PEP-10',    '보고계획',                   ['보고계획', '보고 계획']),
    ('PEP-11-01', '표준화 항목',                ['표준화 계획', '표준화항목', '표준화 항목']),
    ('PEP-11-02', '정보화기반표준',              ['정보화기반표준', '정보화기반 표준']),
    ('PEP-11-03', '공공기관 DB표준화 지침',      ['DB표준화', 'DB 표준화', '데이터베이스 표준']),
    ('PEP-11-04', '전자정부 웹사이트 품질관리 지침', ['웹사이트 품질', '웹 품질', '전자정부 표준']),
    ('PEP-12',    '품질관리계획',               ['품질관리계획', '품질 관리 계획', '품질관리 계획']),
    ('PEP-13',    '위험관리계획',               ['위험관리계획', '위험 관리', '리스크 관리']),
    ('PEP-14',    '보안대책',                   ['보안대책', '보안 대책', '보안계획']),
    ('PEP-15',    '교육계획',                   ['교육계획', '교육 계획', '사용자 교육']),
    ('PEP-16',    '발주기관 협조요청사항',       ['협조요청', '협조 요청', '발주기관 협조']),
]


def _find_keyword_position(text: str, keywords: list[str]) -> int | None:
    best = None
    for kw in keywords:
        idx = text.find(kw)
        if idx != -1 and (best is None or idx < best):
            best = idx
    return best


def parse_execution_plan(text: str) -> dict:
    """
    과업수행계획서 텍스트를 파싱해 PEP 코드 체계 기반 JSON 반환.

    Args:
        text: extract_text()로 추출된 원문 텍스트

    Returns:
        {
            "PEP-01": {"label": "사업명", "content": "...", "found": True},
            "PEP-02": {"label": "사업기간", "content": "...", "found": True},
            ...
        }
    """
    # 각 코드의 시작 위치 탐색
    anchor_positions: list[tuple[str, str, int]] = []

    for code, label, keywords in _PEP_SECTION_KEYWORDS:
        pos = _find_keyword_position(text, keywords)
        if pos is not None:
            anchor_positions.append((code, label, pos))

    anchor_positions.sort(key=lambda x: x[2])

    sections: dict[str, Any] = {}

    for i, (code, label, start) in enumerate(anchor_positions):
        end = anchor_positions[i + 1][2] if i + 1 < len(anchor_positions) else len(text)
        content = text[start:end].strip()

        if len(content) > 2000:
            content = content[:2000] + ' [이하 생략]'

        sections[code] = {
            'label': label,
            'content': content,
            'found': True,
        }

    # 찾지 못한 섹션 채움
    found_codes = {item[0] for item in anchor_positions}
    for code, label, _ in _PEP_SECTION_KEYWORDS:
        if code not in found_codes:
            sections[code] = {'label': label, 'content': '', 'found': False}

    return sections


# ─────────────────────────────────────────────────────────────────────────────
# RFP ↔ PEP 구조적 비교 (LLM 없음)
# ─────────────────────────────────────────────────────────────────────────────

# RFP 코드 → 대응 PEP 코드 매핑
# 'required': True → 해당 PEP 항목이 없으면 미충족으로 분류
_RFP_TO_PEP_MAPPING: list[dict] = [
    {
        'rfp_code': 'RFP-01-02',
        'rfp_subtitle': '추진배경 및 필요성',
        'pep_codes': ['PEP-03-01', 'PEP-03-02'],
        'required': True,
        'description': '사업 목적 및 추진 배경',
    },
    {
        'rfp_code': 'RFP-01-04',
        'rfp_subtitle': '사업 범위',
        'pep_codes': ['PEP-04-01', 'PEP-04-02', 'PEP-04-03'],
        'required': True,
        'description': '개발 대상 업무 및 환경',
    },
    {
        'rfp_code': 'RFP-03-02',
        'rfp_subtitle': '추진 체계',
        'pep_codes': ['PEP-05-01', 'PEP-05-02'],
        'required': True,
        'description': '사업 추진 체계',
    },
    {
        'rfp_code': 'RFP-03-03',
        'rfp_subtitle': '추진일정',
        'pep_codes': ['PEP-08'],
        'required': True,
        'description': '일정 계획',
    },
    {
        'rfp_code': 'RFP-04-04-01',
        'rfp_subtitle': '기능 요구사항',
        'pep_codes': ['PEP-04-01', 'PEP-06'],
        'required': True,
        'description': '기능 요구사항 대응 개발 계획',
    },
    {
        'rfp_code': 'RFP-04-04-06',
        'rfp_subtitle': '테스트 요구사항',
        'pep_codes': ['PEP-07', 'PEP-06'],
        'required': False,
        'description': '테스트 계획 및 산출물',
    },
    {
        'rfp_code': 'RFP-04-04-07',
        'rfp_subtitle': '보안 요구사항',
        'pep_codes': ['PEP-14'],
        'required': True,
        'description': '보안 대책',
    },
    {
        'rfp_code': 'RFP-04-04-08',
        'rfp_subtitle': '품질 요구사항',
        'pep_codes': ['PEP-12'],
        'required': False,
        'description': '품질 관리 계획',
    },
    {
        'rfp_code': 'RFP-04-04-09',
        'rfp_subtitle': '제약사항',
        'pep_codes': ['PEP-11-01', 'PEP-11-02', 'PEP-11-03', 'PEP-11-04'],
        'required': False,
        'description': '표준화 및 제약사항 준수 계획',
    },
    {
        'rfp_code': 'RFP-04-04-10',
        'rfp_subtitle': '프로젝트 관리',
        'pep_codes': ['PEP-09', 'PEP-10'],
        'required': False,
        'description': '인력 및 보고 계획',
    },
    {
        'rfp_code': 'RFP-04-04-11',
        'rfp_subtitle': '프로젝트 지원',
        'pep_codes': ['PEP-15', 'PEP-13'],
        'required': False,
        'description': '교육 및 위험 관리 계획',
    },
]

_MIN_CONTENT_LENGTH = 30  # 이 이상이어야 "내용 있음"으로 판단


def _has_content(pep_sections: dict, pep_code: str) -> bool:
    """PEP 섹션에 실질적인 내용이 있는지 확인한다."""
    section = pep_sections.get(pep_code, {})
    content = section.get('content', '')
    return section.get('found', False) and len(content.strip()) >= _MIN_CONTENT_LENGTH


def compare_rfp_and_pep(rfp_json: dict, pep_json: dict) -> dict:
    """
    파싱된 RFP JSON과 PEP JSON을 구조적으로 비교한다.

    Returns:
        {
            "overall_score": 85,
            "total_items": 11,
            "satisfied_count": 8,
            "partial_count": 2,
            "unsatisfied_count": 1,
            "satisfied": [...],
            "partial": [...],
            "unsatisfied": [...],
        }
    """
    rfp_sections = rfp_json.get('sections', {})
    pep_sections = pep_json  # PEP는 최상위가 바로 섹션 딕셔너리

    satisfied = []
    partial = []
    unsatisfied = []

    for mapping in _RFP_TO_PEP_MAPPING:
        rfp_code = mapping['rfp_code']
        rfp_section = rfp_sections.get(rfp_code, {})

        # RFP 섹션 자체가 없으면 건너뜀
        if not rfp_section.get('found', False):
            continue

        pep_codes = mapping['pep_codes']
        found_pep = [c for c in pep_codes if _has_content(pep_sections, c)]
        missing_pep = [c for c in pep_codes if not _has_content(pep_sections, c)]

        coverage = len(found_pep) / len(pep_codes) if pep_codes else 0

        entry: dict[str, Any] = {
            'rfp_code': rfp_code,
            'rfp_subtitle': mapping['rfp_subtitle'],
            'description': mapping['description'],
            'required': mapping['required'],
            'pep_codes_checked': pep_codes,
            'pep_codes_found': found_pep,
            'pep_codes_missing': missing_pep,
            'coverage': round(coverage * 100),
        }

        if coverage == 1.0:
            # 모든 대응 PEP 섹션에 내용 있음
            satisfied.append(entry)
        elif coverage > 0:
            # 일부만 있음
            entry['gap'] = f"다음 항목이 미흡합니다: {', '.join(missing_pep)}"
            partial.append(entry)
        else:
            # 대응 PEP 섹션이 하나도 없음
            entry['issue'] = f"대응 PEP 항목({', '.join(pep_codes)})에 내용이 없습니다."
            unsatisfied.append(entry)

    # 점수 계산
    # required 항목 미충족 시 감점 가중치 2, 선택 항목 미충족 시 감점 가중치 1
    total_weight = 0
    earned_weight = 0

    for mapping in _RFP_TO_PEP_MAPPING:
        rfp_code = mapping['rfp_code']
        rfp_section = rfp_sections.get(rfp_code, {})
        if not rfp_section.get('found', False):
            continue

        w = 2 if mapping['required'] else 1
        total_weight += w

        pep_codes = mapping['pep_codes']
        found_pep = [c for c in pep_codes if _has_content(pep_sections, c)]
        coverage = len(found_pep) / len(pep_codes) if pep_codes else 0
        earned_weight += w * coverage

    overall_score = round((earned_weight / total_weight * 100) if total_weight else 0)

    return {
        'overall_score': overall_score,
        'total_items': len(satisfied) + len(partial) + len(unsatisfied),
        'satisfied_count': len(satisfied),
        'partial_count': len(partial),
        'unsatisfied_count': len(unsatisfied),
        'satisfied': satisfied,
        'partial': partial,
        'unsatisfied': unsatisfied,
    }