"""
contracts/utils.py

- extract_text(): 업로드 파일에서 텍스트 추출 (PDF / DOCX / TXT)
- parse_to_workit(): RAG+sLLM 결과 → Workit AIReviewResult 형식 변환
"""


def extract_text(file_path: str) -> str:
    """업로드된 파일 경로를 받아 텍스트 문자열 반환."""
    ext = file_path.lower().rsplit('.', 1)[-1]

    if ext == 'hwp':
        # 구형 HWP는 pdfplumber/docx로 직접 못 읽으므로
        # LibreOffice(H2Orestart 확장)로 PDF 변환 후 같은 함수에 재귀 호출해 처리한다.
        import sys
        import os
        import tempfile

        rag_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rag')
        if rag_dir not in sys.path:
            sys.path.insert(0, rag_dir)

        from hwp_converter import convert_hwp_to_pdf

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = convert_hwp_to_pdf(file_path, tmp_dir)
            return extract_text(pdf_path)  # 변환된 PDF를 아래 'pdf' 분기로 재처리

    if ext == 'pdf':
        import pdfplumber
        texts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
        return '\n'.join(texts)

    elif ext == 'docx':
        from docx import Document
        doc = Document(file_path)
        return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())

    elif ext in ('txt', 'md', 'hwpx'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    else:
        # 지원하지 않는 형식 — 빈 문자열 반환, 호출부에서 처리
        return ''


def parse_to_workit(inference_results: list) -> dict:
    """
    jihye_inference.run_inference() 반환값 →  Workit AIReviewResult 형식으로 변환.

    inference_results 각 항목 구조:
    {
        "clause_number": "제1조",
        "clause_text":   "...",
        "risk_names":    ["손해배상 범위 일방적 제한", ...],
        "prediction":    "위험 조항입니다. 근거: ..."   ← sLLM 출력 텍스트
    }

    반환 형식 (AIReviewResult 모델에 저장):
    {
        "blanks":       [],   # RAG/sLLM 미지원 → 빈 리스트
        "typos":        [],   # RAG/sLLM 미지원 → 빈 리스트
        "legal_issues": [
            {
                "location":      "제1조",
                "original_text": "...",
                "issue":         "손해배상 범위 일방적 제한, ...",
                "legal_ref":     "... (sLLM 판정 전문)"
            },
            ...
        ]
    }

    필터링 기준:
    - "판정: 정상" 항목 제외
    - 동일 조항 번호 중복 제거 (첫 번째 위반/누락 항목만 사용)
    """
    legal_issues = []
    seen_locations = set()

    for item in inference_results:
        prediction = (item.get('prediction') or '').strip()
        risk_names = item.get('risk_names') or []
        location = item.get('clause_number', '')

        # sLLM 판정 결과 없는 항목 제외
        if not prediction:
            continue

        # 정상 판정 제외 (계약서='일치', 산출물='충족')
        if ('판정: 일치' in prediction) or ('판정: 충족' in prediction):
            continue

        # 동일 조항 번호 중복 제거
        if location in seen_locations:
            continue
        seen_locations.add(location)

        legal_issues.append({
            'location':      location,
            'original_text': item.get('clause_text', '')[:300],
            'issue':         ', '.join(risk_names) if risk_names else '위험 조항 감지',
            'legal_ref':     prediction,
            'page':          item.get('page'),
            'bbox':          item.get('bbox'),
            'fragments':     item.get('fragments'),
        })

    return {
        'blanks':       [],
        'typos':        [],
        'legal_issues': legal_issues,
    }

import re
from datetime import date

def extract_contract_period(text: str):
    """
    ex.) "2026년 6월 1일부터 2026년 7월 31일까지" 패턴 추출
    """
    pattern = r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일부터\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일까지"
    match = re.search(pattern, text)
    if match:
        y1, m1, d1, y2, m2, d2 = map(int, match.groups())
        return date(y1, m1, d1), date(y2, m2, d2)
    return None, None