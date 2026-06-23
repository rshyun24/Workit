import os
import sys
from celery import shared_task

MAX_CLAUSES = 3  # 시연용 조항 수 제한 (CPU 환경)

@shared_task(bind=True)
def analyze_document_task(self, doc_id):
    """AI 분석 비동기 태스크"""
    from contracts.models import ContractDocument, AIReviewResult
    from contracts.utils import extract_text, parse_to_workit

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in [os.path.join(BASE_DIR, 'rag'), os.path.join(BASE_DIR, 'data')]:
        if p not in sys.path:
            sys.path.insert(0, p)

    doc = ContractDocument.objects.get(pk=doc_id)

    try:
        # Step 1. 텍스트 추출
        file_text = extract_text(doc.file.path)
        if not file_text.strip():
            return {'status': 'error', 'message': '텍스트 추출 실패'}

        # Step 1.5. 조항/항별 좌표(fragments) 추출 (PDF/HWP 모두 지원, 실패 시 빈 매핑으로 fallback)
        # clause_locator.extract_clause_positions() 반환 형식:
        #   { "제2조": {"fragments": [{page,x,y,width,height}, ...]},
        #     "제2조제1항": {"fragments": [...]}, ... }
        # 항이 있는 조항은 "제N조제M항" 키로, 항이 없으면 "제N조" 키로 직접 매칭한다.
        # (페이지 경계를 넘는 영역은 fragments가 여러 개로 분할되어 있다 — 프론트에서 각각 그린다)
        clause_positions = {}
        file_path_lower = doc.file.path.lower()

        try:
            from clause_locator import extract_clause_positions

            if file_path_lower.endswith('.pdf'):
                clause_positions = extract_clause_positions(doc.file.path)

            elif file_path_lower.endswith('.hwp'):
                from hwp_converter import convert_hwp_to_pdf
                import tempfile
                with tempfile.TemporaryDirectory() as tmp_dir:
                    converted_pdf = convert_hwp_to_pdf(doc.file.path, tmp_dir)
                    clause_positions = extract_clause_positions(converted_pdf)
        except Exception:
            clause_positions = {}

        # Step 2. RAG (BGE-M3 Dense+Sparse 하이브리드 검색)
        from qdrant_client import QdrantClient
        from yoonha_contract_rag import (
            load_model,
            load_laws_ref,
            review_contract,
            results_to_json,
        )

        embed_model = load_model()  # BAAI/bge-m3 (BGEM3FlagModel)
        qdrant_client = QdrantClient(url="http://localhost:6333")
        laws_ref = load_laws_ref()

        clause_results = review_contract(
            contract_text=file_text,
            client=qdrant_client,
            model=embed_model,
            laws_ref=laws_ref,
        )
        rag_results = results_to_json(clause_results)

        # 메모리 해제 (BGE-M3 다 썼으니 EXAONE 로딩 전에 비움 — Segfault 방지)
        del embed_model
        del qdrant_client
        import gc
        gc.collect()

        # 조항(항/호)번호 기준으로 좌표 정보(fragments) 병합.
        # "제5조제1항제1호"가 clause_positions에 직접 있으면 그걸 쓰고,
        # 없으면 "제5조제1항"(항 단위)로 fallback, 그것도 없으면 "제5조"(조 단위)로 fallback.
        import re as _re

        def _fallback_keys(num: str) -> list[str]:
            """clause_number로부터 [원본, 항단위, 조단위] 순으로 fallback 키 목록 생성."""
            keys = [num]
            # 호 제거: "제5조제1항제1호" → "제5조제1항"
            m_hang = _re.match(r"(제\d+조(?:의\d+)?제\d+항)", num or "")
            if m_hang and m_hang.group(1) != num:
                keys.append(m_hang.group(1))
            # 조까지만: "제5조"
            m_jo = _re.match(r"(제\d+조(?:의\d+)?)", num or "")
            if m_jo and m_jo.group(1) not in keys:
                keys.append(m_jo.group(1))
            return keys

        for item in rag_results:
            clause_number = item.get('clause_number')
            pos = None
            for key in _fallback_keys(clause_number):
                pos = clause_positions.get(key)
                if pos:
                    break

            if pos and pos.get('fragments'):
                item['fragments'] = pos['fragments']
                # 하위 호환: 기존 프론트/리포트 코드가 단일 page/bbox를 참조하는 곳이
                # 있을 수 있으므로 첫 fragment를 대표값으로 같이 내려준다.
                first = pos['fragments'][0]
                item['page'] = first['page']
                item['bbox'] = {
                    'x': first['x'], 'y': first['y'],
                    'width': first['width'], 'height': first['height'],
                }
            else:
                item['fragments'] = None
                item['page'] = None
                item['bbox'] = None

        # Step 3. sLLM 추론 (CPU 환경 대비 상위 MAX_CLAUSES개만)
        from jihye_inference import load_model as load_llm_model, predict

        llm_model, tokenizer = load_llm_model()

        # law_refs 있는 항목만 필터링 후 MAX_CLAUSES개 제한
        filtered = [r for r in rag_results if r.get('law_refs')][:MAX_CLAUSES]
        total = len(filtered)
        done = 0

        inference_results = []
        for item in filtered:
            prediction = predict(
                clause_text=item['clause_text'],
                law_refs=item['law_refs'],
                model=llm_model,
                tokenizer=tokenizer,
            )
            inference_results.append({
                'clause_number': item['clause_number'],
                'clause_text':   item['clause_text'],
                'risk_names':    item.get('categories', []),
                'page':          item.get('page'),
                'bbox':          item.get('bbox'),
                'fragments':     item.get('fragments'),
                'prediction':    prediction,
            })
            done += 1
            self.update_state(
                state='PROGRESS',
                meta={'current': done, 'total': total}
            )

        # Step 4. 결과 저장
        parsed = parse_to_workit(inference_results)
        AIReviewResult.objects.update_or_create(
            document=doc,
            defaults={
                'blanks':       parsed['blanks'],
                'typos':        parsed['typos'],
                'legal_issues': parsed['legal_issues'],
            }
        )

        return {
            'status': 'ok',
            'total': len(parsed['legal_issues']),
            'blanks': parsed['blanks'],
            'typos': parsed['typos'],
            'legal_issues': parsed['legal_issues'],
        }

    except Exception as e:
        import traceback
        return {'status': 'error', 'message': traceback.format_exc()}