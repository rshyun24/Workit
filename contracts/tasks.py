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

        # Step 2. RAG
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        from yoonha_contract_rag import review_contract, results_to_json

        QDRANT_PATH = os.path.join(BASE_DIR, 'vectorstore', 'qdrant_storage')
        embed_model = SentenceTransformer('BAAI/bge-m3')
        qdrant_client = QdrantClient(url="http://localhost:6333")

        clause_results = review_contract(
            contract_text=file_text,
            client=qdrant_client,
            model=embed_model,
            risk_only=True,
        )
        rag_results = results_to_json(clause_results)

        # Step 3. sLLM 추론 (CPU 환경 대비 상위 MAX_CLAUSES개만)
        from jihye_inference import load_model, predict

        llm_model, tokenizer = load_model()

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
                'risk_names':    item.get('risk_names', []),
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