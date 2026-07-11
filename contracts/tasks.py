import os
import re
import sys
from celery import shared_task

# RunPod 등 원격 GPU 추론 서버 URL (SSH 로컬 포트포워딩 경유, 예: http://localhost:18000).
# 임베딩/리랭커(BGE-M3 계열)와 LLM(kanana)은 서로 다른 transformers 버전을 요구해
# RunPod에서 별도 venv·별도 포트의 서버 두 개로 띄운다. 둘 다 비어있으면 기존처럼
# 로컬에서 직접 모델을 로드해 추론한다.
EMBED_SERVER_URL = os.environ.get('EMBED_SERVER_URL', '').strip() or None
LLM_SERVER_URL = os.environ.get('LLM_SERVER_URL', '').strip() or None
USE_REMOTE_INFERENCE = bool(EMBED_SERVER_URL and LLM_SERVER_URL)

_CLAUSE_HEADER_RE = re.compile(r"제(\d+)조(?:의(\d+))?(?:\s*\([^)]*\))?")


def _chunk_contract(text: str) -> list[dict]:
    """
    계약서 텍스트를 '제N조(...)' 단위 조항으로 분할한다.
    새 law_rag_pipeline.search_jo()는 조항 하나(query_text)씩 검색하는
    순수 검색 함수라, 계약서 전체를 조항 단위로 쪼개는 건 호출하는
    쪽(여기)의 책임이다. (rag/pdfver_yoonha_contract_rag.py의 예전
    chunk_contract()와 동일 로직 — 그 모듈은 없는 패키지에 의존해서 못 씀)

    "제N조" 패턴은 실제 조항 헤더 말고도 본문 중간의 법령 인용(예: "소프트웨어진흥법
    제38조에 근거") 이나 요약표의 오기재("지식재산권 제23조(지식재산권)에서 정하는
    바에...")에도 나타난다. 이런 인용은 지금까지 확정된 조항 번호와 이어지지 않으므로,
    직전에 확정한 조항 번호와 같거나(같은 조의 하위 항) 정확히 다음 번호일 때만
    진짜 헤더로 인정한다 — 실제 계약서 조항은 항상 1부터 순차적으로 증가한다.
    """
    text = text.strip()

    matches = []
    last_num = 0
    for m in _CLAUSE_HEADER_RE.finditer(text):
        num = int(m.group(1))
        if num == last_num or num == last_num + 1:
            matches.append(m)
            last_num = num

    clauses = []
    for idx, m in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        raw_header = m.group(0).strip()
        body = text[m.end():end].strip()
        clause_number = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
        clause_text = f"{raw_header} {body}".strip()

        if body:
            clauses.append({'clause_number': clause_number, 'clause_text': clause_text})

    if not clauses:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        clauses = [
            {'clause_number': f'단락{idx + 1}', 'clause_text': paragraph}
            for idx, paragraph in enumerate(paragraphs)
        ]

    return clauses


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

    # 1. 계약서 1~2페이지 요약 표(계약 조건 + 계약당사자) 빈칸 검사
    # 규칙 기반이라 LLM과 완전히 무관하게 항상 먼저 실행한다. 
    # 아래 LLM 파이프라인이 (모델 미연결 등으로) 실패하더라도 이 결과는 그대로 저장·반환되어야 한다.
    blanks = []
    try:
        from contracts.contract_field_checker import check_contract_fields
        blanks = check_contract_fields(doc.file.path)
    except Exception:
        import traceback
        print(f'[analyze_document_task] 계약서 빈칸 검사 실패 (doc_id={doc_id}):\n{traceback.format_exc()}')

    typos: list = []
    legal_issues: list = []

    # 2. RAG + sLLM 법률 조항 검토 (실패해도 위 빈칸 결과는 유지한 채 계속 진행)
    try:
        file_text = extract_text(doc.file.path)
        if not file_text.strip():
            raise ValueError('텍스트 추출 실패')

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

        from qdrant_client import QdrantClient
        from law_rag_pipeline import search_jo, DEFAULT_MIN_SCORE

        if USE_REMOTE_INFERENCE:
            from remote_inference_client import RemoteEmbedModel, RemoteReranker
            embed_model = RemoteEmbedModel(EMBED_SERVER_URL)
            reranker = RemoteReranker(EMBED_SERVER_URL)
        else:
            from law_rag_pipeline import load_embed_model, load_reranker
            embed_model = load_embed_model()
            reranker = load_reranker()

        qdrant_client = QdrantClient(
            host=os.environ.get('QDRANT_HOST', 'localhost'),
            port=int(os.environ.get('QDRANT_PORT', '6333')),
        )

        clauses = _chunk_contract(file_text)

        rag_results = []
        for clause in clauses:
            law_refs = search_jo(
                clause['clause_text'],
                qdrant_client,
                embed_model,
                reranker=reranker,
            )
            rag_results.append({
                'clause_number': clause['clause_number'],
                'clause_text': clause['clause_text'],
                'law_refs': [
                    {
                        'law_name': ref.law_name,
                        'article_number': ref.article_id,
                        'chunk_text': ref.text,
                        'source_full': f'{ref.law_name} {ref.article_id}'.strip(),
                        'score': ref.score,
                    }
                    for ref in law_refs
                ],
                # 새 search_jo()는 리스크 카테고리 태깅을 제공하지 않는다 —
                # jihye_inference.build_user_content()는 비어있으면 "기타"로 처리한다.
                'risk_names': [],
            })

        del embed_model
        del reranker
        del qdrant_client
        import gc
        gc.collect()

        def _fallback_keys(num: str) -> list[str]:
            keys = [num]
            m_hang = re.match(r"(제\d+조(?:의\d+)?제\d+항)", num or "")
            if m_hang and m_hang.group(1) != num:
                keys.append(m_hang.group(1))
            m_jo = re.match(r"(제\d+조(?:의\d+)?)", num or "")
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

        if USE_REMOTE_INFERENCE:
            from remote_inference_client import remote_predict

            def predict(item, *_):
                return remote_predict(item, LLM_SERVER_URL)

            llm_model, tokenizer = None, None
        else:
            from jihye_inference import load_model as load_llm_model, predict

            llm_model, tokenizer = load_llm_model()

        # law_refs가 있어도 search_jo()의 fallback(관련 조문이 하나도 threshold를
        # 못 넘기면 그냥 top_k를 반환)때문에 사실상 모든 조항이 law_refs를 갖게 된다.
        # 개수로 자르는 대신, 실제로 min_score를 넘긴 "진짜 관련 법령"이 있는
        # 조항만 LLM 판정으로 넘긴다 — fallback으로 채워진 저품질 후보는 제외.
        def _has_relevant_law_ref(item):
            return any((ref.get('score') or 0) >= DEFAULT_MIN_SCORE for ref in item.get('law_refs', []))

        filtered = [r for r in rag_results if _has_relevant_law_ref(r)]
        total = len(filtered)
        done = 0

        inference_results = []
        for item in filtered:
            prediction = predict(item, llm_model, tokenizer)
            inference_results.append({
                'clause_number': item['clause_number'],
                'clause_text': item['clause_text'],
                'risk_names': item.get('risk_names', []),
                'page': item.get('page'),
                'bbox': item.get('bbox'),
                'fragments': item.get('fragments'),
                # parse_to_workit()은 prediction을 문자열(sLLM 판정 원문)로 기대한다.
                'prediction': prediction.get('raw', ''),
            })
            done += 1
            self.update_state(
                state='PROGRESS',
                meta={'current': done, 'total': total}
            )

        parsed = parse_to_workit(inference_results)
        typos = parsed['typos']
        legal_issues = parsed['legal_issues']

    except Exception:
        import traceback
        print(
            f'[analyze_document_task] LLM 법률 검토 실패 (doc_id={doc_id}) - '
            f'규칙 기반 빈칸 검사 결과는 그대로 저장/반환합니다:\n{traceback.format_exc()}'
        )

    # LLM 단계가 실패해도 규칙 기반 빈칸 검사 결과(blanks)는 항상 저장·반환한다
    # status='done'으로 바꿔야 분석화면이 "분석중" 대신 결과를 보여준다.
    AIReviewResult.objects.update_or_create(
        document=doc,
        defaults={
            'blanks': blanks,
            'typos': typos,
            'legal_issues': legal_issues,
            'status': 'done',
        }
    )

    # 화면을 나가 있어도 분석은 celery worker에서 계속 진행되므로, 완료 시점에
    # 알림을 띄워 사용자가 다시 들어와서 결과를 확인할 수 있게 한다.
    try:
        from performance.models import Notification

        owner = doc.contract.created_by
        if owner.notification_enabled:
            Notification.objects.create(
                user=owner,
                message=f'AI 분석이 완료되었습니다: {doc}',
                url=f'/contracts/document/{doc_id}/analyze/',
            )
    except Exception:
        import traceback
        print(f'[analyze_document_task] 완료 알림 생성 실패 (doc_id={doc_id}):\n{traceback.format_exc()}')

    return {
        'status': 'ok',
        'total': len(legal_issues),
        'blanks': blanks,
        'typos': typos,
        'legal_issues': legal_issues,
    }


# RFP 파싱 태스크 (규칙 기반, LLM 없음)
# 실행 시점 : document_complete_review (이행관리 이관) 직후 비동기
# 파서 : contracts.parsers.parse_rfp (키워드·정규식 기반)
# 결과 : contracts.models.RFPParsedData.parsed_json (RDS)

@shared_task(bind=True, max_retries=2, default_retry_delay=10)
def parse_rfp_task(self, rfp_doc_id: int):
    """
    RFP 문서를 규칙 기반으로 파싱해 RFPParsedData에 저장한다.

    contracts.parsers.parse_rfp() 를 사용하므로 LLM 호출이 없고,
    텍스트 추출 후 즉시 완료된다(보통 1~2초).
    """
    from django.utils import timezone
    from contracts.models import ContractDocument, RFPParsedData
    from contracts.utils import extract_text
    from contracts.parsers import parse_rfp, to_qa_agent_records  # ← 규칙 기반 파서

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    llm_dir = os.path.join(BASE_DIR, 'LLM')
    if llm_dir not in sys.path:
        sys.path.insert(0, llm_dir)

    rfp_doc = ContractDocument.objects.select_related('contract').get(pk=rfp_doc_id)

    parsed, _ = RFPParsedData.objects.get_or_create(document=rfp_doc)
    parsed.parse_status = 'processing'
    parsed.error_message = ''
    parsed.save(update_fields=['parse_status', 'error_message'])

    try:
        text = extract_text(rfp_doc.file.path)
        if not text.strip():
            raise ValueError('RFP 텍스트 추출 실패 — 파일을 확인하세요.')

        result_json = parse_rfp(text)

        found_count = sum(1 for s in result_json.get('sections', {}).values() if s.get('found'))
        total_count = len(result_json.get('sections', {}))

        # 소제목 매핑 QA 검수 (LLM/qa_agent). 검수 자체가 실패해도 파싱 성공은
        # 그대로 살려야 하므로 별도 try/except로 감싸고, 실패 시 리포트만 비워둔다.
        # (사업수행계획서 AI 분석 화면의 "2단계 · RFP 매핑" 탭에서 사용)
        qa_report = {}
        try:
            from qa_agent.engine import review_section_mapping

            qa_report = review_section_mapping(
                original_text=text,
                parsed_sections=to_qa_agent_records(result_json),
                document_type='rfp',
            )
        except Exception:
            import traceback
            print(f'[parse_rfp_task] QA 검수 실패 — doc_id={rfp_doc_id}\n{traceback.format_exc()}')

        qa_issues = qa_report.get('issues', [])
        if qa_issues:
            print(
                f'[QA] RFP(doc_id={rfp_doc_id}) 1단계 QA 이슈 {len(qa_issues)}건 발견 '
                f'(review_status={qa_report.get("review_status")}): '
                f'{[issue.get("issue_type") for issue in qa_issues]}'
            )
        else:
            print(f'[QA] RFP(doc_id={rfp_doc_id}) 1단계 QA 이슈 없음 (review_status={qa_report.get("review_status")})')

        parsed.parsed_json = result_json
        parsed.qa_report = qa_report
        parsed.parse_status = 'done'
        parsed.parsed_at = timezone.now()
        parsed.save(update_fields=['parsed_json', 'qa_report', 'parse_status', 'parsed_at'])

        print(f'[parse_rfp_task] 완료 — doc_id={rfp_doc_id}, 섹션 {found_count}/{total_count} 발견')
        return {'status': 'ok', 'doc_id': rfp_doc_id, 'found': found_count, 'total': total_count,
                'qa_status': qa_report.get('review_status')}

    except Exception as exc:
        import traceback
        err = traceback.format_exc()
        parsed.parse_status = 'failed'
        parsed.error_message = err[:2000]
        parsed.save(update_fields=['parse_status', 'error_message'])
        print(f'[parse_rfp_task] 실패 — doc_id={rfp_doc_id}\n{err}')
        raise self.retry(exc=exc)