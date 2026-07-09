import json
import logging
import os
from django.core.exceptions import ValidationError
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST
from contracts.models import Contract
from .models import Performance, Deliverable
from .models import Deliverable, Notification, Performance
from django.utils import timezone
from accounts.audit import log_audit
from accounts.models import AuditLog
from accounts.file_validators import validate_uploaded_file
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_COLORS = [
    '#4F63D2', '#00B894', '#6C5CE7', '#E67E22',
    '#E74C3C', '#1ABC9C', '#2980B9', '#8E44AD',
]

# 산출물 유형별 평가 기준 문서 매핑
DELIVERABLE_CRITERIA = {
    'kickoff': '사업수행계획서 품질평가 기준 (16개 목차 항목 × 9개 품질특성)',
    'tech_apply': '기술적용결과표 체크박스 정합성 검증 (적용/부분적용/미적용/해당없음 + 사유)',
    'test_plan': None,
    'test_result': None,
    'final': '사업추진결과보고서 소제목 매핑 QA 검수 · RFP 대응 비교',
}


@login_required
def performance_list(request):
    contracts = Contract.objects.filter(
        created_by=request.user,
        status__in=['in_progress', 'completed']
    ).order_by('-created_at')

    # Ensure Performance objects exist for all in-progress contracts
    for c in contracts:
        Performance.objects.get_or_create(contract=c)

    performances = Performance.objects.filter(
        contract__created_by=request.user,
        contract__status__in=['in_progress', 'completed']
    ).order_by('-created_at').select_related('contract')

    TARGET_DELIVERABLE_TYPES = ['tech_apply', 'final']

    calendar_events = []

    for perf in performances:
        color    = PROJECT_COLORS[perf.contract_id % len(PROJECT_COLORS)]
        contract = perf.contract
        is_done  = contract.status == 'completed'

        # 산출물(기술적용결과표·결과보고서) 제출 일자만 달력에 표시한다
        # (계약 기간 줄은 더 이상 달력에 표시하지 않음 — 이행 현황 카드에 텍스트로 노출)
        target_deliverables = perf.deliverables.filter(
            deliverable_type__in=TARGET_DELIVERABLE_TYPES,
        ).order_by('due_date')

        for d in target_deliverables:
            if not d.due_date:
                continue  # due_date 없으면 달력 표시 안 함

            calendar_events.append({
                'id': d.id,
                'performance_id': perf.id,
                'project_name': contract.project_name,
                'deliverable_type': d.get_deliverable_type_display(),
                'start_date': d.due_date.strftime('%Y-%m-%d'),
                'end_date': d.due_date.strftime('%Y-%m-%d'),
                'due_date': d.due_date.strftime('%Y-%m-%d'),
                'submitted_date': d.submitted_date.strftime('%Y-%m-%d') if d.submitted_date else None,
                'status': d.status,
                'color': '#999999' if (is_done or d.status == 'submitted') else color,
                'is_completed': is_done or d.status == 'submitted',
            })

    return render(request, 'performance/performance_list.html', {
        'performances': performances,
        'calendar_events_json': json.dumps(calendar_events, ensure_ascii=False),
        'project_colors': {
            perf.id: PROJECT_COLORS[perf.contract_id % len(PROJECT_COLORS)]
            for perf in performances
        },
})


@login_required
def performance_detail_api(request, pk):
    perf = get_object_or_404(Performance, pk=pk, contract__created_by=request.user)
    contract = perf.contract

    # Build deliverable list with all 4 types
    TYPE_ORDER = ['kickoff', 'tech_apply', 'final']
    TYPE_LABELS = {
        'kickoff': '사업수행계획서',
        'tech_apply': '기술적용결과표',
        'final': '사업추진결과보고서',
    }

    existing = {d.deliverable_type: d for d in perf.deliverables.all()}
    deliverables_data = []
    for t in TYPE_ORDER:
        d = existing.get(t)
        deliverables_data.append({
            'id': d.id if d else None,
            'type': t,
            'type_display': TYPE_LABELS[t],
            'filename': d.filename() if d else '',
            'due_date': d.due_date.strftime('%Y-%m-%d') if d and d.due_date else '',
            'submitted_date': d.submitted_date.strftime('%Y-%m-%d') if d and d.submitted_date else '',
            'status': d.status if d else 'pending',
            'has_file': bool(d and d.file),
            # 분석 지원 여부 (평가기준서가 준비된 유형만 True)
            'analyzable': DELIVERABLE_CRITERIA.get(t) is not None,
        })

    # Contract docs - 계약서·RFP·요구사항정의서 3종 (뷰어 전용)
    DOC_TYPE_ORDER = ['contract', 'rfp', 'requirements']
    DOC_TYPE_LABELS = {
        'contract': '계약서',
        'rfp': 'RFP (제안요청서)',
        'requirements': '요구사항정의서',
    }
    existing_docs = {doc.doc_type: doc for doc in contract.documents.all()}
    contract_docs = []
    for dtype in DOC_TYPE_ORDER:
        doc = existing_docs.get(dtype)
        contract_docs.append({
            'id': doc.id if doc else None,
            'doc_type': dtype,
            'doc_type_display': DOC_TYPE_LABELS[dtype],
            'filename': doc.filename() if doc else '',
            'review_status': doc.review_status if doc else 'pending',
            'has_file': bool(doc and doc.file),
        })

    progress = sum(1 for d in deliverables_data if d['status'] == 'submitted')

    return JsonResponse({
        'id': perf.id,
        'contract_id': contract.id,
        'project_name': contract.project_name,
        'company_name': contract.company_name,
        'issuing_org': contract.issuing_org,
        'budget': contract.budget,
        'contact_person': contract.contact_person,
        'status': contract.status,
        'status_display': contract.get_status_display(),
        'created_at': contract.created_at.strftime('%Y-%m-%d'),
        'deliverables': deliverables_data,
        'contract_docs': contract_docs,
        'progress': progress,
        'total': len(TYPE_ORDER),
    })


def _reset_kickoff_analysis(d, perf):
    """
    사업수행계획서(kickoff)의 1단계 QA 검수·2단계 비교 결과를 모두 폐기해
    분석 화면이 "분석 시작" 초기 상태부터 다시 노출되게 한다.
    새 파일 업로드 시(deliverable_upload)와 반려(deliverable_reject_qa) 시
    둘 다 "지금 붙어있는 분석 결과는 더 이상 유효하지 않다"는 같은 상황이라
    같은 로직을 공유한다. RFP 자체의 QA(2단계)는 이 산출물이 아니라 계약
    문서 소속이라 건드리지 않는다.
    """
    from .models import ExecutionPlanParsedData, RFPComparisonResult, PEPFinalComparisonResult
    ExecutionPlanParsedData.objects.filter(deliverable=d).update(
        parsed_json={}, qa_report={}, parse_status='pending',
        parsed_at=None, error_message='',
    )
    RFPComparisonResult.objects.filter(performance=perf).delete()
    # 사업추진결과보고서(final)의 2단계 비교(PEP ↔ RPT)도 이 사업수행계획서를
    # 근거로 만든 것이라 함께 폐기한다.
    PEPFinalComparisonResult.objects.filter(performance=perf).delete()


def _reset_final_analysis(d, perf):
    """사업추진결과보고서(final)판 _reset_kickoff_analysis. 같은 이유로 같은 방식."""
    from .models import FinalReportParsedData, PEPFinalComparisonResult
    FinalReportParsedData.objects.filter(deliverable=d).update(
        parsed_json={}, qa_report={}, parse_status='pending',
        parsed_at=None, error_message='',
    )
    PEPFinalComparisonResult.objects.filter(performance=perf).delete()


@login_required
@require_POST
def deliverable_upload(request, perf_id):
    perf = get_object_or_404(Performance, pk=perf_id, contract__created_by=request.user)
    d_type = request.POST.get('deliverable_type')
    f = request.FILES.get('file')
    due_date = request.POST.get('due_date') or None
    submitted_date = request.POST.get('submitted_date') or None

    if not d_type:
        return JsonResponse({'status': 'error', 'message': '산출물 유형이 없습니다.'}, status=400)

    if f:
        try:
            validate_uploaded_file(f)
        except ValidationError as e:
            return JsonResponse({'status': 'error', 'message': str(e.message)}, status=400)

    defaults = {'status': 'submitted' if f else 'pending'}
    if f:
        defaults['file'] = f
        defaults['original_filename'] = f.name
    if due_date:
        defaults['due_date'] = due_date
    if submitted_date:
        defaults['submitted_date'] = submitted_date

    d, created = Deliverable.objects.update_or_create(
        performance=perf,
        deliverable_type=d_type,
        defaults=defaults,
    )
    if f:
        log_audit(request, AuditLog.ACTION_UPLOAD, 'deliverable', d.id, detail=f'{d_type} 업로드')

    # 사업수행계획서(kickoff) 파일이 새로 바뀌면, 예전 파일 기준으로 만들어진
    # QA 검수 결과·RFP 비교 결과는 더 이상 유효하지 않으므로 폐기한다.
    # (재분석 전까지는 deliverable_analyze 화면에서 "분석 시작" 단계부터 다시 노출됨)
    if d_type == 'kickoff' and f:
        _reset_kickoff_analysis(d, perf)

        # 그 안의 산출물계획 표를 파싱해 기술적용결과표/사업추진결과보고서
        # 마감일을 비동기로 자동 반영
        from .tasks import sync_deliverable_dates_from_kickoff_task
        sync_deliverable_dates_from_kickoff_task.delay(d.id)

    # 기술적용결과표 파일이 새로 바뀌면, 예전 파일 기준 검증 결과는 더 이상
    # 유효하지 않으므로 폐기한다 (재분석 전까지는 "분석 시작" 초기 화면부터 다시 노출됨)
    if d_type == 'tech_apply' and f:
        from .models import TechApplyCheckResult
        TechApplyCheckResult.objects.filter(deliverable=d).delete()

    # 사업추진결과보고서(final) 파일이 새로 바뀌면, kickoff와 동일한 이유로
    # 이전 QA 검수 결과·RFP 비교 결과를 폐기한다.
    if d_type == 'final' and f:
        from .models import FinalReportParsedData, PEPFinalComparisonResult
        FinalReportParsedData.objects.filter(deliverable=d).update(
            parsed_json={}, qa_report={}, parse_status='pending',
            parsed_at=None, error_message='',
        )
        PEPFinalComparisonResult.objects.filter(performance=perf).delete()

    return JsonResponse({'status': 'ok', 'filename': d.filename(), 'deliverable_id': d.id})


@login_required
@require_POST
def deliverable_update_due_date(request, perf_id):
    perf = get_object_or_404(Performance, pk=perf_id, contract__created_by=request.user)
    data = json.loads(request.body)
    d_type = data.get('deliverable_type')
    due_date = data.get('due_date')

    d, _ = Deliverable.objects.get_or_create(performance=perf, deliverable_type=d_type)
    if due_date:
        d.due_date = due_date
        d.save()
    return JsonResponse({'status': 'ok'})


# ──────────────────────────────────────────────────────────────
# 산출물 문서 뷰어 / AI 분석
# ──────────────────────────────────────────────────────────────

@login_required
def contract_doc_view(request, doc_id):
    """이행관리에서 계약 문서(계약서·RFP·요구사항정의서) 보기 — '목록'은 이행관리로.

    이미지는 contracts 앱의 기존 엔드포인트(/contracts/document/<id>/pages|page)를 재사용.
    """
    from contracts.models import ContractDocument
    doc = get_object_or_404(
        ContractDocument, pk=doc_id,
        contract__created_by=request.user,
    )
    log_audit(request, AuditLog.ACTION_VIEW, 'contract_document', doc.id)
    return render(request, 'performance/contract_doc_view.html', {
        'doc': doc,
        'contract': doc.contract,
    })


@login_required
def deliverable_view(request, del_id):
    """산출물 이미지 뷰어 (AI 패널 없음)"""
    d = get_object_or_404(
        Deliverable, pk=del_id,
        performance__contract__created_by=request.user,
    )
    log_audit(request, AuditLog.ACTION_VIEW, 'deliverable', d.id)
    return render(request, 'performance/deliverable_view.html', {
        'deliverable': d,
        'contract': d.performance.contract,
    })


@login_required
def deliverable_analyze(request, del_id):
    """산출물 AI 분석 화면 (뷰어 + 평가 패널)"""
    d = get_object_or_404(
        Deliverable, pk=del_id,
        performance__contract__created_by=request.user,
    )
    log_audit(request, AuditLog.ACTION_VIEW, 'deliverable', d.id)
    analyzable = DELIVERABLE_CRITERIA.get(d.deliverable_type) is not None
    is_kickoff = d.deliverable_type == 'kickoff'
    is_final = d.deliverable_type == 'final'

    # 이미 실행된 QA 검수 / 2단계 비교 결과가 있으면 새로고침해도 그대로 복원.
    # kickoff(사업수행계획서)는 RFP와, final(사업추진결과보고서)은 사업수행계획서(PEP)와
    # 비교한다는 점만 다르고 "파싱 → qa_agent 검수 → 2단계 비교" 흐름 자체는 동일하다.
    qa_data = None
    comparison_data = None
    if is_kickoff:
        parsed = getattr(d, 'parsed_data', None)
        comparison_qs = d.performance.rfp_comparisons
    elif is_final:
        parsed = getattr(d, 'final_parsed_data', None)
        comparison_qs = d.performance.pep_final_comparisons
    else:
        parsed = None
        comparison_qs = None

    if is_kickoff or is_final:
        if parsed and parsed.parse_status in ('done', 'failed'):
            qa_data = {
                'parse_status': parsed.parse_status,
                'qa_report': parsed.qa_report,
                'error_message': parsed.error_message,
            }
        comparison = comparison_qs.first() if comparison_qs is not None else None
        if comparison:
            comparison_data = comparison.comparison_json

    # 사업수행계획서(kickoff) 전용 "2단계 · RFP 매핑" — RFP는 계약관리→이행관리 이관
    # 시점에 이미 자동으로 파싱·QA가 끝나 있는 경우가 대부분이라, 그 결과를 그대로
    # 복원한다. 아직 안 끝났으면(또는 예전 계약이라 qa_report가 비어 있으면)
    # 프론트에서 "RFP AI 분석 시작" 버튼을 눌러 새로 실행할 수 있다.
    rfp_qa_data = None
    rfp_doc_id = None
    if is_kickoff:
        rfp_doc = d.performance.contract.documents.filter(doc_type='rfp').first()
        if rfp_doc:
            rfp_doc_id = rfp_doc.id
        rfp_parsed = getattr(rfp_doc, 'rfp_parsed', None) if rfp_doc else None
        if rfp_parsed and rfp_parsed.parse_status in ('done', 'failed') and rfp_parsed.qa_report:
            rfp_qa_data = {
                'parse_status': rfp_parsed.parse_status,
                'qa_report': rfp_parsed.qa_report,
                'error_message': rfp_parsed.error_message,
            }

    # 기술적용결과표: 이미 실행된 체크 검증 결과가 있으면 새로고침해도 그대로 복원
    # (파일이 재업로드되면 deliverable_upload에서 이 레코드를 지우므로 여기서 못 찾음)
    tech_apply_data = None
    if d.deliverable_type == 'tech_apply':
        tech_apply_result = getattr(d, 'tech_apply_result', None)
        if tech_apply_result:
            tech_apply_data = {
                'status': 'ok',
                'analysis_type': 'tech_apply_check',
                **tech_apply_result.result_json,
            }

    return render(request, 'performance/deliverable_analyze.html', {
        'deliverable': d,
        'contract': d.performance.contract,
        'analyzable': analyzable,
        'criteria_label': DELIVERABLE_CRITERIA.get(d.deliverable_type) or '',
        'is_kickoff': is_kickoff,
        'is_final': is_final,
        'has_qa_flow': is_kickoff or is_final,
        'has_rfp_qa_step': is_kickoff,
        'rfp_doc_id': rfp_doc_id,
        'qa_data': qa_data,
        'rfp_qa_data': rfp_qa_data,
        'comparison_data': comparison_data,
        'tech_apply_data': tech_apply_data,
    })


@login_required
@require_POST
def deliverable_parse_qa(request, del_id):
    """
    사업수행계획서(kickoff)/사업추진결과보고서(final) 파일을 규칙 기반으로
    파싱하고, LLM/qa_agent로 '원본 ↔ 파싱 결과'의 소제목 매핑을 검수한다.

    호출 시점: 이행관리에서 "분석 시작" 버튼 클릭.
    """
    d = get_object_or_404(
        Deliverable, pk=del_id,
        performance__contract__created_by=request.user,
    )
    if d.deliverable_type not in ('kickoff', 'final'):
        return JsonResponse({'status': 'error', 'message': '사업수행계획서·사업추진결과보고서만 지원합니다.'}, status=400)
    if not d.file:
        return JsonResponse({'status': 'error', 'message': '파일이 없습니다.'}, status=400)

    if d.deliverable_type == 'kickoff':
        from .tasks import parse_execution_plan_task
        parse_execution_plan_task.apply(args=(d.id,)).get()
        parsed = d.parsed_data
    else:
        from .tasks import parse_final_report_task
        parse_final_report_task.apply(args=(d.id,)).get()
        parsed = d.final_parsed_data

    parsed.refresh_from_db()
    if parsed.parse_status == 'failed':
        return JsonResponse({
            'status': 'error',
            'message': parsed.error_message or '파싱 중 오류가 발생했습니다.',
        }, status=400)

    return JsonResponse({
        'status': 'ok',
        'parse_status': parsed.parse_status,
        'qa_report': parsed.qa_report,
    })


@login_required
@require_POST
def deliverable_parse_rfp_qa(request, del_id):
    """
    사업수행계획서(kickoff) AI 분석 화면의 "2단계 · RFP 매핑" — 이 계약의 RFP
    문서를 다시 파싱하고 LLM/qa_agent로 '원본 ↔ 파싱 결과'의 소제목 매핑을
    검수한다. RFP는 보통 계약관리→이행관리 이관 시점에 이미 자동으로 한 번
    파싱되지만, 여기서 다시 실행해도 결과는 동일하다(멱등).

    호출 시점: 이 화면의 2단계 탭에서 "RFP AI 분석 시작" 버튼 클릭.
    """
    d = get_object_or_404(
        Deliverable, pk=del_id,
        performance__contract__created_by=request.user,
    )
    if d.deliverable_type != 'kickoff':
        return JsonResponse({'status': 'error', 'message': '사업수행계획서에서만 지원합니다.'}, status=400)

    rfp_doc = d.performance.contract.documents.filter(doc_type='rfp').first()
    if not rfp_doc:
        return JsonResponse({'status': 'error', 'message': 'RFP 문서가 없습니다.'}, status=400)
    if not rfp_doc.file:
        return JsonResponse({'status': 'error', 'message': 'RFP 파일이 없습니다.'}, status=400)

    from contracts.tasks import parse_rfp_task
    result = parse_rfp_task.apply(args=(rfp_doc.id,)).get()
    if result.get('status') != 'ok':
        return JsonResponse(result, status=400)

    rfp_parsed = rfp_doc.rfp_parsed
    if rfp_parsed.parse_status == 'failed':
        return JsonResponse({
            'status': 'error',
            'message': rfp_parsed.error_message or 'RFP 분석 중 오류가 발생했습니다.',
        }, status=400)

    return JsonResponse({
        'status': 'ok',
        'parse_status': rfp_parsed.parse_status,
        'qa_report': rfp_parsed.qa_report,
    })


@login_required
@require_POST
def deliverable_compare_rfp(request, del_id):
    """
    사업수행계획서(kickoff)는 RFP와, 사업추진결과보고서(final)는 사업수행계획서(PEP)와
    구조적으로 비교한다 — final은 "계획한 대로 실제로 이행됐는지"를 확인한다.

    호출 시점: QA 검수 결과 확인 후 "그대로 진행" 버튼 클릭 (반려 없이 비교로 넘어갈 때).
    """
    d = get_object_or_404(
        Deliverable, pk=del_id,
        performance__contract__created_by=request.user,
    )
    if d.deliverable_type not in ('kickoff', 'final'):
        return JsonResponse({'status': 'error', 'message': '사업수행계획서·사업추진결과보고서만 지원합니다.'}, status=400)

    # 반려하지 않고 그대로 2단계로 넘어온 경우, 1단계에 이슈가 있었는지 남겨둔다
    # (반려 여부는 deliverable_reject_qa에서 별도로 기록한다).
    from .models import AIAnalysisLog
    parsed = d.parsed_data if d.deliverable_type == 'kickoff' else d.final_parsed_data
    qa_issues = (parsed.qa_report or {}).get('issues', []) if parsed else []
    if qa_issues:
        logger.info(
            '[QA] %s(deliverable_id=%s) 1단계 이슈 %d건 있었지만 반려 없이 2단계 비교분석 진행함',
            d.get_deliverable_type_display(), d.id, len(qa_issues),
        )
        AIAnalysisLog.log(
            d, 'proceed_with_issue', issue_count=len(qa_issues), user=request.user,
        )
    else:
        logger.info(
            '[QA] %s(deliverable_id=%s) 1단계 이슈 없이 2단계 비교분석 진행',
            d.get_deliverable_type_display(), d.id,
        )
        AIAnalysisLog.log(d, 'proceed_no_issue', user=request.user)

    if d.deliverable_type == 'kickoff':
        from .tasks import compare_rfp_execution_plan_task
        result = compare_rfp_execution_plan_task.apply(args=(d.performance_id,)).get()
        comparison_qs = d.performance.rfp_comparisons
    else:
        from .tasks import compare_pep_final_report_task
        result = compare_pep_final_report_task.apply(args=(d.performance_id,)).get()
        comparison_qs = d.performance.pep_final_comparisons

    if result.get('status') != 'ok':
        return JsonResponse(result, status=400)

    comparison = comparison_qs.first()
    return JsonResponse({
        'status': 'ok',
        'comparison': comparison.comparison_json if comparison else {},
    })


@login_required
@require_POST
def deliverable_reject_qa(request, del_id):
    """
    1단계 QA 검수에서 "반려 (다시 업로드)"를 눌렀을 때 호출된다.
    실제 처리(파일 재업로드 유도)는 프론트에서 하고, 여기서는 반려 사실만 기록한다.
    """
    d = get_object_or_404(
        Deliverable, pk=del_id,
        performance__contract__created_by=request.user,
    )
    if d.deliverable_type not in ('kickoff', 'final'):
        return JsonResponse({'status': 'error', 'message': '사업수행계획서·사업추진결과보고서만 지원합니다.'}, status=400)

    from .models import AIAnalysisLog
    parsed = d.parsed_data if d.deliverable_type == 'kickoff' else d.final_parsed_data
    qa_issues = (parsed.qa_report or {}).get('issues', []) if parsed else []
    logger.info(
        '[QA] %s(deliverable_id=%s) 1단계 이슈 %d건 확인 후 반려(다시 업로드) 선택함',
        d.get_deliverable_type_display(), d.id, len(qa_issues),
    )
    AIAnalysisLog.log(d, 'reject', issue_count=len(qa_issues), user=request.user)
    return JsonResponse({'status': 'ok'})


@login_required
def deliverable_page_count(request, del_id):
    """산출물 PDF 총 페이지 수 반환"""
    d = get_object_or_404(
        Deliverable, pk=del_id,
        performance__contract__created_by=request.user,
    )
    if not d.file:
        return JsonResponse({'pages': 0})
    try:
        from pdf2image import pdfinfo_from_path
        poppler_path = r"C:\poppler-24.08.0\Library\bin"
        info = pdfinfo_from_path(
            d.file.path,
            poppler_path=poppler_path if os.name == 'nt' else None,
        )
        return JsonResponse({'pages': info['Pages']})
    except Exception:
        return JsonResponse({'pages': 1})


@login_required
def deliverable_page_image(request, del_id, page):
    """산출물 PDF 한 페이지를 PNG 이미지로 렌더링"""
    d = get_object_or_404(
        Deliverable, pk=del_id,
        performance__contract__created_by=request.user,
    )
    if not d.file:
        return HttpResponse(status=404)
    try:
        from pdf2image import convert_from_path
        import io, shutil, tempfile

        poppler_path = r"C:\poppler-24.08.0\Library\bin"

        # 한글 경로 문제 해결 - 임시 파일로 복사
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name
            shutil.copy2(d.file.path, tmp_path)

        images = convert_from_path(
            tmp_path,
            dpi=150,
            first_page=page,
            last_page=page,
            poppler_path=poppler_path if os.name == 'nt' else None,
        )
        os.unlink(tmp_path)

        if not images:
            return HttpResponse(status=404)

        buf = io.BytesIO()
        images[0].save(buf, format='PNG')
        buf.seek(0)
        return HttpResponse(buf.read(), content_type='image/png')

    except Exception:
        import traceback
        return HttpResponse(traceback.format_exc(), content_type='text/plain', status=500)


@login_required
@require_POST
def deliverable_ai_analyze(request, del_id):
    """
    산출물 AI 분석 (현재 2번 평가만 구현: 평가기준서 기반 누락·일관성 판정).

    ※ 백엔드는 목업(더미 결과 반환).
       실제 연동 시 RAG(평가기준서) + sLLM(EXAONE Fine-tuned) 추론 결과로 교체.

    구현 예정(자리만):
      1. 양식 추가/누락 항목 검사 (계약서 검토와 동일 방식)
      3. 계약서·요구사항정의서·수행계획서 간 날짜/항목 정합성 검사
    """
    d = get_object_or_404(
        Deliverable, pk=del_id,
        performance__contract__created_by=request.user,
    )

    analyzable = DELIVERABLE_CRITERIA.get(d.deliverable_type) is not None
    if not analyzable:
        return JsonResponse({
            'status': 'unsupported',
            'message': '해당 산출물 유형은 아직 평가기준서가 준비되지 않았습니다.',
        }, status=400)

    # ── 기술적용결과표: 체크박스 정합성 검증 (규칙 기반, LLM 없음) ──
    if d.deliverable_type == 'tech_apply':
        if not d.file:
            return JsonResponse({'status': 'error', 'message': '파일이 없습니다.'}, status=400)

        from .tech_apply_checker import check_tech_apply
        try:
            result = check_tech_apply(d.file.path)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'분석 중 오류가 발생했습니다: {e}'}, status=400)

        # 결과를 저장해두면, 화면을 나갔다 다시 들어와도 재분석 없이 복원된다
        # (파일이 바뀌면 deliverable_upload에서 이 레코드를 지워서 초기화한다)
        from .models import TechApplyCheckResult
        TechApplyCheckResult.objects.update_or_create(
            deliverable=d,
            defaults={'result_json': result, 'checked_at': timezone.now()},
        )

        from .models import AIAnalysisLog
        if result['error_count']:
            logger.info(
                '[QA] 기술적용결과표(deliverable_id=%s) AI 분석 이슈 %d건 발견 (전체 %d건 중)',
                d.id, result['error_count'], result['total'],
            )
            AIAnalysisLog.log(
                d, 'analysis_issue', issue_count=result['error_count'],
                detail={'total': result['total']}, user=request.user,
            )
        else:
            logger.info(
                '[QA] 기술적용결과표(deliverable_id=%s) AI 분석 이슈 없음 (전체 %d건)',
                d.id, result['total'],
            )
            AIAnalysisLog.log(
                d, 'analysis_ok', detail={'total': result['total']}, user=request.user,
            )

        return JsonResponse({
            'status': 'ok',
            'analysis_type': 'tech_apply_check',
            'total': result['total'],
            'error_count': result['error_count'],
            'items': result['items'],
        })

    # ── 그 외(kickoff): 목업 결과 (평가기준서: 16개 목차 항목 × 9개 품질특성) ──
    result = _mock_kickoff_analysis()
    return JsonResponse({
        'status': 'ok',
        'analysis_type': 'kickoff_quality',
        'qualities': result['qualities'],
        'sections': result['sections'],
    })


@login_required
@require_POST
def deliverable_export_pdf(request, del_id):
    """산출물 AI 품질분석 결과를 PDF로 다운로드.

    백엔드 분석이 목업(미저장)이라, 프런트에서 분석 결과 JSON을 POST로 전달받아 PDF 생성.
    (실제 결과 저장 모델 도입 시 GET + DB 조회 방식으로 전환 가능)
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import io
    from datetime import datetime, timezone
    from pathlib import Path
    

    d = get_object_or_404(
        Deliverable, pk=del_id,
        performance__contract__created_by=request.user,
    )

    try:
        payload = json.loads(request.body or '{}')
    except Exception:
        payload = {}
    qualities = payload.get('qualities') or []
    sections = payload.get('sections') or []

    if not sections:
        return HttpResponse("분석 결과가 없습니다. AI 분석을 먼저 실행해주세요.", status=400)

    # 한국어 폰트 등록 (Windows 맑은 고딕)
    # FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
    # FONT_BOLD_PATH = r"C:\Windows\Fonts\malgunbd.ttf"
    _bundle = Path(__file__).resolve().parent.parent / "assets" / "fonts"
    if (_bundle / "NanumGothic.ttf").exists():
        FONT_PATH, FONT_BOLD_PATH = str(_bundle / "NanumGothic.ttf"), str(_bundle / "NanumGothicBold.ttf")
    elif os.name == "nt":
        FONT_PATH, FONT_BOLD_PATH = r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf"
    else:
        FONT_PATH, FONT_BOLD_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"

    try:
        if "MalgunGothic" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("MalgunGothic", FONT_PATH))
            pdfmetrics.registerFont(TTFont("MalgunGothicBold", FONT_BOLD_PATH))
        FONT = "MalgunGothic"
        FONT_BOLD = "MalgunGothicBold"
    except Exception:
        FONT = "Helvetica"
        FONT_BOLD = "Helvetica-Bold"

    def style(name, **kwargs):
        if 'fontName' not in kwargs:
            kwargs['fontName'] = FONT
        return ParagraphStyle(name, **kwargs)

    S = {
        "title": style("title", fontName=FONT_BOLD, fontSize=18, leading=26, spaceAfter=4),
        "subtitle": style("subtitle", fontSize=10, textColor=colors.HexColor("#666666"), spaceAfter=4),
        "section": style("section", fontName=FONT_BOLD, fontSize=12, leading=18, spaceBefore=12, spaceAfter=6),
        "secsub": style("secsub", fontName=FONT_BOLD, fontSize=10.5, leading=15, spaceBefore=8, spaceAfter=3,
                        textColor=colors.HexColor("#4f46e5")),
        "body": style("body", fontSize=9.5, leading=14, spaceAfter=2, leftIndent=10),
        "footer": style("footer", fontSize=8, textColor=colors.HexColor("#aaaaaa")),
    }

    LEVEL_LABEL = {"bad": "부적합", "warn": "보완 권고", "ok": "적합"}
    LEVEL_COLOR = {
        "bad": colors.HexColor("#dc2626"),
        "warn": colors.HexColor("#d97706"),
        "ok": colors.HexColor("#2563eb"),
    }

    def score_color(sc):
        return {
            5: colors.HexColor("#2563eb"), 4: colors.HexColor("#60a5fa"),
            3: colors.HexColor("#facc15"), 2: colors.HexColor("#f97316"),
            1: colors.HexColor("#dc2626"),
        }.get(sc, colors.HexColor("#cccccc"))

    def score_text_color(sc):
        return colors.HexColor("#5b4d00") if sc == 3 else colors.white

    buffer = io.BytesIO()
    W = A4[0] - 40 * mm
    pdf = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )

    story = []
    today = datetime.now(timezone.utc).strftime("%Y년 %m월 %d일")
    contract = d.performance.contract

    # 집계
    bad_cnt = warn_cnt = ok_cnt = 0
    for s in sections:
        for issue in (s.get('issues') or []):
            lv = issue[0] if issue else 'ok'
            if lv == 'bad':
                bad_cnt += 1
            elif lv == 'warn':
                warn_cnt += 1
            else:
                ok_cnt += 1

    # 헤더
    story.append(Paragraph("AI 산출물 품질분석 결과보고서", S["title"]))
    story.append(Paragraph("Workit — 정보화사업 산출물 AI 품질평가 플랫폼", S["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#4f46e5")))
    story.append(Spacer(1, 6))

    info = [
        ["산출물", f"{d.get_deliverable_type_display()} — {d.filename()}"],
        ["프로젝트명", contract.project_name],
        ["수행 업체", contract.company_name],
        ["평가 일자", today],
        ["평가 기준", "사업수행계획서 품질평가 기준 (16개 목차 항목 × 9개 품질특성)"],
        ["분석 요약", f"부적합 {bad_cnt}건 · 보완 권고 {warn_cnt}건 · 적합 {ok_cnt}건"],
    ]
    t = Table(info, colWidths=[28*mm, W - 28*mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), FONT),
        ("FONTNAME", (0,0), (0,-1), FONT_BOLD),
        ("FONTSIZE", (0,0), (-1,-1), 9.5),
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#4f46e5")),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f5f3ff")),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#e0e0e0")),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "※ 본 보고서는 sLLM이 평가기준서를 근거로 자동 생성한 품질평가 의견입니다. 누락·일관성 보완점을 표시하며 법적 검토는 포함하지 않습니다.",
        S["footer"]
    ))
    story.append(Spacer(1, 8))

    # 품질특성 점수표
    story.append(Paragraph("목차 항목별 품질특성 점수", S["section"]))
    q_short = ['완전', '정확', '명확', '일관', '특이', '검증', '수정', '추적', '이해']
    header = ["목차 항목"] + q_short
    table_data = [header]
    cell_styles = []
    for ri, s in enumerate(sections, start=1):
        scores = s.get('scores') or []
        row = [f"{s.get('no')}. {s.get('name')}"] + [str(x) for x in scores]
        table_data.append(row)
        for ci, sc in enumerate(scores, start=1):
            cell_styles.append(("BACKGROUND", (ci, ri), (ci, ri), score_color(sc)))
            cell_styles.append(("TEXTCOLOR", (ci, ri), (ci, ri), score_text_color(sc)))

    name_w = 44 * mm
    q_w = (W - name_w) / 9.0
    qt = Table(table_data, colWidths=[name_w] + [q_w]*9, repeatRows=1)
    qt.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), FONT),
        ("FONTNAME", (0,0), (-1,0), FONT_BOLD),
        ("FONTSIZE", (0,0), (-1,0), 8),
        ("FONTSIZE", (0,1), (-1,-1), 8.5),
        ("FONTNAME", (0,1), (0,-1), FONT_BOLD),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a237e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("ALIGN", (0,0), (0,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#dddddd")),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,1), (0,-1), 6),
    ] + cell_styles))
    story.append(qt)
    story.append(Spacer(1, 4))
    story.append(Paragraph("점수: 5 우수 · 4 양호 · 3 보통 · 2 미흡 · 1 불량", S["footer"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd")))

    # 보완 필요 항목 상세
    story.append(Paragraph("보완이 필요한 항목", S["section"]))
    any_issue = False
    for s in sections:
        flagged = [iss for iss in (s.get('issues') or []) if iss and iss[0] != 'ok']
        if not flagged:
            continue
        any_issue = True
        story.append(Paragraph(f"{s.get('no')}. {s.get('name')}", S["secsub"]))
        for iss in flagged:
            lv = iss[0]
            txt = iss[1] if len(iss) > 1 else ''
            label = LEVEL_LABEL.get(lv, '')
            color_hex = LEVEL_COLOR.get(lv, colors.black).hexval()[2:]
            story.append(Paragraph(
                f'<font color="#{color_hex}"><b>[{label}]</b></font> {txt}', S["body"]
            ))
        story.append(Spacer(1, 3))
    if not any_issue:
        story.append(Paragraph("보완이 필요한 항목이 발견되지 않았습니다.", S["body"]))

    # 푸터
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"본 보고서는 Workit이 {today}에 자동 생성했습니다.", S["footer"]))

    pdf.build(story)

    base_name = d.filename().rsplit('.', 1)[0] if d.filename() else d.get_deliverable_type_display()
    filename = f"{base_name}_AI품질분석결과.pdf"
    response = HttpResponse(buffer.getvalue(), content_type="application/pdf")
    # 한글 파일명 인코딩 (RFC 5987)
    from urllib.parse import quote
    response["Content-Disposition"] = f"attachment; filename*=UTF-8''{quote(filename)}"
    return response


def _mock_kickoff_analysis():
    """
    사업수행계획서(착수보고서) 평가기준서 기반 목업 분석 결과.
    실제 sLLM 연동 시 이 함수의 반환값만 교체하면 됨.

    구조:
      qualities: 9개 품질특성 이름 (열 순서)
      sections: 16개 목차 항목, 각 항목에 9개 점수 + 이슈 리스트
        issue level: 'ok'(적합) / 'warn'(권고) / 'bad'(부적합)
    """
    qualities = ['완전성', '정확성', '명확성', '일관성', '특이성',
                 '검증가능성', '수정용이성', '추적성', '이해가능성']

    sections = [
        {'no': 1, 'name': '사업명', 'scores': [4, 4, 5, 4, 3, 3, 4, 3, 5], 'issues': [
            ['ok', '사업명·계약명·과제번호가 일치하여 식별이 명확함'],
            ['warn', '사업 유형(신규구축/고도화 등) 구분 표기 없음'],
            ['bad', '관련 근거 법령·지침 번호 미기재로 추적 불가'],
        ]},
        {'no': 2, 'name': '사업기간', 'scores': [4, 4, 5, 4, 3, 4, 4, 3, 5], 'issues': [
            ['ok', '계약기간·착수보고·최종보고·검수완료 일정이 분리되어 기재됨'],
            ['warn', '검수 완료일과 최종보고일 간 차이 발생 사유 미설명'],
            ['bad', '단계별 Phase 기간과 본 항목 일정 간 공식적 연결 기재 없음'],
        ]},
        {'no': 3, 'name': '사업목적', 'scores': [4, 4, 4, 4, 3, 3, 4, 3, 4], 'issues': [
            ['ok', '추진배경에 현행 문제점을 수치로 제시함'],
            ['warn', '목적 항목 간 우선순위·중요도 구분 없음'],
            ['warn', '사업목적과 사업범위 간 직접 매핑 부재(추적성 미흡)'],
        ]},
        {'no': 4, 'name': '사업범위', 'scores': [3, 4, 4, 3, 1, 2, 1, 1, 4], 'issues': [
            ['bad', '기능 항목에 고유 ID(기능ID/요구사항ID)가 없어 추적·수정 곤란'],
            ['bad', '우선순위·중요도·난이도·필수/선택 구분 전무'],
            ['bad', '각 기능의 인수 기준(Acceptance Criteria) 미정의로 검증 불가'],
            ['warn', '비기능 요구사항(성능·보안·접근성)이 범위 항목에 미포함'],
            ['ok', '개발/운영 환경을 표로 구분하여 명확히 제시'],
        ]},
        {'no': 5, 'name': '사업추진체계', 'scores': [4, 4, 4, 3, 3, 3, 3, 2, 4], 'issues': [
            ['ok', '발주기관·이용기관·사업자 역할 구분이 명확함'],
            ['warn', '참여인력과 산출물계획 담당자 간 연결(추적) 미흡'],
            ['bad', '에스컬레이션 경로가 조직도에서 파악되지 않음'],
        ]},
        {'no': 6, 'name': '사업추진절차', 'scores': [3, 3, 3, 2, 2, 3, 3, 2, 4], 'issues': [
            ['warn', 'Task명이 일정계획·산출물계획 명칭과 부분적으로 불일치'],
            ['bad', '일부 단계의 산출물이 산출물계획 항목과 대응되지 않음'],
        ]},
        {'no': 7, 'name': '산출물계획', 'scores': [3, 4, 4, 2, 2, 3, 3, 2, 4], 'issues': [
            ['warn', '제출일정이 일정계획 Task 완료 시점과 일부 어긋남'],
            ['bad', '각 산출물이 사업추진절차의 어느 Task에서 생성되는지 연결 부재'],
        ]},
        {'no': 8, 'name': '일정계획', 'scores': [3, 3, 3, 2, 2, 2, 2, 2, 3], 'issues': [
            ['warn', '간트차트는 있으나 Task 간 의존성(선후관계) 표기 미흡'],
            ['bad', 'Task명이 사업추진절차 단위업무명과 통일되지 않음'],
        ]},
        {'no': 9, 'name': '공정별 투입인력계획', 'scores': [3, 3, 3, 2, 2, 2, 3, 2, 3], 'issues': [
            ['warn', '직무명이 업무분장표 역할명과 용어가 다름(일관성 미흡)'],
            ['bad', '단계별 투입량 합계가 일정계획 기간과 논리적으로 어긋남'],
        ]},
        {'no': 10, 'name': '보고계획', 'scores': [4, 4, 4, 3, 3, 3, 4, 3, 4], 'issues': [
            ['ok', '주간·월간·단계별·최종보고 유형이 모두 포함됨'],
            ['warn', '품질보증활동보고 주기가 품질보증계획 주기와 일부 불일치'],
        ]},
        {'no': 11, 'name': '표준화계획', 'scores': [4, 4, 4, 3, 3, 3, 3, 3, 4], 'issues': [
            ['ok', '표준화 항목·적용 대상·시점이 구분되어 기재됨'],
            ['warn', '"준수한다" 수준의 선언적 표현이 일부 존재(실행 수단 부족)'],
        ]},
        {'no': 12, 'name': '품질보증계획', 'scores': [3, 4, 4, 2, 3, 3, 3, 2, 4], 'issues': [
            ['warn', '품질 목표가 일부 정성적 표현에 머물러 측정 기준 부족'],
            ['bad', '테스트 결과서 등 산출물과의 연결(추적) 미흡'],
        ]},
        {'no': 13, 'name': '위험관리계획', 'scores': [4, 4, 4, 3, 3, 3, 4, 3, 4], 'issues': [
            ['ok', '위험 유형별 식별과 대응방안이 표로 정리됨'],
            ['warn', '위험 항목과 사업추진절차 단계 간 연계 표기 보완 권고'],
        ]},
        {'no': 14, 'name': '보안대책', 'scores': [4, 4, 4, 3, 3, 3, 3, 3, 4], 'issues': [
            ['ok', '문서·통신·시스템·개인정보·시큐어코딩 항목이 포함됨'],
            ['warn', '암호화 프로토콜 버전 등 기술 명세 일부 누락'],
        ]},
        {'no': 15, 'name': '교육계획', 'scores': [4, 4, 4, 3, 3, 3, 4, 3, 4], 'issues': [
            ['ok', '교육과목·일정·대상·지원사항이 기재됨'],
            ['warn', '교육 이행 완료 확인 방법(수료증·만족도 등) 명시 권고'],
        ]},
        {'no': 16, 'name': '발주기관 협조요청사항', 'scores': [4, 4, 4, 3, 3, 4, 4, 3, 4], 'issues': [
            ['ok', '협조 항목·시기·담당자가 표로 정리됨'],
            ['warn', '협조 필요 시기가 일정계획 단계 시작 전으로 설정되었는지 확인 필요'],
        ]},
    ]

    return {'qualities': qualities, 'sections': sections}

@login_required
def notification_list(request):
    """드롭다운용 - 최신 5개 (읽음 여부 무관) + 안읽은 개수"""
    recent = Notification.objects.filter(user=request.user).order_by('-created_at')[:5]
    unread_count = Notification.objects.filter(user=request.user, is_read=False).count()
    return JsonResponse({
        'unread_count': unread_count,
        'notifications': [
            {
                'id':         n.id,
                'message':    n.message,
                'url':        n.url,
                'created_at': timezone.localtime(n.created_at).strftime('%Y-%m-%d %H:%M'),  # localtime 적용
                'is_read':    n.is_read,
            }
            for n in recent
        ],
    })


@login_required
def notification_page(request):
    """전체보기 페이지 - 전체 이력"""
    notifications = Notification.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'performance/notification_page.html', {
        'notifications': notifications,
    })


@login_required
@require_POST
def notification_read(request, pk):
    Notification.objects.filter(pk=pk, user=request.user).update(is_read=True)
    return JsonResponse({'status': 'ok'})


@login_required
@require_POST
def notification_read_all(request):
    Notification.objects.filter(user=request.user, is_read=False).update(is_read=True)
    return JsonResponse({'status': 'ok'})