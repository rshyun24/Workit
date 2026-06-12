import json
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from contracts.models import Contract
from .models import Performance, Deliverable

PROJECT_COLORS = [
    '#4F63D2', '#00B894', '#6C5CE7', '#E67E22',
    '#E74C3C', '#1ABC9C', '#2980B9', '#8E44AD',
]


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

    # Build calendar events - 산출물별 기간(start~end) 바 형태로 표시
    # 산출물 순서에 따라 이전 산출물 due_date 다음날 ~ 현재 산출물 due_date 를 기간으로 설정
    calendar_events = []
    for idx, perf in enumerate(performances):
        color = PROJECT_COLORS[idx % len(PROJECT_COLORS)]
        contract = perf.contract
        if contract.contract_start and contract.contract_end:
            is_done = contract.status == 'completed'
            calendar_events.append({
                'id': f'contract_period_{perf.id}',
                'performance_id': perf.id,
                'project_name': contract.project_name,
                'deliverable_type': '계약 기간',
                'start_date': contract.contract_start.strftime('%Y-%m-%d'),
                'end_date': contract.contract_end.strftime('%Y-%m-%d'),
                'due_date': contract.contract_end.strftime('%Y-%m-%d'),
                'submitted_date': None,
                'status': contract.status,
                'color': '#999999' if is_done else color,
                'is_completed': is_done,
                'is_contract_period': True,
            })

    return render(request, 'performance/performance_list.html', {
        'performances': performances,
        'calendar_events_json': json.dumps(calendar_events, ensure_ascii=False),
        'project_colors': {
            perf.id: PROJECT_COLORS[idx % len(PROJECT_COLORS)]
            for idx, perf in enumerate(performances)
        },
    })


@login_required
def performance_detail_api(request, pk):
    perf = get_object_or_404(Performance, pk=pk, contract__created_by=request.user)
    contract = perf.contract

    # Build deliverable list with all 4 types
    TYPE_ORDER = ['kickoff', 'test_plan', 'test_result', 'final']
    TYPE_LABELS = {
        'kickoff': '착수보고서',
        'test_plan': '테스트 계획서',
        'test_result': '테스트 결과 보고서',
        'final': '완료보고서',
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
        })

    # Contract docs
    contract_docs = []
    for doc in contract.documents.all():
        contract_docs.append({
            'id': doc.id,
            'doc_type': doc.doc_type,
            'doc_type_display': doc.get_doc_type_display(),
            'filename': doc.filename(),
            'review_status': doc.review_status,
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
        'total': 4,
    })


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
