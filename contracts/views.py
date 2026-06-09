import json
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from .models import Contract, ContractDocument, AIReviewResult


@login_required
def contract_list(request):
    contracts = Contract.objects.filter(created_by=request.user).order_by('-created_at')
    return render(request, 'contracts/contract_list.html', {'contracts': contracts})


@login_required
def contract_create(request):
    if request.method == 'POST':
        contract = Contract.objects.create(
            project_name=request.POST.get('project_name'),
            company_name=request.POST.get('company_name'),
            issuing_org=request.POST.get('issuing_org', ''),
            budget=request.POST.get('budget', ''),
            contact_person=request.POST.get('contact_person', ''),
            created_by=request.user,
            status='reviewing',
        )
        # Handle file uploads
        doc_fields = [
            ('requirements_doc', 'requirements'),
            ('rfp_doc', 'rfp'),
            ('contract_doc', 'contract'),
        ]
        for field_name, doc_type in doc_fields:
            f = request.FILES.get(field_name)
            if f:
                doc = ContractDocument.objects.create(
                    contract=contract,
                    doc_type=doc_type,
                    file=f,
                    original_filename=f.name,
                )
        return JsonResponse({'status': 'ok', 'id': contract.id, 'name': contract.project_name})
    return JsonResponse({'status': 'error'}, status=400)


@login_required
def contract_detail_api(request, pk):
    contract = get_object_or_404(Contract, pk=pk, created_by=request.user)
    docs = []
    for doc in contract.documents.all():
        docs.append({
            'id': doc.id,
            'doc_type': doc.doc_type,
            'doc_type_display': doc.get_doc_type_display(),
            'filename': doc.filename(),
            'review_status': doc.review_status,
            'url': doc.file.url,
        })
    return JsonResponse({
        'id': contract.id,
        'project_name': contract.project_name,
        'company_name': contract.company_name,
        'issuing_org': contract.issuing_org,
        'budget': contract.budget,
        'contact_person': contract.contact_person,
        'status': contract.status,
        'status_display': contract.get_status_display(),
        'created_at': contract.created_at.strftime('%Y-%m-%d'),
        'documents': docs,
    })


@login_required
def document_analyze(request, doc_id):
    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)
    try:
        result = doc.review_result
    except AIReviewResult.DoesNotExist:
        result = None
    return render(request, 'contracts/document_analyze.html', {
        'doc': doc,
        'contract': doc.contract,
        'result': result,
    })


@login_required
@require_POST
def document_ai_analyze(request, doc_id):
    """AI 분석 실행 (mock 데이터 반환 - 실제 sLLM 연동 시 교체)"""
    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)

    mock_blanks = [
        {"location": "p.3 · 3.2 사업 범위", "description": "총 사업비 금액이 공란입니다. 계약 체결 전 반드시 기재가 필요합니다.", "text": "본 사업의 총 사업비는 ( )원으로 하며, 부가가치세를 포함한다."},
        {"location": "p.5 · 4.3 납품 일정", "description": "최종 납품일자가 기재되지 않았습니다.", "text": "최종 납품 기한은 ____년 __월 __일로 한다."},
    ]
    mock_typos = [
        {"location": "p.4 · 4.1 과업 기간", "original": "검수 기관", "corrected": "검수 기간", "description": "\"검수 기관\" → \"검수 기간\"으로 추정되는 오탈자가 있습니다.", "text": "계약 체결일로부터 180일간 과업을 수행하며, 검수 기관은 30일로 한다."},
        {"location": "p.7 · 5.1 과업 내용", "original": "분석·설계·구현·시흠", "corrected": "분석·설계·구현·시험", "description": "\"시흠\" → \"시험\"으로 추정되는 오탈자가 있습니다.", "text": "수급인은 과업내용서에 따라 시스템 분석·설계·구현·시흠을 수행한다."},
    ]
    mock_legal = [
        {"location": "p.6 · 6.3 지체상금", "text": "계약상대자가 납품 기한 내에 계약을 이행하지 아니한 경우 지체상금을 부과한다.", "issue": "지체상금 요율 및 상한액이 명시되지 않아 분쟁 시 기준이 불명확할 수 있습니다.", "legal_ref": "국가계약법 시행령 제74조 관련"},
        {"location": "p.8 · 7.1 하자보수", "text": "납품 후 하자 발생 시 수급인이 책임진다.", "issue": "하자보수 기간 및 범위가 불명확하게 기재되어 있어 추후 분쟁 소지가 있습니다.", "legal_ref": "국가계약법 시행령 제60조 관련"},
        {"location": "p.9 · 8.2 지식재산권", "text": "본 사업의 결과물에 대한 권리는 발주기관에 귀속한다.", "issue": "오픈소스 등 제3자 지식재산권 처리 방식이 명시되지 않았습니다.", "legal_ref": "저작권법 제9조 관련"},
    ]

    result, _ = AIReviewResult.objects.update_or_create(
        document=doc,
        defaults={
            'blanks': mock_blanks,
            'typos': mock_typos,
            'legal_issues': mock_legal,
        }
    )

    total = len(mock_blanks) + len(mock_typos) + len(mock_legal)
    return JsonResponse({
        'status': 'ok',
        'total': total,
        'blanks': mock_blanks,
        'typos': mock_typos,
        'legal_issues': mock_legal,
    })


@login_required
@require_POST
def document_complete_review(request, doc_id):
    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)
    doc.review_status = 'reviewed'
    doc.save()
    contract = doc.contract
    contract.status = 'in_progress'
    contract.save()
    return JsonResponse({'status': 'ok', 'redirect': '/performance/'})


@login_required
@require_POST
def contract_update_file(request, pk):
    contract = get_object_or_404(Contract, pk=pk, created_by=request.user)
    doc_type = request.POST.get('doc_type')
    f = request.FILES.get('file')
    if not f or not doc_type:
        return JsonResponse({'status': 'error', 'message': '파일 또는 문서 유형이 없습니다.'}, status=400)

    existing = contract.documents.filter(doc_type=doc_type).first()
    if existing:
        existing.file = f
        existing.original_filename = f.name
        existing.review_status = 'pending'
        existing.save()
    else:
        ContractDocument.objects.create(
            contract=contract,
            doc_type=doc_type,
            file=f,
            original_filename=f.name,
        )
    return JsonResponse({'status': 'ok', 'filename': f.name})
