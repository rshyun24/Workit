import json
import os
import re
import sys
from django.core.exceptions import ValidationError
from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from .models import Contract, ContractDocument, AIReviewResult
from accounts.audit import log_audit
from accounts.models import AuditLog
from accounts.file_validators import validate_uploaded_file

# rag/hwp_converter.py 등을 import할 수 있도록 프로젝트 루트의 rag 폴더를 경로에 추가
_RAG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rag')
if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)


@login_required
def contract_list(request):
    # 계약 검토중 상태만 표시 (이행중/완료는 이행관리로 이동했으므로 제외)
    contracts = Contract.objects.filter(
        created_by=request.user,
        status='reviewing',
    ).order_by('-created_at')
    return render(request, 'contracts/contract_list.html', {'contracts': contracts})


@login_required
def contract_create(request):
    if request.method == 'POST':
        doc_fields = [
            ('requirements_doc', 'requirements'),
            ('rfp_doc', 'rfp'),
            ('contract_doc', 'contract'),
        ]

        # 계약을 만들기 전에 첨부 파일을 전부 먼저 검증한다 — 하나라도
        # 형식이 안 맞으면 빈 계약 레코드가 남지 않도록 함
        pending_files = []
        for field_name, doc_type in doc_fields:
            f = request.FILES.get(field_name)
            if f:
                try:
                    validate_uploaded_file(f)
                except ValidationError as e:
                    return JsonResponse({'status': 'error', 'message': str(e.message)}, status=400)
                pending_files.append((doc_type, f))

        contract = Contract.objects.create(
            project_name=request.POST.get('project_name'),
            company_name=request.POST.get('company_name'),
            issuing_org=request.POST.get('issuing_org', ''),
            budget=request.POST.get('budget', ''),
            contact_person=request.POST.get('contact_person', ''),
            created_by=request.user,
            status='reviewing',
        )
        for doc_type, f in pending_files:
            doc = ContractDocument.objects.create(
                contract=contract,
                doc_type=doc_type,
                file=f,
                original_filename=f.name,
            )
            log_audit(request, AuditLog.ACTION_UPLOAD, 'contract_document', doc.id, detail=f'{doc_type} 최초 업로드')

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
def document_view(request, doc_id):
    """문서 이미지 뷰어 (AI 패널 없음)"""
    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)
    log_audit(request, AuditLog.ACTION_VIEW, 'contract_document', doc.id)
    return render(request, 'contracts/document_view.html', {
        'doc': doc,
        'contract': doc.contract,
    })


@login_required
def document_analyze(request, doc_id):
    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)
    log_audit(request, AuditLog.ACTION_VIEW, 'contract_document', doc.id)
    try:
        result = doc.review_result
    except AIReviewResult.DoesNotExist:
        result = None

    # JSONField(blanks/typos/legal_issues) 안에 Python None/True/False가 섞여 있으면
    # 템플릿에서 {{ ... |safe }}로 그대로 출력될 때 JS 입장에서 깨진 문법(None, True, False)이
    # 되어버린다(JS에는 null/true/false만 있음). json.dumps로 미리 직렬화해서 넘긴다.
    result_json = None
    if result:
        result_json = json.dumps({
            'blanks': result.blanks or [],
            'typos': result.typos or [],
            'legal_issues': result.legal_issues or [],
        })

    return render(request, 'contracts/document_analyze.html', {
        'doc': doc,
        'contract': doc.contract,
        'result': result,
        'result_json': result_json,
    })


@login_required
@require_POST
def document_ai_analyze(request, doc_id):
    """AI 분석 태스크 시작"""
    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)
    from contracts.tasks import analyze_document_task
    task = analyze_document_task.delay(doc_id)
    return JsonResponse({'status': 'started', 'task_id': task.id})


@login_required
def document_ai_status(request, task_id):
    """태스크 진행 상태 조회"""
    from celery.result import AsyncResult
    result = AsyncResult(task_id)

    if result.state == 'PENDING':
        return JsonResponse({'state': 'pending', 'current': 0, 'total': 1})

    elif result.state == 'PROGRESS':
        meta = result.info or {}
        return JsonResponse({
            'state': 'progress',
            'current': meta.get('current', 0),
            'total': meta.get('total', 1),
        })

    elif result.state == 'SUCCESS':
        data = result.result or {}
        return JsonResponse({'state': 'success', **data})

    else:
        return JsonResponse({'state': 'error', 'message': str(result.info)})

@login_required
@require_POST
def document_complete_review(request, doc_id):
    import re
    from datetime import date
    from performance.models import Performance

    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)
    doc.review_status = 'reviewed'
    doc.save()

    contract = doc.contract
    contract.status = 'in_progress'

    # ── 이관 전 검증 ──────────────────────────────────────────────────────────
    # 1. 필수 문서 3종 확인
    REQUIRED_DOCS = [
        ('requirements', '요구사항정의서'),
        ('rfp', 'RFP(제안요청서)'),
        ('contract', '계약서'),
    ]
    existing_types = set(contract.documents.values_list('doc_type', flat=True))
    for dtype, dlabel in REQUIRED_DOCS:
        if dtype not in existing_types:
            return JsonResponse({
                'status': 'error',
                'message': f'{dlabel} 문서가 없어 이관이 불가합니다.',
            }, status=400)

    # 2. 필수 정보 확인
    REQUIRED_FIELDS = [
        ('issuing_org', '발주기관'),
        ('contact_person', '계약 담당자'),
        ('budget', '사업 예산'),
        ('company_name', '수행 업체'),
    ]
    for field, label in REQUIRED_FIELDS:
        value = getattr(contract, field, None)
        if not value or not str(value).strip():
            return JsonResponse({
                'status': 'error',
                'message': f'{label} 정보가 없어 이관이 불가합니다.',
            }, status=400)

    # 검증 통과 → 이관 처리
    doc.review_status = 'reviewed'
    doc.save()

    contract.status = 'in_progress'

    try:
        from contracts.utils import extract_text
        text = extract_text(doc.file.path)

        # 1. 계약기간 추출
        # "2026년 6월 1일부터 2026년 7월 31일까지" / "2025년 03월 01일 ~ 2025년 12월 31일" 등
        # 발주기관마다 표기(부터·까지 문구 vs ~/- 구분자)가 달라 둘 다 인식한다.
        period_pattern = (
            r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*"
            r"(?:부터|[~\-])\s*"
            r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*(?:까지)?"
        )
        match = re.search(period_pattern, text)
        if match:
            y1, m1, d1, y2, m2, d2 = map(int, match.groups())
            contract.contract_start = date(y1, m1, d1)
            contract.contract_end   = date(y2, m2, d2)
            print(f"[계약기간] {contract.contract_start} ~ {contract.contract_end}")

        contract.save()

        # 2. Performance 레코드 확보
        # 이행관리에서 사업수행계획서(kickoff)를 업로드하면 그 안의 '산출물계획' 표를 정확히 파싱해 자동 반영된다.
        # (performance.deliverable_date_sync.sync_deliverable_dates_from_kickoff)
        Performance.objects.get_or_create(contract=contract)

        # 3. RFP 파싱 비동기 태스크 시작 ← 여기만 새로 추가 
        # RFP 파일은 이관 시점에 확정됐다고 볼 수 있으므로, 이 시점에 파싱을 시작하는 것이 적절하다.
        # 파싱은 Celery 워커에서 비동기로 돌아가므로 응답 속도에 영향 없음.
        try:
            rfp_doc = contract.documents.filter(doc_type='rfp').first()
            if rfp_doc:
                from contracts.tasks import parse_rfp_task
                parse_rfp_task.delay(rfp_doc.id)
                print(f'[이관] RFP 파싱 태스크 시작 — rfp_doc_id={rfp_doc.id}')
        except Exception as task_exc:
            # 파싱 태스크 시작 실패가 이관 자체를 막아선 안 됨 — 로그만 남김
            print(f'[이관] RFP 파싱 태스크 시작 실패 (이관은 계속): {task_exc}')

    except Exception as e:
        print(f"[document_complete_review] 처리 실패: {e}")
        import traceback
        traceback.print_exc()

    return JsonResponse({'status': 'ok', 'redirect': '/performance/'})

    # 계약서에서 계약기간 추출
    # try:
    #     from contracts.utils import extract_text
    #     import re
    #     from datetime import date

    #     text = extract_text(doc.file.path)
    #     pattern = r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일부터\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일까지"
    #     match = re.search(pattern, text)
    #     if match:
    #         y1, m1, d1, y2, m2, d2 = map(int, match.groups())
    #         contract.contract_start = date(y1, m1, d1)
    #         contract.contract_end   = date(y2, m2, d2)
    # except Exception:
    #     pass

    # contract.save()
    # return JsonResponse({'status': 'ok', 'redirect': '/performance/'})


@login_required
@require_POST
def contract_update_info(request, pk):
    """계약 기본 정보 수정"""
    contract = get_object_or_404(Contract, pk=pk, created_by=request.user)
    contract.project_name = request.POST.get('project_name', contract.project_name)
    contract.company_name = request.POST.get('company_name', contract.company_name)
    contract.issuing_org = request.POST.get('issuing_org', contract.issuing_org)
    contract.budget = request.POST.get('budget', contract.budget)
    contract.contact_person = request.POST.get('contact_person', contract.contact_person)
    contract.save()
    return JsonResponse({'status': 'ok'})


@login_required
@require_POST
def contract_update_file(request, pk):
    contract = get_object_or_404(Contract, pk=pk, created_by=request.user)
    doc_type = request.POST.get('doc_type')
    f = request.FILES.get('file')
    if not f or not doc_type:
        return JsonResponse({'status': 'error', 'message': '파일 또는 문서 유형이 없습니다.'}, status=400)

    try:
        validate_uploaded_file(f)
    except ValidationError as e:
        return JsonResponse({'status': 'error', 'message': str(e.message)}, status=400)

    existing = contract.documents.filter(doc_type=doc_type).first()
    if existing:
        existing.file = f
        existing.original_filename = f.name
        existing.review_status = 'pending'
        existing.save()
        # 파일이 바뀌면 기존 AI 분석 결과는 유효하지 않으므로 폐기
        AIReviewResult.objects.filter(document=existing).delete()
        target_id = existing.id
    else:
        doc = ContractDocument.objects.create(
            contract=contract,
            doc_type=doc_type,
            file=f,
            original_filename=f.name,
        )
        target_id = doc.id
    log_audit(request, AuditLog.ACTION_UPLOAD, 'contract_document', target_id, detail=f'{doc_type} 재업로드')
    return JsonResponse({'status': 'ok', 'filename': f.name})

def _get_pdf_path_for_view(doc):
    """
    뷰어/페이지수 조회용 PDF 경로를 반환한다.
    HWP 파일이면 LibreOffice로 변환한 PDF를 media/contracts/_hwp_cache/ 에 캐시해두고
    재사용한다(매 요청마다 변환하면 느리기 때문).
    PDF면 원본 경로를 그대로 반환한다.
    """
    file_path = doc.file.path
    if not file_path.lower().endswith('.hwp'):
        return file_path

    from django.conf import settings
    cache_dir = os.path.join(settings.MEDIA_ROOT, 'contracts', '_hwp_cache')
    os.makedirs(cache_dir, exist_ok=True)

    cached_pdf_path = os.path.join(
        cache_dir,
        os.path.splitext(os.path.basename(file_path))[0] + '.pdf'
    )

    # 이미 변환된 캐시가 있고 원본보다 최신이면 그대로 사용
    if os.path.exists(cached_pdf_path) and os.path.getmtime(cached_pdf_path) >= os.path.getmtime(file_path):
        return cached_pdf_path

    from hwp_converter import convert_hwp_to_pdf
    converted_path = convert_hwp_to_pdf(file_path, cache_dir)

    # convert_hwp_to_pdf가 만든 파일명이 cached_pdf_path와 다를 수 있으니 통일
    if converted_path != cached_pdf_path and os.path.exists(converted_path):
        import shutil as _shutil
        _shutil.move(converted_path, cached_pdf_path)

    return cached_pdf_path


@login_required
def document_page_image(request, doc_id, page):
    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)
    
    try:
        from pdf2image import convert_from_path
        import io, shutil, tempfile

        poppler_path = r"C:\poppler-24.08.0\Library\bin"
        poppler_path = r"C:\poppler-24.08.0\Library\bin" if os.name == 'nt' else None

        source_path = _get_pdf_path_for_view(doc)  # HWP면 변환된 PDF 경로, PDF면 원본 그대로

        # 한글 경로 문제 해결 - 임시 파일로 복사
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name
            # print(f"tmp_path: {tmp_path}")
            shutil.copy2(source_path, tmp_path)

        images = convert_from_path(
            tmp_path,
            dpi=150,
            first_page=page,
            last_page=page,
            poppler_path=poppler_path,
        )

        os.unlink(tmp_path)  # 임시 파일 삭제

        if not images:
            return HttpResponse(status=404)

        buf = io.BytesIO()
        images[0].save(buf, format='PNG')
        buf.seek(0)
        return HttpResponse(buf.read(), content_type='image/png')

    except Exception as e:
        import traceback
        return HttpResponse(traceback.format_exc(), content_type='text/plain', status=500)

@login_required  
def document_page_count(request, doc_id):
    """PDF 총 페이지 수 반환"""
    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)
    
    try:
        from pdf2image import pdfinfo_from_path
        import io
        
        poppler_path = r"C:\poppler-24.08.0\Library\bin"

        source_path = _get_pdf_path_for_view(doc)  # HWP면 변환된 PDF 경로, PDF면 원본 그대로

        info = pdfinfo_from_path(
            source_path,
            poppler_path=poppler_path if os.name == 'nt' else None,
        )
        return JsonResponse({'pages': info['Pages']})
    
    except Exception as e:
        return JsonResponse({'pages': 1})
    
@login_required
def document_export_pdf(request, doc_id):
    """AI 분석 결과를 PDF로 다운로드"""
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

    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)

    try:
        result = doc.review_result
    except AIReviewResult.DoesNotExist:
        return HttpResponse("분석 결과가 없습니다. AI 분석을 먼저 실행해주세요.", status=404)

    # 한국어 폰트 등록 
    # Windows로 변경
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

    # 스타일 
    def style(name, **kwargs):
        if 'fontName' not in kwargs:
            kwargs['fontName'] = FONT
        return ParagraphStyle(name, **kwargs)

    S = {
        "title": style("title", fontName=FONT_BOLD, fontSize=18, leading=26, spaceAfter=4),
        "subtitle": style("subtitle", fontSize=10, textColor=colors.HexColor("#666666"), spaceAfter=4),
        "section": style("section", fontName=FONT_BOLD, fontSize=12, leading=18, spaceBefore=12, spaceAfter=6),
        "body": style("body", fontSize=10, leading=15, spaceAfter=3),
        "quote": style("quote", fontSize=9,  leading=14, leftIndent=12,
                          textColor=colors.HexColor("#555555"), spaceAfter=3),
        "ref": style("ref", fontSize=8,  leading=12,
                          textColor=colors.HexColor("#888888"), spaceAfter=6),
        "footer": style("footer", fontSize=8, textColor=colors.HexColor("#aaaaaa")),
    }

    TAG_COLOR = {
        "blank": colors.HexColor("#3c36ac"),
        "typo": colors.HexColor("#d97706"),
        "legal": colors.HexColor("#881818"),
    }
    TAG_BG = {
        "blank": colors.HexColor("#eef2ff"),
        "typo": colors.HexColor("#fffbeb"),
        "legal": colors.HexColor("#fef2f2"),
    }
    TAG_LABEL = {"blank": "빈칸", "typo": "오탈자", "legal": "법률 검토"}

    # PDF 생성
    buffer = io.BytesIO()
    W = A4[0] - 40 * mm
    contract = doc.contract

    pdf = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
        # PDF 내부 제목(메타데이터)도 채워야 브라우저 뷰어 탭에 "(anonymous)" 대신
        # 실제 문서명이 뜬다 (다운로드 파일명과 별개 — Content-Disposition은 그대로 유지)
        title=f"{contract.project_name}_{contract.company_name}_계약서_AI분석",
    )

    story = []
    today = datetime.now(timezone.utc).strftime("%Y년 %m월 %d일")

    blanks = result.blanks or []
    typos = result.typos or []
    legal = result.legal_issues or []
    total = len(blanks) + len(typos) + len(legal)

    # 헤더
    story.append(Paragraph("AI 계약서 검토 결과보고서", S["title"]))
    story.append(Paragraph("Workit — 정보화사업 계약서 AI 검토 플랫폼", S["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#040072")))
    story.append(Spacer(1, 6))

    # 기본 정보 테이블
    info = [
        ["검토 파일", doc.filename()],
        ["프로젝트명", contract.project_name],
        ["수행 업체", contract.company_name],
        ["검토 일자", today],
        ["확인 항목", f"총 {total}건 (빈칸 {len(blanks)}건 · 오탈자 {len(typos)}건 · 법률 {len(legal)}건)"],
    ]
    t = Table(info, colWidths=[32*mm, W - 32*mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), FONT),
        ("FONTNAME", (0,0), (0,-1), FONT_BOLD),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#18145c")),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f5f3ff")),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#e0e0e0")),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "※ 본 보고서는 AI가 자동 생성한 검토 의견입니다. 확인이 필요한 항목만 표시하며 수정안은 제공하지 않습니다.",
        S["footer"]
    ))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd")))

    # 항목별 카드 출력 함수
    def add_cards(items, tag):
        if not items:
            return

        story.append(Spacer(1, 8))
        story.append(Paragraph(f"{TAG_LABEL[tag]} ({len(items)}건)", S["section"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e0e0e0")))
        story.append(Spacer(1, 6))

        for item in items:
            location = item.get("location", "")
            text     = item.get("text", "") or item.get("original_text", "")

            # ── legal_ref 파싱 (판정/유형/근거 분리) ──
            if tag == "legal":
                raw_ref  = item.get("legal_ref", "")
                desc     = item.get("issue", "")           # 리스크명 목록

                판정_m = re.search(r"판정:\s*(.+)", raw_ref)
                유형_m = re.search(r"유형:\s*(.+)", raw_ref)
                근거_m = re.search(r"근거:\s*([\s\S]+)", raw_ref)

                판정 = 판정_m.group(1).strip() if 판정_m else ""
                유형 = 유형_m.group(1).strip() if 유형_m else ""
                근거 = 근거_m.group(1).strip() if 근거_m else raw_ref
            else:
                desc = item.get("description", "")
                판정 = 유형 = 근거 = ""
                ref  = item.get("legal_ref", "")

            # ── 카드 외곽 박스 (테이블로 배경+테두리 구현) ──
            card_rows = []

            # 1행: 태그 뱃지 + 위치
            card_rows.append([
                Paragraph(TAG_LABEL[tag], ParagraphStyle(
                    "badge", fontName=FONT_BOLD, fontSize=9,
                    textColor=TAG_COLOR[tag], alignment=1,
                )),
                Paragraph(location, ParagraphStyle(
                    "loc", fontName=FONT, fontSize=9,
                    textColor=colors.HexColor("#888888"), alignment=2,
                )),
            ])

            card_t = Table(card_rows, colWidths=[22*mm, W - 22*mm])
            card_t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, 0),  TAG_BG[tag]),
                ("BACKGROUND", (1, 0), (1, 0),  colors.HexColor("#fafafa")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                # ("BOX", (0, 0), (-1, -1), 1, TAG_COLOR[tag]),
                ("LINEABOVE", (0, 0), (-1, 0),  1, TAG_COLOR[tag]),
                ("LINEBEFORE", (0, 0), (0, -1),  1, TAG_COLOR[tag]),
                ("LINEAFTER", (-1, 0), (-1, -1), 1, TAG_COLOR[tag]),
                ("LINEBELOW", (0, -1), (-1, -1), 1, TAG_COLOR[tag]),
            ]))
            story.append(card_t)

            # 2행: 리스크명(issue) / description — 굵게
            if desc:
                story.append(Table(
                    [[Paragraph(desc, ParagraphStyle(
                        "desc", fontName=FONT_BOLD, fontSize=10,
                        leading=15, textColor=colors.HexColor("#111111"),
                    ))]],
                    colWidths=[W],
                    style=TableStyle([
                        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#fafafa")),
                        ("TOPPADDING", (0,0), (-1,-1), 6),
                        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
                        ("LEFTPADDING", (0,0), (-1,-1), 10),
                        ("RIGHTPADDING", (0,0), (-1,-1), 10),
                        # ("LINEBELOW", (0,0), (-1,-1), 0.3, colors.HexColor("#e0e0e0")),
                    ])
                ))

            # 3행: 판정 뱃지 + 유형 (legal 전용)
            if tag == "legal" and (판정 or 유형):
                VERDICT_COLOR = {"정상": "#16a34a", "누락": "#d97706", "위반": "#dc2626"}
                VERDICT_BG    = {"정상": "#f0fdf4", "누락": "#fffbeb", "위반": "#fef2f2"}
                vc = colors.HexColor(VERDICT_COLOR.get(판정, "#6b7280"))
                vb = colors.HexColor(VERDICT_BG.get(판정,  "#f9fafb"))

                badge_row = []
                if 판정:
                    badge_row.append(Paragraph(f"판정: {판정}", ParagraphStyle(
                        "verdict", fontName=FONT_BOLD, fontSize=9,
                        textColor=vc, backColor=vb,
                    )))
                if 유형:
                    badge_row.append(Paragraph(f"유형: {유형}", ParagraphStyle(
                        "vtype", fontName=FONT, fontSize=9,
                        textColor=colors.HexColor("#475569"),
                    )))

                col_w = W / len(badge_row)
                vt = Table([badge_row], colWidths=[col_w] * len(badge_row))
                vt.setStyle(TableStyle([
                    ("BACKGROUND", (0,0), (0,0),  vb),
                    ("BACKGROUND", (1,0), (1,0),  colors.HexColor("#f1f5f9")) if len(badge_row) > 1 else ("SPAN", (0,0),(0,0)),
                    ("TOPPADDING", (0,0), (-1,-1), 5),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                    ("LEFTPADDING", (0,0), (-1,-1), 10),
                    ("BOX", (0,0), (-1,-1), 0.3, colors.HexColor("#e0e0e0")),
                ]))
                story.append(vt)

            # 4행: 근거 텍스트 (legal) 또는 법률근거 (일반)
            ref_text = 근거 if tag == "legal" else (item.get("legal_ref", ""))
            if ref_text:
                story.append(Table(
                    [[Paragraph(ref_text, ParagraphStyle(
                        "ref2", fontName=FONT, fontSize=9,
                        leading=14, textColor=colors.HexColor("#374151"),
                    ))]],
                    colWidths=[W],
                    style=TableStyle([
                        ("BACKGROUND", (0,0), (-1,-1), colors.white),
                        ("TOPPADDING", (0,0), (-1,-1), 6),
                        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                        ("LEFTPADDING", (0,0), (-1,-1), 10),
                        ("RIGHTPADDING", (0,0), (-1,-1), 10),
                        ("BOX", (0,0), (-1,-1), 0.3, colors.HexColor("#e0e0e0")),
                    ])
                ))

            # 5행: 원문 인용 (있을 때만)
            if text:
                short = text[:120] + ("..." if len(text) > 120 else "")
                story.append(Table(
                    [[Paragraph(f'"{short}"', ParagraphStyle(
                        "qt", fontName=FONT, fontSize=8,
                        leading=13, textColor=colors.HexColor("#555555"),
                        leftIndent=4,
                    ))]],
                    colWidths=[W],
                    style=TableStyle([
                        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f8f9fa")),
                        ("TOPPADDING", (0,0), (-1,-1), 5),
                        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
                        ("LEFTPADDING", (0,0), (-1,-1), 12),
                        ("RIGHTPADDING", (0,0), (-1,-1), 10),
                        ("LINEAFTER", (0,0), (0,-1),  2, colors.HexColor("#d1d5db")),
                    ])
                ))

            story.append(Spacer(1, 10))  # 카드 간 여백

    add_cards(blanks, "blank")
    add_cards(typos, "typo")
    add_cards(legal, "legal")

    # 푸터
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"본 보고서는 Workit이 {today}에 자동 생성했습니다.", S["footer"]))

    pdf.build(story)

    # 응답 
    from urllib.parse import quote

    filename = f"{contract.project_name}_{contract.company_name}_계약서_AI분석.pdf"
    encoded_filename = quote(filename)

    response = HttpResponse(buffer.getvalue(), content_type="application/pdf")
    response["Content-Disposition"] = f"attachment; filename*=UTF-8''{encoded_filename}"
    return response