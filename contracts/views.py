import json
import os
from django.http import HttpResponse
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


# @login_required
# @require_POST
# def document_ai_analyze(request, doc_id):
#     """RAG + sLLM(EXAONE Fine-tuned) 기반 계약서 AI 분석"""
#     import sys
#     import os
#     import traceback

#     doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)

#     # ── rag/, data/ 디렉토리를 import 경로에 추가 ──
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     for extra_path in [
#         os.path.join(BASE_DIR, 'rag'),
#         os.path.join(BASE_DIR, 'data'),
#     ]:
#         if extra_path not in sys.path:
#             sys.path.insert(0, extra_path)

#     try:
#         from contracts.utils import extract_text, parse_to_workit

#         # ── Step 1. 파일에서 텍스트 추출 ──
#         file_text = extract_text(doc.file.path)
#         if not file_text.strip():
#             return JsonResponse(
#                 {'status': 'error', 'message': '텍스트를 추출할 수 없는 파일 형식입니다.'},
#                 status=400,
#             )

#         # ── Step 2. RAG: 조항 청킹 + Qdrant 법령 검색 ──
#         from sentence_transformers import SentenceTransformer
#         from qdrant_client import QdrantClient
#         from yoonha_contract_rag import review_contract, results_to_json

#         QDRANT_PATH = os.path.join(BASE_DIR, 'vectorstore', 'qdrant_storage')
#         embed_model = SentenceTransformer('BAAI/bge-m3')
#         qdrant_client = QdrantClient(path=QDRANT_PATH)

#         clause_results = review_contract(
#             contract_text=file_text,
#             client=qdrant_client,
#             model=embed_model,
#             risk_only=True,   # 위험 조항 관련 법령만 검색
#         )
#         # contract_review_output.json 과 동일한 구조의 list[dict]
#         rag_results = results_to_json(clause_results)

#         # ── Step 3. sLLM(EXAONE Fine-tuned) 추론 ──
#         from jihye_inference import load_model, predict

#         llm_model, tokenizer = load_model()

#         inference_results = []
#         for item in rag_results:
#             if not item.get('law_refs'):
#                 continue

#             print(f"[{rag_results.index(item)+1}/{len(rag_results)}] 판정 중: {item['clause_number']}", flush=True)

#             prediction = predict(
#                 clause_text=item['clause_text'],
#                 law_refs=item['law_refs'],
#                 model=llm_model,
#                 tokenizer=tokenizer,
#             )
#             inference_results.append({
#                 'clause_number': item['clause_number'],
#                 'clause_text':   item['clause_text'],
#                 'risk_names':    item.get('risk_names', []),
#                 'prediction':    prediction,
#             })

#         # ── Step 4. Workit 화면 형식으로 변환 ──
#         parsed = parse_to_workit(inference_results)

#         # ── Step 5. DB 저장 ──
#         AIReviewResult.objects.update_or_create(
#             document=doc,
#             defaults={
#                 'blanks':       parsed['blanks'],
#                 'typos':        parsed['typos'],
#                 'legal_issues': parsed['legal_issues'],
#             },
#         )

#         total = (
#             len(parsed['blanks'])
#             + len(parsed['typos'])
#             + len(parsed['legal_issues'])
#         )
#         return JsonResponse({
#             'status':       'ok',
#             'total':        total,
#             'blanks':       parsed['blanks'],
#             'typos':        parsed['typos'],
#             'legal_issues': parsed['legal_issues'],
#         })

#     except Exception as e:
#         traceback.print_exc()
#         return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

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


# @login_required
# @require_POST
# def document_complete_review(request, doc_id):
#     doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)
#     doc.review_status = 'reviewed'
#     doc.save()
#     contract = doc.contract
#     contract.status = 'in_progress'
#     contract.save()
#     return JsonResponse({'status': 'ok', 'redirect': '/performance/'})

@login_required
@require_POST
def document_complete_review(request, doc_id):
    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)
    doc.review_status = 'reviewed'
    doc.save()

    contract = doc.contract
    contract.status = 'in_progress'

    # 계약서에서 계약기간 추출
    try:
        from contracts.utils import extract_text
        import re
        from datetime import date

        text = extract_text(doc.file.path)
        pattern = r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일부터\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일까지"
        match = re.search(pattern, text)
        if match:
            y1, m1, d1, y2, m2, d2 = map(int, match.groups())
            contract.contract_start = date(y1, m1, d1)
            contract.contract_end   = date(y2, m2, d2)
    except Exception:
        pass

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

@login_required
def document_page_image(request, doc_id, page):
    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)
    
    try:
        from pdf2image import convert_from_path
        import io, shutil, tempfile

        poppler_path = r"C:\poppler-24.08.0\Library\bin"

        # 한글 경로 문제 해결 - 임시 파일로 복사
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name
            # print(f"tmp_path: {tmp_path}")
            shutil.copy2(doc.file.path, tmp_path)

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

        info = pdfinfo_from_path(
            doc.file.path,
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

    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)

    try:
        result = doc.review_result
    except AIReviewResult.DoesNotExist:
        return HttpResponse("분석 결과가 없습니다. AI 분석을 먼저 실행해주세요.", status=404)

    # 한국어 폰트 등록 
    # Windows로 변경
    FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"        # 맑은 고딕
    FONT_BOLD_PATH = r"C:\Windows\Fonts\malgunbd.ttf" # 맑은 고딕 Bold

    try:
        if "MalgunGothic" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("MalgunGothic", FONT_PATH))
            pdfmetrics.registerFont(TTFont("MalgunGothicBold", FONT_BOLD_PATH))
        FONT = "MalgunGothic"
        FONT_BOLD = "MalgunGothicBold"
    except Exception:
        FONT = "Helvetica"
        FONT_BOLD = "Helvetica-Bold"
    # FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    # FONT_BOLD_PATH = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
    # try:
    #     if "NanumGothic" not in pdfmetrics.getRegisteredFontNames():
    #         pdfmetrics.registerFont(TTFont("NanumGothic", FONT_PATH))
    #         pdfmetrics.registerFont(TTFont("NanumGothicBold", FONT_BOLD_PATH))
    #     FONT = "NanumGothic"
    #     FONT_BOLD = "NanumGothicBold"
    # except Exception:
    #     # 폰트 없으면 기본 폰트 사용 (한글 깨질 수 있음)
    #     FONT = "Helvetica"
    #     FONT_BOLD = "Helvetica-Bold"

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
        "blank": colors.HexColor("#4f46e5"),
        "typo": colors.HexColor("#d97706"),
        "legal": colors.HexColor("#dc2626"),
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

    pdf = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )

    story = []
    today = datetime.now(timezone.utc).strftime("%Y년 %m월 %d일")
    contract = doc.contract

    blanks = result.blanks or []
    typos = result.typos or []
    legal = result.legal_issues or []
    total = len(blanks) + len(typos) + len(legal)

    # 헤더
    story.append(Paragraph("AI 계약서 검토 결과보고서", S["title"]))
    story.append(Paragraph("Workit — 정보화사업 계약서 AI 검토 플랫폼", S["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#4f46e5")))
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
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#4f46e5")),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f5f3ff")),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#e0e0e0")),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "※ 본 보고서는 sLLM이 자동 생성한 검토 의견입니다. 확인이 필요한 항목만 표시하며 수정안은 제공하지 않습니다.",
        S["footer"]
    ))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd")))

    # 항목별 카드 출력 함수
    def add_cards(items, tag):
        if not items:
            return
        story.append(Paragraph(
            f"{TAG_LABEL[tag]} ({len(items)}건)", S["section"]
        ))
        for item in items:
            location = item.get("location", "")
            desc = item.get("description") or item.get("issue", "")
            text = item.get("text", "")
            ref = item.get("legal_ref", "")

            # 태그 + 위치 행
            badge = Table(
                [[Paragraph(TAG_LABEL[tag], ParagraphStyle(
                    "badge", fontName=FONT_BOLD, fontSize=9,
                    textColor=TAG_COLOR[tag], alignment=1
                  )),
                  Paragraph(location, S["footer"])]],
                colWidths=[18*mm, W - 18*mm]
            )
            badge.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (0,0),   TAG_BG[tag]),
                ("BACKGROUND", (1,0), (1,0),   colors.HexColor("#fafafa")),
                ("BOX", (0,0), (-1,-1), 0.5, TAG_COLOR[tag]),
                ("TOPPADDING", (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                ("LEFTPADDING", (0,0), (-1,-1), 8),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ]))
            story.append(badge)
            if desc:
                story.append(Paragraph(desc, S["body"]))
            if text:
                story.append(Paragraph(f'"{text}"', S["quote"]))
            if ref:
                story.append(Paragraph(f"근거: {ref}", S["ref"]))
            story.append(Spacer(1, 4))

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
    filename = f"{doc.filename().rsplit('.', 1)[0]}_AI검토결과.pdf"
    response = HttpResponse(buffer.getvalue(), content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response