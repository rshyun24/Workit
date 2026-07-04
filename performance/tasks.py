from celery import shared_task
from django.utils import timezone
from datetime import timedelta

TARGET_TYPES = ['tech_apply', 'final']

@shared_task
def check_deadlines():
    from performance.models import Deliverable, Notification

    today = timezone.now().date()

    for days in [7, 3, 1]:
        target_date = today + timedelta(days=days)

        deliverables = Deliverable.objects.filter(
            due_date=target_date,
            status='pending',
            deliverable_type__in=TARGET_TYPES,
        ).select_related('performance__contract__created_by')

        for d in deliverables:
            contract = d.performance.contract
            user = contract.created_by
            label = d.get_deliverable_type_display()
            project = contract.project_name

            if not user.notification_enabled:
                continue

            message = f"[{project}] {label} 제출 마감 {days}일 전입니다."

            # 사이트 내 알림
            Notification.objects.create(
                user=user,
                message=message,
                url='/performance/',
            )

            # 이메일 발송
            if user.email:
                try:
                    from django.core.mail import send_mail
                    send_mail(
                        subject=f'[Workit] {label} 마감 {days}일 전 알림',
                        message=(
                            f"안녕하세요, {user.korean_name()}님.\n\n"
                            f"'{project}'의 {label} 제출 마감일이 "
                            f"{days}일 후({target_date.strftime('%Y년 %m월 %d일')})입니다.\n\n"
                            f"Workit에 접속하여 산출물을 등록해 주세요.\n"
                            f"http://your-domain.com/performance/"
                        ),
                        from_email=None,           # ← DEFAULT_FROM_EMAIL 사용
                        recipient_list=[user.email],  # ← 각 사용자 이메일로
                        fail_silently=True,
                    )
                except Exception as e:
                    print(f"[알림] 이메일 발송 실패: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 과업수행계획서 파싱 태스크 (규칙 기반, LLM 없음)
#
# 실행 시점 : 과업수행계획서 파일 업로드 직후 비동기
# 파서      : performance.parsers.parse_execution_plan (키워드·정규식 기반)
# 결과      : performance.models.ExecutionPlanParsedData.parsed_json (RDS)
# ─────────────────────────────────────────────────────────────────────────────

@shared_task(bind=True, max_retries=2, default_retry_delay=10)
def parse_execution_plan_task(self, deliverable_id: int):
    """
    과업수행계획서 파일을 규칙 기반으로 파싱해 ExecutionPlanParsedData에 저장한다.

    performance.parsers.parse_execution_plan() 를 사용하므로 LLM 없이 동작한다.
    호출 시점: 과업수행계획서 파일 업로드 직후.
    """
    from performance.models import Deliverable, ExecutionPlanParsedData
    from contracts.utils import extract_text
    from performance.parsers import parse_execution_plan

    deliverable = Deliverable.objects.select_related('performance__contract').get(pk=deliverable_id)

    parsed, _ = ExecutionPlanParsedData.objects.get_or_create(deliverable=deliverable)
    parsed.parse_status = 'processing'
    parsed.error_message = ''
    parsed.save(update_fields=['parse_status', 'error_message'])

    try:
        if not deliverable.file:
            raise ValueError('과업수행계획서 파일이 없습니다.')

        text = extract_text(deliverable.file.path)
        if not text.strip():
            raise ValueError('과업수행계획서 텍스트 추출 실패 — 파일을 확인하세요.')

        result_json = parse_execution_plan(text)

        found_count = sum(1 for s in result_json.values() if s.get('found'))
        total_count = len(result_json)

        parsed.parsed_json = result_json
        parsed.parse_status = 'done'
        parsed.parsed_at = timezone.now()
        parsed.save(update_fields=['parsed_json', 'parse_status', 'parsed_at'])

        print(
            f'[parse_execution_plan_task] 완료 — deliverable_id={deliverable_id}, '
            f'섹션 {found_count}/{total_count} 발견'
        )
        return {'status': 'ok', 'deliverable_id': deliverable_id, 'found': found_count, 'total': total_count}

    except Exception as exc:
        import traceback
        err = traceback.format_exc()
        parsed.parse_status = 'failed'
        parsed.error_message = err[:2000]
        parsed.save(update_fields=['parse_status', 'error_message'])
        print(f'[parse_execution_plan_task] 실패 — deliverable_id={deliverable_id}\n{err}')
        raise self.retry(exc=exc)


# ─────────────────────────────────────────────────────────────────────────────
# RFP ↔ 과업수행계획서 비교 태스크 (구조적 비교, LLM 없음)
#
# 실행 시점 : 비교 버튼 클릭 → rfp_compare view
# 비교 로직  : performance.parsers.compare_rfp_and_pep (코드 매핑 기반)
# 결과      : performance.models.RFPComparisonResult.comparison_json (RDS)
# ─────────────────────────────────────────────────────────────────────────────

@shared_task(bind=True, max_retries=1, default_retry_delay=10)
def compare_rfp_execution_plan_task(self, performance_id: int):
    """
    파싱된 RFP와 과업수행계획서를 구조적으로 비교해 RFPComparisonResult에 저장한다.

    performance.parsers.compare_rfp_and_pep() 를 사용하므로 LLM 없이 동작한다.
    호출 시점: 비교 버튼 클릭 → rfp_compare view.
    """
    from performance.models import Performance, ExecutionPlanParsedData, RFPComparisonResult
    from contracts.models import RFPParsedData
    from performance.parsers import compare_rfp_and_pep

    performance = Performance.objects.select_related('contract').get(pk=performance_id)
    contract = performance.contract

    # ── 전제 조건 확인 ──────────────────────────────────────────────────────

    rfp_doc = contract.documents.filter(doc_type='rfp').first()
    if not rfp_doc:
        return {'status': 'error', 'message': 'RFP 문서가 없습니다.'}

    try:
        rfp_parsed = rfp_doc.rfp_parsed
    except RFPParsedData.DoesNotExist:
        return {'status': 'error', 'message': 'RFP가 아직 파싱되지 않았습니다.'}

    if rfp_parsed.parse_status != 'done':
        return {'status': 'error', 'message': f'RFP 파싱 상태: {rfp_parsed.parse_status}'}

    execution_plan = performance.deliverables.filter(deliverable_type='execution_plan').first()
    if not execution_plan:
        return {'status': 'error', 'message': '과업수행계획서 산출물이 없습니다.'}

    try:
        pep_parsed = execution_plan.parsed_data
    except ExecutionPlanParsedData.DoesNotExist:
        return {'status': 'error', 'message': '과업수행계획서가 아직 파싱되지 않았습니다.'}

    if pep_parsed.parse_status != 'done':
        return {'status': 'error', 'message': f'과업수행계획서 파싱 상태: {pep_parsed.parse_status}'}

    # ── 구조적 비교 ─────────────────────────────────────────────────────────

    try:
        comparison_json = compare_rfp_and_pep(rfp_parsed.parsed_json, pep_parsed.parsed_json)

        # 이전 결과 삭제 후 최신 1건 저장
        RFPComparisonResult.objects.filter(performance=performance).delete()
        RFPComparisonResult.objects.create(
            performance=performance,
            rfp_parsed=rfp_parsed,
            execution_plan_parsed=pep_parsed,
            comparison_json=comparison_json,
        )

        print(
            f'[compare_rfp_execution_plan_task] 완료 — performance_id={performance_id}, '
            f'점수={comparison_json["overall_score"]}'
        )
        return {
            'status': 'ok',
            'performance_id': performance_id,
            'overall_score': comparison_json['overall_score'],
        }

    except Exception as exc:
        import traceback
        err = traceback.format_exc()
        print(f'[compare_rfp_execution_plan_task] 실패 — performance_id={performance_id}\n{err}')
        raise self.retry(exc=exc)