import logging

from celery import shared_task
from django.utils import timezone
from datetime import timedelta
import os

SITE_URL = os.environ.get('SITE_URL', 'http://localhost:8000')

logger = logging.getLogger(__name__)

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
                            f"{SITE_URL}/performance/"
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
    import os
    import sys

    from performance.models import Deliverable, ExecutionPlanParsedData
    from contracts.utils import extract_text
    from performance.parsers import parse_execution_plan, to_qa_agent_records

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    llm_dir = os.path.join(BASE_DIR, 'LLM')
    if llm_dir not in sys.path:
        sys.path.insert(0, llm_dir)

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

        # 소제목 매핑 QA 검수 (LLM/qa_agent). 검수 자체가 실패해도 파싱 성공은
        # 그대로 살려야 하므로 별도 try/except로 감싸고, 실패 시 리포트만 비워둔다.
        qa_report = {}
        try:
            from qa_agent.engine import review_section_mapping

            qa_report = review_section_mapping(
                original_text=text,
                parsed_sections=to_qa_agent_records(result_json),
                document_type='pep',
            )
        except Exception:
            import traceback
            print(f'[parse_execution_plan_task] QA 검수 실패 — deliverable_id={deliverable_id}\n{traceback.format_exc()}')

        qa_issues = qa_report.get('issues', [])
        from performance.models import AIAnalysisLog
        if qa_issues:
            logger.info(
                '[QA] 사업수행계획서(deliverable_id=%s) 1단계 QA 이슈 %d건 발견 (review_status=%s): %s',
                deliverable_id, len(qa_issues), qa_report.get('review_status'),
                [issue.get('issue_type') for issue in qa_issues],
            )
            AIAnalysisLog.log(
                deliverable, 'analysis_issue', issue_count=len(qa_issues),
                detail={'review_status': qa_report.get('review_status'),
                        'issue_types': [issue.get('issue_type') for issue in qa_issues]},
            )
        else:
            logger.info(
                '[QA] 사업수행계획서(deliverable_id=%s) 1단계 QA 이슈 없음 (review_status=%s)',
                deliverable_id, qa_report.get('review_status'),
            )
            AIAnalysisLog.log(
                deliverable, 'analysis_ok',
                detail={'review_status': qa_report.get('review_status')},
            )

        parsed.parsed_json = result_json
        parsed.qa_report = qa_report
        parsed.parse_status = 'done'
        parsed.parsed_at = timezone.now()
        parsed.save(update_fields=['parsed_json', 'qa_report', 'parse_status', 'parsed_at'])

        print(
            f'[parse_execution_plan_task] 완료 — deliverable_id={deliverable_id}, '
            f'섹션 {found_count}/{total_count} 발견, QA={qa_report.get("review_status", "N/A")}'
        )
        return {'status': 'ok', 'deliverable_id': deliverable_id, 'found': found_count, 'total': total_count,
                'qa_status': qa_report.get('review_status')}

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

    execution_plan = performance.deliverables.filter(deliverable_type='kickoff').first()
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


# ─────────────────────────────────────────────────────────────────────────────
# 사업추진결과보고서 파싱 태스크 (규칙 기반, LLM 없음)
#
# 실행 시점 : 산출물 분석 화면에서 "분석 시작" 클릭
# 파서      : performance.parsers.parse_final_report (키워드·정규식 기반)
# 결과      : performance.models.FinalReportParsedData.parsed_json (RDS)
# ─────────────────────────────────────────────────────────────────────────────

@shared_task(bind=True, max_retries=2, default_retry_delay=10)
def parse_final_report_task(self, deliverable_id: int):
    """
    사업추진결과보고서 파일을 규칙 기반으로 파싱해 FinalReportParsedData에 저장한다.

    performance.parsers.parse_final_report() 를 사용하므로 LLM 없이 동작한다.
    parse_execution_plan_task와 동일한 구조 — RPT 코드 체계로 파싱 후 qa_agent로
    소제목 매핑 QA 검수까지 함께 수행한다.
    """
    import os
    import sys

    from performance.models import Deliverable, FinalReportParsedData
    from contracts.utils import extract_text
    from performance.parsers import parse_final_report, to_qa_agent_records

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    llm_dir = os.path.join(BASE_DIR, 'LLM')
    if llm_dir not in sys.path:
        sys.path.insert(0, llm_dir)

    deliverable = Deliverable.objects.select_related('performance__contract').get(pk=deliverable_id)

    parsed, _ = FinalReportParsedData.objects.get_or_create(deliverable=deliverable)
    parsed.parse_status = 'processing'
    parsed.error_message = ''
    parsed.save(update_fields=['parse_status', 'error_message'])

    try:
        if not deliverable.file:
            raise ValueError('사업추진결과보고서 파일이 없습니다.')

        text = extract_text(deliverable.file.path)
        if not text.strip():
            raise ValueError('사업추진결과보고서 텍스트 추출 실패 — 파일을 확인하세요.')

        result_json = parse_final_report(text)

        found_count = sum(1 for s in result_json.values() if s.get('found'))
        total_count = len(result_json)

        # 소제목 매핑 QA 검수 (LLM/qa_agent). 검수 자체가 실패해도 파싱 성공은
        # 그대로 살려야 하므로 별도 try/except로 감싸고, 실패 시 리포트만 비워둔다.
        qa_report = {}
        try:
            from qa_agent.engine import review_section_mapping

            qa_report = review_section_mapping(
                original_text=text,
                parsed_sections=to_qa_agent_records(result_json),
                document_type='rpt',
            )
        except Exception:
            import traceback
            print(f'[parse_final_report_task] QA 검수 실패 — deliverable_id={deliverable_id}\n{traceback.format_exc()}')

        qa_issues = qa_report.get('issues', [])
        from performance.models import AIAnalysisLog
        if qa_issues:
            logger.info(
                '[QA] 사업추진결과보고서(deliverable_id=%s) 1단계 QA 이슈 %d건 발견 (review_status=%s): %s',
                deliverable_id, len(qa_issues), qa_report.get('review_status'),
                [issue.get('issue_type') for issue in qa_issues],
            )
            AIAnalysisLog.log(
                deliverable, 'analysis_issue', issue_count=len(qa_issues),
                detail={'review_status': qa_report.get('review_status'),
                        'issue_types': [issue.get('issue_type') for issue in qa_issues]},
            )
        else:
            logger.info(
                '[QA] 사업추진결과보고서(deliverable_id=%s) 1단계 QA 이슈 없음 (review_status=%s)',
                deliverable_id, qa_report.get('review_status'),
            )
            AIAnalysisLog.log(
                deliverable, 'analysis_ok',
                detail={'review_status': qa_report.get('review_status')},
            )

        parsed.parsed_json = result_json
        parsed.qa_report = qa_report
        parsed.parse_status = 'done'
        parsed.parsed_at = timezone.now()
        parsed.save(update_fields=['parsed_json', 'qa_report', 'parse_status', 'parsed_at'])

        print(
            f'[parse_final_report_task] 완료 — deliverable_id={deliverable_id}, '
            f'섹션 {found_count}/{total_count} 발견, QA={qa_report.get("review_status", "N/A")}'
        )
        return {'status': 'ok', 'deliverable_id': deliverable_id, 'found': found_count, 'total': total_count,
                'qa_status': qa_report.get('review_status')}

    except Exception as exc:
        import traceback
        err = traceback.format_exc()
        parsed.parse_status = 'failed'
        parsed.error_message = err[:2000]
        parsed.save(update_fields=['parse_status', 'error_message'])
        print(f'[parse_final_report_task] 실패 — deliverable_id={deliverable_id}\n{err}')
        raise self.retry(exc=exc)


# ─────────────────────────────────────────────────────────────────────────────
# PEP(사업수행계획서) ↔ 사업추진결과보고서 비교 태스크 (구조적 비교, LLM 없음)
#
# RFP 대비 이행 여부는 PEP 쪽(compare_rfp_execution_plan_task)에서 이미 확인하므로,
# 여기서는 "계획(PEP)한 대로 실제로 이행됐는지"를 PEP 대비로 비교한다.
#
# 실행 시점 : QA 검수 결과 확인 후 "그대로 진행" 버튼 클릭
# 비교 로직  : performance.parsers.compare_pep_and_final (코드 매핑 기반)
# 결과      : performance.models.PEPFinalComparisonResult.comparison_json (RDS)
# ─────────────────────────────────────────────────────────────────────────────

@shared_task(bind=True, max_retries=1, default_retry_delay=10)
def compare_pep_final_report_task(self, performance_id: int):
    """
    파싱된 사업수행계획서(PEP)와 사업추진결과보고서(RPT)를 구조적으로 비교해
    PEPFinalComparisonResult에 저장한다.

    performance.parsers.compare_pep_and_final() 를 사용하므로 LLM 없이 동작한다.
    """
    from performance.models import Performance, ExecutionPlanParsedData, FinalReportParsedData, PEPFinalComparisonResult
    from performance.parsers import compare_pep_and_final

    performance = Performance.objects.select_related('contract').get(pk=performance_id)

    # ── 전제 조건 확인 ──────────────────────────────────────────────────────

    kickoff_doc = performance.deliverables.filter(deliverable_type='kickoff').first()
    if not kickoff_doc:
        return {'status': 'error', 'message': '사업수행계획서 산출물이 없습니다.'}

    try:
        pep_parsed = kickoff_doc.parsed_data
    except ExecutionPlanParsedData.DoesNotExist:
        return {'status': 'error', 'message': '사업수행계획서가 아직 파싱되지 않았습니다.'}

    if pep_parsed.parse_status != 'done':
        # 사업수행계획서 파일이 재업로드되면 deliverable_upload()에서 parse_status를
        # 'pending'으로 되돌린다 — 여기서 그 상태를 그대로 막아서, 옛 사업수행계획서
        # 기준으로 만들어진 데이터를 사업추진결과보고서와 잘못 비교하지 않게 한다.
        if pep_parsed.parse_status == 'pending':
            message = '사업수행계획서가 파일 변경 후 아직 재분석되지 않았습니다. 이행관리에서 사업수행계획서를 먼저 분석해주세요.'
        elif pep_parsed.parse_status == 'processing':
            message = '사업수행계획서를 아직 분석 중입니다. 잠시 후 다시 시도해주세요.'
        else:
            message = f'사업수행계획서 파싱 상태: {pep_parsed.parse_status}'
        return {'status': 'error', 'message': message}

    final_doc = performance.deliverables.filter(deliverable_type='final').first()
    if not final_doc:
        return {'status': 'error', 'message': '사업추진결과보고서 산출물이 없습니다.'}

    try:
        final_parsed = final_doc.final_parsed_data
    except FinalReportParsedData.DoesNotExist:
        return {'status': 'error', 'message': '사업추진결과보고서가 아직 파싱되지 않았습니다.'}

    if final_parsed.parse_status != 'done':
        return {'status': 'error', 'message': f'사업추진결과보고서 파싱 상태: {final_parsed.parse_status}'}

    # ── 구조적 비교 ─────────────────────────────────────────────────────────

    try:
        comparison_json = compare_pep_and_final(pep_parsed.parsed_json, final_parsed.parsed_json)

        # 이전 결과 삭제 후 최신 1건 저장
        PEPFinalComparisonResult.objects.filter(performance=performance).delete()
        PEPFinalComparisonResult.objects.create(
            performance=performance,
            execution_plan_parsed=pep_parsed,
            final_report_parsed=final_parsed,
            comparison_json=comparison_json,
        )

        print(
            f'[compare_pep_final_report_task] 완료 — performance_id={performance_id}, '
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
        print(f'[compare_pep_final_report_task] 실패 — performance_id={performance_id}\n{err}')
        raise self.retry(exc=exc)


# 사업수행계획서 → 산출물 일정 자동 반영 태스크 (규칙 기반, LLM 없음)
# 실행 시점 : 사업수행계획서(kickoff) 파일 업로드 직후 비동기
# 파서 : performance.deliverable_date_extractor.parse_output_plan (표 파싱)
# 결과 : 같은 Performance의 tech_apply/final Deliverable.due_date 자동 채움
#             (이미 수동 입력된 값은 덮어쓰지 않음)

@shared_task(bind=True, max_retries=1, default_retry_delay=10)
def sync_deliverable_dates_from_kickoff_task(self, deliverable_id: int):
    """
    사업수행계획서(kickoff) Deliverable을 파싱해 그 안의 '산출물계획' 표에서
    기술적용결과표(tech_apply)/사업추진결과보고서(final)의 제출일자를 찾아
    같은 Performance의 Deliverable.due_date에 자동 반영한다.
    """
    from performance.models import Deliverable
    from performance.deliverable_date_sync import sync_deliverable_dates_from_kickoff

    deliverable = Deliverable.objects.select_related('performance__contract').get(pk=deliverable_id)

    try:
        result = sync_deliverable_dates_from_kickoff(deliverable)
        print(
            f'[sync_deliverable_dates_from_kickoff_task] 완료 — '
            f'deliverable_id={deliverable_id}, {result}'
        )
        return {'status': 'ok', 'deliverable_id': deliverable_id, **result}

    except Exception as exc:
        import traceback
        err = traceback.format_exc()
        print(f'[sync_deliverable_dates_from_kickoff_task] 실패 — deliverable_id={deliverable_id}\n{err}')
        raise self.retry(exc=exc)
