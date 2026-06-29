from celery import shared_task
from django.utils import timezone
from datetime import timedelta

TARGET_TYPES = ['tech_apply', 'result_report']

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
                        from_email=None,
                        recipient_list=[user.email],
                        fail_silently=True,
                    )
                except Exception as e:
                    print(f"[알림] 이메일 발송 실패: {e}")