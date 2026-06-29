from datetime import timedelta

from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone


class User(AbstractUser):
    """Workit 사용자 계정.

    운영 정책:
    - 관리자가 초기 비밀번호로 계정을 생성한다.
    - 최초 로그인 또는 관리자 잠금해제/초기화 후 비밀번호 변경을 강제한다.
    - 5회 연속 로그인 실패 또는 90일 비밀번호 만료 시 계정을 잠근다.
    - 동일 계정 동시접속 제어를 위해 현재 세션키를 저장한다.
    """

    LOCK_REASON_FAILED_LOGIN = "FAILED_LOGIN"
    LOCK_REASON_PASSWORD_EXPIRED = "PASSWORD_EXPIRED"
    LOCK_REASON_ADMIN = "ADMIN"

    LOCK_REASON_CHOICES = (
        (LOCK_REASON_FAILED_LOGIN, "로그인 5회 연속 실패"),
        (LOCK_REASON_PASSWORD_EXPIRED, "비밀번호 90일 만료"),
        (LOCK_REASON_ADMIN, "관리자 잠금"),
    )

    # AbstractUser 기본 필드를 재정의해서 관리자 계정 생성 시 필수 입력으로 만든다.
    first_name = models.CharField("이름", max_length=150)
    last_name = models.CharField("성", max_length=150)
    email = models.EmailField("이메일")

    department = models.CharField("부서", max_length=100)
    position = models.CharField("직급", max_length=50)
    phone = models.CharField("전화번호", max_length=20)
    organization = models.CharField("소속기관", max_length=100)

    force_password_change = models.BooleanField("다음 로그인 시 비밀번호 변경", default=True)
    password_changed_at = models.DateTimeField("비밀번호 변경 일시", null=True, blank=True)
    failed_login_attempts = models.PositiveSmallIntegerField("연속 로그인 실패 횟수", default=0)
    locked_at = models.DateTimeField("잠금 일시", null=True, blank=True)
    lock_reason = models.CharField("잠금 사유", max_length=30, choices=LOCK_REASON_CHOICES, blank=True)
    current_session_key = models.CharField("현재 세션키", max_length=40, null=True, blank=True)
    notification_enabled = models.BooleanField("알림 수신", default=True)

    REQUIRED_FIELDS = [
        "last_name",
        "first_name",
        "email",
        "phone",
        "department",
        "position",
        "organization",
    ]

    class Meta:
        verbose_name = "사용자"
        verbose_name_plural = "사용자 목록"

    def korean_name(self):
        """성 + 이름 순서로 반환 (한국식)."""
        if self.last_name and self.first_name:
            return f"{self.last_name}{self.first_name}"
        return self.last_name or self.first_name or self.username

    @property
    def is_locked(self):
        return self.locked_at is not None

    @property
    def must_change_password(self):
        return self.force_password_change

    @property
    def password_expires_at(self):
        if not self.password_changed_at:
            return None
        max_age_days = getattr(settings, "ACCOUNTS_PASSWORD_MAX_AGE_DAYS", 90)
        return self.password_changed_at + timedelta(days=max_age_days)

    @property
    def is_password_expired(self):
        # 최초/관리자 초기화 상태는 사용자가 직접 변경할 기회를 주고, 만료 잠금은 적용하지 않는다.
        if self.force_password_change:
            return False
        if not self.password_changed_at:
            return True
        return timezone.now() >= self.password_expires_at

    def set_initial_password(self):
        self.set_password(getattr(settings, "ACCOUNTS_INITIAL_PASSWORD", "Workit2026!"))
        self.force_password_change = True
        self.password_changed_at = None
        self.failed_login_attempts = 0
        self.locked_at = None
        self.lock_reason = ""
        self.current_session_key = None

    def mark_password_changed(self, session_key=None):
        self.force_password_change = False
        self.password_changed_at = timezone.now()
        self.failed_login_attempts = 0
        self.locked_at = None
        self.lock_reason = ""
        if session_key is not None:
            self.current_session_key = session_key
        self.save(
            update_fields=[
                "force_password_change",
                "password_changed_at",
                "failed_login_attempts",
                "locked_at",
                "lock_reason",
                "current_session_key",
            ]
        )

    def register_login_success(self, session_key):
        self.failed_login_attempts = 0
        self.current_session_key = session_key
        self.save(update_fields=["failed_login_attempts", "current_session_key"])

    def register_login_failure(self):
        max_attempts = getattr(settings, "ACCOUNTS_MAX_FAILED_LOGIN_ATTEMPTS", 5)
        self.failed_login_attempts += 1
        update_fields = ["failed_login_attempts"]
        if self.failed_login_attempts >= max_attempts:
            self.lock(User.LOCK_REASON_FAILED_LOGIN)
            return
        self.save(update_fields=update_fields)

    def lock(self, reason=LOCK_REASON_ADMIN):
        self.locked_at = timezone.now()
        self.lock_reason = reason
        self.current_session_key = None
        self.save(update_fields=["locked_at", "lock_reason", "current_session_key"])

    def unlock(self, force_password_change=True):
        self.locked_at = None
        self.lock_reason = ""
        self.failed_login_attempts = 0
        self.current_session_key = None
        if force_password_change:
            self.force_password_change = True
        self.save(
            update_fields=[
                "locked_at",
                "lock_reason",
                "failed_login_attempts",
                "current_session_key",
                "force_password_change",
            ]
        )

    def __str__(self):
        return f"{self.korean_name()} ({self.department})"
