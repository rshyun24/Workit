from datetime import timedelta

from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone


class LoginHistory(models.Model):
    """로그인 시도 접속기록.

    User.current_session_key는 "현재" 세션 1개만 덮어쓰기 때문에 감사 목적에는
    부족하다. 이 모델은 성공/실패 모든 로그인 시도를 누적 기록하여
    관리자가 계정별 접속 이력을 추적할 수 있게 한다.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="login_histories"
    )
    ip_address = models.GenericIPAddressField("접속 IP", null=True, blank=True)
    success = models.BooleanField("성공 여부")
    user_agent = models.CharField("User-Agent", max_length=255, blank=True)
    created_at = models.DateTimeField("접속 일시", auto_now_add=True)

    class Meta:
        verbose_name = "접속기록"
        verbose_name_plural = "접속기록 목록"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user} - {'성공' if self.success else '실패'} ({self.created_at:%Y-%m-%d %H:%M})"


class Organization(models.Model):
    """소속부서 단위.

    하나의 Workit 시스템은 하나의 소속기관(예: ○○교육청)에 설치되어 운영되며,
    그 기관 산하의 부서들을 시스템 관리자가 이 모델로 직접 생성·관리한다.
    사용자(User)와 문서(계약/이행)의 접근 범위는 이 부서 단위로 격리된다.
    서로 다른 부서에 속한 사용자는 서로의 문서를 볼 수 없다.
    """

    name = models.CharField("부서명", max_length=100, unique=True)
    code = models.CharField("부서코드", max_length=20, unique=True)
    is_active = models.BooleanField("사용 여부", default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField("수정 일시", auto_now=True)

    class Meta:
        verbose_name = "소속부서"
        verbose_name_plural = "소속부서 목록"
        ordering = ["name"]

    def __str__(self):
        return self.name


class User(AbstractUser):
    """Workit 사용자 계정.

    운영 정책:
    - 관리자(is_superuser)가 초기 비밀번호로 계정을 생성한다.
    - 최초 로그인 또는 관리자 잠금해제/초기화 후 비밀번호 변경을 강제한다.
    - 5회 연속 로그인 실패 또는 90일 비밀번호 만료 시 계정을 잠근다.
    - 동일 계정 동시접속 제어를 위해 현재 세션키를 저장한다.
    - 사용자는 자신이 속한 부서(Organization)의 문서만 조회할 수 있다.
    - "관리자" 권한은 별도 role 필드 없이 Django 기본 is_superuser를 그대로 사용한다.
      (is_staff/groups는 Django 자체 admin 사이트 권한 체계이며, 이 서비스에서는 사용하지 않는다.)
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

    # department(CharField)는 Organization(소속부서) FK 도입 이후 더 이상 사용하지 않는다.
    # 컬럼 삭제는 별도 마이그레이션으로 추후 처리 (현재는 폼/템플릿에서만 미노출).
    department = models.CharField("소속부서(미사용, deprecated)", max_length=100, blank=True)
    position = models.CharField("직급", max_length=50)
    phone = models.CharField("전화번호", max_length=20)

    # 시스템 전체는 하나의 소속기관(예: ○○교육청)에 설치되어 운영된다.
    # 그 기관 산하 부서는 관리자가 이 FK를 통해 직접 생성·관리한다.
    # 관리자(is_superuser) 계정은 특정 부서에 속하지 않아도 되므로 null 허용.
    organization = models.ForeignKey(
        Organization,
        on_delete=models.PROTECT,
        related_name="users",
        verbose_name="소속부서",
        null=True,
        blank=True,
    )

    force_password_change = models.BooleanField("다음 로그인 시 비밀번호 변경", default=True)
    password_changed_at = models.DateTimeField("비밀번호 변경 일시", null=True, blank=True)
    failed_login_attempts = models.PositiveSmallIntegerField("연속 로그인 실패 횟수", default=0)
    locked_at = models.DateTimeField("잠금 일시", null=True, blank=True)
    lock_reason = models.CharField("잠금 사유", max_length=30, choices=LOCK_REASON_CHOICES, blank=True)
    current_session_key = models.CharField("현재 세션키", max_length=40, null=True, blank=True)
    notification_enabled = models.BooleanField("알림 수신", default=True)
    updated_at = models.DateTimeField("수정 일시", auto_now=True)

    REQUIRED_FIELDS = [
        "last_name",
        "first_name",
        "email",
        "phone",
        "position",
        # "department"는 deprecated(미사용)되어 필수 입력에서 제외한다.
        # "organization"은 관리자(is_superuser) 계정이 부서 미배정일 수 있어
        # createsuperuser 필수 입력에서 제외한다.
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
    def is_system_admin(self):
        """관리자 여부. 별도 role 필드 없이 Django 기본 is_superuser를 그대로 사용한다."""
        return self.is_superuser

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
        return f"{self.korean_name()} ({self.organization})"