from django.contrib import messages
from django.contrib.auth import logout
from django.shortcuts import redirect
from django.urls import NoReverseMatch, reverse


def _safe_message(request, level, text):
    """MessageMiddleware가 아직 없거나 제거된 환경에서도 예외 없이 넘어간다."""
    if not hasattr(request, "_messages"):
        return
    getattr(messages, level)(request, text)


def _safe_reverse(name, fallback):
    try:
        return reverse(name)
    except NoReverseMatch:
        return fallback


# 시스템관리자(is_system_admin)는 대시보드/계약관리/이행관리 등 일반 업무 화면에
# 접근할 수 없다. URL을 직접 입력해도 막아야 하므로 화이트리스트 방식으로 제한한다.
# (블랙리스트로 "이건 막자"를 하나씩 추가하면 새 view가 생길 때 빠뜨리기 쉽다.)
SYSTEM_ADMIN_ALLOWED_PREFIXES = (
    "/manage/",       # 계정관리, 부서관리, 접속기록
    "/mypage/",       # 마이페이지 조회/수정
    "/help/",         # 도움말
    "/notification/", # 알림 토글
    "/static/",       # 정적 파일(DEBUG 모드에서 미들웨어를 통과함)
)


def _is_system_admin_allowed_path(path, login_path, logout_path, change_password_path):
    if path in {login_path, logout_path, change_password_path}:
        return True
    return any(path.startswith(prefix) for prefix in SYSTEM_ADMIN_ALLOWED_PREFIXES)


class AccountSecurityMiddleware:
    """계정 잠금, 비밀번호 변경 강제, 동일계정 세션 제어를 담당한다.

    주의: settings.MIDDLEWARE에서 MessageMiddleware 뒤에 위치해야 한다.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        user = getattr(request, "user", None)

        if user and user.is_authenticated:
            path = request.path_info or request.path or ""

            login_path = _safe_reverse("login", "/login/")
            logout_path = _safe_reverse("logout", "/logout/")
            change_password_path = _safe_reverse("change_password", "/password/change/")

            is_admin_path = path.startswith("/admin/")
            is_allowed_path = path in {login_path, logout_path, change_password_path}

            if not is_admin_path:
                if user.is_locked:
                    logout(request)
                    _safe_message(request, "error", "잠긴 계정입니다. 관리자에게 잠금해제를 요청하세요.")
                    return redirect("login")

                if user.current_session_key and user.current_session_key != request.session.session_key:
                    logout(request)
                    _safe_message(request, "warning", "동일 계정의 다른 접속으로 현재 세션이 종료되었습니다.")
                    return redirect("login")

                if user.is_password_expired:
                    user.lock(user.LOCK_REASON_PASSWORD_EXPIRED)
                    logout(request)
                    _safe_message(request, "error", "비밀번호 사용기간 90일이 만료되어 계정이 잠겼습니다. 관리자에게 잠금해제를 요청하세요.")
                    return redirect("login")

                if user.must_change_password and not is_allowed_path:
                    _safe_message(request, "info", "계속 사용하려면 먼저 비밀번호를 변경해야 합니다.")
                    return redirect("change_password")

                if getattr(user, "is_system_admin", False):
                    if not _is_system_admin_allowed_path(path, login_path, logout_path, change_password_path):
                        return redirect("account_list")

        return self.get_response(request)