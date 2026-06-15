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

        return self.get_response(request)
