from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.sessions.models import Session
from django.http import JsonResponse
from django.shortcuts import redirect, render

from contracts.models import Contract
from performance.models import Performance

from .forms import WorkitPasswordChangeForm
from .models import User


def _delete_session(session_key):
    if session_key:
        Session.objects.filter(session_key=session_key).delete()


def _handle_single_session_policy(request, user, suppress_block_message=False):
    """동일계정 동시접속 정책 처리.

    기본 정책은 기존 세션 종료(KILL_OLD)이다.
    settings.ACCOUNTS_SINGLE_SESSION_POLICY = "BLOCK_NEW" 로 바꾸면 신규 접속 차단 방식으로 전환된다.
    """

    existing_session_key = user.current_session_key
    if not existing_session_key or existing_session_key == request.session.session_key:
        return True

    existing_session_exists = Session.objects.filter(session_key=existing_session_key).exists()
    if not existing_session_exists:
        user.current_session_key = None
        user.save(update_fields=["current_session_key"])
        return True

    policy = getattr(settings, "ACCOUNTS_SINGLE_SESSION_POLICY", "KILL_OLD")
    if policy == "BLOCK_NEW":
        if not suppress_block_message:
            messages.error(request, "이미 동일 계정으로 접속 중입니다. 기존 세션을 종료한 뒤 다시 로그인하세요.")
        return False

    _delete_session(existing_session_key)
    return True


def login_view(request):
    if request.user.is_authenticated:
        if request.user.must_change_password:
            return redirect('change_password')
        return redirect('home')

    if request.method == 'POST':
        username = (request.POST.get('username') or '').strip()
        password = request.POST.get('password') or ''
        candidate = User.objects.filter(username=username).first()

        if candidate and candidate.is_locked:
            messages.error(request, '잠긴 계정입니다. 관리자에게 잠금해제를 요청하세요.')
            return render(request, 'accounts/login.html', {'username': username})

        if candidate and not candidate.is_active:
            messages.error(request, '비활성화된 계정입니다. 관리자에게 문의하세요.')
            return render(request, 'accounts/login.html', {'username': username})

        user = authenticate(request, username=username, password=password)

        if not user:
            if candidate and candidate.is_active and not candidate.is_locked:
                candidate.register_login_failure()
                max_attempts = getattr(settings, 'ACCOUNTS_MAX_FAILED_LOGIN_ATTEMPTS', 5)
                remaining = max_attempts - candidate.failed_login_attempts
                if candidate.is_locked:
                    messages.error(request, '로그인 5회 연속 실패로 계정이 잠겼습니다. 관리자에게 잠금해제를 요청하세요.')
                else:
                    messages.error(request, f'아이디 또는 비밀번호가 올바르지 않습니다. 남은 시도 횟수: {remaining}회')
            else:
                messages.error(request, '아이디 또는 비밀번호가 올바르지 않습니다.')
            return render(request, 'accounts/login.html', {'username': username})

        if user.is_password_expired:
            user.lock(User.LOCK_REASON_PASSWORD_EXPIRED)
            messages.error(request, '비밀번호 사용기간 90일이 만료되어 계정이 잠겼습니다. 관리자에게 잠금해제를 요청하세요.')
            return render(request, 'accounts/login.html', {'username': username})

        suppress_single_session_message = request.COOKIES.get('workit_password_changed_recently') == '1'
        if not _handle_single_session_policy(request, user, suppress_block_message=suppress_single_session_message):
            response = render(request, 'accounts/login.html', {'username': username})
            if suppress_single_session_message:
                response.delete_cookie('workit_password_changed_recently')
            return response

        login(request, user)
        if not request.session.session_key:
            request.session.save()
        user.register_login_success(request.session.session_key)

        if user.must_change_password:
            messages.info(request, '최초 로그인 또는 관리자 초기화 계정입니다. 비밀번호를 먼저 변경하세요.')
            return redirect('change_password')

        return redirect('home')

    return render(request, 'accounts/login.html')


def logout_view(request):
    if request.user.is_authenticated and request.user.current_session_key == request.session.session_key:
        request.user.current_session_key = None
        request.user.save(update_fields=['current_session_key'])
    logout(request)
    return redirect('login')


@login_required
def change_password_view(request):
    if request.method == 'POST':
        form = WorkitPasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            # update_session_auth_hash()가 세션 키를 갱신할 수 있으므로,
            # 갱신 이후의 세션 키를 사용자 current_session_key에 저장해야
            # 비밀번호 변경 직후 동일계정 동시접속으로 오인되지 않는다.
            update_session_auth_hash(request, user)
            if not request.session.session_key:
                request.session.save()
            user.mark_password_changed(request.session.session_key)
            messages.success(request, '비밀번호가 변경되었습니다. 마이페이지에서 계정 정보를 확인하세요.')
            response = redirect('mypage')
            response.set_cookie(
                'workit_password_changed_recently',
                '1',
                max_age=120,
                httponly=True,
                samesite='Lax',
            )
            return response
        messages.error(request, '비밀번호 변경 내용을 확인하세요.')
    else:
        form = WorkitPasswordChangeForm(request.user)

    return render(request, 'accounts/change_password.html', {'form': form})


@login_required
def home_view(request):
    contracts = Contract.objects.filter(created_by=request.user).order_by('-created_at')
    performances = Performance.objects.filter(contract__created_by=request.user).order_by('-created_at')

    total = contracts.count()
    in_review = contracts.filter(status='reviewing').count()
    in_progress = contracts.filter(status='in_progress').count()
    completed = contracts.filter(status='completed').count()

    context = {
        'total': total,
        'in_review': in_review,
        'in_progress': in_progress,
        'completed': completed,
        'recent_contracts': contracts[:5],
        'recent_performances': performances[:5],
    }
    return render(request, 'home.html', context)


@login_required
def mypage_view(request):
    return render(request, 'mypage/mypage.html', {'user': request.user})


@login_required
def mypage_update(request):
    if request.method == 'POST':
        required_fields = {
            'last_name': '성',
            'first_name': '이름',
            'email': '이메일',
            'phone': '전화번호',
            'department': '부서',
            'position': '직급',
            'organization': '소속기관',
        }
        missing = [label for field, label in required_fields.items() if not (request.POST.get(field) or '').strip()]
        if missing:
            return JsonResponse({'status': 'error', 'message': f"필수 항목을 입력하세요: {', '.join(missing)}"}, status=400)

        user = request.user
        user.first_name = request.POST.get('first_name').strip()
        user.last_name = request.POST.get('last_name').strip()
        user.email = request.POST.get('email').strip()
        user.phone = request.POST.get('phone').strip()
        user.department = request.POST.get('department').strip()
        user.position = request.POST.get('position').strip()
        user.organization = request.POST.get('organization').strip()
        user.save(update_fields=['first_name', 'last_name', 'email', 'phone', 'department', 'position', 'organization'])
        return JsonResponse({'status': 'ok', 'message': '정보가 수정되었습니다.'})

    return JsonResponse({'status': 'error', 'message': '잘못된 요청입니다.'}, status=400)
