from functools import wraps
from datetime import timedelta

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.sessions.models import Session
from django.core.paginator import Paginator
from django.db.models import ProtectedError
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from contracts.models import Contract
from performance.models import Performance

from .forms import (
    AccountEditForm,
    AdminUserCreationForm,
    OrganizationCreateForm,
    OrganizationEditForm,
    WorkitPasswordChangeForm,
)
from .models import LoginHistory, Organization, User

from django.views.decorators.http import require_POST


def _delete_session(session_key):
    if session_key:
        Session.objects.filter(session_key=session_key).delete()


def _get_client_ip(request):
    """프록시(X-Forwarded-For) 환경을 고려한 클라이언트 IP 추출.

    nginx 등 리버스 프록시 뒤에 있다면 REMOTE_ADDR은 프록시 자신의 IP가 되므로
    X-Forwarded-For 헤더의 첫 값을 우선 사용한다.
    """
    xff = request.META.get('HTTP_X_FORWARDED_FOR')
    if xff:
        return xff.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR')


def _record_login_history(request, user, success):
    LoginHistory.objects.create(
        user=user,
        ip_address=_get_client_ip(request),
        success=success,
        user_agent=(request.META.get('HTTP_USER_AGENT') or '')[:255],
    )


def _check_single_session_conflict(request, user):
    """동일 계정이 다른(살아있는) 세션에서 로그인 중인지 확인."""
    existing_session_key = user.current_session_key
    if not existing_session_key or existing_session_key == request.session.session_key:
        return False

    if not Session.objects.filter(session_key=existing_session_key).exists():
        # 이미 끊긴 세션이면 정리하고 충돌 아님으로 처리
        user.current_session_key = None
        user.save(update_fields=["current_session_key"])
        return False

    return True


def _finalize_login(request, user):
    """기존 세션 종료(필요 시) 후 실제 로그인 처리."""
    existing_session_key = user.current_session_key
    if existing_session_key and existing_session_key != request.session.session_key:
        _delete_session(existing_session_key)

    login(request, user)
    if not request.session.session_key:
        request.session.save()
    user.register_login_success(request.session.session_key)
    _record_login_history(request, user, success=True)

    if user.must_change_password:
        messages.info(request, '최초 로그인 또는 관리자 초기화 계정입니다. 비밀번호를 먼저 변경하세요.')
        return redirect('change_password')

    if user.is_system_admin:
        return redirect('account_list')

    return redirect('home')


def login_view(request):
    if request.user.is_authenticated:
        if request.user.must_change_password:
            return redirect('change_password')
        if request.user.is_system_admin:
            return redirect('account_list')
        return redirect('home')

    if request.method == 'POST':
        # ── 확인창에서 "계속" / "취소"를 누른 2차 제출 처리 ──
        if request.POST.get('confirm_force_login') == '1':
            pending_user_id = request.session.pop('pending_force_login_user_id', None)
            user = User.objects.filter(pk=pending_user_id).first() if pending_user_id else None
            if not user:
                messages.error(request, '로그인 확인 시간이 초과되었습니다. 다시 로그인해 주세요.')
                return redirect('login')
            return _finalize_login(request, user)

        if request.POST.get('cancel_force_login') == '1':
            request.session.pop('pending_force_login_user_id', None)
            return redirect('login')

        # ── 일반적인 1차 로그인 제출 ──
        request.session.pop('pending_force_login_user_id', None)  # 이전에 남아있던 대기 상태 정리

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
                _record_login_history(request, candidate, success=False)
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
        policy = getattr(settings, 'ACCOUNTS_SINGLE_SESSION_POLICY', 'KILL_OLD')
        conflict = _check_single_session_conflict(request, user)

        if conflict and policy == 'BLOCK_NEW':
            if not suppress_single_session_message:
                messages.error(request, '이미 동일 계정으로 접속 중입니다. 기존 세션을 종료한 뒤 다시 로그인하세요.')
            response = render(request, 'accounts/login.html', {'username': username})
            if suppress_single_session_message:
                response.delete_cookie('workit_password_changed_recently')
            return response

        if conflict:  # policy == KILL_OLD → 확인창 띄우기
            request.session['pending_force_login_user_id'] = user.pk
            request.session.save()
            return render(request, 'accounts/login.html', {
                'username': username,
                'need_session_confirm': True,
            })

        return _finalize_login(request, user)

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

    # "계약 검토 목록"은 계약관리 화면(검토중 상태만 표시)과 동일한 범위를 보여줘야 한다.
    # 이행으로 넘어간 계약(in_progress/completed)이 섞여 보이면 안 되므로 reviewing만 필터링.
    recent_contracts = contracts.filter(status='reviewing')[:5]

    context = {
        'total': total,
        'in_review': in_review,
        'in_progress': in_progress,
        'completed': completed,
        'recent_contracts': recent_contracts,
        'recent_performances': performances[:5],
    }
    return render(request, 'home.html', context)


@login_required
def mypage_view(request):
    institution_name = getattr(settings, 'WORKIT_INSTITUTION_NAME', '소속기관 미설정')
    return render(request, 'mypage/mypage.html', {'user': request.user, 'institution_name': institution_name})


@login_required
def mypage_update(request):
    if request.method == 'POST':
        required_fields = {
            'last_name': '성',
            'first_name': '이름',
            'email': '이메일',
            'phone': '전화번호',
            'position': '직급',
            # 'department'는 deprecated(미사용) — 부서는 organization FK로 관리자가 배정.
        }
        missing = [label for field, label in required_fields.items() if not (request.POST.get(field) or '').strip()]
        if missing:
            return JsonResponse({'status': 'error', 'message': f"필수 항목을 입력하세요: {', '.join(missing)}"}, status=400)

        user = request.user
        user.first_name = request.POST.get('first_name').strip()
        user.last_name = request.POST.get('last_name').strip()
        user.email = request.POST.get('email').strip()
        user.phone = request.POST.get('phone').strip()
        user.position = request.POST.get('position').strip()
        user.save(update_fields=['first_name', 'last_name', 'email', 'phone', 'position'])
        return JsonResponse({'status': 'ok', 'message': '정보가 수정되었습니다.'})

    return JsonResponse({'status': 'error', 'message': '잘못된 요청입니다.'}, status=400)


@login_required
def help_page(request):
    if request.user.is_system_admin:
        return render(request, 'help/help_admin.html')
    return render(request, 'help/help.html')


@login_required
@require_POST
def toggle_notification(request):
    user = request.user
    user.notification_enabled = not user.notification_enabled
    user.save(update_fields=['notification_enabled'])
    return JsonResponse({
        'status': 'ok',
        'notification_enabled': user.notification_enabled,
    })


# ──────────────────────────────────────────────
# 관리자(is_superuser) 전용 계정/부서 관리
# ──────────────────────────────────────────────

def system_admin_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated or not request.user.is_system_admin:
            messages.error(request, '접근 권한이 없습니다.')
            return redirect('home')
        return view_func(request, *args, **kwargs)
    return wrapper


@system_admin_required
def account_list_view(request):
    accounts = (
        User.objects.select_related('organization')
        .exclude(is_superuser=True)
        .order_by('-is_active', '-date_joined')
    )

    # current_session_key가 있어도 실제로는 만료된 세션일 수 있으므로,
    # 만료되지 않은(expire_date > now) 세션 키만 "로그인 중"으로 판단한다.
    session_keys = [a.current_session_key for a in accounts if a.current_session_key]
    online_keys = set(
        Session.objects.filter(
            session_key__in=session_keys, expire_date__gt=timezone.now()
        ).values_list('session_key', flat=True)
    )
    active_count = 0
    for account in accounts:
        account.is_online = bool(account.current_session_key) and account.current_session_key in online_keys
        if account.is_active:
            active_count += 1

    total_count = len(accounts)

    organizations = Organization.objects.filter(is_active=True)
    return render(request, 'accounts/account_list.html', {
        'accounts': accounts,
        'organizations': organizations,
        'total_count': total_count,
        'active_count': active_count,
        'inactive_count': total_count - active_count,
    })


@system_admin_required
def account_create_view(request):
    if request.method == 'POST':
        form = AdminUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(
                request,
                f'{user.korean_name()}님의 계정이 생성되었습니다. 초기 비밀번호로 로그인 후 변경이 필요합니다.',
            )
            return redirect('account_list')
        messages.error(request, '입력 내용을 확인하세요.')
    else:
        form = AdminUserCreationForm()

    return render(request, 'accounts/account_create.html', {'form': form})


@system_admin_required
@require_POST
def account_lock_toggle_view(request, user_id):
    target = get_object_or_404(User, pk=user_id)
    if target.is_system_admin:
        messages.error(request, '관리자 계정은 잠금 처리할 수 없습니다.')
        return redirect('account_list')

    if target.is_locked:
        target.unlock()
        messages.success(request, f'{target.korean_name()}님의 계정 잠금이 해제되었습니다.')
    else:
        target.lock(User.LOCK_REASON_ADMIN)
        messages.success(request, f'{target.korean_name()}님의 계정을 잠금 처리했습니다.')
    return redirect('account_list')


@system_admin_required
@require_POST
def account_edit_view(request, user_id):
    target = get_object_or_404(User, pk=user_id)
    if target.is_system_admin and target.pk != request.user.pk:
        messages.error(request, '다른 관리자 계정은 이 화면에서 수정할 수 없습니다.')
        return redirect('account_list')

    form = AccountEditForm(request.POST, instance=target)
    if form.is_valid():
        form.save()
        messages.success(request, f'{target.korean_name()}님의 정보가 수정되었습니다.')
    else:
        error_text = ' / '.join(f"{field}: {', '.join(errs)}" for field, errs in form.errors.items())
        messages.error(request, f'입력 내용을 확인하세요. ({error_text})')
    return redirect('account_list')


@system_admin_required
def organization_list_view(request):
    if request.method == 'POST':
        form = OrganizationCreateForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, '부서가 추가되었습니다.')
            return redirect('organization_list')
        messages.error(request, '입력 내용을 확인하세요.')
    else:
        form = OrganizationCreateForm()

    organizations = Organization.objects.all()
    org_total = len(organizations)
    org_active = sum(1 for o in organizations if o.is_active)
    return render(request, 'accounts/organization_list.html', {
        'organizations': organizations,
        'form': form,
        'org_total': org_total,
        'org_active': org_active,
        'org_inactive': org_total - org_active,
    })


@system_admin_required
@require_POST
def organization_edit_view(request, org_id):
    organization = get_object_or_404(Organization, pk=org_id)

    form = OrganizationEditForm(request.POST, instance=organization)
    if form.is_valid():
        form.save()
        messages.success(request, f'{organization.name} 부서 정보가 수정되었습니다.')
    else:
        error_text = ' / '.join(f"{field}: {', '.join(errs)}" for field, errs in form.errors.items())
        messages.error(request, f'입력 내용을 확인하세요. ({error_text})')
    return redirect('organization_list')


@system_admin_required
@require_POST
def account_reset_password_view(request, user_id):
    target = get_object_or_404(User, pk=user_id)
    if target.is_system_admin:
        messages.error(request, '관리자 계정은 이 화면에서 초기화할 수 없습니다.')
        return redirect('account_list')

    target.set_initial_password()
    target.save()
    messages.success(request, f'{target.korean_name()}님의 비밀번호가 초기 비밀번호로 재설정되었습니다.')
    return redirect('account_list')


@system_admin_required
@require_POST
def account_toggle_active_view(request, user_id):
    target = get_object_or_404(User, pk=user_id)
    if target.is_system_admin:
        messages.error(request, '관리자 계정은 비활성화할 수 없습니다.')
        return redirect('account_list')

    target.is_active = not target.is_active
    update_fields = ['is_active']

    if not target.is_active:
        # 비활성화 시 현재 세션도 함께 종료한다.
        _delete_session(target.current_session_key)
        target.current_session_key = None
        update_fields.append('current_session_key')

    target.save(update_fields=update_fields)

    if target.is_active:
        messages.success(request, f'{target.korean_name()}님의 계정이 다시 활성화되었습니다.')
    else:
        messages.success(request, f'{target.korean_name()}님의 계정이 비활성화되었습니다.')
    return redirect('account_list')


@system_admin_required
@require_POST
def account_force_logout_view(request, user_id):
    target = get_object_or_404(User, pk=user_id)
    if not target.current_session_key:
        messages.info(request, f'{target.korean_name()}님은 현재 로그인 중인 세션이 없습니다.')
        return redirect('account_list')

    _delete_session(target.current_session_key)
    target.current_session_key = None
    target.save(update_fields=['current_session_key'])
    messages.success(request, f'{target.korean_name()}님을 강제 로그아웃 처리했습니다.')
    return redirect('account_list')


@system_admin_required
def login_history_view(request):
    one_year_ago = timezone.now() - timedelta(days=365)
    histories = (
        LoginHistory.objects.select_related('user', 'user__organization')
        .filter(created_at__gte=one_year_ago)
        .order_by('-created_at')
    )
    paginator = Paginator(histories, 50)
    page_obj = paginator.get_page(request.GET.get('page'))
    return render(request, 'accounts/login_history.html', {'page_obj': page_obj})


@system_admin_required
@require_POST
def organization_delete_view(request, org_id):
    organization = get_object_or_404(Organization, pk=org_id)
    try:
        name = organization.name
        organization.delete()
        messages.success(request, f'{name} 부서가 삭제되었습니다.')
    except ProtectedError:
        messages.error(
            request,
            f'{organization.name} 부서에 소속된 사용자가 있어 삭제할 수 없습니다. '
            f'먼저 해당 사용자들을 다른 부서로 옮기거나 계정을 비활성화하세요.',
        )
    return redirect('organization_list')