from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from .models import User
from contracts.models import Contract
from performance.models import Performance


def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, '아이디 또는 비밀번호가 올바르지 않습니다.')
    return render(request, 'accounts/login.html')


def logout_view(request):
    logout(request)
    return redirect('login')


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
        user = request.user
        user.first_name = request.POST.get('first_name', user.first_name)
        user.last_name = request.POST.get('last_name', user.last_name)
        user.email = request.POST.get('email', user.email)
        user.phone = request.POST.get('phone', user.phone)
        user.department = request.POST.get('department', user.department)
        user.position = request.POST.get('position', user.position)
        user.organization = request.POST.get('organization', user.organization)
        user.save()
        return JsonResponse({'status': 'ok', 'message': '정보가 수정되었습니다.'})
    return JsonResponse({'status': 'error'}, status=400)
