#!/usr/bin/env python
"""
초기 설정 스크립트: 마이그레이션 + 슈퍼유저 생성 + 샘플 데이터 생성
실행: python setup.py
"""
import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
django.setup()

from django.core.management import call_command
from accounts.models import User
from contracts.models import Contract, ContractDocument
from performance.models import Performance, Deliverable
from datetime import date, timedelta

print("🔄 마이그레이션 실행 중...")
call_command('makemigrations', '--no-input')
call_command('migrate', '--no-input')

print("👤 기본 사용자 생성 중...")
if not User.objects.filter(username='admin').exists():
    user = User.objects.create_superuser(
        username='admin',
        password='admin1234',
        email='admin@gov.kr',
        first_name='담당',
        last_name='김',
        department='정보화사업팀',
        position='주무관',
        phone='02-1234-5678',
        organization='행정안전부',
    )
    print("  ✅ admin / admin1234 생성 완료")
else:
    user = User.objects.get(username='admin')
    print("  ✅ 기존 admin 사용자 사용")

print("📁 샘플 데이터 생성 중...")
today = date.today()

# Contract 1 - in_progress
c1, _ = Contract.objects.get_or_create(
    project_name='스마트 민원 처리 시스템 고도화',
    defaults={
        'company_name': '(주)디지털라인',
        'issuing_org': 'OO광역시청',
        'budget': '4억 8,000만원',
        'contact_person': '김도윤 주무관',
        'status': 'in_progress',
        'created_by': user,
    }
)
perf1, _ = Performance.objects.get_or_create(contract=c1)
Deliverable.objects.get_or_create(
    performance=perf1, deliverable_type='kickoff',
    defaults={'due_date': today - timedelta(days=30), 'status': 'submitted', 'submitted_date': today - timedelta(days=28)}
)
Deliverable.objects.get_or_create(
    performance=perf1, deliverable_type='test_plan',
    defaults={'due_date': today + timedelta(days=20), 'status': 'pending'}
)
Deliverable.objects.get_or_create(
    performance=perf1, deliverable_type='test_result',
    defaults={'due_date': today + timedelta(days=50), 'status': 'pending'}
)
Deliverable.objects.get_or_create(
    performance=perf1, deliverable_type='final',
    defaults={'due_date': today + timedelta(days=80), 'status': 'pending'}
)

# Contract 2 - reviewing
c2, _ = Contract.objects.get_or_create(
    project_name='통합 행정 시스템 구축',
    defaults={
        'company_name': '(주)테크솔루션',
        'issuing_org': 'OO시청',
        'budget': '3억 2,000만원',
        'contact_person': '이정민 주무관',
        'status': 'reviewing',
        'created_by': user,
    }
)

# Contract 3 - completed
c3, _ = Contract.objects.get_or_create(
    project_name='공공도서관 통합 검색서비스 구축',
    defaults={
        'company_name': '(주)북스데이터',
        'issuing_org': 'OO교육청',
        'budget': '3억 5,000만원',
        'contact_person': '박성호 주무관',
        'status': 'completed',
        'created_by': user,
    }
)
perf3, _ = Performance.objects.get_or_create(contract=c3)
for dt in ['kickoff', 'test_plan', 'test_result', 'final']:
    Deliverable.objects.get_or_create(
        performance=perf3, deliverable_type=dt,
        defaults={'due_date': today - timedelta(days=60), 'status': 'submitted', 'submitted_date': today - timedelta(days=55)}
    )

print("  ✅ 샘플 데이터 3건 생성 완료")
print("\n🎉 설정 완료!")
print("=" * 40)
print("실행 명령: python manage.py runserver")
print("접속 주소: http://127.0.0.1:8000")
print("아이디: admin  /  비밀번호: admin1234")
print("=" * 40)
