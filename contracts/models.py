from django.db import models
from django.conf import settings


class Contract(models.Model):
    STATUS_CHOICES = [
        ('reviewing', '계약 검토중'),
        ('in_progress', '이행중'),
        ('completed', '완료'),
    ]

    project_name = models.CharField('프로젝트명', max_length=200)
    company_name = models.CharField('업체명', max_length=200)
    issuing_org = models.CharField('발주기관', max_length=200, blank=True)
    budget = models.CharField('사업 예산', max_length=100, blank=True)
    contact_person = models.CharField('계약 담당자', max_length=100, blank=True)
    status = models.CharField('상태', max_length=20, choices=STATUS_CHOICES, default='reviewing')

    requirements_doc = models.FileField('요구사항정의서', upload_to='contracts/requirements/', blank=True, null=True)
    rfp_doc = models.FileField('RFP(제안요청서)', upload_to='contracts/rfp/', blank=True, null=True)
    contract_doc = models.FileField('계약서', upload_to='contracts/contract/', blank=True, null=True)

    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='contracts')
    created_at = models.DateTimeField('등록일', auto_now_add=True)
    updated_at = models.DateTimeField('수정일', auto_now=True)

    contract_start = models.DateField('계약 시작일', null=True, blank=True)
    contract_end = models.DateField('계약 종료일', null=True, blank=True)

    class Meta:
        verbose_name = '계약'
        verbose_name_plural = '계약 목록'
        ordering = ['-created_at']

    def __str__(self):
        return self.project_name

    def get_status_display_color(self):
        colors = {
            'reviewing': 'status-reviewing',
            'in_progress': 'status-progress',
            'completed': 'status-completed',
        }
        return colors.get(self.status, '')


class ContractDocument(models.Model):
    DOC_TYPES = [
        ('requirements', '요구사항정의서'),
        ('rfp', 'RFP (제안요청서)'),
        ('contract', '계약서'),
    ]
    REVIEW_STATUS = [
        ('pending', '미검토'),
        ('reviewed', '검토완료'),
    ]

    contract = models.ForeignKey(Contract, on_delete=models.CASCADE, related_name='documents')
    doc_type = models.CharField('문서 유형', max_length=30, choices=DOC_TYPES)
    file = models.FileField('파일', upload_to='contracts/docs/')
    original_filename = models.CharField('원본 파일명', max_length=255, blank=True)
    review_status = models.CharField('검토 상태', max_length=20, choices=REVIEW_STATUS, default='pending')
    uploaded_at = models.DateTimeField('업로드일', auto_now_add=True)

    class Meta:
        verbose_name = '계약 문서'
        verbose_name_plural = '계약 문서 목록'

    def __str__(self):
        return f"{self.contract.project_name} - {self.get_doc_type_display()}"

    def filename(self):
        return self.original_filename or self.file.name.split('/')[-1]


class AIReviewResult(models.Model):
    STATUS_CHOICES = [
        ('idle',       '대기'),
        ('processing', '분석중'),
        ('done',       '완료'),
        ('failed',     '실패'),
    ]

    document = models.OneToOneField(ContractDocument, on_delete=models.CASCADE, related_name='review_result')
    blanks = models.JSONField('빈칸/미기재', default=list)
    typos = models.JSONField('오탈자', default=list)
    legal_issues = models.JSONField('법률 관련', default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    # 분석 화면을 나갔다가 돌아와도 진행 상태를 보여주기 위한 필드.
    # task_id는 브라우저 JS 변수에만 있으면 새로고침 시 사라지므로 DB에 같이 저장한다.
    status = models.CharField('분석 상태', max_length=20, choices=STATUS_CHOICES, default='idle')
    task_id = models.CharField('Celery task id', max_length=255, blank=True, default='')
    started_at = models.DateTimeField('분석 시작 시각', null=True, blank=True)

    class Meta:
        verbose_name = 'AI 검토 결과'

class RFPParsedData(models.Model):
    """
    RFP 문서에서 파싱한 정형화 데이터.

    - 원본 파일은 S3(ContractDocument.file) 에 저장
    - 파싱 결과 JSON은 이 모델(RDS)에 저장
    - 파싱 시점: 이행관리 이관(document_complete_review) 직후 비동기
    - parse_status 흐름: pending → processing → done | failed
    """

    document = models.OneToOneField(
        'ContractDocument',
        on_delete=models.CASCADE,
        related_name='rfp_parsed',
        # doc_type이 'rfp'인 문서에만 연결됨 (DB 제약은 없음, 앱 로직으로 보장)
    )
    # 노션 RFP 코드 체계(RFP-01-01 ~ RFP-04-04-11) 기반 정형화 JSON
    parsed_json = models.JSONField(default=dict, blank=True)

    # 소제목 매핑 QA 검수 리포트 (LLM/qa_agent.review_section_mapping 결과)
    # — 사업수행계획서 AI 분석 화면의 "2단계 · RFP 매핑" 탭에서 사용
    qa_report = models.JSONField('QA 검수 리포트', default=dict, blank=True)

    parse_status = models.CharField(
        max_length=20,
        choices=[
            ('pending',    '대기'),
            ('processing', '처리중'),
            ('done',       '완료'),
            ('failed',     '실패'),
        ],
        default='pending',
    )
    parsed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)

    class Meta:
        verbose_name = 'RFP 파싱 결과'
        verbose_name_plural = 'RFP 파싱 결과 목록'

    def __str__(self):
        return f'RFP 파싱 — {self.document} [{self.parse_status}]'