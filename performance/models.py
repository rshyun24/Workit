from django.db import models
from contracts.models import Contract
from django.contrib.auth import get_user_model


class Performance(models.Model):
    contract = models.OneToOneField(Contract, on_delete=models.CASCADE, related_name='performance')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = '이행 관리'
        verbose_name_plural = '이행 관리 목록'

    def __str__(self):
        return f"{self.contract.project_name} 이행관리"

    def progress_count(self):
        valid_types = [t for t, _ in Deliverable.TYPE_CHOICES]
        return self.deliverables.filter(
            status='submitted',
            deliverable_type__in=valid_types,
        ).count()

    def total_count(self):
        return 3  # 사업수행계획서, 기술적용결과표, 사업추진결과보고서

    def next_deliverable_label(self):
        existing = {d.deliverable_type: d for d in self.deliverables.all()}
        for t, label in Deliverable.TYPE_CHOICES:
            d = existing.get(t)
            if not d or d.status == 'pending':
                return label
        return ''


class Deliverable(models.Model):
    TYPE_CHOICES = [
        ('kickoff', '사업수행계획서'),
        ('tech_apply',  '기술적용결과표'),
        ('final', '사업추진결과보고서'),
    ]
    STATUS_CHOICES = [
        ('pending', '미등록'),
        ('submitted', '제출완료'),
    ]
    TYPE_ORDER = ['kickoff', 'tech_apply', 'final']

    performance = models.ForeignKey(Performance, on_delete=models.CASCADE, related_name='deliverables')
    deliverable_type = models.CharField('산출물 유형', max_length=20, choices=TYPE_CHOICES)
    file = models.FileField('파일', upload_to='performance/deliverables/', blank=True, null=True)
    original_filename = models.CharField('원본 파일명', max_length=255, blank=True)
    due_date = models.DateField('제출 예정일', null=True, blank=True)
    submitted_date = models.DateField('실제 제출일', null=True, blank=True)
    status = models.CharField('상태', max_length=20, choices=STATUS_CHOICES, default='pending')

    class Meta:
        verbose_name = '산출물'
        verbose_name_plural = '산출물 목록'
        ordering = ['deliverable_type']

    def __str__(self):
        return f"{self.performance.contract.project_name} - {self.get_deliverable_type_display()}"

    def filename(self):
        return self.original_filename or (self.file.name.split('/')[-1] if self.file else '')

    def type_order(self):
        return self.TYPE_ORDER.index(self.deliverable_type) if self.deliverable_type in self.TYPE_ORDER else 99

User = get_user_model()

class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    message = models.CharField('메시지', max_length=255)
    url = models.CharField('이동 경로', max_length=255, blank=True, default='/performance/')
    is_read = models.BooleanField('읽음 여부', default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'notifications'
        ordering = ['-created_at']
        verbose_name = '알림'
        verbose_name_plural = '알림 목록'

    def __str__(self):
        return f"[{self.user}] {self.message}"