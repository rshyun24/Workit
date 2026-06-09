from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    department = models.CharField('부서', max_length=100, blank=True)
    position = models.CharField('직급', max_length=50, blank=True)
    phone = models.CharField('전화번호', max_length=20, blank=True)
    organization = models.CharField('소속기관', max_length=100, blank=True)

    class Meta:
        verbose_name = '사용자'
        verbose_name_plural = '사용자 목록'

    def korean_name(self):
        """성 + 이름 순서로 반환 (한국식)"""
        if self.last_name and self.first_name:
            return f"{self.last_name}{self.first_name}"
        return self.last_name or self.first_name or self.username

    def __str__(self):
        return f"{self.korean_name()} ({self.department})"
