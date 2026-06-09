from django.contrib import admin
from .models import Performance, Deliverable

@admin.register(Performance)
class PerformanceAdmin(admin.ModelAdmin):
    list_display = ['contract', 'created_at']

admin.site.register(Deliverable)
