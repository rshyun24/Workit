from django.contrib import admin
from .models import Contract, ContractDocument, AIReviewResult

@admin.register(Contract)
class ContractAdmin(admin.ModelAdmin):
    list_display = ['project_name', 'company_name', 'status', 'created_by', 'created_at']
    list_filter = ['status']

admin.site.register(ContractDocument)
admin.site.register(AIReviewResult)
