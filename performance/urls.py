from django.urls import path
from . import views

urlpatterns = [
    path('', views.performance_list, name='performance_list'),
    path('<int:pk>/detail/', views.performance_detail_api, name='performance_detail_api'),
    path('<int:perf_id>/deliverable/upload/', views.deliverable_upload, name='deliverable_upload'),
    path('<int:perf_id>/deliverable/due-date/', views.deliverable_update_due_date, name='deliverable_update_due_date'),

    # 산출물 뷰어 / AI 분석
    path('contract-doc/<int:doc_id>/view/', views.contract_doc_view, name='perf_contract_doc_view'),
    path('deliverable/<int:del_id>/view/', views.deliverable_view, name='deliverable_view'),
    path('deliverable/<int:del_id>/analyze/', views.deliverable_analyze, name='deliverable_analyze'),
    path('deliverable/<int:del_id>/pages/', views.deliverable_page_count, name='deliverable_page_count'),
    path('deliverable/<int:del_id>/page/<int:page>/', views.deliverable_page_image, name='deliverable_page_image'),
    path('deliverable/<int:del_id>/ai-analyze/', views.deliverable_ai_analyze, name='deliverable_ai_analyze'),
    path('deliverable/<int:del_id>/export-pdf/', views.deliverable_export_pdf, name='deliverable_export_pdf'),
]