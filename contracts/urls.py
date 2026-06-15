from django.urls import path
from . import views

urlpatterns = [
    path('', views.contract_list, name='contract_list'),
    path('create/', views.contract_create, name='contract_create'),
    path('<int:pk>/detail/', views.contract_detail_api, name='contract_detail_api'),
    path('<int:pk>/update-file/', views.contract_update_file, name='contract_update_file'),
    path('<int:pk>/update-info/', views.contract_update_info, name='contract_update_info'),
    path('document/<int:doc_id>/view/', views.document_view, name='document_view'),
    path('document/<int:doc_id>/analyze/', views.document_analyze, name='document_analyze'),
    path('document/<int:doc_id>/ai-analyze/', views.document_ai_analyze, name='document_ai_analyze'),
    path('document/<int:doc_id>/complete/', views.document_complete_review, name='document_complete_review'),
    path('document/<int:doc_id>/page/<int:page>/', views.document_page_image, name='document_page_image'),
    path('document/<int:doc_id>/pages/', views.document_page_count, name='document_page_count'),
    path('document/<int:doc_id>/ai-analyze/', views.document_ai_analyze, name='document_ai_analyze'),
    path('task/<str:task_id>/status/', views.document_ai_status, name='document_ai_status'),
    path('document/<int:doc_id>/export-pdf/', views.document_export_pdf, name='document_export_pdf'),
]