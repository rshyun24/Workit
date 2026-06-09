from django.urls import path
from . import views

urlpatterns = [
    path('', views.contract_list, name='contract_list'),
    path('create/', views.contract_create, name='contract_create'),
    path('<int:pk>/detail/', views.contract_detail_api, name='contract_detail_api'),
    path('<int:pk>/update-file/', views.contract_update_file, name='contract_update_file'),
    path('document/<int:doc_id>/analyze/', views.document_analyze, name='document_analyze'),
    path('document/<int:doc_id>/ai-analyze/', views.document_ai_analyze, name='document_ai_analyze'),
    path('document/<int:doc_id>/complete/', views.document_complete_review, name='document_complete_review'),
]
