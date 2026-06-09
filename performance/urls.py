from django.urls import path
from . import views

urlpatterns = [
    path('', views.performance_list, name='performance_list'),
    path('<int:pk>/detail/', views.performance_detail_api, name='performance_detail_api'),
    path('<int:perf_id>/deliverable/upload/', views.deliverable_upload, name='deliverable_upload'),
    path('<int:perf_id>/deliverable/due-date/', views.deliverable_update_due_date, name='deliverable_update_due_date'),
]
