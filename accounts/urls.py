from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('password/change/', views.change_password_view, name='change_password'),
    path('mypage/', views.mypage_view, name='mypage'),
    path('mypage/update/', views.mypage_update, name='mypage_update'),
    path('help/', views.help_page, name='help'),
    path('notification/toggle/', views.toggle_notification, name='toggle_notification'),
]
