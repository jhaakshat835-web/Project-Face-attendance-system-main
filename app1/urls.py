from django.urls import path
from . import views

urlpatterns = [
    # --- Home & Auth ---
    path('', views.home, name='home'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),

    # --- Student Registration & Capture ---
    path('capture_student/', views.capture_student, name='capture_student'),
    path('selfie-success/', views.selfie_success, name='selfie_success'),
    
    # --- Attendance Marking ---
    path('capture-and-recognize/', views.capture_and_recognize, name='capture_and_recognize'),

    # --- Manage Students (Ye Alag Page Hai) ---
    path('students/', views.student_list, name='student-list'), 
    path('students/<int:pk>/', views.student_detail, name='student-detail'),
    path('students/<int:pk>/authorize/', views.student_authorize, name='student-authorize'),
    path('students/<int:pk>/delete/', views.student_delete, name='student-delete'),

    # --- Attendance Details (Ye Alag Page Hai) ---
    path('students/attendance/', views.student_attendance_list, name='student_attendance_list'),

    # --- Camera Settings ---
    path('camera-config/', views.camera_config_create, name='camera_config_create'),
    path('camera-config/list/', views.camera_config_list, name='camera_config_list'),
    path('camera-config/update/<int:pk>/', views.camera_config_update, name='camera_config_update'),
    path('camera-config/delete/<int:pk>/', views.camera_config_delete, name='camera_config_delete'),
]