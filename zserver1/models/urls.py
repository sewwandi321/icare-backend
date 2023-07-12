from .views import (tablets_userList, tablets_userDetail, 
                    tabletsList, tabletsDetail, 
                    lab_reportList, lab_reportDetail, 
                    blistersList, blistersDetail, 
                    blister_userList, blister_userDetail, 
                    predefined_precautionsList, predefined_precautionsDetail, 
                    prescriptionList, prescriptionDetail,
                    lab_report_precautionsList, lab_report_precautionsDetail,
                    lab_report_details,get_tablet_details)
from django.urls import path, include
from rest_framework.urlpatterns import format_suffix_patterns

# blister_user, lab_report, blisters,  predefined_precautions, prescription, tablets, tablets_user

urlpatterns = [
    path('tablets_user/', tablets_userList.as_view()),
    path('tablets_user/<int:pk>/', tablets_userDetail.as_view()),
    
    path('tablets/', tabletsList.as_view()),
    path('tablets/<int:pk>/', tabletsDetail.as_view()),
    
    path('lab_report/', lab_reportList.as_view()),
    path('lab_report/<int:pk>/', lab_reportDetail.as_view()),
    
    path('blisters/', blistersList.as_view()),
    path('blisters/<int:pk>/', blistersDetail.as_view()),
    
    path('blister_user/', blister_userList.as_view()),
    path('blister_user/<int:pk>/', blister_userDetail.as_view()),
    
    path('predefined_precautions/', predefined_precautionsList.as_view()),
    path('predefined_precautions/<int:pk>/', predefined_precautionsDetail.as_view()),
    
    path('prescription/', prescriptionList.as_view()),
    path('prescription/<int:pk>/', prescriptionDetail.as_view()),

    path('lab_report_precautions/', lab_report_precautionsList.as_view()),
    path('lab_report_precautions/<int:pk>/', lab_report_precautionsDetail.as_view()),

    path('lab_report_details/<int:pk>/', lab_report_details),
    path('get_tablet_details/<str:pk>/', get_tablet_details),
]

urlpatterns = format_suffix_patterns(urlpatterns)