from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name='index'),
    # path("detail/<str:id>", views.detail, name='detail'),
    path("shoe_list/detail/<str:id>", views.detail, name='detail'),
    path("crop", views.crop_action, name='crop_action'),
    # path(, views.detail, name="detail"),
    path("shoe_list/<str:id>", views.shoe_list, name='shoe_list')
    
]