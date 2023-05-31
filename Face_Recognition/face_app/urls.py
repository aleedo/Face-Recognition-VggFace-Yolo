from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path("from_url/", views.from_url, name="from_url"),
    path("from_file/", views.from_file, name="from_file"),
]
