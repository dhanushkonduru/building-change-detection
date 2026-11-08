from django.urls import path
from . import views

app_name = "buildings"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/detect-changes/", views.detect_changes, name="detect_changes"),
    path("results/<str:key>/<str:name>", views.result_file, name="result_file"),
    path("sample/<str:role>.png", views.sample, name="sample"),
]
