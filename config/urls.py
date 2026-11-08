from django.urls import include, path

urlpatterns = [path("", include("buildings.urls"))]
