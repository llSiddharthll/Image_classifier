# urls.py
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.predict_image, name='predict_image'),
    path('result/', views.result, name='result'),
]

# Include the media URL configuration for development only
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
