from django.contrib import admin
from django.urls import path
from webbidder.views import upload_csv
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', upload_csv, name='upload_csv'),
    path('favicon.ico', RedirectView.as_view(url='/static/favicon.ico')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)