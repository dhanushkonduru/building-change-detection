from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "local-atlas-development-only")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = ["localhost", "127.0.0.1", "testserver"]
ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"
INSTALLED_APPS = ["django.contrib.staticfiles", "buildings.apps.BuildingsConfig"]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
]
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": ["django.template.context_processors.request"]
        },
    }
]
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
IMAGERY_DIR = Path(os.environ.get("IMAGERY_DIR", BASE_DIR / "data" / "samples")).resolve()
RESULTS_DIR = BASE_DIR / "var" / "results"
MODEL_PATH = BASE_DIR / "models" / "hot-buildings.onnx"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

MAX_IMAGE_UPLOAD_BYTES = 50 * 1024 * 1024
