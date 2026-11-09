"""Settings of the AutoAD web interface.

The interface is a local tool without user accounts: run it on your own machine, not
on a public server. Everything it stores (database, uploads, fitted models, results)
lives in one data directory.

Environment variables:

``AUTOAD_WEB_DATA_DIR``
    Data directory, default ``webui/data``.
``AUTOAD_WEB_PORT``
    Default port of ``manage.py runserver``, default 8080 (see
    ``detection/management/commands/runserver.py``).
``AUTOAD_WEB_DEBUG``
    ``1`` (default) or ``0``. With ``0``, static files must be served by a web server
    (``manage.py collectstatic``).
``AUTOAD_WEB_SECRET_KEY``
    Secret key. If unset, one is generated once and stored in the data directory.
``AUTOAD_WEB_ALLOWED_HOSTS``
    Comma separated host names, default ``127.0.0.1,localhost``.
``AUTOAD_WEB_TIME_ZONE``
    Time zone for displayed dates, default ``UTC``.
``AUTOAD_WEB_MAX_UPLOAD_MB``
    Largest accepted file, default 100.
``AUTOAD_WEB_MAX_TRAIN_ROWS``
    Larger tables are fitted on a random sample of this many rows (all rows are
    scored), default 50000.
``AUTOAD_WEB_MAX_PLOT_POINTS``
    Most points per chart; all anomalies are always shown, default 20000.
"""

import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("AUTOAD_WEB_DATA_DIR", BASE_DIR / "data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _secret_key() -> str:
    key = os.environ.get("AUTOAD_WEB_SECRET_KEY")
    if key:
        return key
    path = DATA_DIR / "secret_key.txt"
    if not path.exists():
        path.write_text(secrets.token_urlsafe(50), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


SECRET_KEY = _secret_key()
DEBUG = _env_bool("AUTOAD_WEB_DEBUG", True)
ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get("AUTOAD_WEB_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
    if host.strip()
]

INSTALLED_APPS = [
    "django.contrib.messages",
    # Before staticfiles, so its runserver command (default port 8080) takes precedence.
    "detection",
    "django.contrib.staticfiles",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "autoad_web.urls"
WSGI_APPLICATION = "autoad_web.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": DATA_DIR / "db.sqlite3",
    }
}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# No sessions and no user accounts: messages live in a signed cookie.
MESSAGE_STORAGE = "django.contrib.messages.storage.cookie.CookieStorage"

LANGUAGE_CODE = "en"
TIME_ZONE = os.environ.get("AUTOAD_WEB_TIME_ZONE", "UTC")
USE_I18N = False
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = DATA_DIR / "static"
# plotly.min.js is served by a view straight from the installed plotly package (no CDN).

# Uploads and results are never served directly, only through views.
MEDIA_ROOT = DATA_DIR / "files"

X_FRAME_OPTIONS = "DENY"

AUTOAD_WEB_MAX_UPLOAD_MB = _env_int("AUTOAD_WEB_MAX_UPLOAD_MB", 100)
AUTOAD_WEB_MAX_TRAIN_ROWS = _env_int("AUTOAD_WEB_MAX_TRAIN_ROWS", 50_000)
AUTOAD_WEB_MAX_PLOT_POINTS = _env_int("AUTOAD_WEB_MAX_PLOT_POINTS", 20_000)
AUTOAD_WEB_ROWS_PER_PAGE = 50

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "loggers": {
        "detection": {"handlers": ["console"], "level": "INFO"},
        "autoad": {"handlers": ["console"], "level": "INFO"},
    },
}
