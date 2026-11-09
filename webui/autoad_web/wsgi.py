"""WSGI entry point of the AutoAD web interface."""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autoad_web.settings")

application = get_wsgi_application()
