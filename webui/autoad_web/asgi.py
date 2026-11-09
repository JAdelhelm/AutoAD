"""ASGI entry point of the AutoAD web interface."""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autoad_web.settings")

application = get_asgi_application()
