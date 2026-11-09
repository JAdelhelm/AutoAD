"""``runserver`` on port 8090 by default.

Django's default port 8000 is often taken, for example by an editor such as Cursor or VS
Code on Windows, where Django then misleadingly reports "You don't have permission to
access that port". Set ``AUTOAD_WEB_PORT`` to use another default; an explicit port
(``runserver 9000``) always wins.
"""

import os

from django.contrib.staticfiles.management.commands.runserver import Command as StaticFilesRunserver


class Command(StaticFilesRunserver):
    # Subclasses the staticfiles version, so static files are still served in development.
    default_port = os.environ.get("AUTOAD_WEB_PORT", "8090")
