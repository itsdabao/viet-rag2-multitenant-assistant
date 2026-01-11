"""
Compatibility shim.

The project is being refactored into a modular `app/` package.
Import from `app.services.documents` going forward.
"""

from app.services.documents import *  # noqa: F401,F403

