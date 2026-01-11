"""
Compatibility shim.

The project is being refactored into a modular `app/` package.
Import from `app.services.rag_service` going forward.
"""

from app.services.rag_service import *  # noqa: F401,F403

