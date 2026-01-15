"""
Compatibility shim.

The project is being refactored into a modular `app/` package.
Import from `app.services.ingestion` going forward.
"""

from app.services.ingestion import *  # noqa: F401,F403