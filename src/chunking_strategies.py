"""
Compatibility shim.

The project is being refactored into a modular `app/` package.
Import chunking strategies from `app.services.chunking` going forward.
"""

from app.services.chunking import *  # noqa: F401,F403
