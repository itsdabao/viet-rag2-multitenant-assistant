"""
Compatibility shim.

The project is being refactored into a modular `app/` package.
Import from `app.services.retrieval.bm25` going forward.
"""

from app.services.retrieval.bm25 import *  # noqa: F401,F403

