"""
Compatibility shim.

The project is being refactored into a modular `app/` package.
Import from `app.services.retrieval.vector_store` going forward.
"""

from app.services.retrieval.vector_store import *  # noqa: F401,F403

