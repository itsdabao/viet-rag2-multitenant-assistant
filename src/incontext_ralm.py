"""
Compatibility shim.

The project is being refactored into a modular `app/` package.
Import from `app.services.rag.incontext_ralm` going forward.
"""

from app.services.rag.incontext_ralm import *  # noqa: F401,F403

