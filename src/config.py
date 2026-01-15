"""
Compatibility shim.

The project is being refactored into a modular `app/` package.
Import config from `app.core.config` going forward.
"""

from app.core.config import *  # noqa: F401,F403

