"""
Compatibility shim.

The project is being refactored into a modular `app/` package.
Import from `app.services.guardrails.domain_guard` going forward.
"""

from app.services.guardrails.domain_guard import *  # noqa: F401,F403

