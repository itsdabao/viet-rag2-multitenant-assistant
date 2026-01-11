"""
Compatibility shim.

The project is being refactored into a modular `app/` package.
Import embedding setup from `app.core.llama` going forward.
"""

from app.core.llama import setup_embedding  # noqa: F401

